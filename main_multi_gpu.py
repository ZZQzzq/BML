from __future__ import print_function
import os
import numpy as np
from IPython import embed
import time
import datetime
import sys
from tqdm import tqdm
import json

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import torch.optim as optim
from sklearn import metrics

from dataloader.dataloader import bml_dataloder, global_dataloder
from model.BML import BMLBuilder
from model.baseline_global import BaseGlobalBuilder
from model.baseline_local import BaseLocalBuilder
from config import parse_option
from logger import create_logger
from utils.util import prepare_label, mean_confidence_interval, accuracy, AverageMeter, \
    save_checkpoint, load_checkpoint, DistillKL, update_lr, auto_resume_helper, vis_hard_p_n
try:
    from apex import amp
except Exception:
    amp = None
    print('WARNING: could not import pygcransac')
    pass


def build_model(config):
    if config.mode == "local":
        return BaseLocalBuilder(config)
    elif config.mode == "global":
        return BaseGlobalBuilder(config)
    elif config.mode == "bml":
        return BMLBuilder(config)
    else:
        raise ValueError('Dont support {}'.format(config.mode))


def build_dataloader(config, world_size):
    is_distributed = (world_size > 0)
    if config.mode in ["local", "bml"]:
        return bml_dataloder(config, is_distributed=is_distributed)
    elif config.mode == "global":
        return global_dataloder(config)
    else:
        raise ValueError('Dont support {}'.format(config.mode))


def build_optimizer(model, config):
    if config.optim == "adam":
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=config.initial_lr)
    else:
        optimizer = optim.SGD(model.parameters(),
                              lr=config.initial_lr,
                              momentum=config.momentum,
                              weight_decay=config.weight_decay)
    return optimizer


def train_bml_on_epoch(config, model, optimizer, criterion_Intra, criterion_Cross, epoch, meta_trainloader):
    model.train()

    num_steps = len(meta_trainloader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    mutual_meter = AverageMeter()

    start = time.time()
    end = time.time()
    _idx = 0
    for idx, (samples, global_targets) in enumerate(meta_trainloader):
        samples = samples.cuda(non_blocking=True)
        global_targets = global_targets.cuda(non_blocking=True)

        meta_targets = prepare_label(config, split="train")
        # =================== forward =====================
        global_logits, local_logits, global_feat, local_feat = model(samples,
                                                                     label=meta_targets,
                                                                     split="train",
                                                                     epoch=epoch,
                                                                     max_epoch=config.epochs)

        # =================== cal loss =====================
        global_loss = 0
        for i in range(global_logits.size(2)):
            global_loss += criterion_Intra(global_logits[..., i], global_targets) / global_logits.size(2)
        local_loss = 0
        for i in range(local_logits.size(2)):
            local_loss += criterion_Intra(local_logits[..., i], meta_targets) / local_logits.size(2)
        mutual_loss = criterion_Cross(global_feat, local_feat) + criterion_Cross(local_feat, global_feat)
        # mutual_loss = criterion_Cross(global_feat, local_feat)
        loss = config.w[0] * global_loss + config.w[1] * local_loss + config.w[2] * mutual_loss

        # ===================backward=====================
        optimizer.zero_grad()

        if config.amp_opt_level != "O0":
            if amp is not None:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else: # not support amp
                loss.backward()
        else:
            loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

        # ===================summery=====================
        batch_time.update(time.time() - end)
        mutual_meter.update(mutual_loss.item(), 25)
        loss_meter.update(loss.item(), 25)
        end = time.time()

        if idx % config.print_freq == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.epochs}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'mutual diff {mutual_meter.val:.4f} ({mutual_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


def train_single_on_epoch(config, model, optimizer, criterion_Intra, epoch, meta_trainloader):
    model.train()

    num_steps = len(meta_trainloader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, global_targets) in enumerate(meta_trainloader):
        samples = samples.cuda(non_blocking=True)
        global_targets = global_targets.cuda(non_blocking=True)
        meta_targets = prepare_label(config, split="train")
        logits = model(samples, split="train")
        if config.mode == "local":
            targets = meta_targets
        else:
            assert config.mode == "global"
            targets = global_targets
        if config.spatial:
            loss = 0
            for i in range(logits.size(2)):
                loss += criterion_Intra(logits[..., i], targets) / logits.size(2)
        else:
            loss = criterion_Intra(logits, targets)

        # ===================backward=====================
        optimizer.zero_grad()
        if config.amp_opt_level != "O0":
            if amp is not None:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
        else:
            loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        # ===================summery=====================
        batch_time.update(time.time() - end)
        loss_meter.update(loss.item(), 25)
        end = time.time()

        # print info
        if idx % config.print_freq == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Train: [{epoch}/{config.epochs}][{idx}/{num_steps}]\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


def main(config, world_size):
    logger.info(f"Prepare Dataset:{config.dataset}/{config.transform}")
    dataset_train, _, dataset_test = build_dataloader(config, world_size)

    logger.info(f"Creating model:{config.mode}/{config.backbone}")
    model = build_model(config)
    model.cuda()
    logger.info(str(model))

    criterion_Intra = nn.CrossEntropyLoss()
    criterion_Cross = DistillKL(T=config.T)

    if config.is_continue:
        resume_file = auto_resume_helper(config.save_folder)
        config.ckp_path = resume_file
        config.initial_lr, start_epoch = update_lr(config, logger)
    else:
        start_epoch = 1

    optimizer = build_optimizer(model, config)
    lrstep1, lrstep2, max_accuracy = 50, 70, 0
    if config.amp_opt_level != "O0" and amp is not None:
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.amp_opt_level)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.local_rank], broadcast_buffers=False)
    model_without_ddp = model.module
    if config.is_continue:
        load_checkpoint(config, model_without_ddp, logger)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters / 1e6} M")
    if config.is_eval:
        load_checkpoint(config, model_without_ddp, logger)
        accuracy = validate(model, dataset_test, config)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, config.epochs + 1):
        if config.mode == "bml":
            train_bml_on_epoch(config, model, optimizer, criterion_Intra, criterion_Cross, epoch, dataset_train)
        else:
            train_single_on_epoch(config, model, optimizer, criterion_Intra, epoch, dataset_train)
        # evaluating
        if epoch % config.eval_freq == 0:
            accuracy = validate(model, dataset_test, config)
            logger.info(f"Accuracy of the network on the {len(dataset_test) * 4} test images: {accuracy:.2f}%")
            max_accuracy = max(max_accuracy, accuracy)
            logger.info(f'Max accuracy: {max_accuracy:.2f}%')

        # saving
        if dist.get_rank() == 0 and (epoch % config.eval_freq == 0 or epoch == config.epochs):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, logger)

        if config.dataset == "tieredImageNet":
            if epoch % config.decay_step == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
                    print('-------Decay Learning Rate to ', param_group['lr'], '------')
        else:
            decay1 = 0.06 if config.optim == 'SGD' else 0.1
            decay2 = 0.2 if config.optim == 'SGD' else 0.1

            if epoch == lrstep1:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= decay1
                    print('-------Decay Learning Rate to ', param_group['lr'], '------')
            if epoch == lrstep2:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= decay2
                    print('-------Decay Learning Rate to ', param_group['lr'], '------')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    # eval result
    print("==== [best {}-way {}-shot]: acc: {}".format(config.n_ways, config.n_shots, max_accuracy))


@torch.no_grad()
def validate(model, testloader, config):
    criterion = torch.nn.CrossEntropyLoss()

    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    end = time.time()

    acc, scores = [], []
    for idx, (images, _) in enumerate(testloader):
        images = images.cuda(non_blocking=True)
        # compute output
        output = model(images)
        if config.mode == "bml":
            output = output[0] + output[1]
        target = prepare_label(config, split="test")
        target = target.cuda(non_blocking=True)

        # measure accuracy and record loss
        loss = criterion(output, target)
        loss_meter.update(loss.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if idx % config.print_freq == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(testloader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Mem {memory_used:.0f}MB')
        logits = torch.argmax(output, dim=1)
        logits = logits.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        acc.append(metrics.accuracy_score(target, logits))
    acc_list = [i * 100 for i in acc]
    ci95 = 1.96 * np.std(acc_list, axis=0) / np.sqrt(len(acc_list))
    logger.info(f' * Acc on {config.n_ways} way-{config.n_shots} shot: {np.mean(acc_list):.3f}({ci95:.3f})')
    if config.is_eval:
        os.makedirs("eval_results", exist_ok=True)
        with open(os.path.join("eval_results", config.dataset + "_bml_eval_results.txt"), "a") as f:
            f.write(str("{}: {}\n".format(config.seed, np.mean(acc_list))))
        f.close()
    return np.mean(acc_list)


if __name__ == "__main__":
    config = parse_option()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    torch.cuda.set_device(config.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = int(config.seed) + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    logger = create_logger(output_dir=config.save_folder, dist_rank=dist.get_rank(), name=f"{config.mode}")
    if dist.get_rank() == 0:
        path = os.path.join(config.save_folder, "config.json")
        with open(path, "w") as f:
            f.write(str(vars(config)))
        logger.info(f"Full config saved to {path}")
    # print config
    logger.info(vars(config))
    main(config, world_size)
