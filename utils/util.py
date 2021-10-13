import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
import scipy
from scipy.stats import t

from IPython import embed
import os
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt


class DistillKL(nn.Module):
    """KL divergence for distillation"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / y_s.shape[0]
        return loss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class PairInfoMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.mean = 0
        self.hard = 0

    def update(self, mean_simi, hard_simi):
        self.mean = mean_simi
        self.hard = hard_simi


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1 + confidence) / 2., n - 1)
    min_ac, max_acc = np.min(a), np.max(a)
    return m, h, min_ac, max_acc


def prepare_label(opt, split="test"):
    # prepare one-hot label
    if split in ["test", "val"]:
        label = torch.arange(opt.n_ways, dtype=torch.int16).repeat(opt.n_queries)
    else:
        label = torch.arange(opt.n_train_ways, dtype=torch.int16).repeat(opt.n_train_queries)
    label = label.type(torch.LongTensor)
    if torch.cuda.is_available():
        label = label.cuda()
    return label


def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.

    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth]))
    if indices.is_cuda:
        encoded_indicies = encoded_indicies.cuda()
    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)

    return encoded_indicies


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def update_lr(config, logger):
    checkpoint = torch.load(config.ckp_path)
    epoch = checkpoint["epoch"]
    config = checkpoint["config"]
    if config.dataset == "tieredImageNet":
        init_lr = 1e-3 * (epoch // config.decay_step)
    else:
        if epoch > 70:
            init_lr = 0.1 * 0.2 * 0.06
        elif epoch > 50:
            init_lr = 0.1 * 0.2
        else:
            init_lr = 0.1
    logger.info(f"==> Update learning rate to {init_lr}....................")
    del checkpoint
    torch.cuda.empty_cache()
    return init_lr, epoch


def load_checkpoint(config, model, logger):
    logger.info(f"==> Loading form {config.ckp_path}....................")
    model_dict = model.state_dict()
    checkpoint = torch.load(config.ckp_path)
    exist_pretrained_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict}
    model_dict.update(exist_pretrained_dict)
    msg = model.load_state_dict(model_dict, strict=False)
    logger.info(msg)
    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(config, epoch, model, max_accuracy, logger):
    save_state = {'model': model.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}

    save_path = os.path.join(config.save_folder, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def vis_hard_p_n(prob_pos_list, prob_neg_list, config, start_idx):
    plt.figure()
    all_cnt = len(prob_pos_list)
    x_list = list(range(start_idx, start_idx+all_cnt))
    plt.plot(x_list, prob_pos_list, label='P', color='forestgreen')
    plt.plot(x_list, prob_neg_list, label='N', color='tomato')
    plt.legend()
    plt.ylabel("prob")
    plt.xlabel("iteration")
    plt.savefig(os.path.join(config.save_folder, "prob.png"))
    plt.close()


def mean_test_acc(dataset_name):
    file = open(os.path.join("eval_results", dataset_name + "_bml_eval_results.txt"), "r")
    acc_list = file.readlines()
    acc = 0
    for acc_ in acc_list:
        acc += float(acc_[:-1].split(" ")[-1])
    print("==> Mean ACC on {}: {}".format(dataset_name, acc/len(acc_list)))
    with open(os.path.join("eval_results", dataset_name + "_bml_eval_results.txt"), "a") as f:
        f.write(str("==> Mean ACC on {}: {}\n".format(dataset_name, acc/len(acc_list))))
    f.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('eval')
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet',
                                                                                'CIFAR-FS', 'FC100', "CUB"])
    config = parser.parse_args()
    mean_test_acc(config.dataset)





