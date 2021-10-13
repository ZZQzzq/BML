import argparse
import torch
import os
from dataloader.transform.transform_cfg import transforms_list


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument("--mode", type=str, choices=["bml", "global", "local"], default="bml")
    parser.add_argument('--backbone', type=str, choices=["Res12", "Res18"], default="Res12")
    parser.add_argument('--eval_freq', type=int, default=20, help='meta-eval frequency')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=20, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')

    # optimization
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument("--decay_step", type=int, default=40)
    parser.add_argument('--optim', type=str, choices=["adam", "SGD"], default="SGD")
    # dataset
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet',
                                                                                'CIFAR-FS', 'FC100', "CUB"])
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)
    # meta setting
    parser.add_argument('--n_ways', type=int, default=5, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=5, metavar='N',
                        help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N',
                        help='Number of query in test')
    parser.add_argument('--n_train_ways', type=int, default=15, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--n_train_shots', type=int, default=5, metavar='N',
                        help='Number of shots during train')
    parser.add_argument('--n_train_queries', type=int, default=7, metavar='N',
                        help='Number of query in test')
    parser.add_argument('--episodes', default=2000, help="test episode num")
    # bml setting
    parser.add_argument("--weights", type=str, default="2_1_0.5")
    parser.add_argument("--alpha_1", type=float, default=5.5)  # 2.5
    parser.add_argument("--alpha_2", type=float, default=0.1)
    parser.add_argument("--T", type=float, default=4.0)
    # spatial for single view training
    parser.add_argument("--spatial", action="store_true")
    # continue train
    parser.add_argument("--is_continue", action="store_true")
    # eval
    parser.add_argument("--is_eval", action="store_true")
    parser.add_argument("--ckp_path", type=str, default="")
    # save setting
    parser.add_argument('-s', '--save_folder', type=str, default='params')

    # DistributedDataParallel
    parser.add_argument("--local_rank", type=int, required=True, default=0, help='local rank for DistributedDataParallel')

    parser.add_argument('--amp_opt_level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument("--seed", default='0', type=str)
    
    opt = parser.parse_args()
    opt.w = [float(i) for i in opt.weights.split('_')]
    assert (opt.transform == "A" and opt.dataset == "miniImageNet") or \
           (opt.transform == "B" and opt.dataset == "tieredImageNet") or \
           (opt.transform == "C" and opt.dataset == "FC100") or \
           (opt.transform == "D" and opt.dataset == "CIFAR-FS") or \
           (opt.transform == "E" and opt.dataset == "CUB")
    if opt.dataset == "miniImageNet" or opt.dataset == "CIFAR-FS":
        opt.n_cls = 64
    elif opt.dataset == "tieredImageNet":
        opt.n_cls = 351 + 97
    elif opt.dataset == 'FC100':
        opt.n_cls = 60
    elif opt.dataset == "CUB":
        opt.n_cls = 100
    else:
        raise ValueError("wrong dataset")
    if opt.mode == "global":
        opt.model_name = '{}_{}_{}_Trans_{}_bs{}_Optim_{}_SIM_{}'.format(
            opt.mode, opt.backbone, opt.dataset, opt.transform, opt.batch_size,
            opt.optim, opt.spatial)
    elif opt.mode == "local":
        opt.model_name = '{}_{}_{}_Trans_{}_N{}Q{}S{}_Optim_{}_SIM_{}'.format(
            opt.mode, opt.backbone, opt.dataset, opt.transform, opt.n_train_ways, opt.n_train_queries, opt.n_train_shots,
            opt.optim, opt.spatial)
    else:
        opt.model_name = '{}_{}_{}_Trans_{}_N{}Q{}S{}_Optim_{}_a1_{}_a2_{}_w_{}_T_{}'.format(
            opt.mode, opt.backbone, opt.dataset, opt.transform, opt.n_train_ways, opt.n_train_queries, opt.n_train_shots,
            opt.optim, opt.alpha_1, opt.alpha_2, opt.weights, opt.T)
    # if not opt.is_eval:
    os.makedirs(os.path.join(opt.save_folder, opt.model_name), exist_ok=True)
    opt.save_folder = os.path.join(opt.save_folder, opt.model_name)
    opt.SEED = 0
    if opt.is_eval:
        assert len(opt.ckp_path)
    if opt.optim == 'SGD':
        opt.initial_lr = 1e-1
    else:
        opt.initial_lr = 1e-3

    return opt