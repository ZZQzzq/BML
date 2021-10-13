from torch.utils.data import Dataset, DataLoader
from dataloader.dataset import ImageNet, CUB, CategoriesSampler, DCategoriesSampler
from dataloader.transform.transform_cfg import transforms_options, transforms_list


def get_val_test(opt, train_trans, test_trans, is_distributed=False):
    if opt.dataset == "CUB":
        valset = CUB(train_transform=train_trans, test_transform=test_trans, split="val")
    else:
        valset = ImageNet(args=opt, train_transform=train_trans, test_transform=test_trans, split="val")
    if not is_distributed:
        val_sampler = CategoriesSampler(label=valset.labels,
                                        n_batch=opt.episodes,
                                        n_cls=opt.n_ways,
                                        n_per=opt.n_shots + opt.n_queries)
    else:
        val_sampler = DCategoriesSampler(label=valset.labels,
                                         n_batch=opt.episodes,
                                         n_cls=opt.n_ways,
                                         n_per=opt.n_shots + opt.n_queries)
    meta_valloader = DataLoader(dataset=valset,
                                num_workers=opt.num_workers,
                                batch_sampler=val_sampler,
                                pin_memory=True)
    if opt.dataset == "CUB":
        testset = CUB(train_transform=train_trans, test_transform=test_trans, split="test")
    else:
        testset = ImageNet(args=opt, train_transform=train_trans, test_transform=test_trans, split="test")
    if not is_distributed:
        test_sampler = CategoriesSampler(label=testset.labels,
                                         n_batch=opt.episodes,
                                         n_cls=opt.n_ways,
                                         n_per=opt.n_shots + opt.n_queries)
    else:
        test_sampler = DCategoriesSampler(label=testset.labels,
                                          n_batch=opt.episodes,
                                          n_cls=opt.n_ways,
                                          n_per=opt.n_shots + opt.n_queries)
    meta_testloader = DataLoader(dataset=testset,
                                 num_workers=opt.num_workers,
                                 batch_sampler=test_sampler,
                                 pin_memory=True)
    return meta_valloader, meta_testloader


def bml_dataloder(opt, is_val=True, is_distributed=False):
    train_trans, test_trans, test_trans_plus = transforms_options[opt.transform]
    if opt.dataset == "CUB":
        trainset = CUB(split="train", train_transform=train_trans, test_transform=test_trans)
    else:
        trainset = ImageNet(args=opt, train_transform=train_trans, test_transform=test_trans, split="train")
    if not is_distributed:
        train_sampler = CategoriesSampler(label=trainset.labels,
                                          n_batch=500,
                                          n_cls=opt.n_train_ways,
                                          n_per=opt.n_train_shots + opt.n_train_queries)
    else:
        train_sampler = DCategoriesSampler(label=trainset.labels,
                                           n_batch=500,
                                           n_cls=opt.n_train_ways,
                                           n_per=opt.n_train_shots + opt.n_train_queries)
    meta_trainloader = DataLoader(dataset=trainset,
                                  num_workers=opt.num_workers,
                                  batch_sampler=train_sampler,
                                  pin_memory=True)
    # test & val
    if is_val:
        meta_valloader, meta_testloader = get_val_test(opt, train_trans, test_trans, is_distributed)
    else:
        meta_valloader, meta_testloader = None, None
    return meta_trainloader, meta_valloader, meta_testloader


def global_dataloder(opt, is_val=True):
    # transforms
    train_trans, test_trans, test_trans_plus = transforms_options[opt.transform]
    # # train
    if opt.dataset == "CUB":
        trainset = CUB(train_transform=train_trans, test_transform=test_trans, split="train")
    else:
        trainset = ImageNet(args=opt, train_transform=train_trans, test_transform=test_trans, split="train")
    meta_trainloader = DataLoader(dataset=trainset, batch_size=opt.batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers,
                                  pin_memory=True)
    # test & val
    if is_val:
        meta_valloader, meta_testloader = get_val_test(opt, train_trans, test_trans)
    else:
        meta_valloader, meta_testloader = None, None
    return meta_trainloader, meta_valloader, meta_testloader


def global_224_dataloder(opt, is_val=True):
    from dataloader.transformer.transform_cfg_big import transforms_options
    from dataloader.miniImageNet_224 import ImageNet, CategoriesSampler
    # transforms
    train_trans, test_trans, test_trans_plus = transforms_options[opt.transform]
    # # train
    trainset = ImageNet(args=opt, train_transform=train_trans, test_transform=test_trans, split="train")
    meta_trainloader = DataLoader(dataset=trainset, batch_size=opt.batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers,
                                  pin_memory=True)
    # test & val
    if is_val:
        meta_valloader, meta_testloader = get_val_test(opt, train_trans, test_trans)
    else:
        meta_valloader, meta_testloader = None, None
    return meta_trainloader, meta_valloader, meta_testloader


def bml_224_dataloder(opt, is_val=True):
    from dataset.transform_cfg_big import transforms_options
    from models.big_dataloader import ImageNet, CategoriesSampler
    # transforms
    train_trans, test_trans, test_trans_plus = transforms_options[opt.transform]
    # # train
    trainset = ImageNet(args=opt, train_transform=train_trans, test_transform=test_trans, split="train")
    train_sampler = CategoriesSampler(label=trainset.labels,
                                      n_batch=500,
                                      n_cls=opt.n_train_ways,
                                      n_per=opt.n_shots + opt.n_queries)
    meta_trainloader = DataLoader(dataset=trainset,
                                  num_workers=opt.num_workers,
                                  batch_sampler=train_sampler,
                                  pin_memory=True)

    # test & val
    if is_val:
        testset = ImageNet(args=opt, train_transform=train_trans, test_transform=test_trans, split="test")
        test_sampler = CategoriesSampler(label=testset.labels,
                                         n_batch=600,
                                         n_cls=opt.n_train_ways,
                                         n_per=opt.n_shots + opt.n_queries)
        meta_testloader = DataLoader(dataset=testset,
                                     num_workers=opt.num_workers,
                                     batch_sampler=test_sampler,
                                     pin_memory=True)
    else:
        meta_testloader = None
    meta_valloader = None

    return meta_trainloader, meta_valloader, meta_testloader
