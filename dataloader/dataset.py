from __future__ import print_function

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data.sampler import Sampler
import torch.distributed as dist

import os
import cv2
import pickle
from PIL import Image
import numpy as np
from IPython import embed
from collections import defaultdict
import time
import sys
import cv2


class ImageNet(Dataset):
    def __init__(self, args, root_path=None, train_transform=None, test_transform=None, strong_train_trans=None,
                 fix_seed=True, split="train"):
        super(Dataset, self).__init__()
        self.fix_seed = fix_seed
        self.split = split
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.strong_train_trans = strong_train_trans

        self.dataset = args.dataset
        print("Current train dataset: ", self.dataset, self.split)
        if root_path is None:
            root_path = "/data/few_shot/data"
        if self.dataset == "tieredImageNet":
            self.imgs = np.load(os.path.join(root_path, self.dataset, "{}_images.npz").format(self.split))["images"]
            self.origin_labels = \
                pickle.load(open(os.path.join(root_path, self.dataset, "{}_labels.pkl").format(self.split), "rb"),
                            encoding='latin1')["labels"]
            if self.split == "train":
                self.val_imgs = np.load(os.path.join(root_path, self.dataset, "val_images.npz"))["images"]
                self.val_labels = pickle.load(open(os.path.join(root_path, self.dataset, "val_labels.pkl"), "rb"),
                                              encoding='latin1')["labels"]
                self.imgs = np.concatenate([self.imgs, self.val_imgs], axis=0)
                self.origin_labels.extend(self.val_labels)
        elif self.dataset == "miniImageNet":
            if self.split == "test":
                split_name = "{}_category_split_test.pickle".format(self.dataset)
            elif self.split == "val":
                split_name = "{}_category_split_val.pickle".format(self.dataset)
            else:
                split_name = "{}_category_split_train_phase_train.pickle".format(self.dataset)
            with open(os.path.join(root_path, self.dataset, split_name), 'rb') as f:
                data = pickle.load(f, encoding='latin1')
                self.imgs = data['data']
                self.origin_labels = data['labels']
        elif self.dataset == "FC100" or self.dataset == "CIFAR-FS":
            with open(os.path.join(root_path, self.dataset, "{}.pickle".format(self.split)), 'rb') as f:
                data = pickle.load(f, encoding='latin1')
                self.imgs = data['data']
                self.origin_labels = data['labels']

        self.Label2idx = dict(zip(list(set(self.origin_labels)),
                                  range(len(list(set(self.origin_labels))))))
        self.labels = []
        for l in self.origin_labels:
            self.labels.append(self.Label2idx[l])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        data, label = self.imgs[item], self.labels[item]
        if self.split == "train":
            if self.strong_train_trans is not None:
                data = self.strong_train_trans.augment_image(data)
            else:
                data = data
            image = self.train_transform(data)
        else:
            image = self.test_transform(data)
        return image, label


class CUB(Dataset):
    def __init__(self, split, train_transform=None, test_transform=None):
        self.split = split
        self.wnids = []
        if self.split == "train":  # train with raw img
            self.root_path = "/data/few_shot/data/CUB_200_2011"
            self.IMAGE_PATH = os.path.join(self.root_path, "images")
            self.SPLIT_PATH = os.path.join(self.root_path, "split")
        else:  # test with cropped img based on bounding box
            self.IMAGE_PATH = "/data/few_shot/data/CUB_test/"
            self.SPLIT_PATH = os.path.join(self.IMAGE_PATH, "split")
        txt_path = os.path.join(self.SPLIT_PATH, split + '.csv')
        self.data, self.labels = self.parse_csv(txt_path)
        self.num_class = np.unique(np.array(self.labels)).shape[0]
        print("Current {} dataset: CUB, {}_ways".format(split, self.num_class))
        self.train_transform = train_transform
        self.test_transform = test_transform

    def parse_csv(self, txt_path):
        data = []
        label = []
        lb = -1
        lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]
        for l in lines:
            context = l.split(',')
            name = context[0]
            wnid = context[1]
            if self.split == "train":
                path = os.path.join(self.IMAGE_PATH, wnid, name)
            else:
                path = os.path.join(self.IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        return data, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label = self.data[i], self.labels[i]
        if self.split == "train":
            image = self.train_transform(Image.open(data).convert('RGB'))
        else:
            image = self.test_transform(Image.open(data).convert('RGB'))
        return image, label


class CategoriesSampler():
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.catlocs = []
        for i in range(min(label), max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.catlocs.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for _ in range(self.n_batch):
            batch = []
            choose_classes = torch.randperm(len(self.catlocs))[:self.n_cls]
            for c in choose_classes:
                l = self.catlocs[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


class DCategoriesSampler(Sampler):
    def __init__(self, label, n_batch, n_cls, n_per, ep_per_batch=8,
                 num_replicas=None, rank=None):
        super().__init__(self)
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank

        self.n_batch = n_batch  # batchs for each epoch
        self.n_cls = n_cls  # ways
        self.n_per = n_per  # shots
        self.num_samples = self.n_cls * self.n_per
        self.ep_per_batch = ep_per_batch
        label = np.array(label)
        self.catlocs = []
        for i in range(min(label), max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.catlocs.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            for i_ep in range(self.ep_per_batch):
                episode = []
                choose_classes = torch.randperm(len(self.catlocs))[:self.n_cls]
                for c in choose_classes:
                    l = self.catlocs[c]
                    samples = torch.randperm(len(l))[:self.n_per]
                    episode.append(l[samples])
                batch.append(torch.stack(episode).t().reshape(-1))
            batch = torch.stack(batch).reshape(-1)
            # subsample
            offset = self.num_samples * self.rank
            batch = batch[offset: offset + self.num_samples]
            # print(">" * 50, self.rank, len(batch), self.num_samples)
            assert len(batch) == self.num_samples

            yield batch