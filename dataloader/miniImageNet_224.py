from __future__ import print_function

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.optim as optim

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
import json
from meghair.utils.imgproc import imdecode
import nori2 as nori
train_list = '/data/few_shot/data/miniImageNet/origin_data/imagenet.train.nori.list'  # 全部数据
fetcher = nori.Fetcher()
nori_dict, labels = dict(), []
with open(train_list) as f:
    lines = f.readlines()
for line in lines:
    image_path = line.split('\t')[0]
    image_name = line.split('\t')[-1].split('/')[-1][:-1]
    nori_dict[image_name] = image_path


class ImageNet(Dataset):

    def __init__(self, args, train_transform=None, test_transform=None, strong_train_trans=None,
                 fix_seed=True, split="train"):
        super(Dataset, self).__init__()
        self.fix_seed = fix_seed
        self.split = split
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.strong_train_trans = strong_train_trans
        if self.split == "train":
            data_file = "base.json"
        else:
            data_file = "noval.json"
        with open(os.path.join("/data/few_shot/data/miniImageNet/origin_data", data_file), 'r') as f:
            self.meta = json.load(f)
        train_list = defaultdict(list)
        all_imgs = self.meta['image_names']
        for img_path in all_imgs:
            cls_name = img_path.split("/")[-2]
            img_name = img_path.split("/")[-1]
            train_list[cls_name].append(nori_dict[img_name])
        cls2idxs = dict(zip(list(train_list.keys()), range(len(train_list.keys()))))

        self.train_info = []
        self.labels = []
        for key, values in train_list.items():
            for value in values:
                self.train_info.append([cls2idxs[key], value])
            self.labels.extend([cls2idxs[key] for _ in range(len(values))])

        self.dataset = args.dataset
        print("Current train dataset: ", self.dataset, self.split)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        label, nori_path = self.train_info[item]
        img = imdecode(fetcher.get(nori_path))[..., :3]
        if self.split == "train":
            img = self.train_transform(img)
        else:
            img = self.test_transform(img)
        return img, label


class CategoriesSampler():  # n_batch次的类别采样
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(min(label), max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)  # 按照类别划分样本， 0-599: 0类， 600-1199： 1类

        self.score_list = []

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for _ in range(self.n_batch):
            batch = []
            choose_classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in choose_classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch
