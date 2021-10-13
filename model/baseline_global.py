from __future__ import print_function

import torch
from torch import nn, Tensor
import numpy as np

from model.base import BaseBuilder


class BaseGlobalBuilder(BaseBuilder):
    def __init__(self, opt):
        super().__init__(opt)
        if opt.backbone == 'Res12':
            from backbone.resnet12 import resnet12
            self.encoder = resnet12()
            hdim = 640
        elif opt.backbone == 'Res18':
            from backbone.resnet18 import resnet18
            self.encoder = resnet18()
            hdim = 512
        else:
            raise ValueError('')

        if opt.spatial:
            self.global_w = nn.Conv2d(in_channels=hdim, out_channels=self.opt.n_cls, kernel_size=1, stride=1)
        else:
            self.global_w = nn.Linear(hdim, self.opt.n_cls)
        nn.init.xavier_uniform_(self.global_w.weight)

    def forward(self, x, split="test"):
        instance_embs, instance_embs_spatial = self.encoder(x)
        if split == "train":
            if self.opt.spatial:
                logits = self.global_w(instance_embs_spatial)
                logits = logits.flatten(start_dim=2)
            else:
                logits = self.global_w(instance_embs)
        else:
            support_idx, query_idx = self.split_instances(self.opt.n_ways, self.opt.n_queries, self.opt.n_shots)
            emb = instance_embs.size(-1)
            support = instance_embs[support_idx.flatten()].view(
                *(support_idx.shape + (-1,)))
            query = instance_embs[query_idx.flatten()].view(*(query_idx.shape + (-1,)))
            proto = support.mean(dim=1)
            logits = - self.compute_logits(proto, query, emb, query_idx)
        return logits