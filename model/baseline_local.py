from __future__ import print_function

import torch
from torch import nn, Tensor
import numpy as np

from model.base import BaseBuilder


class BaseLocalBuilder(BaseBuilder):
    def __init__(self, opt):
        super().__init__(opt)
        if opt.backbone == 'Res12':
            from backbone.resnet12 import resnet12
            self.encoder = resnet12()
        elif opt.backbone == 'Res18':
            from backbone.resnet18 import resnet18
            self.encoder = resnet18()
        else:
            raise ValueError('')

    def forward(self, x, split="test"):
        instance_embs, instance_embs_spatial = self.encoder.forward(x)
        if split in ["test", "val"]:
            n_ways = self.opt.n_ways
            n_queries = self.opt.n_queries
            n_shots = self.opt.n_shots
        else:
            n_ways = self.opt.n_train_ways
            n_queries = self.opt.n_train_queries
            n_shots = self.opt.n_train_shots
        support_idx, query_idx = self.split_instances(n_ways, n_queries, n_shots)
        if split == "train" and self.opt.spatial:
            bs, emb, w, h = instance_embs_spatial.shape
            logits = torch.zeros((n_queries * n_ways, n_ways, w * h)).cuda()
            support = instance_embs_spatial[support_idx.flatten()].view(*(support_idx.shape + (emb, w * h)))
            query = instance_embs_spatial[query_idx.flatten()].view(*(query_idx.shape + (emb, w * h)))
            proto = support.mean(dim=1).mean(dim=-1)
            for i in range(query.size(-1)):
                cur_logits = - self.compute_logits(proto, query[..., i], emb, query_idx)
                logits[..., i] = cur_logits
        else:
            emb = instance_embs.size(-1)
            support = instance_embs[support_idx.flatten()].view(
                *(support_idx.shape + (-1,)))
            query = instance_embs[query_idx.flatten()].view(*(query_idx.shape + (-1,)))
            proto = support.mean(dim=1)
            logits = - self.compute_logits(proto, query, emb, query_idx)
        return logits