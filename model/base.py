from __future__ import print_function

import torch
from torch import nn, Tensor
import numpy as np


class BaseBuilder(nn.Module):
    def __init__(self, opt):
        super(BaseBuilder, self).__init__()
        self.opt = opt

    def split_instances(self, _ways, _queries, _shots):
        return (torch.Tensor(np.arange(_ways * _shots)).long().view(1, _shots, _ways),
                torch.Tensor(np.arange(_ways * _shots, _ways * (_shots + _queries)))
                .long().view(1, _queries, _ways))

    def compute_logits(self, proto, query, emb_dim, query_idx):
        num_batch = proto.shape[0]  # 1
        num_proto = proto.shape[1]  # 5
        num_query = int(np.prod(query_idx.shape[-2:]))

        query = query.reshape(-1, emb_dim).unsqueeze(1)

        proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim)
        proto = proto.contiguous().view(num_batch * num_query, num_proto, emb_dim)
        logits = torch.sum((proto - query) ** 2, 2)
        return logits
