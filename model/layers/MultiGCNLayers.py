# -*- coding: utf-8 -*-
"""
   File Name：     GCNlayers
   Description : 图卷积层，多个channal
   Author :       mingjian
   date：          2021/3/11
"""
__author__ = 'mingjian'

import torch
import torch.nn as nn
import torch_geometric.nn as gnn



class MultiGCNLayers(nn.Module):

    def __init__(self, d_in, d_h, d_out, sz_c, sz_l, drop_rate, device, g_norm=False):
        super().__init__()
        self.sz_c = sz_c
        self.d_in = d_in
        self.d_h = d_h
        self.d_out = d_out
        self.sz_l = sz_l
        self.drop_rate = drop_rate
        self.device = device
        self.gnn_type = gnn.GCNConv
        self.gcn_layer = nn.ModuleList(self.channal_block(g_norm))
        self.layer_norm = nn.LayerNorm(d_out, eps=1e-6)
        # self.Dropout = nn.Dropout(self.drop_rate)

    def reset_parameters(self):
        for conv in self.gcn_layer:
            if type(conv) == self.gnn_type:
                conv.reset_parameters()

    def channal_block(self, g_norm):
        layer = []
        for c in range(self.sz_c):
            t_layer = []
            for l in range(self.sz_l):
                if l == 0:
                    t_layer.append(self.gnn_type(self.d_in, self.d_h))
                elif l == (self.sz_l - 1):
                    t_layer.append(self.gnn_type(self.d_h, self.d_out))
                else:
                    t_layer.append(self.gnn_type(self.d_h, self.d_h))
                # t_layer.append(nn.Dropout(self.drop_rate))
                if g_norm:
                    t_layer.append(gnn.GraphSizeNorm())
                    if l == (self.sz_l - 1):
                        t_layer.append(gnn.BatchNorm(self.d_out))
                    else:
                        t_layer.append(gnn.BatchNorm(self.d_h))
                t_layer.append(nn.ReLU())
            layer.append(nn.ModuleList(t_layer))
            # layer.append(nn.Sequential(*t_layer))
        return layer
        # pass

    def forward(self, x, edge, batch):
        channal = torch.empty([self.sz_c, x.size(0), self.d_out]).to(self.device)
        # channal = []
        for i in range(self.sz_c):
            h = x
            m = self.gcn_layer[i]
            for el in m:
                rh = h
                if isinstance(el, self.gnn_type):
                    h = el(h, edge)
                else:
                    h = el(h)
                if isinstance(el, nn.ReLU) and h.size(1) == rh.size(1):
                    h = h + rh  # 残差

            channal[i] = h
        # channal =
        batchs = torch.ones([self.sz_c, batch.size(0)]).to(self.device) * batch
        return self.layer_norm(channal), batchs

# print(isinstance(model,MultiGCNLayers))
