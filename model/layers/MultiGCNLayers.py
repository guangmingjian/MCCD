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
import torch.nn.functional as F

gnn_dict = {
    "GCN": gnn.GCNConv,
    "GraphSAGE": gnn.SAGEConv,
    "GAT": gnn.GATConv
}


class MultiGCNLayers(nn.Module):
    def __init__(self, d_in, d_h, sz_c, sz_l, drop_rate, device, g_name='GCN', g_norm=False):
        super().__init__()
        self.sz_c = sz_c
        self.d_in = d_in
        self.d_h = d_h // sz_c
        self.sz_l = sz_l
        self.drop_rate = drop_rate
        self.device = device
        self.gcn_layer = nn.ModuleList(self.channal_block(d_in, self.d_h, g_name, g_norm))
        self.g_norm = g_norm
        self.dropout = nn.Dropout(self.drop_rate)
        self.layer_norm = nn.LayerNorm(self.d_h, eps=1e-6)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.gcn_layer:
            for f in layer:
                if type(f) in list(gnn_dict.values()) + [gnn.BatchNorm]:
                    f.reset_parameters()
        self.layer_norm.reset_parameters()

    def get_gnn(self, d_in, d_out, g_name):
        """"""
        if g_name == 'GAT':
            return gnn.GATConv(d_in, d_out, 1)  # 1 head
        else:
            return gnn_dict[g_name](d_in, d_out)

    def channal_block(self, d_in, d_h, g_name, g_norm):
        layer = []
        for c in range(self.sz_c):
            t_layer = []
            for l in range(self.sz_l):
                if l == 0:
                    t_layer.append(self.get_gnn(d_in, d_h, g_name))
                else:
                    t_layer.append(self.get_gnn(d_h, d_h, g_name))
                t_layer.append(nn.Dropout(self.drop_rate))
                if g_norm:
                    t_layer.append(gnn.GraphSizeNorm())
                    t_layer.append(gnn.BatchNorm(d_h))
                t_layer.append(nn.ReLU())
            layer.append(nn.ModuleList(t_layer))
            # layer.append(nn.Sequential(*t_layer))
        return layer
        # pass

    def forward(self, x, edge, batch):
        channal = torch.empty([self.sz_c, x.size(0), self.d_h]).to(self.device)
        # channal = []
        for c, layer in enumerate(self.gcn_layer):
            h = x
            for el in layer:
                rh = h
                if type(el) in list(gnn_dict.values()):
                    h = el(h, edge)
                elif type(el) == gnn.GraphSizeNorm:
                    h = el(h,batch)
                else:
                    h = el(h)
                if isinstance(el, nn.ReLU) and h.size(1) == rh.size(1):
                    h = h + rh  # 残差
            channal[c] = h
        return self.layer_norm(channal)

# print(isinstance(model,MultiGCNLayers))
