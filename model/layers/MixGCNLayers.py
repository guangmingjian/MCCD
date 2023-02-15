# -*- coding: utf-8 -*-
"""
   File Name：     MixGCNLayers
   Description :
   Author :       mingjian
   date：          2021/3/23
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


class MixGCNLayers(nn.Module):

    def __init__(self, d_in, d_h, sz_l, drop_rate, device, g_norm=False):
        super().__init__()
        # self.pooling_ratio = pooling_ratio
        self.d_in = d_in
        self.d_h = d_h
        self.sz_l = sz_l
        self.drop_rate = drop_rate
        self.device = device
        self.g_norm = g_norm
        self.gcn_layer = nn.ModuleList(self.channal_block('GCN'))
        self.graph_conv_layer = nn.ModuleList(self.channal_block('GraphSAGE'))
        self.gat_layer = nn.ModuleList(self.channal_block('GAT'))
        # self.Dropout = nn.Dropout(self.drop_rate)

    def reset_parameters(self):
        all_gnn_type = [gnn.GATConv, gnn.GCNConv, gnn.SAGEConv, gnn.BatchNorm]
        for conv in self.gcn_layer:
            if type(conv) in all_gnn_type:
                conv.reset_parameters()
        for conv in self.gat_layer:
            if type(conv) in all_gnn_type:
                conv.reset_parameters()
        for conv in self.graph_conv_layer:
            if type(conv) in all_gnn_type:
                conv.reset_parameters()

    def get_gnn(self, d_in, d_out, g_name,i):
        """"""
        if g_name == 'GAT':
            if i ==0:
                return gnn.GATConv(d_in, d_out // 8, 8)
            else:
                return gnn.GATConv(d_in // 8 * 8, d_out, 1)  # 1 head
        else:
            return gnn_dict[g_name](d_in, d_out)

    def channal_block(self, g_name):
        gat_list = []
        for i in range(self.sz_l):
            if i == 0:
                gat_list.append(self.get_gnn(self.d_in, self.d_h, g_name,i))
            else:
                gat_list.append(self.get_gnn(self.d_h, self.d_h, g_name,i))
            if self.g_norm:
                gat_list.append(gnn.GraphSizeNorm())
                gat_list.append(gnn.BatchNorm(self.d_h))
            gat_list = gat_list + [nn.ReLU(), nn.Dropout(self.drop_rate)]
        return gat_list

    def forward(self, x, edge, batch):
        channel = torch.empty([3, x.size(0), self.d_h]).to(self.device)
        for i, layer in enumerate([self.gcn_layer, self.gat_layer, self.graph_conv_layer]):
            h = x
            for f in layer:
                if type(f) in gnn_dict.values():
                    h = f(h, edge)
                elif type(f) == gnn.GraphSizeNorm:
                    h = f(h, batch)
                else:
                    h = f(h)
            channel[i] = h
        return channel
