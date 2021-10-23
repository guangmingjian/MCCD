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


class MixGCNLayers(nn.Module):

    def __init__(self, d_in, d_h, d_out, sz_l, drop_rate, device, g_norm=False):
        super().__init__()
        # self.pooling_ratio = pooling_ratio
        self.d_in = d_in
        self.d_h = d_h
        self.d_out = d_out
        self.sz_l = sz_l
        self.drop_rate = drop_rate
        self.device = device
        self.g_norm = g_norm
        self.gcn_layer = nn.ModuleList(self.channal_block(gnn.GCNConv))
        self.gat_layer = nn.ModuleList(self.channal_gat_block())
        self.graph_conv_layer = nn.ModuleList(self.channal_block(gnn.SAGEConv))
        self.layer_norm = nn.LayerNorm(d_out, eps=1e-6)
        # self.Dropout = nn.Dropout(self.drop_rate)

    def reset_parameters(self):
        all_gnn_type = [gnn.GATConv,gnn.GCNConv,gnn.SAGEConv]
        for conv in self.gcn_layer:
            if type(conv) in all_gnn_type:
                conv.reset_parameters()
        for conv in self.gat_layer:
            if type(conv) in all_gnn_type:
                conv.reset_parameters()
        for conv in self.graph_conv_layer:
            if type(conv) in all_gnn_type:
                conv.reset_parameters()

    # def channal_gat_block(self):
    #     return [
    #         gnn.GATConv(self.d_in, self.d_h // 8, 8),
    #         nn.ReLU(),
    #         gnn.GraphSizeNorm(),
    #         gnn.BatchNorm(self.d_h // 8 * 8),
    #         gnn.GATConv(self.d_h // 8 * 8, self.d_out, 1),
    #         nn.ReLU(),
    #         gnn.GraphSizeNorm(),
    #         gnn.BatchNorm(self.d_out),
    #     ]

    def channal_gat_block(self):
        return [
            gnn.GATConv(self.d_in, self.d_h // 8, 8),
            nn.ReLU(),
            # gnn.GraphSizeNorm(),
            # gnn.BatchNorm(self.d_h // 8 * 8),
            gnn.GATConv(self.d_h // 8 * 8, self.d_out, 1),
            nn.ReLU()
            # gnn.GraphSizeNorm(),
            # gnn.BatchNorm(self.d_out),
        ]

    # def channal_sagepool_block(self):
    #     net_paras = {
    #         "gcn_nums" : 4,
    #         "gcn_h_dim": 128,
    #         "residual": True,
    #         "readout": "mean",
    #         "dropout": 0.5,
    #         "pooling_ratio": 0.5,
    #         "graph_norm": True
    #     }
    #     return SAGPooling(net_paras)

    def channal_block(self, graphconv):
        t_layer = []
        for l in range(self.sz_l):
            if l == 0:
                t_layer.append(graphconv(self.d_in, self.d_h))
            elif l == (self.sz_l - 1):
                t_layer.append(graphconv(self.d_h, self.d_out))
            else:
                t_layer.append(graphconv(self.d_h, self.d_h))
            # t_layer.append(nn.Dropout(self.drop_rate))
            if self.g_norm:
                t_layer.append(gnn.GraphSizeNorm())
                if l == (self.sz_l - 1):
                    t_layer.append(gnn.BatchNorm(self.d_out))
                else:
                    t_layer.append(gnn.BatchNorm(self.d_h))
            t_layer.append(nn.ReLU())

            # layer.append(nn.Sequential(*t_layer))
        return t_layer
        # pass

    def get_hiddens(self, layer, x, edge):
        h = x
        for m in layer:
            rh = h
            if isinstance(m, gnn.GCNConv) or isinstance(m, gnn.SAGEConv) or isinstance(m, gnn.GATConv):
                h = m(h, edge)
            else:
                # print(type(m))
                h = m(h)
            if isinstance(m, nn.ReLU) and h.size(1) == rh.size(1):
                h = h + rh  # 残差
        return h

    def forward(self, x, edge, batch):
        channal = torch.empty([3, x.size(0), self.d_out]).to(self.device)
        batchs = torch.ones([3,batch.size(0)]).to(self.device) * batch
        # batchs = torch.empty([3,batch.size(0),batch.size(1)])
        # for i in range()
        # channal = []
        # for i in range(self.sz_c):
        #     h = x
        #     m = self.gcn_layer[i]
        #     for el in m:
        #         rh = h
        #         if isinstance(el,gnn.GCNConv):
        #             h = el(h,edge)
        #         else:
        #             h = el(h)
        #         if isinstance(el,nn.ReLU) and h.size(1) == rh.size(1):
        #             h = h + rh # 残差

        # channal[i] = h
        # channal =
        for i, layer in zip(torch.arange(3), [self.gcn_layer, self.gat_layer, self.graph_conv_layer]):
            channal[i] = self.get_hiddens(layer, x, edge)
        return self.layer_norm(channal),batchs
