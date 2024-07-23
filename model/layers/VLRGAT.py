#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/21 21:36
# @Author  : mingjian
# @Version : 1.0
# @File    : VLRGAT.py


import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

class VLRGAT(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, sz_c, d_kv, device, n_head, gcn_h_dim, att_drop_ratio):
        super().__init__()
        self.sz_c = sz_c
        self.d_kv = d_kv
        self.device = device
        self.MHA = MultiHeadAttention(n_head, gcn_h_dim, d_kv, att_drop_ratio)
        self.layer_norm = nn.LayerNorm(d_kv, eps=1e-6)

    def reset_parameters(self):
        self.MHA.reset_parameters()
        self.layer_norm.reset_parameters()

    # input is :
    # gnn_out : sz_c x n x gcn_out
    # batch:
    # target : from sz_c x n x gcn_out  → sz_b x sz_c x d_kv x gcn_out
    def forward(self, gnn_out, batch):
        sz_c = gnn_out.size(0)
        sz_b = len(th.unique(batch))
        batch_data = th.empty([sz_b, sz_c, self.d_kv, self.d_kv]).to(self.device)

        for cha in range(sz_c):
            # sz_b x n x dh
            batch_cha, mask = to_dense_batch(gnn_out[cha], batch)
            z_out = self.MHA(batch_cha, batch_cha, batch_cha)
            batch_data[:, cha, :, :] = z_out
        return self.layer_norm(batch_data)


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_kv, att_drop_ratio):
        super().__init__()

        self.n_head = n_head
        self.d_v = d_kv

        self.w_qs = nn.Linear(d_model, n_head * d_kv, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_kv, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_kv, bias=False)
        self.fc = nn.Linear(n_head * d_kv, d_kv, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_kv ** 0.5, attn_dropout=att_drop_ratio)
        self.layer_norm = nn.LayerNorm(d_kv, eps=1e-6)
        # self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        self.w_qs.reset_parameters()
        self.w_ks.reset_parameters()
        self.w_vs.reset_parameters()
        self.fc.reset_parameters()
        self.layer_norm.reset_parameters()

    # @torchsnooper.snoop()
    # multi channal  input : sz_c, d_n, d_model
    def forward(self, q, k, v, mask=None):
        d_v, n_head = self.d_v, self.n_head
        sz_c, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        # print("head size {:d}".format(n_head),"d_model size {:d}".format(q.size(-1)))
        # print("head size {:03d}, d_model size {:03d}",format(n_head,q.size(-1)))
        # print(self.w_qs(q).size())
        q = self.w_qs(q).view(sz_c, len_q, n_head, d_v)
        k = self.w_ks(k).view(sz_c, len_k, n_head, d_v)
        v = self.w_vs(v).view(sz_c, len_v, n_head, d_v)
        # print(v.size())

        # Transpose for attention dot product: c x n_h x n x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # print(v.size())

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: c x dv x n_h x dv
        # Combine the last two dimensions to concatenate all the heads together:  c x dv x (n_h *dv)
        q = q.transpose(1, 2).contiguous().view(sz_c, d_v, -1)

        # q = self.dropout(self.fc(q))
        # q =
        # c x dv x (n_h *d_model)
        # q += residual
        # print(q.size())

        q = self.fc(q)

        return self.layer_norm(q)


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    # input is : c x n_h x n x dv
    def forward(self, q, k, v, mask=None):
        #  c x n_h x n x dv -> c x n_h x n x n
        attn = th.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        # print(attn.size())
        #  c x n_h x n x dv -> c x n_h x dv x n
        output = th.matmul(v.transpose(2, 3), attn)
        # print(output.size())
        # print(v.size())
        #  c x n_h x n x dv -> c x n_h x dv x dv
        output = th.matmul(output, v)
        # print(output.size())

        return output, attn


if __name__ == '__main__':
    x = th.randn([3, 10, 5]).float()
    batch = th.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2]).long()
    model = VLRGAT(3, 32, "cpu", 1, 5, 0.0)
    print(model(x, batch).size())
