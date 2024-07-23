#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/6 9:13
# @Author  : mingjian
# @Version : 1.0
# @File    : LeNet.py

import torch.nn as nn
import torch.nn.functional as F
import torch


class BasicLeNet(nn.Module):
    def __init__(self, in_dim=16 * 5 * 5, num_class=2, dropout=0.5):
        super(BasicLeNet, self).__init__()
        # self.fc1 = nn.Linear(in_dim, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_dim, 32)
        self.fc2 = nn.Linear(32, num_class)

    def read_out(self, out):
        out = out.view(out.size(0), -1)
        # print(out.size())
        # print(out.size())
        # out = self.dropout(self.fc1(out))
        # print(out.size())
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        return out


'''
in_dim is : 32 x 32
'''


class LeNet1(BasicLeNet):
    def __init__(self, size_c, **kwargs):
        super(LeNet1, self).__init__(in_dim=18 * 5 * 5, **kwargs)
        self.conv1 = nn.Conv2d(size_c, 16, 5)
        self.conv2 = nn.Conv2d(16, 18, 5)

    def reset_parameters(self):
        all_res = [self.conv1, self.conv2, self.fc1, self.fc2]
        for res in all_res:
            res.reset_parameters()

    def forward(self, x):
        out = F.relu(self.conv1(x))
        # out1 = out.detach()
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        # out2 = out.detach()
        out = F.max_pool2d(out, 2)
        return self.read_out(out)


class LeNet(nn.Module):
    def __init__(self, d_kv, size_c, kernel, hidden_kernel,out_kernel,padding=0,cnn_dropout=0.0,is_bn=False):
        super(LeNet, self).__init__()
        self.d_kv = d_kv
        self.padding = padding
        self.kernel = kernel
        self.conv1 = nn.Conv2d(size_c, hidden_kernel, kernel, padding=padding)
        self.conv2 = nn.Conv2d(hidden_kernel, out_kernel, kernel, padding=padding)
        if is_bn:
            self.bn = nn.ModuleList([nn.BatchNorm2d(hidden_kernel),nn.BatchNorm2d(out_kernel)])
        self.dropout = nn.Dropout(cnn_dropout)
        self.is_bn = is_bn
        self.reset_parameters()

    def reset_parameters(self):
        all_res = [self.conv1, self.conv2]
        for res in all_res:
            res.reset_parameters()
        if self.is_bn:
            for bn in self.bn:
                bn.reset_parameters()

    def cal_conv_size(self, ):
        conv_l = self.d_kv
        for _ in range(2):
            conv_l = (conv_l - self.kernel + 2 * self.padding) // 1 + 1
            # print(conv_l)
            conv_l = (conv_l - 2 + 2 * 0) // 2 + 1
            # print(conv_l)
        return conv_l

    def forward(self, x):
        for i,conv in enumerate([self.conv1,self.conv2]):
            x = conv(x)
            if self.is_bn:
                x = self.bn[i](x)
            x = self.dropout(F.relu(x))
            x = F.max_pool2d(x, 2)
        return x.view(x.size(0), -1)

if __name__ == '__main__':
    x = torch.randn([4,3,16,16])
    model = LeNet(16,3,3,1)
    print(model(x).size())
    print(model.cal_conv_size())