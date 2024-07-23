from model.layers.MultiGCNLayers import MultiGCNLayers
from model.layers.MixGCNLayers import MixGCNLayers
import torch as th
from model.layers.EmbeddingTransform import EmbeddingTransform
from model.layers.LeNet import LeNet
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from model.layers.VLRGAT import VLRGAT as TransformerET



class MCCD(nn.Module):
    def __init__(self, in_channels, out_channels, gcn_nums, dropout,
                 sz_c, graph_norm, gcn_h_dim, alpha, device, arc_type,beta,
                 d_kv, att_drop_ratio, gcn_dropout=0.0, cnn_out_ker=8,cnn_dropout=0.0,
                 hidden_kernel=8,is_cbn=False,is_em=False,is_transformer=False):
        # nnf,150,32,3,4,10,32,name_model,num_class,device,0.0,0.8,True
        super(MCCD, self).__init__()
        # pprint(net_params)
        self.beta = beta
        self.att_drop_ratio = att_drop_ratio
        self.d_kv = d_kv
        self.arc_type = arc_type
        self.device = device
        self.alpha = alpha
        self.gcn_h_dim = gcn_h_dim
        self.graph_norm = graph_norm
        self.sz_c = sz_c
        self.dropout = dropout
        self.gcn_nums = gcn_nums
        self.out_dim = out_channels
        self.in_dim = in_channels
        self.is_em = is_em
        if is_em:
            self.em_layer = nn.Sequential(nn.Linear(in_channels,in_channels//2),nn.ReLU(),
                                          nn.Linear(in_channels//2,64))
            in_channels = 64
        if self.arc_type != "ensemble":
            self.MGL = MultiGCNLayers(in_channels, gcn_h_dim, sz_c, gcn_nums, gcn_dropout, device, arc_type, graph_norm)
            self.gcn_h_dim = gcn_h_dim // sz_c
        else:
            self.gcn_h_dim = 64
            self.MGL = MixGCNLayers(in_channels, self.gcn_h_dim, self.gcn_nums, gcn_dropout,
                                    self.device, self.graph_norm)
            self.sz_c = 3
        # self.sortn = SortNodeLayer(self.gcn_h_dim)
        # ***********************************************************************************************************
        # self.VLRGAT = VLRGAT(self.sz_c, self.d_kv, self.device, self.n_head, self.gcn_h_dim, net_params["att_drop_ratio"])
        if not is_transformer:
            self.EmTran = EmbeddingTransform(self.d_kv, self.device, self.gcn_h_dim, self.att_drop_ratio)
        else:
            self.EmTran = TransformerET(sz_c,self.d_kv,device,1,self.gcn_h_dim, self.att_drop_ratio)
        kernal = 5
        if d_kv == 32:
            padding = 0
        else:
            padding = 1
        self.cnn_net = LeNet(self.d_kv, self.sz_c, kernal, hidden_kernel, cnn_out_ker, padding,cnn_dropout,is_cbn)
        self.out1 = None
        cnn_size = self.cnn_net.cal_conv_size()
        cnn_out_dim = cnn_out_ker * cnn_size * cnn_size
        self.fc1 = nn.Linear(cnn_out_dim, self.gcn_h_dim)
        self.fc2 = nn.Linear(self.gcn_h_dim, out_channels)
        self.gfc = nn.Linear(self.gcn_h_dim, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        all_res = [self.cnn_net, self.MGL, self.EmTran, self.fc1, self.fc2, self.gfc]
        for res in all_res:
            if res != None:
                res.reset_parameters()

    # @torchsnooper.snoop()
    def forward(self, data, edge_index, batch):
        if self.is_em:
            data = self.em_layer(data)
        #  multi-channel encoder
        z = self.MGL(data, edge_index, batch)
        # weak decoder
        self.out1 = th.mean(z, dim=0)
        self.out1 = gnn.global_mean_pool(self.out1, batch)
        self.out1 = self.gfc(self.out1)
        # self.out1 = F.log_softmax(self.out1, dim=-1)
        # CNN Decoder
        z = self.EmTran(z, batch)  # VLR
        # CPB
        z = self.cnn_net(z)
        z = F.relu(self.dropout(self.fc1(z)))
        z = self.fc2(z)
        return z
        # return F.log_softmax(z, dim=-1)

    def loss(self, y_pre, y_true):
        la = F.cross_entropy(self.out1, y_true)
        lb = F.cross_entropy(y_pre, y_true)
        # return lb + (self.alpha * la - self.beta * F.relu(la - lb)) ** 2
        with th.no_grad():
            self.l1 = la
            self.l2 = 1 - th.tanh(la - lb)
        # return lb + (self.alpha * la - self.beta * F.tanh(la - lb))**2
        # return lb + self.alpha * la + self.beta * torch.exp(-F.tanh(la - lb))
        return lb + self.alpha * la + self.beta * (1 - th.tanh(la - lb))
