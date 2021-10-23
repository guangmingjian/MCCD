from model.layers.MultiGCNLayers import MultiGCNLayers
from model.layers.MixGCNLayers import MixGCNLayers
import torch as th
from model.layers.EmbeddingTransform import EmbeddingTransform
from model import CNNs
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class MCCD(nn.Module):
    def __init__(self, net_params):
        # nnf,150,32,3,4,10,32,name_model,num_class,device,0.0,0.8,True
        super(MCCD, self).__init__()
        # pprint(net_params)
        self.in_dim = net_params['in_dim']

        self.out_dim = net_params['out_dim']
        self.gcn_nums = net_params['gcn_nums']
        self.dropout = net_params['dropout']

        self.graph_norm = False
        self.sz_c = net_params['sz_c']
        self.gcn_h_dim = net_params["gcn_h_dim"]
        self.alpha = net_params["alpha"]
        # self.gcn_out = net_params["gcn_out"]
        self.device = net_params["device"]
        # **********************************************************************************************************
        self.arc_type = net_params["arc_type"]  # gcn / ensemble
        cnn_name = "LeNet1"
        self.d_kv = 32
        self.cnn_net = CNNs.__dict__[cnn_name](size_c = self.sz_c,num_class=self.out_dim, dropout=self.dropout)

        if self.arc_type == "gcn":
            self.MGL = MultiGCNLayers(self.in_dim, self.gcn_h_dim, self.gcn_h_dim, self.sz_c, self.gcn_nums,
                                      self.dropout,
                                      self.device, self.graph_norm)
        elif self.arc_type == "ensemble":
            self.MGL = MixGCNLayers(self.in_dim, self.gcn_h_dim, self.gcn_h_dim, self.gcn_nums, self.dropout,
                                    self.device, self.graph_norm)

        # self.sortn = SortNodeLayer(self.gcn_h_dim)
        # ***********************************************************************************************************

        # self.VLRGAT = VLRGAT(self.sz_c, self.d_kv, self.device, self.n_head, self.gcn_h_dim, net_params["att_drop_ratio"])
        self.EmTran = EmbeddingTransform(self.d_kv,self.device,self.gcn_h_dim,net_params["att_drop_ratio"])
        self.out1 = None
        self.fc = nn.Linear(self.gcn_h_dim, self.out_dim)



    def reset_parameters(self):

        all_res = [self.cnn_net, self.MGL, self.EmTran, self.out1, self.fc]
        for res in all_res:
            if res != None:
                res.reset_parameters()

    def forward(self, data, edge_index, batch):
        #  multi-channel encoder
        z, batches = self.MGL(data, edge_index, batch)
        # weak decoder
        gcn_z = self.fc(z)
        gcn_z = th.mean(gcn_z, dim=0)
        gcn_z = gnn.global_mean_pool(gcn_z, batch)
        self.out1 = F.log_softmax(gcn_z, dim=-1)

        # CNN Decoder
        batch_data = self.EmTran(z, batch)  # VLR
        batch_data = self.cnn_net(batch_data) # CPB
        return F.log_softmax(batch_data, dim=-1)

    def loss(self, y_pre, y_true):
        la = F.nll_loss(self.out1, y_true)
        lb = F.nll_loss(y_pre, y_true)
        return lb + th.exp((self.alpha / th.exp(la) - (1 - self.alpha) * th.tanh(la - lb)) ** 2)
