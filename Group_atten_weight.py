
import numpy as np
from numpy import random
import math

import torch as t
import torch.nn as nn
from Params import args
import torch.nn.functional as F

class interout_group_attention(nn.Module):
    def __init__(self, behaviors):
        super(interout_group_attention, self).__init__()
        self.behaviors = behaviors
        k = args.rank

        self.meta_netu1 = nn.Linear(args.hidden_dim * 3, args.hidden_dim, bias=True)
        self.mlp = MLP(args.hidden_dim, args.hidden_dim * k, args.hidden_dim // 2, k)
        self.mlp1 = MLP(args.hidden_dim, args.hidden_dim * k, args.hidden_dim // 2, args.hidden_dim)

        self.meta_netu2 = nn.Linear(args.hidden_dim * 2, args.hidden_dim, bias=True)
        self.mlp2 = MLP(args.hidden_dim, args.hidden_dim * k, args.hidden_dim // 2, k)
        self.mlp3 = MLP(args.hidden_dim, args.hidden_dim * k, args.hidden_dim // 2, args.hidden_dim)

    def forward(self,user_embeddings,user_embedding, infoNCELoss_group_list,meta_behavior_loss_list, SSL_user_step_index,
                 meta_user_index_list,user_id_groups,item_embedding_gk, user_embedding_gk):
        k = args.rank
        self.user_embeddings = user_embeddings

        self.infoNCELoss_group_list = infoNCELoss_group_list #
        self.behavior_loss_list = meta_behavior_loss_list
        self.user_step_index = SSL_user_step_index
        self.user_index_list = meta_user_index_list

        self.user_id_groups = user_id_groups  #
        self.item_embedding_gk = item_embedding_gk
        self.user_embedding_gk = user_embedding_gk

        weightu_all = []
        weightu_group = [None] * args.groups

        for i in range(self.behaviors):
            for g in range(args.groups):
                group_num = self.user_id_groups[g].shape[0]
                if group_num != 0:
                    group_index = self.user_id_groups[g].long()

                    temp3 = t.cat((self.infoNCELoss_group_list[i][g][:group_num].unsqueeze(1).repeat(1, args.hidden_dim).to(t.float32),
                                   self.user_embeddings[i][group_index],
                                   self.user_embedding_gk[i][g].weight[group_index].cuda()), dim=1).detach()
                    tembedu = (self.meta_netu1(temp3))  #.detach()
                    metau1 = self.mlp(tembedu).reshape(-1, k)
                    metau2 = self.mlp1(tembedu).reshape(-1, args.hidden_dim)

                    meta_biasu = (t.mean(metau1, dim=0))
                    meta_biasu1 = (t.mean(metau2, dim=0))

                    low_weightu1 = t.sum(F.softmax(metau1 + meta_biasu, dim=1),1).squeeze()
                    low_weightu2 = t.sum(F.softmax(metau2 + meta_biasu1, dim=1),1).squeeze()
                    weightu_group[g] = (low_weightu1 + low_weightu2) / 2


                else:
                    weightu_group[g] = t.zeros(1,1).cuda()
            weightu_all.append(weightu_group)


        weightu_behavior = []

        for b in range(self.behaviors):
            embu = self.meta_netu2(t.cat((self.behavior_loss_list[b].unsqueeze(1).repeat(1, args.hidden_dim),
                                          self.user_embeddings[b][self.user_index_list[b]]), dim=1).detach())
            embu1 = self.mlp2(embu).reshape(-1, k)  # d*k
            embu2 = self.mlp3(embu).reshape(-1, args.hidden_dim)  # k*d
            biasu1 = (t.mean(embu1, dim=0))
            biasu2 = (t.mean(embu2, dim=0))
            low_weightu3 = t.sum(F.softmax(embu1 + biasu1, dim=1), 1).squeeze()  # k行
            low_weightu4 = t.sum(F.softmax(embu2 + biasu2, dim=1), 1).squeeze()
            weightu_behavior.append((low_weightu3 + low_weightu4) / 2)  #

        weights = []
        weight = [None] * args.groups

        for i in range(self.behaviors):
            for g in range(args.groups):
                group_num = self.user_id_groups[g].shape[0]
                if group_num != 0:
                    group_index = self.user_id_groups[g].long()

                    tempembed_gk = t.mean(self.user_embedding_gk[i][g].weight[group_index].cuda(),dim=1).unsqueeze(0)
                    temlat = FC(tempembed_gk, tempembed_gk.shape[1] // 2, useBias=True, reg=True, activation='relu',reuse=True)
                    weight[g] = FC(temlat, 1, useBias=True, reg=True, reuse=True)
                else:
                    weight[g] = t.zeros(1,1).cuda() # t.zeros(1,1).cuda()
            weights.append(weight)

        return weightu_all, weightu_behavior,weights

    def multi_head_attenion(self):
        return

class MLP(t.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(MLP, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim, bias=True) #
        else:
            self.linear_first = nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)]) #
        self.linear_out = nn.Linear(feature_dim, output_dim, bias=True) #

    def forward(self, data):
        x = data
        if self.feature_pre:
            x = self.linear_pre(x) #d*k
        prelu = nn.PReLU().cuda()
        x = prelu(x)
        for i in range(self.layer_num - 2):
            x = self.linear_hidden[i](x)
            x = F.tanh(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.linear_out(x) # d*k
        x = F.normalize(x, p=2, dim=-1)
        return x

class BN(t.nn.Module):
    def __init__(self, num_features, eps=1e-8, momentum=0.5):
        super(BN, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.register_buffer('running_mean', t.zeros(num_features))
        self.register_buffer('running_var', t.ones(num_features))
        self.scale = t.nn.Parameter(t.ones(num_features))
        self.shift = t.nn.Parameter(t.zeros(num_features))

    def forward(self, inp):
        if self.training:
            mean, var = t.mean(inp, dim=0), t.var(inp, dim=0)
            with t.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * var)
        else:
            mean, var = self.running_mean, self.running_var
        ret = t.nn.functional.batch_norm(
            inp, mean, var, self.shift, self.scale, eps=self.eps, training=True)
        return ret

def Bias(data, name=None, reg=False, reuse=False):
    inDim = data.shape[-1]

    bias = t.zeros(inDim, dtype=t.float32, requires_grad=True).cuda()
    return (data + bias).cuda()


def FC(inp, outDim, name=None, useBias=False, activation=None, reg=False, useBN=False, dropout=None,initializer='xavier', reuse=False):
    global params
    global regParams
    global leaky
    inDim = inp.shape[1]
    W = nn.Parameter(t.nn.init.xavier_uniform_(t.empty([inDim, outDim], dtype=t.float32)), requires_grad=True).cuda()

    if dropout != None:
        dropout = nn.Dropout(p=dropout)
        ret = dropout(inp) @ W
    else:
        ret = (inp @ W).cuda()
    if useBias:
        ret = Bias(ret, name=name, reuse=reuse).cuda()
    if useBN:
        ret = BN(ret).cuda()
    if activation != None:
        ret = Activate(ret, activation).cuda()
    return ret.cuda()

def ActivateHelp(data, method):
    if method == 'relu':
        ret = t.nn.functional.relu(data)
    elif method == 'sigmoid':
        sigmoid= t.nn.Sigmoid()
        ret = sigmoid(data)
    elif method == 'tanh':
        tanh = t.nn.Tanh()
        ret = tanh(data)
    elif method == 'softmax':
        softmax = t.nn.Softmax()
        ret = softmax(data, axis=-1)
    elif method == 'leakyRelu':
        ret = t.maximum(leaky * data, data)
    elif method == 'twoWayLeakyRelu':
        temMask = (t.greater(data, 1.0)).to(t.float)
        ret = temMask * (1 + leaky * (data - 1)) + (1 - temMask) * t.maximum(leaky * data, data)
    elif method == '-1relu':
        ret = t.maximum(-1.0, data)
    elif method == 'relu6':
        ret = t.maximum(0.0, t.minimum(6.0, data))
    elif method == 'relu3':
        ret = t.maximum(0.0, t.minimum(3.0, data))
    else:
        raise Exception('Error Activation Function')
    return ret


def Activate(data, method, useBN=False):
    global leaky
    if useBN:
        ret = BN(data)
    else:
        ret = data
    ret = ActivateHelp(ret, method)
    return ret
