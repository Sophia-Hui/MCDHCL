import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from Params import args


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class myModel(nn.Module):
    def __init__(self, userNum, itemNum, behavior, behavior_mats): 
        super(myModel, self).__init__()
        self.userNum = userNum
        self.itemNum = itemNum
        self.behavior = behavior
        self.behavior_mats = behavior_mats
        self.subgradivflag = 1
        self.groups = args.groups
        self.group_user_list = [[None]*self.groups for i in range(len(self.behavior))]
        self.group_item_list = [[None]*self.groups for i in range(len(self.behavior))]
        
        self.embedding_dict = self.init_embedding()
        self.weight_dict = self.init_weight()

        self.user_embedding0, self.item_embedding0 = self.init_embeddings()

        self.W_subdiv_1, self.b_subdiv_1, self.W_subdiv_2, self.b_subdiv_2, self.W_subdiv, self.b_subdiv = self.init_gradiv_weight()

        self.gcn = GCN(self.userNum, self.itemNum, self.behavior, self.behavior_mats, self.subgradivflag,
                       self.group_user_list, self.group_item_list)


        user_embed1, item_embed1, user_embeds1, item_embeds1,self.item_embedding_g,self.user_embedding_g = self.gcn()

        ego_embedding = torch.cat([self.user_embedding0, self.item_embedding0],dim = 0)

        embedding_1 = torch.cat([user_embed1, item_embed1],dim=0)

        user_group_embeddings_side = ego_embedding.cuda() + embedding_1.cuda()

        user_group_embeddings_hidden_1 = torch.nn.functional.relu(torch.matmul(user_group_embeddings_side,self.W_subdiv_1.cuda()) + self.b_subdiv_1.cuda())

        m = torch.nn.Dropout(p=0.6)  #
        user_group_embeddings_hidden_d1 = m(user_group_embeddings_hidden_1 )
        user_group_embeddings_sum = torch.matmul(user_group_embeddings_hidden_d1.cuda(), self.W_subdiv.cuda()) + self.b_subdiv.cuda()
        top_g,top_idx_g = torch.topk(user_group_embeddings_sum,k=1,dim=1)
        user_group_embeddings = (torch.eq(user_group_embeddings_sum,top_g)).float()
        u_group_embeddings, i_group_embeddings = torch.split(user_group_embeddings, [self.userNum, self.itemNum], 0)
        i_group_embeddings = torch.ones(i_group_embeddings.shape)


        self.u_groupdiv_embeddings = torch.transpose(u_group_embeddings,1,0)
        self.i_groupdiv_embeddings = torch.transpose(i_group_embeddings,1,0)


        for i in range(len(self.behavior)):
            for g in range(0,self.groups):
                temp_item = (self.behavior_mats[i]['A'].to_dense()).__mul__(self.i_groupdiv_embeddings[g].cuda()).__mul__(
                    self.u_groupdiv_embeddings[g].cuda().unsqueeze(1))
                temp_user = (self.behavior_mats[i]['AT'].to_dense()).__mul__(self.u_groupdiv_embeddings[g].cuda()).__mul__(
                    self.i_groupdiv_embeddings[g].cuda().unsqueeze(1))

                temp_item = temp_item.to_sparse()
                temp_user = temp_user.to_sparse()
                self.group_item_list[i][g] = temp_item
                self.group_user_list[i][g] = temp_user

        self.gcn = GCN(self.userNum, self.itemNum, self.behavior, self.behavior_mats, self.subgradivflag,
                       self.group_user_list, self.group_item_list)


    def forward(self):

        self.subgradivflag=0


        user_embed, item_embed, user_embeds, item_embeds,item_embedding_g,user_embedding_g = self.gcn()
        return user_embed, item_embed, user_embeds, item_embeds,self.u_groupdiv_embeddings,item_embedding_g,user_embedding_g

    def init_gradiv_weight(self):
        W_subdiv_1 = nn.Parameter(torch.Tensor(args.hidden_dim, args.hidden_dim))
        b_subdiv_1 = nn.Parameter(torch.Tensor(1, args.hidden_dim))
        W_subdiv_2 = nn.Parameter(torch.Tensor(args.hidden_dim, args.hidden_dim))
        b_subdiv_2 = nn.Parameter(torch.Tensor(1, args.hidden_dim))
        W_subdiv = nn.Parameter(torch.Tensor(args.hidden_dim, self.groups))
        b_subdiv = nn.Parameter(torch.Tensor(1, self.groups))

        init.xavier_uniform_(W_subdiv_1)
        init.xavier_uniform_(b_subdiv_1)
        init.xavier_uniform_(W_subdiv_2)
        init.xavier_uniform_(b_subdiv_2)
        init.xavier_uniform_(W_subdiv)
        init.xavier_uniform_(b_subdiv)

        return W_subdiv_1,b_subdiv_1,W_subdiv_2,b_subdiv_2,W_subdiv,b_subdiv

    def init_embeddings(self):

        user_embedding = nn.Parameter(torch.Tensor(self.userNum, args.hidden_dim))
        item_embedding = nn.Parameter(torch.Tensor(self.itemNum, args.hidden_dim))
        nn.init.xavier_uniform_(user_embedding)
        nn.init.xavier_uniform_(item_embedding)

        return user_embedding, item_embedding

    def init_embedding(self):
        embedding_dict = {  
            'user_embedding': None,
            'item_embedding': None,
            'user_embeddings': None,
            'item_embeddings': None,
        }
        return embedding_dict

    def init_weight(self):  
        initializer = nn.init.xavier_uniform_
        
        weight_dict = nn.ParameterDict({
            'w_self_attention_item': nn.Parameter(initializer(torch.empty([args.hidden_dim, args.hidden_dim]))),
            'w_self_attention_user': nn.Parameter(initializer(torch.empty([args.hidden_dim, args.hidden_dim]))),
            'w_self_attention_cat': nn.Parameter(initializer(torch.empty([args.head_num*args.hidden_dim, args.hidden_dim]))),
            'alpha': nn.Parameter(torch.ones(2)),
        })      
        return weight_dict

    def para_dict_to_tenser(self, para_dict): 
    
        tensors = []
        for beh in para_dict.keys():
            tensors.append(para_dict[beh])
        tensors = torch.stack(tensors, dim=0)

        return tensors.float()

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_parameters(), source_params):
                name_t, param_t = tgt
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_parameters()(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  
                    self.set_param(self, name, param)

class GCN(nn.Module):
    def __init__(self, userNum, itemNum, behavior, behavior_mats,subgradivflag,group_user_list, group_item_list):
        super(GCN, self).__init__()  
        self.userNum = userNum
        self.itemNum = itemNum
        self.hidden_dim = args.hidden_dim

        self.behavior = behavior
        self.behavior_mats = behavior_mats

        self.user_embedding, self.item_embedding,self.relation_embedding0 = self.init_embedding()
        self.group = args.groups
        self.sub_flag = subgradivflag
        self.group_user_list = group_user_list
        self.group_item_list = group_item_list

        self.item_embedding_g = [[None]*self.group for i in range(len(self.behavior))]
        self.user_embedding_g = [[None]*self.group for i in range(len(self.behavior))]

        for i in range(len(self.behavior)):
            for g in range(0, self.group):
                self.item_embedding_g[i][g] = self.item_embedding
                self.user_embedding_g[i][g] = self.user_embedding

        self.alpha, self.i_concatenation_w, self.u_concatenation_w, self.i_input_w, self.u_input_w = self.init_weight()

        #-------------------------------------------------
        self.sigmoid = torch.nn.Sigmoid()
        self.act = torch.nn.PReLU()

        self.dropout = torch.nn.Dropout(args.drop_rate) #
        self.gnn_layer = eval(args.gnn_layer)
        self.layers_ego = nn.ModuleList()
        self.Subgraph_layers = nn.ModuleList()

        self.layers_ego.append(GCNLayer(args.hidden_dim, args.hidden_dim, self.userNum, self.itemNum, self.behavior, self.behavior_mats))

        for i in range(0, len(self.gnn_layer)):  #
            self.Subgraph_layers.append(subGCNLayer(args.hidden_dim, args.hidden_dim, self.userNum, self.itemNum, self.behavior,
                                            self.behavior_mats,self.group_user_list, self.group_item_list))

    def init_embedding(self):

        user_embedding = torch.nn.Embedding(self.userNum, args.hidden_dim)
        item_embedding = torch.nn.Embedding(self.itemNum, args.hidden_dim)
        relation_embedding = nn.Parameter(torch.Tensor(len(self.behavior), args.hidden_dim))

        nn.init.xavier_uniform_(user_embedding.weight)  #
        nn.init.xavier_uniform_(item_embedding.weight)
        nn.init.xavier_uniform_(relation_embedding)

        return user_embedding, item_embedding,relation_embedding

    def init_weight(self):
        alpha = nn.Parameter(torch.ones(2))

        i_concatenation_w = nn.Parameter(torch.Tensor(len(eval(args.gnn_layer))*args.hidden_dim, args.hidden_dim))
        u_concatenation_w = nn.Parameter(torch.Tensor(len(eval(args.gnn_layer))*args.hidden_dim, args.hidden_dim))
        i_input_w = nn.Parameter(torch.Tensor(args.hidden_dim, args.hidden_dim))
        u_input_w = nn.Parameter(torch.Tensor(args.hidden_dim, args.hidden_dim))
        init.xavier_uniform_(i_concatenation_w)
        init.xavier_uniform_(u_concatenation_w)
        init.xavier_uniform_(i_input_w)
        init.xavier_uniform_(u_input_w)

        return alpha, i_concatenation_w, u_concatenation_w, i_input_w, u_input_w


    def forward(self, user_embedding_input=None, item_embedding_input=None):
        all_user_embeddings = []
        all_item_embeddings = []
        all_user_embeddingss = []
        all_item_embeddingss = []

        user_embedding = self.user_embedding.weight
        item_embedding = self.item_embedding.weight
        for i, layer in enumerate(self.layers_ego):
            user_embedding_ego, item_embedding_ego, user_embeddings_ego, item_embeddings_ego = layer(user_embedding, item_embedding) #

            all_user_embeddings.append(user_embedding_ego)
            all_item_embeddings.append(item_embedding_ego)
            all_user_embeddingss.append(user_embeddings_ego)
            all_item_embeddingss.append(item_embeddings_ego)

        for i, layer in enumerate(self.Subgraph_layers): # i=0,1,2
            if self.sub_flag == 1:

                for i, layer in enumerate(self.layers_ego):
                    user_embedding, item_embedding, user_embeddings, item_embeddings = layer(user_embedding, item_embedding)
                return user_embedding, item_embedding, user_embeddings, item_embeddings,self.item_embedding_g,self.user_embedding_g
            else:

                user_embedding, item_embedding, user_embeddings, item_embeddings,self.item_embedding_g,self.user_embedding_g = \
                    layer(self.item_embedding_g,self.user_embedding_g)
            all_user_embeddings.append(user_embedding)
            all_item_embeddings.append(item_embedding)
            all_user_embeddingss.append(user_embeddings)
            all_item_embeddingss.append(item_embeddings)

        user_embedding = torch.cat(all_user_embeddings, dim=1)
        item_embedding = torch.cat(all_item_embeddings, dim=1)
        user_embeddings = torch.cat(all_user_embeddingss, dim=2)
        item_embeddings = torch.cat(all_item_embeddingss, dim=2)

        user_embedding = torch.matmul(user_embedding, self.u_concatenation_w)
        item_embedding = torch.matmul(item_embedding, self.i_concatenation_w)
        user_embeddings = torch.matmul(user_embeddings, self.u_concatenation_w)
        item_embeddings = torch.matmul(item_embeddings, self.i_concatenation_w)

        return user_embedding, item_embedding, user_embeddings, item_embeddings ,self.item_embedding_g,self.user_embedding_g # [31882, 16], [31882, 16], [4, 31882, 16], [4, 31882, 16]


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, userNum, itemNum, behavior, behavior_mats):
        super(GCNLayer, self).__init__()
        self.behavior = behavior
        self.behavior_mats = behavior_mats

        self.userNum = userNum
        self.itemNum = itemNum

        self.act = torch.nn.Sigmoid()
        self.i_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.u_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        # 无用
        self.ii_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        init.xavier_uniform_(self.i_w)
        init.xavier_uniform_(self.u_w)

    def forward(self, user_embedding, item_embedding): #

        user_embedding_list = [None]*len(self.behavior)
        item_embedding_list = [None]*len(self.behavior)


        for i in range(len(self.behavior)):
            user_embedding_list[i] = torch.spmm(self.behavior_mats[i]['A'], item_embedding.cuda())
            item_embedding_list[i] = torch.spmm(self.behavior_mats[i]['AT'], user_embedding.cuda())

  
        #torch.stack： #把list转为大的tensor
        user_embeddings = torch.stack(user_embedding_list, dim=0)
        item_embeddings = torch.stack(item_embedding_list, dim=0)

        user_embedding = self.act(torch.matmul(torch.mean(user_embeddings, dim=0), self.u_w.cuda())).cuda()
        item_embedding = self.act(torch.matmul(torch.mean(item_embeddings, dim=0), self.i_w.cuda())).cuda()

        user_embeddings = self.act(torch.matmul(user_embeddings, self.u_w.cuda())).cuda()
        item_embeddings = self.act(torch.matmul(item_embeddings, self.i_w.cuda())).cuda()


        return user_embedding, item_embedding, user_embeddings, item_embeddings             

#------------------------------------------------------------------------------------------------------------------------------------------------

class subGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, userNum, itemNum, behavior, behavior_mats,group_user_list, group_item_list):
        super(subGCNLayer, self).__init__()
        self.behavior = behavior
        self.behavior_mats = behavior_mats

        self.userNum = userNum
        self.itemNum = itemNum

        self.act = torch.nn.Sigmoid()
        self.i_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.u_w = nn.Parameter(torch.Tensor(in_dim, out_dim))

        self.groups = args.groups
        self.group_user_list = group_user_list
        self.group_item_list = group_item_list
        init.xavier_uniform_(self.i_w)
        init.xavier_uniform_(self.u_w)


        self.ii_w = nn.Parameter(torch.Tensor(in_dim, out_dim))

    def forward(self, item_embedding_g,user_embedding_g):

        embedding_user_f = []
        embedding_item_f = []
        user_embeddings_ego = []
        item_embeddings_ego = []
        
        for i in range(len(self.behavior)):
            for g in range(0,self.group):

                temp_embed_user = torch.spmm(self.group_item_list[i][g], item_embedding_g[i][g])
                temp_embed_item = torch.spmm(self.group_user_list[i][g], user_embedding_g[i][g])

                item_embedding_g[i][g] = item_embedding_g[i][g] + temp_embed_item
                user_embedding_g[i][g] = user_embedding_g[i][g] + temp_embed_user

                embedding_user_f[g] = torch.spmm(self.self.behavior_mats[i]['A'],temp_embed_item)
                embedding_item_f[g] = torch.spmm(self.self.behavior_mats[i]['AT'], temp_embed_user)

            user_embeddings_ego[i] = torch.mean(embedding_user_f[g],0)
            item_embeddings_ego[i] = torch.mean(embedding_item_f[g],0)

        user_embeddings = torch.stack(user_embeddings_ego, dim=0)
        item_embeddings = torch.stack(item_embeddings_ego, dim=0)


        user_embedding = self.act(torch.matmul(torch.mean(user_embeddings, dim=0), self.u_w))
        item_embedding = self.act(torch.matmul(torch.mean(item_embeddings, dim=0), self.i_w))

        user_embeddings = self.act(torch.matmul(user_embeddings, self.u_w))
        item_embeddings = self.act(torch.matmul(item_embeddings, self.i_w))

        return user_embedding, item_embedding, user_embeddings, item_embeddings,item_embedding_g,user_embedding_g