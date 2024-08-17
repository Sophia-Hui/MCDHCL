
import os
import numpy as np
from numpy import random
import pickle
from scipy.sparse import csr_matrix
import math
import gc
import time
import random
import datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch as t
import torch.nn as nn

import torch.utils.data as dataloader
import torch.nn.functional as F

import graph_utils
import DataHandler

import BGNN
import Group_atten_weight

from Params import args
from Utils.TimeLogger import log
from tqdm import tqdm


t.backends.cudnn.benchmark = True

if t.cuda.is_available():
    use_cuda = True
else:
    use_cuda = True

MAX_FLAG = 0x7FFFFFFF

now_time = datetime.datetime.now()
modelTime = datetime.datetime.strftime(now_time, '%Y_%m_%d__%H_%M_%S')

t.autograd.set_detect_anomaly(True)

class Model():
    def __init__(self):

        self.trn_file = args.path + args.dataset + '/trn_'
        self.tst_file = args.path + args.dataset + '/tst_int'

        self.meta_multi_single_file = args.path + args.dataset + '/meta_multi_single_beh_user_index_shuffle'
        #
        self.meta_multi_single = pickle.load(open(self.meta_multi_single_file, 'rb'))

        self.t_max = -1
        self.t_min = 0x7FFFFFFF
        self.time_number = -1
        self.user_num = -1
        self.item_num = -1

        self.behavior_mats = {}
        self.behaviors = []
        self.behaviors_data = {}

        self.train_loss = []
        self.his_hr = []
        self.his_ndcg = []
        gc.collect()

        self.relu = t.nn.ReLU()
        self.sigmoid = t.nn.Sigmoid()
        self.curEpoch = 0

        #
        if args.dataset == 'Tmall':
            self.behaviors_SSL = ['pv', 'fav', 'cart', 'buy']
            self.behaviors = ['pv', 'fav', 'cart', 'buy']
            # self.behaviors = ['buy']
        elif args.dataset == 'IJCAI_15':
            self.behaviors = ['click', 'fav', 'cart', 'buy']
            # self.behaviors = ['buy']
            self.behaviors_SSL = ['click', 'fav', 'cart', 'buy']

        elif args.dataset == 'MultiInt-ML10M':
            self.behaviors = ['review', 'browse', 'buy']
            self.behaviors_SSL = ['review', 'browse', 'buy']

        elif args.dataset == 'JD':
            self.behaviors = ['review', 'browse', 'buy']
            self.behaviors_SSL = ['review', 'browse', 'buy']

        elif args.dataset == 'retailrocket':
            self.behaviors = ['catDict', 'cart','fav','buy']
            # self.behaviors = ['buy']
            self.behaviors_SSL = ['catDict', 'cart','fav','buy']

        for i in range(0, len(self.behaviors)):
            with open(self.trn_file + self.behaviors[i], 'rb') as fs:
                data = pickle.load(fs)
                self.behaviors_data[i] = data  #
                if data.get_shape()[0] > self.user_num:

                    self.user_num = data.get_shape()[0]
                if data.get_shape()[1] > self.item_num:
                    self.item_num = data.get_shape()[1]

                if data.data.max() > self.t_max:
                    self.t_max = data.data.max()
                if data.data.min() < self.t_min:
                    self.t_min = data.data.min()

                if self.behaviors[i] == args.target:
                    self.trainMat = data
                    self.trainLabel = 1 * (self.trainMat != 0)  #
                    self.labelP = np.squeeze(np.array(np.sum(self.trainLabel, axis=0)))  #
        time = datetime.datetime.now()
        print("Start building:  ", time,file=code_log)
        print("Start building:  ", time)

        for i in range(0, len(self.behaviors)):
            '''behavior_mats = {}
            behavior_mats['A'] = matrix_to_tensor(normalize_adj(behaviors_data))
            behavior_mats['AT'] = matrix_to_tensor(normalize_adj(behaviors_data.T))'''
            self.behavior_mats[i] = graph_utils.get_use(self.behaviors_data[i])
        time = datetime.datetime.now()
        print ("End building:", time,file=code_log)
        print ("End building:", time)

        print("user_num: ", self.user_num,file=code_log)
        print("user_num: ", self.user_num)
        print("item_num: ", self.item_num,file=code_log)
        print("item_num: ", self.item_num)
        print("\n",file=code_log)


        train_u, train_v = self.trainMat.nonzero()
        train_data = np.hstack((train_u.reshape(-1, 1), train_v.reshape(-1,1))).tolist()

        train_dataset = DataHandler.RecDataset_beh(self.behaviors, train_data, self.item_num, self.behaviors_data, True)
        # args.batch:8192
        self.train_loader = dataloader.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=4,
                                                  pin_memory=True)
        with open(self.tst_file, 'rb') as fs:
            data = pickle.load(fs)

        test_user = np.array([idx for idx, i in enumerate(data) if i is not None])
        test_item = np.array([i for idx, i in enumerate(data) if i is not None])

        test_data = np.hstack((test_user.reshape(-1, 1), test_item.reshape(-1, 1))).tolist()
        test_dataset = DataHandler.RecDataset(test_data, self.item_num, self.trainMat, 0, False)  #
        self.test_loader = dataloader.DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=4,
                                                 pin_memory=True)

    def negSamp(self, temLabel, sampSize, nodeNum):
        negset = [None] * sampSize
        cur = 0
        while cur < sampSize:
            rdmItm = np.random.choice(nodeNum)
            if temLabel[rdmItm] == 0:
                negset[cur] = rdmItm
                cur += 1
        return negset

    def sampleTrainBatch(self, batIds, labelMat):
        temLabel = labelMat[batIds.cpu()].toarray()
        batch = len(batIds)
        user_id = []
        item_id_pos = []
        item_id_neg = []
        cur = 0
        for i in range(batch):
            posset = np.reshape(np.argwhere(temLabel[i] != 0), [-1])
            sampNum = min(args.sampNum, len(posset))
            if sampNum == 0:
                poslocs = [np.random.choice(labelMat.shape[1])]
                neglocs = [poslocs[0]]
            else:
                poslocs = np.random.choice(posset, sampNum)
                neglocs = self.negSamp(temLabel[i], sampNum, labelMat.shape[1])

            for j in range(sampNum):
                user_id.append(batIds[i].item())
                item_id_pos.append(poslocs[j].item())
                item_id_neg.append(neglocs[j])
                cur += 1

        return t.as_tensor(np.array(user_id)).cuda(), t.as_tensor(np.array(item_id_pos)).cuda(), t.as_tensor(
            np.array(item_id_neg)).cuda()

    def innerProduct(self, u, i, j):
        pred_i = t.sum(t.mul(u, i), dim=1) * args.inner_product_mult
        pred_j = t.sum(t.mul(u, j), dim=1) * args.inner_product_mult
        return pred_i, pred_j

    def SSL(self, user_embeddings, item_embeddings, target_user_embeddings, target_item_embeddings,
            user_step_index,group_user_list):

        def one_neg_sample_pair_index(i, step_index, embedding1, embedding2):

            index_set = set(np.array(step_index))
            index_set.remove(i.item())
            neg2_index = t.as_tensor(np.array(list(index_set))).long().cuda()

            neg1_index = t.ones((2,), dtype=t.long)
            neg1_index = neg1_index.new_full((len(index_set),), i)

            neg_score_pre = t.sum(compute(embedding1, embedding2, neg1_index, neg2_index).squeeze())
            return neg_score_pre

        def multi_groups_pos_pair(batch_user_group, user_id_group, embedding1,embedding2):  #

            group_set = set(np.array(user_id_group.cpu())) # {806} {436}
            batch_group_set = set(np.array(batch_user_group.cpu())) # {18} {9}
            pos2_group_set = group_set - batch_group_set  # beh {set:788} {428}
            pos2_group = t.as_tensor(np.array(list(pos2_group_set))).long().cuda()
            pos2_group = t.unsqueeze(pos2_group, 0)  # [1,910]  [1,788]  {Tensor(1,428)}

            pos2_group = pos2_group.repeat(len(batch_user_group), 1)  #
            pos2_group = t.reshape(pos2_group, (1, -1))  # [1, 91000]  [1,14184]
            pos2_group = pos2_group.squeeze(0)

            pos1_group = batch_user_group.long().cuda()
            pos1_group = t.unsqueeze(pos1_group, 1)  # [100, 1]
            pos1_group = pos1_group.repeat(1, len(pos2_group_set))
            pos1_group = t.reshape(pos1_group, (1, -1))  # [1, 91000]  [1,14184]
            pos1_group = pos1_group.squeeze(0) # [91000]  [14184]

            temp10 = compute(embedding1, embedding2, pos1_group, pos2_group).squeeze(-1).view(len(batch_user_group), -1)

            pos_score_pre = t.sum(temp10,-1)

            return pos_score_pre  # [100]  [18]

        def multi_groups_neg_pair(batch_user_group,user_id_group,step_index, embedding1,embedding2):
            all_groups_set = set(np.array(step_index.cpu())) #{806}
            group_set = set(np.array(user_id_group.cpu())) #{18}
            neg2_groups_set = all_groups_set - group_set  #
            neg2_groups = t.as_tensor(np.array(list(neg2_groups_set))).long().cuda()  # [910]
            neg2_groups = t.unsqueeze(neg2_groups, 0)  # [1,910]  [1,788]

            neg2_groups = neg2_groups.repeat(len(batch_user_group), 1)
            neg2_groups = t.reshape(neg2_groups, (1, -1))  # [1, 91000]  [1,14184]
            neg2_groups = neg2_groups.squeeze(0) #n

            neg1_group = batch_user_group.long().cuda()  # [100] [18,]
            neg1_group = t.unsqueeze(neg1_group, 1)  # [100, 1]
            neg1_group = neg1_group.repeat(1, len(neg2_groups_set))
            neg1_group = t.reshape(neg1_group, (1, -1))  # [1, 91000]  [1,14184]
            neg1_group = neg1_group.squeeze(0)

            temp20 = compute(embedding1, embedding2, neg1_group, neg2_groups).squeeze(-1).view(len(batch_user_group), -1)
            neg_score_pre = t.sum(temp20,-1)

            return neg_score_pre  # [100]  [18]

        def compute(x1, x2, neg1_index=None, neg2_index=None, τ=0.05):

            if neg1_index != None:
                #print(neg1_index)
                x1 = x1[neg1_index]
                x2 = x2[neg2_index]

            N = x1.shape[0]
            D = x1.shape[1]

            x1 = x1 #[5157,16]
            x2 = x2 #[5157,16]

            temp1 = x1.view(N, 1, D) # {Tensor(5157,1,16)}
            temp2 = x2.view(N, D, 1) # {Tensor(5157,16,1)}
            temp3 = t.bmm(temp1,temp2)
            temp4 = temp3.view(N, 1) # {Tensor(5157,1)}
            temp5 = t.div(temp4,np.power(D,1) + 1e-8)
            scores = t.exp( temp5 )

            return scores

        def single_infoNCE_loss_one_by_one(embedding1, embedding2, step_index,user_id_groups):  # target, beh

            con_loss_groups = [None] * args.groups

            maxid = 0
            for group in range(args.groups):
                user_id_group = user_id_groups[group] # {Tensor(339,)}

                N = user_id_group.shape[0]
                if N>maxid: maxid = N

                if N != 0:
                    steps = int(np.ceil(N / args.SSL_batch))  # 46     separate the batch to smaller one
                    pos_group_score = t.zeros((N,), dtype=t.float32).cuda()  #dtype=t.float64
                    neg_group_score = t.zeros((N,), dtype=t.float32).cuda()

                    for i in range(steps):
                        st = i * args.SSL_batch
                        ed = min((i + 1) * args.SSL_batch, N)
                        batch_user_group = user_id_group[st: ed] # 18

                        pos_score = multi_groups_pos_pair(batch_user_group, user_id_group, embedding1,embedding2) #.squeeze() # 1   ##[batch_user_group] [819]

                        neg_score = multi_groups_neg_pair(batch_user_group,user_id_group,step_index,embedding1,embedding2)  # 18    #[batch_user_group] #{Tensor:(18,)}

                        if i == 0:
                            pos_group_score = pos_score
                            neg_group_score = neg_score
                        else:
                            pos_group_score = t.cat((pos_group_score, pos_score), 0) #[user_id_group]
                            neg_group_score = t.cat((neg_group_score, neg_score), 0)

                    con_loss_group = -t.log(1e-8 + t.div(pos_group_score, neg_group_score + 1e-8))
                    assert not t.any(t.isnan(con_loss_group))
                    assert not t.any(t.isinf(con_loss_group))

                    con_loss_groups[group] = con_loss_group
                else:
                    con_loss_groups[group] = t.zeros(1).cuda()

            for g in range(args.groups):
                con_loss_groups[g] = t.nn.functional.pad(con_loss_groups[g],
                                                         pad=(0,maxid-(con_loss_groups[g].shape[0])),mode='constant', value=0)
            con_loss_groups = t.stack(con_loss_groups)

            return t.where(t.isnan(con_loss_groups), t.full_like(con_loss_groups, 0 + 1e-8),con_loss_groups)

        user_con_loss_list = []
        user_id_groups = [None] * args.groups

        if user_step_index.shape[0]>800:
            SSL_len = int(user_step_index.shape[0] / 8)

            user_step_index = t.as_tensor(np.random.choice(user_step_index.cpu(), size=SSL_len, replace=False, p=None)).cuda()

        user_step_index = t.sort(user_step_index)[0].cuda()
        user_groups_temp = [None] * (args.groups)

        for g in range(args.groups):
            temp_tensor = t.nonzero(group_user_list[g]).squeeze(-1)
            user_groups_temp[g] = [int(user_step_index[i] in temp_tensor) for i in range(len(user_step_index))]  # list2

        user_groups_temp = (t.as_tensor(user_groups_temp).cuda()).__mul__(user_step_index.cuda())

        #
        for a in range(args.groups):

            temp1 = (t.nonzero(user_groups_temp[a])).squeeze(-1)
            user_id_groups[a] = user_groups_temp[a][temp1]
        a = 1

        for i in range(len(self.behaviors_SSL)):

            user_con_loss_list.append(single_infoNCE_loss_one_by_one(user_embeddings[-1], user_embeddings[i], user_step_index,user_id_groups)) # user_step_index:排序后的index

        return user_con_loss_list,user_step_index,user_id_groups  # 4*[1024]

    def trainEpoch(self):
        train_loader = self.train_loader  #
        time = datetime.datetime.now()
        print("start_ng_samp:  ", time,file=code_log)
        print("start_ng_samp:  ", time)
        train_loader.dataset.ng_sample()
        time = datetime.datetime.now()
        print("end_ng_samp:  ", time,file=code_log)
        print("end_ng_samp:  ", time)

        epoch_loss = 0
        # -----------------------------------------------------------------------------------
        self.behavior_loss_list = [None] * len(self.behaviors)  #

        self.user_id_list = [None] * len(self.behaviors)  #
        self.item_id_pos_list = [None] * len(self.behaviors)  #
        self.item_id_neg_list = [None] * len(self.behaviors)

        self.meta_start_index = 0
        self.meta_end_index = self.meta_start_index + args.meta_batch
        # ----------------------------------------------------------------------------------
        cnt = 0

        for user, item_i, item_j in tqdm(train_loader):
            user = user.long().cuda() # {Tensor(8192,)}
            self.user_step_index = user


            self.meta_user = t.as_tensor(self.meta_multi_single[self.meta_start_index:self.meta_end_index]).cuda()
            # --------
            if self.meta_end_index == self.meta_multi_single.shape[0]:
                self.meta_start_index = 0
            else:
                self.meta_start_index = (self.meta_start_index + args.meta_batch) % (
                        self.meta_multi_single.shape[0] - 1)
            self.meta_end_index = min(self.meta_start_index + args.meta_batch, self.meta_multi_single.shape[0])

            self.infoNCELoss_list = [[None] * args.groups for i in range(len(self.behaviors))]
            self.infoNCELoss_lst = [[None] * args.groups for i in range(len(self.behaviors))]

            user_embed, item_embed, user_embeds, item_embeds,group_user_list,item_embedding_gk,user_embedding_gk = self.model()

            maxnum = 0
            for index in range(len(self.behaviors)):
                not_zero_index = np.where(item_i[index].cpu().numpy() != -1)[0]
                self.user_id_list[index] = user[not_zero_index].long().cuda()
                self.item_id_pos_list[index] = item_i[index][not_zero_index].long().cuda()
                self.item_id_neg_list[index] = item_j[index][not_zero_index].long().cuda()

                userEmbed = user_embed[self.user_id_list[index]]
                posEmbed = item_embed[self.item_id_pos_list[index]]
                negEmbed = item_embed[self.item_id_neg_list[index]]

                pred_i, pred_j = 0, 0
                pred_i, pred_j = self.innerProduct(userEmbed, posEmbed, negEmbed)

                self.behavior_loss_list[index] = - (pred_i.view(-1) - pred_j.view(-1)).sigmoid().log()

                if (self.behavior_loss_list[index].shape[0] > maxnum): maxnum = self.behavior_loss_list[index].shape[0]

            self.infoNCELoss_listt, SSL_user_step_index,user_id_groups = self.SSL(user_embeds, item_embeds, user_embed, item_embed,
                                                                  self.meta_user,group_user_list) # index:self.meta_user

            self.group_atten_weight3, self.weight_behavior3, self.group_weight3 = self.getgroupWeights(
                                                                        user_embeds, user_embed,
                                                                        self.infoNCELoss_listt,self.behavior_loss_list,
                                                                        SSL_user_step_index, self.user_id_list, user_id_groups,
                                                                        item_embedding_gk, user_embedding_gk)

            for i in range(len(self.behaviors)):
                for g in range(args.groups):
                    group_num = user_id_groups[g].shape[0]
                    self.infoNCELoss_list[i][g] = (self.infoNCELoss_listt[i][g][:group_num] * self.group_atten_weight3[i][g]).sum()

            for i in range(len(self.behaviors)): #
                self.infoNCELoss_list[i] = t.sum((t.stack(self.infoNCELoss_list[i]) * t.stack(self.group_weight3[i]).squeeze()),dim=-1)

            infoNCELoss = sum(self.infoNCELoss_list) / len(self.infoNCELoss_list)

            for i in range(len(self.behaviors)):
                self.behavior_loss_list[i] = (self.behavior_loss_list[i] * self.weight_behavior3[i]).sum()
            bprloss = sum(self.behavior_loss_list) / len(self.behavior_loss_list)

            regLoss = (t.norm(userEmbed) ** 2 + t.norm(posEmbed) ** 2 + t.norm(negEmbed) ** 2)

            loss = (bprloss + args.reg * regLoss + args.beta * infoNCELoss) / args.batch

            self.Groupweights_opt.zero_grad(set_to_none=True)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)

            nn.utils.clip_grad_norm_(self.getgroupWeights.parameters(), max_norm=20, norm_type=2)
            self.Groupweights_opt.step()
            self.opt.step()
            # ---round three---------------------------------------------------------------------------------------------
            cnt += 1

        return epoch_loss, user_embed, item_embed, user_embeds, item_embeds

    # ----------------------------
    def sampleTestBatch(self, batch_user_id, batch_item_id):

        batch = len(batch_user_id)
        tmplen = (batch * 100)

        sub_trainMat = self.trainMat[batch_user_id].toarray()
        user_item1 = batch_item_id
        user_compute = [None] * tmplen
        item_compute = [None] * tmplen
        user_item100 = [None] * (batch)

        cur = 0
        for i in range(batch):
            pos_item = user_item1[i]
            negset = np.reshape(np.argwhere(sub_trainMat[i] == 0), [-1])
            pvec = self.labelP[negset]
            pvec = pvec / np.sum(pvec)

            random_neg_sam = np.random.permutation(negset)[:99]
            user_item100_one_user = np.concatenate((random_neg_sam, np.array([pos_item])))
            user_item100[i] = user_item100_one_user

            for j in range(100):
                user_compute[cur] = batch_user_id[i]
                item_compute[cur] = user_item100_one_user[j]
                cur += 1

        return user_compute, item_compute, user_item1, user_item100

    def calcRes(self, pred_i, user_item1, user_item100):
        hit = 0
        ndcg = 0
        for j in range(pred_i.shape[0]):

            _, shoot_index = t.topk(pred_i[j], args.shoot)
            shoot_index = shoot_index.cpu()
            shoot = user_item100[j][shoot_index]
            shoot = shoot.tolist()
            if type(shoot) != int and (user_item1[j] in shoot):
                hit += 1
                ndcg += np.reciprocal(np.log2(shoot.index(user_item1[j]) + 2))
            elif type(shoot) == int and (user_item1[j] == shoot):
                hit += 1
                ndcg += np.reciprocal(np.log2(0 + 2))

        return hit, ndcg  # int, float

    def testEpoch(self, data_loader, save=False):
        epochHR, epochNDCG = [0] * 2
        with t.no_grad():
            user_embed, item_embed, user_embeds, item_embeds,group_user_list,item_embedding_gk,user_embedding_gk = self.model()

        cnt = 0
        tot = 0
        for user, item_i in data_loader:
            user_compute, item_compute, user_item1, user_item100 = self.sampleTestBatch(user, item_i)
            userEmbed = user_embed[user_compute]  # [614400, 16], [147894, 16]
            itemEmbed = item_embed[item_compute]

            pred_i = t.sum(t.mul(userEmbed, itemEmbed), dim=1)  # torch.Size([729900])

            hit, ndcg = self.calcRes(t.reshape(pred_i, [user.shape[0], 100]), user_item1, user_item100)
            epochHR = epochHR + hit
            epochNDCG = epochNDCG + ndcg  #
            cnt += 1
            tot += user.shape[0]

        result_HR = epochHR / tot
        result_NDCG = epochNDCG / tot
        print(f"Step {cnt}:  hit:{result_HR}, ndcg:{result_NDCG}",file=code_log)
        print(f"Step {cnt}:  hit:{result_HR}, ndcg:{result_NDCG}")

        return result_HR, result_NDCG

    # ---------------------------------------------------------------------------------------------
    def setRandomSeed(self):
        np.random.seed(args.seed)
        t.manual_seed(args.seed)
        t.cuda.manual_seed(args.seed)
        random.seed(args.seed)

    # -------------------------------------

    def getModelName( self):
        title = args.title
        ModelName = \
            args.point + \
            "_" + title + \
            "_" + args.dataset + \
            "_" + modelTime + \
            "_lr_" + str(args.lr) + \
            "_reg_" + str(args.reg) + \
            "_batch_size_" + str(args.batch) + \
            "_gnn_layer_" + str(args.gnn_layer)

        return ModelName

    def loadModel(self, loadPath):
        ModelName = self.modelName

        loadPath = loadPath
        checkpoint = t.load(loadPath)
        self.model = checkpoint['model']

        self.curEpoch = checkpoint['epoch'] + 1

        history = checkpoint['history']
        self.train_loss = history['loss']
        self.his_hr = history['HR']
        self.his_ndcg = history['NDCG']

    def saveHistory(self):
        history = dict()
        history['loss'] = self.train_loss
        history['HR'] = self.his_hr
        history['NDCG'] = self.his_ndcg
        ModelName = self.modelName

        with open(args.path_ + r'/History/' + args.dataset + r'/' + ModelName + '.his', 'wb') as fs:
            pickle.dump(history, fs)

    def saveModel(self):
        ModelName = self.modelName
        history = dict()
        history['loss'] = self.train_loss
        history['HR'] = self.his_hr
        history['NDCG'] = self.his_ndcg
        savePath = args.path_ + r'/Model/' + args.dataset + r'/' + ModelName + r'.pth'

        params = {
            'epoch': self.curEpoch,
            # 'lr': self.lr,
            'model': self.model,
            # 'reg': self.reg,
            'history': history,
            'user_embed': self.user_embed,
            'user_embeds': self.user_embeds,
            'item_embed': self.item_embed,
            'item_embeds': self.item_embeds,
        }
        t.save(params, savePath)

    def prepareModel(self):
        self.modelName = self.getModelName()
        self.gnn_layer = eval(args.gnn_layer)
        self.hidden_dim = args.hidden_dim

        if args.isload == True:
            self.loadModel(args.loadModelPath)
        else:
            self.model = BGNN.myModel(self.user_num, self.item_num, self.behaviors,self.behavior_mats)

            self.getgroupWeights = Group_atten_weight.interout_group_attention(len(self.behaviors)).cuda()

        self.opt = t.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.opt_weight_decay)

        self.Groupweights_opt = t.optim.AdamW(self.getgroupWeights.parameters(), lr=args.Groupweights_lr,
                                      weight_decay=args.Groupweights_opt_weight_decay)

        self.scheduler = t.optim.lr_scheduler.CyclicLR(self.opt, args.opt_base_lr, args.opt_max_lr, step_size_up=5,
                                                       step_size_down=10, mode='triangular', gamma=0.99, scale_fn=None,
                                                       scale_mode='cycle', cycle_momentum=False, base_momentum=0.8,
                                                       max_momentum=0.9, last_epoch=-1)
        self.groupweights_scheduler = t.optim.lr_scheduler.CyclicLR(self.Groupweights_opt, args.Groupweights_opt_base_lr, args.Groupweights_opt_max_lr,
                                                            step_size_up=3, step_size_down=7, mode='triangular',
                                                            gamma=0.98, scale_fn=None, scale_mode='cycle',
                                                            cycle_momentum=False, base_momentum=0.9, max_momentum=0.99,
                                                            last_epoch=-1)


        if use_cuda:
            device = t.device('cuda' if t.cuda.is_available() else 'cpu')
            self.model = self.model.to(device)

    def run(self):
        self.prepareModel()
        if args.isload == True:
            print("----------------------pre test:",file=code_log)
            print("----------------------pre test:")#
            HR, NDCG = self.testEpoch(self.test_loader)
            print(f"HR: {HR} , NDCG: {NDCG}",file=code_log)
            print(f"HR: {HR} , NDCG: {NDCG}")
        log('Model Prepared')

        cvWait = 0  #
        self.best_HR = 0
        self.best_NDCG = 0
        flag = 0

        self.user_embed = None
        self.item_embed = None
        self.user_embeds = None
        self.item_embeds = None
        print("Test before train:",file=code_log)
        print("Test before train:")
        HR, NDCG = self.testEpoch(self.test_loader)

        for e in range(self.curEpoch, args.epoch + 1):
            self.curEpoch = e

            self.meta_flag = 0
            if e % args.meta_slot == 0:
                self.meta_flag = 1

            log("*****************Start epoch: %d ************************" % e)
            print("*****************Start epoch: %d ************************" % e,file=code_log)
            print("*****************Start epoch: %d ************************" % e)

            if args.isJustTest == False:
                epoch_loss, user_embed, item_embed, user_embeds, item_embeds = self.trainEpoch()
                self.train_loss.append(epoch_loss)
                print(f"epoch {e / args.epoch},  epoch loss{epoch_loss}",file=code_log)
                print(f"epoch {e / args.epoch},  epoch loss{epoch_loss}",)
                self.train_loss.append(epoch_loss)
            else:
                break

            HR, NDCG = self.testEpoch(self.test_loader)
            self.his_hr.append(HR)
            self.his_ndcg.append(NDCG)

            self.scheduler.step()
            self.groupweights_scheduler.step()


            if HR > self.best_HR:
                self.best_HR = HR
                self.best_epoch = self.curEpoch
                cvWait = 0
                print(
                    "--------------------------------------------------------------------------------------------------------------------------best_HR",
                    self.best_HR,file=code_log)
                print(
                    "--------------------------------------------------------------------------------------------------------------------------best_HR",
                    self.best_HR)
                # print("--------------------------------------------------------------------------------------------------------------------------NDCG", self.best_NDCG)
                self.user_embed = user_embed
                self.item_embed = item_embed
                self.user_embeds = user_embeds
                self.item_embeds = item_embeds

                self.saveHistory()
                self.saveModel()

            if NDCG > self.best_NDCG:
                self.best_NDCG = NDCG
                self.best_epoch = self.curEpoch
                cvWait = 0
                print(
                    "--------------------------------------------------------------------------------------------------------------------------best_NDCG",
                    self.best_NDCG,file=code_log)
                print(
                    "--------------------------------------------------------------------------------------------------------------------------best_NDCG",
                    self.best_NDCG)
                self.user_embed = user_embed
                self.item_embed = item_embed
                self.user_embeds = user_embeds
                self.item_embeds = item_embeds

                self.saveHistory()
                self.saveModel()

            if (HR < self.best_HR) and (NDCG < self.best_NDCG):
                cvWait += 1

            if cvWait == args.patience:
                print(f"Early stop at {self.best_epoch} :  best HR: {self.best_HR}, best_NDCG: {self.best_NDCG} \n",file=code_log)
                print(f"Early stop at {self.best_epoch} :  best HR: {self.best_HR}, best_NDCG: {self.best_NDCG} \n")
                self.saveHistory()
                self.saveModel()
                break

        HR, NDCG = self.testEpoch(self.test_loader)
        self.his_hr.append(HR)
        self.his_ndcg.append(NDCG)


if __name__ == '__main__':

    code_log = open('Design-Tmall-solo-code.txt', mode='a', encoding='utf-8')
    print(args,file=code_log)
    print(args)


    my_model = Model()
    my_model.run()
    log.close()  #


