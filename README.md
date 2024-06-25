# MCD-HCL

This repository contains PyTorch codes and datasets for the paper:

> Shan, Shuhui and Zhou, Wei and Gao, Min and Zhang, Hongyu and Wen, Junhao. Muti-Cluster Division Fine-Grained Heterogeneous Graph Contrastive Learning for Recommendation.This paper has been submitted to TOIS.


## Introduction
Multi-relational recommender systems model multiple sources of heterogeneous data and mine users’ potential interest characteristics and specific interaction patterns. Existing work faces three challenges: the data sparsity and interaction noise problems within auxiliary behaviors, the sample sampling bias problem in contrastive learning, and the lack of personalization when extracting heterogeneous interaction information. This paper proposes a heterogeneous multi-relational recommendation approach called Muti-Cluster Division Fine-Grained Heterogeneous Graph Contrastive Learning (MCD-HCL). The heterogeneous multi-relational graphs are divided according to user preference features, and higher-order convolution is performed within each heterogeneous multi-cluster graph. Based on the embedded feature representation, a fine-grained self-supervised model is introduced from multi-cluster and multi-relationship perspectives to mitigate the noisy information introduced by the higher-order convolution of auxiliary behaviors for the target behavior from the graph structure perspective. We further explore the user’s personalized information extraction and transfer as a way to optimize the overall recommendation task from two perspectives: heterogeneous multi-relational supervised learning and relation-dependent self-supervised learning, respectively


## Environment

The codes of MCD-HCL are implemented and tested under the following development environment:

- Python 3.6
- torch==1.8.1+cu111
- scipy==1.6.2
- tqdm==4.61.2



## Datasets

#### Raw data：
- IJCAI contest:  https://tianchi.aliyun.com/dataset/dataDetail?dataId=47
- Tmall:  https://tianchi.aliyun.com/dataset/dataDetail?dataId=649 
#### Processed data：
- The processed IJCAI and Tmall are under the /datasets folder.


## Usage
You need to create the `History/` and the `Models/` directories. 
- mkdir /History
- mkdir /Model 

The command to train MCD-HCL on the Tmall datasets are as follows. The commands specify the hyperparameter settings that generate the reported results in the paper.

* Tmall
```
python .\main.py --dataset=Tmall --opt_base_lr=1e-3 --opt_max_lr=5e-3 --opt_weight_decay=1e-4 --Groupweights_opt_base_lr=1e-4 --Groupweights_opt_max_lr=2e-3 --Groupweights_opt_weight_decay=1e-4 --Groupweights_lr=1e-3 --batch=8192 --SSL_batch=9
```
         







