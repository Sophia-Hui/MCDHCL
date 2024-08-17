import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')

    parser.add_argument('--hidden_dim', default=64, type=int, help='embedding size')
    parser.add_argument('--gnn_layer', default="[64,64,64]", type=str, help='gnn layers: number + dim')
    parser.add_argument('--dataset', default='IJCAI_15', type=str, help='name of dataset')
    parser.add_argument('--point', default='for_meta_hidden_dim', type=str, help='')
    parser.add_argument('--title', default='dim__8', type=str, help='title of model')
    parser.add_argument('--sampNum', default=10, type=int, help='batch size for sampling')
    parser.add_argument('--groups', type=int, default=2, help='Number of group.')
    parser.add_argument('--rank', type=int, default=3,help='the dimension of low rank matrix decomposition')
    parser.add_argument('--lr', default=3e-4, type=float, help='learning rate')
    parser.add_argument('--opt_base_lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--opt_max_lr', default=2e-3, type=float, help='learning rate')
    parser.add_argument('--opt_weight_decay', default=1e-4, type=float, help='weight decay regularizer')
    parser.add_argument('--Groupweights_opt_base_lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--Groupweights_opt_max_lr', default=2e-3, type=float, help='learning rate')
    parser.add_argument('--Groupweights_opt_weight_decay', default=1e-4, type=float, help='weight decay regularizer')
    parser.add_argument('--Groupweights_lr', default=1e-3, type=float, help='_meta_learning rate')
    parser.add_argument('--batch', default=8192, type=int, help='batch size')
    parser.add_argument('--meta_batch', default=128, type=int, help='batch size')
    parser.add_argument('--SSL_batch', default=30, type=int, help='batch size')
    parser.add_argument('--reg', default=1e-3, type=float, help='weight decay regularizer')
    parser.add_argument('--beta', default=0.005, type=float, help='scale of infoNCELoss')
    parser.add_argument('--epoch', default=500, type=int, help='number of epochs')

    parser.add_argument('--shoot', default=10, type=int, help='K of top k')
    parser.add_argument('--inner_product_mult', default=1, type=float, help='multiplier for the result')
    parser.add_argument('--drop_rate', default=0.8, type=float, help='drop_rate')
    parser.add_argument('--drop_rate1', default=0.5, type=float, help='drop_rate')
    parser.add_argument('--seed', type=int, default=6)
    parser.add_argument('--slope', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--path', default='/home/shanshuhui/',type=str, help='data path')
    parser.add_argument('--path_', default='/home/shanshuhui/', type=str,help='data path')
    parser.add_argument('--save_path', default='tem', help='file name to save model and training record')
    parser.add_argument('--load_model', default=None, help='model name to load')
    parser.add_argument('--target', default='buy', type=str, help='target behavior to predict on')
    parser.add_argument('--isload', default=False , type=bool, help='whether load model')
    parser.add_argument('--isJustTest', default=False , type=bool, help='whether load model')
    parser.add_argument('--loadModelPath', default='/home/shanshuhui/', type=str, help='loadModelPath')
    parser.add_argument('--head_num', default=4, type=int, help='head_num_of_multihead_attention')
    parser.add_argument('--beta_multi_behavior', default=0.005, type=float, help='scale of infoNCELoss')
    parser.add_argument('--sampNum_slot', default=30, type=int, help='SSL_step')
    parser.add_argument('--SSL_slot', default=1, type=int, help='SSL_step')
    parser.add_argument('--k', default=2, type=float, help='MFB')
    parser.add_argument('--meta_time_rate', default=0.8, type=float, help='gating rate')
    parser.add_argument('--meta_behavior_rate', default=0.8, type=float, help='gating rate')
    parser.add_argument('--meta_slot', default=2, type=int, help='epoch number for each SSL')
    parser.add_argument('--time_slot', default=60*60*24*360, type=float, help='length of time slots')
    parser.add_argument('--hidden_dim_meta', default=16, type=int, help='embedding size')

    return parser.parse_args()


args = parse_args()

