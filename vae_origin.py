import os
import time
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm
from collections import defaultdict
import sys

import torch
import torch.utils.data as data

import warnings
warnings.filterwarnings('ignore')

from daisy.utils.sampler import Sampler
from daisy.utils.parser import parse_args
from daisy.utils.splitter import split_test
from daisy.utils.data import PointData, PairData, UAEData
from daisy.utils.loader import load_rate, get_ur, convert_npy_mat, get_adj_mat, build_candidates_set
from daisy.utils.metrics import precision_at_k, recall_at_k, map_at_k, hr_at_k, ndcg_at_k, mrr_at_k

def df_to_sparse(df, shape):
    users = df.user
    items = df.item
    ratings = df.rating

    sp_matrix = sp.csr_matrix((ratings, (users, items)), shape=shape)
    return sp_matrix


if __name__ == '__main__':
    ''' all parameter part '''
    args = parse_args()

    # store running time in time_log file
    result_save_path = './res/time_log_jupyter/'
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)
    time_file = f'{args.dataset}_{args.prepro}_{args.test_method}_{args.problem_type}_{args.algo_name}_{args.loss_type}'
    time_log = open(f'./res/time_log_jupyter/{time_file}.txt', 'a') 
    if args.early_stop == 1:
        early_stop = True
    else:
        early_stop = False
    ''' Test Process for Metrics Exporting '''
    train_set = pd.read_csv(f'./experiment_datasets/train_{args.dataset}_{args.prepro}_{args.test_method}.dat')
    test_set = pd.read_csv(f'./experiment_datasets/test_{args.dataset}_{args.prepro}_{args.test_method}.dat')
    if args.dataset in ['yelp']:
        train_set['timestamp'] = pd.to_datetime(train_set['timestamp'])
        test_set['timestamp'] = pd.to_datetime(test_set['timestamp'])
    df = pd.concat([train_set, test_set], ignore_index=True)
    user_num = df['user'].nunique()
    item_num = df['item'].nunique()

    train_set['rating'] = 1.0
    test_set['rating'] = 1.0

    # get ground truth
    test_ur = get_ur(test_set)
    total_train_ur = get_ur(train_set)
    # initial candidate item pool
    item_pool = set(range(item_num))
    candidates_num = args.cand_num

    print('='*50, '\n')
    # train_dataset = UAEData(user_num, item_num, train_set, test_set)
    train_matrix = df_to_sparse(train_set, (user_num, item_num))
    training_mat = convert_npy_mat(user_num, item_num, train_set)

    from daisy.model.VAERecommender_origin import VAE
    model = VAE(
            train_data=train_matrix,
            rating_mat=training_mat,
            q=args.dropout,
            epochs=args.epochs,
            lr=args.lr,
            reg_1=args.reg_1,
            reg_2=args.reg_2,
            beta=args.kl_reg,
            loss_type=args.loss_type,
            gpuid=args.gpu,
            device=args.device,
            early_stop=early_stop,
            optimizer=args.optimizer,
            initializer=args.initializer
        )
     
    # build recommender model
    s_time = time.time()
    # train_loader = data.DataLoader(
    #     train_dataset, 
    #     batch_size=args.batch_size, 
    #     shuffle=True, 
    #     num_workers=8,
    #     pin_memory=True,)

    model.fit(args.batch_size)

    elapsed_time = time.time() - s_time
    time_log.write(f'{args.dataset}_{args.prepro}_{args.test_method}_{args.problem_type}_{args.algo_name}_{args.loss_type}_{args.sample_method},{elapsed_time:.4f}' + '\n')
    time_log.close()
    print('training complete')
    if args.test_time == 1:
        sys.exit(0)

    print('Start Calculating Metrics......')
    test_ucands = build_candidates_set(test_ur, total_train_ur, item_pool, candidates_num)

    # get predict result
    print('')
    print('Generate recommend list...')
    print('')
    preds = {}
    if args.algo_name in ['vae', 'cdae', 'itemknn', 'puresvd', 'slim'] and args.problem_type == 'point':
        for u in tqdm(test_ucands.keys()):
            pred_rates = [model.predict(u, i) for i in test_ucands[u]]
            rec_idx = np.argsort(pred_rates)[::-1][:args.topk]
            top_n = np.array(test_ucands[u])[rec_idx]
            preds[u] = top_n
    

    # convert rank list to binary-interaction
    for u in preds.keys():
        preds[u] = [1 if i in test_ur[u] else 0 for i in preds[u]]

    # process topN list and store result for reporting KPI
    print('Save metric@k result to res folder...')
    result_save_path = f'./res/{args.dataset}/{args.prepro}/{args.test_method}/'
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    res = pd.DataFrame({'metric@K': ['pre', 'rec', 'hr', 'map', 'mrr', 'ndcg']})
    for k in [1, 5, 10, 20, 30, 50]:
        if k > args.topk:
            continue
        tmp_preds = preds.copy()        
        tmp_preds = {key: rank_list[:k] for key, rank_list in tmp_preds.items()}

        pre_k = np.mean([precision_at_k(r, k) for r in tmp_preds.values()])
        rec_k = recall_at_k(tmp_preds, test_ur, k)
        hr_k = hr_at_k(tmp_preds, test_ur)
        map_k = map_at_k(tmp_preds.values())
        mrr_k = mrr_at_k(tmp_preds, k)
        ndcg_k = np.mean([ndcg_at_k(r, k) for r in tmp_preds.values()])

        if k == 10:
            print(f'Precision@{k}: {pre_k:.4f}')
            print(f'Recall@{k}: {rec_k:.4f}')
            print(f'HR@{k}: {hr_k:.4f}')
            print(f'MAP@{k}: {map_k:.4f}')
            print(f'MRR@{k}: {mrr_k:.4f}')
            print(f'NDCG@{k}: {ndcg_k:.4f}')

        res[k] = np.array([pre_k, rec_k, hr_k, map_k, mrr_k, ndcg_k])

    # common_prefix = f'with_{args.sample_ratio}{args.sample_method}'
    if args.reg_2 != 0:
        reg = 1
    else:
        reg = 0

    if args.dropout != 0:
        dropout = 1
    else:
        dropout = 0   

    if args.mess_dropout != 0:
        mess_dropout = 1
    else:
        mess_dropout = 0

    if args.node_dropout != 0:
        node_dropout = 1
    else:
        node_dropout = 0

    if args.kl_reg != 0:
        kl_reg  = 1
    else:
        kl_reg  = 0

    if args.algo_name == 'mf' or args.algo_name == 'fm':
        common_prefix = f'with_{reg}_{args.early_stop}'
    elif args.algo_name == 'neumf' or args.algo_name == 'nfm':
        common_prefix = f'with_{reg}_{dropout}_{args.early_stop}'
    elif args.algo_name == 'ngcf':
        common_prefix = f'with_{reg}_{node_dropout}_{mess_dropout}_{args.early_stop}'
    elif args.algo_name == 'vae':
        common_prefix = f'with_{reg}_{kl_reg}_{dropout}_{args.early_stop}'
    else:
        common_prefix = f'with'
    

    algo_prefix = f'{args.loss_type}_{args.problem_type}_{args.algo_name}'

    res.to_csv(
        f'{result_save_path}{algo_prefix}_{common_prefix}_{args.optimizer}_{args.initializer}_{args.sample_method}_{args.sample_ratio}results.csv', 
        index=False
    )
