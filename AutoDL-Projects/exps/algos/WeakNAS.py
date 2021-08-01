##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
##################################################################
# Regularized Evolution for Image Classifier Architecture Search #
##################################################################
import os, sys;
sys.path.append(os.path.join(os.getcwd()))
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch
import numpy as np
from tqdm.auto import tqdm, trange
from collections import OrderedDict
import random
import numpy as np
import xgboost as xgb
import pandas as pd
from functools import reduce
from copy import deepcopy
import argparse
# import shap
from embedding import *
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
# from sklearn_plus.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from scipy import stats
import lightgbm as lgb
import matplotlib.pyplot as plt
import time

def get_dict(dataDict, mapList):
  # print(mapList)
  return reduce(dict.get, mapList, dataDict)

def prepare_seed(rand_seed):
  random.seed(rand_seed)
  np.random.seed(rand_seed)
  torch.manual_seed(rand_seed)
  torch.cuda.manual_seed(rand_seed)
  torch.cuda.manual_seed_all(rand_seed)

def lanas_org(adj, ops):
  assert len(ops) <= 7
  adj = [item for sublist in adj for item in sublist]
  if len(adj) >= 42:
      adj = adj[0:42]
  else:
      adj  += [0] * (42 - len(adj) )
  if len(ops) < 7:
      ops += [0] * (7 - len(ops) )
  emb\
    = adj + ops
  assert len(emb) == 49
  return emb

def run(args, num_sample_train_list):
  start_time = time.time()
  all_index = np.arange(len(args.df_dict_all))
  all_index_selected = deepcopy(all_index)
  keep_index_train = np.array([]).astype(np.int)
  num_sample_train_sum = 0
  args.index_unique_sample_list = []
  args.feature_name_list = list(args.df_dict_all.columns)
  # if args.flops is not None:
  #   df_dict_all_1 = args.df_dict_all[args.df_dict_all['FLOPs'] > args.flops[0]]
  #   df_dict_all_flops = df_dict_all_1[df_dict_all_1['FLOPs'] < args.flops[1]]
  #   index_flops = np.array(df_dict_all_flops.index.tolist())
  # label_dict_all = np.array([get_dict(value, args.label_key_dict[stage]) for key, value in arch_dict_data[stage].items()]).astype(np.float64)
  args.shap_values_list = []
  args.index_top_list = []
  args.acc_train_top_list = []
  args.acc_test_top_list = []
  # args.test_index_list = []
  args.index_train_list= []
  args.pred_list = []
  # args.label_list = []
  # args.index_unique_sampled_list = []
  args.acc_train_optimal_pred_list = []
  args.acc_test_optimal_pred_list = []
  args.index_optimal_pred_list = []
  # args.num_sampled = []
  args.num_unique_sample_list = []
  args.kendall_tau_train_list = []
  args.kendall_tau_test_list = []

  args.regr = []
  info_dict = {}
  shap_dict = {}
  info_dict_more = {}
  info_dict_tree = {}
  args.log_file = os.path.join(args.save_dir, 'seeds-{}.pth.tar'.format(rand_seed))
  args.log_file_more = os.path.join(args.save_dir, 'more-{}.pth.tar'.format(rand_seed))
  args.log_file_tree = os.path.join(args.save_dir, 'tree-{}.pth.tar'.format(rand_seed))
  args.log_file_shap = os.path.join(args.save_dir, 'shap-{}.pth.tar'.format(rand_seed))
  if os.path.exists(args.log_file):
    print(f'{args.log_file} already exists')
    return
  log_path = os.path.dirname(args.log_file)
  if not os.path.exists(log_path):
    try:
      os.makedirs(log_path)
    except:
      pass
  # index_set_unique_sampled = set()
  # pbar = trange(len(num_sample_train_list))
  train_optimal_found = False
  test_optimal_found = False
  acc_test_optimal_pred_max = 0.0
  for z, (num_sample_train, top_acc) in enumerate(zip(tqdm(num_sample_train_list), args.top_acc_list)):
    num_sample_train_sum += num_sample_train
    if len(all_index_selected) == 0:
      print(f'len all_index_selected = 0')
      break
    if z == 0:
      #             print('init uniform')
      train_index = np.random.choice(all_index_selected, size=min(num_sample_train, len(all_index_selected)), replace=False)

    else:
      if keep_index_train.size != 0:
        all_index_sample = all_index_selected[~np.isin(all_index_selected, keep_index_train)]
      else:
        all_index_sample = all_index_selected
      if num_sample_train == 0:
        print(f'len num_sample_train = 0')
        break
      if args.sampling_method == 'uniform':
        # if args.flops is not None:
        #   all_index_sample = np.intersect1d(all_index_sample, index_flops)
        # assert len(all_index_sample) >= num_sample_train, f'{len(all_index_sample)} < {num_sample_train}'
        # print(f'num_sample_train {num_sample_train}, all_index_sample {len(all_index_sample)}')
        if num_sample_train <= len(all_index_sample):
          train_index_sample = np.random.choice(all_index_sample, size=min(num_sample_train, len(all_index_sample)), replace=False)
          # train_index_sample = np.random.choice(allindex_sample[args.index_by_acc_test[:1000]], size=min(num_sample_train, len(all_index_sample)), replace=False)
          # train_index_sample = np.random.choice(all_index_sample[args.index_by_acc_test[:len(all_index_sample)]], size=min(num_sample_train, len(all_index_sample)), replace=False)
        else:
          # print(f'use top {num_sample_train} out of {len(all_index_sample)}')
          train_index_sample = []
          for index in all_index_by_acc:
            if index in keep_index_train:
              pass
            else:
              train_index_sample.append(index)
            if len(train_index_sample) >= num_sample_train:
              break
          train_index_sample = np.array(train_index_sample)
      elif args.sampling_method == 'gaussian':
        #                 print('gaussian')
        train_index_sample = sample_half_gaussian(sample_list=all_index_selected,
                                                  sample_num=min(num_sample_train, len(all_index_selected)),
                                                  std=args.gaussian_std, reject_list=keep_index_train)
      #                 inter = set(train_index_sample) & set(keep_index_train)
      else:
        raise ValueError
      if keep_index_train.size != 0:
        assert len(np.intersect1d(keep_index_train, train_index_sample)) == 0
        train_index = np.concatenate((keep_index_train, train_index_sample))
      else:
        train_index = train_index_sample
    assert len(train_index) != 0
    # test_index = np.setdiff1d(all_index, train_index)
    # assert len(test_index) + len(train_index) == len(all_index)
    # if len(test_index) == 0:
    #   test_index = train_index
    # assert len(test_index) != 0, f'{len(all_index)} {len(train_index)}'
    args.index_train_list.append(deepcopy(train_index))
    # args.test_index_list.append(deepcopy(test_index))
    # print(f'train samples {len(train_index)}')
    # arch_dict_data['train'] = OrderedDict(list(map(all_items.__getitem__, train_index)))
    # arch_dict_data['test'] = OrderedDict(list(map(all_items.__getitem__, test_index)))
    # #         assert len(arch_dict_data['train']) != 0
    # #         assert len(arch_dict_data['test']) != 0
    # #         print(f'optimal in train index: {index_optimal in train_index}')
    # #         print(f'optimal in test index: {index_optimal in test_index}')
    # df_dict = {}
    # label_dict = {}
    # stage_list = ['train', 'test']
    # for stage in stage_list:
    #   df_dict[stage] = pd.DataFrame()
    #   for feature in feature_list:
    #     feature = feature.replace('<', '')
    #     data_list = [get_dict(value, disp2key[feature]) for key, value in arch_dict_data[stage].items()]
    #     if len(data_list) == 0:
    #       print('Exception!')
    #       return arch_dict_data[stage], feature
    #     data_emp = data_list[0]
    #     if type(data_emp) is dict:
    #       for key in data_emp.keys():
    #         key_new = key.replace('<', '')
    #         df_dict[stage][f'{feature}: {key_new}'] = [float(data[key]) for data in data_list]
    #     elif type(data_emp) == float or type(data_emp) == np.ndarray or type(data_emp) == np.float64:
    #       if feature == 'SuperNet Acc':
    #         #                         print('SuperNet Acc / 100')
    #         df_dict[stage][f'{feature}'] = [float(data) / 100.0 for data in data_list]
    #       else:
    #         df_dict[stage][f'{feature}'] = [float(data) for data in data_list]
    #     elif type(data_emp) is tuple:
    #       for i, d in enumerate(data_emp):
    #         df_dict[stage][f'{feature}: {i + 1}'] = [float(data[i]) for data in data_list]
    #     else:
    #       print(feature, type(data_emp))
    #       raise ValueError
    # df_dict = {}
    # df_dict['train'] = df_dict_all.iloc[train_index]
    # df_dict['test'] = df_dict_all.iloc[test_index]
    # df_dict['all'] = df_dict_all
    # label_dict['train'] = args.label_all[train_index]
    # label_dict['test'] = args.label_all[test_index]
    # label_dict['train'] = np.array([get_dict(value, args.label_key_dict) for key, value in list(map(all_items.__getitem__, train_index))]).astype(np.float64)
    # label_dict['test'] = np.array([get_dict(value, args.label_key_dict) for key, value in list(map(all_items.__getitem__, test_index))]).astype(np.float64)
    #   label_dict[stage] = np.array([get_dict(value, args.label_key_dict[stage]) for key, value in arch_dict_data[stage].items()]).astype(np.float64)
    # unique_sample.update({key: get_dict(value, args.label_key_dict['train']) for key, value in list(map(args.all_items.__getitem__, train_index))})
    # args.index_unique_sample_list.append(deepcopy(unique_sample))
    args.num_unique_sample_list.append(len(train_index))
    if 'XGBoost' in args.predictor:
      if args.gpu:
        params = {'booster': 'gbtree',
                  'max_depth': args.max_depth,
                  'objective': args.loss,
                  'gpu_id': 0,
                  'tree_method': 'gpu_hist'}
      else:
        params = {'booster': 'gbtree',
                  'max_depth': args.max_depth,
                  'objective': args.loss}  #rank:pairwise
      dtrain = xgb.DMatrix(data=args.df_dict_all.iloc[train_index], label=args.norm_label_all_train[train_index])
      dall = xgb.DMatrix(data=args.df_dict_all, label=args.norm_label_all_train)
      # train_start = time.time()
      regr = xgb.train(params=params, dtrain=dtrain, num_boost_round=args.n_trees)
      # print(f'{args.predictor}, {len(train_index)} samples, train time: {time.time() - train_start}')
      # eval_start = time.time()
      pred_all = regr.predict(dall)
      # print(f'{args.predictor}, {len(args.df_dict_all)} samples, eval time: {time.time()-eval_start}')

    elif 'LGBoost' in args.predictor:
      params = {
        'boosting_type': args.lgboost_type, # 'gbdt' 'rf' 'dart'
        'objective': 'regression',
        'metric': args.lgboost_loss,
        'num_leaves': args.lgboost_leaves,
        'learning_rate': args.lgboost_lr,
        'feature_fraction': 0.9, #0.9 1.0
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        # 'tree_learner': 'feature',
        'verbose': 0
      }
      if args.lgboost_loss in ['lambdarank', 'rank_xendcg', 'rank_xendcg']:
        train_label_numpy = args.norm_label_all_train[train_index].argsort().argsort()
      else:
        train_label_numpy = args.norm_label_all_train[train_index]

      train_numpy = args.df_dict_all.iloc[train_index].to_numpy()
      all_numpy = args.df_dict_all.to_numpy()
      dtrain = lgb.Dataset(data=train_numpy, label=train_label_numpy)
      regr = lgb.train(params, dtrain, num_boost_round=args.lgboost_n_trees)
      pred_all = regr.predict(all_numpy, num_iteration=regr.best_iteration)
    else:
      if 'SVR' in args.predictor:
        regr = SVR(kernel=args.svr_kernel) # kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’}, degree=3 ‘poly’. epsilon=0.2
      elif 'RandomForest' in args.predictor:
        regr = RandomForestRegressor(n_estimators=args.random_forest_num) # max_depth=10, n_estimators=500
      elif 'MLP' in args.predictor:
        regr = MLPRegressor(hidden_layer_sizes=args.mlp_size, max_iter=args.mlp_iter) #max_iter=200  hidden_layer_sizes=(100, 100), max_iter=200, solver={‘lbfgs’, ‘sgd’, ‘adam’}
      elif 'DecisionTree' in args.predictor:
        regr = DecisionTreeRegressor(criterion="mse") # , max_depth=50 criterion{“mse”, “friedman_mse”, “mae”}
      elif 'LinearRegression' in args.predictor:
        regr = LinearRegression()
      if 'Bagging' in args.predictor:
        regr = BaggingRegressor(base_estimator=regr, n_estimators=100) #n_estimators
      elif 'AdaBoost' in args.predictor:
        regr = AdaBoostRegressor(base_estimator=regr, n_estimators=100)

      # train_start = time.time()
      regr.fit(args.df_dict_all.iloc[train_index], args.norm_label_all_train[train_index])
      # print(f'{args.predictor}, {len(train_index)} samples, train time: {time.time()-train_start}')
      eval_start = time.time()
      pred_all = regr.predict(args.df_dict_all)
      # df_dict_all = pd.concat([args.df_dict_all] * 3, ignore_index=True)
      # pred_all = regr.predict(df_dict_all)
      # print(f'{args.predictor}, {len(args.df_dict_all)} samples, eval time: {time.time()-eval_start}')

    args.regr.append(deepcopy(regr))
    args.pred_list.append(deepcopy(pred_all))
    # rank_gt = (-args.label_all[args.index_test]).argsort().argsort()
    all_index_by_acc = (-pred_all).argsort()

    kendall_tau_dict_train = {}
    kendall_tau_dict_test = {}
    # print(len(pred_all))
    if args.rank_top == -1 or args.rank_top == '-1':
      pass
    else:
      for rank_top in args.rank_top:
        if rank_top == 0:
          if args.bench == 'nasbench101':
            rank_top = 423624
          elif args.bench == 'nasbench201':
            rank_top = 15625
        rank_pred_train = (-(-pred_all[args.index_by_acc_train[:rank_top]])).argsort()
        rank_pred_test = (-(-pred_all[args.index_by_acc_test[:rank_top]])).argsort()

        rank_gt_train = (-(-args.label_all_train[args.index_by_acc_train[:rank_top]]).argsort()).argsort()
        rank_gt_test = (-(-args.label_all_test[args.index_by_acc_test[:rank_top]]).argsort()).argsort()

      # print(len(rank_pred_train), len(rank_pred_test), len(rank_gt_train))
      kendall_tau_train = stats.kendalltau(rank_pred_train, rank_gt_train)[0]
      kendall_tau_test = stats.kendalltau(rank_pred_test, rank_gt_test)[0]
      # print(f'kendall_tau_train: {kendall_tau_train:.4f}, kendall_tau_test: {kendall_tau_test:.4f}')

      kendall_tau_dict_train[rank_top] = kendall_tau_train
      kendall_tau_dict_test[rank_top] = kendall_tau_test

      args.kendall_tau_train_list.append(deepcopy(kendall_tau_dict_train))
      args.kendall_tau_test_list.append(deepcopy(kendall_tau_dict_test))

    # all_index_by_acc = np.array([i for (i, _) in sorted(zip(np.arange(len(pred_all)), pred_all), key=lambda pair: pair[-1], reverse=True)]).astype(np.int)
    all_index_selected = all_index_by_acc[:top_acc]
    # if args.save_top:
    index_top = deepcopy(all_index_by_acc[:max(top_acc, args.save_top)])
    args.index_top_list.append(index_top)
    args.acc_train_top_list.append(args.label_all_train[index_top])
    args.acc_test_top_list.append(args.label_all_test[index_top])
    if args.predictor == 'xgboost' and args.shap and args.repeat_iteration % args.shap_step == 0:
      explainer = shap.TreeExplainer(regr)
      shap_values = explainer.shap_values(args.df_dict_all, check_additivity=False)
      # print(dtrain.num_row(), dtrain.num_col(), df_dict_all.shape)
      args.shap_values_list.append(deepcopy(shap_values))
      del explainer
    if args.keep_old == 'none':
      keep_index_train = np.array([]).astype(np.int)
    elif args.keep_old == 'top':
      keep_index_train = np.array([i for i in all_index_selected if i in train_index]).astype(np.int)
    elif args.keep_old == 'all':
      keep_index_train = train_index
    #         print(f'plot points, train: {len(label_dict["train"])} test: {len(label_dict["test"])}', end='')
    index_optimal_pred = np.argmax(pred_all)
    args.index_optimal_pred_list.append(deepcopy(index_optimal_pred))
    if args.index_optimal_train in train_index:
      train_optimal_found = True
      acc_train_optimal_pred = args.label_all_train[args.index_optimal_train]
      args.acc_train_optimal_pred_list.append(deepcopy(acc_train_optimal_pred))
      acc_test_optimal_pred = args.label_all_test[args.index_optimal_train]
      args.acc_test_optimal_pred_list.append(deepcopy(acc_test_optimal_pred))
    else:
      acc_train_optimal_pred = args.label_all_train[index_optimal_pred]
      args.acc_train_optimal_pred_list.append(deepcopy(acc_train_optimal_pred))
      acc_test_optimal_pred = args.label_all_test[index_optimal_pred]
      args.acc_test_optimal_pred_list.append(deepcopy(acc_test_optimal_pred))
    # arch_optimal_pred = args.all_items[index_optimal_pred][0]
    # args.arch_optimal_pred_list.append(deepcopy(arch_optimal_pred))
    # arch_optimal_pred = deepcopy(args.all_items[index_optimal_pred][0])
    # args.index_optimal_pred_list.append(deepcopy(index_optimal_pred))
    # args.acc_optimal_pred.append(deepcopy(acc_optimal_pred))
    # args.arch_optimal_pred.append(deepcopy(arch_optimal_pred))
    # if use_top_delta_acc:
    #   pred_all_old = deepcopy(pred_all)
    # del regr
    if acc_train_optimal_pred == args.acc_optimal_train:
      # print('train optimal found')
      train_optimal_found = True
    if acc_test_optimal_pred == args.acc_optimal_test:
      # print('test optimal found')
      test_optimal_found = True

    if acc_test_optimal_pred > acc_test_optimal_pred_max:
      acc_test_optimal_pred_max = acc_test_optimal_pred

    # if train_optimal_found and test_optimal_found:
    # if args.label_all_test[index_optimal_pred] == args.acc_optimal_test:
    #   print_str += ' (train optimal found)'

    # pbar.set_description(print_str)
    # pbar.update(1)
    # info_dict_s['num_sampled'] = deepcopy(args.num_sampled)
    info_dict['num_unique_sample_list'] = deepcopy(args.num_unique_sample_list)
    # info_dict['test_index_list'] = deepcopy(args.test_index_list)
    # if 'operator_sync' in args.arch_shuffle or 'operator_unsync' in args.arch_shuffle:
    #   info_dict['ops_order_list'] = deepcopy(args.ops_order_list)
    # if 'node' in args.arch_shuffle:
    #   info_dict['node_order_list'] = deepcopy(args.node_order_list)
    info_dict['acc_train_optimal_pred_list'] = deepcopy(args.acc_train_optimal_pred_list)
    info_dict['acc_test_optimal_pred_list'] = deepcopy(args.acc_test_optimal_pred_list)
    info_dict['index_optimal_pred_list'] = deepcopy(args.index_optimal_pred_list)

    info_dict['acc_optimal_test_oracle'] = deepcopy(args.acc_optimal_test_oracle)

    info_dict['acc_optimal_train'] = deepcopy(args.acc_optimal_train)
    # info_dict['acc_optimal_train'] = deepcopy(args.acc_optimal_train)
    info_dict['index_optimal_train'] = deepcopy(args.index_optimal_train)

    info_dict['acc_optimal_test'] = deepcopy(args.acc_optimal_test)
    info_dict['index_optimal_test'] = deepcopy(args.index_optimal_test)
    info_dict['top_acc_list'] = deepcopy(args.top_acc_list)

    info_dict['kendall_tau_train_list'] = deepcopy(args.kendall_tau_train_list)
    info_dict['kendall_tau_test_list'] = deepcopy(args.kendall_tau_test_list)

    info_dict['index_train_list'] = deepcopy(args.index_train_list)
    info_dict['feature_name_list'] = deepcopy(args.feature_name_list)
    info_dict['index_top_list'] = deepcopy(args.index_top_list)
    info_dict['acc_train_top_list'] = deepcopy(args.acc_train_top_list)
    info_dict['acc_test_top_list'] = deepcopy(args.acc_test_top_list)

    if args.save_more_info:
      info_dict_more['pred_list'] = deepcopy(args.pred_list)

    # if args.save_more_info:
    #   # info_dict['arch_optimal_pred_list'] = deepcopy(args.arch_optimal_pred_list)
    #   info_dict_more['index_top_list'] = deepcopy(args.index_top_list)
    #   info_dict_more['index_train_list'] = deepcopy(args.index_train_list)
    #   info_dict_more['feature_name_list'] = deepcopy(args.feature_name_list)
    # info_dict['label_all_train'] = deepcopy(args.label_all_train)
    # info_dict['norm_label_all_train'] = deepcopy(args.norm_label_all_train)
    if args.save_tree_info:
      # info_dict_tree['pred_list'] = deepcopy(args.pred_list)
      info_dict_tree['regr'] = deepcopy(args.regr)
    # info_dict['arch_optimal'] = deepcopy(args.arch_optimal)
    # deepcopy(sorted(unique_sample.items(), key=lambda x: x[1], reverse=True)[0][0])
    # if args.save_more_info:
    # info_dict['index_unique_sample_list'] = deepcopy(args.index_unique_sample_list)
    # info_dict['index_unique_sampled_list'] = deepcopy(args.index_unique_sampled_list)
    # info_dict[args.num_unique_sample] = deepcopy(info_dict)
    # print(f'num_unique_sample: {args.num_unique_sample}')
    # info_dict['feature_name_list'] = args.feature_name_list
    # info_dict['label'] = deepcopy(args.label_list)
    # info_dict['acc_optimal'] = deepcopy(args.acc_optimal)
    # info_dict['arch_optimal'] = deepcopy(args.arch_optimal)
    # info_dict['index_op timal'] = deepcopy(args.index_optimal)
    if args.shap and args.repeat_iteration % args.shap_step == 0:
      shap_dict = deepcopy(info_dict)
      shap_dict['shap_values_list'] = deepcopy(args.shap_values_list)

    if (time.time()-start_time) >= args.save_interval:
      start_time = time.time()
      torch.save(info_dict, args.log_file)
      if args.save_more_info:
        torch.save(info_dict_more, args.log_file_more)
      if args.save_tree_info:
        torch.save(info_dict_tree, args.log_file_tree)
      if args.shap:
        torch.save(shap_dict, args.log_file_shap)

    # if train_optimal_found:
    #   torch.save(info_dict, args.log_file)
    #   if args.save_more_info:
    #     torch.save(info_dict_more, args.log_file_more)
    #   if args.save_tree_info:
    #     torch.save(info_dict_tree, args.log_file_tree)
    #   if args.shap:
    #     torch.save(shap_dict, args.log_file_shap)
    #   print_str = f'num_samples: {args.num_unique_sample_list[z]}, ' \
    #               f'train: {acc_train_optimal_pred:.5f} (optimal {args.acc_optimal_train:.5f}) ' \
    #               f'test: {acc_test_optimal_pred:.5f}, max: {acc_test_optimal_pred_max:.5f} (optimal {args.acc_optimal_test:.5f}) (oracle {args.acc_optimal_test_oracle:.5f}) ' \
    #               f'top {args.top_acc_list[z]} kendall_tau_train: '
    #   for key, value in kendall_tau_dict_train.items():
    #     print_str += f' {key}: {value:.5f}'
    #   if index_optimal_pred == args.index_optimal_train:
    #     print_str += ' (oracle  found)'
    #   print_str += '\n'
    #   print(print_str)
    #   break
    # else:
    if z + 1 == len(num_sample_train_list):
      print_str = f'num_samples: {args.num_unique_sample_list[z]}, ' \
                  f'train: {acc_train_optimal_pred:.5f} (optimal {args.acc_optimal_train:.5f}) ' \
                  f'test: {acc_test_optimal_pred:.5f}, max: {acc_test_optimal_pred_max:.5f} (optimal {args.acc_optimal_test:.5f}) (oracle {args.acc_optimal_test_oracle:.5f}) ' \
                  f'top {args.top_acc_list[z]} kendall_tau_train: '
      for key, value in kendall_tau_dict_train.items():
        print_str += f' {key}: {value:.5f}'
      if index_optimal_pred == args.index_optimal_train:
        print_str += ' (oracle  found)'
      print_str += '\n'
      print(print_str)
    # if index_optimal_pred == args.index_optimal_train:
    #   print_str += ' (oracle  found)'
    #   torch.save(info_dict, args.log_file)
    #   if args.save_more_info:
    #     torch.save(info_dict_more, args.log_file_more)
    #   if args.save_tree_info:
    #     torch.save(info_dict_tree, args.log_file_tree)
    #   if args.shap:
    #     torch.save(shap_dict, args.log_file_shap)
    #   break
  torch.save(info_dict, args.log_file)
  if args.save_more_info:
    torch.save(info_dict_more, args.log_file_more)
  if args.save_tree_info:
    torch.save(info_dict_tree, args.log_file_tree)
  if args.shap:
    torch.save(shap_dict, args.log_file_shap)
  return args

def get_y(x, decay, max_, min_):
  y = np.power(decay, x)
  y = (y-y.min())/(y.max()-y.min())
  y = y * (max_-min_) + min_
  y = y.astype(int)
  return y

# import matplotlib.pyplot as plt
def sample_scheduler(decay, iteration, start, end):
  if iteration == 1:
    y = np.array([start])
  else:
    base_end = 100
    x = np.linspace(0, base_end-1, num=iteration, endpoint=True)
    y = get_y(x=x, decay=decay, max_=start, min_=end)
    # print_index = [int(f) for f in np.linspace(0, 99, num=10, endpoint=True)]
    # y_print = y[print_index].astype(int)
    # plt.plot(x, y, label=f'decay {decay}\n{y_print}')
    # plt.show()
  return y

def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def main(args, bench_dict, arch_dict):
  # assert torch.cuda.is_available(), 'CUDA is not available.'
  # lower = args.filter[0]
  # upper = args.filter[1]
  if args.sample_decay == 'none':
    if args.init_sample > args.sample_each_iter:
      num_sample_train_list = [args.init_sample] + [args.sample_each_iter] * int(np.ceil((args.max_sample-args.init_sample+args.sample_each_iter) / args.sample_each_iter - 1))
    else:
      num_sample_train_list = [args.sample_each_iter] * int(np.ceil((args.max_sample-args.init_sample+args.sample_each_iter) / args.sample_each_iter))
  elif args.sample_decay == 'step':
    num_sample_train_list = [args.init_sample]
    # for i in range(len(args.sample_decay_step)-1):
    #   sample_start, sample_each = args.sample_decay_step[i]
    #   sample_end, _ = args.sample_decay_step[i+1]
      # for sample in range(start =sample_start, stop=sample_end, step=sample_each):
    i = 0
    while sum(num_sample_train_list) <= args.max_sample:
      sample_end, sample_each = args.sample_decay_step[i]
      # sample_end, _ = args.sample_decay_step[i+1]
      num_sample_train_list.append(sample_each)
      if sum(num_sample_train_list) > sample_end:
        i += 1
  elif args.sample_decay == 'linear':
    prob_list = [step for step in np.linspace(start=args.sample_linear_start,
                                              stop=args.sample_linear_end,
                                              num=args.sample_linear_iteration, endpoint=True)]
    prob_list = [p / sum(prob_list) for p in prob_list]
    sample_list = [p*10000 for p in prob_list]
    sample_candidate_list = [int(np.round(step)) for step in sample_list]
    num_sample_train_list = [args.init_sample]
    i = 0
    while sum(num_sample_train_list) <= args.max_sample:
      num_sample_train_list.append(sample_candidate_list[i])
      i += 1
    print()

  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.set_num_threads( args.workers )
  if arch_dict:
    arch_dict = {key: {**values, **bench_dict[key]} for (key, values) in arch_dict.items()}
  else:
    arch_dict = bench_dict
  args.all_items = list(arch_dict.items())
  if args.bench == 'nasbench201':
    if args.dataset == 'cifar10':
      setname_list = ['train', 'ori-test']
    elif args.dataset == 'cifar10-valid':
      setname_list = ['train', 'x-valid', 'ori-test']
    elif args.dataset == 'cifar100' or args.dataset == 'ImageNet16-120':
      setname_list = ['train', 'x-valid', 'x-test', 'ori-test']
    # assert args.setname in setname_list
    assert args.train_set in setname_list
    disp2key = {'SuperNet-Acc': ['supernet', args.train_set, 'acc'],
                'SuperNet-Loss': ['supernet', args.train_set, 'loss'],
                'Supernet-Entropy': ['supernet', args.train_set, 'entropy'],
                'Arch': ['arch'],
                'FLOPs': ['flops'],
                'Params': ['params'],
                'Latency': ['latency'],
                'Layer-Wise-Gradient-Norm': ['supernet', args.train_set, 'grad_norm'],
                'Cell-Wise-Gradient-Norm': ['supernet', args.train_set, 'cell_wise_grad_norm'],
                'Layer-Wise-Angle': ['supernet', 'angle'],
                'Cell-Wise-Angle': ['supernet', 'cell_wise_angle'],
                'Train Iter': ['supernet', 'train_counts']}
  elif args.bench == 'nasbench101':
    disp2key = {'Arch': ['arch'],
                'Params': ['params']}
  args.df_dict_all = pd.DataFrame()
  for feature in args.feature_list:
    feature = feature.replace('<', '')
    if feature == 'Arch':
      data_list = [key for key, value in args.all_items]
    else:
      data_list = [get_dict(value, disp2key[feature]) for key, value in args.all_items]
    if len(data_list) == 0:
      print('Exception!')
      raise ValueError
    data_emp = data_list[0]
    if type(data_emp) is dict:
      for key in data_emp.keys():
        key_new = key.replace('<', '')
        args.df_dict_all[f'{feature}: {key_new}'] = [float(data[key]) for data in data_list]
    elif type(data_emp) == float or type(data_emp) == np.ndarray or type(data_emp) == np.float64:
      if feature == 'SuperNet Acc':
        #                         print('SuperNet Acc / 100')
        args.df_dict_all[f'{feature}'] = [float(data) / 100.0 for data in data_list]
      else:
        args.df_dict_all[f'{feature}'] = [float(data) for data in data_list]
    elif type(data_emp) is tuple:
      assert feature == 'Arch'
      if args.bench == 'nasbench201':
        assert len(data_emp) == 6
        # if 'MLP' in args.predictor and args.mlp_onehot:
        if args.mlp_onehot:
          # args.node_order_list = node_order_list
          data_list = np.array([one_hot(a=np.array(data), num_classes=5).flatten() for data in data_list])
          data_emp = data_list[0]
          for i, d in enumerate(data_emp):
            arch_array = np.array([float(data[i]) for data in data_list])
            arch_list = list(arch_array)
            args.df_dict_all[f'{feature}: {i + 1}'] = arch_list
        else:
          node_order_list = np.arange(6)
          args.node_order_list = node_order_list
          if 'node' in args.arch_shuffle:
            np.random.seed(args.seed)
            np.random.shuffle(node_order_list)
          if 'operator_sync' in args.arch_shuffle or 'operator_unsync' in args.arch_shuffle:
            ops_order_list = np.arange(5)
            np.random.seed(args.seed)
            np.random.shuffle(ops_order_list)
            args.ops_order_list = ops_order_list
          for i, (node_idx, d) in enumerate(zip(node_order_list, data_emp)):
            arch_array = np.array([float(data[node_idx]) for data in data_list])
            if 'none' in args.arch_shuffle:
              pass
            elif 'operator_sync' in args.arch_shuffle or 'operator_unsync' in args.arch_shuffle:
              mask_dict = {}
              for p in range(5):
                mask_dict[p] = arch_array == p
              if 'operator_sync' in args.arch_shuffle:
                for old, new in enumerate(ops_order_list):
                  arch_array[mask_dict[old]] = new
              elif 'operator_unsync' in args.arch_shuffle:
                np.random.shuffle(ops_order_list)
                for old, new in enumerate(ops_order_list):
                  arch_array[mask_dict[old]] = new
            arch_list = list(arch_array)
            args.df_dict_all[f'{feature}: {i + 1}'] = arch_list
          if 'sample_all' in args.arch_shuffle:
            np.random.seed(args.seed)
            args.df_dict_all = args.df_dict_all.reindex(np.random.permutation(args.df_dict_all.index)).reset_index(drop=True)
      elif args.bench == 'nasbench101':
        # adj, ops = data_emp
        # arch_emp = lanas_org(adj, ops)
        arch_list = []
        for data in data_list:
          assert len(data) == 2
          adj, ops = data
          if args.emb_type == ['gbdt-nas']:
            arch = convert_arch_to_seq(deepcopy(adj), deepcopy(ops))
          else:
            arch = lanas_mod(deepcopy(adj), deepcopy(ops), transform=args.emb_type)
          # if args.emb_type == 'lanas':
          #   arch = lanas_org(adj, ops)
          # elif args.emb_type == 'lanas_mod':
          #   arch = lanas_mod(deepcopy(adj), deepcopy(ops), transform=['-1', 'last'])
          # elif args.emb_type == 'lanas_mod':
          #   arch = lanas_mod(deepcopy(adj), deepcopy(ops), transform=['-1', 'mid'])
          # elif args.emb_type == 'lanas_mod':
          #   arch = lanas_mod(deepcopy(adj), deepcopy(ops), transform=['-1', 'mid', 'swap'])
          arch_list.append(arch)
        # print(f'len of arch: {len(arch_list[0])}')
        arch_array = np.array(arch_list)
        # args.df_dict_all[f'{feature}'] = arch_array
        # arch_list_new = [arch[i] for arch in arch_list for i in range(len(arch_emp))]
        for i in range(arch_array.shape[1]):
          args.df_dict_all[f'{feature}: {i + 1}'] = arch_array[:,i].tolist()
      elif args.bench == 'mobilenet':
        pass
    else:
      print(feature, type(data_emp))
      raise ValueError

  if args.bench == 'nasbench201':
    args.label_all_train = np.array([get_dict(value, args.label_key_dict['train']) for i, (key, value) in enumerate(arch_dict.items())]).astype(np.float64)
    args.label_all_test = np.array([get_dict(value, args.label_key_dict['test']) for i, (key, value) in enumerate(arch_dict.items())]).astype(np.float64)
  elif args.bench == 'nasbench101':
    args.label_all_train = np.array([get_dict(value, args.label_key_dict['train']+['avg']) for i, (key, value) in enumerate(arch_dict.items())]).astype(np.float64)
    # args.label_all_train = np.array([get_dict(value, args.label_key_dict['train']+[random.randint(0,2)]) for i, (key, value) in enumerate(arch_dict.items())]).astype(np.float64)
    args.label_all_test = np.array([get_dict(value, args.label_key_dict['test']+['avg']) for i, (key, value) in enumerate(arch_dict.items())]).astype(np.float64)
    # args.label_all_test = np.array([get_dict(value, args.label_key_dict['test']+[random.randint(0,2)]) for i, (key, value) in enumerate(arch_dict.items())]).astype(np.float64)
  elif args.bench == 'mobilenet':
    args.label_all_train = np.array([get_dict(value, args.label_key_dict['train']) for i, (key, value) in enumerate(arch_dict.items())]).astype(np.float64)
    args.label_all_test = np.array([get_dict(value, args.label_key_dict['test']) for i, (key, value) in enumerate(arch_dict.items())]).astype(np.float64)
  # args.flops_all_train = np.array([get_dict(value, ['flops']) for i, (key, value) in enumerate(arch_dict.items())]).astype(np.float64)
  # args.flops_all_test = np.array([get_dict(value, ['flops']) for i, (key, value) in enumerate(arch_dict.items())]).astype(np.float64)
  #
  # args.params_all_train = np.array([get_dict(value, ['params']) for i, (key, value) in enumerate(arch_dict.items())]).astype(np.float64)
  # args.params_all_test = np.array([get_dict(value, ['params']) for i, (key, value) in enumerate(arch_dict.items())]).astype(np.float64)
  #
  # args.filter_index = np.logical_and(args.flops_all_train < args.flops, args.params_all_train < args.params)
  # args.df_dict_all = args.df_dict_all[args.filter_index]
  # args.label_all_train = args.label_all_train[args.filter_index]
  # args.label_all_test = args.label_all_test[args.filter_index]
  args.norm_label_all_train = (args.label_all_train - np.mean(args.label_all_train)) / np.std(args.label_all_train)

  # args.rank_gt_train = (-(-args.label_all_train).argsort()).argsort()
  args.index_by_acc_train = (-args.label_all_train).argsort()

  args.acc_optimal_train = max(args.label_all_train)
  args.index_optimal_train = np.argmax(args.label_all_train)

  # args.norm_label_all_test = (args.label_all_test - np.mean(args.label_all_test)) / np.std(args.label_all_test)
  # args.rank_gt_test = (-(-args.label_all_test).argsort()).argsort()
  args.index_by_acc_test = (-args.label_all_test).argsort()

  args.acc_optimal_test = max(args.label_all_test)
  args.index_optimal_test = np.argmax(args.label_all_test)

  args.acc_optimal_test_oracle = args.label_all_test[args.index_optimal_train]

  # label_all_test = [x for _, x in sorted(zip(args.label_all_train, args.label_all_test))]
  # label_all_test = label_all_test[::-1]
  # print()

  # print(args.label_all_train[args.index_optimal_train], args.label_all_test[args.index_optimal_train])
  # print(args.index_optimal_train, args.index_optimal_test)

  # assert args.index_optimal_test == args.index_optimal_train
  # arch_optimal = [(i, value[0]) for i, value in filter(lambda value: get_dict(value[-1][1], args.label_key_dict['test']) == args.acc_optimal, enumerate(arch_dict.items()))]
  # args.index_optimal = [i for (i, a) in arch_optimal]
  # args.arch_optimal = [a for (i, a) in arch_optimal]
  # if len(arch_optimal) == 1:
  #   args.index_optimal, args.arch_optimal = args.index_optimal[0], args.arch_optimal[0]
  # init_sample = max(args.init_sample, args.sample_each_iter)
  # num_sample_list = [int(s) for s in np.linspace(start=init_sample, stop=args.max_sample,
  #                                                num=int((args.max_sample-init_sample+args.sample_each_iter) / args.sample_each_iter), endpoint=True)]
  # init_sample = max(args.init_sample, args.sample_each_iter)
  # assert args.init_sample >= args.sample_each_iter, f'init_sample: {args.init_sample}, sample_each_iter: {args.sample_each_iter}'
  # num_iterations = int((args.max_sample-args.init_sample+args.sample_each_iter) / args.sample_each_iter)
    # num_sample_train_list = [args.init_sample] + [args.sample_each_iter] * int(np.ceil((args.max_sample-args.init_sample+args.sample_each_iter) / args.sample_each_iter - 1))
  # num_sample_list = [sample if sample >= args.sample_each_iter else args.sample_each_iter for sample in num_sample_list]
  args.iteration = len(num_sample_train_list)
  # print(f'dataset: {args.dataset}, setname: {args.setname}, arch_dim :{len(arch_list[0])}, num_runs: {len(num_sample_list)}, repeat: {args.repeat}')
  # print(f'top_acc_list: {args.top_acc_list}, top_acc_list_delta: {args.top_acc_list_delta}')
  # pbar = tqdm(total=len(num_sample_list), leave=True)
  # for num_sample in trange(len(num_sample_list)):
  if not args.top_increase:
    assert args.top_start >= args.top_end, f'top_start: {args.top_start}, top_end: {args.top_end}'
  args.top_acc_list = sample_scheduler(args.top_decay, args.iteration, args.top_start, args.top_end)
  # for i, num_sample in zip(pbar, num_sample_list):
  #   # args.iteration = int(num_sample/args.sample_each_iter)
  #   args.top_acc_list = sample_scheduler(args.top_decay, args.iteration, args.top_start, args.top_end)
  #   # print(f'top_acc_list: {args.top_acc_list}\n', end='')
  #   # print(f'num_samples: {num_sample}', end='')
  #   assert args.iteration == len(args.top_acc_list)
  print(f'num_sample_train_list: {num_sample_train_list}')
  args = run(args=args, num_sample_train_list=num_sample_train_list)
  if args.plot:
    # acc_valid_list = []
    acc_test_list = []
    num_sample_list = []
    test_sample = 1
    # train_acc_max = 0.0
    test_acc_max = 0.0
    num_sample_sum = 0
    fig1, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    for (train_top, test_top, num_sample) in zip(args.acc_train_top_list, args.acc_test_top_list, args.num_unique_sample_list):
      if num_sample < 0:
        # train_acc_max = max(train_acc_max, train_top[0])
        # acc_valid_list.append(train_acc_max)
        test_acc_max = max(test_acc_max, test_top[0])
        acc_test_list.append(test_acc_max)
        num_sample_list.append(num_sample + num_sample_sum)
      else:
        num_sample_sum += test_sample
        # train_acc_max = max(train_acc_max, max(train_top[:test_sample]))
        # acc_valid_list.append(train_acc_max)
        test_acc_max = max(test_acc_max, max(test_top[:test_sample]))
        acc_test_list.append(test_acc_max)
        num_sample_list.append(num_sample + num_sample_sum)
    ax.plot(num_sample_list, acc_test_list, linewidth=4, label=None)
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Best Validation Accuracy')
    ax.axhline(y=args.acc_optimal_train, linestyle='dashed', color='black', linewidth=4, label='oracle')
    plt.show()
  # pbar.update(1)
  # pbar.update()
  # pbar.refresh()
  # info_dict_s = {}
  # shap_dict[num_sample] = deepcopy(shap_dict)
  # return deepcopy(info_dict), deepcopy(info_dict_more), deepcopy(shap_dict)

if __name__ == '__main__':

  def coords(s):
    try:
      x, y = map(int, s.split(','))
      return x, y
    except:
      raise argparse.ArgumentTypeError("Coordinates must be x,y")

  parser = argparse.ArgumentParser("Regularized Evolution Algorithm")
  parser.add_argument('--repeat',      type=int,   default=1000, help='The total time cost budge for searching (in seconds).')
  parser.add_argument('--save_dir',           type=str,   help='Folder to save checkpoints and log.')
  parser.add_argument('--rand_seed',          type=int,   default=-1,   help='manual seed')
  parser.add_argument('--sampling_method', type=str,  default='uniform')
  parser.add_argument('--num_sample_method', type=str,  default='uniform')
  parser.add_argument('--arch_shuffle', type=str, nargs='+',  default=['none'])
  parser.add_argument('--dataset', type=str,  default='cifar100')
  parser.add_argument('--bench', type=str,  default='nasbench201')
  parser.add_argument('--setname', type=str,  default='x-valid')
  parser.add_argument('--keep_old', type=str,  default='all')
  parser.add_argument('--max_sample',          type=int,   default=1000)
  parser.add_argument('--init_sample',          type=int,   default=10)
  parser.add_argument('--n_trees',          type=int,   default=5000)
  parser.add_argument('--max_depth',          type=int,   default=10)
  parser.add_argument('--iteration',          type=int,   default=5)
  parser.add_argument('--top_max',          type=int,   default=2000,   help='manual seed')
  parser.add_argument('--top_min',          type=int,   default=80,   help='manual seed')
  # parser.add_argument('--top_acc_list', type=int, nargs='+', default=[2000, 667, 444, 296, 198]) #[2000, 667, 444, 296, 198] [2000, 500, 400, 300, 200]
  # parser.add_argument('--top_acc_list_delta', type=int, nargs='+', default=[10000, 8000, 6000, 4000, 2000]) # [2000, 4000, 6000, 8000, 10000]
  # parser.add_argument('--feature_list',  type=str, nargs='*', default=['SuperNet Acc','SuperNet Loss','Supernet Entropy','Arch','FLOPs','Params','Latency','Gradient Norm','Cell-Wise Gradient-Norm','Angle','Cell-Wise Angle','Train Iter'])
  parser.add_argument('--feature_list',  type=str, default='Arch')
  parser.add_argument('--emb_type', type=str, default='gbdt-nas')
  parser.add_argument('--info_path', type=str,  default=None)
  parser.add_argument('--emb', type=str,  default=None)
  parser.add_argument('--bench_path', type=str,  default='DATASET/nasbench201/nasbench201-cifar100-info.pth.tar')
  parser.add_argument('--train_epoch',          type=int,   default=2000)
  parser.add_argument('--epoch',          type=int,   default=2000)
  parser.add_argument('--gaussian_std',          type=float,   default=0.2)
  parser.add_argument('--scratch_train_epoch',          type=int,   default=200)
  parser.add_argument('--scratch_test_epoch',          type=int,   default=200)
  parser.add_argument('--workers',            type=int,   default=8,    help='number of data loading workers (default: 2)')
  parser.add_argument('--use_top_delta_acc', default=False, action='store_true')
  parser.add_argument('--shap', default=False, action='store_true')
  parser.add_argument('--shap_step',          type=int,   default=100)
  parser.add_argument('--flops', type=float, default=None) # mean 54.89653436992, max 220.11969, min 7.78305
  parser.add_argument('--params', type=float, default=None) # mean 54.89653436992, max 220.11969, min 7.78305
  parser.add_argument('--top_start',          type=int,   default=80000)
  parser.add_argument('--top_end',          type=int,   default=500)
  parser.add_argument('--top_decay',          type=float,   default=0.96)
  parser.add_argument('--sample_each_iter',          type=int,   default=10)
  parser.add_argument('--save_more_info', default=False, action='store_true')
  parser.add_argument('--save_tree_info', default=False, action='store_true')
  parser.add_argument('--train_set', type=str,  default='valid')
  parser.add_argument('--test_set', type=str,  default='test')
  parser.add_argument('--save_interval', type=int,   default=1800)
  parser.add_argument('--save_top', type=int,   default=5000)
  parser.add_argument('--top_increase', default=False, action='store_true')
  parser.add_argument('--loss', type=str,  default='reg:squarederror') #rank:pairwise
  parser.add_argument('--predictor', type=str, nargs='+', default=['XGBoost'])
  parser.add_argument('--rank_top', type=int, nargs='+',  default=[100, 1000, 10000])
  parser.add_argument('--mlp_size', type=int, nargs='+',  default=[100])
  parser.add_argument('--mlp_iter', type=int,  default=200)
  parser.add_argument('--mlp_onehot', default=False, action='store_true')
  parser.add_argument('--random_forest_num', type=int, default=500)
  parser.add_argument('--svr_kernel', type=str, default='rbf')
  parser.add_argument('--lgboost_leaves', type=int, default=31)
  parser.add_argument('--lgboost_lr', type=float, default=0.05) # 0.05
  parser.add_argument('--lgboost_n_trees', type=int, default=5000)
  parser.add_argument('--lgboost_type', type=str, default='gbdt') #'gbdt' 'rf' 'dart'
  parser.add_argument('--lgboost_loss', type=str, default='l2') #'gbdt' 'rf' 'dart'
  parser.add_argument('--plot', default=False, action='store_true')
  parser.add_argument('--filter', type=float, nargs='+',  default=[0.875, 0.925])
  parser.add_argument('--gpu', default=False, action='store_true')
  parser.add_argument('--sample_decay', type=str, default='none') #'gbdt' 'rf' 'dart'
  parser.add_argument('--sample_decay_step', help="step", type=coords, nargs='+')
  parser.add_argument('--sample_linear_iteration', help="step", type=int, default=10)
  parser.add_argument('--sample_linear_end', help="step", type=int, default=10)
  parser.add_argument('--sample_linear_start', help="step", type=int, default=1)

  args = parser.parse_args()
  # print(f'arch_shuffle: {args.arch_shuffle}')
  args.feature_list = [x for x in args.feature_list.split(",")]
  args.emb_type = [x for x in args.emb_type.split(",")]
  if args.bench == 'nasbench101':
    args.dataset == 'cifar10'
  print(f'Saving to {args.save_dir}')
  if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir, exist_ok=True)
  #if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
  # args.ea_fast_by_api = args.ea_fast_by_api > 0
  bench_dict = torch.load(args.bench_path)
  print(f'loaded bench form {args.bench_path}')
  if args.bench == 'nasbench201':
    # pass
    arch_dict = {}
    # info_name = os.path.join(args.info_path, 'info.pth.tar')
    # arch_dict = torch.load(info_name)
    # assert len(arch_dict) == 15625, f'{info_name}'
    # if args.dataset == 'cifar10-valid':
    #   if args.setname == 'train':
    #     args.args.acc_optimal = 100.0
    #   elif args.setname == 'x-valid':
    #     args.acc_optimal = 91.60666665039064
    #   elif args.setname == 'x-test':
    #     args.acc_optimal = 91.52333333333333
    # elif args.dataset == 'cifar10':
    #   if args.setname == 'train':
    #     args.acc_optimal = 99.994
    #   elif args.setname == 'ori-test':
    #     args.acc_optimal = 94.37333333333333
    # elif args.dataset == 'cifar100':
    #   if args.setname == 'train':
    #     args.acc_optimal = 99.93733333333334
    #   elif args.setname == 'x-valid':
    #     args.acc_optimal = 73.4933333577474
    #   elif args.setname == 'x-test':
    #     args.acc_optimal = 73.51333332112631
    #   elif args.setname == 'ori-test':
    #     args.acc_optimal = 73.50333333333333
    # elif args.dataset == 'ImageNet16-120':
    #   if args.setname == 'train':
    #     args.acc_optimal = 73.22918040138735
    #   elif args.setname == 'x-valid':
    #     args.acc_optimal = 46.73333327229818
    #   elif args.setname == 'x-test':
    #     args.acc_optimal = 47.31111100599501
    #   elif args.setname == 'ori-test':
    #     args.acc_optimal = 46.8444444647895

    if args.scratch_train_epoch < 200:
      raise ValueError
      # args.label_key_dict = {'train': ['scratch', 'ori-test', 'acc', args.scratch_train_epoch, 'avg'],
      #                   'test': ['scratch', 'ori-test', 'acc', args.scratch_test_epoch, 'avg'],
      #                   'test_exclude': ['scratch', 'ori-test', 'acc', args.scratch_test_epoch, 'avg']}
      # args.label_key_dict = ['scratch', args.setname, 'acc', args.scratch_test_epoch, 'avg']
    else:
      assert args.scratch_train_epoch == 200
      # args.label_key_dict = {'train': ['scratch', args.setname, 'acc', args.scratch_train_epoch, 'avg'],
      #                   'test': ['scratch', args.setname, 'acc', args.scratch_test_epoch, 'avg'],
      #                   'test_exclude': ['scratch', args.setname, 'acc', args.scratch_test_epoch, 'avg']}
      # args.label_key_dict = ['scratch', args.setname, 'acc', args.scratch_test_epoch, 'avg']
      args.label_key_dict = {'train': ['scratch', args.train_set, 'acc', args.scratch_test_epoch, 'avg'],
                             'test': ['scratch', args.test_set, 'acc', args.scratch_test_epoch, 'avg']}
  elif args.bench == 'nasbench101':
    arch_dict = {}
    # if args.setname == 'train':
    #   args.acc_optimal = 1.0
    # elif args.setname == 'valid':
    #   args.acc_optimal = 0.9505542318026224
    # elif args.setname == 'test':
    #   args.acc_optimal = 0.943175752957662
    # args.label_key_dict = {'train' :['scratch', 'valid', 'acc', args.scratch_test_epoch, 'avg'],
    #                        'test' :['scratch', 'test', 'acc', args.scratch_test_epoch, 'avg']}
    # args.label_key_dict = {'train' :['scratch', 'valid', 'acc', args.scratch_test_epoch, 'avg'],
    #                        'test' :['scratch', 'valid', 'acc', args.scratch_test_epoch, 'avg']}
    # args.label_key_dict = {'train' :['scratch', args.train_set, 'acc', args.scratch_test_epoch, 'avg'],
    #                        'test' :['scratch', args.test_set, 'acc', args.scratch_test_epoch, 'avg']}
    args.label_key_dict = {'train' :['scratch', args.train_set, 'acc', args.scratch_test_epoch],
                           'test' :['scratch', args.test_set, 'acc', args.scratch_test_epoch]}
    # args.label_key_dict = {'train' :['scratch', args.train_set, 'acc', args.scratch_test_epoch, 0],
    #                        'test' :['scratch', args.test_set, 'acc', args.scratch_test_epoch, 0]}
  # assert args.acc_optimal in [get_dict(value, args.label_key_dict) for key, value in bench_dict.items()]
  if args.rand_seed < 0:
    for args.repeat_iteration in tqdm(range(args.repeat)):
    # for i in trange(args.repeat):
    # for i in range(args.repeat):
      # print ('{:} : {:03d}/{:03d}'.format(time_string(), i, num))
      rand_seed = random.randint(0, 2**32-1)
      args.seed = rand_seed
      prepare_seed(rand_seed)
      main(args, bench_dict, arch_dict)
  else:
      pass

