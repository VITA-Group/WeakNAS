import os
import torch
import random
import numpy as np
from tqdm import tqdm
import xgboost as xgb
import pandas as pd
from functools import reduce
from copy import deepcopy
import argparse
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import pickle

INPUT = 0
CONV1X1 = 1
CONV3X3 = 2
MAXPOOL3X3 = 3
OUTPUT = 4

def coords(s):
  try:
    x, y = map(int, s.split(','))
    return x, y
  except:
    raise argparse.ArgumentTypeError("Coordinates must be x,y")

def flatten_list(adj):
  return [item for sublist in adj for item in sublist]

def convert_arch_to_seq(matrix, ops, max_n=7):
  seq = []
  n = len(matrix)
  max_n = 7
  assert n == len(ops)
  for col in range(1, max_n):
    if col >= n:
      seq += [0 for i in range(col)]
      seq += [0, 0, 0, 0]
    else:
      for row in range(col):
        seq.append(matrix[row][col])
      if ops[col] == CONV1X1:
        seq += [1, 0, 0, 0]
      elif ops[col] == CONV3X3:
        seq += [0, 1, 0, 0]
      elif ops[col] == MAXPOOL3X3:
        seq += [0, 0, 1, 0]
      elif ops[col] == OUTPUT:
        seq += [0, 0, 0, 1]
  assert len(seq) == (5 + max_n + 3) * (max_n - 1) / 2
  return seq

def lanas_mod(adj, ops, transform=[]):
  assert len(adj) == len(adj[0])
  assert len(adj) == len(ops)
  res = 7 - len(ops)
  adj_len = len(adj)
  assert len(adj) == len(ops), f'{len(adj)} {len(ops)}'
  if res > 0:
    adj_remove = adj[adj_len - 1:]
    assert all(x == 0 for xx in adj_remove for x in xx)
  adj = adj[:adj_len - 1]
  if '0' in transform:
    adj_pad = 0
  elif '-1' in transform:
    adj_pad = -1
  if 'onehot' in transform:
    ops_num_list = [2, 3, 4, 5, 6, 7]
    emb = []
    for ops_num in ops_num_list:
      if ops_num != adj_len:
        if 'fill' in transform:
          emb += [adj_pad] * ops_num * (ops_num-1)
        else:
          emb += [adj_pad] * ops_num ** 2
      else:
        if 'fill' in transform:
          for i in range(len(adj)):
            for j, op in enumerate(ops):
              if adj[i][j] != 0:
                adj[i][j] = op
          emb += flatten_list(adj)
        else:
          emb += flatten_list(adj) + ops
    emb_flatten = emb
    if 'fill' in transform:
      assert len(emb_flatten) == 112, f'{len(emb_flatten)}'
    else:
      assert len(emb_flatten) == 139, f'{len(emb_flatten)}'
  else:
    assert 'mid' in transform or 'last' in transform
    assert '-1' in transform or '0' in transform
    assert len(ops) in [2,3,4,5,6,7]
    if 'mid' in transform:
      adj = [sublist + [adj_pad] * res for sublist in adj]
      for i in range(res):
        adj.append([adj_pad] * 7)
    elif 'last' in transform:
      if len(flatten_list(adj)) < 42:
        adj.append([adj_pad] * (42 - len(flatten_list(adj))))
    if 'fill' in transform:
      if 'mid' in transform:
        for i in range(len(adj)):
          for j, op in enumerate(ops):
            if adj[i][j] != 0:
              adj[i][j] = op
      elif 'last' in transform:
        for i in range(len(adj)):
          for j, op in enumerate(ops):
            if adj[i][j] != 0:
              adj[i][j] = op
      emb = adj
    else:
      emb = adj
      ops = ops + [0] * res
      emb.append(ops)
    emb_flatten = flatten_list(emb)
    if 'fill' in transform:
      assert len(emb_flatten) == 42, f'{len(emb_flatten)}'
    else:
      assert len(emb_flatten) == 49, f'{len(emb_flatten)}'
  return emb_flatten

def get_dict(dataDict, mapList):
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

def acq_fn(predictions, ytrain=None, stds=None, explore_type='ei'):
    predictions = - np.array(predictions)

    if stds is None:
        stds = np.sqrt(np.var(predictions, axis=0))

    # Upper confidence bound (UCB) acquisition function
    if explore_type == 'ucb':
        explore_factor = 0.5
        mean = np.mean(predictions, axis=0)
        ucb = mean - explore_factor * stds
        sorted_indices = np.argsort(ucb)

    # Expected improvement (EI) acquisition function
    elif explore_type == 'ei':
        ei_calibration_factor = 5.
        mean = list(np.mean(predictions, axis=0))
        factored_stds = list(stds / ei_calibration_factor)
        min_y = ytrain.min()
        gam = [(min_y - mean[i]) / factored_stds[i] for i in range(len(mean))]
        ei = [-1 * factored_stds[i] * (gam[i] * norm.cdf(gam[i]) + norm.pdf(gam[i]))
              for i in range(len(mean))]
        sorted_indices = np.argsort(ei)

    # Probability of improvement (PI) acquisition function
    elif explore_type == 'pi':
        mean = list(np.mean(predictions, axis=0))
        stds = list(stds)
        min_y = ytrain.min()
        pi = [-1 * norm.cdf(min_y, loc=mean[i], scale=stds[i]) for i in range(len(mean))]
        sorted_indices = np.argsort(pi)

    # Thompson sampling (TS) acquisition function
    elif explore_type == 'ts':
        rand_ind = np.random.randint(predictions.shape[0])
        ts = predictions[rand_ind,:]
        sorted_indices = np.argsort(ts)

    # Top exploitation
    elif explore_type == 'percentile':
        min_prediction = np.min(predictions, axis=0)
        sorted_indices = np.argsort(min_prediction)

    # Top mean
    elif explore_type == 'mean':
        mean = np.mean(predictions, axis=0)
        sorted_indices = np.argsort(mean)

    elif explore_type == 'confidence':
        confidence_factor = 2
        mean = np.mean(predictions, axis=0)
        conf = mean + confidence_factor * stds
        sorted_indices = np.argsort(conf)

    # Independent Thompson sampling (ITS) acquisition function
    elif explore_type == 'its':
        mean = np.mean(predictions, axis=0)
        samples = np.random.normal(mean, stds)
        sorted_indices = np.argsort(samples)

    else:
        print('{} is not a valid exploration type'.format(explore_type))
        raise NotImplementedError()

    return sorted_indices

def run(args, num_sample_train_list):
  all_index = np.arange(len(args.df_dict_all))
  all_index_selected = deepcopy(all_index)
  keep_index_train = np.array([]).astype(np.int)
  info_dict = {}
  info_dict['acc_valid'] = args.label_all_train
  info_dict['acc_test'] = args.label_all_test
  info_dict['num_unique_sample'] = []
  info_dict['index_unique_sample'] = []
  info_dict['index_pred_top'] = []
  args.log_file = os.path.join(args.save_dir, 'seeds-{}.pkl'.format(rand_seed))
  if os.path.exists(args.log_file):
    print(f'{args.log_file} already exists')
    return
  log_path = os.path.dirname(args.log_file)
  if not os.path.exists(log_path):
    try:
      os.makedirs(log_path)
    except:
      pass

  for z, (num_sample_train, top_acc) in enumerate(zip(tqdm(num_sample_train_list), args.top_acc_list)):
    random.seed(args.seed+z*args.num_ensemble)
    np.random.seed(args.seed+z*args.num_ensemble)
    if len(all_index_selected) == 0:
      print(f'len all_index_selected = 0')
      break
    if z == 0:
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
        if num_sample_train <= len(all_index_sample):
          train_index_sample = np.random.choice(all_index_sample, size=min(num_sample_train, len(all_index_sample)), replace=False)
        else:
          train_index_sample = []
          for index in all_index_by_acc:
            if index in keep_index_train:
              pass
            else:
              train_index_sample.append(index)
            if len(train_index_sample) >= num_sample_train:
              break
          train_index_sample = np.array(train_index_sample)
      elif args.sampling_method == 'ei':
        train_index_sample = all_index_sample[:num_sample_train]

      if keep_index_train.size != 0:
        assert len(np.intersect1d(keep_index_train, train_index_sample)) == 0
        train_index = np.concatenate((keep_index_train, train_index_sample))
      else:
        train_index = train_index_sample
    assert len(train_index) != 0

    pred_all_list = []
    for i in range(args.num_ensemble):
      random.seed(args.seed + z * args.num_ensemble + i)
      np.random.seed(args.seed + z * args.num_ensemble + i)
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
        regr = xgb.train(params=params, dtrain=dtrain, num_boost_round=args.n_trees)
        pred_all = regr.predict(dall)
      else:
        if 'RandomForest' in args.predictor:
          regr = RandomForestRegressor(n_estimators=args.random_forest_num, max_depth=10) # max_depth=10, n_estimators=500
        elif 'MLP' in args.predictor:
          regr = MLPRegressor(hidden_layer_sizes=args.mlp_size, max_iter=args.mlp_iter) #max_iter=200  hidden_layer_sizes=(100, 100), max_iter=200, solver={‘lbfgs’, ‘sgd’, ‘adam’}
        regr.fit(args.df_dict_all.iloc[train_index], args.norm_label_all_train[train_index])
        pred_all = regr.predict(args.df_dict_all)
      pred_all_list.append(pred_all)

    all_index_by_acc = (-pred_all).argsort()
    if args.sampling_method == 'ei':
      ytrain=args.norm_label_all_train[train_index]
      all_index_selected = acq_fn(pred_all_list, ytrain=ytrain, explore_type='ei')
    else:
      all_index_selected = all_index_by_acc[:top_acc]

    info_dict['num_unique_sample'].append(len(train_index))
    info_dict['index_unique_sample'].append(train_index)
    info_dict['index_pred_top'].append(all_index_by_acc[:args.save_top])

    if args.keep_old == 'none':
      keep_index_train = np.array([]).astype(np.int)
    elif args.keep_old == 'top':
      keep_index_train = np.array([i for i in all_index_selected if i in train_index]).astype(np.int)
    elif args.keep_old == 'all':
      keep_index_train = train_index
  with open(args.log_file, 'wb') as handle:
    pickle.dump(info_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
  return args

def get_y(x, decay, max_, min_):
  y = np.power(decay, x)
  y = (y-y.min())/(y.max()-y.min())
  y = y * (max_-min_) + min_
  y = y.astype(int)
  return y

def sample_scheduler(decay, iteration, start, end):
  if iteration == 1:
    y = np.array([start])
  else:
    base_end = 100
    x = np.linspace(0, base_end-1, num=iteration, endpoint=True)
    y = get_y(x=x, decay=decay, max_=start, min_=end)
  return y

def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def main(args, bench_dict):
  if args.sample_decay == 'none':
    if args.init_sample > args.sample_each_iter:
      num_sample_train_list = [args.init_sample] + [args.sample_each_iter] * int(np.ceil((args.max_sample-args.init_sample+args.sample_each_iter) / args.sample_each_iter - 1))
    else:
      num_sample_train_list = [args.sample_each_iter] * int(np.ceil((args.max_sample-args.init_sample+args.sample_each_iter) / args.sample_each_iter))
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.set_num_threads( args.workers )
  arch_dict = bench_dict
  args.all_items = list(arch_dict.items())
  if args.bench == 'nasbench201':
    if args.dataset == 'cifar10':
      setname_list = ['train', 'ori-test']
    elif args.dataset == 'cifar10-valid':
      setname_list = ['train', 'x-valid', 'ori-test']
    elif args.dataset == 'cifar100' or args.dataset == 'ImageNet16-120':
      setname_list = ['train', 'x-valid', 'x-test', 'ori-test']
    assert args.train_set in setname_list
    disp2key = {'Arch': ['arch']}
  elif args.bench == 'nasbench101':
    disp2key = {'Arch': ['arch']}
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
    if type(data_emp) is tuple:
      assert feature == 'Arch'
      if args.bench == 'nasbench201':
        assert len(data_emp) == 6
        if args.mlp_onehot:
          data_list = np.array([one_hot(a=np.array(data), num_classes=5).flatten() for data in data_list])
          data_emp = data_list[0]
          for i, d in enumerate(data_emp):
            arch_array = np.array([float(data[i]) for data in data_list])
            arch_list = list(arch_array)
            args.df_dict_all[f'{feature}: {i + 1}'] = arch_list
        else:
          node_order_list = np.arange(6)
          args.node_order_list = node_order_list
          for i, (node_idx, d) in enumerate(zip(node_order_list, data_emp)):
            arch_array = np.array([float(data[node_idx]) for data in data_list])
            arch_list = list(arch_array)
            args.df_dict_all[f'{feature}: {i + 1}'] = arch_list
      elif args.bench == 'nasbench101':
        arch_list = []
        for data in data_list:
          assert len(data) == 2
          adj, ops = data
          if args.emb_type == ['org']:
            arch = convert_arch_to_seq(deepcopy(adj), deepcopy(ops))
          else:
            arch = lanas_mod(deepcopy(adj), deepcopy(ops), transform=args.emb_type)
          arch_list.append(arch)
        arch_array = np.array(arch_list)
        for i in range(arch_array.shape[1]):
          args.df_dict_all[f'{feature}: {i + 1}'] = arch_array[:,i].tolist()
    else:
      print(feature, type(data_emp))
      raise ValueError
  if args.deterministic:
    args.label_all_train = np.array([get_dict(value, args.label_key_dict['train']+['avg']) for i, (key, value) in enumerate(arch_dict.items())]).astype(np.float64)*100
    args.label_all_test = np.array([get_dict(value, args.label_key_dict['test']+['avg']) for i, (key, value) in enumerate(arch_dict.items())]).astype(np.float64)*100
  else:
    args.label_all_train = np.array([get_dict(value, args.label_key_dict['train']+[random.randint(0,2)]) for i, (key, value) in enumerate(arch_dict.items())]).astype(np.float64)*100
    args.label_all_test = np.array([get_dict(value, args.label_key_dict['test']+['avg']) for i, (key, value) in enumerate(arch_dict.items())]).astype(np.float64)*100
  args.norm_label_all_train = (args.label_all_train - np.mean(args.label_all_train)) / np.std(args.label_all_train)
  args.index_by_acc_train = (-args.label_all_train).argsort()
  args.acc_optimal_train = max(args.label_all_train)
  args.index_optimal_train = np.argmax(args.label_all_train)
  args.index_by_acc_test = (-args.label_all_test).argsort()
  args.acc_optimal_test = max(args.label_all_test)
  args.index_optimal_test = np.argmax(args.label_all_test)
  args.acc_optimal_test_oracle = args.label_all_test[args.index_optimal_train]
  args.iteration = len(num_sample_train_list)
  if not args.top_increase:
    assert args.top_start >= args.top_end, f'top_start: {args.top_start}, top_end: {args.top_end}'
  args.top_acc_list = sample_scheduler(args.top_decay, args.iteration, args.top_start, args.top_end)
  run(args=args, num_sample_train_list=num_sample_train_list)

if __name__ == '__main__':

  parser = argparse.ArgumentParser("Regularized Evolution Algorithm")
  parser.add_argument('--repeat', type=int, default=1)
  parser.add_argument('--save_dir', type=str, default='')
  parser.add_argument('--rand_seed', type=int, default=-1)
  parser.add_argument('--sampling_method', type=str,  default='uniform')
  parser.add_argument('--dataset', type=str,  default='cifar10')
  parser.add_argument('--bench', type=str,  default='nasbench101')
  parser.add_argument('--keep_old', type=str,  default='all')
  parser.add_argument('--max_sample', type=int,   default=1000)
  parser.add_argument('--num_ensemble', type=int,   default=5)
  parser.add_argument('--init_sample', type=int,   default=100)
  parser.add_argument('--n_trees', type=int,   default=1000)
  parser.add_argument('--max_depth', type=int,   default=20)
  parser.add_argument('--iteration', type=int,   default=5)
  parser.add_argument('--top_max', type=int,   default=100)
  parser.add_argument('--top_min', type=int,   default=100)
  parser.add_argument('--feature_list', type=str, default='Arch')
  parser.add_argument('--emb_type', type=str, default='org')
  parser.add_argument('--info_path', type=str, default=None)
  parser.add_argument('--emb', type=str, default=None)
  parser.add_argument('--bench_path', type=str, default='DATASET/nasbench101_minimal.pth.tar')
  parser.add_argument('--workers', type=int, default=8)
  parser.add_argument('--top_start', type=int, default=80000)
  parser.add_argument('--top_end', type=int, default=500)
  parser.add_argument('--top_decay', type=float, default=0.96)
  parser.add_argument('--sample_each_iter',  type=int,   default=10)
  parser.add_argument('--train_set', type=str,  default='valid')
  parser.add_argument('--test_set', type=str,  default='test')
  parser.add_argument('--save_top', type=int,   default=1000)
  parser.add_argument('--top_increase', default=False, action='store_true')
  parser.add_argument('--loss', type=str,  default='reg:squarederror')
  parser.add_argument('--predictor', type=str, nargs='+', default=['MLP'])
  parser.add_argument('--mlp_size', type=int, nargs='+',  default=[1000, 1000, 1000, 1000])
  parser.add_argument('--mlp_iter', type=int,  default=100)
  parser.add_argument('--mlp_onehot', default=True, action='store_true')
  parser.add_argument('--random_forest_num', type=int, default=1000)
  parser.add_argument('--gpu', default=False, action='store_true')
  parser.add_argument('--sample_decay', type=str, default='none')
  parser.add_argument('--sample_decay_step', help="step", type=coords, nargs='+')
  parser.add_argument('--sample_linear_iteration', help="step", type=int, default=10)
  parser.add_argument('--sample_linear_end', help="step", type=int, default=10)
  parser.add_argument('--sample_linear_start', help="step", type=int, default=1)
  parser.add_argument('--deterministic', default=False)
  args = parser.parse_args()
  if args.sampling_method == 'uniform':
    args.num_ensemble = 1
  args.feature_list = [x for x in args.feature_list.split(",")]
  args.emb_type = [x for x in args.emb_type.split(",")]
  if args.bench == 'nasbench101':
    args.dataset == 'cifar10'
  print(f'Saving to {args.save_dir}')
  if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir, exist_ok=True)
  bench_dict = torch.load(args.bench_path)
  print(f'loaded bench from {args.bench_path}')
  if args.bench == 'nasbench201':
    args.label_key_dict = {'train': ['scratch', args.train_set, 'acc', 200],
                           'test': ['scratch', args.test_set, 'acc', 200]}
  elif args.bench == 'nasbench101':
    args.label_key_dict = {'train' :['scratch', args.train_set, 'acc', 108],
                           'test' :['scratch', args.test_set, 'acc', 108]}
  if args.rand_seed < 0:
    for args.repeat_iteration in tqdm(range(args.repeat)):
      rand_seed = random.randint(0, 2**32-1)
      args.seed = rand_seed
      prepare_seed(rand_seed)
      main(args, bench_dict)