import numpy as np

# INPUT = 'input'
# CONV1X1 = 'conv1x1-bn-relu'
# CONV3X3 = 'conv3x3-bn-relu'
# MAXPOOL3X3 = 'maxpool3x3'
# OUTPUT = 'output'

INPUT = 0
CONV1X1 = 1
CONV3X3 = 2
MAXPOOL3X3 = 3
OUTPUT = 4

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


def convert_seq_to_arch(seq):
  n = 7
  matrix = [[0 for _ in range(n)] for _ in range(n)]
  ops = [INPUT]
  for i in range(n - 1):
    offset = (i + 9) * i // 2
    for j in range(i + 1):
      matrix[j][i + 1] = seq[offset + j]
    if seq[offset + i + 1] == 1:
      op = CONV1X1
    elif seq[offset + i + 2] == 1:
      op = CONV3X3
    elif seq[offset + i + 3] == 1:
      op = MAXPOOL3X3
    elif seq[offset + i + 4] == 1:
      op = OUTPUT
    ops.append(op)
  return matrix, ops


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