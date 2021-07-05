##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##############################################################################
# Random Search and Reproducibility for Neural Architecture Search, UAI 2019 # 
##############################################################################
import torch, random
import torch.nn as nn
from copy import deepcopy
from ..cell_operations import ResNetBasicblock
from .search_cells     import NAS201SearchCell as SearchCell
from .genotypes        import Structure
from .fbnet import IRFBlock, Hsigmoid, Identity
import numpy as np
from .ops import Hswish
from .fbnet import OPS_1


class TinyNetworkRANDOM(nn.Module):

  def __init__(self, C, N, max_nodes, num_classes, search_space, affine, track_running_stats):
    super(TinyNetworkRANDOM, self).__init__()
    self._C        = C
    self._layerN   = N
    self.max_nodes = max_nodes
    self.stem = nn.Sequential(
                    nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(C))
  
    layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N    
    layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

    C_prev, num_edge, edge2index = C, None, None
    self.cells = nn.ModuleList()
    for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
      if reduction:
        cell = ResNetBasicblock(C_prev, C_curr, 2)
      else:
        cell = SearchCell(C_prev, C_curr, 1, max_nodes, search_space, affine, track_running_stats)
        if num_edge is None: num_edge, edge2index = cell.num_edges, cell.edge2index
        else: assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(num_edge, cell.num_edges)
      self.cells.append( cell )
      C_prev = cell.out_dim
    self.op_names   = deepcopy( search_space )
    self._Layer     = len(self.cells)
    self.edge2index = edge2index
    self.lastact    = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)
    self.arch_cache = None
    
  def get_message(self):
    string = self.extra_repr()
    for i, cell in enumerate(self.cells):
      string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
    return string

  def extra_repr(self):
    return ('{name}(C={_C}, Max-Nodes={max_nodes}, N={_layerN}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__))

  def random_genotype(self, set_cache):
    genotypes = []
    for i in range(1, self.max_nodes):
      xlist = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        op_name  = random.choice( self.op_names )
        xlist.append((op_name, j))
      genotypes.append( tuple(xlist) )
    arch = Structure( genotypes )
    if set_cache: self.arch_cache = arch
    return arch

  def forward(self, inputs):

    feature = self.stem(inputs)
    for i, cell in enumerate(self.cells):
      if isinstance(cell, SearchCell):
        feature = cell.forward_dynamic(feature, self.arch_cache)
      else: feature = cell(feature)

    out = self.lastact(feature)
    out = self.global_pooling( out )
    out = out.view(out.size(0), -1)
    logits = self.classifier(out)
    return out, logits


class MobileNetRANDOM(nn.Module):

  def __init__(self, n_class=100, stages=None, zero_gamma=False, affine=False):
    super(MobileNetRANDOM, self).__init__()
    assert stages != None
    self.zero_gamma = zero_gamma
    self.affine = affine
    self.stages = stages
    self.stages_len = [len(s) for s in self.stages]
    self.stages_fine = [s for stage in self.stages for s in stage]
    self.stage_out_channels = [16, 16, 24, 24, 40, 40, 40, 48, 48, 96, 96, 96, 576]
    self.exp_size = [16, 72, 88, 96, 240, 240, 120, 144, 288, 576, 576, 576]
    self.downsample_list = [False, False, True, False, True, False, False, False, False, True, False, False, False]
    self.se_list = [False, True, False, False, True, True, True, True, True, True, True, True, True]
    self.act_list = ["hswish", "hswish", "hswish", "hswish", "hswish", "hswish", "hswish", "hswish", "hswish", "hswish",
                     "hswish", "hswish", "hswish"]
    input_channel = self.stage_out_channels[0]
    self.first_conv = nn.Sequential(
        nn.Conv2d(3, input_channel, kernel_size=3, padding=1),
        nn.BatchNorm2d(input_channel, affine=self.affine),
        Hswish(inplace=True),
    )
    self.features = torch.nn.ModuleList()
    for i, PRIMITIVES in enumerate(self.stages_fine):
        output_channel = self.stage_out_channels[i + 1]
        act_func = self.act_list[i + 1]
        downsample = self.downsample_list[i + 1]
        se = self.se_list[i + 1]
        self.features.append(torch.nn.ModuleList())
        for PRIMITIVE in PRIMITIVES:
            inp, outp = input_channel, output_channel
            if downsample:
                stride = 2
            else:
                stride = 1
            base_mid_channels = outp // 2
            mid_channels = int(base_mid_channels)
            self.features[-1].append(OPS_1[PRIMITIVE](inp, outp, mid_channels, stride, act_func, se, self.affine))
        input_channel = output_channel
    self.conv_last = nn.Sequential(
        nn.Conv2d(input_channel, self.stage_out_channels[-1], kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(self.stage_out_channels[-1], affine=self.affine),
        Hswish(inplace=True),
    )
    self.globalpool = nn.AdaptiveAvgPool2d(1)
    self.last = nn.Sequential(
        nn.Conv2d(self.stage_out_channels[-1], 1024, kernel_size=1, stride=1, padding=0),
        nn.Flatten(),
        Hswish(inplace=True),
        nn.Dropout(p=0.8),
        # nn.Linear(1024, n_class),
    )
    self.classifier = nn.Linear(1024, n_class)
    self.arch_cache = None
    self.op_names   = [0,1,2,3,4,5]
    self._initialize_weights()

  def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        if self.zero_gamma:
            last_bn_list = [m.pwl.bn for m in self.modules() if isinstance(m, IRFBlock)]
            for last_bn in last_bn_list:
                nn.init.constant_(last_bn.weight, 0.0)

  def get_message(self):
    string = self.extra_repr()
    for i, cell in enumerate(self.cells):
      string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
    return string

  def extra_repr(self):
    return (f'{self.__class__.__name__,}(groups={[len(s) for s in self.stages]}, ops={self.stages[0][0]}, affine: {self.affine}, zero_gamma: {self.zero_gamma})')

  def random_genotype(self, set_cache):
      arch = [random.choice(self.op_names) for _ in self.stages]
      if set_cache:
          self.arch_cache = arch
      return tuple(arch)

  def update_arch(self, _arch):
    self.arch_cache = _arch

  # def random_genotype(self, set_cache):
  #   genotypes = []
  #   for i in range(1, self.max_nodes):
  #     xlist = []
  #     for j in range(i):
  #       node_str = '{:}<-{:}'.format(i, j)
  #       op_name = random.choice(self.op_names)
  #       xlist.append((op_name, j))
  #     genotypes.append(tuple(xlist))
  #   arch = Structure(genotypes)
  #   if set_cache:
  #     self.arch_cache = arch
  #   return arch

  def forward(self, x):

    x = self.first_conv(x)
    arch_cache_full = [self.arch_cache[i] for i, s in enumerate(self.stages_len) for _ in range(s)]
    for i, (features, index) in enumerate(zip(self.features, arch_cache_full)):
        x = features[index](x)
    x = self.conv_last(x)
    x = self.globalpool(x)
    x = self.last(x)
    x = x.view(x.size(0), -1)
    logits = self.classifier(x)
    return x, logits
    # feature = self.stem(inputs)
    # for i, cell in enumerate(self.cells):
    #   if isinstance(cell, SearchCell):
    #     feature = cell.forward_dynamic(feature, self.arch_cache)
    #   else:
    #     feature = cell(feature)
    #
    # out = self.lastact(feature)
    # out = self.global_pooling(out)
    # out = out.view(out.size(0), -1)
    # logits = self.classifier(out)
    # return out, logits

