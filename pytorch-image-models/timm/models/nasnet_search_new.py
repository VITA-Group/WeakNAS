import torch
import torch.nn as nn
from copy import deepcopy
from .cell_operations import DARTS_SPACE
# from .search_cells     import NAS201SearchCell
from .search_cells     import NASNetSearchCell
from .genotypes        import Structure
from collections import namedtuple
from .registry import register_model

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

WeakNAS_imagenet = Genotype(normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('dil_conv_5x5', 3), ('sep_conv_3x3', 0), ('dil_conv_3x3', 4)],
                            normal_concat=[2, 3, 4, 5],
                            reduce=[('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 4)],
                            reduce_concat=[2, 3, 4, 5])

WeakNAS_imagenet = Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('skip_connect', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 3)],
                            normal_concat=[2, 3, 4, 5],
                            reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_5x5', 2), ('sep_conv_3x3', 4)],
                            reduce_concat=[2, 3, 4, 5])

default_cfgs = {
    'nasnet_800': WeakNAS_imagenet,
}
# , auxiliary=False, genotype=default_cfgs['nasnet_800']
# @register_model
# def nasnet_800(pretrained=False, **kwargs):
#     model = NASNetSuperNet(C=36, N=20, steps=4, multiplier=4, stem_multiplier=3, num_classes=1000, search_space=DARTS_SPACE, affine=True, track_running_stats=True, **kwargs)
#     return model


class NASNetSuperNet(nn.Module):

  def __init__(self, C, N, steps, multiplier, stem_multiplier, num_classes, search_space, affine, track_running_stats, drop_rate=0.1, drop_path_rate=0.0):
    super(NASNetSuperNet, self).__init__()
    self._C = C
    self._layerN = N
    self._steps = steps
    self._multiplier = multiplier
    self.num_classes = num_classes
    self.stem = nn.Sequential(
      nn.Conv2d(3, C * stem_multiplier, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(C * stem_multiplier))

    # config for each layer
    layer_channels = [C] * N + [C * 2] + [C * 2] * (N - 1) + [C * 4] + [C * 4] * (N - 1)
    layer_reductions = [False] * N + [True] + [False] * (N - 1) + [True] + [False] * (N - 1)

    num_edge, edge2index = None, None
    C_prev_prev, C_prev, C_curr, reduction_prev = C * stem_multiplier, C * stem_multiplier, C, False

    self.cells = nn.ModuleList()
    for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
      cell = NASNetSearchCell(search_space, steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, affine,
                        track_running_stats)
      if num_edge is None:
        num_edge, edge2index = cell.num_edges, cell.edge2index
      else:
        assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(num_edge,
                                                                                                           cell.num_edges)
      self.cells.append(cell)
      C_prev_prev, C_prev, reduction_prev = C_prev, multiplier * C_curr, reduction
    self.op_names = deepcopy(search_space)
    self._Layer = len(self.cells)
    self.edge2index = edge2index
    self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

  def update_arch(self, _arch):
    if _arch is None:
      self.sampled_arch = None
    elif isinstance(_arch, Structure):
      self.sampled_arch = _arch
    elif isinstance(_arch, (list, tuple)):
      genotypes = []
      index = 0
      for i in range(1, self.max_nodes):
        xlist = []
        for j in range(i):
          node_str = '{:}<-{:}'.format(i, j)
          # print(self.edge2index[node_str])
          op_index = _arch[ self.edge2index[node_str] ]
          op_name  = self.op_names_list[index][ op_index ]
          xlist.append((op_name, j))
          index += 1
        genotypes.append( tuple(xlist) )
      self.sampled_arch = Structure(genotypes)
    else:
      raise ValueError('invalid type of input architecture : {:}'.format(_arch))
    return self.sampled_arch

  def get_message(self):
    string = self.extra_repr()
    for i, cell in enumerate(self.cells):
      string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
    return string

  def extra_repr(self):
    return ('{name}(C={_C}, N={_layerN}, steps={_steps}, multiplier={_multiplier}, L={_Layer})'.format(
      name=self.__class__.__name__, **self.__dict__))

  def forward(self, inputs):

    s0 = s1 = self.stem(inputs)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        ww = reduce_w
      else:
        ww = normal_w
      s0, s1 = s1, cell.forward_darts(s0, s1, ww)
    out = self.lastact(s1)
    out = self.global_pooling(out)
    out = out.view(out.size(0), -1)
    logits = self.classifier(out)

    return out, logits
