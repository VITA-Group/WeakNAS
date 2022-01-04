import torch
import torch.nn as nn
from .layers import OPS, ReLUConvBN, FactorizedReduce, Identity, drop_path
from timm.models.ofa.utils.layers import LinearLayer
# from operations import OPS, ReLUConvBN, FactorizedReduce, Identity
from collections import namedtuple
from .registry import register_model

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


# WeakNAS_imagenet = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('avg_pool_3x3', 1)], normal_concat=[2, 3, 4, 5],
#                             reduce=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 0), ('max_pool_3x3', 2), ('sep_conv_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 3)], reduce_concat=[2, 3, 4, 5])

# WeakNAS_imagenet = Genotype(normal=[('sep_conv_5x5', 1), ('dil_conv_5x5', 0), ('skip_connect', 2), ('sep_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('sep_conv_5x5', 1)], normal_concat=[2, 3, 4, 5],
#                             reduce=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 3), ('dil_conv_5x5', 2), ('dil_conv_5x5', 4), ('sep_conv_5x5', 3)], reduce_concat=[2, 3, 4, 5])
#
# WeakNAS_imagenet = Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('skip_connect', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 3)], normal_concat=[2, 3, 4, 5],
#                             reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_5x5', 2), ('sep_conv_3x3', 4)], reduce_concat=[2, 3, 4, 5])

WeakNAS_imagenet = Genotype(normal=[('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('skip_connect', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 3)], normal_concat=[2, 3, 4, 5],
                            reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 3), ('sep_conv_5x5', 2), ('sep_conv_5x5', 4), ('sep_conv_3x3', 4)], reduce_concat=[2, 3, 4, 5])

default_cfgs = {
    'nasnet_800': WeakNAS_imagenet,
}

@register_model
def nasnet_800(pretrained=False, **kwargs):
    # model = NetworkImageNet(C=36, num_classes=1000, layers=20, genotype=default_cfgs['nasnet_800'], **kwargs)
    model = NetworkImageNet(C=46, num_classes=1000, layers=14, genotype=default_cfgs['nasnet_800'], **kwargs)
    return model


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        # print(C_prev_prev, C_prev, C)

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class NetworkImageNet(nn.Module):

    def __init__(self, C, num_classes, layers, drop_rate=0.1, drop_path_rate=0.0, genotype=default_cfgs['nasnet_800']):
        super(NetworkImageNet, self).__init__()
        self._layers = layers
        self.num_classes = num_classes
        self.drop_path_prob = drop_path_rate

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = LinearLayer(C_prev, num_classes, dropout_rate=drop_rate)
        # self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            # print(s0.shape, s1.shape)
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits
