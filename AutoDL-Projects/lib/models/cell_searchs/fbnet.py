import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import BatchNorm2d
from torch.nn.modules.utils import _ntuple
from .ops import Hswish, Hsigmoid, Swish

OPS_1 = {
'mobv3_k1_e2': lambda C_in, C_out, mid_channels, stride, act_func, se, affine, **kwargs: IRFBlock(
                    C_in, C_out, 2, stride, kernel=1, se=se, act_func=act_func,
                    se_last_act=Hsigmoid, se_pos='mid', affine=affine, **kwargs),
'mobv3_k1_e4': lambda C_in, C_out, mid_channels, stride, act_func, se, affine, **kwargs: IRFBlock(
                    C_in, C_out, 4, stride, kernel=1, se=se, act_func=act_func,
                    se_last_act=Hsigmoid, se_pos='mid', affine=affine, **kwargs),
'mobv3_k1_e6': lambda C_in, C_out, mid_channels, stride, act_func, se, affine, **kwargs: IRFBlock(
                    C_in, C_out, 6, stride, kernel=1, se=se, act_func=act_func,
                    se_last_act=Hsigmoid, se_pos='mid', affine=affine, **kwargs),
'mobv3_k3_e1': lambda C_in, C_out, mid_channels, stride, act_func, se, affine, **kwargs: IRFBlock(
                    C_in, C_out, 1, stride, kernel=3, se=se, act_func=act_func,
                    se_last_act=Hsigmoid, se_pos='mid', affine=affine, **kwargs),
'mobv3_k3_e2': lambda C_in, C_out, mid_channels, stride, act_func, se, affine, **kwargs: IRFBlock(
                    C_in, C_out, 2, stride, kernel=3, se=se, act_func=act_func,
                    se_last_act=Hsigmoid, se_pos='mid', affine=affine, **kwargs),
'mobv3_k3_e3': lambda C_in, C_out, mid_channels, stride, act_func, se, affine, **kwargs: IRFBlock(
                    C_in, C_out, 3, stride, kernel=3, se=se, act_func=act_func,
                    se_last_act=Hsigmoid, se_pos='mid', affine=affine, **kwargs),
'mobv3_k3_e4': lambda C_in, C_out, mid_channels, stride, act_func, se, affine, **kwargs: IRFBlock(
                    C_in, C_out, 4, stride, kernel=3, se=se, act_func=act_func,
                    se_last_act=Hsigmoid, se_pos='mid', affine=affine, **kwargs),
'mobv3_k3_e4.5': lambda C_in, C_out, mid_channels, stride, act_func, se, affine, **kwargs: IRFBlock(
                    C_in, C_out, 4.5, stride, kernel=3, se=se, act_func=act_func,
                    se_last_act=Hsigmoid, se_pos='mid', affine=affine, **kwargs),
'mobv3_k3_e6': lambda C_in, C_out, mid_channels, stride, act_func, se, affine, **kwargs: IRFBlock(
                    C_in, C_out, 6, stride, kernel=3, se=se, act_func=act_func,
                    se_last_act=Hsigmoid, se_pos='mid', affine=affine, **kwargs),
'mobv3_k5_e2': lambda C_in, C_out, mid_channels, stride, act_func, se, affine, **kwargs: IRFBlock(
                    C_in, C_out, 2, stride, kernel=5, se=se, act_func=act_func,
                    se_last_act=Hsigmoid, se_pos='mid', affine=affine, **kwargs),
'mobv3_k5_e3': lambda C_in, C_out, mid_channels, stride, act_func, se, affine, **kwargs: IRFBlock(
                    C_in, C_out, 3, stride, kernel=5, se=se, act_func=act_func,
                    se_last_act=Hsigmoid, se_pos='mid', affine=affine, **kwargs),
'mobv3_k5_e4': lambda C_in, C_out, mid_channels, stride, act_func, se, affine, **kwargs: IRFBlock(
                    C_in, C_out, 4, stride, kernel=5, se=se, act_func=act_func,
                    se_last_act=Hsigmoid, se_pos='mid', affine=affine, **kwargs),
'mobv3_k5_e6': lambda C_in, C_out, mid_channels, stride, act_func, se, affine, **kwargs: IRFBlock(
                    C_in, C_out, 6, stride, kernel=5, se=se, act_func=act_func,
                    se_last_act=Hsigmoid, se_pos='mid', affine=affine, **kwargs),
'mobv3_k7_e3': lambda C_in, C_out, mid_channels, stride, act_func, se, affine, **kwargs: IRFBlock(
                    C_in, C_out, 3, stride, kernel=7, se=se, act_func=act_func,
                    se_last_act=Hsigmoid, se_pos='mid', affine=affine, **kwargs),
'mobv3_k7_e6': lambda C_in, C_out, mid_channels, stride, act_func, se, affine, **kwargs: IRFBlock(
                    C_in, C_out, 6, stride, kernel=1, se=se, act_func=act_func,
                    se_last_act=Hsigmoid, se_pos='mid', affine=affine, **kwargs),
'mobv3_k3_e1': lambda C_in, C_out, mid_channels, stride, act_func, se, affine, **kwargs: IRFBlock(
                    C_in, C_out, 1, stride, kernel=3, se=se, act_func=act_func,
                    se_last_act=Hsigmoid, se_pos='mid', affine=affine, **kwargs),
'mobv3_k5_e4': lambda C_in, C_out, mid_channels, stride, act_func, se, affine, **kwargs: IRFBlock(
                    C_in, C_out, 4, stride, kernel=5, se=se, act_func=act_func,
                    se_last_act=Hsigmoid, se_pos='mid', affine=affine, **kwargs),
'mobv3_k3_e11/3': lambda C_in, C_out, mid_channels, stride, act_func, se, affine, **kwargs: IRFBlock(
                    C_in, C_out, 11/3, stride, kernel=3, se=se, act_func=act_func,
                    se_last_act=Hsigmoid, se_pos='mid', affine=affine, **kwargs),
'conv1x1': lambda C_in, C_out, mid_channels, stride, act_func, se, affine, **kwargs: nn.Conv2d(
                    C_in, C_out, kernel_size=1, padding=0),
'conv3x3': lambda C_in, C_out, mid_channels, stride, act_func, se, affine, **kwargs: nn.Conv2d(
                    C_in, C_out, kernel_size=3, padding=1),
'conv5x5': lambda C_in, C_out, mid_channels, stride, act_func, se, affine, **kwargs: nn.Conv2d(
                    C_in, C_out, kernel_size=5, padding=2),
'conv7x7': lambda C_in, C_out, mid_channels, stride, act_func, se, affine, **kwargs: nn.Conv2d(
                    C_in, C_out, kernel_size=7, padding=3),
'shuffle_3x3': lambda C_in, C_out, mid_channels, stride, act_func, se, affine, **kwargs: Shufflenet(
                    C_in, C_out, mid_channels=C_out//2, ksize=3, stride=stride, affine=affine),
'shuffle_5x5': lambda C_in, C_out, mid_channels, stride, act_func, se, affine, **kwargs: Shufflenet(
                    C_in, C_out, mid_channels=C_out//2, ksize=5, stride=stride, affine=affine),
'shuffle_7x7': lambda C_in, C_out, mid_channels, stride, act_func, se, affine, **kwargs: Shufflenet(
                    C_in, C_out, mid_channels=C_out//2, ksize=7, stride=stride, affine=affine),
'shuffle_xception': lambda C_in, C_out, mid_channels, stride, act_func, se, affine, **kwargs: Shuffle_Xception(
                    C_in, C_out, mid_channels=C_out//2, stride=stride, affine=affine),
'skip':             lambda C_in, C_out, mid_channels, stride, act_func, se, affine, **kwargs: Identity(
                    C_in, C_out, stride, act_func=act_func, affine=affine, **kwargs),
}

class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


class Conv2d(torch.nn.Conv2d):
    def forward(self, x):
        if x.numel() > 0:
            return super(Conv2d, self).forward(x)
        # get output shape

        output_shape = [
            (i + 2 * p - (di * (k - 1) + 1)) // d + 1
            for i, p, di, k, d in zip(
                x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
            )
        ]
        output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class ConvTranspose2d(torch.nn.ConvTranspose2d):
    def forward(self, x):
        if x.numel() > 0:
            return super(ConvTranspose2d, self).forward(x)
        # get output shape

        output_shape = [
            (i - 1) * d - 2 * p + (di * (k - 1) + 1) + op
            for i, p, di, k, d, op in zip(
                x.shape[-2:],
                self.padding,
                self.dilation,
                self.kernel_size,
                self.stride,
                self.output_padding,
            )
        ]
        output_shape = [x.shape[0], self.bias.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


def interpolate(
        input, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    if input.numel() > 0:
        return torch.nn.functional.interpolate(
            input, size, scale_factor, mode, align_corners
        )

    def _check_size_scale_factor(dim):
        if size is None and scale_factor is None:
            raise ValueError("either size or scale_factor should be defined")
        if size is not None and scale_factor is not None:
            raise ValueError("only one of size or scale_factor should be defined")
        if (
                scale_factor is not None
                and isinstance(scale_factor, tuple)
                and len(scale_factor) != dim
        ):
            raise ValueError(
                "scale_factor shape must match input shape. "
                "Input is {}D, scale_factor size is {}".format(dim, len(scale_factor))
            )

    def _output_size(dim):
        _check_size_scale_factor(dim)
        if size is not None:
            return size
        scale_factors = _ntuple(dim)(scale_factor)
        # math.floor might return float in py2.7
        return [
            int(math.floor(input.size(i + 2) * scale_factors[i])) for i in range(dim)
        ]

    output_shape = tuple(_output_size(2))
    output_shape = input.shape[:-2] + output_shape
    return _NewEmptyTensorOp.apply(input, output_shape)


def _py2_round(x):
    return math.floor(x + 0.5) if x >= 0.0 else math.ceil(x - 0.5)


def get_divisible_by(num, divisible_by, min_val):
    ret = int(num)
    if divisible_by > 0 and num % divisible_by != 0:
        ret = int((_py2_round(num / divisible_by) or min_val) * divisible_by)
    if ret < 0.9 * num:
        ret += divisible_by
    return ret


class Identity(nn.Module):
    def __init__(self, C_in, C_out, stride, act_func, bn_type="bn", affine=False):
        super(Identity, self).__init__()
        self.conv = (
            ConvBNAct(
                C_in,
                C_out,
                kernel=1,
                stride=stride,
                pad=(1 // 2),
                group=1,
                no_bias=1,
                act_func=act_func,
                bn_type=bn_type,
                affine=affine,
            )
            if C_in != C_out or stride != 1
            else None
        )

    def forward(self, x):
        if self.conv:
            out = self.conv(x)
        else:
            out = x
        return out


class CascadeConv3x3(nn.Sequential):
    def __init__(self, C_in, C_out, stride, affine):
        assert stride in [1, 2]
        ops = [
            Conv2d(C_in, C_in, 3, stride, 1, bias=False),
            BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=True),
            Conv2d(C_in, C_out, 3, 1, 1, bias=False),
            BatchNorm2d(C_out, affine=affine),
        ]
        super(CascadeConv3x3, self).__init__(*ops)
        self.res_connect = (stride == 1) and (C_in == C_out)

    def forward(self, x):
        y = super(CascadeConv3x3, self).forward(x)
        if self.res_connect:
            y += x
        return y


class Shift(nn.Module):
    def __init__(self, C, kernel_size, stride, padding):
        super(Shift, self).__init__()
        self.C = C
        kernel = torch.zeros((C, 1, kernel_size, kernel_size), dtype=torch.float32)
        ch_idx = 0

        assert stride in [1, 2]
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.dilation = 1

        hks = kernel_size // 2
        ksq = kernel_size ** 2

        for i in range(kernel_size):
            for j in range(kernel_size):
                if i == hks and j == hks:
                    num_ch = C // ksq + C % ksq
                else:
                    num_ch = C // ksq
                kernel[ch_idx : ch_idx + num_ch, 0, i, j] = 1
                ch_idx += num_ch

        self.register_parameter("bias", None)
        self.kernel = nn.Parameter(kernel, requires_grad=False)

    def forward(self, x):
        if x.numel() > 0:
            return nn.functional.conv2d(
                x,
                self.kernel,
                self.bias,
                (self.stride, self.stride),
                (self.padding, self.padding),
                self.dilation,
                self.C,  # groups
            )

        output_shape = [
            (i + 2 * p - (di * (k - 1) + 1)) // d + 1
            for i, p, di, k, d in zip(
                x.shape[-2:],
                (self.padding, self.dilation),
                (self.dilation, self.dilation),
                (self.kernel_size, self.kernel_size),
                (self.stride, self.stride),
            )
        ]
        output_shape = [x.shape[0], self.C] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class ShiftBlock5x5(nn.Sequential):
    def __init__(self, C_in, C_out, expansion, stride, affine):
        assert stride in [1, 2]
        self.res_connect = (stride == 1) and (C_in == C_out)

        C_mid = get_divisible_by(C_in * expansion, 8, 1)

        ops = [
            # pw
            Conv2d(C_in, C_mid, 1, 1, 0, bias=False),
            BatchNorm2d(C_mid, affine=affine),
            nn.ReLU(inplace=True),
            # shift
            Shift(C_mid, 5, stride, 2),
            # pw-linear
            Conv2d(C_mid, C_out, 1, 1, 0, bias=False),
            BatchNorm2d(C_out, affine=affine),
        ]
        super(ShiftBlock5x5, self).__init__(*ops)

    def forward(self, x):
        y = super(ShiftBlock5x5, self).forward(x)
        if self.res_connect:
            y += x
        return y


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        assert C % g == 0, "Incompatible group size {} for input channel {}".format(
            g, C
        )
        return (
            x.view(N, g, int(C / g), H, W)
                .permute(0, 2, 1, 3, 4)
                .contiguous()
                .view(N, C, H, W)
        )


class ConvBNAct(nn.Sequential):
    def __init__(
            self,
            input_depth,
            output_depth,
            kernel,
            stride,
            pad,
            no_bias,
            act_func,
            bn_type,
            group=1,
            affine=False,
            *args,
            **kwargs
    ):
        super(ConvBNAct, self).__init__()

        assert act_func in ["relu", "hswish", 'swish', None], 'invalid act_func: %s' % act_func
        if isinstance(bn_type, (list, tuple)):
            assert len(bn_type) == 2
            assert bn_type[0] == "gn"
            gn_group = bn_type[1]
            bn_type = bn_type[0]
        assert bn_type in ["bn", "af", "gn", None]
        assert stride in [1, 2, 4]

        op = Conv2d(
                    input_depth,
                    output_depth,
                    kernel_size=kernel,
                    stride=stride,
                    padding=pad,
                    bias=not no_bias,
                    groups=group,
                    *args,
                    **kwargs
                )
        nn.init.kaiming_normal_(op.weight, mode="fan_out", nonlinearity="relu")
        if op.bias is not None:
            nn.init.constant_(op.bias, 0.0)
        self.add_module("conv", op)

        if bn_type == "bn":
            bn_op = BatchNorm2d(output_depth, affine=affine)
        elif bn_type == "gn":
            bn_op = nn.GroupNorm(num_groups=gn_group, num_channels=output_depth, affine=affine)
        if bn_type is not None:
            self.add_module("bn", bn_op)

        if act_func == "relu":
            self.add_module("relu", nn.ReLU(inplace=True))
        elif act_func == "hswish":
            self.add_module("act_func", Hswish(inplace=True))
        elif act_func == 'swish':
            self.add_module('act_func', Swish(inplace=True))


class ConvOpBNAct(nn.Sequential):
    def __init__(
            self, conv_op, output_depth,
            act_func,
            bn_type,
            affine
    ):
        super(ConvOpBNAct, self).__init__()

        assert act_func in ["relu", "hswish", 'swish', None]
        if isinstance(bn_type, (list, tuple)):
            assert len(bn_type) == 2
            assert bn_type[0] == "gn"
            gn_group = bn_type[1]
            bn_type = bn_type[0]
        assert bn_type in ["bn", "af", "gn", None]
        if hasattr(conv_op, 'weight') and conv_op.weight is not None:
            nn.init.kaiming_normal_(conv_op.weight, mode="fan_out", nonlinearity="relu")
        if hasattr(conv_op, 'bias') and conv_op.bias is not None:
            nn.init.constant_(conv_op.bias, 0.0)
        self.add_module("conv", conv_op)

        if bn_type == "bn":
            bn_op = BatchNorm2d(output_depth, affine=affine)
        elif bn_type == "gn":
            bn_op = nn.GroupNorm(num_groups=gn_group, num_channels=output_depth, affine=affine)
        if bn_type is not None:
            self.add_module("bn", bn_op)

        if act_func == "relu":
            self.add_module("relu", nn.ReLU(inplace=True))
        elif act_func == "hswish":
            self.add_module("act_func", Hswish(inplace=True))
        elif act_func == 'swish':
            self.add_module('act_func', Swish(inplace=True))


class SEModule(nn.Module):
    def __init__(self, C, reduce_base, reduction=0.25, inner_act=nn.ReLU, last_act=nn.Sigmoid):
        super(SEModule, self).__init__()
        self.reduction = reduction
        mid = get_divisible_by(int(reduce_base * self.reduction), 8, 1)
        conv1 = Conv2d(C, mid, 1, 1, 0)
        conv2 = Conv2d(mid, C, 1, 1, 0)

        self.op = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                conv1, inner_act(inplace=True),
                                conv2, last_act())

    def forward(self, x):
        return x * self.op(x)


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=None):
        super(Upsample, self).__init__()
        self.scale = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return interpolate(
            x, scale_factor=self.scale, mode=self.mode,
            align_corners=self.align_corners
        )


def _get_upsample_op(stride):
    assert (
            stride in [1, 2, 4]
            or stride in [-1, -2, -4]
            or (isinstance(stride, tuple) and all(x in [-1, -2, -4] for x in stride))
    )

    scales = stride
    ret = None
    if isinstance(stride, tuple) or stride < 0:
        scales = [-x for x in stride] if isinstance(stride, tuple) else -stride
        stride = 1
        ret = Upsample(scale_factor=scales, mode="nearest", align_corners=None)

    return ret, stride


class IRFBlock(nn.Module):
    def __init__(
            self,
            input_depth,
            output_depth,
            expansion,
            stride,
            bn_type="bn",
            kernel=3,
            width_divisor=1,
            shuffle_type=None,
            pw_group=1,
            se=False,
            cdw=False,
            full=False,
            dw_skip_bn=False,
            dw_skip_relu=False,
            act_func='relu',
            se_reduce_mid=True,
            se_inner_act=nn.ReLU,
            se_last_act=nn.Sigmoid,
            se_pos='last',
            affine=False,
    ):
        super(IRFBlock, self).__init__()

        assert kernel in [1, 3, 5, 7], kernel
        assert se_pos in ['mid', 'last'], 'se_pos [%s] should be in mid | last' % se_pos
        self.se_pos = se_pos
        self.use_res_connect = stride == 1 and input_depth == output_depth
        self.output_depth = output_depth

        mid_depth = int(input_depth * expansion)
        mid_depth = get_divisible_by(mid_depth, width_divisor, 1)

        # print(input_depth, mid_depth, output_depth)
        # pw
        self.pw = ConvBNAct(
            input_depth,
            mid_depth,
            kernel=1,
            stride=1,
            pad=0,
            no_bias=1,
            act_func=act_func,
            bn_type=bn_type,
            group=pw_group,
            affine=affine,
        )

        # negative stride to do upsampling
        self.upscale, stride = _get_upsample_op(stride)

        # dw
        if kernel == 1:
            self.dw = nn.Sequential()
        elif cdw:
            dw1 = ConvBNAct(
                mid_depth,
                mid_depth,
                kernel=kernel,
                stride=stride,
                pad=(kernel // 2),
                group=mid_depth,
                no_bias=1,
                act_func=act_func,
                bn_type=bn_type,
                affine=affine,
            )
            dw2 = ConvBNAct(
                mid_depth,
                mid_depth,
                kernel=kernel,
                stride=1,
                pad=(kernel // 2),
                group=mid_depth,
                no_bias=1,
                act_func=act_func if not dw_skip_relu else None,
                bn_type=bn_type if not dw_skip_bn else None,
                affine=affine,
            )
            self.dw = nn.Sequential(OrderedDict([("dw1", dw1), ("dw2", dw2)]))
        elif full:
            self.dw = ConvBNAct(
                mid_depth,
                mid_depth,
                kernel=kernel,
                stride=stride,
                pad=(kernel // 2),
                group=1,
                no_bias=1,
                act_func=act_func if not dw_skip_relu else None,
                bn_type=bn_type if not dw_skip_bn else None,
                affine=affine,
            )
        else:
            self.dw = ConvBNAct(
                mid_depth,
                mid_depth,
                kernel=kernel,
                stride=stride,
                pad=(kernel // 2),
                group=mid_depth,
                no_bias=1,
                act_func=act_func if not dw_skip_relu else None,
                bn_type=bn_type if not dw_skip_bn else None,
                affine=affine,
            )

        # pw-linear
        self.pwl = ConvBNAct(
            mid_depth,
            output_depth,
            kernel=1,
            stride=1,
            pad=0,
            no_bias=1,
            act_func=None,
            bn_type=bn_type,
            group=pw_group,
            affine=affine,
        )

        self.shuffle_type = shuffle_type
        if shuffle_type is not None:
            self.shuffle = ChannelShuffle(pw_group)

        if self.se_pos == 'mid': # MobileNetV3
            se_inc = mid_depth
        elif self.se_pos == 'last':
            se_inc = output_depth
        else:
            raise ValueError('invalid self.se_pos')

        se_reduce_base = se_inc if se_reduce_mid else input_depth
        self.se4 = SEModule(se_inc, se_reduce_base, inner_act=se_inner_act, last_act=se_last_act) if se else None

        self.output_depth = output_depth

    def forward(self, x):
        # in_ = x.shape[1]
        y = self.pw(x)
        if self.shuffle_type == "mid":
            y = self.shuffle(y)
        if self.upscale is not None:
            y = self.upscale(y)
        y = self.dw(y)
        # mid_ = y.shape[1]
        if self.se4 and self.se_pos == 'mid':
            y = self.se4(y)

        y = self.pwl(y)
        if self.use_res_connect:
            y += x

        if self.se4 and self.se_pos == 'last':
            y = self.se4(y)
        # out_ = y.shape[1]
        # print(in_, mid_, out_)
        return y




skip = lambda C_in, C_out, stride, **kwargs: Identity(
    C_in, C_out, stride
)
ir_c3_e1 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 1, stride, kernel=3, full=True, **kwargs
)
ir_k3_e1 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 1, stride, kernel=3, **kwargs
)
ir_k3_e3 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 3, stride, kernel=3, **kwargs
)
ir_k3_e6 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 6, stride, kernel=3, **kwargs
)
ir_k3_s4 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 4, stride, kernel=3, shuffle_type="mid", pw_group=4, **kwargs
)
ir_c5_e1 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 1, stride, kernel=5, full=True, **kwargs
)
ir_k5_e1 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 1, stride, kernel=5, **kwargs
)
ir_k5_e3 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 3, stride, kernel=5, **kwargs
)
ir_k5_e6 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 6, stride, kernel=5, **kwargs
)
ir_k5_s4 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 4, stride, kernel=5, shuffle_type="mid", pw_group=4, **kwargs
)
# layer search se
ir_k3_e1_se = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 1, stride, kernel=3, se=True, **kwargs
)
ir_k3_e3_se = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 3, stride, kernel=3, se=True, **kwargs
) 
ir_k3_e6_se = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 6, stride, kernel=3, se=True, **kwargs
)
ir_k3_s4_se = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in,
    C_out,
    4,
    stride,
    kernel=3,
    shuffle_type="mid",
    pw_group=4,
    se=True,
    **kwargs
)
ir_k5_e1_se = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 1, stride, kernel=5, se=True, **kwargs
)
ir_k5_e3_se = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 3, stride, kernel=5, se=True, **kwargs
)
ir_k5_e6_se = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 6, stride, kernel=5, se=True, **kwargs
)
ir_k5_s4_se = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in,
    C_out,
    4,
    stride,
    kernel=5,
    shuffle_type="mid",
    pw_group=4,
    se=True,
    **kwargs
)

ir_k3_s2 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 1, stride, kernel=3, shuffle_type="mid", pw_group=2, **kwargs
)
ir_k5_s2 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 1, stride, kernel=5, shuffle_type="mid", pw_group=2, **kwargs
)
ir_k3_s2_se = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in,
    C_out,
    1,
    stride,
    kernel=3,
    shuffle_type="mid",
    pw_group=2,
    se=True,
    **kwargs
)
ir_k5_s2_se = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in,
    C_out,
    1,
    stride,
    kernel=5,
    shuffle_type="mid",
    pw_group=2,
    se=True,
    **kwargs
)
ir_k33_e1 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 1, stride, kernel=3, cdw=True, **kwargs
)
ir_k33_e3 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 3, stride, kernel=3, cdw=True, **kwargs
)
ir_k33_e6 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 6, stride, kernel=3, cdw=True, **kwargs
)
ir_k7_e1 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 1, stride, kernel=7, **kwargs
)
ir_k7_e3 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 3, stride, kernel=7, **kwargs
)
ir_k7_e6 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 6, stride, kernel=7, **kwargs
)
ir_k7_sep_e1 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 1, stride, kernel=7, cdw=True, **kwargs
)
ir_k7_sep_e3 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 3, stride, kernel=7, cdw=True, **kwargs
)
ir_k7_sep_e6 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 6, stride, kernel=7, cdw=True, **kwargs
)


# inplanes, outplanes, stride=1, midplanes=None, norm_layer=nn.BatchNorm2d):



