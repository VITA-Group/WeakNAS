# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import copy
import random

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.ofa.utils.layers import ConvLayer, IdentityLayer, LinearLayer, MBConvLayer, ResidualBlock
from timm.models.ofa.imagenet_classification.networks import MobileNetV3
from timm.models.ofa.utils import make_divisible, val2list, MyNetwork
from .registry import register_model

__all__ = ['OFAMobileNetV3']


def _cfg(**kwargs):
    return {
        'num_classes': 1000,
        **kwargs
    }


default_cfgs = {
    'ofa_mbv3': _cfg(
	width_mult=1.2,
	ks_list=[5, 3, 3, 3,
			 5, 5, 5, 3,
			 7, 5, 7, 5,
			 3, 5, 5, 3,
		     7, 5, 3, 5],
	expand_ratio_list=[3, 3, 4, 3,
					   4, 6, 4, 4,
					   4, 6, 6, 6,
					   6, 6, 6, 4,
					   6, 6, 6, 3],
	depth_list=[4, 3, 3, 3, 4],
	),
}

@register_model
def ofa_mbv3(pretrained=False, **kwargs):
    model = OFAMobileNetV3(**default_cfgs['ofa_mbv3'], **kwargs)
    return model


class OFAMobileNetV3(MobileNetV3):

	def __init__(self, num_classes=1000, bn_param=(0.1, 1e-5), drop_rate=0.1, drop_path_rate=0.0, width_mult=1.0,
	             ks_list=3, expand_ratio_list=6, depth_list=4):

		self.width_mult = width_mult
		self.ks_list = val2list(ks_list, 1)
		self.expand_ratio_list = val2list(expand_ratio_list, 1)
		self.depth_list = val2list(depth_list, 1)
		self.num_classes = num_classes
		self.drop_path_rate = drop_path_rate
		# self.ks_list.sort()
		# self.expand_ratio_list.sort()
		# self.depth_list.sort()
		base_stage_width = [16, 16, 24, 40, 80, 112, 160, 960, 1280]

		final_expand_width = make_divisible(base_stage_width[-2] * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE)
		last_channel = make_divisible(base_stage_width[-1] * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE)

		stride_stages = [1, 2, 2, 2, 1, 2]
		act_stages = ['relu', 'relu', 'relu', 'h_swish', 'h_swish', 'h_swish']
		se_stages = [False, False, True, False, True, True]
		# n_block_list = [1] + [max(self.depth_list)] * 5
		n_block_list = [1] + self.depth_list
		width_list = []
		for base_width in base_stage_width[:-2]:
			width = make_divisible(base_width * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE)
			width_list.append(width)

		input_channel, first_block_dim = width_list[0], width_list[1]
		# first conv layer
		first_conv = ConvLayer(3, input_channel, kernel_size=3, stride=2, act_func='h_swish')
		print(input_channel)
		first_block_conv = MBConvLayer(
			in_channels=input_channel, out_channels=first_block_dim, kernel_size=3, stride=stride_stages[0],
			expand_ratio=1, act_func=act_stages[0], use_se=se_stages[0],
		)
		first_block = ResidualBlock(
			first_block_conv,
			IdentityLayer(first_block_dim, first_block_dim) if input_channel == first_block_dim else None,
			drop_path_rate=self.drop_path_rate,
		)
		print(first_block_dim)
		# inverted residual blocks
		self.block_group_info = []
		blocks = [first_block]
		_block_index = 1
		feature_dim = first_block_dim
		for j, (width, n_block, s, act_func, use_se) in enumerate(zip(width_list[2:], n_block_list[1:],
		                                               stride_stages[1:], act_stages[1:], se_stages[1:])):
			self.block_group_info.append([_block_index + i for i in range(n_block)])
			output_channel = width
			for i in range(n_block):
				if i == 0:
					stride = s
				else:
					stride = 1
				# print(f'Index {_block_index}: {j} {i} {4*j+i}')
				mobile_inverted_conv = MBConvLayer(
					in_channels=feature_dim, out_channels=output_channel,
					kernel_size=self.ks_list[4*j+i], expand_ratio=self.expand_ratio_list[4*j+i],
					stride=stride, act_func=act_func, use_se=use_se,
				)
				print(output_channel)
				_block_index += 1
				if stride == 1 and feature_dim == output_channel:
					shortcut = IdentityLayer(feature_dim, feature_dim)
				else:
					shortcut = None
				blocks.append(ResidualBlock(mobile_inverted_conv, shortcut, drop_path_rate=self.drop_path_rate))
				feature_dim = output_channel
		# final expand layer, feature mix layer & classifier
		final_expand_layer = ConvLayer(feature_dim, final_expand_width, kernel_size=1, act_func='h_swish')
		print(feature_dim, final_expand_width)
		feature_mix_layer = ConvLayer(
			final_expand_width, last_channel, kernel_size=1, bias=False, use_bn=False, act_func='h_swish',
		)
		print(final_expand_width, last_channel)
		classifier = LinearLayer(last_channel, num_classes, dropout_rate=drop_rate)

		super(OFAMobileNetV3, self).__init__(first_conv, blocks, final_expand_layer, feature_mix_layer, classifier)

		# set bn param
		self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

		# runtime_depth
		self.runtime_depth = [len(block_idx) for block_idx in self.block_group_info]

	""" MyNetwork required methods """

	@staticmethod
	def name():
		return 'OFAMobileNetV3'

	def forward(self, x):
		# first conv
		x = self.first_conv(x)
		# first block
		x = self.blocks[0](x)
		# blocks
		for stage_id, block_idx in enumerate(self.block_group_info):
			for idx in block_idx:
				x = self.blocks[idx](x)
		x = self.final_expand_layer(x)
		x = x.mean(3, keepdim=True).mean(2, keepdim=True)  # global average pooling
		x = self.feature_mix_layer(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

	@property
	def module_str(self):
		_str = self.first_conv.module_str + '\n'
		_str += self.blocks[0].module_str + '\n'

		for stage_id, block_idx in enumerate(self.block_group_info):
			depth = self.runtime_depth[stage_id]
			active_idx = block_idx[:depth]
			for idx in active_idx:
				_str += self.blocks[idx].module_str + '\n'

		_str += self.final_expand_layer.module_str + '\n'
		_str += self.feature_mix_layer.module_str + '\n'
		_str += self.classifier.module_str + '\n'
		return _str

	@property
	def config(self):
		return {
			'name': OFAMobileNetV3.__name__,
			'bn': self.get_bn_param(),
			'first_conv': self.first_conv.config,
			'blocks': [
				block.config for block in self.blocks
			],
			'final_expand_layer': self.final_expand_layer.config,
			'feature_mix_layer': self.feature_mix_layer.config,
			'classifier': self.classifier.config,
		}

	@staticmethod
	def build_from_config(config):
		raise ValueError('do not support this function')

	@property
	def grouped_block_index(self):
		return self.block_group_info

	def load_state_dict(self, state_dict, **kwargs):
		model_dict = self.state_dict()
		for key in state_dict:
			if '.mobile_inverted_conv.' in key:
				new_key = key.replace('.mobile_inverted_conv.', '.conv.')
			else:
				new_key = key
			if new_key in model_dict:
				pass
			elif '.bn.bn.' in new_key:
				new_key = new_key.replace('.bn.bn.', '.bn.')
			elif '.conv.conv.weight' in new_key:
				new_key = new_key.replace('.conv.conv.weight', '.conv.weight')
			elif '.linear.linear.' in new_key:
				new_key = new_key.replace('.linear.linear.', '.linear.')
			##############################################################################
			elif '.linear.' in new_key:
				new_key = new_key.replace('.linear.', '.linear.linear.')
			elif 'bn.' in new_key:
				new_key = new_key.replace('bn.', 'bn.bn.')
			elif 'conv.weight' in new_key:
				new_key = new_key.replace('conv.weight', 'conv.conv.weight')
			else:
				raise ValueError(new_key)
			assert new_key in model_dict, '%s' % new_key
			model_dict[new_key] = state_dict[key]
		super(OFAMobileNetV3, self).load_state_dict(model_dict)

	""" set, sample and get active sub-networks """

	def set_max_net(self):
		self.set_active_subnet(ks=max(self.ks_list), e=max(self.expand_ratio_list), d=max(self.depth_list))


	def set_constraint(self, include_list, constraint_type='depth'):
		if constraint_type == 'depth':
			self.__dict__['_depth_include_list'] = include_list.copy()
		elif constraint_type == 'expand_ratio':
			self.__dict__['_expand_include_list'] = include_list.copy()
		elif constraint_type == 'kernel_size':
			self.__dict__['_ks_include_list'] = include_list.copy()
		else:
			raise NotImplementedError

	def clear_constraint(self):
		self.__dict__['_depth_include_list'] = None
		self.__dict__['_expand_include_list'] = None
		self.__dict__['_ks_include_list'] = None

	def sample_active_subnet(self):
		ks_candidates = self.ks_list if self.__dict__.get('_ks_include_list', None) is None \
			else self.__dict__['_ks_include_list']
		expand_candidates = self.expand_ratio_list if self.__dict__.get('_expand_include_list', None) is None \
			else self.__dict__['_expand_include_list']
		depth_candidates = self.depth_list if self.__dict__.get('_depth_include_list', None) is None else \
			self.__dict__['_depth_include_list']

		# sample kernel size
		ks_setting = []
		if not isinstance(ks_candidates[0], list):
			ks_candidates = [ks_candidates for _ in range(len(self.blocks) - 1)]
		for k_set in ks_candidates:
			k = random.choice(k_set)
			ks_setting.append(k)

		# sample expand ratio
		expand_setting = []
		if not isinstance(expand_candidates[0], list):
			expand_candidates = [expand_candidates for _ in range(len(self.blocks) - 1)]
		for e_set in expand_candidates:
			e = random.choice(e_set)
			expand_setting.append(e)

		# sample depth
		depth_setting = []
		if not isinstance(depth_candidates[0], list):
			depth_candidates = [depth_candidates for _ in range(len(self.block_group_info))]
		for d_set in depth_candidates:
			d = random.choice(d_set)
			depth_setting.append(d)

		self.set_active_subnet(ks_setting, expand_setting, depth_setting)

		return {
			'ks': ks_setting,
			'e': expand_setting,
			'd': depth_setting,
		}

	""" Width Related Methods """

	def re_organize_middle_weights(self, expand_ratio_stage=0):
		for block in self.blocks[1:]:
			block.conv.re_organize_middle_weights(expand_ratio_stage)


if __name__ == "__main__":
	import torch
	net_setting = {"ks": [5, 3, 3, 3,
						  5, 5, 5, 3,
						  7, 5, 7, 5,
						  3, 5, 5, 3,
						  7, 5, 3, 5],
				   "e":  [3, 3, 4, 3,
						  4, 6, 4, 4,
						  4, 6, 6, 6,
						  6, 6, 6, 4,
						  6, 6, 6, 3],
				   "d":  [4, 3, 3, 3, 4]}
	net = OFAMobileNetV3(drop_rate=0, width_mult=1.2, ks_list=net_setting["ks"], expand_ratio_list=net_setting["e"], depth_list=net_setting["d"],)
	# net.set_active_subnet(**net_setting)
	x = torch.randn(10, 3, 236, 236)
	output = net(x)
	print(output.shape)
