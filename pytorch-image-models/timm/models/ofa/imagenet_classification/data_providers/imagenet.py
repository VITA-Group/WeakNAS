# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import warnings
import os
import math
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from torchvision.datasets import DatasetFolder
import json
from .base_provider import DataProvider
from ofa.utils.my_dataloader import MyRandomResizedCrop, MyDistributedSampler

__all__ = ['ImagenetDataProvider']

def pil_loader(path):
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		img = Image.open(f)
		return img.convert('RGB')


class ImageNetFolder(DatasetFolder):
	def __init__(self, root, split='train', suffix='', transform=None, target_transform=None,
				 loader=pil_loader, repeat_num=1, shuffle=False):
		self.root = root
		self.transform = transform
		self.target_transform = target_transform
		self.loader = loader
		metas = []
		for sp in split.split(','):
			assert sp in ['train', 'val', 'test'], 'split should be train | val | test'
			meta_file = os.path.join(root, sp + suffix + '_meta.json')
			assert os.path.exists(meta_file), \
				'meta file %s under root %s not found' % (os.path.basename(meta_file), root)

			with open(meta_file, 'r') as f:
				meta = json.load(f)
			metas.append(meta)
		self.classes = metas[0]['classes']
		self.class_to_idx = metas[0]['class_to_idx']
		self.samples = []
		for meta in metas:
			self.samples += meta['samples']
		self.num_sample = len(self.samples)
		self.repeat_num = repeat_num
		self.allsamples = self.samples
		self.shuffle = shuffle
		self.sample_indexes = []
		self.get_sample_indexes()
		print('Total %d images in split: %s' % (len(self.samples), split))

	def get_sample_indexes(self):
		self.sample_indexes = []
		for idx in range(self.repeat_num):
			curr_indexes = np.random.permutation(self.num_sample) if self.shuffle else range(self.num_sample)
			self.sample_indexes += list(curr_indexes)

	def select_class(self, cls):
		new_samples = [sample for sample in self.allsamples if sample[-1] in cls]
		self.samples = new_samples
		self.num_sample = len(self.samples)
		self.get_sample_indexes()

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index



		Returns:
			tuple: (sample, target) where target is class_index of the target class.
		"""
		# index = index % self.num_sample
		index = self.sample_indexes[index]
		img_path, target = self.samples[index]
		sample = self.loader(os.path.join(self.root, img_path))
		if self.transform is not None:
			sample = self.transform(sample)
		if self.target_transform is not None:
			target = self.target_transform(target)

		return sample, target

	def __len__(self):
		return len(self.samples) * self.repeat_num


class ImagenetDataProvider(DataProvider):
	DEFAULT_PATH = '/dataset/imagenet'
	DEFAULT_SUBSAMPLE = 1

	def __init__(self, save_path=None, train_batch_size=256, test_batch_size=512, valid_size=None, n_worker=32,
	             resize_scale=0.08, distort_color=None, image_size=224,
	             num_replicas=None, rank=None, subsample=1):

		warnings.filterwarnings('ignore')
		self._save_path = save_path

		self.image_size = image_size  # int or list of int
		self.distort_color = 'None' if distort_color is None else distort_color
		self.resize_scale = resize_scale
		subsample = self.DEFAULT_SUBSAMPLE
		self.subsample = subsample

		self._valid_transform_dict = {}
		if not isinstance(self.image_size, int):
			from ofa.utils.my_dataloader import MyDataLoader
			assert isinstance(self.image_size, list)
			self.image_size.sort()  # e.g., 160 -> 224
			MyRandomResizedCrop.IMAGE_SIZE_LIST = self.image_size.copy()
			MyRandomResizedCrop.ACTIVE_SIZE = max(self.image_size)

			for img_size in self.image_size:
				self._valid_transform_dict[img_size] = self.build_valid_transform(img_size)
			self.active_img_size = max(self.image_size)  # active resolution for test
			valid_transforms = self._valid_transform_dict[self.active_img_size]
			train_loader_class = MyDataLoader  # randomly sample image size for each batch of training image
		else:
			self.active_img_size = self.image_size
			valid_transforms = self.build_valid_transform()
			train_loader_class = torch.utils.data.DataLoader

		train_dataset = self.train_dataset(self.build_train_transform())

		if valid_size is not None:
			if not isinstance(valid_size, int):
				assert isinstance(valid_size, float) and 0 < valid_size < 1
				valid_size = int(len(train_dataset) * valid_size)

			valid_dataset = self.train_dataset(valid_transforms)
			train_indexes, valid_indexes = self.random_sample_valid_set(len(train_dataset), valid_size)

			if num_replicas is not None:
				train_sampler = MyDistributedSampler(train_dataset, num_replicas, rank, True, np.array(train_indexes))
				valid_sampler = MyDistributedSampler(valid_dataset, num_replicas, rank, True, np.array(valid_indexes))
			else:
				train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indexes)
				valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indexes)
			if subsample == 1:
				self.train = train_loader_class(
					train_dataset, batch_size=train_batch_size, sampler=train_sampler,
					num_workers=n_worker, pin_memory=True,
				)
			else:
				train_dataset = torch.utils.data.Subset(train_dataset, indices=[int(s) for s in np.linspace(start=0, stop=len(train_dataset), num=int(len(train_dataset)*subsample), endpoint=False)])
				self.train = train_loader_class(
					train_dataset, batch_size=train_batch_size, sampler=train_sampler,
					num_workers=n_worker, pin_memory=True,
				)

			self.valid = torch.utils.data.DataLoader(
				valid_dataset, batch_size=test_batch_size, sampler=valid_sampler,
				num_workers=n_worker, pin_memory=True,
			)
		else:
			if num_replicas is not None:
				train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas, rank)
				self.train = train_loader_class(
					train_dataset, batch_size=train_batch_size, sampler=train_sampler,
					num_workers=n_worker, pin_memory=True
				)
			else:
				if subsample == 1:
					self.train = train_loader_class(
						train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=n_worker, pin_memory=True,
					)
				else:
					train_dataset = torch.utils.data.Subset(train_dataset, indices=[int(s) for s in np.linspace(start=0, stop=len(train_dataset), num=int(len(train_dataset) * subsample), endpoint=False)])
					self.train = train_loader_class(
						train_dataset, batch_size=train_batch_size, shuffle=False, num_workers=n_worker, pin_memory=True,
					)

			self.valid = None

		test_dataset = self.test_dataset(valid_transforms)
		if num_replicas is not None:
			test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas, rank)
			self.test = torch.utils.data.DataLoader(
				test_dataset, batch_size=test_batch_size, sampler=test_sampler, num_workers=n_worker, pin_memory=True,
			)
		else:
			self.test = torch.utils.data.DataLoader(
				test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=n_worker, pin_memory=True,
			)

		if self.valid is None:
			self.valid = self.test

	@staticmethod
	def name():
		return 'imagenet'

	@property
	def data_shape(self):
		return 3, self.active_img_size, self.active_img_size  # C, H, W

	@property
	def n_classes(self):
		return 1000

	@property
	def save_path(self):
		if self._save_path is None:
			self._save_path = self.DEFAULT_PATH
			if not os.path.exists(self._save_path):
				self._save_path = os.path.expanduser('~/dataset/imagenet')
		return self._save_path

	@property
	def data_url(self):
		raise ValueError('unable to download %s' % self.name())

	def train_dataset(self, _transforms):
		# return datasets.ImageFolder(self.train_path, _transforms)
		return ImageNetFolder(self.save_path, split='train', transform=_transforms)

	def test_dataset(self, _transforms):
		# return datasets.ImageFolder(self.valid_path, _transforms)
		return ImageNetFolder(self.save_path, split='test', transform=_transforms)

	@property
	def train_path(self):
		return os.path.join(self.save_path, 'train')

	@property
	def valid_path(self):
		return os.path.join(self.save_path, 'val')

	@property
	def normalize(self):
		return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	def build_train_transform(self, image_size=None, print_log=True):
		if image_size is None:
			image_size = self.image_size
		if print_log:
			print('Color jitter: %s, resize_scale: %s, img_size: %s' %
			      (self.distort_color, self.resize_scale, image_size))

		if isinstance(image_size, list):
			resize_transform_class = MyRandomResizedCrop
			print('Use MyRandomResizedCrop: %s, \t %s' % MyRandomResizedCrop.get_candidate_image_size(),
			      'sync=%s, continuous=%s' % (MyRandomResizedCrop.SYNC_DISTRIBUTED, MyRandomResizedCrop.CONTINUOUS))
		else:
			resize_transform_class = transforms.RandomResizedCrop
		if self.subsample == 1:
			# random_resize_crop -> random_horizontal_flip
			train_transforms = [
				resize_transform_class(image_size, scale=(self.resize_scale, 1.0)),
				transforms.RandomHorizontalFlip(),
			]

			# color augmentation (optional)
			color_transform = None
			if self.distort_color == 'torch':
				color_transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
			elif self.distort_color == 'tf':
				color_transform = transforms.ColorJitter(brightness=32. / 255., saturation=0.5)
			if color_transform is not None:
				train_transforms.append(color_transform)
		else:
			train_transforms = [transforms.Resize(int(math.ceil(image_size / 0.875))),
			                    transforms.CenterCrop(image_size),]

		train_transforms += [
			transforms.ToTensor(),
			self.normalize,
		]

		train_transforms = transforms.Compose(train_transforms)
		return train_transforms

	def build_valid_transform(self, image_size=None):
		if image_size is None:
			image_size = self.active_img_size
		return transforms.Compose([
			transforms.Resize(int(math.ceil(image_size / 0.875))),
			transforms.CenterCrop(image_size),
			transforms.ToTensor(),
			self.normalize,
		])

	def assign_active_img_size(self, new_img_size):
		self.active_img_size = new_img_size
		if self.active_img_size not in self._valid_transform_dict:
			self._valid_transform_dict[self.active_img_size] = self.build_valid_transform()
		# change the transform of the valid and test set
		self.valid.dataset.transform = self._valid_transform_dict[self.active_img_size]
		self.test.dataset.transform = self._valid_transform_dict[self.active_img_size]

	def build_sub_train_loader(self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None):
		# used for resetting BN running statistics
		if self.__dict__.get('sub_train_%d' % self.active_img_size, None) is None:
			if num_worker is None:
				num_worker = self.train.num_workers

			n_samples = len(self.train.dataset)
			g = torch.Generator()
			g.manual_seed(DataProvider.SUB_SEED)
			rand_indexes = torch.randperm(n_samples, generator=g).tolist()

			new_train_dataset = self.train_dataset(
				self.build_train_transform(image_size=self.active_img_size, print_log=False))
			chosen_indexes = rand_indexes[:n_images]
			if num_replicas is not None:
				sub_sampler = MyDistributedSampler(new_train_dataset, num_replicas, rank, True, np.array(chosen_indexes))
			else:
				sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(chosen_indexes)
			sub_data_loader = torch.utils.data.DataLoader(
				new_train_dataset, batch_size=batch_size, sampler=sub_sampler,
				num_workers=num_worker, pin_memory=True,
			)
			self.__dict__['sub_train_%d' % self.active_img_size] = []
			for images, labels in sub_data_loader:
				self.__dict__['sub_train_%d' % self.active_img_size].append((images, labels))
		return self.__dict__['sub_train_%d' % self.active_img_size]
