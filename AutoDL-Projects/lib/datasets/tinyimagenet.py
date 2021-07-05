import os
import argparse
import copy
import torchvision
import torch
import torch.utils.data as data
from torchvision.transforms import transforms
import numpy as np

def identity(x):
    return x

class ImageFolderTensor(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, mean=0, std=0):
        self.img, self.label = torch.load(root)
        # self.img = F.normalize(self.img, mean, std, True)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        img = self.img[index,:,:,:]
        label = self.label[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return self.img.shape[0]

class ImageFolderPIL(data.Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        # self.img, self.label = torch.load(root)
        self.img, self.label = copy.deepcopy(data)
        # print(f'loaded from {root}')
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img = self.img[index]
        label = self.label[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.img)


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def identity(x):
    return x


def folder2lmdb(path, outpath, mean=0, std=1):
    directory = os.path.expanduser(path)
    print("Loading dataset from %s" % directory)
    dataset = torchvision.datasets.ImageFolder(directory, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]))
    # dataset = torchvision.datasets.ImageFolder(directory, transform=None)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, collate_fn=identity)
    # data_loader = DataLoader(dataset, num_workers=16, collate_fn=lambda x: x)outpath):
    #     #     os.makedirs(outpath)
    # if not os.path.exists(

    image_list = []
    label_list = []
    # for idx, data in enumerate(data_loader):
    #     image, label = data[0]
    #     image_list.append(image)
    #     label_list.append(label)
    # torch.save((image_list, label_list), outpath)
    for idx, data in enumerate(data_loader):
        image, label = data[0]
        image_list.append(image)
        # print(label)
        label_list.append(torch.from_numpy(np.array(label, dtype=np.int)))
        # label_list.append(torch.LongTensor(label))
        # label_list.append(label)
    # input = torch.cat(image_list)
    # label = torch.cat(label_list)
    input = torch.stack(image_list, axis=0)
    label = torch.stack(label_list, axis=0)
    # torch.mean(input, dim=(0)).permute(2, 1, 0).shape
    print(torch.mean(input, dim=(0,2,3)), torch.std(input, dim=(0,2,3)))
    torch.save((input, label), outpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default='D:/Dropbox/MSR/DATASET/nasbench201/TinyImageNet', help="Path to original image dataset folder")
    parser.add_argument("-o", "--outpath", default='D:/Dropbox/MSR/DATASET/nasbench201/TinyImageNet', help="Path to output LMDB file")
    args = parser.parse_args()

    # train_path = os.path.join(args.dataset, 'train')
    # val_path = os.path.join(args.dataset, 'val')
    test_path = os.path.join(args.dataset, 'test')

    # train_out = os.path.join(args.outpath, 'train.tensor')
    # val_out = os.path.join(args.outpath, 'val.tensor')
    test_out = os.path.join(args.outpath, 'test.tensor')

    # train_out = os.path.join(args.outpath, 'train.pil')
    # val_out = os.path.join(args.outpath, 'val.pil')

    # mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    # mean, std = [0.4802, 0.4481, 0.3975], [0.2764, 0.2689, 0.2816]
    # mean, std = [0.4824, 0.4495, 0.3981], [0.2765, 0.2691, 0.2825]

    # folder2lmdb(train_path, train_out)
    # folder2lmdb(val_path, val_out)
    folder2lmdb(test_path, test_out)