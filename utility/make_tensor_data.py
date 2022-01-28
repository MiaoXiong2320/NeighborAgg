import os
import os.path as osp
import glob

import pickle
import numpy as np
import imageio
from PIL import Image

import torch
#import torch.utils.data as data
from torchvision import transforms
from torchvision import datasets

from torch.utils.data import Subset

import pickle
import cv2

import numpy as np
import numpy.random as npr

import pdb
import torchvision

dataset = 'svhn'
data_dir = ".."
sav_folder = "./data/Tensor"+dataset.upper()

if dataset.lower() == "mnist":
    aug_transform = [transforms.ToTensor(),
                     transforms.Normalize([0.1307,], [0.3081,])]
    aug_transform = transforms.Compose(aug_transform)

    data_dir = "../classifier/code/confidnet/data/mnist-data/"
    train_dataset_all = datasets.MNIST(root=data_dir,
                                        train=True,
                                        download=False,
                                        transform=aug_transform)

    test_dataset = datasets.MNIST(root=data_dir,
                                        train=False,
                                        download=False,
                                        transform=aug_transform)

    resume_folder = "../classifier/mnist_pretrained/baseline/"
    train_idx = np.load(resume_folder + "train_idx.npy")
    val_idx = np.load(resume_folder + "val_idx.npy")

    train_dataset = Subset(train_dataset_all, train_idx)
    val_dataset = Subset(train_dataset_all, val_idx)

elif dataset.lower() == 'svhn':
    data_path = os.path.join(data_dir, dataset.lower())
    transform = transforms.Compose([transforms.ToTensor()])
    ds = getattr(torchvision.datasets, dataset.upper())
    train_set = ds(root=data_path, split='train', download=True, transform=transform)
    train_dataset, val_dataset = torch.utils.data.random_split(train_set, [58605, 14652])
    test_dataset = ds(root=data_path, split='test', download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=100,
    shuffle=False,
    pin_memory=True,
    num_workers=1,
)
val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=100,
    shuffle=False,
    pin_memory=True,
    num_workers=1,
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=100,
    shuffle=False,
    pin_memory=True,
    num_workers=1,
)


train_xs, train_ys = [], []
for itr, (xs, ys) in enumerate(train_loader):
    train_xs.append(xs)
    train_ys.append(ys)

train_xs = torch.cat(train_xs)
train_ys = torch.cat(train_ys)
np.save(osp.join(sav_folder, "train_xs.npy"), train_xs.detach().numpy())
np.save(osp.join(sav_folder, "train_ys.npy"), train_ys.detach().numpy())

val_xs, val_ys = [], []
for itr, (xs, ys) in enumerate(val_loader):
    val_xs.append(xs)
    val_ys.append(ys)

val_xs = torch.cat(val_xs)
val_ys = torch.cat(val_ys)

np.save(osp.join(sav_folder, "val_xs.npy"), val_xs.detach().numpy())
np.save(osp.join(sav_folder, "val_ys.npy"), val_ys.detach().numpy())

##### TEST ######
test_xs, test_ys = [], []
for itr, (xs, ys) in enumerate(test_loader):
    test_xs.append(xs)
    test_ys.append(ys)

test_xs = torch.cat(test_xs)
test_ys = torch.cat(test_ys)

np.save(osp.join(sav_folder, "test_xs.npy"), test_xs.detach().numpy())
np.save(osp.join(sav_folder, "test_ys.npy"), test_ys.detach().numpy())

print("DONE.")
