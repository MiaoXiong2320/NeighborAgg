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

from torch.utils.data import Subset

import pickle

import pdb

import torchvision

def make_sup_data_loaders(
        dataset,
        data_path, 
        batch_size, 
        num_workers, 
        data_size=32,
        shuffle_train=True,
        ):
    """
    data_path: is the direcotry where all dataset file locate; all dataset file is in lower case
    data_size: is prepared for reducing data size 
    """
    data_path = os.path.join(data_path, dataset.lower())
    aug_transform = [transforms.ToTensor(),
                 transforms.Normalize([0.1307,], [0.3081,])]
    transform = transforms.Compose(aug_transform)
    

    if dataset.lower() in ["fashionmnist"]:
        ds = torchvision.datasets.FashionMNIST
        train_set = ds(root=data_path, train=True, download=True, transform=transform)
        test_set = ds(root=data_path, train=False, download=True, transform=transform)
    elif dataset.lower() in ["cifar10", "mnist"]:
        ds = getattr(torchvision.datasets, dataset.upper())
        train_set = ds(root=data_path, train=True, download=True, transform=transform)
        test_set = ds(root=data_path, train=False, download=True, transform=transform)
    elif dataset.lower() == "celeba":
        ds = getattr(torchvision.datasets, "CelebA")
        transform = transforms.Compose([transforms.Resize((data_size, data_size)),
                                        transforms.ToTensor()
                                        ])
        train_set = ds(root=data_path, split='train', download=True, transform=transform)
        test_set = ds(root=data_path, split='test', download=True, transform=transform) 
    elif dataset.lower() == "svhn":
        ds = getattr(torchvision.datasets, dataset.upper())
        train_set = ds(root=data_path, split='train', download=True, transform=transform)
        test_set = ds(root=data_path, split='test', download=True, transform=transform) 
    else:
        raise NotImplementedError      

    print("{} Prepared.".format(dataset))
    print('Number of train samples: ', len(train_set))
    print('Number of test samples: ', len(test_set))

    train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True and shuffle_train,
                num_workers=num_workers,
                pin_memory=True
            )
    test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )

    return train_loader, test_loader




class TensorData(torch.utils.data.Dataset):
    def __init__(self, feature_dir, data_dir, split="val"):

        train_xs = np.load(osp.join(data_dir, "train_xs.npy"))
        train_ys = np.load(osp.join(data_dir, "train_ys.npy"))

        if split == "train":
            val_xs = np.load(osp.join(data_dir, "train_xs.npy"))
            val_ys = np.load(osp.join(data_dir, "train_ys.npy"))
            self.val2neighbors_idx = np.load(osp.join(feature_dir, "train2train_idx_allclass.npy")) # index dict
            self.val2neighbors_dist = np.load(osp.join(feature_dir, "train2train_dist_allclass.npy"))

        elif split == "val":
            val_xs = np.load(osp.join(data_dir, "val_xs.npy"))
            val_ys = np.load(osp.join(data_dir, "val_ys.npy"))
            self.val2neighbors_idx = np.load(osp.join(feature_dir, "val2train_idx_allclass.npy")) # index dict
            self.val2neighbors_dist = np.load(osp.join(feature_dir, "val2train_dist_allclass.npy"))
        elif split == "test":
            val_xs = np.load(osp.join(data_dir, "test_xs.npy"))
            val_ys = np.load(osp.join(data_dir, "test_ys.npy"))   
            self.val2neighbors_idx = np.load(osp.join(feature_dir, "test2train_idx_allclass.npy"))         
            self.val2neighbors_dist = np.load(osp.join(feature_dir, "test2train_dist_allclass.npy"))
        #device = "cuda:0"
        self.train_xs = torch.from_numpy(train_xs)#.to(device)
        self.train_ys = torch.from_numpy(train_ys)#.to(device)
        self.val_xs = torch.from_numpy(val_xs)#.to(device)
        self.val_ys = torch.from_numpy(val_ys)#.to(device)


    def __getitem__(self, index):
        val_datum = self.val_xs[index] # [1, 28, 28]
        val_label = self.val_ys[index]

        # for neighbors
        dists = self.val2neighbors_dist[index]
        neighbor_indices = self.val2neighbors_idx[index] #[self.val_idx[index]]
        neighbor_data = self.train_xs[neighbor_indices]
        # neighbor_label = self.train_ys[neighbor_indices]

        return val_datum, val_label, neighbor_data, dists, neighbor_indices

    def __len__(self):
        return self.val_ys.size(0)

    def show_neighbors(self, save_dir, index=None):
        if index == None:
            inds = np.random.randint(self.val_xs.shape[0], size=(100, 1))
        else:
            inds = [index]
        for index in inds:
            val_datum = self.val_xs[index] # [1, 28, 28]
            val_label = self.val_ys[index]
            # for neighbors
            dists = self.val2neighbors_dist[index]
            neighbor_indices = self.val2neighbors_idx[index] #[self.val_idx[index]]
            neighbor_data = self.train_xs[neighbor_indices]
            neighbor_label = self.train_ys[neighbor_indices]

            # 如果image是一组照片，会自动调用 make_grid 函数
            os.makedirs("output/toy", exist_ok=True)
            import heapq
            maxK = heapq.nsmallest(10, range(len(dists.flatten())), dists.flatten().take)
            # maxK = np.argmax(dists)
            images = neighbor_data[maxK]
            nn_images = torch.cat((val_datum, images))
            # images = torchvision.utils.make_grid(sample_array, nrow=11, normalize=False)
            # torchvision.utils.save_image(images, "output/toy/0.jpg")            

            
            data_list = []
            for cls in range(10):
                if cls == val_label:
                    node = val_datum
                else:
                    node = torch.ones_like(val_datum)
                sample_array = torch.cat((node, neighbor_data[cls*10:(cls+1)*10]))
                data_list.append(sample_array)
            data_list = torch.cat(data_list, dim=0)
            data_list = torch.cat((data_list, nn_images), dim=0)
            images = torchvision.utils.make_grid(data_list, nrow=11, normalize=False)
            # save_dir = "output/toy/cifar10/resnet18/e20/"
            os.makedirs(save_dir, exist_ok=True)
            torchvision.utils.save_image(images, os.path.join(save_dir, "{}.jpg".format(index)) )



class TrustMNIST7Digits(torch.utils.data.Dataset):
    def __init__(self, train=True, classwise=True):

        #data_dir = "./data/TensorMNIST/"
        data_dir = "../classifier/code/simple_cls/data/TensorMNIST3Digits"
        train_xs = np.load(osp.join(data_dir, "train_xs.npy"))
        train_ys = np.load(osp.join(data_dir, "train_ys.npy"))

        val_xs = np.load(osp.join(data_dir, "ood_val_xs.npy"))
        val_ys = np.load(osp.join(data_dir, "ood_val_ys.npy"))

        #device = "cuda:0"
        self.train_xs = torch.from_numpy(train_xs)#.to(device)
        self.train_ys = torch.from_numpy(train_ys)#.to(device)
        self.val_xs = torch.from_numpy(val_xs)#.to(device)
        self.val_ys = torch.from_numpy(val_ys)#.to(device)
                    
        # self.val2neighbors = np.load("../classifier/code/simple_cls/output/mnist3digits_smallerconvnet/feats/smallerconvnet_val2train_ood.npy")   
        # self.val2neighbors_dist = np.load("../classifier/code/simple_cls/output/mnist3digits_smallerconvnet/feats/smallerconvnet_val2train_dist_ood.npy")  

        self.val2neighbors = np.load("../classifier/code/simple_cls/output/mnist3digits_smallerconvnet/feats/smallerconvnet_val2train_ood_allclass.npy")   
        self.val2neighbors_dist = np.load("../classifier/code/simple_cls/output/mnist3digits_smallerconvnet/feats/smallerconvnet_val2train_dist_ood_allclass.npy")         
          
     

    def __getitem__(self, index):
        val_datum = self.val_xs[index] # [1, 28, 28]
        val_label = self.val_ys[index]

        neighbor_indices = self.val2neighbors[index] #[self.val_idx[index]]
        distances = self.val2neighbors_dist[index]
        neighbor_data = self.train_xs[neighbor_indices]
        neighbor_label = self.train_ys[neighbor_indices]
        return val_datum, val_label, neighbor_data, distances, neighbor_label


    def __len__(self):
        return self.val_ys.size(0)

class TrustMNIST3Digits(torch.utils.data.Dataset):
    def __init__(self, train=True, classwise=True):

        #data_dir = "./data/TensorMNIST/"
        data_dir = "../classifier/code/simple_cls/data/TensorMNIST3Digits"
        train_xs = np.load(osp.join(data_dir, "train_xs.npy"))
        train_ys = np.load(osp.join(data_dir, "train_ys.npy"))

        val_xs = np.load(osp.join(data_dir, "val_xs.npy"))
        val_ys = np.load(osp.join(data_dir, "val_ys.npy"))

        #device = "cuda:0"
        self.train_xs = torch.from_numpy(train_xs)#.to(device)
        self.train_ys = torch.from_numpy(train_ys)#.to(device)
        self.val_xs = torch.from_numpy(val_xs)#.to(device)
        self.val_ys = torch.from_numpy(val_ys)#.to(device)

        # load the dict
        #self.val2neighbors = np.load("../classifier/mnist_pretrained/baseline/feats/smallconvnet_val2train_classwise.npy")
        if classwise:
            self.val2neighbors = np.load("../classifier/code/simple_cls/output/mnist3digits_smallerconvnet/feats/smallerconvnet_val2train_classwise.npy")
        else:
            self.val2neighbors = np.load("../classifier/code/simple_cls/output/mnist3digits_smallerconvnet/feats/smallerconvnet_val2train.npy")   
            self.val2neighbors_dist = np.load("../classifier/code/simple_cls/output/mnist3digits_smallerconvnet/feats/smallerconvnet_val2train_dist.npy")            

    def __getitem__(self, index):
        val_datum = self.val_xs[index] # [1, 28, 28]
        val_label = self.val_ys[index]

        neighbor_indices = self.val2neighbors[index] #[self.val_idx[index]]
        distances = self.val2neighbors_dist[index]
        neighbor_data = self.train_xs[neighbor_indices]
        neighbor_label = self.train_ys[neighbor_indices]
        return val_datum, val_label, neighbor_data, distances, neighbor_label


    def __len__(self):
        return self.val_ys.size(0)

class TrustMNIST3Digits_NeighAgg(torch.utils.data.Dataset):
    def __init__(self, train=True, classwise=True):

        #data_dir = "./data/TensorMNIST/"
        data_dir = "../classifier/code/simple_cls/data/TensorMNIST3Digits"
        train_xs = np.load(osp.join(data_dir, "train_xs.npy"))
        train_ys = np.load(osp.join(data_dir, "train_ys.npy"))

        val_xs = np.load(osp.join(data_dir, "val_xs.npy"))
        val_ys = np.load(osp.join(data_dir, "val_ys.npy"))

        #device = "cuda:0"
        self.train_xs = torch.from_numpy(train_xs)#.to(device)
        self.train_ys = torch.from_numpy(train_ys)#.to(device)
        self.val_xs = torch.from_numpy(val_xs)#.to(device)
        self.val_ys = torch.from_numpy(val_ys)#.to(device)

        # load the dict
        #self.val2neighbors = np.load("../classifier/mnist_pretrained/baseline/feats/smallconvnet_val2train_classwise.npy")
        self.val2neighbors = np.load("../classifier/code/simple_cls/output/mnist3digits_smallerconvnet/feats/smallerconvnet_val2train_allclass.npy")
        self.val2neighbors_dist = np.load("../classifier/code/simple_cls/output/mnist3digits_smallerconvnet/feats/smallerconvnet_val2train_dist_allclass.npy")            

    def __getitem__(self, index):
        val_datum = self.val_xs[index] # [1, 28, 28]
        val_label = self.val_ys[index]

        neighbor_indices = self.val2neighbors[index] #[self.val_idx[index]]
        distances = self.val2neighbors_dist[index]
        neighbor_data = self.train_xs[neighbor_indices]
        neighbor_label = self.train_ys[neighbor_indices]
        return val_datum, val_label, neighbor_data, distances, neighbor_label


    def __len__(self):
        return self.val_ys.size(0)

class TrustMNIST3Digits_2Neigh(torch.utils.data.Dataset):
    def __init__(self, train=True, classwise=True):

        #data_dir = "./data/TensorMNIST/"
        data_dir = "../classifier/code/simple_cls/data/TensorMNIST3Digits"
        train_xs = np.load(osp.join(data_dir, "train_xs.npy"))
        train_ys = np.load(osp.join(data_dir, "train_ys.npy"))

        val_xs = np.load(osp.join(data_dir, "val_xs.npy"))
        val_ys = np.load(osp.join(data_dir, "val_ys.npy"))

        #device = "cuda:0"
        self.train_xs = torch.from_numpy(train_xs)#.to(device)
        self.train_ys = torch.from_numpy(train_ys)#.to(device)
        self.val_xs = torch.from_numpy(val_xs)#.to(device)
        self.val_ys = torch.from_numpy(val_ys)#.to(device)

        # load the dict
        #self.val2neighbors = np.load("../classifier/mnist_pretrained/baseline/feats/smallconvnet_val2train_classwise.npy")

        self.val2neighbors_classwise = np.load("../classifier/code/simple_cls/output/mnist3digits_smallerconvnet/feats/smallerconvnet_val2train_classwise.npy")
        self.val2neighbors = np.load("../classifier/code/simple_cls/output/mnist3digits_smallerconvnet/feats/smallerconvnet_val2train.npy")            

    def __getitem__(self, index):
        val_datum = self.val_xs[index] # [1, 28, 28]
        val_label = self.val_ys[index]
        neighbor_indices = self.val2neighbors_classwise[index] #[self.val_idx[index]]
        
        neighbor_data = self.train_xs[neighbor_indices]
        neighbor_label = self.train_ys[neighbor_indices]

        neighbor_indices_2 = self.val2neighbors[index] #[self.val_idx[index]]
        neighbor_data_2 = self.train_xs[neighbor_indices_2]
        neighbor_label_2 = self.train_ys[neighbor_indices_2]
        return val_datum, val_label, neighbor_data, neighbor_label, neighbor_data_2, neighbor_label_2


    def __len__(self):
        return self.val_ys.size(0)


if __name__ == '__main__':
    feature_dir = "../classifier/code/simple_cls/output/mnist/mnist_smallerconvnet/feats/finetune_e20"
    data_dir = "../classifier/code/simple_cls/data/TensorMNIST" 
    save_dir = "output/toy/mnist/mnist_smallerconvnet/finetune_e20/"
    # save_dir = "output/toy/cifar10/resnet18/e20/"
    dataseet = TensorData(feature_dir, data_dir, split='val')
    dataseet.show_neighbors(save_dir)