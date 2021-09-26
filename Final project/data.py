# **********************************
# *      INFO8010 - Project        *
# *  CycleGAN for style transfer   *
# *             ---                *
# *  Antoine DEBOR & Pierre NAVEZ  *
# *       ULiège, May 2021         *
# **********************************

import os
import itertools
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import config

class XDataset(Dataset):

    def __init__(self, root_dir, transform=None, train=True):

        self.samples = []
        self.root_dir = root_dir
        self.transform = transforms.Compose(transform[:3]) if transform else None
        self.train = train

        if train == True:
            print("Building the training X set...")
            for img in os.listdir(self.root_dir):
                image = np.asarray(Image.open(self.root_dir+'/'+img).convert('RGB'))
                if self.transform:
                    self.samples.append(self.transform(image))

            print('Complete !\n')

        else:
            print("Building the testing X set...")
            for img in os.listdir(self.root_dir):
                image = np.asarray(Image.open(self.root_dir+'/'+img).convert('RGB'))
                if self.transform:
                    self.samples.append(self.transform(image))

            print('Complete !\n')

    def __len__(self):
    	  return len(self.samples)

    def __getitem__(self,idx):
        return self.samples[idx]

class YDataset(Dataset):

    def __init__(self, root_dir, transform=None, train=True, len_X=0):

        self.samples = []
        self.root_dir = root_dir
        self.transform = transforms.Compose(transform[:3]) if transform else None
        self.transform_hf = transforms.Compose(transform[:4]) if transform else None
        self.transform_rot = transforms.Compose(transform[:3] + [transform[4]]) if transform else None
        self.transform_aff = transforms.Compose(transform[:3] + [transform[5]]) if transform else None
        self.len_X = len_X
        self.train = train

        if train == True:
            print("Building the training Y set...")
            for img in os.listdir(self.root_dir):
                image = Image.open(self.root_dir+'/'+img).convert('RGB')
                image = np.asarray(image)
                if self.transform:
                    self.samples.append(self.transform(image))
                    if config.MODE == "F2I":
                      self.samples.append(self.transform_hf(image))
                      self.samples.append(self.transform_rot(image))
                      self.samples.append(self.transform_aff(image))

            print('Complete !\n')

        else:
            print("Building the testing Y set...")
            for img in os.listdir(self.root_dir):
                image = Image.open(self.root_dir+'/'+img).convert('RGB')
                image = np.asarray(image)
                if self.transform:
                    self.samples.append(self.transform(image))

            print('Complete !\n')

    def __len__(self):
        if self.len_X > len(self.samples):   # Virtual length
            return self.len_X
        else:
            return len(self.samples)

    def __getitem__(self,idx):
        if self.len_X > len(self.samples):   # Virtual length
            return self.samples[idx % len(self.samples)]
        else:
            return self.samples[idx]

def make_loader(X_path, Y_path, train=True):
    ret_tuple = ()

    if train == True:
        X_trainset = XDataset(X_path, transform=config.TRANSFORM_LIST, train=True)
        len_X = X_trainset.__len__()
        X_trainloader = DataLoader(X_trainset, batch_size=10, shuffle=True, num_workers=2)

        ret_tuple = ret_tuple + (X_trainloader, )

        Y_trainset = YDataset(Y_path, transform=config.TRANSFORM_LIST, train=True, len_X=len_X)
        Y_trainloader = DataLoader(Y_trainset, batch_size=10, shuffle=True, num_workers=2)

        ret_tuple = ret_tuple + (Y_trainloader, )

    else:
        X_testset = XDataset(X_path, transform=config.TRANSFORM_LIST, train=False)
        X_testloader = DataLoader(X_testset, batch_size=10, shuffle=True, num_workers=2)

        ret_tuple = ret_tuple + (X_testloader, )

        Y_testset = YDataset(Y_path, transform=config.TRANSFORM_LIST, train=False)
        Y_testloader = DataLoader(Y_testset, batch_size=10, shuffle=True, num_workers=2)

        ret_tuple = ret_tuple + (Y_testloader, )

    return ret_tuple
