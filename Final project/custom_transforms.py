# **********************************
# *      INFO8010 - Project        *
# *  CycleGAN for style transfer   *
# *             ---                *
# *  Antoine DEBOR & Pierre NAVEZ  *
# *       ULiège, May 2021         *
# **********************************

import torch

class AddGaussianNoise(object):
    "Gathered from https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745"
    def __init__(self, mean=0., std=0.2):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
