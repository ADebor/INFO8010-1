# **********************************
# *      INFO8010 - Project        *
# *  CycleGAN for style transfer   *
# *             ---                *
# *  Antoine DEBOR & Pierre NAVEZ  *
# *       ULiège, May 2021         *
# **********************************

import torch
import torch.nn as nn

class DBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=True),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

    def forward(self,x):
        return self.conv(x)

class Discriminator(nn.Module):
    """ PatchGAN """
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):

        super(Discriminator, self).__init__()

        layers = []

        # Initial layer
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ))

        # Inner layers
        in_channels = features[0]

        for feature in features[1:]:
            layers.append(DBlock(in_channels, feature, stride = 1 if feature == features[-1] else 2))
            in_channels = feature

        # Last layer
        layers.append(nn.Sequential(nn.Conv2d(feature, 1, kernel_size=4, stride=1, padding=1)))

        self.model = nn.Sequential(*layers)


    def forward(self,x):
        return self.model(x)
