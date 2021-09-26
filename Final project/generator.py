# **********************************
# *      INFO8010 - Project        *
# *  CycleGAN for style transfer   *
# *             ---                *
# *  Antoine DEBOR & Pierre NAVEZ  *
# *       ULiège, May 2021         *
# **********************************

import torch
import torch.nn as nn

class GInitBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=0, bias=True),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.conv(x)

class GDownsamplingBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.conv(x)

class GResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(out_channels),
            nn.Identity()
            )

    def forward(self, x):
        return x + self.conv(x) # Skip connections

class GUpsamplingBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=1, bias=True),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.conv(x)

class GLastBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=0),
                nn.Tanh()
        )

    def forward(self, x):
        return self.conv(x)

class Generator(nn.Module):
    """ ResNet-based generator """
    def __init__(self, in_channels=3, n_features=64, n_residuals=9):

        super(Generator,self).__init__()

        layers = []

        # Initial layer
        layers.append(GInitBlock(in_channels, n_features))

        # Encoder layers (Downsampling)
        for i in range(1,3):
            layers.append(GDownsamplingBlock(n_features*i, n_features*2*i, stride=2))

        # Transformer layers
        for i in range(n_residuals):
            layers.append(GResidualBlock(n_features*4, n_features*4))

        # Decoder layers (Upsampling)
        for i in range(1,3):
            layers.append(GUpsamplingBlock(int(n_features*4/i),int(n_features*4/(2*i)), stride=2))

        # Final layer
        layers.append(GLastBlock(n_features, in_channels))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
