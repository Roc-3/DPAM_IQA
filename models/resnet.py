import torch
import torch.nn as nn
import torchvision.models as models

import torch.nn.functional as F

class ResBlockGroup(nn.Module):
    def __init__(self, dim, num_blocks):
        super(ResBlockGroup, self).__init__()
        self.blocks = nn.Sequential(*[ResBlock(dim) for _ in range(num_blocks)])

    def forward(self, x):
        return self.blocks(x)

class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class ConvBlock(nn.Module):
    def __init__(self, channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(1, 1))
        self.bn2 = nn.BatchNorm2d(channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_texture_res):
        x_weight = self.conv1(x_texture_res)
        x_weight = self.bn1(x_weight)
        x_weight = self.sigmoid(x_weight)
        x_weight = self.conv2(x_weight)
        x_weight = self.bn2(x_weight)
        x_weight = self.sigmoid(x_weight)
        x_weight = F.interpolate(x_weight, size=(28, 28), mode='bilinear', align_corners=False)
        x_texture_res = F.interpolate(x_texture_res, size=(28, 28), mode='bilinear', align_corners=False)
        x_weight = torch.mul(x_texture_res, x_weight)
        out = torch.mul(x, x_weight)
        # out = torch.mul(x, x_weight).sum(dim=(2, 3)) / (x_weight.sum(dim=(2, 3)) + 0.0001)
        return out

class IQANet(nn.Module):
    def __init__(self, channels):
        super(IQANet, self).__init__()
        self.resnet101_freeze = nn.Sequential(*list(models.resnet101(True).children())[:7])
        self.resnet101 = nn.Sequential(*list(models.resnet101(True).children())[7:8])
        self.texconv = nn.Conv2d(2048, channels, 1, 1, 0)
        self.wsp = ConvBlock(channels)

        # freeze conv and weight of batchnorm
        for para in self.resnet101_freeze.parameters():
            para.requires_grad = False

        # freeze running mean and var of barchnorm
        self.resnet101_freeze.eval()

    def forward(self, x, x_texture):
        x_texture_res = self.resnet101_freeze(x_texture)
        x_texture_res = self.resnet101(x_texture_res)
        x_texture_res = self.texconv(x_texture_res)
        x = self.wsp(x, x_texture_res)
        return x

    def train(self, mode=True):
        self.training = mode

        for m in [self.resnet101, self.texconv, self.wsp]:
            m.training = mode
            for module in m.children():
                module.train(mode)

        return self
    