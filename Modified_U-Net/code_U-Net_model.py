from collections import OrderedDict
import torch
import torch.nn as nn 
from torch.autograd import Variable 
import torch.nn.functional as F 
import torchvision.transforms as transforms 
from torch.utils.data import Dataset, DataLoader
import albumentations as A 
from albumentations.pytorch import ToTensorV2 
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 


class Conv(nn.Module): 
        def __init__(self, in_ch, out_ch):
                super(Conv, self).__init__()
                self.conv = nn.Sequential( 
                        nn.Conv2d(in_ch, out_ch, 3, padding=1), 
                        nn.BatchNorm2d(out_ch), 
                        nn.ReLU()) 
        def forward(self, x):
                return self.conv(x)


class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch, is_res=True):
                super(ConvBlock, self).__init__()
                self.conv1 = Conv(in_ch, out_ch)
                self.conv2 = Conv(out_ch, out_ch)
                self.conv3 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
                self.bn = nn.BatchNorm2d(out_ch)
                self.relu = nn.ReLU()
                self.is_res = is_res
        def forward(self, x):
                x = self.conv1(x)
                y = self.conv2(x)
                y = self.conv3(y)
                y = self.bn(y)
                if self.is_res:
                        y += x 
                return self.relu(y)


class DeconvBlock(nn.Module): 
        def __init__(self, in_ch, out_ch, attn=None):
                super(DeconvBlock, self).__init__()
                self.deconv = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
                self.conv = ConvBlock(in_ch, out_ch)
                self.attn = attn
        def forward(self, x, bridge):
                x = self.deconv(x)
                if self.attn:
                        bridge = self.attn(bridge, x) 
                x = torch.cat([x, bridge], dim=1) 
                return self.conv(x)


class Encoder(nn.Module):
        def __init__(self, in_ch=3, out_ch=64, depth=5):
                super(Encoder, self).__init__()
                self.pool = nn.MaxPool2d(2)
                self.convs = nn.ModuleList()
                for _ in range(depth):
                        self.convs.append(ConvBlock(in_ch, out_ch)) 
                        in_ch = out_ch
                        out_ch *= 2
        def forward(self, x):
                res = []
                for i, m in enumerate(self.convs):
                        if i > 0:
                                x = self.pool(x) 
                        x = m(x)
                        res.append(x) 
                return res


class Attn(nn.Module):
        def __init__(self, ch):
                super(Attn, self).__init__()
                self.wx = nn.Conv2d(ch, ch, 1)
                self.wg = nn.Conv2d(ch, ch, 1)
                self.psi = nn.Conv2d(ch, ch, 1)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()
        def forward(self, x, g): 
                identity = x
                x = self.wx(x)
                g = self.wg(g)
                x = self.relu(x + g)
                x = self.psi(x)
                x = self.sigmoid(x)
                return identity * (x + 1)


class Decoder(nn.Module):
        def __init__(self, in_ch=1024, depth=4, attn=True):
                super(Decoder, self).__init__()
                self.depth = depth
                self.deconvs = nn.ModuleList()
                for _ in range(depth):
                        self.deconvs.append(DeconvBlock(in_ch, in_ch // 2, Attn(in_ch // 2) if attn else None))
                        in_ch //= 2

        def forward(self, x_list):
                for i in range(self.depth):
                        if i == 0:
                                x = x_list.pop() 
                        bridge = x_list.pop() 
                        x = self.deconvs[i](x, bridge)
                return x


class UNet(nn.Module): 
        def __init__(self, in_ch=3, out_ch=1, encoder_depth=5, regressive=True, attn=True):
                super(UNet, self).__init__()
                self.encoder = Encoder(in_ch, 64, encoder_depth)
                self.decoder = Decoder(1024, encoder_depth-1, attn)
                self.conv = nn.Conv2d(64, out_ch, 1)
                self.sigmoid = nn.Sigmoid()
                self.regressive = regressive
        def forward(self, x):
                x = self.encoder(x) 
                x = self.decoder(x) 
                x = self.conv(x) 
                if self.regressive:
                        return x 
                else:
                        return self.sigmoid(x).clamp(1e-4, 1 - 1e-4) 
