""" Net 网络结构 """

import logging
import numpy as np
from functools import partial
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
'''
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
'''


# 定义网络结构
class MLP1(nn.Module):
    def __init__(self, img_size=224, in_chans=3):
        super(MLP1, self).__init__()
        self.fc1 = nn.Linear(img_size*img_size*in_chans, 128)
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # ([64, 1, 28, 28])->(64,784)
        x = x.view(x.size()[0], -1)
        print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# sml


class CNN(nn.Module):
    def __init__(self, img_H=96, img_W=180, in_chans=1):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_chans, out_channels=64,
                                             kernel_size=(5, 3), stride=(3, 1), padding=(5+img_H, 2), dilation=(3, 2)),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(), nn.MaxPool2d((2, 1)))
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=(5, 3),
                stride=(3, 1),
                padding=(5 + img_H // 2, 2),
                dilation=(3, 2)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 1)))
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=(5, 3),
                stride=(3, 1),
                padding=(5 + img_H // 4, 2),
                dilation=(3, 2)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 1)))
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=(5, 3),
                stride=(3, 1),
                padding=(5 + img_H // 8, 2),
                dilation=(3, 2)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 1)))

        self.fc1 = nn.Sequential(nn.Linear(in_features=184320 * 3, out_features=2),
                                 nn.Dropout(p=0.5), nn.Softmax(dim=1))

        self.weights_init_xavier()

    def weights_init_xavier(self):
        if isinstance(self, torch.nn.Linear) or isinstance(self, torch.nn.Conv2d):
            init.xavier_uniform_(self.weight)
        # if self.bias is not None:
        #    init.zeros_(self.bias)

    def forward(self, x):
        # mnist: ([batch_size, 1, 28, 28]) / cifar10: ([batch_size, 3, 32, 32])
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = self.conv4(x)
        # print(x.shape)
        x = x.view(x.size()[0], -1)
        # print(x.shape)
        x = self.fc1(x)
        # print(x.shape)
        return x

# cyy


class CNN2(nn.Module):  # 虽然图片是黑白的，但还是有3个channel的
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(5, 3), stride=(3, 1), padding=(8, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5, 3), stride=(3, 1), padding=(8, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(5, 3), stride=(3, 1), padding=(8, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(5, 3), stride=(3, 1), padding=(8, 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(184320, 2),
        )
        self.softmax = nn.Softmax(dim=1)
        self.weights_init_xavier()

    def weights_init_xavier(self):
        if isinstance(self, torch.nn.Linear) or isinstance(self, torch.nn.Conv2d):
            init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = x.reshape(-1, 3, 96, 180)
        # 自动匹配batch size，channels=3, height=96, and width=180
        x = self.layer1(x)
        # print(x.shape)
        # 计算公式
        # For conv layer:
        # Output_height = ((Input_height + 2 * Padding_height - Dilation_height * (Kernel_height - 1) - 1) / Stride_height) + 1
        # Output_width = ((Input_width + 2 * Padding_width - Dilation_width * (Kernel_width - 1) - 1) / Stride_width) + 1
        # For maxpool layer:
        # Output_height = ((Input_height - Kernel_height) / Stride_height) + 1
        # Output_width = ((Input_width - Kernel_width) / Stride_width) + 1

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(-1, 184320)
        x = self.fc1(x)
        x = self.softmax(x)

        return x


# fz
class CNN3(nn.Module):
    def __init__(self):
        super(CNN3, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(52480, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class ResNet(nn.Module):
    def __init__(self, img_H=180, img_W=96, in_chans=3):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_chans, out_channels=64,
                                   kernel_size=5, stride=1, padding=2), nn.ReLU())
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64,
                                   kernel_size=5, stride=1, padding=2), nn.ReLU())
        self.maxpool2 = nn.MaxPool2d(2, 2)
        # 由于MaxPooling降采样两次，特征图缩减为原图的 1/4 * 1/4
        self.fc1 = nn.Sequential(nn.Linear(in_features=64*int(img_H/4)*int(img_W/4),
                                 out_features=1000), nn.Dropout(p=0.4), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(in_features=1000, out_features=10), nn.Softmax(dim=1))
        self.weights_init_xavier()

    def weights_init_xavier(self):
        if isinstance(self, torch.nn.Linear) or isinstance(self, torch.nn.Conv2d):
            init.xavier_uniform_(self.weight)
        # if self.bias is not None:
            # init.zeros_(self.bias)

    def forward(self, x):
        # mnist: ([batch_size, 1, 28, 28]) / cifar10: ([batch_size, 3, 32, 32])
        # print(x.shape)
        # print(self.conv1(x).shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.maxpool1(x)
        # print(x.shape)
        # print(self.conv2(x).shape)
        x = self.conv2(x) + x
        x = self.maxpool2(x)
        # print(x.shape)
        # 合并后两维通道, 保留第一维(batch_size)
        x = x.view(x.size()[0], -1)
        # print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        # print(x.shape)
        return x


class CNN_with_decoder(nn.Module):  # 虽然图片是黑白的，但还是有3个channel的
    def __init__(self, decoder_output_dim=3):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(5, 3), stride=(3, 1), padding=(8, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5, 3), stride=(3, 1), padding=(8, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(5, 3), stride=(3, 1), padding=(8, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(5, 3), stride=(3, 1), padding=(8, 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.decoder = nn.Sequential(
            nn.Linear(184320, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, decoder_output_dim)
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(184320, 2),
        )
        self.softmax = nn.Softmax(dim=1)
        self.weights_init_xavier()

    def weights_init_xavier(self):
        if isinstance(self, torch.nn.Linear) or isinstance(self, torch.nn.Conv2d):
            init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = x.reshape(-1, 3, 96, 180)
        # 自动匹配batch size，channels=3, height=96, and width=180
        x = self.layer1(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(-1, 184320)
        x = self.fc1(x)
        x = self.softmax(x)

        return x


class Decoder(CNN_with_decoder):

    def forward(self, x):
        x = x.reshape(-1, 3, 96, 180)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(-1, 184320)
        x = self.decoder(x)

        return x
