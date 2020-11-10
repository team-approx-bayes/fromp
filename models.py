import torch
import torch.nn as nn
import torch.nn.functional as F


# Fully connected network, size: input_size, hidden_size,... , output_size
class MLP(nn.Module):
    def __init__(self, size, act='sigmoid'):
        super(type(self), self).__init__()
        self.num_layers = len(size) - 1
        lower_modules = []
        for i in range(self.num_layers - 1):
            lower_modules.append(nn.Linear(size[i], size[i+1]))
            if act == 'relu':
                lower_modules.append(nn.ReLU())
            elif act == 'sigmoid':
                lower_modules.append(nn.Sigmoid())
            else:
                raise ValueError("%s activation layer hasn't been implemented in this code" %act)
        self.layer_1 = nn.Sequential(*lower_modules)
        self.layer_2 = nn.Linear(size[-2], size[-1])


    def forward(self, x):
        o = self.layer_1(x)
        o = self.layer_2(o)
        return o


class SplitMLP(nn.Module):
    def __init__(self, size, act='sigmoid'):
        super(type(self), self).__init__()
        self.num_layers = len(size) - 1
        lower_modules = []
        for i in range(self.num_layers - 1):
            lower_modules.append(nn.Linear(size[i], size[i+1]))
            if act == 'relu':
                lower_modules.append(nn.ReLU())
            elif act == 'sigmoid':
                lower_modules.append(nn.Sigmoid())
            else:
                raise ValueError("%s activation layer hasn't been implemented in this code" %act)
        self.layer_1 = nn.Sequential(*lower_modules)
        self.layer_2 = nn.Linear(size[-2], size[-1])


    def forward(self, x, label_set):
        o = self.layer_1(x)
        o = self.layer_2(o)
        o = o[:, label_set]
        return o


class CifarNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(type(self), self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.25),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.25)
        )
        self.linear_block = nn.Sequential(
            nn.Linear(64*6*6, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.out_block = nn.Linear(512, out_channels)


    def weight_init(self):
        nn.init.constant_(self.out_block.weight, 0)
        nn.init.constant_(self.out_block.bias, 0)


    def forward(self, x, label_set):
        o = self.conv_block(x)
        o = torch.flatten(o, 1)
        o = self.linear_block(o)
        o = self.out_block(o)
        o = o[:, label_set]
        return o
