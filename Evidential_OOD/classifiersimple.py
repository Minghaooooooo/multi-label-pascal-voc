import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

# GroupNorm

class clssimp(nn.Module):
    def __init__(self, ch=2880, num_classes=20):

        super(clssimp, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.way1 = nn.Sequential(
            nn.Linear(ch, 1000, bias=True),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
        )
        self.cls= nn.Linear(1000, num_classes,bias=True)

    def forward(self, x):
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.way1(x)
        logits = self.cls(x)
        return logits

    def intermediate_forward(self, x):
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.way1(x)
        return x


class LeNetAdapter(nn.Module):
    def __init__(self, input_dim=1024, num_classes=20):
        super(LeNetAdapter, self).__init__()

        # Pool first to handle DenseNet output
        self.pool = nn.AdaptiveAvgPool2d(output_size=(8, 8))  # Change to a spatial size of 8x8

        # LeNet-inspired convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=5, stride=1, padding=2),  # Match DenseNet output channels
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fully connected layers for classification
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 2 * 2, 120),  # Adjust for final pooled size
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        # Apply pooling to prepare for convolutional layers
        x = self.pool(x)

        # Pass through convolutional layers
        x = self.conv_layers(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        logits = self.fc_layers(x)
        return logits

    def intermediate_forward(self, x):
        # Pool first to extract meaningful features
        x = self.pool(x)
        return x


class segclssimp_group(nn.Module):
    def __init__(self, ch=2880, num_classes=21):

        super(segclssimp_group, self).__init__()
        self.depthway1 = nn.Sequential(
            nn.Conv2d(ch, 1000, kernel_size=1),
            nn.GroupNorm(4,1000),
            nn.ReLU(inplace=True),
        )
        self.depthway2 = nn.Sequential(
            nn.Conv2d(1000, 1000, kernel_size=1),
            nn.GroupNorm(4,1000),
            nn.ReLU(inplace=True),
        )
        self.depthway3 = nn.Sequential(
            nn.Conv2d(1000, 512, kernel_size=1),
            nn.GroupNorm(4,512),
            nn.ReLU(inplace=True),
        )

        self.clsdepth = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        # bp()

        seg = self.depthway1(x)
        seg = self.depthway2(seg)
        seg = self.depthway3(seg)
        seg = self.clsdepth(seg)

        return seg
