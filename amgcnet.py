import torch
import torch.nn as nn
import torch.nn.functional as F

class AMGCBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        self.bn1   = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, k, 1, p, bias=False)
        self.bn2   = nn.BatchNorm2d(c_out)
        self.short = nn.Conv2d(c_in, c_out, 1, s, 0, bias=False) if (s != 1 or c_in != c_out) else nn.Identity()

    def forward(self, x):
        skip = self.short(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + skip, inplace=True)

class AMGCNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, width=64):
        super().__init__()
        C = width
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, C, 7, 2, 3, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        self.stage1 = AMGCBlock(C,   C)
        self.stage2 = AMGCBlock(C, 2*C, s=2)
        self.stage3 = AMGCBlock(2*C, 4*C, s=2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4*C, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)

