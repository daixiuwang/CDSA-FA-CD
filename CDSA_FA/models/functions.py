import torch
import torch.nn as nn
from .AFF import AFF, MS_CAM


class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels , 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)

        y = self.fc(y)
        return x * y

class Dlt(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.d5 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(dim)
        )
        self.d3 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(dim)
        )
        self.d1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(dim)
        )
        self.relu = nn.ReLU(inplace=True)
        #self.c = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        x1 = self.d1(x)
        x3 = self.d3(x)
        x5 = self.d5(x)
        x = self.relu(x1 + x3 + x5)
        return x



class Exchange(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.oc = nn.Conv2d(dim // 2, 1, kernel_size=1)
        self.cbr = DoubleConv(dim, dim // 2)
        self.up = nn.ConvTranspose2d(dim * 2, dim, kernel_size=2, stride=2)

    def forward(self, x1, x2, y=None):
        if y is not None:
            y = self.up(y)
            x11 = x1 + y
            x22 = x2 + y
        else:
            x_diff = abs(x1 - x2)
            x11 = x1 + x_diff
            x22 = x2 + x_diff

        x11_1, x11_2 = torch.split(x11, x11.size(1) // 2, dim=1)
        x22_1, x22_2 = torch.split(x22, x22.size(1) // 2, dim=1)
        y1 = torch.cat([x11_1, x22_2], dim = 1)
        y2 = torch.cat([x22_1, x11_2], dim = 1)

        y1 = self.cbr(y1)
        y1 = self.oc(y1)

        y2 = self.cbr(y2)
        y2 = self.oc(y2)
        map = y1 + y2
        sigmoid_map = torch.sigmoid(map)
        return sigmoid_map

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x= self.single_conv(x)
        return x


class PA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.aff = AFF(out_channels)
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
        self.cbr3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.cbr5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        y3 = self.cbr3(x)
        y5 = self.cbr5(x)
        y = self.aff(y3, y5)
        x = self.conv(x)
        z = x + y
        return z

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
