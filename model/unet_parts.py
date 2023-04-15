""" U-netのモデルのパーツ類"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """_summary_
        Conv2d → BatchNorm2d → ReLu を2セット

    Args:
        in_channels (int): 特徴マップの入力チャンネル数
        out_channels (int): 特徴マップの出力チャンネル数
    """
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """_summary_
        MaxPool2d → DoubleConv を１セット

    Args:
        in_channels (int): 特徴マップの入力チャンネル数
        out_channels (int): 特徴マップの出力チャンネル数
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    

class Up(nn.Module):
    """_summary_
        ConvTranspose2d → DoubleConv を1セット
        UpSampleした後skip connectionをcropして結合

    Args:
        in_channels (int): 特徴マップの入力チャンネル数
        out_channels (int): 特徴マップの出力チャンネル数
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def crop(self, tensor_ref_size, tensor_to_crop):
        _, _, h, w = tensor_to_crop.shape
        _, _, h_new, w_new = tensor_ref_size.shape

        ch, cw = h//2, w//2
        ch_new, cw_new = h_new//2, w_new//2
        x1 = int(cw - cw_new)
        y1 = int(ch - ch_new)
        x2 = int(x1 + w_new)
        y2 = int(y1 + h_new)
        return tensor_to_crop[:, :, y1:y2, x1:x2]

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.crop(x1, x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    

class OutConv(nn.Module):
    """_summary_
        最終出力

    Args:
        in_channels (int): 特徴マップの入力チャンネル数
        out_channels (int): 特徴マップの出力チャンネル数
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)