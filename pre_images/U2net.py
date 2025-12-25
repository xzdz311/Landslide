import cv2

import random
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast  # 混合精度训练
import numpy as np
from tqdm import tqdm
import os
from predict import create_test_loader_from_csv, predict_and_evaluate




class RSU7(nn.Module):
    """RSU-7模块: 高度为7的残差U块"""

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()
        self.out_ch = out_ch

        # 编码器部分
        self.conv0 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)

        self.conv1 = nn.Conv2d(out_ch, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)

        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)

        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv3 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_ch)

        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv4 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(mid_ch)

        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv5 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(mid_ch)

        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv6 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(mid_ch)

        # 最底层的卷积
        self.conv7 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, dilation=2, padding=2, bias=False)
        self.bn7 = nn.BatchNorm2d(mid_ch)

        # 解码器部分
        self.conv6d = nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn6d = nn.BatchNorm2d(mid_ch)

        self.conv5d = nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn5d = nn.BatchNorm2d(mid_ch)

        self.conv4d = nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn4d = nn.BatchNorm2d(mid_ch)

        self.conv3d = nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn3d = nn.BatchNorm2d(mid_ch)

        self.conv2d = nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn2d = nn.BatchNorm2d(mid_ch)

        self.conv1d = nn.Conv2d(mid_ch * 2, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1d = nn.BatchNorm2d(out_ch)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 保存原始输入用于残差连接
        x_input = x

        # 第一层卷积
        hx = self.conv0(x_input)
        hx_in = hx  # 保存用于残差连接

        # 编码器路径
        hx1 = self.relu(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        hx2 = self.relu(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.relu(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)

        hx5 = self.relu(self.bn5(self.conv5(hx)))
        hx = self.pool5(hx5)

        hx6 = self.relu(self.bn6(self.conv6(hx)))

        hx7 = self.relu(self.bn7(self.conv7(hx6)))

        # 解码器路径
        hx6d = self.relu(self.bn6d(self.conv6d(torch.cat((hx6, hx7), 1))))
        hx6dup = F.interpolate(hx6d, size=hx5.shape[2:], mode='bilinear', align_corners=True)

        hx5d = self.relu(self.bn5d(self.conv5d(torch.cat((hx5, hx6dup), 1))))
        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:], mode='bilinear', align_corners=True)

        hx4d = self.relu(self.bn4d(self.conv4d(torch.cat((hx4, hx5dup), 1))))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=True)

        hx3d = self.relu(self.bn3d(self.conv3d(torch.cat((hx3, hx4dup), 1))))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=True)

        hx2d = self.relu(self.bn2d(self.conv2d(torch.cat((hx2, hx3dup), 1))))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=True)

        hx1d = self.relu(self.bn1d(self.conv1d(torch.cat((hx1, hx2dup), 1))))

        # 确保hx1d和hx_in大小一致，如果不一致则调整hx_in
        if hx1d.shape != hx_in.shape:
            # 调整hx_in的大小以匹配hx1d
            hx_in_adjusted = F.interpolate(hx_in, size=hx1d.shape[2:], mode='bilinear', align_corners=True)
        else:
            hx_in_adjusted = hx_in

        # 残差连接
        return hx1d + hx_in_adjusted


class RSU6(nn.Module):
    """RSU-6模块: 高度为6的残差U块"""

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()
        self.out_ch = out_ch

        self.conv0 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)

        self.conv1 = nn.Conv2d(out_ch, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)

        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)

        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv3 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_ch)

        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv4 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(mid_ch)

        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv5 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(mid_ch)

        self.conv6 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, dilation=2, padding=2, bias=False)
        self.bn6 = nn.BatchNorm2d(mid_ch)

        # 解码器
        self.conv5d = nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn5d = nn.BatchNorm2d(mid_ch)

        self.conv4d = nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn4d = nn.BatchNorm2d(mid_ch)

        self.conv3d = nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn3d = nn.BatchNorm2d(mid_ch)

        self.conv2d = nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn2d = nn.BatchNorm2d(mid_ch)

        self.conv1d = nn.Conv2d(mid_ch * 2, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1d = nn.BatchNorm2d(out_ch)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 保存原始输入用于残差连接
        x_input = x

        # 第一层卷积
        hx = self.conv0(x_input)
        hx_in = hx  # 保存用于残差连接

        hx1 = self.relu(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        hx2 = self.relu(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.relu(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)

        hx5 = self.relu(self.bn5(self.conv5(hx)))

        hx6 = self.relu(self.bn6(self.conv6(hx5)))

        hx5d = self.relu(self.bn5d(self.conv5d(torch.cat((hx5, hx6), 1))))
        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:], mode='bilinear', align_corners=True)

        hx4d = self.relu(self.bn4d(self.conv4d(torch.cat((hx4, hx5dup), 1))))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=True)

        hx3d = self.relu(self.bn3d(self.conv3d(torch.cat((hx3, hx4dup), 1))))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=True)

        hx2d = self.relu(self.bn2d(self.conv2d(torch.cat((hx2, hx3dup), 1))))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=True)

        hx1d = self.relu(self.bn1d(self.conv1d(torch.cat((hx1, hx2dup), 1))))

        # 确保hx1d和hx_in大小一致
        if hx1d.shape != hx_in.shape:
            hx_in_adjusted = F.interpolate(hx_in, size=hx1d.shape[2:], mode='bilinear', align_corners=True)
        else:
            hx_in_adjusted = hx_in

        # 残差连接
        return hx1d + hx_in_adjusted


class RSU5(nn.Module):
    """RSU-5模块: 高度为5的残差U块"""

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()
        self.out_ch = out_ch

        self.conv0 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)

        self.conv1 = nn.Conv2d(out_ch, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)

        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)

        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv3 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_ch)

        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv4 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(mid_ch)

        self.conv5 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, dilation=2, padding=2, bias=False)
        self.bn5 = nn.BatchNorm2d(mid_ch)

        # 解码器
        self.conv4d = nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn4d = nn.BatchNorm2d(mid_ch)

        self.conv3d = nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn3d = nn.BatchNorm2d(mid_ch)

        self.conv2d = nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn2d = nn.BatchNorm2d(mid_ch)

        self.conv1d = nn.Conv2d(mid_ch * 2, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1d = nn.BatchNorm2d(out_ch)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 保存原始输入用于残差连接
        x_input = x

        # 第一层卷积
        hx = self.conv0(x_input)
        hx_in = hx  # 保存用于残差连接

        hx1 = self.relu(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        hx2 = self.relu(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.relu(self.bn4(self.conv4(hx)))

        hx5 = self.relu(self.bn5(self.conv5(hx4)))

        hx4d = self.relu(self.bn4d(self.conv4d(torch.cat((hx4, hx5), 1))))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=True)

        hx3d = self.relu(self.bn3d(self.conv3d(torch.cat((hx3, hx4dup), 1))))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=True)

        hx2d = self.relu(self.bn2d(self.conv2d(torch.cat((hx2, hx3dup), 1))))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=True)

        hx1d = self.relu(self.bn1d(self.conv1d(torch.cat((hx1, hx2dup), 1))))

        # 确保hx1d和hx_in大小一致
        if hx1d.shape != hx_in.shape:
            hx_in_adjusted = F.interpolate(hx_in, size=hx1d.shape[2:], mode='bilinear', align_corners=True)
        else:
            hx_in_adjusted = hx_in

        # 残差连接
        return hx1d + hx_in_adjusted


class RSU4(nn.Module):
    """RSU-4模块: 高度为4的残差U块"""

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()
        self.out_ch = out_ch

        self.conv0 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)

        self.conv1 = nn.Conv2d(out_ch, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)

        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)

        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv3 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_ch)

        self.conv4 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, dilation=2, padding=2, bias=False)
        self.bn4 = nn.BatchNorm2d(mid_ch)

        # 解码器
        self.conv3d = nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn3d = nn.BatchNorm2d(mid_ch)

        self.conv2d = nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn2d = nn.BatchNorm2d(mid_ch)

        self.conv1d = nn.Conv2d(mid_ch * 2, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1d = nn.BatchNorm2d(out_ch)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 保存原始输入用于残差连接
        x_input = x

        # 第一层卷积
        hx = self.conv0(x_input)
        hx_in = hx  # 保存用于残差连接

        hx1 = self.relu(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        hx2 = self.relu(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu(self.bn3(self.conv3(hx)))

        hx4 = self.relu(self.bn4(self.conv4(hx3)))

        hx3d = self.relu(self.bn3d(self.conv3d(torch.cat((hx3, hx4), 1))))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=True)

        hx2d = self.relu(self.bn2d(self.conv2d(torch.cat((hx2, hx3dup), 1))))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=True)

        hx1d = self.relu(self.bn1d(self.conv1d(torch.cat((hx1, hx2dup), 1))))

        # 确保hx1d和hx_in大小一致
        if hx1d.shape != hx_in.shape:
            hx_in_adjusted = F.interpolate(hx_in, size=hx1d.shape[2:], mode='bilinear', align_corners=True)
        else:
            hx_in_adjusted = hx_in

        # 残差连接
        return hx1d + hx_in_adjusted


class RSU4F(nn.Module):
    """RSU-4F模块: 无下采样的RSU-4（使用空洞卷积）"""

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()
        self.out_ch = out_ch

        self.conv0 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)

        self.conv1 = nn.Conv2d(out_ch, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)

        self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, dilation=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)

        self.conv3 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, dilation=4, padding=4, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_ch)

        self.conv4 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, dilation=8, padding=8, bias=False)
        self.bn4 = nn.BatchNorm2d(mid_ch)

        # 解码器
        self.conv3d = nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn3d = nn.BatchNorm2d(mid_ch)

        self.conv2d = nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn2d = nn.BatchNorm2d(mid_ch)

        self.conv1d = nn.Conv2d(mid_ch * 2, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1d = nn.BatchNorm2d(out_ch)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 保存原始输入用于残差连接
        x_input = x

        # 第一层卷积
        hx = self.conv0(x_input)
        hx_in = hx  # 保存用于残差连接

        hx1 = self.relu(self.bn1(self.conv1(hx)))
        hx2 = self.relu(self.bn2(self.conv2(hx1)))
        hx3 = self.relu(self.bn3(self.conv3(hx2)))
        hx4 = self.relu(self.bn4(self.conv4(hx3)))

        hx3d = self.relu(self.bn3d(self.conv3d(torch.cat((hx3, hx4), 1))))
        hx2d = self.relu(self.bn2d(self.conv2d(torch.cat((hx2, hx3d), 1))))
        hx1d = self.relu(self.bn1d(self.conv1d(torch.cat((hx1, hx2d), 1))))

        # RSU4F没有下采样，所以尺寸应该保持不变
        # 残差连接
        return hx1d + hx_in


class U2NET(nn.Module):
    """U^2-Net模型 - 早期融合版本，输入输出与原始U-Net保持一致"""

    def __init__(self, n_channels=4, n_classes=1):
        super(U2NET, self).__init__()

        # 编码器 (RSU模块)
        self.stage1 = RSU7(n_channels, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        # 解码器
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        # 侧边输出
        self.side1 = nn.Conv2d(64, n_classes, kernel_size=3, padding=1)
        self.side2 = nn.Conv2d(64, n_classes, kernel_size=3, padding=1)
        self.side3 = nn.Conv2d(128, n_classes, kernel_size=3, padding=1)
        self.side4 = nn.Conv2d(256, n_classes, kernel_size=3, padding=1)
        self.side5 = nn.Conv2d(512, n_classes, kernel_size=3, padding=1)
        self.side6 = nn.Conv2d(512, n_classes, kernel_size=3, padding=1)

        # 最终融合层
        self.outconv = nn.Conv2d(6 * n_classes, n_classes, kernel_size=1)

    def forward(self, optical, dem):
        # 早期融合: 在通道维度拼接
        x = torch.cat([optical, dem], dim=1)

        # 编码路径
        hx1 = self.stage1(x)
        hx = self.pool12(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        hx6 = self.stage6(hx)
        hx6up = F.interpolate(hx6, size=hx5.shape[2:], mode='bilinear', align_corners=True)

        # 解码路径
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:], mode='bilinear', align_corners=True)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=True)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=True)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=True)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # 侧边输出
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = F.interpolate(d2, size=x.shape[2:], mode='bilinear', align_corners=True)

        d3 = self.side3(hx3d)
        d3 = F.interpolate(d3, size=x.shape[2:], mode='bilinear', align_corners=True)

        d4 = self.side4(hx4d)
        d4 = F.interpolate(d4, size=x.shape[2:], mode='bilinear', align_corners=True)

        d5 = self.side5(hx5d)
        d5 = F.interpolate(d5, size=x.shape[2:], mode='bilinear', align_corners=True)

        d6 = self.side6(hx6)
        d6 = F.interpolate(d6, size=x.shape[2:], mode='bilinear', align_corners=True)

        # 融合所有侧边输出
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return d0



def get_simple_training_config():
    """获取简单训练配置"""

    # 1. 创建简单模型
    model = U2NET(n_channels=4, n_classes=1)

    # 2. 使用标准损失函数（先排除复杂的损失函数）
    def simple_loss(pred, target):
        """简单的BCE损失函数"""
        return nn.BCEWithLogitsLoss()(pred, target)

    # 或者联合损失
    def combined_loss(pred, target):
        """BCE + Dice损失"""
        bce = nn.BCEWithLogitsLoss()(pred, target)

        # Dice损失
        probs = torch.sigmoid(pred)
        smooth = 1e-6
        intersection = (probs * target).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice.mean()

        return bce + dice_loss

    def combined_loss_v1(pred, target, alpha=0.25, gamma=2.0, dice_weight=0.5):
        """
        Focal Loss + Dice Loss
        优点：自动处理类别不平衡，对简单样本降权
        适合：FP过多，正负样本极不平衡的情况
        """
        # Focal Loss部分
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        focal_loss = focal_loss.mean()

        # Dice Loss部分
        probs = torch.sigmoid(pred)
        smooth = 1e-6
        intersection = (probs * target).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice.mean()

        return focal_loss + dice_weight * dice_loss

    # 3. 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    # 4. 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
    )

    return model, combined_loss, optimizer, scheduler


def main():
    """主训练函数"""

    # 获取配置
    model, criterion, optimizer, scheduler = get_simple_training_config()

    print(f"模型架构: {model.__class__.__name__}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    device_ids = list(range(torch.cuda.device_count()))
    print(f"可用的GPU: {device_ids}")

    # 数据准备（使用新函数）
    data_dir = r"F:\zx\datasets\Bijie-landslide-dataset"
    csv_path = r"F:\zx\landslit\Landslide\pre_images\detailed_results.csv"  # 根据实际情况修改

    # 创建测试集加载器
    test_loader, test_dataset = create_test_loader_from_csv(
        csv_path=csv_path,
        data_dir=data_dir,
        batch_size=8,
        num_workers=2
    )

    # model.load_state_dict(torch.load(r'D:\ly\landsint\result\final_U2NET_model.pth', map_location=torch.device('cpu')))
    model.load_state_dict(torch.load(r'F:\zx\模型结果及参数\final_U2NET_model.pth', map_location=torch.device('cpu')))
    model.eval()

    # 2. 运行评估
    results = predict_and_evaluate(
        model=model,
        test_loader=test_loader,  # 你的测试数据加载器
        device='cpu',
        # save_dir=r'D:\ly\landsint\result\predictions_results_U2net',
        save_dir=r'F:\zx\predictions_results\predictions_results_U2net2',
        multigpu=True
    )


# 运行调试
if __name__ == "__main__":
    main()

