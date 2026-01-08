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

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 核心模块 1: Coordinate Attention (保持0.7593版的成功设计)
# ==========================================
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out


# ==========================================
# 核心模块 2: SE-Fusion (新加入：优化侧边输出的融合)
# 作用：在生成 d0 时，自动给 d1-d6 分配权重
# ==========================================
class SEFusion(nn.Module):
    def __init__(self, in_channel=6, out_channel=1):
        super(SEFusion, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 1)

        # 通道注意力，用于给6个侧边输出打分
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // 2 + 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // 2 + 1, in_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [Batch, 6, H, W] (6个侧边输出的拼接)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # 加权融合
        return self.conv(x * y.expand_as(x))


# ==========================================
# 基础组件
# ==========================================
class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))


def _upsample_like(src, tar):
    return F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=True)


# ==========================================
# RSU 模块 (基于0.7593版，微调Bottleneck)
# ==========================================

class RSU7(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)  # 保持Dilation
        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)
        self.ca = CoordAtt(out_ch, out_ch)  # 保持CoordAtt

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin);
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx);
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx);
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx);
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx);
        hx = self.pool5(hx5)
        hx6 = self.rebnconv6(hx)
        hx7 = self.rebnconv7(hx6)
        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1));
        hx6dup = _upsample_like(hx6d, hx5)
        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1));
        hx5dup = _upsample_like(hx5d, hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1));
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1));
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1));
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        hx1d = self.ca(hx1d)
        return hx1d + hxin


class RSU6(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1);
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1);
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1);
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1);
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)
        self.ca = CoordAtt(out_ch, out_ch)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin);
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx);
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx);
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx);
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx6 = self.rebnconv6(hx5)
        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1));
        hx5dup = _upsample_like(hx5d, hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1));
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1));
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1));
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        hx1d = self.ca(hx1d)
        return hx1d + hxin


class RSU5(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1);
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1);
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1);
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)
        self.ca = CoordAtt(out_ch, out_ch)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin);
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx);
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx);
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx5 = self.rebnconv5(hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1));
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1));
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1));
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        hx1d = self.ca(hx1d)
        return hx1d + hxin


class RSU4(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1);
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1);
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)
        self.ca = CoordAtt(out_ch, out_ch)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin);
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx);
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1));
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1));
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        hx1d = self.ca(hx1d)
        return hx1d + hxin


# ==========================================
# RSU4F_Drop (新修改：在Bottleneck加入Dropout)
# ==========================================
class RSU4F_Drop(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, drop_prob=0.3):
        super(RSU4F_Drop, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)
        self.ca = CoordAtt(out_ch, out_ch)
        # [新增] 正则化，防止深层过拟合
        self.dropout = nn.Dropout2d(p=drop_prob)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)

        # 在最深层特征处使用Dropout
        hx4 = self.dropout(hx4)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))
        hx1d = self.ca(hx1d)
        return hx1d + hxin


# RSU4F 标准版 (用于非瓶颈层)
class RSU4F(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)
        self.ca = CoordAtt(out_ch, out_ch)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))
        hx1d = self.ca(hx1d)
        return hx1d + hxin


# ==========================================
# 终极稳定优化版: U2Net + RSU_CA + Drop + SE_Fusion
# ==========================================
class U2Net_RS_Final(nn.Module):
    def __init__(self, in_ch=4, out_ch=1):
        super(U2Net_RS_Final, self).__init__()

        # Encoder
        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # Stage 5 和 6 (最深层) 使用带 Dropout 的版本，防止过拟合
        self.stage5 = RSU4F_Drop(512, 256, 512, drop_prob=0.2)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage6 = RSU4F_Drop(512, 256, 512, drop_prob=0.3)

        # Decoder (保持 RSU4F 标准版)
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        # Side Outputs
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        # [改进点] 使用 SEFusion 代替普通的 Conv 融合
        # 输入是6个侧边输出的concat (6 * 1 = 6通道)
        self.outconv = SEFusion(in_channel=6 * out_ch, out_channel=out_ch)

    def forward(self, rgb, dem):
        # 1. 拼接
        x = torch.cat([rgb, dem], dim=1)

        # 2. Encoder
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

        # 3. Decoder
        hx6up = _upsample_like(hx6, hx5)
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)
        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # 4. Side Outputs
        d1 = self.side1(hx1d)
        d2 = self.side2(hx2d)
        d3 = self.side3(hx3d)
        d4 = self.side4(hx4d)
        d5 = self.side5(hx5d)
        d6 = self.side6(hx6)

        d2 = _upsample_like(d2, d1)
        d3 = _upsample_like(d3, d1)
        d4 = _upsample_like(d4, d1)
        d5 = _upsample_like(d5, d1)
        d6 = _upsample_like(d6, d1)

        # 5. Fusion (使用SEFusion自动加权)
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return d0

class BoundarySensitiveDiceLoss(nn.Module):
    """
    边界敏感Dice Loss：专门针对滑坡模糊边界优化
    """

    def __init__(self, boundary_width=2, alpha=1.0, beta=2.0, eps=1e-6):
        """
        Args:
            boundary_width: 边界像素宽度
            alpha: 内部区域权重
            beta: 边界区域权重（beta > alpha表示更关注边界）
            eps: 数值稳定性
        """
        super().__init__()
        self.boundary_width = boundary_width
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1, H, W] 网络预测的概率图或logits
            target: [B, 1, H, W] 真实二值掩码
        """
        # 确保值范围在[0, 1]
        if pred.min() < 0 or pred.max() > 1:
            pred_sigmoid = torch.sigmoid(pred)
        else:
            pred_sigmoid = pred

        if target.max() > 1:
            target_binary = (target > 0.5).float()
        else:
            target_binary = target

        # 提取边界区域
        boundary_mask = self.extract_boundary_with_weights(target_binary)

        # 计算加权Dice Loss
        # 边界区域使用beta权重，内部区域使用alpha权重
        weight_map = self.alpha + (self.beta - self.alpha) * boundary_mask

        # 加权交集和并集
        intersection = (weight_map * pred_sigmoid * target_binary).sum(dim=(2, 3))
        union = (weight_map * (pred_sigmoid + target_binary)).sum(dim=(2, 3))

        # Dice系数
        dice = (2. * intersection + self.eps) / (union + self.eps)
        loss = 1 - dice.mean()

        return loss

    def extract_boundary_with_weights(self, mask):
        """
        提取带权重的边界区域
        """
        B, C, H, W = mask.shape

        # 高斯模糊模拟边界不确定性
        blurred = F.avg_pool2d(mask, kernel_size=5, stride=1, padding=2)

        # 计算梯度幅度
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                device=mask.device).view(1, 1, 3, 3).float()
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                device=mask.device).view(1, 1, 3, 3).float()

        grad_x = F.conv2d(blurred, kernel_x, padding=1)
        grad_y = F.conv2d(blurred, kernel_y, padding=1)
        gradient = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

        # 归一化
        gradient_norm = gradient / (gradient.max() + 1e-6)

        # 非线性增强：让边界区域更突出
        boundary_weights = 2 * torch.sigmoid(5 * gradient_norm - 2.5) - 0.5
        boundary_weights = torch.clamp(boundary_weights, 0, 1)

        return boundary_weights


class ConnectivityLoss(nn.Module):
    """
    连接性损失：强制滑坡区域保持空间连续性
    """

    def __init__(self, temperature=0.1, connectivity_weight=2.0):
        super().__init__()
        self.temperature = temperature
        self.connectivity_weight = connectivity_weight

        # 4邻域卷积核
        self.register_buffer('neighbor_kernel',
                             torch.tensor([[[[0, 1, 0],
                                             [1, 0, 1],
                                             [0, 1, 0]]]], dtype=torch.float32))

    def forward(self, pred, target):
        # 二值化
        pred_binary = (torch.sigmoid(pred) > 0.5).float()
        target_binary = (target > 0.5).float()

        B, C, H, W = pred.shape

        # 计算连通组件数量
        pred_components = self.count_connected_components(pred_binary)
        target_components = self.count_connected_components(target_binary)

        # 组件数量差异损失
        component_loss = F.l1_loss(pred_components, target_components)

        # 边界连续性损失
        boundary_continuity_loss = self.boundary_continuity_loss(pred_binary, target_binary)

        # 总连接性损失
        total_loss = (component_loss * 0.6 + boundary_continuity_loss * 0.4)

        return total_loss * self.connectivity_weight

    def count_connected_components(self, binary_mask):
        """快速估算连通组件数量"""
        B = binary_mask.shape[0]
        components = []

        # 简化的组件计数（不需要scipy）
        for b in range(B):
            mask = binary_mask[b, 0]
            # 使用形态学操作近似
            kernel = torch.ones(1, 1, 3, 3, device=binary_mask.device)
            # 膨胀后计算连通区域
            dilated = F.conv2d(mask.unsqueeze(0).unsqueeze(0), kernel, padding=1) > 0
            # 简单估算：通过局部极大值数量
            local_max = F.max_pool2d(dilated.float(), 3, stride=1, padding=1)
            num_components = (local_max == dilated.float()).sum().item()
            components.append(num_components)

        return torch.tensor(components, dtype=torch.float32, device=binary_mask.device)

    def boundary_continuity_loss(self, pred, target):
        """边界连续性损失"""
        # 提取边界
        kernel = torch.ones(1, 1, 3, 3, device=pred.device)
        pred_dilated = F.conv2d(pred, kernel, padding=1) > 0
        pred_eroded = F.conv2d(pred, kernel, padding=1) < kernel.sum()
        pred_boundary = (pred_dilated.float() - pred_eroded.float()).abs()

        # 检查边界连续性
        neighbor_conv = F.conv2d(pred_boundary, self.neighbor_kernel, padding=1)
        has_neighbor = (neighbor_conv > 0).float()

        # 连续性得分
        pred_score = (has_neighbor * pred_boundary).sum() / (pred_boundary.sum() + 1e-6)

        # 对target做相同操作
        target_dilated = F.conv2d(target, kernel, padding=1) > 0
        target_eroded = F.conv2d(target, kernel, padding=1) < kernel.sum()
        target_boundary = (target_dilated.float() - target_eroded.float()).abs()

        neighbor_conv_target = F.conv2d(target_boundary, self.neighbor_kernel, padding=1)
        has_neighbor_target = (neighbor_conv_target > 0).float()
        target_score = (has_neighbor_target * target_boundary).sum() / (target_boundary.sum() + 1e-6)

        return F.l1_loss(pred_score, target_score)


class MultiScaleIoULoss(nn.Module):
    """
    多尺度IoU损失：同时优化不同大小的滑坡区域
    """

    def __init__(self, scales=[0.5, 1.0, 2.0], weights=None):
        super().__init__()
        self.scales = scales

        if weights is None:
            self.weights = torch.tensor([0.4, 0.3, 0.3])
        else:
            self.weights = torch.tensor(weights)

        self.weights = self.weights / self.weights.sum()

    def forward(self, pred, target):
        B, C, H, W = pred.shape

        total_loss = 0.0

        for i, scale in enumerate(self.scales):
            if scale != 1.0:
                new_H, new_W = int(H * scale), int(W * scale)
                pred_scaled = F.interpolate(pred, size=(new_H, new_W),
                                            mode='bilinear', align_corners=True)
                target_scaled = F.interpolate(target, size=(new_H, new_W),
                                              mode='nearest')
            else:
                pred_scaled = pred
                target_scaled = target

            scale_loss = self.single_scale_iou_loss(pred_scaled, target_scaled)
            total_loss += scale_loss * self.weights[i]

        return total_loss

    def single_scale_iou_loss(self, pred, target):
        """单尺度IoU损失"""
        pred_prob = torch.sigmoid(pred)

        intersection = (pred_prob * target).sum(dim=(2, 3))
        union = pred_prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection

        iou = (intersection + 1e-6) / (union + 1e-6)
        iou_loss = 1 - iou.mean()

        return iou_loss


class EnhancedFocalLoss(nn.Module):
    """
    增强版Focal Loss：专门针对滑坡样本不平衡和困难样本
    """

    def __init__(self, alpha=0.75, gamma=3.0, landslide_weight=2.0,
                 hard_sample_threshold=0.3):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.landslide_weight = landslide_weight
        self.hard_sample_threshold = hard_sample_threshold
        self.hard_sample_ratio = 0.0

    def forward(self, pred, target):
        if target.max() > 1:
            target = (target > 0.5).float()

        pred_prob = torch.sigmoid(pred)

        # 基础交叉熵
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        # Focal调制因子
        p_t = target * pred_prob + (1 - target) * (1 - pred_prob)
        focal_weight = (1 - p_t) ** self.gamma

        # 类别平衡权重
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)

        # 滑坡区域额外权重
        with torch.no_grad():
            kernel = torch.ones(1, 1, 3, 3, device=target.device)
            dilated = F.conv2d(target, kernel, padding=1) > 0
            eroded = F.conv2d(target, kernel, padding=1) < kernel.sum()
            boundary_mask = (dilated.float() - eroded.float()).abs()
            landslide_weight_map = 1.0 + (self.landslide_weight - 1.0) * (target + boundary_mask * 0.5)

        # 困难样本识别
        hard_sample_mask = self.identify_hard_samples(pred_prob, target)
        hard_sample_weight = 1.0 + 2.0 * hard_sample_mask

        # 组合权重
        total_weight = focal_weight * alpha_weight * landslide_weight_map * hard_sample_weight

        # 加权损失
        weighted_loss = total_weight * bce_loss

        # 记录困难样本比例
        self.hard_sample_ratio = hard_sample_mask.float().mean().item()

        return weighted_loss.mean()

    def identify_hard_samples(self, pred_prob, target):
        """识别困难样本"""
        pred_error = torch.abs(pred_prob - target)
        hard_low = pred_prob > self.hard_sample_threshold
        hard_high = pred_prob < (1 - self.hard_sample_threshold)
        hard_sample_mask = (hard_low & hard_high).float()

        with torch.no_grad():
            kernel = torch.ones(1, 1, 3, 3, device=target.device)
            target_dilated = F.conv2d(target, kernel, padding=1) > 0
            target_eroded = F.conv2d(target, kernel, padding=1) < kernel.sum()
            boundary_mask = (target_dilated.float() - target_eroded.float()).abs()

        hard_sample_mask = torch.max(hard_sample_mask, boundary_mask)

        return hard_sample_mask


class LandslideOptimizedLoss(nn.Module):
    """完整版的滑坡专用损失函数组合"""

    def __init__(self,
                 boundary_weight=0.4,
                 connect_weight=0.2,
                 multiscale_weight=0.2,
                 focal_weight=0.2,
                 adaptive_weights=True):
        super().__init__()

        # 各组件损失
        self.boundary_dice = BoundarySensitiveDiceLoss(beta=3.0)  # 修复：添加beta参数
        self.shape_loss = ConnectivityLoss(connectivity_weight=2.0)
        self.multiscale_iou = MultiScaleIoULoss(scales=[0.5, 1.0, 2.0])
        self.hard_example_focal = EnhancedFocalLoss(alpha=0.75, gamma=3.0)

        # 初始权重
        self.boundary_weight = boundary_weight
        self.connect_weight = connect_weight
        self.multiscale_weight = multiscale_weight
        self.focal_weight = focal_weight

        # 是否使用自适应权重
        self.adaptive_weights = adaptive_weights

        if adaptive_weights:
            self.learnable_weights = nn.Parameter(torch.ones(4) / 4)

    def forward(self, pred, target):
        # 计算各项损失
        l_boundary = self.boundary_dice(pred, target)
        l_shape = self.shape_loss(pred, target)
        l_multiscale = self.multiscale_iou(pred, target)
        l_focal = self.hard_example_focal(pred, target)

        if self.adaptive_weights:
            weights = F.softmax(self.learnable_weights, dim=0)
            total_loss = (weights[0] * l_boundary +
                          weights[1] * l_shape +
                          weights[2] * l_multiscale +
                          weights[3] * l_focal)

            self.current_weights = weights.detach().cpu().numpy()
        else:
            total_loss = (self.boundary_weight * l_boundary +
                          self.connect_weight * l_shape +
                          self.multiscale_weight * l_multiscale +
                          self.focal_weight * l_focal)

        return total_loss

    def get_loss_breakdown(self, pred, target):
        """获取各项损失的详细数值"""
        with torch.no_grad():
            l_boundary = self.boundary_dice(pred, target).item()
            l_shape = self.shape_loss(pred, target).item()
            l_multiscale = self.multiscale_iou(pred, target).item()
            l_focal = self.hard_example_focal(pred, target).item()

            hard_ratio = self.hard_example_focal.hard_sample_ratio

        return {
            'boundary_loss': l_boundary,
            'connectivity_loss': l_shape,
            'multiscale_iou_loss': l_multiscale,
            'focal_loss': l_focal,
            'hard_sample_ratio': hard_ratio,
            'current_weights': getattr(self, 'current_weights', None)
        }

def get_simple_training_config_enhance():
    """获取简单训练配置"""

    # 1. 创建简单模型
    model = U2Net_RS_Final(in_ch=4, out_ch=1)

    # 2. 使用滑坡专用损失函数（直接替换这里！）
    # ============ 替换开始 ============
    # 原来的：
    # def combined_loss(pred, target):
    #     bce = nn.BCEWithLogitsLoss()(pred, target)
    #     ...
    #     return bce + dice_loss

    # 替换为：
    criterion = LandslideOptimizedLoss(
        boundary_weight=0.4,  # 边界损失权重（重点优化边界）
        connect_weight=0.2,  # 连接性损失权重
        multiscale_weight=0.2,  # 多尺度损失权重
        focal_weight=0.2,  # 困难样本损失权重
        adaptive_weights=True  # 让网络自己学习最佳权重组合
    ) # 重要：必须放到GPU上

    # 包装成函数形式，保持接口一致
    def combined_loss(pred, target):
        """滑坡专用损失函数"""
        return criterion(pred, target)

    # ============ 替换结束 ============

    # 3. 优化器（建议调整为更激进的学习率）
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,  # 可以尝试增加到2e-4
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

def get_simple_training_config():
    """获取简单训练配置"""

    # 1. 创建简单模型
    model = U2Net_RS_Final(in_ch=4, out_ch=1)

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

    model.load_state_dict(torch.load(r'F:\zx\模型结果及参数\U2Net_RS_model.pth', map_location=torch.device('cpu')))

    model.eval()

    # 2. 运行评估
    results = predict_and_evaluate(
        model=model,
        test_loader=test_loader,  # 你的测试数据加载器
        device='cpu',
        save_dir='F:\zx\predictions_results\predictions_results_U2Net_RS',
        multigpu=True
    )


# 运行调试
if __name__ == "__main__":
    main()

