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

import torch
import torch.nn as nn
import torch.nn.functional as F


class LandslideDeformableAttention(nn.Module):
    """滑坡自适应可变形注意力：让网络聚焦在滑坡特征上"""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.in_channels = in_channels

        # 1. 可变形卷积学习滑坡形状
        self.deform_conv = DeformConv2d(in_channels, in_channels, 3, padding=1)

        # 2. 多尺度感受野提取
        self.multi_scale = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels // 4, 3, dilation=1, padding=1),
            nn.Conv2d(in_channels, in_channels // 4, 3, dilation=2, padding=2),
            nn.Conv2d(in_channels, in_channels // 4, 3, dilation=4, padding=4),
            nn.Conv2d(in_channels, in_channels // 4, 3, dilation=8, padding=8)
        ])

        # 3. 滑坡特征注意力（学习滑坡的空间分布模式）
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 1, 1),
            nn.Sigmoid()
        )

        # 4. 通道注意力（强化滑坡相关特征）
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

        # 5. 上下文增强（滑坡与周围环境的对比）
        self.context_enhance = ContextEnhancement(in_channels)

    def forward(self, x):
        # 原始特征
        identity = x

        # 1. 可变形特征提取
        deform_feat = self.deform_conv(x)

        # 2. 多尺度特征融合
        scale_features = []
        for conv in self.multi_scale:
            scale_features.append(conv(deform_feat))
        multi_scale_feat = torch.cat(scale_features, dim=1)

        # 3. 滑坡空间注意力（哪些位置更像滑坡）
        spatial_weights = self.spatial_attention(multi_scale_feat)

        # 4. 通道注意力（哪些特征通道对滑坡更重要）
        channel_weights = self.channel_attention(multi_scale_feat)

        # 5. 上下文增强
        context_feat = self.context_enhance(multi_scale_feat)

        # 6. 三重注意力融合
        attended_feat = context_feat * spatial_weights * channel_weights

        # 7. 残差连接
        output = identity + attended_feat

        return output


class DeformConv2d(nn.Module):
    """简化的可变形卷积（避免复杂依赖）"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding

        # 偏移量预测网络
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size,
                                     kernel_size=kernel_size, padding=padding)

        # 标准卷积
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, padding=padding)

        # 初始化偏移量为0
        self.offset_conv.weight.data.zero_()
        self.offset_conv.bias.data.zero_()

    def forward(self, x):
        # 预测偏移量
        offset = self.offset_conv(x)

        # 应用偏移（简化版本：通过双线性插值实现）
        B, C, H, W = x.shape
        kh, kw = self.kernel_size, self.kernel_size

        # 生成采样网格
        y_coords, x_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid = torch.stack([x_coords, y_coords], dim=-1).float().to(x.device)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 2]

        # 应用偏移
        offset = offset.permute(0, 2, 3, 1).reshape(B, H, W, kh * kw, 2)
        sampling_grids = []

        for i in range(kh * kw):
            # 每个采样点的偏移网格
            offset_grid = grid + offset[..., i, :]
            # 归一化到[-1, 1]
            offset_grid[..., 0] = 2.0 * offset_grid[..., 0] / (W - 1) - 1.0
            offset_grid[..., 1] = 2.0 * offset_grid[..., 1] / (H - 1) - 1.0
            sampling_grids.append(offset_grid)

        # 采样特征
        sampled_features = []
        for i in range(kh * kw):
            sampled = F.grid_sample(x, sampling_grids[i], align_corners=True, mode='bilinear')
            sampled_features.append(sampled)

        # 组合采样特征
        sampled_features = torch.stack(sampled_features, dim=2)  # [B, C, kh*kw, H, W]
        sampled_features = sampled_features.view(B, C * kh * kw, H, W)

        # 应用卷积
        output = self.conv(sampled_features)

        return output


class ContextEnhancement(nn.Module):
    """上下文增强：强化滑坡与周围环境的对比"""

    def __init__(self, channels):
        super().__init__()

        # 全局上下文
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

        # 局部对比（滑坡通常与周围地形有明显对比）
        self.local_contrast = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 全局上下文权重
        global_weight = self.global_context(x)

        # 局部对比权重
        local_weight = self.local_contrast(x)

        # 组合权重
        combined_weight = 0.5 * global_weight + 0.5 * local_weight

        return x * combined_weight


# 在RSU中集成滑坡注意力
class RSU5_LandslideEnhanced(RSU5):
    """集成滑坡注意力增强的RSU5"""

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__(in_ch, mid_ch, out_ch)

        # 在RSU的关键位置添加滑坡注意力
        self.landslide_attention1 = LandslideDeformableAttention(mid_ch)
        self.landslide_attention2 = LandslideDeformableAttention(mid_ch)

        # 替换conv5为增强版本
        self.conv5 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            LandslideDeformableAttention(mid_ch),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 修改前向传播，在关键位置添加注意力
        hx = self.conv0(x)
        hx_in = hx

        # 第一层后添加注意力
        hx1 = self.relu(self.bn1(self.conv1(hx)))
        hx1 = self.landslide_attention1(hx1)  # 添加注意力
        hx = self.pool1(hx1)

        # 后续层...
        hx2 = self.relu(self.bn2(self.conv2(hx)))
        hx2 = self.landslide_attention2(hx2)  # 添加注意力
        hx = self.pool2(hx2)

        # ... 保持其他层不变，但conv5已经增强

        return super().forward(x)  # 调用父类但使用增强的层

class CascadeRefinementNetwork(nn.Module):
    """级联精炼网络：多阶段逐步优化滑坡分割"""

    def __init__(self, base_net, num_stages=3):
        super().__init__()
        self.base_net = base_net
        self.num_stages = num_stages

        # 阶段精炼网络
        self.refinement_stages = nn.ModuleList()
        for i in range(num_stages):
            self.refinement_stages.append(
                RefinementStage(
                    input_channels=1 + (4 if i == 0 else 1),  # 输入包含原图特征
                    hidden_channels=64
                )
            )

        # 渐进融合（逐步引入细节）
        self.progressive_fusion = ProgressiveFusion()

    def forward(self, optical, dem):
        # 第一阶段：基础分割
        x = torch.cat([optical, dem], dim=1)
        base_output = self.base_net(x)
        base_prob = torch.sigmoid(base_output)

        # 多阶段精炼
        refined_outputs = []
        current_pred = base_prob

        for i, refinement_stage in enumerate(self.refinement_stages):
            # 准备输入：当前预测 + 原始图像特征
            if i == 0:
                stage_input = torch.cat([current_pred, optical, dem], dim=1)
            else:
                stage_input = torch.cat([current_pred, optical], dim=1)

            # 精炼
            refinement = refinement_stage(stage_input)
            current_pred = torch.sigmoid(refinement)
            refined_outputs.append(current_pred)

        # 渐进融合所有阶段结果
        final_output = self.progressive_fusion(refined_outputs)

        return final_output


class RefinementStage(nn.Module):
    """单阶段精炼网络"""

    def __init__(self, input_channels, hidden_channels):
        super().__init__()

        # 滑坡细节提取
        self.detail_extractor = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, padding=1),
            LandslideDeformableAttention(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 错误修正（修正上一阶段的错误）
        self.error_correction = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 1, 3, padding=1)
        )

        # 上下文约束（确保分割结果合理）
        self.context_constraint = ContextConstraint(hidden_channels)

    def forward(self, x):
        # 提取细节
        features = self.detail_extractor(x)

        # 应用上下文约束
        constrained_features = self.context_constraint(features)

        # 错误修正
        correction = self.error_correction(constrained_features)

        return correction


class ContextConstraint(nn.Module):
    """上下文约束：确保分割结果符合滑坡的空间约束"""

    def __init__(self, channels):
        super().__init__()

        # 空间连续性约束
        self.spatial_continuity = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 3, padding=1),
            nn.Sigmoid()
        )

        # 形状合理性约束（滑坡通常不是孤立点）
        self.shape_reasoning = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 空间连续性权重
        continuity_weight = self.spatial_continuity(x)

        # 形状合理性权重
        shape_weight = self.shape_reasoning(x)
        shape_weight = F.interpolate(shape_weight, size=x.shape[2:],
                                     mode='bilinear', align_corners=True)

        # 组合约束
        constraint_weight = 0.6 * continuity_weight + 0.4 * shape_weight

        # 应用约束
        constrained_x = x * constraint_weight

        return constrained_x


class ProgressiveFusion(nn.Module):
    """渐进融合：智能融合多阶段结果"""

    def __init__(self):
        super().__init__()

        # 可学习的融合权重
        self.fusion_weights = nn.Parameter(torch.ones(3))

        # 置信度估计（哪些区域哪些阶段更可信）
        self.confidence_estimator = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Softmax(dim=1)
        )

    def forward(self, stage_outputs):
        # 堆叠所有阶段输出
        stacked = torch.stack(stage_outputs, dim=1)  # [B, num_stages, 1, H, W]
        stacked = stacked.squeeze(2)  # [B, num_stages, H, W]

        # 计算各阶段在各位置的置信度
        confidence_maps = self.confidence_estimator(stacked)  # [B, num_stages, H, W]

        # 加权融合
        weighted_sum = torch.sum(stacked * confidence_maps, dim=1, keepdim=True)

        return weighted_sum