import cv2

import random
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast  # 混合精度训练
import numpy as np
from tqdm import tqdm
import os
from predict import create_test_loader_from_csv, predict_and_evaluate



import warnings
warnings.filterwarnings('ignore')


class PatchEmbed(nn.Module):
    """图像分块嵌入"""

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class WindowAttention(nn.Module):
    """窗口注意力机制（修改版）"""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 相对位置偏置表 - 动态计算大小
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        # 生成相对位置索引（一次性计算）
        coords = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords, coords, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 获取相对位置偏置
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

        # 确保形状匹配
        if attn.shape[2:] != relative_position_bias.shape[1:]:
            # 动态调整窗口大小（如果实际窗口与预设不同）
            actual_window_size = int(N ** 0.5)
            if actual_window_size != self.window_size:
                # 重新计算相对位置索引
                coords = torch.arange(actual_window_size)
                coords = torch.stack(torch.meshgrid(coords, coords, indexing='ij'))
                coords_flatten = torch.flatten(coords, 1)
                relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
                relative_coords = relative_coords.permute(1, 2, 0).contiguous()
                relative_coords[:, :, 0] += actual_window_size - 1
                relative_coords[:, :, 1] += actual_window_size - 1
                relative_coords[:, :, 0] *= 2 * actual_window_size - 1
                relative_position_index = relative_coords.sum(-1)

                # 重新计算偏置
                relative_position_bias = self.relative_position_bias_table[
                    relative_position_index.view(-1)].view(
                    actual_window_size * actual_window_size,
                    actual_window_size * actual_window_size, -1)
                relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

        attn = attn + relative_position_bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer块"""

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)

        # 如果需要shift窗口
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # 分区窗口
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)  # nW*B, window_size*window_size, C

        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # 反向shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class Mlp(nn.Module):
    """MLP模块"""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    将特征图划分为窗口（支持任意尺寸）
    Args:
        x: (B, H, W, C)
        window_size (int): 窗口大小
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape

    # 如果尺寸不能被窗口大小整除，进行填充
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))  # 在H和W维度上填充
        H, W = H + pad_h, W + pad_w

    # 计算窗口数量
    num_windows_h = H // window_size
    num_windows_w = W // window_size

    # 重新塑造为窗口
    x = x.view(B, num_windows_h, window_size, num_windows_w, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

    return windows, (H, W), (pad_h, pad_w)


def window_reverse(windows, window_size, H, W, pad_h, pad_w):
    """
    将窗口恢复为特征图（处理填充）
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): 窗口大小
        H (int): 原始高度
        W (int): 原始宽度
        pad_h (int): 高度方向填充
        pad_w (int): 宽度方向填充
    Returns:
        x: (B, H, W, C)
    """
    H_padded = H + pad_h
    W_padded = W + pad_w
    num_windows_h = H_padded // window_size
    num_windows_w = W_padded // window_size

    B = int(windows.shape[0] / (num_windows_h * num_windows_w))

    x = windows.view(B, num_windows_h, num_windows_w, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_padded, W_padded, -1)

    # 去除填充
    if pad_h > 0 or pad_w > 0:
        x = x[:, :H, :W, :].contiguous()

    return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth)"""

    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class PatchMerging(nn.Module):
    """下采样：2倍降采样，维度翻倍"""

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        # 输出维度是输入维度的2倍
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H, W, C
        返回: B, H/2, W/2, 2C
        """
        B, H, W, C = x.shape

        # 确保尺寸是偶数
        if H % 2 != 0 or W % 2 != 0:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
            H = H + (H % 2)
            W = W + (W % 2)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)  # B H/2 W/2 2*C

        return x

# 纯Swin Transformer分割模型（完全离线）
class PureSwinSegmentation(nn.Module):
    """纯Swin Transformer语义分割模型 - 完全离线版"""

    def __init__(self, n_channels=4, n_classes=1, swin_type='tiny'):
        super().__init__()

        # 配置参数
        configs = {
            'tiny': {
                'embed_dim': 96,
                'depths': [2, 2, 6, 2],
                'num_heads': [3, 6, 12, 24],
                'window_size': 7,
                'drop_path_rate': 0.2
            },
            'small': {
                'embed_dim': 96,
                'depths': [2, 2, 18, 2],
                'num_heads': [3, 6, 12, 24],
                'window_size': 7,
                'drop_path_rate': 0.3
            },
            'base': {
                'embed_dim': 128,
                'depths': [2, 2, 18, 2],
                'num_heads': [4, 8, 16, 32],
                'window_size': 7,
                'drop_path_rate': 0.5
            }
        }

        config = configs[swin_type]
        self.embed_dim = config['embed_dim']
        self.depths = config['depths']
        self.num_heads = config['num_heads']
        self.window_size = config['window_size']

        # 重要修正：Swin Transformer的实际输出通道数
        # 注意：每个BasicLayer（除了最后一个）内部有PatchMerging，会将维度翻倍
        # 所以实际的输出通道数序列应该是：
        # 输入: embed_dim
        # stage1输出: embed_dim * 2 (如果stage1有downsample)
        # stage2输出: embed_dim * 4 (如果stage2有downsample)
        # stage3输出: embed_dim * 8 (如果stage3有downsample)
        # stage4输出: embed_dim * 8 (最后一个stage没有downsample)

        # 计算每个阶段的输出维度
        self.stage_channels = []
        current_dim = self.embed_dim

        for i, depth in enumerate(self.depths):
            # 除了最后一个阶段，其他阶段都有PatchMerging会将维度翻倍
            if i < len(self.depths) - 1:
                output_dim = current_dim * 2
            else:
                output_dim = current_dim
            self.stage_channels.append(output_dim)

            # 更新下一个阶段的输入维度
            current_dim = output_dim

        print(f"各阶段输出通道数: {self.stage_channels}")

        # 1. 输入适配层
        self.input_adapter = nn.Sequential(
            nn.Conv2d(n_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        # 2. Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=224,
            patch_size=4,
            in_chans=3,
            embed_dim=self.embed_dim,
            norm_layer=nn.LayerNorm
        )

        # 3. Swin Transformer阶段
        self.num_layers = len(self.depths)
        dpr = [x.item() for x in torch.linspace(0, config['drop_path_rate'], sum(self.depths))]

        self.layers = nn.ModuleList()
        current_dim = self.embed_dim

        for i_layer in range(self.num_layers):
            print(
                f"构建第{i_layer + 1}阶段: input_dim={current_dim}, depth={self.depths[i_layer]}, heads={self.num_heads[i_layer]}, "
                f"output_dim={self.stage_channels[i_layer]}")

            # 这个阶段是否有下采样
            downsample = PatchMerging if (i_layer < self.num_layers - 1) else None

            layer = BasicLayer(
                dim=current_dim,  # 输入维度
                depth=self.depths[i_layer],
                num_heads=self.num_heads[i_layer],
                window_size=self.window_size,
                drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                downsample=downsample
            )
            self.layers.append(layer)

            # 更新下一个阶段的输入维度为当前阶段的输出维度
            current_dim = self.stage_channels[i_layer]

        # 4. 特征金字塔网络 - 使用实际计算的通道数
        self.fpn = FPNModule(
            in_channels=self.stage_channels,  # 使用实际计算出的通道数
            out_channels=256
        )

        # 5. 分割头
        self.seg_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, n_classes, 1)
        )

        self.apply(self._init_weights)

        print(f"\n初始化离线Swin Transformer模型 ({swin_type}):")
        print(f"  输入通道: {n_channels}")
        print(f"  输出类别: {n_classes}")
        print(f"  嵌入维度: {self.embed_dim}")
        print(f"  网络深度: {self.depths}")
        print(f"  各阶段输出通道数: {self.stage_channels}")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, optical, dem):
        # 1. 融合输入
        x = torch.cat([optical, dem], dim=1)
        input_size = x.shape[2:]

        # 2. 输入适配
        x = self.input_adapter(x)

        # 3. Patch Embedding
        x = self.patch_embed(x)  # B, N, C

        # 4. Swin Transformer层
        features = []
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.view(B, H, W, C)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            # 将特征保存为[B, C, H, W]格式
            B, H, W, C = x.shape
            features.append(x.permute(0, 3, 1, 2).contiguous())

        # 5. FPN融合特征
        fused = self.fpn(features)

        # 6. 上采样到输入尺寸
        output = F.interpolate(fused, size=input_size,
                               mode='bilinear', align_corners=True)

        # 7. 分割头
        output = self.seg_head(output)

        return output


class BasicLayer(nn.Module):
    """Swin Transformer基本层（简化版）"""

    def __init__(self, dim, depth, num_heads, window_size,
                 drop_path=0., downsample=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.downsample = downsample

        # 构建块
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path
            )
            for i in range(depth)
        ])

        # 下采样层（如果有）
        if downsample is not None:
            self.downsample_layer = downsample(dim=dim)
        else:
            self.downsample_layer = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        if self.downsample_layer is not None:
            x = self.downsample_layer(x)

        return x

class FPNModule(nn.Module):
    """特征金字塔网络（修复版）"""

    def __init__(self, in_channels, out_channels=256):
        """
        参数:
            in_channels: 各层输入通道数列表，例如 [96, 192, 384, 768]
            out_channels: 输出通道数
        """
        super().__init__()

        self.in_channels = in_channels

        # 横向连接：将各层特征映射到统一维度
        self.lateral_convs = nn.ModuleList()
        for i, in_channel in enumerate(in_channels):
            print(f"FPN 第{i + 1}层: in={in_channel}, out={out_channels}")
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

        # 融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        print(f"FPN初始化完成 - 输入通道: {in_channels}, 输出通道: {out_channels}")

    def forward(self, features):
        """
        前向传播

        参数:
            features: 特征列表 [feat1, feat2, feat3, feat4]
                    每个特征的形状: [B, C_i, H_i, W_i]
        """
        # 检查特征数量是否与通道数匹配
        if len(features) != len(self.in_channels):
            print(f"警告: 特征数量({len(features)})与通道数({len(self.in_channels)})不匹配")
            print(f"特征通道数: {[f.shape[1] for f in features]}")
            print(f"期望通道数: {self.in_channels}")

            # 自适应调整：取最匹配的层数
            min_len = min(len(features), len(self.in_channels))
            features = features[:min_len]
            print(f"调整后使用前{min_len}层特征")

        # 自顶向下的特征融合
        last_idx = len(features) - 1

        # 如果特征数量少于预期的层数，使用最后一个可用的特征
        if last_idx >= len(self.lateral_convs):
            last_idx = len(self.lateral_convs) - 1

        fused_feature = self.lateral_convs[last_idx](features[last_idx])

        # 自顶向下的特征融合
        for i in range(len(features) - 2, -1, -1):
            if i >= len(self.lateral_convs):
                continue

            # 上采样到与当前层相同的分辨率
            target_size = features[i].shape[2:]
            fused_feature = F.interpolate(
                fused_feature,
                size=target_size,
                mode='bilinear',
                align_corners=True
            )

            # 横向连接
            lateral_feature = self.lateral_convs[i](features[i])

            # 特征融合（逐元素相加）
            fused_feature = fused_feature + lateral_feature

        # 最终的融合卷积
        fused_feature = self.fusion_conv(fused_feature)

        return fused_feature

class SwinTransformerBlock(nn.Module):
    """Swin Transformer块（修改版）"""

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)

        # 计算实际窗口大小（适应输入尺寸）
        actual_window_size = min(self.window_size, H, W)

        # 如果需要shift窗口
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # 分区窗口（带填充处理）
        windows, padded_size, padding = window_partition(shifted_x, actual_window_size)
        windows = windows.view(-1, actual_window_size * actual_window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(windows)

        # 合并窗口（考虑填充）
        attn_windows = attn_windows.view(-1, actual_window_size, actual_window_size, C)
        shifted_x = window_reverse(attn_windows, actual_window_size, H, W, *padding)

        # 反向shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

def get_simple_training_config():
    """获取简单训练配置"""

    # 1. 创建简单模型
    model = PureSwinSegmentation(
        n_channels=4,
        n_classes=1,
        swin_type='base'  # 或 'small', 'base'
    )

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



    model.load_state_dict(torch.load(r'F:\zx\模型结果及参数\final_SwinTransformer_model.pth', map_location=torch.device('cpu')))
    model.eval()

    # 2. 运行评估
    results = predict_and_evaluate(
        model=model,
        test_loader=test_loader,  # 你的测试数据加载器
        device='cpu',
        save_dir=r'F:\zx\predictions_results\predictions_results_SwinTransformer',
        multigpu=True
    )


# 运行调试
if __name__ == "__main__":
    main()

