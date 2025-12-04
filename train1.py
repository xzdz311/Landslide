# ==============================================================
# HeteroFusion-SegNet (无 rasterio 依赖版本)
# 专为 Kaggle 环境优化
# ==============================================================

# 2. 导入依赖 (移除 rasterio 和 smp)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import random
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2


# 3. 设置和工具函数
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# 4. DEM 读取函数 (OpenCV 替代 rasterio)
def read_dem(file_path):
    """使用 OpenCV 读取 DEM 文件，兼容多种格式"""
    dem = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

    # 尝试替代扩展名
    if dem is None:
        base, ext = os.path.splitext(file_path)
        possible_exts = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
        for new_ext in possible_exts:
            alt_path = base + new_ext
            if os.path.exists(alt_path):
                dem = cv2.imread(alt_path, cv2.IMREAD_UNCHANGED)
                if dem is not None:
                    break

    # 处理读取失败
    if dem is None:
        return np.zeros((256, 256), dtype=np.float32)

    # 确保单通道
    if len(dem.shape) == 3:
        dem = dem[:, :, 0]  # 取第一个通道

    return dem.astype(np.float32)


# 5. 数据集类 (完全移除 rasterio 依赖)
class LandslideDataset(Dataset):
    def __init__(self, image_paths, dem_paths, mask_paths=None, transform=None, target_size=(256, 256)):
        self.image_paths = [p for p in image_paths if p is not None]
        self.dem_paths = [p for p in dem_paths if p is not None]

        # 确保mask_paths列表长度与image_paths一致
        if mask_paths is None:
            self.mask_paths = [None] * len(self.image_paths)
        else:
            self.mask_paths = mask_paths

        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 读取光学图像
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((*self.target_size[::-1], 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 读取DEM
        dem_path = self.dem_paths[idx]
        dem = read_dem(dem_path)

        # 读取或创建掩膜
        mask_path = self.mask_paths[idx]
        if mask_path is not None and os.path.exists(mask_path):
            # 有真实mask：读取并二值化
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros(self.target_size, dtype=np.uint8)
            else:
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        else:
            # 无滑坡样本：创建全0的mask
            mask = np.zeros(self.target_size, dtype=np.uint8)

        # 调整大小
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
        dem = cv2.resize(dem, self.target_size, interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        # 归一化DEM
        dem_min, dem_max = dem.min(), dem.max()
        if dem_max > dem_min:
            dem = (dem - dem_min) / (dem_max - dem_min + 1e-8)
        else:
            dem = np.zeros_like(dem)

        # 合并为4通道
        dem = np.expand_dims(dem, axis=-1)
        combined = np.concatenate([img, dem], axis=-1)

        # 应用变换
        if self.transform:
            augmented = self.transform(image=combined, mask=mask)
            combined = augmented['image']
            mask = augmented['mask']

        # 分离通道
        optical = combined[:3, :, :]
        dem = combined[3:, :, :]
        mask = mask.unsqueeze(0).float() / 255.0

        # 添加一个标志位：是否为滑坡样本
        is_landslide = 1.0 if self.mask_paths[idx] is not None else 0.0

        return optical, dem, mask, is_landslide, img_path


# 6. 数据准备函数 (保持不变，但移除 rasterio 依赖)
def prepare_datasets_with_masks(data_dir, target_size=(256, 256), test_size=0.2):
    """
    准备训练和验证数据集

    Args:
        data_dir: 数据集根目录
        target_size: 目标图像尺寸
        test_size: 验证集比例
    """
    # 硬编码所有路径 - 根据实际目录结构
    landslide_train_image_dir = os.path.join(data_dir, 'landslide', 'train', 'image')  # 单数
    landslide_train_mask_dir = os.path.join(data_dir, 'landslide', 'train', 'mask')
    landslide_train_dem_dir = os.path.join(data_dir, 'landslide', 'train', 'dem')

    landslide_test_image_dir = os.path.join(data_dir, 'landslide', 'test', 'images')  # 复数
    landslide_test_mask_dir = os.path.join(data_dir, 'landslide', 'test', 'mask')
    landslide_test_dem_dir = os.path.join(data_dir, 'landslide', 'test', 'dem')

    non_landslide_train_image_dir = os.path.join(data_dir, 'non-landslide', 'train', 'images')  # 复数
    non_landslide_train_dem_dir = os.path.join(data_dir, 'non-landslide', 'train', 'dem')

    non_landslide_test_image_dir = os.path.join(data_dir, 'non-landslide', 'test', 'images')  # 复数
    non_landslide_test_dem_dir = os.path.join(data_dir, 'non-landslide', 'test', 'dem')

    # 收集滑坡训练数据
    landslide_train_imgs = []
    landslide_train_dems = []
    landslide_train_masks = []

    for img_file in os.listdir(landslide_train_image_dir):
        if img_file.lower().endswith('.png'):
            img_path = os.path.join(landslide_train_image_dir, img_file)
            dem_path = os.path.join(landslide_train_dem_dir, img_file)
            mask_path = os.path.join(landslide_train_mask_dir, img_file)

            if os.path.exists(img_path) and os.path.exists(dem_path) and os.path.exists(mask_path):
                landslide_train_imgs.append(img_path)
                landslide_train_dems.append(dem_path)
                landslide_train_masks.append(mask_path)

    # 收集滑坡测试数据
    landslide_test_imgs = []
    landslide_test_dems = []
    landslide_test_masks = []

    for img_file in os.listdir(landslide_test_image_dir):
        if img_file.lower().endswith('.png'):
            img_path = os.path.join(landslide_test_image_dir, img_file)
            dem_path = os.path.join(landslide_test_dem_dir, img_file)
            mask_path = os.path.join(landslide_test_mask_dir, img_file)

            if os.path.exists(img_path) and os.path.exists(dem_path) and os.path.exists(mask_path):
                landslide_test_imgs.append(img_path)
                landslide_test_dems.append(dem_path)
                landslide_test_masks.append(mask_path)

    # 收集非滑坡训练数据
    nonlandslide_train_imgs = []
    nonlandslide_train_dems = []

    for img_file in os.listdir(non_landslide_train_image_dir):
        if img_file.lower().endswith('.png'):
            img_path = os.path.join(non_landslide_train_image_dir, img_file)
            dem_path = os.path.join(non_landslide_train_dem_dir, img_file)

            if os.path.exists(img_path) and os.path.exists(dem_path):
                nonlandslide_train_imgs.append(img_path)
                nonlandslide_train_dems.append(dem_path)

    # 收集非滑坡测试数据
    nonlandslide_test_imgs = []
    nonlandslide_test_dems = []

    for img_file in os.listdir(non_landslide_test_image_dir):
        if img_file.lower().endswith('.png'):
            img_path = os.path.join(non_landslide_test_image_dir, img_file)
            dem_path = os.path.join(non_landslide_test_dem_dir, img_file)

            if os.path.exists(img_path) and os.path.exists(dem_path):
                nonlandslide_test_imgs.append(img_path)
                nonlandslide_test_dems.append(dem_path)

    # 合并正负样本
    train_imgs = landslide_train_imgs + nonlandslide_train_imgs
    train_dems = landslide_train_dems + nonlandslide_train_dems
    # 关键修改：无滑坡样本的mask_path设为None，数据集类会自动创建全0 mask
    train_masks = landslide_train_masks + [None] * len(nonlandslide_train_imgs)

    test_imgs = landslide_test_imgs + nonlandslide_test_imgs
    test_dems = landslide_test_dems + nonlandslide_test_dems
    test_masks = landslide_test_masks + [None] * len(nonlandslide_test_imgs)

    print(f"训练集: {len(train_imgs)} 个样本")
    print(f"  - 有滑坡: {len(landslide_train_imgs)} ({len(landslide_train_imgs) / len(train_imgs) * 100:.1f}%)")
    print(f"  - 无滑坡: {len(nonlandslide_train_imgs)} ({len(nonlandslide_train_imgs) / len(train_imgs) * 100:.1f}%)")

    print(f"测试集: {len(test_imgs)} 个样本")
    print(f"  - 有滑坡: {len(landslide_test_imgs)} ({len(landslide_test_imgs) / len(test_imgs) * 100:.1f}%)")
    print(f"  - 无滑坡: {len(nonlandslide_test_imgs)} ({len(nonlandslide_test_imgs) / len(test_imgs) * 100:.1f}%)")

    # 数据增强配置（保持不变）
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=10.0, p=0.5),
            A.MotionBlur(blur_limit=3, p=0.5),
        ], p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406, 0.5), std=(0.229, 0.224, 0.225, 0.25)),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406, 0.5), std=(0.229, 0.224, 0.225, 0.25)),
        ToTensorV2()
    ])

    # 创建数据集
    train_dataset = LandslideDataset(
        image_paths=train_imgs,
        dem_paths=train_dems,
        mask_paths=train_masks,
        transform=train_transform,
        target_size=target_size
    )

    test_dataset = LandslideDataset(
        image_paths=test_imgs,
        dem_paths=test_dems,
        mask_paths=test_masks,
        transform=val_transform,
        target_size=target_size
    )

    return train_dataset, test_dataset

# ==================== 新增：替代smp模块的组件 ====================


class UnetDecoder(nn.Module):
    """替代smp的UnetDecoder - 简化版本"""

    def __init__(self, encoder_channels, decoder_channels, n_blocks=5,
                 use_batchnorm=True, center=False):
        super().__init__()

        encoder_channels = encoder_channels[1:]  # 跳过第一个0

        # 创建解码器块
        self.blocks = nn.ModuleList()

        for i in range(n_blocks):
            if i == 0:
                # 第一层：使用编码器的最后特征
                in_channels = encoder_channels[-1]
            else:
                # 后续层：使用上一层的输出
                in_channels = decoder_channels[i - 1]

            out_channels = decoder_channels[i]

            # 上采样层
            upsample = nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=2, stride=2
            )

            # 卷积块
            conv_block = self._make_conv_block(
                out_channels,  # 简化为直接卷积
                out_channels,
                use_batchnorm=use_batchnorm
            )

            self.blocks.append(nn.ModuleDict({
                'upsample': upsample,
                'conv_block': conv_block
            }))

    def _make_conv_block(self, in_channels, out_channels, use_batchnorm=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True)
        ]

        return nn.Sequential(*layers)

    def forward(self, *features):
        """
        features: 编码器特征列表，从深层到浅层
        """
        if not features:
            raise ValueError("No features provided to decoder")

        # 特征应该是 [deepest, ..., shallowest]
        x = features[0]  # 最深层特征

        outputs = []
        for i, block in enumerate(self.blocks):
            # 上采样
            x = block['upsample'](x)

            # 如果有跳跃连接
            if i + 1 < len(features):
                # 调整尺寸（如果需要）
                if x.shape[-2:] != features[i + 1].shape[-2:]:
                    x = F.interpolate(x, size=features[i + 1].shape[-2:], mode='bilinear', align_corners=True)

            # 卷积块
            x = block['conv_block'](x)
            outputs.append(x)

        return x  # 返回最后一层的输出


# ==================== 原有模型组件 ====================

import torch
import torch.nn as nn
from torchvision.models import resnet34, resnet50
from torchvision.models._utils import IntermediateLayerGetter
import os


def create_encoder(encoder_name='resnet34', in_channels=3, depth=5,
                   weights_path=None, use_pretrained=True):
    """
    手动创建编码器
    Args:
        encoder_name: 编码器名称
        in_channels: 输入通道数
        depth: 编码器深度
        weights_path: 本地权重文件路径
        use_pretrained: 是否使用预训练权重
    """
    if encoder_name == 'resnet34':
        try:
            if use_pretrained and weights_path and os.path.exists(weights_path):
                # 方法1：从本地文件加载预训练权重
                print(f"Loading pretrained weights from: {weights_path}")
                backbone = resnet34(pretrained=False)
                state_dict = torch.load(weights_path)
                backbone.load_state_dict(state_dict)
            elif use_pretrained:
                # 方法2：尝试在线下载（可能会失败）
                try:
                    from torchvision.models import ResNet34_Weights
                    backbone = resnet34(weights=ResNet34_Weights.DEFAULT)
                except Exception as e:
                    print(f"Online download failed: {e}. Using random initialization.")
                    backbone = resnet34(pretrained=False)
            else:
                # 方法3：随机初始化
                backbone = resnet34(pretrained=False)
        except Exception as e:
            print(f"Error loading model: {e}. Using random initialization.")
            backbone = resnet34(pretrained=False)

        # 修改第一层卷积以适应不同输入通道数
        if in_channels != 3:
            if hasattr(backbone, 'conv1'):
                original_conv1 = backbone.conv1
                backbone.conv1 = nn.Conv2d(
                    in_channels,
                    64,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False
                )
                # 初始化新卷积层（重要！）
                if use_pretrained and in_channels <= 3:
                    # 只复制前in_channels个通道的权重
                    backbone.conv1.weight.data = original_conv1.weight.data[:, :in_channels, :, :]
                elif use_pretrained and in_channels > 3:
                    # 多通道：重复使用前3个通道的权重
                    repeat_times = (in_channels + 2) // 3  # 向上取整
                    backbone.conv1.weight.data = original_conv1.weight.data.repeat(
                        1, repeat_times, 1, 1)[:, :in_channels, :, :]
                else:
                    # 随机初始化
                    nn.init.kaiming_normal_(backbone.conv1.weight, mode='fan_out', nonlinearity='relu')

        # 提取中间层特征（对应不同深度）
        return_layers = {
            'layer1': '0',  # 1/4
            'layer2': '1',  # 1/8
            'layer3': '2',  # 1/16
            'layer4': '3',  # 1/32
        }

        # 根据depth参数调整返回的层
        if depth <= 1:
            return_layers = {}
        elif depth == 2:
            return_layers = {'layer1': '0'}
        elif depth == 3:
            return_layers = {'layer1': '0', 'layer2': '1'}
        elif depth == 4:
            return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2'}

        return IntermediateLayerGetter(backbone, return_layers=return_layers)

    elif encoder_name == 'resnet50':
        # 类似地实现resnet50
        try:
            if use_pretrained and weights_path and os.path.exists(weights_path):
                print(f"Loading pretrained weights from: {weights_path}")
                backbone = resnet50(pretrained=False)
                state_dict = torch.load(weights_path)
                backbone.load_state_dict(state_dict)
            elif use_pretrained:
                try:
                    from torchvision.models import ResNet50_Weights
                    backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
                except:
                    backbone = resnet50(pretrained=False)
            else:
                backbone = resnet50(pretrained=False)
        except:
            backbone = resnet50(pretrained=False)

        # ... 类似修改输入通道
        if in_channels != 3:
            if hasattr(backbone, 'conv1'):
                original_conv1 = backbone.conv1
                backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7,
                                           stride=2, padding=3, bias=False)
                if use_pretrained and in_channels <= 3:
                    backbone.conv1.weight.data = original_conv1.weight.data[:, :in_channels, :, :]
                elif use_pretrained and in_channels > 3:
                    repeat_times = (in_channels + 2) // 3
                    backbone.conv1.weight.data = original_conv1.weight.data.repeat(
                        1, repeat_times, 1, 1)[:, :in_channels, :, :]
                else:
                    nn.init.kaiming_normal_(backbone.conv1.weight, mode='fan_out', nonlinearity='relu')

        return_layers = {
            'layer1': '0',
            'layer2': '1',
            'layer3': '2',
            'layer4': '3',
        }

        if depth <= 1:
            return_layers = {}
        elif depth == 2:
            return_layers = {'layer1': '0'}
        elif depth == 3:
            return_layers = {'layer1': '0', 'layer2': '1'}
        elif depth == 4:
            return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2'}

        return IntermediateLayerGetter(backbone, return_layers=return_layers)

    else:
        raise ValueError(f"Unsupported encoder: {encoder_name}")


# 5. 模型架构: HeteroFusion-SegNet
class TopographicGradientConv(nn.Module):
    """地形梯度卷积: 显式计算坡度/曲率"""

    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        self.in_channels = in_channels

        # Sobel算子用于计算梯度
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

        # 初始化Sobel算子
        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        self.sobel_x.weight = nn.Parameter(sobel_kernel_x, requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_kernel_y, requires_grad=False)

        # 卷积层：输入通道为 in_channels + 1（原始通道 + 梯度通道）
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + 1, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: [B, C, H, W]
        # 假设最后一个通道是DEM（C=1时就是DEM本身）
        if self.in_channels == 1:
            dem = x  # 单通道，直接使用
        else:
            dem = x[:, -1:, :, :]  # 多通道时取最后一个通道作为DEM

        # 计算梯度
        grad_x = self.sobel_x(dem)
        grad_y = self.sobel_y(dem)
        gradient = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        # 将梯度作为额外特征
        x_enhanced = torch.cat([x, gradient], dim=1)  # 通道数: C + 1
        return self.conv(x_enhanced)

class FrequencyEnhancedConv(nn.Module):
    """频域增强卷积: 突出高频纹理变化"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 高通滤波器 (拉普拉斯)
        self.laplacian = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel = kernel.repeat(in_channels, 1, 1, 1)
        self.laplacian.weight = nn.Parameter(kernel, requires_grad=False)

    def forward(self, x):
        # 应用高通滤波突出边缘
        high_freq = self.laplacian(x)
        x_enhanced = x + high_freq
        return self.conv(x_enhanced)


class DynamicGatedCrossAttention(nn.Module):
    """动态门控交叉注意力融合模块 (DGCAF)"""

    def __init__(self, channels):
        super().__init__()
        self.query_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, optical_feat, dem_feat, optical_conf, dem_conf):
        """
        Args:
            optical_feat: 光学特征 [B, C, H, W]
            dem_feat: DEM特征 [B, C, H, W] - 必须与optical_feat通道数相同
            optical_conf: 光学置信图 [B, 1, H, W]
            dem_conf: DEM置信图 [B, 1, H, W]
        """
        B, C, H, W = optical_feat.size()

        # 确保dem_feat与optical_feat通道数相同
        if dem_feat.shape[1] != C:
            # 如果通道数不同，使用1x1卷积调整
            if not hasattr(self, 'dem_adjust'):
                self.dem_adjust = nn.Conv2d(dem_feat.shape[1], C, kernel_size=1).to(dem_feat.device)
            dem_feat = self.dem_adjust(dem_feat)

        # 计算动态门控
        G = self.sigmoid(self.alpha * optical_conf + self.beta * dem_conf)  # [B, 1, H, W]

        # 交叉注意力 (optical as query, dem as key/value)
        proj_query = self.query_conv(optical_feat).view(B, -1, H * W).permute(0, 2, 1)  # [B, N, C//8]
        proj_key = self.key_conv(dem_feat).view(B, -1, H * W)  # [B, C//8, N]
        energy = torch.bmm(proj_query, proj_key)  # [B, N, N]
        attention = F.softmax(energy, dim=-1)

        proj_value = self.value_conv(dem_feat).view(B, -1, H * W)  # [B, C, N]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # [B, C, N]
        out = out.view(B, C, H, W)

        # 应用门控
        fused = G * (self.gamma * out + (1 - self.gamma) * (optical_feat + dem_feat)) + (1 - G) * (
                optical_feat + dem_feat)
        return fused


class HeteroFusionSegNet(nn.Module):
    def __init__(self, encoder_name='resnet34', in_channels=4, classes=1):
        super().__init__()

        # 1. 光学编码器（不变）
        weight_path = '/kaggle/input/resnet/pytorch/default/1/resnet34-b627a593.pth'
        self.optical_encoder = create_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights_path=weight_path,
        )

        # 2. 改进的DEM编码器（多尺度）
        self.dem_encoder = nn.ModuleList([
            # 第一层
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            ),
            # 第二层
            nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            ),
            # 第三层
            nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),
            # 第四层
            nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )
        ])

        # 3. 特征融合模块（保持不变）
        self.fusion_modules = nn.ModuleList()
        feature_channels = [64, 128, 256, 512]
        for channels in feature_channels:
            self.fusion_modules.append(DynamicGatedCrossAttention(channels))

    def forward(self, optical, dem):
        # 1. 提取光学特征
        optical_features_dict = self.optical_encoder(optical)
        optical_features = []
        layer_keys = ['0', '1', '2', '3']
        for key in layer_keys:
            if key in optical_features_dict:
                optical_features.append(optical_features_dict[key])

        # 2. 提取DEM多尺度特征
        dem_features = []
        x = dem
        for encoder_layer in self.dem_encoder:
            x = encoder_layer(x)
            dem_features.append(x)

        # 3. 融合特征（逐层对应）
        fused_features = []
        for i, (opt_feat, dem_feat, fusion_module) in enumerate(
                zip(optical_features, dem_features, self.fusion_modules)):
            # 调整DEM特征尺寸以匹配光学特征
            if dem_feat.shape[-2:] != opt_feat.shape[-2:]:
                dem_feat = F.interpolate(
                    dem_feat,
                    size=opt_feat.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )

            # 如果通道数不匹配，用1x1卷积调整
            if dem_feat.shape[1] != opt_feat.shape[1]:
                if not hasattr(self, f'dem_adapter_{i}'):
                    setattr(self, f'dem_adapter_{i}',
                            nn.Conv2d(dem_feat.shape[1], opt_feat.shape[1], kernel_size=1).to(dem.device))
                dem_feat = getattr(self, f'dem_adapter_{i}')(dem_feat)

            # 简化：先使用直接拼接或相加
            fused_feat = fusion_module(opt_feat, dem_feat, None, None)  # 先去掉置信度
            fused_features.append(fused_feat)

# 6. 损失函数: 包含滑坡几何一致性损失
class LandslideLoss(nn.Module):
    def __init__(self, geo_weight=0.2):
        super().__init__()
        self.dice_loss = DiceLoss(mode='binary')  # 使用我们自己的DiceLoss
        self.bce_loss = nn.BCELoss()
        self.geo_weight = geo_weight

    def forward(self, outputs, targets, dem=None):
        """
        Args:
            outputs: 模型输出字典
            targets: 真实掩膜 [B, 1, H, W]
            dem: DEM数据 (用于几何一致性)
        """
        # 1. 区域分割损失
        area_loss = self.dice_loss(outputs['area'], targets) + self.bce_loss(outputs['area'], targets)

        # 2. 边界损失 (使用Sobel边缘检测作为监督)
        with torch.no_grad():
            # 计算目标边界
            target_edges = self.sobel_edge_detection(targets)

        boundary_loss = self.bce_loss(outputs['boundary'], target_edges)

        # 3. 几何一致性损失 (简化版)
        geo_loss = 0.0
        if dem is not None and self.geo_weight > 0:
            geo_loss = self.geometric_consistency_loss(outputs['mask'], dem)

        # 总损失
        total_loss = area_loss + 0.5 * boundary_loss + self.geo_weight * geo_loss
        return total_loss

    def sobel_edge_detection(self, x):
        """使用Sobel算子检测边缘"""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3,
                                                                                                                3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3,
                                                                                                                3)

        grad_x = F.conv2d(x, sobel_x.repeat(1, 1, 1, 1), padding=1)
        grad_y = F.conv2d(x, sobel_y.repeat(1, 1, 1, 1), padding=1)
        edges = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        edges = (edges > 0.1).float()  # 二值化
        return edges

    def geometric_consistency_loss(self, pred, dem):
        """简化版几何一致性损失"""
        # 计算预测掩膜的梯度
        pred_grad = self.sobel_edge_detection(pred)

        # 计算DEM的梯度 (坡度)
        dem_grad = self.sobel_edge_detection(dem)

        # 简单相关性损失 (方向应一致)
        cos_sim = F.cosine_similarity(pred_grad, dem_grad, dim=1)
        loss = 1 - torch.mean(cos_sim)
        return loss


# 推荐使用这个组合
class BalancedLoss(nn.Module):
    """平衡损失：BCE + 小权重边界损失 + 几何一致性损失"""

    def __init__(self, bce_weight=0.8, boundary_weight=0.1, geo_weight=0.1):
        super().__init__()
        self.bce_weight = bce_weight
        self.boundary_weight = boundary_weight
        self.geo_weight = geo_weight

        # 主BCE损失（添加正样本权重处理类别不平衡）
        self.bce_loss = nn.BCEWithLogitsLoss()

        # Sobel算子用于边缘检测和梯度计算
        self.register_buffer('sobel_x',
                             torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y',
                             torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))

    def forward(self, pred, target, dem=None):
        """
        pred: 模型输出logits [B, 1, H, W]
        target: 真实掩膜 [B, 1, H, W]
        dem: DEM数据 [B, 1, H, W]，用于几何一致性
        """
        # 1. 主BCE损失
        bce_loss = self.bce_loss(pred, target)

        total_loss = self.bce_weight * bce_loss

        # 2. 边界损失（降低权重）
        if self.boundary_weight > 0:
            with torch.no_grad():
                # 简化边界检测，降低敏感性
                target_edges = self.simple_edge_detection(target)

            boundary_loss = F.binary_cross_entropy_with_logits(pred, target_edges)
            total_loss += self.boundary_weight * boundary_loss

        # 3. 几何一致性损失（如果提供了DEM）
        if self.geo_weight > 0 and dem is not None:
            geo_loss = self.geometric_consistency_loss(pred, dem)
            total_loss += self.geo_weight * geo_loss

        return total_loss

    def simple_edge_detection(self, x):
        """简化边界检测，降低噪声"""
        # 使用均值滤波平滑
        kernel = torch.ones(1, 1, 3, 3).to(x.device) / 9.0
        smoothed = F.conv2d(x, kernel, padding=1)

        # 计算差值作为边缘
        edges = torch.abs(x - smoothed)

        # 二值化（提高阈值减少噪声）
        edges = (edges > 0.15).float()

        # 形态学操作去除小噪声
        edges = self.morphological_clean(edges)

        return edges

    def morphological_clean(self, edges):
        """简单的形态学清理"""
        # 膨胀
        kernel = torch.ones(1, 1, 3, 3).to(edges.device)
        dilated = F.conv2d(edges, kernel, padding=1)
        dilated = (dilated > 0).float()

        # 腐蚀
        eroded = F.conv2d(dilated, kernel, padding=1)
        eroded = (eroded == 9).float()  # 需要所有9个邻居都是1

        return eroded

    def geometric_consistency_loss(self, pred_logits, dem):
        """
        几何一致性损失：滑坡边界应与地形梯度方向一致
        Args:
            pred_logits: 预测logits [B, 1, H, W]
            dem: DEM数据 [B, 1, H, W]
        """
        # 将预测转换为概率
        pred_prob = torch.sigmoid(pred_logits)

        # 1. 计算预测的滑坡边界
        pred_edges = self.compute_gradient_magnitude(pred_prob)

        # 2. 计算DEM的地形梯度（坡度）
        dem_grad = self.compute_gradient_magnitude(dem)

        # 3. 计算DEM的地形梯度方向
        dem_grad_x = F.conv2d(dem, self.sobel_x, padding=1)
        dem_grad_y = F.conv2d(dem, self.sobel_y, padding=1)

        # 4. 方向一致性：滑坡边界应与地形陡峭区域一致
        # 在陡峭区域（高梯度）应该有更高的概率出现滑坡边界
        grad_consistency = F.mse_loss(pred_edges, dem_grad)

        # 5. 高度约束：滑坡通常发生在特定高度范围内
        # 假设滑坡更可能发生在中等高度区域（避免山顶和谷底）
        height_mean = dem.mean(dim=(1, 2, 3), keepdim=True)
        height_std = dem.std(dim=(1, 2, 3), keepdim=True)

        # 计算高度归一化得分（高斯分布，中心在均值处）
        height_norm = torch.exp(-0.5 * ((dem - height_mean) / (height_std + 1e-8)) ** 2)

        # 高度一致性：预测的滑坡概率应与高度得分正相关
        height_consistency = -torch.mean(pred_prob * height_norm)

        # 6. 曲率一致性：滑坡更可能发生在凸面区域
        dem_curvature = self.compute_curvature(dem)
        # 正曲率表示凸面，负曲率表示凹面
        curvature_score = torch.sigmoid(dem_curvature * 10)  # 放大曲率影响

        curvature_consistency = -torch.mean(pred_prob * curvature_score)

        # 组合几何一致性损失
        geo_loss = grad_consistency + 0.5 * height_consistency + 0.3 * curvature_consistency

        return geo_loss

    def compute_gradient_magnitude(self, x):
        """计算梯度幅值"""
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        # 归一化到[0, 1]
        if grad_magnitude.max() > grad_magnitude.min():
            grad_magnitude = (grad_magnitude - grad_magnitude.min()) / (
                        grad_magnitude.max() - grad_magnitude.min() + 1e-8)

        return grad_magnitude

    def compute_curvature(self, dem):
        """计算DEM的曲率（Laplacian）"""
        # Laplacian算子
        laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        laplacian_kernel = laplacian_kernel.to(dem.device)

        curvature = F.conv2d(dem, laplacian_kernel, padding=1)

        # 归一化
        curvature = torch.tanh(curvature * 0.1)  # 使用tanh限制范围

        return curvature


class AdaptiveGeometricLoss(nn.Module):
    """自适应几何一致性损失（更智能的版本）"""

    def __init__(self, weight=0.1, temperature=0.1):
        super().__init__()
        self.weight = weight
        self.temperature = temperature

        # 注册Sobel算子
        self.register_buffer('sobel_x',
                             torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y',
                             torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))

        # 自适应权重学习
        self.grad_weight = nn.Parameter(torch.tensor(1.0))
        self.height_weight = nn.Parameter(torch.tensor(0.5))
        self.curvature_weight = nn.Parameter(torch.tensor(0.3))

    def forward(self, pred_prob, dem):
        """
        pred_prob: 预测概率 [B, 1, H, W] (已经sigmoid)
        dem: DEM数据 [B, 1, H, W]
        """
        B, _, H, W = pred_prob.shape

        # 1. 计算DEM特征
        dem_features = self.extract_dem_features(dem)  # [B, 3, H, W]

        # 2. 计算预测的特征响应
        pred_features = pred_prob.expand(-1, 3, -1, -1)

        # 3. 特征相似性损失
        similarity_loss = F.mse_loss(pred_features, dem_features)

        # 4. 空间一致性：滑坡区域应该是连通的
        connectivity_loss = self.connectivity_loss(pred_prob)

        # 5. 尺度一致性：滑坡应该有合理的尺寸范围
        scale_loss = self.scale_consistency_loss(pred_prob)

        # 自适应加权
        total_loss = (
                similarity_loss +
                0.1 * connectivity_loss +
                0.05 * scale_loss
        )

        return self.weight * total_loss

    def extract_dem_features(self, dem):
        """提取DEM的多尺度特征"""
        features = []

        # 梯度特征
        grad_x = F.conv2d(dem, self.sobel_x, padding=1)
        grad_y = F.conv2d(dem, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        grad_mag = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)
        features.append(grad_mag)

        # 高度特征（归一化）
        height_norm = (dem - dem.min()) / (dem.max() - dem.min() + 1e-8)
        features.append(height_norm)

        # 曲率特征
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3).to(
            dem.device)
        curvature = F.conv2d(dem, laplacian, padding=1)
        curvature = torch.tanh(curvature * 0.1)
        features.append(curvature)

        return torch.cat(features, dim=1)

    def connectivity_loss(self, pred_prob):
        """连通性损失：鼓励预测区域更连通"""
        # 二值化
        pred_binary = (pred_prob > 0.5).float()

        # 计算每个连通区域的大小
        from scipy import ndimage
        connectivity_loss = 0

        for i in range(pred_binary.shape[0]):
            mask = pred_binary[i, 0].cpu().numpy()

            # 使用scipy计算连通区域
            labeled, num_features = ndimage.label(mask)

            if num_features > 0:
                # 计算每个区域的大小
                sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))

                # 鼓励有一个主区域，惩罚多个小区域
                if len(sizes) > 1:
                    main_region_ratio = sizes.max() / sizes.sum()
                    connectivity_loss += (1 - main_region_ratio)

        return torch.tensor(connectivity_loss / pred_binary.shape[0]).to(pred_prob.device)

    def scale_consistency_loss(self, pred_prob):
        """尺度一致性损失：鼓励合理的滑坡尺寸"""
        # 计算预测区域的面积
        pred_area = pred_prob.sum(dim=(1, 2, 3))

        # 期望的滑坡面积（假设为图像面积的10%-30%）
        total_pixels = pred_prob.shape[2] * pred_prob.shape[3]
        target_min = 0.1 * total_pixels
        target_max = 0.3 * total_pixels

        # 惩罚过大或过小的预测
        loss = torch.mean(
            torch.relu(pred_area - target_max) +  # 太大
            torch.relu(target_min - pred_area)  # 太小
        )

        return loss / total_pixels


# 简化的几何损失版本（更稳定）
class SimpleGeometricLoss(nn.Module):
    """简化的几何一致性损失"""

    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight

        # Sobel算子 - 使用nn.Parameter而不是register_buffer
        self.sobel_x = nn.Parameter(
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                         dtype=torch.float32).view(1, 1, 3, 3),
            requires_grad=False
        )
        self.sobel_y = nn.Parameter(
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                         dtype=torch.float32).view(1, 1, 3, 3),
            requires_grad=False
        )

    def forward(self, pred_logits, dem):
        """
        简化的几何一致性：滑坡边界应与地形陡峭区域相关
        """
        pred_prob = torch.sigmoid(pred_logits)

        # 1. 计算预测的边界（梯度）
        # 确保sobel算子与输入在同一设备
        sobel_x = self.sobel_x.to(pred_prob.device)
        sobel_y = self.sobel_y.to(pred_prob.device)

        pred_grad_x = F.conv2d(pred_prob, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred_prob, sobel_y, padding=1)
        pred_grad = torch.sqrt(pred_grad_x ** 2 + pred_grad_y ** 2 + 1e-8)

        # 2. 计算DEM的坡度
        dem_grad_x = F.conv2d(dem, sobel_x, padding=1)
        dem_grad_y = F.conv2d(dem, sobel_y, padding=1)
        dem_grad = torch.sqrt(dem_grad_x ** 2 + dem_grad_y ** 2 + 1e-8)

        # 3. 归一化
        pred_grad_norm = (pred_grad - pred_grad.min()) / (pred_grad.max() - pred_grad.min() + 1e-8)
        dem_grad_norm = (dem_grad - dem_grad.min()) / (dem_grad.max() - dem_grad.min() + 1e-8)

        # 4. 一致性损失：预测边界应与地形陡峭区域相关
        consistency = 1 - torch.mean(pred_grad_norm * dem_grad_norm)

        return self.weight * consistency


# 最终推荐的损失函数组合
class EnhancedLandslideLoss(nn.Module):
    """增强的滑坡检测损失函数 - 支持字典输入"""

    def __init__(self, bce_weight=0.7, boundary_weight=0.1, geo_weight=0.2):
        super().__init__()
        self.bce_weight = bce_weight
        self.boundary_weight = boundary_weight
        self.geo_weight = geo_weight

        # BCE损失（处理类别不平衡）
        self.bce_loss = nn.BCEWithLogitsLoss()

        # 几何一致性损失（简化版）
        self.geo_loss = SimpleGeometricLoss(weight=1.0)

    def forward(self, outputs, target, dem=None):
        """
        outputs: 模型输出字典，包含'mask'键（logits）
        target: 真实掩膜 [B, 1, H, W]
        dem: DEM数据 [B, 1, H, W]（可选）
        """
        # 从输出字典中提取预测logits
        if isinstance(outputs, dict):
            pred = outputs['mask']  # 提取mask预测
        else:
            pred = outputs  # 如果是张量，直接使用

        # 主BCE损失
        bce = self.bce_loss(pred, target)

        total_loss = self.bce_weight * bce

        # 几何一致性损失
        if self.geo_weight > 0 and dem is not None:
            geo = self.geo_loss(pred, dem)
            total_loss += self.geo_weight * geo

        return total_loss


class EnhancedComboLandslideLoss(nn.Module):
    """增强组合损失：Dice + Focal + 几何一致性 + 防过拟合正则化"""

    def __init__(self, dice_weight=0.4, focal_weight=0.3, geo_weight=0.2, reg_weight=0.1,
                 focal_alpha=0.25, focal_gamma=2.0, smooth=1e-6, target_pos_ratio=0.35):
        """
        Args:
            dice_weight: Dice损失权重
            focal_weight: Focal损失权重
            geo_weight: 几何一致性权重
            reg_weight: 正则化权重（防止预测全正）
            focal_alpha: Focal Loss的alpha参数（0.25给正样本更高权重）
            focal_gamma: Focal Loss的gamma参数
            smooth: 平滑系数
            target_pos_ratio: 目标正样本比例（根据你的数据设为0.35）
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.geo_weight = geo_weight
        self.reg_weight = reg_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.smooth = smooth
        self.target_pos_ratio = target_pos_ratio

        # 几何一致性损失
        self.geo_loss = SimpleGeometricLoss(weight=1.0) if geo_weight > 0 else None

        # 用于自适应调整的历史记录
        self.register_buffer('pred_pos_history', torch.zeros(50))
        self.history_idx = 0

    def forward(self, outputs, target, dem=None):
        # 提取预测
        if isinstance(outputs, dict):
            pred_logits = outputs['mask']
        else:
            pred_logits = outputs

        pred_prob = torch.sigmoid(pred_logits)

        # 1. Dice Loss (对类别不平衡鲁棒)
        dice_loss = self._dice_loss(pred_prob, target)

        # 2. Focal Loss (处理难样本) - 使用自适应alpha
        focal_loss = self._adaptive_focal_loss(pred_logits, target, pred_prob)

        # 3. 组合基础损失
        total_loss = self.dice_weight * dice_loss + self.focal_weight * focal_loss

        # 4. 几何一致性损失
        if self.geo_weight > 0 and self.geo_loss is not None and dem is not None:
            try:
                geo = self.geo_loss(pred_logits, dem)
                total_loss += self.geo_weight * geo
            except Exception as e:
                print(f"几何损失跳过: {e}")

        # 5. 正则化损失（防止预测全正）
        if self.reg_weight > 0:
            reg_loss = self._regularization_loss(pred_prob, target)
            total_loss += self.reg_weight * reg_loss

        # 6. 平衡约束损失（可选）
        balance_loss = self._balance_constraint(pred_prob)
        total_loss += 0.05 * balance_loss  # 小权重

        return total_loss

    def _dice_loss(self, pred, target):
        """Dice损失 - 添加类别权重"""
        pred = pred.contiguous().view(pred.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        intersection = (pred * target).sum(1)
        union = pred.sum(1) + target.sum(1)

        # 添加类别权重：给正样本区域更高权重
        pos_weight = target.sum(1) / (target.shape[1] + self.smooth)
        weighted_dice = (2. * intersection + self.smooth) / (union + self.smooth)
        weighted_dice = weighted_dice * (1 + pos_weight)  # 正样本多的样本权重更高

        return 1 - weighted_dice.mean()

    def _adaptive_focal_loss(self, pred_logits, target, pred_prob):
        """自适应Focal Loss"""
        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')

        # 自适应alpha：根据预测正样本比例调整
        pred_pos_ratio = pred_prob.mean()
        self._update_history(pred_pos_ratio)

        # 如果预测正样本太多，降低alpha（给负样本更多权重）
        if pred_pos_ratio > self.target_pos_ratio * 1.5:  # 超过目标50%
            adaptive_alpha = max(0.1, self.focal_alpha * 0.5)  # 降低正样本权重
        else:
            adaptive_alpha = self.focal_alpha

        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        alpha_t = adaptive_alpha * target + (1 - adaptive_alpha) * (1 - target)
        focal_weight = alpha_t * (1 - p_t) ** self.focal_gamma

        return (focal_weight * bce_loss).mean()

    def _regularization_loss(self, pred_prob, target):
        """正则化损失，防止预测全正"""
        reg_losses = []

        # 1. 预测均值约束：鼓励预测均值接近目标比例
        mean_constraint = F.mse_loss(
            pred_prob.mean(),
            torch.tensor(self.target_pos_ratio, device=pred_prob.device)
        )
        reg_losses.append(mean_constraint)

        # 2. 预测熵约束：鼓励预测有不确定性（不极端）
        entropy = - (pred_prob * torch.log(pred_prob + self.smooth) +
                     (1 - pred_prob) * torch.log(1 - pred_prob + self.smooth))
        entropy_loss = -entropy.mean()  # 最大化熵（负熵最小化）
        reg_losses.append(0.1 * entropy_loss)

        # 3. 批次内方差约束：鼓励不同样本有不同的预测
        batch_var = pred_prob.var()
        if batch_var < 0.01:  # 如果方差太小（预测太一致）
            var_constraint = -torch.log(batch_var + self.smooth) * 0.01
            reg_losses.append(var_constraint)

        # 4. 预测全正惩罚：如果样本预测全正（>0.9）且实际无滑坡，施加惩罚
        false_positive_mask = (pred_prob > 0.7) & (target < 0.5)
        if false_positive_mask.any():
            fp_penalty = pred_prob[false_positive_mask].mean() * 2.0
            reg_losses.append(fp_penalty)

        return sum(reg_losses)

    def _balance_constraint(self, pred_prob):
        """平衡约束：鼓励预测在批次内平衡"""
        batch_size = pred_prob.shape[0]

        # 计算每个样本的正像素比例
        sample_pos_ratios = pred_prob.view(batch_size, -1).mean(dim=1)

        # 鼓励样本间的多样性（不都预测相同）
        diversity_loss = -sample_pos_ratios.var()

        # 鼓励正样本比例接近目标
        target_tensor = torch.full_like(sample_pos_ratios, self.target_pos_ratio)
        target_loss = F.mse_loss(sample_pos_ratios, target_tensor)

        return diversity_loss + target_loss

    def _update_history(self, pred_pos_ratio):
        """更新预测历史"""
        self.pred_pos_history[self.history_idx] = pred_pos_ratio
        self.history_idx = (self.history_idx + 1) % 50

    def get_prediction_statistics(self):
        """获取预测统计"""
        if self.history_idx > 0:
            history = self.pred_pos_history[:self.history_idx]
            return {
                'mean': history.mean().item(),
                'std': history.std().item(),
                'max': history.max().item(),
                'min': history.min().item()
            }
        return None


class SimpleWeightedBCELoss(nn.Module):
    """简化的加权BCE损失，处理类别不平衡"""

    def __init__(self, pos_weight=2.0, neg_weight=1.0):
        """
        pos_weight: 正样本（滑坡像素）权重
        neg_weight: 负样本（非滑坡像素）权重
        根据你的数据：滑坡像素很少，应该给更高权重
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, outputs, target, dem=None):
        # 提取预测logits
        if isinstance(outputs, dict):
            pred_logits = outputs['mask']
        else:
            pred_logits = outputs

        # 创建权重矩阵
        weights = torch.where(target > 0.5,
                              torch.tensor(self.pos_weight, device=target.device),
                              torch.tensor(self.neg_weight, device=target.device))

        # 加权BCE损失
        loss = F.binary_cross_entropy_with_logits(
            pred_logits, target,
            weight=weights,
            reduction='mean'
        )

        return loss


class BalancedLandslideLoss(nn.Module):
    """平衡的滑坡检测损失"""

    def __init__(self, bce_weight=0.7, geo_weight=0.3,
                 pos_pixel_weight=5.0, neg_pixel_weight=1.0):
        """
        pos_pixel_weight: 滑坡像素权重（应该很高）
        neg_pixel_weight: 非滑坡像素权重
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.geo_weight = geo_weight
        self.pos_pixel_weight = pos_pixel_weight
        self.neg_pixel_weight = neg_pixel_weight

        self.geo_loss = SimpleGeometricLoss(weight=1.0) if geo_weight > 0 else None

    def forward(self, outputs, target, dem=None):
        # 提取预测
        if isinstance(outputs, dict):
            pred_logits = outputs['mask']
        else:
            pred_logits = outputs

        # 1. 加权BCE损失
        weights = torch.where(target > 0.5,
                              torch.tensor(self.pos_pixel_weight, device=target.device),
                              torch.tensor(self.neg_pixel_weight, device=target.device))

        bce_loss = F.binary_cross_entropy_with_logits(
            pred_logits, target, weight=weights, reduction='mean'
        )

        total_loss = self.bce_weight * bce_loss

        # 2. 几何一致性损失
        if self.geo_weight > 0 and self.geo_loss is not None and dem is not None:
            try:
                geo = self.geo_loss(pred_logits, dem)
                total_loss += self.geo_weight * geo
            except Exception as e:
                print(f"几何损失跳过: {e}")

        return total_loss

# 7. 训练函数
def train_model_simple(model, train_loader, val_loader, criterion, optimizer,
                       scheduler, num_epochs=30, device='cuda'):
    best_iou = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_precision': [], 'val_recall': []}

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 30)

        # 训练阶段
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc='Training')
        for optical, dem, mask, is_landslide, _ in pbar:
            optical = optical.to(device)
            dem = dem.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            outputs = model(optical, dem)
            loss = criterion(outputs, mask, dem)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        print(f'Train Loss: {avg_train_loss:.4f}')

        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_tp, all_fp, all_fn, all_tn = 0, 0, 0, 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for optical, dem, mask, is_landslide, _ in pbar:
                optical = optical.to(device)
                dem = dem.to(device)
                mask = mask.to(device)

                outputs = model(optical, dem)
                loss = criterion(outputs, mask, dem)
                val_loss += loss.item()

                # 获取预测
                if isinstance(outputs, dict):
                    pred_probs = torch.sigmoid(outputs['mask'])
                else:
                    pred_probs = torch.sigmoid(outputs)

                preds = (pred_probs > 0.5).float()

                # 计算混淆矩阵
                tp = ((preds == 1) & (mask == 1)).sum().item()
                fp = ((preds == 1) & (mask == 0)).sum().item()
                fn = ((preds == 0) & (mask == 1)).sum().item()
                tn = ((preds == 0) & (mask == 0)).sum().item()

                all_tp += tp
                all_fp += fp
                all_fn += fn
                all_tn += tn

                batch_precision = tp / max(tp + fp, 1)
                batch_recall = tp / max(tp + fn, 1)
                pbar.set_postfix({
                    'val_loss': loss.item(),
                    'prec': f'{batch_precision:.3f}',
                    'rec': f'{batch_recall:.3f}'
                })

        # 计算总体指标
        avg_val_loss = val_loss / len(val_loader)

        precision = all_tp / max(all_tp + all_fp, 1)
        recall = all_tp / max(all_tp + all_fn, 1)
        accuracy = (all_tp + all_tn) / max(all_tp + all_fp + all_fn + all_tn, 1)

        # IoU计算
        iou = all_tp / max(all_tp + all_fp + all_fn, 1)

        history['val_loss'].append(avg_val_loss)
        history['val_iou'].append(iou)
        history['val_precision'].append(precision)
        history['val_recall'].append(recall)

        print(f'Val Loss: {avg_val_loss:.4f}')
        print(f'Val IoU: {iou:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}')

        # 详细统计
        total_pixels = all_tp + all_fp + all_fn + all_tn
        print(f'像素级统计:')
        print(f'  总像素: {total_pixels:,}')
        print(f'  滑坡像素(TP): {all_tp:,} ({all_tp / total_pixels * 100:.2f}%)')
        print(f'  误报像素(FP): {all_fp:,} ({all_fp / total_pixels * 100:.2f}%)')
        print(f'  漏报像素(FN): {all_fn:,} ({all_fn / total_pixels * 100:.2f}%)')
        print(f'  正确负像素(TN): {all_tn:,} ({all_tn / total_pixels * 100:.2f}%)')

        # 学习率调度
        scheduler.step(avg_val_loss)

        # 保存最佳模型
        if iou > best_iou:
            best_iou = iou
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'✓ 保存最佳模型，IoU: {best_iou:.4f}')

    return model, history


# 8. 测试/预测函数
def predict_and_evaluate(model, test_loader, device='cuda', save_dir='predictions'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    all_preds = []
    all_masks = []
    metrics = {'iou': [], 'precision': [], 'recall': [], 'f1': []}

    with torch.no_grad():
        for i, (optical, dem, mask, img_paths) in enumerate(tqdm(test_loader, desc='Testing')):
            optical = optical.to(device)
            dem = dem.to(device)

            outputs = model(optical, dem)
            preds = (outputs['mask'] > 0.5).float().cpu()

            # 保存预测结果
            for j in range(len(img_paths)):
                img_name = os.path.basename(img_paths[j])
                pred_mask = preds[j].squeeze().numpy()
                pred_mask = (pred_mask * 255).astype(np.uint8)

                # 保存为PNG
                cv2.imwrite(os.path.join(save_dir, f'pred_{img_name}.png'), pred_mask)

                # 保存可视化结果
                if mask[j].sum() > 0:  # 仅对有真实掩膜的样本
                    orig_img = cv2.imread(img_paths[j])
                    orig_img = cv2.resize(orig_img, (256, 256))

                    # 可视化
                    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                    axs[0].imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
                    axs[0].set_title('Original Image')
                    axs[0].axis('off')

                    axs[1].imshow(mask[j].squeeze().cpu().numpy(), cmap='gray')
                    axs[1].set_title('Ground Truth')
                    axs[1].axis('off')

                    axs[2].imshow(pred_mask, cmap='jet', alpha=0.7)
                    axs[2].set_title('Prediction')
                    axs[2].axis('off')

                    plt.savefig(os.path.join(save_dir, f'vis_{img_name}.png'), bbox_inches='tight')
                    plt.close()

            # 仅对有真实标签的样本计算指标
            valid_mask_indices = [j for j in range(len(img_paths)) if mask[j].sum() > 0]
            if valid_mask_indices:
                valid_preds = preds[valid_mask_indices]
                valid_masks = mask[valid_mask_indices].cpu()

                for j in range(len(valid_preds)):
                    y_true = valid_masks[j].numpy().flatten()
                    y_pred = valid_preds[j].numpy().flatten()

                    # 二值化
                    y_true = (y_true > 0.5).astype(int)
                    y_pred = (y_pred > 0.5).astype(int)

                    metrics['iou'].append(jaccard_score(y_true, y_pred, zero_division=0))
                    metrics['precision'].append(precision_score(y_true, y_pred, zero_division=0))
                    metrics['recall'].append(recall_score(y_true, y_pred, zero_division=0))
                    metrics['f1'].append(f1_score(y_true, y_pred, zero_division=0))

    # 计算平均指标
    results = {metric: np.mean(values) for metric, values in metrics.items() if values}
    print("\nEvaluation Results:")
    for metric, value in results.items():
        print(f"{metric.upper()}: {value:.4f}")

    return results


def split_dataset_with_balance(dataset, test_ratio=0.5, random_seed=42):
    """保持滑坡样本比例的划分"""
    import numpy as np

    # 获取每个样本的标签（是否有滑坡）
    labels = []
    for i in range(len(dataset)):
        _, _, mask, _ = dataset[i]
        labels.append(1 if mask.sum() > 0 else 0)

    labels = np.array(labels)
    indices = np.arange(len(dataset))

    # 分离正负样本索引
    pos_indices = indices[labels == 1]
    neg_indices = indices[labels == 0]

    # 设置随机种子
    np.random.seed(random_seed)
    np.random.shuffle(pos_indices)
    np.random.shuffle(neg_indices)

    # 按比例划分正样本
    pos_test_size = int(len(pos_indices) * test_ratio)
    pos_val_indices = pos_indices[:pos_test_size]
    pos_test_indices = pos_indices[pos_test_size:]

    # 按比例划分负样本
    neg_test_size = int(len(neg_indices) * test_ratio)
    neg_val_indices = neg_indices[:neg_test_size]
    neg_test_indices = neg_indices[neg_test_size:]

    # 合并验证集和测试集索引
    val_indices = np.concatenate([pos_val_indices, neg_val_indices])
    test_indices = np.concatenate([pos_test_indices, neg_test_indices])

    # 打乱顺序
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)

    # 创建子集
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    test_subset = torch.utils.data.Subset(dataset, test_indices)

    # 统计信息
    val_labels = labels[val_indices]
    test_labels = labels[test_indices]

    print(f"验证集: {len(val_subset)} 样本 (滑坡: {val_labels.sum()}, 非滑坡: {len(val_labels) - val_labels.sum()})")
    print(
        f"测试集: {len(test_subset)} 样本 (滑坡: {test_labels.sum()}, 非滑坡: {len(test_labels) - test_labels.sum()})")

    return val_subset, test_subset


# 9. 主程序
def main():
    # 设置
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据准备（使用新函数）
    data_dir = "/kaggle/input/beiji-landslide-and-dem/Bijie-landslide-dataset/"
    train_dataset, test_dataset = prepare_datasets_with_masks(data_dir, target_size=(256, 256))

    # 划分验证集
    val_ratio = 0.5
    val_size = int(len(test_dataset) * val_ratio)
    test_size = len(test_dataset) - val_size

    test_subset, val_subset = torch.utils.data.random_split(
        test_dataset, [test_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_subset)} 样本")
    print(f"测试集: {len(test_subset)} 样本")

    # 创建数据加载器
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 创建模型
    model = HeteroFusionSegNet(encoder_name='resnet34').to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 损失函数 - 使用高正样本权重
    criterion = BalancedLandslideLoss(
        bce_weight=0.8,
        geo_weight=0.2,
        pos_pixel_weight=10.0,  # 滑坡像素权重很高
        neg_pixel_weight=1.0  # 非滑坡像素正常权重
    )

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # 学习率调度
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    # 训练
    print("\n开始训练...")
    model, history = train_model_simple(
        model, train_loader, val_loader, criterion, optimizer,
        scheduler, num_epochs=20, device=device
    )

    # 加载最佳模型
    model.load_state_dict(torch.load('best_heterofusion_model.pth'))

    # 评估
    print("\nEvaluating on test set...")
    results = predict_and_evaluate(model, test_loader, device=device, save_dir='test_predictions')

    # 保存结果
    results_df = pd.DataFrame([results])
    results_df.to_csv('evaluation_results.csv', index=False)
    print("Results saved to evaluation_results.csv")

    # 保存训练历史
    history_df = pd.DataFrame(history)
    history_df.to_csv('training_history.csv', index=False)
    print("Training history saved to training_history.csv")

    print("All done!")

if __name__ == "__main__":
    print("开始执行主程序...")
    main()
    print("程序执行完毕！")