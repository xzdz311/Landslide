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


class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
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


class FCN(nn.Module):
    """全卷积网络FCN-8s版本 - 早期融合适配"""

    def __init__(self, n_channels=4, n_classes=1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # 编码器部分 (基于VGG16结构，但简化以适应遥感数据)
        # Block 1
        self.conv1_1 = nn.Conv2d(n_channels, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # Block 2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # Block 3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # Block 4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # Block 5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # FCN-8s特定层
        # 分类器 (1x1卷积替换全连接层)
        self.fc6 = nn.Conv2d(512, 4096, 7, padding=3)
        self.drop6 = nn.Dropout2d()

        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_classes, 1)  # 1/32预测

        # 跳层连接分数
        self.score_pool4 = nn.Conv2d(512, n_classes, 1)  # 1/16特征
        self.score_pool3 = nn.Conv2d(256, n_classes, 1)  # 1/8特征

        # 上采样层
        self.upscore2 = nn.ConvTranspose2d(
            n_classes, n_classes, 4, stride=2, bias=False)  # 1/16 -> 1/8
        self.upscore4 = nn.ConvTranspose2d(
            n_classes, n_classes, 4, stride=2, bias=False)  # 1/8 -> 1/4
        self.upscore8 = nn.ConvTranspose2d(
            n_classes, n_classes, 16, stride=8, bias=False)  # 1/4 -> 原尺寸

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                # 使用双线性插值初始化转置卷积
                nn.init.constant_(m.weight, 0)
                if m.kernel_size[0] == 16:
                    # 8x上采样
                    self._make_bilinear_weights_16x(m)
                elif m.kernel_size[0] == 4:
                    # 2x上采样
                    self._make_bilinear_weights_4x(m)

    def _make_bilinear_weights_4x(self, layer):
        """创建4x4双线性插值权重"""
        factor = 2
        c = layer.weight.size(0)

        weight = torch.zeros(c, c, 4, 4)
        for i in range(c):
            weight[i, i] = self._bilinear_kernel(4, factor)

        layer.weight.data = weight

    def _make_bilinear_weights_16x(self, layer):
        """创建16x16双线性插值权重"""
        factor = 8
        c = layer.weight.size(0)

        weight = torch.zeros(c, c, 16, 16)
        for i in range(c):
            weight[i, i] = self._bilinear_kernel(16, factor)

        layer.weight.data = weight

    def _bilinear_kernel(self, kernel_size, factor):
        """生成双线性插值核"""
        center = (kernel_size - 1) / 2.0

        og = torch.arange(kernel_size)
        og = og.unsqueeze(0).repeat(kernel_size, 1)

        u = (og - center) / factor
        v = (og.t() - center) / factor

        kernel = (1 - torch.abs(u)) * (1 - torch.abs(v))
        kernel = kernel / kernel.sum()

        return kernel

    def forward(self, optical, dem):
        """
        前向传播

        参数:
            optical: 光学影像 [B, C_optical, H, W]
            dem: 高程数据 [B, C_dem, H, W]

        返回:
            分割结果 [B, n_classes, H, W]
        """
        # 早期融合: 在通道维度拼接
        x = torch.cat([optical, dem], dim=1)

        # 编码路径
        # Block 1
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = self.pool1(h)  # 1/2

        # Block 2
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = self.pool2(h)  # 1/4

        # Block 3
        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        pool3 = h  # 保存pool3特征用于跳层连接
        h = self.pool3(h)  # 1/8

        # Block 4
        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        pool4 = h  # 保存pool4特征用于跳层连接
        h = self.pool4(h)  # 1/16

        # Block 5
        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = self.pool5(h)  # 1/32

        # FCN分类器
        h = F.relu(self.fc6(h))
        h = self.drop6(h)

        h = F.relu(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)  # 1/32预测

        # FCN-8s跳层连接融合
        # 第一次上采样 (1/32 -> 1/16)
        h = self.upscore2(h)
        upscore2 = h  # 1/16尺寸

        # 添加pool4的预测
        score_pool4 = self.score_pool4(pool4)
        h = score_pool4[:, :, 5:5 + upscore2.size(2), 5:5 + upscore2.size(3)]
        h = h + upscore2

        # 第二次上采样 (1/16 -> 1/8)
        upscore4 = self.upscore4(h)  # 1/8尺寸

        # 添加pool3的预测
        score_pool3 = self.score_pool3(pool3)
        h = score_pool3[:, :, 9:9 + upscore4.size(2), 9:9 + upscore4.size(3)]
        h = h + upscore4

        # 最终上采样到原尺寸 (1/8 -> 原尺寸)
        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size(2), 31:31 + x.size(3)].contiguous()

        return h


def get_simple_training_config():
    """获取简单训练配置"""

    # 1. 创建简单模型
    model = FCN(n_channels=4, n_classes=1)

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

    model.load_state_dict(torch.load(r'F:\zx\模型结果及参数\final_FCN_model.pth', map_location=torch.device('cpu')))
    model.eval()

    # 2. 运行评估
    results = predict_and_evaluate(
        model=model,
        test_loader=test_loader,  # 你的测试数据加载器
        device='cpu',
        save_dir=r'F:\zx\predictions_results\predictions_results_fcn',
        multigpu=True
    )


# 运行调试
if __name__ == "__main__":
    main()

