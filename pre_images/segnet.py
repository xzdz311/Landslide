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


class SegNetEncoderBlock(nn.Module):
    """SegNet编码器块: 卷积层 + BatchNorm + ReLU + 池化(带索引存储)"""

    def __init__(self, in_channels, out_channels, num_convs=2):
        super().__init__()
        self.num_convs = num_convs

        # 创建卷积层序列
        convs = []
        for i in range(num_convs):
            conv_in = in_channels if i == 0 else out_channels
            convs.extend([
                nn.Conv2d(conv_in, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])

        self.convs = nn.Sequential(*convs)
        # 使用MaxPool2d并存储索引
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        # 卷积操作
        x = self.convs(x)
        # 池化并存储索引
        x, indices = self.pool(x)
        return x, indices


class SegNetDecoderBlock(nn.Module):
    """SegNet解码器块: 反池化 + 卷积层 + BatchNorm + ReLU"""

    def __init__(self, in_channels, out_channels, num_convs=2):
        super().__init__()
        self.num_convs = num_convs

        # 反池化层 (使用存储的索引)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        # 创建卷积层序列
        convs = []
        for i in range(num_convs):
            conv_in = in_channels if i == 0 else out_channels
            convs.extend([
                nn.Conv2d(conv_in, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True) if i < num_convs - 1 else nn.Identity()
            ])

        self.convs = nn.Sequential(*convs)

    def forward(self, x, indices, output_size):
        # 反池化操作 (使用编码器存储的索引)
        x = self.unpool(x, indices, output_size=output_size)
        # 卷积操作
        x = self.convs(x)
        return x


class SegNet(nn.Module):
    """标准SegNet网络 - 早期融合版本"""

    def __init__(self, n_channels=4, n_classes=1):
        super().__init__()

        # 编码器 (下采样) - 存储池化索引
        self.enc1 = SegNetEncoderBlock(n_channels, 64, num_convs=2)
        self.enc2 = SegNetEncoderBlock(64, 128, num_convs=2)
        self.enc3 = SegNetEncoderBlock(128, 256, num_convs=3)
        self.enc4 = SegNetEncoderBlock(256, 512, num_convs=3)
        self.enc5 = SegNetEncoderBlock(512, 512, num_convs=3)

        # 解码器 (上采样) - 使用存储的索引进行反池化
        self.dec5 = SegNetDecoderBlock(512, 512, num_convs=3)
        self.dec4 = SegNetDecoderBlock(512, 256, num_convs=3)
        self.dec3 = SegNetDecoderBlock(256, 128, num_convs=3)
        self.dec2 = SegNetDecoderBlock(128, 64, num_convs=2)

        # 最后一个解码块（特殊处理，没有ReLU激活）
        self.dec1_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.dec1_bn1 = nn.BatchNorm2d(64)
        self.dec1_conv2 = nn.Conv2d(64, n_classes, kernel_size=3, padding=1, bias=False)

        # 输出层
        self.outc = nn.Conv2d(n_classes, n_classes, kernel_size=1)

    def forward(self, optical, dem):
        # 早期融合: 在通道维度拼接 (与SegNet相同)
        x = torch.cat([optical, dem], dim=1)

        # 编码路径 (存储池化索引)
        x1, idx1 = self.enc1(x)  # 下采样2倍
        x2, idx2 = self.enc2(x1)  # 下采样4倍
        x3, idx3 = self.enc3(x2)  # 下采样8倍
        x4, idx4 = self.enc4(x3)  # 下采样16倍
        x5, idx5 = self.enc5(x4)  # 下采样32倍

        # 解码路径 (使用存储的索引进行反池化)
        x = self.dec5(x5, idx5, output_size=x4.size())
        x = self.dec4(x, idx4, output_size=x3.size())
        x = self.dec3(x, idx3, output_size=x2.size())
        x = self.dec2(x, idx2, output_size=x1.size())

        # 最后一个解码块 (反池化 + 卷积)
        x = nn.functional.max_unpool2d(x, idx1, kernel_size=2, stride=2)
        x = nn.functional.relu(self.dec1_bn1(self.dec1_conv1(x)))
        x = self.dec1_conv2(x)

        # 输出层 (1x1卷积)
        return self.outc(x)


def get_simple_training_config():
    """获取简单训练配置"""

    # 1. 创建简单模型
    model = SegNet(n_channels=4, n_classes=1)

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
    model.load_state_dict(torch.load(r'F:\zx\模型结果及参数\final_SegNet_model.pth', map_location=torch.device('cpu')))
    model.eval()

    # 2. 运行评估
    results = predict_and_evaluate(
        model=model,
        test_loader=test_loader,  # 你的测试数据加载器
        device='cpu',
        save_dir='F:\zx\predictions_results\predictions_results_segnet',
        multigpu=True
    )


# 运行调试
if __name__ == "__main__":
    main()

