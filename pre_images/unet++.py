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


class Up(nn.Module):
    """上采样模块"""

    def __init__(self, in_channels1, in_channels2, out_channels, bilinear=True):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels1 + in_channels2, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels1 // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels1 // 2 + in_channels2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # 处理尺寸不匹配的情况
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        if diffX > 0 or diffY > 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

        # 拼接特征
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetPlusPlus(nn.Module):
    """UNet++网络 - 适配多GPU训练"""

    def __init__(self, n_channels=4, n_classes=1, bilinear=True, deep_supervision=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.deep_supervision = deep_supervision  # 默认关闭深度监督

        # 基础通道数（可以调整以控制模型大小）
        filters = [64, 128, 256, 512, 1024]

        # 编码器
        self.inc = DoubleConv(n_channels, filters[0])
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(filters[0], filters[1])
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(filters[1], filters[2])
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(filters[2], filters[3])
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(filters[3], filters[4])
        )

        # 上采样模块
        self.up_1_0 = Up(filters[1], filters[0], filters[0], bilinear)
        self.up_2_0 = Up(filters[2], filters[1], filters[1], bilinear)
        self.up_3_0 = Up(filters[3], filters[2], filters[2], bilinear)
        self.up_4_0 = Up(filters[4], filters[3], filters[3], bilinear)

        # 嵌套解码路径
        self.up_1_1 = Up(filters[1], filters[0], filters[0], bilinear)
        self.up_2_1 = Up(filters[2], filters[1], filters[1], bilinear)
        self.up_3_1 = Up(filters[3], filters[2], filters[2], bilinear)

        self.up_1_2 = Up(filters[1], filters[0], filters[0], bilinear)
        self.up_2_2 = Up(filters[2], filters[1], filters[1], bilinear)

        self.up_1_3 = Up(filters[1], filters[0], filters[0], bilinear)

        # 用于密集连接的卷积块
        self.conv_0_1 = DoubleConv(filters[0] * 2, filters[0])
        self.conv_0_2 = DoubleConv(filters[0] * 3, filters[0])
        self.conv_0_3 = DoubleConv(filters[0] * 4, filters[0])

        self.conv_1_1 = DoubleConv(filters[1] * 2, filters[1])
        self.conv_1_2 = DoubleConv(filters[1] * 3, filters[1])

        self.conv_2_1 = DoubleConv(filters[2] * 2, filters[2])

        # 输出层
        self.outc = nn.Conv2d(filters[0], n_classes, kernel_size=1)

    def forward(self, optical, dem):
        # 早期融合
        x = torch.cat([optical, dem], dim=1)

        # 编码路径
        X_00 = self.inc(x)
        X_10 = self.down1(X_00)
        X_20 = self.down2(X_10)
        X_30 = self.down3(X_20)
        X_40 = self.down4(X_30)

        # 解码路径 - 第一层
        X_01 = self.up_1_0(X_10, X_00)
        X_11 = self.up_2_0(X_20, X_10)
        X_21 = self.up_3_0(X_30, X_20)
        X_31 = self.up_4_0(X_40, X_30)

        # 解码路径 - 第二层
        X_02_cat = torch.cat([X_00, X_01], dim=1)
        X_02_conv = self.conv_0_1(X_02_cat)
        X_02 = self.up_1_1(X_11, X_02_conv)

        X_12_cat = torch.cat([X_10, X_11], dim=1)
        X_12_conv = self.conv_1_1(X_12_cat)
        X_12 = self.up_2_1(X_21, X_12_conv)

        X_22_cat = torch.cat([X_20, X_21], dim=1)
        X_22_conv = self.conv_2_1(X_22_cat)
        X_22 = self.up_3_1(X_31, X_22_conv)

        # 解码路径 - 第三层
        X_03_cat = torch.cat([X_00, X_01, X_02], dim=1)
        X_03_conv = self.conv_0_2(X_03_cat)
        X_03 = self.up_1_2(X_12, X_03_conv)

        X_13_cat = torch.cat([X_10, X_11, X_12], dim=1)
        X_13_conv = self.conv_1_2(X_13_cat)
        X_13 = self.up_2_2(X_22, X_13_conv)

        # 解码路径 - 第四层
        X_04_cat = torch.cat([X_00, X_01, X_02, X_03], dim=1)
        X_04_conv = self.conv_0_3(X_04_cat)
        X_04 = self.up_1_3(X_13, X_04_conv)

        # 输出 - 始终返回单个张量
        return self.outc(X_04)


def get_simple_training_config():
    """获取简单训练配置"""

    # 1. 创建简单模型
    model = UNetPlusPlus(n_channels=4, n_classes=1)

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

    # 创建数据加载器


    model.load_state_dict(torch.load(r'F:\zx\模型结果及参数\final_unetplusplus_model.pth', map_location=torch.device('cpu')))


    model.eval()

    # 2. 运行评估
    results = predict_and_evaluate(
        model=model,
        test_loader=test_loader,  # 你的测试数据加载器
        device='cpu',
        save_dir='F:\zx\predictions_results\predictions_results_Unet++',
        multigpu=True
    )


# 运行调试
if __name__ == "__main__":
    main()