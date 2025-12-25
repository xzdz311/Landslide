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
from resnet import build_resnet_backbone, BasicBlock, Bottleneck
from predict import create_test_loader_from_csv, predict_and_evaluate


class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""

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


class ASPP(nn.Module):
    """空洞空间金字塔池化模块 - 为DeepLabV3+优化"""

    def __init__(self, in_channels, out_channels=256, atrous_rates=[6, 12, 18]):
        super().__init__()

        # 1x1卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 3x3空洞卷积，不同膨胀率
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rates[0],
                      dilation=atrous_rates[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rates[1],
                      dilation=atrous_rates[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rates[2],
                      dilation=atrous_rates[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 全局平均池化
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 输出融合
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5)
        )

    def forward(self, x):
        h, w = x.shape[2:]

        # 各个分支
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(x)
        conv3_out = self.conv3(x)
        conv4_out = self.conv4(x)

        # 全局池化分支
        global_feat = self.global_avg_pool(x)
        global_feat = F.interpolate(global_feat, size=(h, w),
                                    mode='bilinear', align_corners=True)

        # 拼接所有特征
        out = torch.cat([conv1_out, conv2_out, conv3_out, conv4_out, global_feat], dim=1)
        out = self.fusion(out)

        return out


class DeepLabV3PlusResNet(nn.Module):
    """DeepLabV3+模型 - 适配你的自定义ResNet骨干网络"""

    def __init__(self, n_channels=4, n_classes=1, backbone='resnet50', output_stride=16):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # 早期融合层：将光学影像和DEM在通道维度拼接
        # 3通道光学影像 + 1通道DEM = 4通道
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 使用你的自定义ResNet骨干网络
        # 注意：你的ResNetBackbone默认返回一个特征列表[c2, c3, c4, c5]
        self.backbone = build_resnet_backbone(
            arch=backbone,
            in_channels=64  # 融合后的通道数
        )

        # 根据backbone类型设置通道数
        if backbone in ['resnet18', 'resnet34']:
            # BasicBlock: expansion=1
            low_level_channels = 64  # layer1输出通道数
            high_level_channels = 512  # layer4输出通道数
        else:  # resnet50, resnet101, resnet152
            # Bottleneck: expansion=4
            low_level_channels = 256  # layer1输出: 64*4=256
            high_level_channels = 2048  # layer4输出: 512*4=2048

        # ASPP模块 (处理高层特征)
        self.aspp = ASPP(
            in_channels=high_level_channels,
            out_channels=256
        )

        # 低层特征处理 (来自layer1的输出)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # 解码器部分
        # 输入: 256(ASPP输出) + 48(低层特征) = 304
        self.decoder_conv1 = DoubleConv(256 + 48, 256)
        self.decoder_conv2 = DoubleConv(256, 256)

        # 最终分类层
        self.final_conv = nn.Conv2d(256, n_classes, kernel_size=1)

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

    def forward(self, optical, dem):
        """
        前向传播

        参数:
            optical: 光学影像 [B, 3, H, W]
            dem: 高程数据 [B, 1, H, W]

        返回:
            分割结果 [B, n_classes, H, W]
        """
        # 1. 早期融合: 在通道维度拼接光学影像和DEM
        x = torch.cat([optical, dem], dim=1)

        # 记录输入尺寸用于后续上采样
        input_size = x.shape[2:]

        # 2. 融合卷积层
        x = self.fusion_conv(x)

        # 3. 骨干网络特征提取
        # 根据你的ResNetBackbone实现，它返回一个特征列表[c2, c3, c4, c5]
        # c2: layer1输出 [B, C_low, H/4, W/4]
        # c3: layer2输出 [B, *, H/8, W/8]
        # c4: layer3输出 [B, *, H/16, W/16]
        # c5: layer4输出 [B, C_high, H/32, W/32]
        features = self.backbone(x)

        # 提取需要的特征
        low_level_feat = features[0]  # c2: 低层特征 (1/4分辨率)
        high_level_feat = features[-1]  # c5: 高层特征 (1/32分辨率)

        # 4. ASPP模块处理高层特征
        aspp_feat = self.aspp(high_level_feat)  # [B, 256, H/32, W/32]

        # 5. 处理低层特征
        low_level_feat = self.low_level_conv(low_level_feat)  # [B, 48, H/4, W/4]

        # 6. 解码器部分
        # 上采样ASPP特征到低层特征的分辨率
        aspp_feat_up = F.interpolate(
            aspp_feat,
            size=low_level_feat.shape[2:],
            mode='bilinear',
            align_corners=True
        )  # [B, 256, H/4, W/4]

        # 拼接ASPP特征和低层特征
        decoder_feat = torch.cat([aspp_feat_up, low_level_feat], dim=1)  # [B, 304, H/4, W/4]

        # 解码器卷积
        decoder_feat = self.decoder_conv1(decoder_feat)  # [B, 256, H/4, W/4]
        decoder_feat = self.decoder_conv2(decoder_feat)  # [B, 256, H/4, W/4]

        # 7. 上采样到原尺寸
        output = F.interpolate(
            decoder_feat,
            size=input_size,
            mode='bilinear',
            align_corners=True
        )  # [B, 256, H, W]

        # 8. 最终卷积得到分割结果
        output = self.final_conv(output)  # [B, n_classes, H, W]

        return output


def get_simple_training_config():
    """获取简单训练配置"""

    # 1. 创建简单模型
    model = DeepLabV3PlusResNet(n_channels=4, n_classes=1,backbone='resnet50')

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

    model.load_state_dict(torch.load(r'F:\zx\模型结果及参数\final_deeplabv3+_model.pth', map_location=torch.device('cpu')))

    model.eval()

    # 2. 运行评估
    results = predict_and_evaluate(
        model=model,
        test_loader=test_loader,  # 你的测试数据加载器
        device='cpu',
        save_dir='F:\zx\predictions_results\predictions_results_deeplabv3+',
        multigpu=True
    )


# 运行调试
if __name__ == "__main__":
    main()

