import os

# 替换为你实际的 Graphviz bin 路径
graphviz_bin = r"D:\Program Files\Graphviz\bin"
os.environ["PATH"] += os.pathsep + graphviz_bin

from torchview import draw_graph
import torch
import torch.nn as nn

class EarlyFusionNet(nn.Module):
    """早期融合网络 - 最简单的版本"""

    def __init__(self, classes=1):
        super().__init__()

        # 直接拼接输入，使用简单的编解码器
        self.encoder = nn.Sequential(
            # 下采样路径
            self._conv_block(4, 64),
            nn.MaxPool2d(2),  # H/2
            self._conv_block(64, 128),
            nn.MaxPool2d(2),  # H/4
            self._conv_block(128, 256),
            nn.MaxPool2d(2),  # H/8
            self._conv_block(256, 512),
            nn.MaxPool2d(2),  # H/16
        )

        self.bottleneck = self._conv_block(512, 1024)

        self.decoder = nn.Sequential(
            # 上采样路径
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            self._conv_block(512, 512),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            self._conv_block(256, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            self._conv_block(128, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            self._conv_block(64, 64),
        )

        self.final_conv = nn.Conv2d(64, classes, kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, optical, dem):
        # 早期融合：直接拼接
        x = torch.cat([optical, dem], dim=1)

        # 编码
        x = self.encoder(x)
        x = self.bottleneck(x)

        # 解码
        x = self.decoder(x)

        # 输出
        out = self.final_conv(x)
        return out

model = EarlyFusionNet(classes=1)
optical = torch.randn(1, 3, 256, 256)
dem = torch.randn(1, 1, 256, 256)

model_graph = draw_graph(model, input_data=(optical, dem), expand_nested=True)
model_graph.visual_graph.render("early_fusion_net", format="png")