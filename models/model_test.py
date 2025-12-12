import torch
import torch.nn as nn
import torch.nn.functional as F


class TraditionalEdgeDetection(nn.Module):
    """传统边缘检测算法：Sobel、Canny-like、Laplacian"""

    def __init__(self, use_sobel=True, use_prewitt=True, use_laplacian=True):
        super(TraditionalEdgeDetection, self).__init__()

        self.use_sobel = use_sobel
        self.use_prewitt = use_prewitt
        self.use_laplacian = use_laplacian

        # 创建固定的卷积核（不参与训练）
        if use_sobel:
            self.sobel_x_kernel = self._create_sobel_x_kernel()
            self.sobel_y_kernel = self._create_sobel_y_kernel()

        if use_prewitt:
            self.prewitt_x_kernel = self._create_prewitt_x_kernel()
            self.prewitt_y_kernel = self._create_prewitt_y_kernel()

        if use_laplacian:
            self.laplacian_kernel = self._create_laplacian_kernel()

    def _create_sobel_x_kernel(self):
        """创建Sobel X方向卷积核"""
        kernel = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return kernel

    def _create_sobel_y_kernel(self):
        """创建Sobel Y方向卷积核"""
        kernel = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return kernel

    def _create_prewitt_x_kernel(self):
        """创建Prewitt X方向卷积核"""
        kernel = torch.tensor([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return kernel

    def _create_prewitt_y_kernel(self):
        """创建Prewitt Y方向卷积核"""
        kernel = torch.tensor([
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return kernel

    def _create_laplacian_kernel(self):
        """创建Laplacian卷积核（4邻域）"""
        kernel = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return kernel

    def _apply_kernel(self, x, kernel):
        """应用卷积核（处理多通道输入）"""
        batch_size, channels, height, width = x.shape

        # 为每个通道复制卷积核
        kernel = kernel.repeat(channels, 1, 1, 1)

        # 分组卷积，每个通道独立处理
        edges = F.conv2d(x, kernel, padding=1, groups=channels)

        return edges

    def forward(self, x):
        """提取多尺度边缘特征"""
        edge_maps = []

        # Sobel边缘检测
        if self.use_sobel:
            sobel_x = self._apply_kernel(x, self.sobel_x_kernel)
            sobel_y = self._apply_kernel(x, self.sobel_y_kernel)
            sobel_magnitude = torch.sqrt(sobel_x ** 2 + sobel_y ** 2 + 1e-8)
            edge_maps.append(sobel_magnitude)

        # Prewitt边缘检测
        if self.use_prewitt:
            prewitt_x = self._apply_kernel(x, self.prewitt_x_kernel)
            prewitt_y = self._apply_kernel(x, self.prewitt_y_kernel)
            prewitt_magnitude = torch.sqrt(prewitt_x ** 2 + prewitt_y ** 2 + 1e-8)
            edge_maps.append(prewitt_magnitude)

        # Laplacian边缘检测
        if self.use_laplacian:
            laplacian = self._apply_kernel(x, self.laplacian_kernel)
            laplacian_abs = torch.abs(laplacian)  # 取绝对值
            edge_maps.append(laplacian_abs)

        # 融合所有边缘特征
        if edge_maps:
            fused_edges = torch.stack(edge_maps, dim=1).mean(dim=1)
        else:
            fused_edges = torch.zeros_like(x)

        return fused_edges


class RGBEdgeEnhancement(nn.Module):
    """RGB图像边缘增强模块：提取RGB边缘并与原始特征融合"""

    def __init__(self, edge_weight=0.3):
        super(RGBEdgeEnhancement, self).__init__()

        self.edge_detector = TraditionalEdgeDetection(
            use_sobel=True,
            use_prewitt=True,
            use_laplacian=True
        )

        # 边缘权重（控制边缘信息的重要性）
        self.edge_weight = edge_weight

        # 简单的融合层（没有可学习参数）
        self.fusion_alpha = nn.Parameter(torch.tensor(0.5), requires_grad=False)

    def forward(self, rgb_features):
        """
        Args:
            rgb_features: RGB特征图 [B, C, H, W]
        Returns:
            增强后的特征图
        """
        # 提取RGB边缘
        rgb_edges = self.edge_detector(rgb_features)

        # 标准化边缘特征（0-1范围）
        rgb_edges_normalized = (rgb_edges - rgb_edges.min()) / (rgb_edges.max() - rgb_edges.min() + 1e-8)

        # 增强原始特征：原始特征 + 边缘特征
        enhanced_features = rgb_features + self.edge_weight * rgb_edges_normalized

        return enhanced_features


class EdgeAwareFusion(nn.Module):
    """边缘感知的特征融合模块：将RGB边缘与DEM特征融合"""

    def __init__(self):
        super(EdgeAwareFusion, self).__init__()

        # 提取RGB边缘
        self.rgb_edge_extractor = TraditionalEdgeDetection(
            use_sobel=True,
            use_prewitt=True,
            use_laplacian=False  # 简化版本
        )

    def forward(self, rgb_features, dem_features):
        """
        Args:
            rgb_features: RGB特征 [B, 3, H, W]
            dem_features: DEM特征 [B, 1, H, W]
        Returns:
            融合后的特征 [B, 4, H, W]
        """
        # 提取RGB边缘
        rgb_edges = self.rgb_edge_extractor(rgb_features)

        # 标准化RGB边缘
        rgb_edges_normalized = (rgb_edges - rgb_edges.min()) / (rgb_edges.max() - rgb_edges.min() + 1e-8)

        # 将RGB边缘作为额外的通道
        rgb_edges_mean = rgb_edges.mean(dim=1, keepdim=True)  # 将3个通道合并为1个
        rgb_edges_normalized = (rgb_edges_mean - rgb_edges_mean.min()) / (
                    rgb_edges_mean.max() - rgb_edges_mean.min() + 1e-8)
        # 融合方式：RGB(3) + DEM(1) + RGB_Edge(1) = 5个通道
        fused_features = torch.cat([
            rgb_features,  # 3个通道
            dem_features,  # 1个通道
            rgb_edges_normalized  # 1个通道（边缘信息）
        ], dim=1)

        return fused_features


class RSU7(nn.Module):
    """RSU-7模块: 高度为7的残差U块（添加边缘信息输入）"""

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, edge_channels=0):
        super(RSU7, self).__init__()
        self.out_ch = out_ch

        # 如果包含边缘通道，调整输入通道数
        total_in_ch = in_ch + edge_channels

        # 编码器部分
        self.conv0 = nn.Conv2d(total_in_ch, out_ch, kernel_size=3, padding=1, bias=False)

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

    def forward(self, x, edge_map=None):
        # 如果提供了边缘图，与输入拼接
        if edge_map is not None:
            x = torch.cat([x, edge_map], dim=1)

        # 保存原始输入尺寸
        input_size = x.shape[2:]

        # 第一层卷积
        hx = self.conv0(x)
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

        # 确保输出和输入尺寸一致
        if hx1d.shape[2:] != input_size:
            hx1d = F.interpolate(hx1d, size=input_size, mode='bilinear', align_corners=True)

        # 残差连接
        return hx1d + hx_in


class RSU6(nn.Module):
    """RSU-6模块: 高度为6的残差U块（集成边缘增强）"""

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, edge_channels=0):
        super(RSU6, self).__init__()
        self.out_ch = out_ch

        total_in_ch = in_ch + edge_channels
        self.conv0 = nn.Conv2d(total_in_ch, out_ch, kernel_size=3, padding=1, bias=False)


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
        # 保存原始输入尺寸
        input_size = x.shape[2:]

        # 第一层卷积
        hx = self.conv0(x)
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

        # 确保输出和输入尺寸一致
        if hx1d.shape[2:] != input_size:
            hx1d = F.interpolate(hx1d, size=input_size, mode='bilinear', align_corners=True)

        # 残差连接
        return hx1d + hx_in


class RSU5(nn.Module):
    """RSU-5模块: 高度为5的残差U块（集成边缘增强）"""

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, edge_channels=0):
        super(RSU5, self).__init__()
        self.out_ch = out_ch

        total_in_ch = in_ch + edge_channels

        self.conv0 = nn.Conv2d(total_in_ch, out_ch, kernel_size=3, padding=1, bias=False)


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
        # 保存原始输入尺寸
        input_size = x.shape[2:]

        # 第一层卷积
        hx = self.conv0(x)
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

        # 确保输出和输入尺寸一致
        if hx1d.shape[2:] != input_size:
            hx1d = F.interpolate(hx1d, size=input_size, mode='bilinear', align_corners=True)

        # 残差连接
        return hx1d + hx_in


class RSU4(nn.Module):
    """RSU-4模块: 高度为4的残差U块（集成边缘增强）"""

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, edge_channels=0):
        super(RSU4, self).__init__()
        self.out_ch = out_ch
        total_in_ch = in_ch + edge_channels

        self.conv0 = nn.Conv2d(total_in_ch, out_ch, kernel_size=3, padding=1, bias=False)

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
        # 保存原始输入尺寸
        input_size = x.shape[2:]

        # 第一层卷积
        hx = self.conv0(x)
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

        # 确保输出和输入尺寸一致
        if hx1d.shape[2:] != input_size:
            hx1d = F.interpolate(hx1d, size=input_size, mode='bilinear', align_corners=True)

        # 残差连接
        return hx1d + hx_in


class RSU4F(nn.Module):
    """RSU-4F模块: 无下采样的RSU-4（使用空洞卷积，集成边缘增强）"""

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, edge_channels=0):
        super(RSU4F, self).__init__()
        self.out_ch = out_ch
        total_in_ch = in_ch + edge_channels

        self.conv0 = nn.Conv2d(total_in_ch, out_ch, kernel_size=3, padding=1, bias=False)

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


class U2NET_EdgeEnhanced_Traditional(nn.Module):
    """U^2-Net模型 - 传统边缘增强版本，针对滑坡语义分割优化"""

    def __init__(self, n_classes=1, use_edge_fusion=True):
        super(U2NET_EdgeEnhanced_Traditional, self).__init__()

        self.use_edge_fusion = use_edge_fusion

        # 边缘感知融合模块
        if use_edge_fusion:
            self.edge_fusion = EdgeAwareFusion()
            # 输入通道：RGB(3) + DEM(1) + RGB_Edge(1) = 5
            input_channels = 5
        else:
            # 基本融合：RGB(3) + DEM(1) = 4
            input_channels = 4

        # 编码器 (RSU模块)
        self.stage1 = RSU7(input_channels, 32, 64)
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
        # 边缘感知融合
        if self.use_edge_fusion:
            x = self.edge_fusion(optical, dem)
        else:
            # 基本融合
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

    def extract_rgb_edges(self, optical):
        """提取RGB边缘图（用于可视化）"""
        edge_detector = TraditionalEdgeDetection(
            use_sobel=True,
            use_prewitt=True,
            use_laplacian=False
        )
        return edge_detector(optical)



# 使用示例
# 测试代码
if __name__ == "__main__":
    # 测试传统边缘检测
    edge_detector = TraditionalEdgeDetection()
    test_input = torch.randn(2, 3, 256, 256)
    edges = edge_detector(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Edge output shape: {edges.shape}")

    # 测试边缘增强U2Net
    model = U2NET_EdgeEnhanced_Traditional(n_classes=1, use_edge_fusion=True)

    # 创建测试输入
    batch_size = 2
    optical = torch.randn(batch_size, 3, 256, 256)  # RGB图像
    dem = torch.randn(batch_size, 1, 256, 256)  # DEM数据

    # 前向传播
    output = model(optical, dem)

    print(f"\nModel input - Optical shape: {optical.shape}")
    print(f"Model input - DEM shape: {dem.shape}")
    print(f"Model output shape: {output.shape}")

    # 测试参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 提取边缘可视化
    rgb_edges = model.extract_rgb_edges(optical)
    print(f"RGB edges shape: {rgb_edges.shape}")