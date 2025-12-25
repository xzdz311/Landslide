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
import pandas as pd

def find_file_in_dirs(base_name, search_dirs, extensions='.png'):
    """在多个目录中查找文件（尝试多种扩展名）"""
    for search_dir in search_dirs:
        if search_dir and os.path.exists(search_dir):
            for ext in extensions:
                file_path = os.path.join(search_dir, base_name + ext)
                if os.path.exists(file_path):
                    return file_path

    # 如果没找到，返回可能的第一个路径（用于创建空数据）
    first_dir = next((d for d in search_dirs if d), None)
    if first_dir:
        return os.path.join(first_dir, base_name + '.png')
    return None



class TestDataset(Dataset):
    """简化的测试数据集类"""

    def __init__(self, csv_path, data_dir, target_size=(256, 256)):
        """
        Args:
            csv_path: detailed_results.csv文件路径
            data_dir: 数据集根目录
            target_size: 目标图像尺寸
        """
        self.csv_path = csv_path
        self.data_dir = data_dir
        self.target_size = target_size

        # 读取CSV文件
        self.df = pd.read_csv(csv_path)

        # 定义可能的数据目录结构
        self.landslide_dirs = {
            'image': [
                os.path.join(data_dir, 'landslide', 'test', 'images'),
                os.path.join(data_dir, 'landslide', 'test', 'image'),
                os.path.join(data_dir, 'landslide', 'images'),
                os.path.join(data_dir, 'landslide', 'image')
            ],
            'mask': [
                os.path.join(data_dir, 'landslide', 'test', 'mask'),
                os.path.join(data_dir, 'landslide', 'test', 'masks'),
                os.path.join(data_dir, 'landslide', 'mask'),
                os.path.join(data_dir, 'landslide', 'masks')
            ],
        }

        self.non_landslide_dirs = {
            'image': [
                os.path.join(data_dir, 'non-landslide', 'test', 'images'),
                os.path.join(data_dir, 'non-landslide', 'test', 'image'),
                os.path.join(data_dir, 'non-landslide', 'images'),
                os.path.join(data_dir, 'non-landslide', 'image')
            ],
            'mask': [],  # 非滑坡没有mask
        }

        # 准备图像路径列表
        self.prepare_image_paths()

        # 数据变换（仅归一化）
        self.transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406, 0.5), std=(0.229, 0.224, 0.225, 0.25)),
            ToTensorV2()
        ])

        print(f"从CSV加载了 {len(self.image_paths)} 个测试样本")

    def prepare_image_paths(self):
        """准备所有测试图像的路径"""
        self.image_paths = []
        self.mask_paths = []
        self.base_names = []

        for idx, row in self.df.iterrows():
            image_name = row['image']
            # 移除扩展名（如果有）
            base_name = os.path.splitext(image_name)[0]
            self.base_names.append(base_name)

            # 尝试在滑坡目录中查找
            img_path = find_file_in_dirs(base_name, self.landslide_dirs['image'])
            mask_path = find_file_in_dirs(base_name, self.landslide_dirs['mask'])

            # 如果在滑坡目录没找到，尝试非滑坡目录
            if img_path is None or not os.path.exists(img_path):
                img_path = find_file_in_dirs(base_name, self.non_landslide_dirs['image'])
                mask_path = None  # 非滑坡没有mask

            # 如果还没找到，尝试直接路径查找
            if img_path is None or not os.path.exists(img_path):
                # 尝试直接查找文件
                potential_paths = [
                    os.path.join(self.data_dir, image_name),
                    os.path.join(self.data_dir, 'test', image_name),
                    os.path.join(self.data_dir, 'images', image_name),
                    os.path.join(self.data_dir, 'test_images', image_name)
                ]
                for path in potential_paths:
                    if os.path.exists(path):
                        img_path = path
                        break

            # 添加到列表
            self.image_paths.append(img_path)
            self.mask_paths.append(mask_path)

            # 打印警告（可选）
            if img_path is None or not os.path.exists(img_path):
                print(f"警告: 未找到图像 {image_name}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 获取基础信息
        base_name = self.base_names[idx]
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # 读取光学图像
        if img_path and os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((*self.target_size[::-1], 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.zeros((*self.target_size[::-1], 3), dtype=np.uint8)

        # 读取或创建掩膜
        if mask_path and os.path.exists(mask_path):
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
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)



        combined = np.concatenate([img], axis=-1)

        # 应用变换
        if self.transform:
            augmented = self.transform(image=combined, mask=mask)
            combined = augmented['image']
            mask = augmented['mask']

        # 分离通道
        optical = combined[:3, :, :]
        mask = mask.unsqueeze(0).float() / 255.0

        # 添加一个标志位：是否为滑坡样本（根据是否有mask判断）
        is_landslide = 1.0 if self.mask_paths[idx] is not None else 0.0

        return optical, mask, is_landslide, base_name


def create_test_loader_from_csv(csv_path, data_dir, batch_size=8, num_workers=2):
    """直接从CSV创建测试集数据加载器"""
    # 创建测试数据集
    test_dataset = TestDataset(
        csv_path=csv_path,
        data_dir=data_dir,
        target_size=(256, 256)
    )

    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return test_loader, test_dataset


def predict_and_evaluate(model, test_loader, device='cuda', save_dir='predictions', multigpu=False):
    """
    适配EarlyFusionNet的预测评估函数
    修改：支持5个返回值的数据加载器
    """
    import os
    import cv2
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score

    if multigpu and torch.cuda.device_count() > 1:
        device_ids = list(range(torch.cuda.device_count()))
        print(f"使用多GPU评估: {device_ids}")
        model = nn.DataParallel(model, device_ids=device_ids)
        # 设置主设备
        if isinstance(device, str):
            device = torch.device(f'cuda:{device_ids[0]}')

    model = model.to(device)
    model.eval()

    model.eval()

    all_preds = []
    all_masks = []
    metrics = {'iou': [], 'precision': [], 'recall': [], 'f1': [], 'accuracy': []}
    sample_results = []  # 保存每个样本的结果

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc='Testing')):
            # ===== 修改这里：支持多种数据格式 =====
            if len(batch) == 3:
                # 格式: (optical, dem, mask, img_paths)
                optical, mask, img_paths = batch
                is_landslide = None
            elif len(batch) == 4:
                # 格式: (optical, dem, mask, is_landslide, img_paths)
                optical, mask, is_landslide, img_paths = batch
            else:
                raise ValueError(f"意外的batch长度: {len(batch)}")
            # ===== 修改结束 =====

            optical = optical.to(device)

            mask = mask.cpu()  # 在CPU上处理mask

            # 修改点1: EarlyFusionNet直接输出logits
            outputs = model(optical)

            # 修改点2: 通过sigmoid得到概率，然后阈值化
            pred_probs = torch.sigmoid(outputs).cpu()
            preds = (pred_probs > 0.5).float()

            # 保存预测结果
            for j in range(len(img_paths)):
                img_name = os.path.basename(img_paths[j])
                # 去掉可能的扩展名
                base_name = os.path.splitext(img_name)[0]

                # 保存预测掩膜
                pred_mask = preds[j].squeeze().numpy()  # [H, W]
                pred_mask_uint8 = (pred_mask * 255).astype(np.uint8)

                # 保存原始预测（浮点数概率）
                pred_prob = pred_probs[j].squeeze().numpy()

                # 保存可视化结果（如果有真实掩膜）
                if mask[j].sum() > 0:
                    try:
                        # 尝试读取原始图像
                        if os.path.exists(img_paths[j]):
                            orig_img = cv2.imread(img_paths[j])
                            if orig_img is not None:
                                orig_img = cv2.resize(orig_img, (256, 256))

                                # 可视化
                                fig, axs = plt.subplots(1, 4, figsize=(20, 5))

                                # 原始图像
                                axs[0].imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
                                axs[0].set_title('Original Image')
                                axs[0].axis('off')

                                # 真实掩膜
                                axs[2].imshow(mask[j].squeeze().numpy(), cmap='gray')
                                axs[2].set_title('Ground Truth')
                                axs[2].axis('off')

                                # 预测结果
                                axs[3].imshow(pred_prob, cmap='jet', vmin=0, vmax=1)
                                axs[3].set_title(f'Prediction')
                                axs[3].axis('off')

                                plt.tight_layout()
                                plt.close()
                    except Exception as e:
                        print(f"可视化 {img_name} 时出错: {e}")

            # 仅对有真实标签的样本计算指标
            valid_indices = [j for j in range(len(img_paths)) if mask[j].sum() > 0]
            if valid_indices:
                valid_preds = preds[valid_indices]
                valid_masks = mask[valid_indices]
                valid_names = [os.path.basename(img_paths[j]) for j in valid_indices]

                for idx, (pred, true, name) in enumerate(zip(valid_preds, valid_masks, valid_names)):
                    y_true = true.squeeze().numpy().flatten()
                    y_pred = pred.squeeze().numpy().flatten()

                    # 二值化
                    y_true_bin = (y_true > 0.5).astype(int)
                    y_pred_bin = (y_pred > 0.5).astype(int)

                    # 计算指标
                    iou = jaccard_score(y_true_bin, y_pred_bin, zero_division=0)
                    precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
                    recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
                    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
                    accuracy = np.mean(y_true_bin == y_pred_bin)

                    # 保存每个样本的指标
                    metrics['iou'].append(iou)
                    metrics['precision'].append(precision)
                    metrics['recall'].append(recall)
                    metrics['f1'].append(f1)
                    metrics['accuracy'].append(accuracy)

                    # 记录样本结果
                    sample_results.append({
                        'image': name,
                        'iou': iou,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'accuracy': accuracy,
                        'true_positives': np.sum((y_true_bin == 1) & (y_pred_bin == 1)),
                        'false_positives': np.sum((y_true_bin == 0) & (y_pred_bin == 1)),
                        'false_negatives': np.sum((y_true_bin == 1) & (y_pred_bin == 0)),
                        'true_negatives': np.sum((y_true_bin == 0) & (y_pred_bin == 0))
                    })

    # 计算总体统计
    if metrics['iou']:
        print("\n" + "=" * 60)
        print("总体评估结果:")
        print("=" * 60)

        for metric in ['iou', 'precision', 'recall', 'f1', 'accuracy']:
            values = metrics[metric]
            if values:
                print(f"{metric.upper():12s}: {np.mean(values):.4f} ± {np.std(values):.4f}")
                print(f"  范围: [{np.min(values):.4f}, {np.max(values):.4f}]")

        # 保存详细结果
        import pandas as pd
        df_results = pd.DataFrame(sample_results)
        df_results.to_csv(os.path.join(save_dir, 'detailed_results.csv'), index=False)

        # 保存汇总统计
        summary_stats = {
            'metric': ['iou', 'precision', 'recall', 'f1', 'accuracy'],
            'mean': [np.mean(metrics[m]) for m in ['iou', 'precision', 'recall', 'f1', 'accuracy']],
            'std': [np.std(metrics[m]) for m in ['iou', 'precision', 'recall', 'f1', 'accuracy']],
            'min': [np.min(metrics[m]) for m in ['iou', 'precision', 'recall', 'f1', 'accuracy']],
            'max': [np.max(metrics[m]) for m in ['iou', 'precision', 'recall', 'f1', 'accuracy']]
        }
        pd.DataFrame(summary_stats).to_csv(os.path.join(save_dir, 'summary_stats.csv'), index=False)

        # 混淆矩阵总计
        total_tp = sum([r['true_positives'] for r in sample_results])
        total_fp = sum([r['false_positives'] for r in sample_results])
        total_fn = sum([r['false_negatives'] for r in sample_results])
        total_tn = sum([r['true_negatives'] for r in sample_results])

        print("\n混淆矩阵总计:")
        print(f"True Positives:  {total_tp}")
        print(f"False Positives: {total_fp}")
        print(f"False Negatives: {total_fn}")
        print(f"True Negatives:  {total_tn}")

        # 从总计计算宏观指标
        macro_precision = total_tp / (total_tp + total_fp + 1e-10)
        macro_recall = total_tp / (total_tp + total_fn + 1e-10)
        macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall + 1e-10)
        macro_iou = total_tp / (total_tp + total_fp + total_fn + 1e-10)

        print("\n宏观指标（从总计计算）:")
        print(f"Macro IoU:       {macro_iou:.4f}")
        print(f"Macro Precision: {macro_precision:.4f}")
        print(f"Macro Recall:    {macro_recall:.4f}")
        print(f"Macro F1:        {macro_f1:.4f}")

        results = {
            'micro_iou': np.mean(metrics['iou']),
            'micro_precision': np.mean(metrics['precision']),
            'micro_recall': np.mean(metrics['recall']),
            'micro_f1': np.mean(metrics['f1']),
            'macro_iou': macro_iou,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'sample_count': len(sample_results)
        }
    else:
        print("警告：没有找到有效的真值掩膜进行评估")
        results = {}

    return results




class RSU7(nn.Module):
    """RSU-7模块: 高度为7的残差U块"""

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()
        self.out_ch = out_ch

        # 编码器部分
        self.conv0 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)

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

    def forward(self, x):
        # 保存原始输入用于残差连接
        x_input = x

        # 第一层卷积
        hx = self.conv0(x_input)
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

        # 确保hx1d和hx_in大小一致，如果不一致则调整hx_in
        if hx1d.shape != hx_in.shape:
            # 调整hx_in的大小以匹配hx1d
            hx_in_adjusted = F.interpolate(hx_in, size=hx1d.shape[2:], mode='bilinear', align_corners=True)
        else:
            hx_in_adjusted = hx_in

        # 残差连接
        return hx1d + hx_in_adjusted


class RSU6(nn.Module):
    """RSU-6模块: 高度为6的残差U块"""

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()
        self.out_ch = out_ch

        self.conv0 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)

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
        # 保存原始输入用于残差连接
        x_input = x

        # 第一层卷积
        hx = self.conv0(x_input)
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

        # 确保hx1d和hx_in大小一致
        if hx1d.shape != hx_in.shape:
            hx_in_adjusted = F.interpolate(hx_in, size=hx1d.shape[2:], mode='bilinear', align_corners=True)
        else:
            hx_in_adjusted = hx_in

        # 残差连接
        return hx1d + hx_in_adjusted


class RSU5(nn.Module):
    """RSU-5模块: 高度为5的残差U块"""

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()
        self.out_ch = out_ch

        self.conv0 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)

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
        # 保存原始输入用于残差连接
        x_input = x

        # 第一层卷积
        hx = self.conv0(x_input)
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

        # 确保hx1d和hx_in大小一致
        if hx1d.shape != hx_in.shape:
            hx_in_adjusted = F.interpolate(hx_in, size=hx1d.shape[2:], mode='bilinear', align_corners=True)
        else:
            hx_in_adjusted = hx_in

        # 残差连接
        return hx1d + hx_in_adjusted


class RSU4(nn.Module):
    """RSU-4模块: 高度为4的残差U块"""

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()
        self.out_ch = out_ch

        self.conv0 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)

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
        # 保存原始输入用于残差连接
        x_input = x

        # 第一层卷积
        hx = self.conv0(x_input)
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

        # 确保hx1d和hx_in大小一致
        if hx1d.shape != hx_in.shape:
            hx_in_adjusted = F.interpolate(hx_in, size=hx1d.shape[2:], mode='bilinear', align_corners=True)
        else:
            hx_in_adjusted = hx_in

        # 残差连接
        return hx1d + hx_in_adjusted


class RSU4F(nn.Module):
    """RSU-4F模块: 无下采样的RSU-4（使用空洞卷积）"""

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()
        self.out_ch = out_ch

        self.conv0 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)

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


class U2NET(nn.Module):
    """U^2-Net模型 - 早期融合版本，输入输出与原始U-Net保持一致"""

    def __init__(self, n_channels=4, n_classes=1):
        super(U2NET, self).__init__()

        # 编码器 (RSU模块)
        self.stage1 = RSU7(n_channels, 32, 64)
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

    def forward(self, optical,):
        # 早期融合: 在通道维度拼接
        x = torch.cat([optical], dim=1)

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



def get_simple_training_config():
    """获取简单训练配置"""

    # 1. 创建简单模型
    model = U2NET(n_channels=3, n_classes=1)

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

    model.load_state_dict(torch.load(r'F:\zx\模型结果及参数\final_U2NET_nodem_model.pth', map_location=torch.device('cpu')))
    model.eval()

    # 2. 运行评估
    results = predict_and_evaluate(
        model=model,
        test_loader=test_loader,  # 你的测试数据加载器
        device='cpu',
        save_dir='F:\zx\predictions_results\predictions_results_u2netnodem',
        multigpu=True
    )


# 运行调试
if __name__ == "__main__":
    main()

