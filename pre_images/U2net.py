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


# 检查GPU数量
def setup_multigpu():
    """设置多GPU环境"""
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 个GPU")

    if num_gpus > 1:
        print("启用多GPU训练")
        # 设置设备ID
        device_ids = list(range(num_gpus))
        return device_ids
    else:
        print("单GPU训练")
        return None


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


# 5. 数据集类
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

    os.makedirs(save_dir, exist_ok=True)

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
            if len(batch) == 4:
                # 格式: (optical, dem, mask, img_paths)
                optical, dem, mask, img_paths = batch
                is_landslide = None
            elif len(batch) == 5:
                # 格式: (optical, dem, mask, is_landslide, img_paths)
                optical, dem, mask, is_landslide, img_paths = batch
            else:
                raise ValueError(f"意外的batch长度: {len(batch)}")
            # ===== 修改结束 =====

            optical = optical.to(device)
            dem = dem.to(device)
            mask = mask.cpu()  # 在CPU上处理mask

            # 修改点1: EarlyFusionNet直接输出logits
            outputs = model(optical, dem)

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
                np.save(os.path.join(save_dir, f'prob_{base_name}.npy'), pred_prob)

                # 保存二值化预测
                cv2.imwrite(os.path.join(save_dir, f'pred_{base_name}.png'), pred_mask_uint8)

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

                                # DEM数据（可选）
                                axs[1].imshow(dem[j].squeeze().cpu().numpy(), cmap='terrain')
                                axs[1].set_title('DEM Data')
                                axs[1].axis('off')

                                # 真实掩膜
                                axs[2].imshow(mask[j].squeeze().numpy(), cmap='gray')
                                axs[2].set_title('Ground Truth')
                                axs[2].axis('off')

                                # 预测结果
                                axs[3].imshow(pred_prob, cmap='jet', vmin=0, vmax=1)
                                axs[3].set_title(f'Prediction')
                                axs[3].axis('off')

                                plt.tight_layout()
                                plt.savefig(os.path.join(save_dir, f'vis_{base_name}.png'),
                                            bbox_inches='tight', dpi=100)
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


def visualize_predictions_comparison(model, test_loader, device='cuda', num_samples=5):
    """
    可视化预测对比（单独函数，更清晰）
    """
    import matplotlib.pyplot as plt

    model.eval()

    with torch.no_grad():
        for i, (optical, dem, mask, img_paths) in enumerate(test_loader):
            if i >= 1:  # 只取第一个batch
                break

            optical = optical.to(device)
            dem = dem.to(device)

            outputs = model(optical, dem)
            pred_probs = torch.sigmoid(outputs).cpu()

            # 显示前几个样本
            num_show = min(num_samples, len(optical))

            fig, axes = plt.subplots(num_show, 4, figsize=(16, num_show * 4))
            if num_show == 1:
                axes = axes.reshape(1, -1)

            for idx in range(num_show):
                # 光学图像
                axes[idx, 0].imshow(optical[idx].cpu().permute(1, 2, 0).numpy())
                axes[idx, 0].set_title('Optical Image')
                axes[idx, 0].axis('off')

                # DEM数据
                axes[idx, 1].imshow(dem[idx].cpu().squeeze().numpy(), cmap='terrain')
                axes[idx, 1].set_title('DEM Data')
                axes[idx, 1].axis('off')

                # 真实掩膜
                if mask[idx].sum() > 0:
                    axes[idx, 2].imshow(mask[idx].squeeze().numpy(), cmap='gray')
                axes[idx, 2].set_title('Ground Truth')
                axes[idx, 2].axis('off')

                # 预测结果
                pred_prob = pred_probs[idx].squeeze().numpy()
                im = axes[idx, 3].imshow(pred_prob, cmap='jet', vmin=0, vmax=1)
                axes[idx, 3].set_title('Prediction')
                axes[idx, 3].axis('off')

                # 添加颜色条
                plt.colorbar(im, ax=axes[idx, 3], fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.savefig('predictions_comparison.png', dpi=150, bbox_inches='tight')
            plt.show()
            break



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

    def forward(self, optical, dem):
        # 早期融合: 在通道维度拼接
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



def get_simple_training_config():
    """获取简单训练配置"""

    # 1. 创建简单模型
    model = U2NET(n_channels=4, n_classes=1)

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


def load_model_with_multigpu_support(model, model_path):
    """
    加载模型，自动处理多GPU训练的权重

    参数:
        model: 模型实例
        model_path: 权重文件路径

    返回:
        model: 加载权重后的模型
    """
    # 加载权重
    checkpoint = torch.load(model_path, map_location='cpu')

    # 提取state_dict
    if isinstance(checkpoint, dict):
        # 检查是完整checkpoint还是直接state_dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # 检查是否是多GPU权重
    if any(key.startswith('module.') for key in state_dict.keys()):
        print("🔄 处理多GPU训练权重...")
        # 移除'module.'前缀
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]  # 去掉'module.'
            else:
                new_key = key
            new_state_dict[new_key] = value
        state_dict = new_state_dict

    # 加载权重
    model.load_state_dict(state_dict)
    model.eval()

    print(f" 模型权重已加载（{len(state_dict)}个参数）")
    return model


def main():
    """主训练函数"""

    # 获取配置
    model, criterion, optimizer, scheduler = get_simple_training_config()

    print(f"模型架构: {model.__class__.__name__}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    device_ids = list(range(torch.cuda.device_count()))
    print(f"可用的GPU: {device_ids}")

    # 设置
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据准备（使用新函数）
    data_dir = r"D:\ly\landsint\Bijie-landslide-dataset"
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

    model.load_state_dict(torch.load(r'D:\ly\landsint\result\final_U2NET_model.pth', map_location=torch.device('cpu')))

    model.eval()

    # 2. 运行评估
    results = predict_and_evaluate(
        model=model,
        test_loader=test_loader,  # 你的测试数据加载器
        device='cpu',
        save_dir=r'D:\ly\landsint\result\predictions_results_U2net',
        multigpu=True
    )


# 运行调试
if __name__ == "__main__":
    main()

