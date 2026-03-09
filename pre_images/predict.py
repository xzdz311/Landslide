import cv2
import pandas as pd
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
            'dem': [
                os.path.join(data_dir, 'landslide', 'test', 'dem'),
                os.path.join(data_dir, 'landslide', 'test', 'dems'),
                os.path.join(data_dir, 'landslide', 'dem'),
                os.path.join(data_dir, 'landslide', 'dems')
            ]
        }

        self.non_landslide_dirs = {
            'image': [
                os.path.join(data_dir, 'non-landslide', 'test', 'images'),
                os.path.join(data_dir, 'non-landslide', 'test', 'image'),
                os.path.join(data_dir, 'non-landslide', 'images'),
                os.path.join(data_dir, 'non-landslide', 'image')
            ],
            'mask': [],  # 非滑坡没有mask
            'dem': [
                os.path.join(data_dir, 'non-landslide', 'test', 'dem'),
                os.path.join(data_dir, 'non-landslide', 'test', 'dems'),
                os.path.join(data_dir, 'non-landslide', 'dem'),
                os.path.join(data_dir, 'non-landslide', 'dems')
            ]
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
        self.dem_paths = []
        self.mask_paths = []
        self.base_names = []

        for idx, row in self.df.iterrows():
            image_name = row['image']
            # 移除扩展名（如果有）
            base_name = os.path.splitext(image_name)[0]
            self.base_names.append(base_name)

            # 尝试在滑坡目录中查找
            img_path = find_file_in_dirs(base_name, self.landslide_dirs['image'])
            dem_path = find_file_in_dirs(base_name, self.landslide_dirs['dem'])
            mask_path = find_file_in_dirs(base_name, self.landslide_dirs['mask'])

            # 如果在滑坡目录没找到，尝试非滑坡目录
            if img_path is None or not os.path.exists(img_path):
                img_path = find_file_in_dirs(base_name, self.non_landslide_dirs['image'])
                dem_path = find_file_in_dirs(base_name, self.non_landslide_dirs['dem'])
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
            self.dem_paths.append(dem_path)
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
        dem_path = self.dem_paths[idx]
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

        # 读取DEM
        if dem_path and os.path.exists(dem_path):
            dem = read_dem(dem_path)
        else:
            dem = np.zeros((256, 256), dtype=np.float32)

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

        # 添加一个标志位：是否为滑坡样本（根据是否有mask判断）
        is_landslide = 1.0 if self.mask_paths[idx] is not None else 0.0

        return optical, dem, mask, is_landslide, base_name


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


def visualize_predictions_comparison(model, test_loader, device='cpu', num_samples=5):
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


def predict_and_evaluate_att(model, test_loader, device='cuda', save_dir='predictions', multigpu=False, use_tta=True,
                         threshold=0.5):
    """
    U2Net_RS_Final 专用预测评估函数 (集成 TTA)

    参数:
        use_tta (bool): 是否开启测试时增强 (Test Time Augmentation)，推荐开启以冲击高分
        threshold (float): 二值化阈值，建议尝试 0.4 或 0.5
    """
    import os
    import cv2
    import numpy as np
    import torch
    import torch.nn as nn  # 确保引入 nn
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
    import pandas as pd  # 移到这里

    os.makedirs(save_dir, exist_ok=True)

    # 多GPU处理
    if multigpu and torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU进行评估")
        # 只有当模型还不是DataParallel时才包装
        if not isinstance(model, nn.DataParallel):
            model = nn.DataParallel(model)

    model = model.to(device)
    model.eval()

    metrics = {'iou': [], 'precision': [], 'recall': [], 'f1': [], 'accuracy': []}
    sample_results = []

    print(f"开始评估 (TTA={'开启' if use_tta else '关闭'}, Threshold={threshold})...")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc='Testing')):
            # 1. 解包数据
            if len(batch) == 4:
                optical, dem, mask, img_paths = batch
            elif len(batch) == 5:
                optical, dem, mask, is_landslide, img_paths = batch
            else:
                continue  # 跳过异常batch

            optical = optical.to(device)
            dem = dem.to(device)
            # mask 保持在 CPU 用于计算指标

            # 2. 模型预测 (核心修改：集成 TTA)
            if use_tta:
                # A. 原图预测
                out_raw = model(optical, dem)
                if isinstance(out_raw, (tuple, list)): out_raw = out_raw[0]  # 兼容多输出
                pred_raw = torch.sigmoid(out_raw)

                # B. 水平翻转预测
                opt_h = torch.flip(optical, [3])
                dem_h = torch.flip(dem, [3])
                out_h = model(opt_h, dem_h)
                if isinstance(out_h, (tuple, list)): out_h = out_h[0]
                pred_h = torch.sigmoid(out_h)
                pred_h = torch.flip(pred_h, [3])  # 翻转回来

                # C. 垂直翻转预测
                opt_v = torch.flip(optical, [2])
                dem_v = torch.flip(dem, [2])
                out_v = model(opt_v, dem_v)
                if isinstance(out_v, (tuple, list)): out_v = out_v[0]
                pred_v = torch.sigmoid(out_v)
                pred_v = torch.flip(pred_v, [2])  # 翻转回来

                # D. 融合 (取平均)
                pred_probs = (pred_raw + pred_h + pred_v) / 3.0
            else:
                # 不使用 TTA
                outputs = model(optical, dem)
                # 兼容性检查：如果返回的是 tuple (d0, d1...), 取 d0
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                pred_probs = torch.sigmoid(outputs)

            # 转回 CPU
            pred_probs = pred_probs.cpu()

            # 3. 阈值化
            preds = (pred_probs > threshold).float()

            # 4. 逐样本处理
            for j in range(len(img_paths)):
                img_path = img_paths[j]
                img_name = os.path.basename(img_path)
                base_name = os.path.splitext(img_name)[0]

                # 获取当前样本的预测和真值
                curr_pred_prob = pred_probs[j].squeeze().numpy()
                curr_pred_mask = preds[j].squeeze().numpy()
                curr_true_mask = mask[j].squeeze().numpy() if mask is not None else None

                # 保存预测图 (可选，防止磁盘写满，可以注释掉)
                # cv2.imwrite(os.path.join(save_dir, f'pred_{base_name}.png'), (curr_pred_mask * 255).astype(np.uint8))

                # --- 指标计算 ---
                if curr_true_mask is not None and curr_true_mask.max() > 0:  # 只有当存在真值时才计算
                    # 展平
                    y_true = (curr_true_mask > 0.5).astype(int).flatten()
                    y_pred = curr_pred_mask.astype(int).flatten()

                    # 计算各项指标
                    iou = jaccard_score(y_true, y_pred, zero_division=0)
                    precision = precision_score(y_true, y_pred, zero_division=0)
                    recall = recall_score(y_true, y_pred, zero_division=0)
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                    acc = np.mean(y_true == y_pred)

                    # 记录
                    metrics['iou'].append(iou)
                    metrics['precision'].append(precision)
                    metrics['recall'].append(recall)
                    metrics['f1'].append(f1)
                    metrics['accuracy'].append(acc)

                    # 混淆矩阵计数
                    tp = np.sum((y_true == 1) & (y_pred == 1))
                    fp = np.sum((y_true == 0) & (y_pred == 1))
                    fn = np.sum((y_true == 1) & (y_pred == 0))
                    tn = np.sum((y_true == 0) & (y_pred == 0))

                    sample_results.append({
                        'image': img_name,
                        'iou': iou,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
                    })

                    # 可视化 (仅对IoU较低或特定的样本，避免生成太多图)
                    # 例如：只保存 IoU < 0.5 的坏样本，或者是那个特定的坏案例
                    if iou < 0.5:
                        try:
                            if os.path.exists(img_path):
                                orig_img = cv2.imread(img_path)
                                if orig_img is not None:
                                    orig_img = cv2.resize(orig_img, (curr_pred_mask.shape[1], curr_pred_mask.shape[0]))

                                    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                                    axs[0].imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
                                    axs[0].set_title(f'Image (IoU={iou:.3f})')
                                    axs[1].imshow(curr_true_mask, cmap='gray')
                                    axs[1].set_title('Ground Truth')
                                    axs[2].imshow(curr_pred_prob, cmap='jet', vmin=0, vmax=1)
                                    axs[2].set_title('Prediction (Prob)')
                                    [ax.axis('off') for ax in axs]
                                    plt.savefig(os.path.join(save_dir, f'bad_case_{base_name}.png'))
                                    plt.close()
                        except:
                            pass

    # 5. 汇总统计
    if metrics['iou']:
        print("\n" + "=" * 60)
        print(f"评估完成 (TTA={use_tta})")
        print("=" * 60)

        # 保存详细结果
        df = pd.DataFrame(sample_results)
        df.to_csv(os.path.join(save_dir, 'detailed_metrics.csv'), index=False)

        # 计算宏观指标 (Macro Average - 基于混淆矩阵总和)
        total_tp = df['tp'].sum()
        total_fp = df['fp'].sum()
        total_fn = df['fn'].sum()

        macro_iou = total_tp / (total_tp + total_fp + total_fn + 1e-10)
        macro_f1 = 2 * total_tp / (2 * total_tp + total_fp + total_fn + 1e-10)
        macro_recall = total_tp / (total_tp + total_fn + 1e-10)
        macro_prec = total_tp / (total_tp + total_fp + 1e-10)

        # 计算微观指标 (Micro Average - 基于每张图指标的平均)
        micro_iou = np.mean(metrics['iou'])
        micro_f1 = np.mean(metrics['f1'])

        print(f"{'Metric':<15} {'Micro (Mean)':<15} {'Macro (Global)':<15}")
        print("-" * 45)
        print(f"{'IoU':<15} {micro_iou:.4f}           {macro_iou:.4f}")
        print(f"{'F1 Score':<15} {micro_f1:.4f}           {macro_f1:.4f}")
        print(f"{'Precision':<15} {np.mean(metrics['precision']):.4f}           {macro_prec:.4f}")
        print(f"{'Recall':<15} {np.mean(metrics['recall']):.4f}           {macro_recall:.4f}")

        # 统计分析
        stats = {
            'metric': ['iou', 'f1', 'precision', 'recall'],
            'mean': [micro_iou, micro_f1, np.mean(metrics['precision']), np.mean(metrics['recall'])],
            'std': [np.std(metrics['iou']), np.std(metrics['f1']), np.std(metrics['precision']),
                    np.std(metrics['recall'])],
            'min': [np.min(metrics['iou']), np.min(metrics['f1']), np.min(metrics['precision']),
                    np.min(metrics['recall'])]
        }
        pd.DataFrame(stats).to_csv(os.path.join(save_dir, 'summary_stats.csv'), index=False)

        return {
            'iou': micro_iou,
            'f1': micro_f1,
            'macro_iou': macro_iou
        }
    else:
        return {}