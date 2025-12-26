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


# 优化版本：添加了更多的性能优化
def train_model_multigpu_optimized(model, train_loader, val_loader, criterion, optimizer,
                                   scheduler, num_epochs=30, device_ids=None):
    """
    多GPU训练函数（优化版）

    优化点：
    1. 混合精度训练
    2. 梯度累积（处理大批次）
    3. 内存优化
    4. 更高效的进度显示
    """

    # GPU设置
    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))

    num_gpus = len(device_ids)

    if num_gpus > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
        device = torch.device(f'cuda:{device_ids[0]}')
        print(f"使用 {num_gpus} 个GPU并行训练")
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"使用单GPU训练")

    model = model.to(device)

    # 混合精度训练
    scaler = GradScaler()

    # 梯度累积步数（模拟更大的batch size）
    accumulation_steps = 4

    best_iou = 0.0
    history = {
        'train_loss': [], 'val_loss': [], 'val_iou': [],
        'val_precision': [], 'val_recall': [], 'learning_rate': []
    }

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 40)

        # 训练阶段
        model.train()
        train_loss = 0.0
        batch_count = 0

        # 使用enumerate获取batch索引
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc='Training')

        optimizer.zero_grad()

        for batch_idx, (optical, dem, mask, is_landslide, _) in pbar:
            batch_count += 1

            optical = optical.to(device, non_blocking=True)
            dem = dem.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            # 混合精度前向传播
            with autocast():
                outputs = model(optical, dem)

                if hasattr(criterion, '__code__') and criterion.__code__.co_argcount > 2:
                    loss = criterion(outputs, mask, dem)
                else:
                    loss = criterion(outputs, mask)

                # 梯度累积：损失除以累积步数
                loss = loss / accumulation_steps

            # 反向传播
            scaler.scale(loss).backward()

            # 梯度累积：每accumulation_steps步更新一次
            if (batch_idx + 1) % accumulation_steps == 0:
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # 更新参数
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps

            # 更新进度条
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item() * accumulation_steps:.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })

        # 如果有剩余的梯度，执行一次更新
        if batch_count % accumulation_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        print(f'Train Loss: {avg_train_loss:.4f}')

        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_tp, all_fp, all_fn, all_tn = 0, 0, 0, 0

        # 验证阶段不使用混合精度
        with torch.no_grad(), autocast(enabled=False):
            pbar = tqdm(val_loader, desc='Validation')
            for optical, dem, mask, is_landslide, _ in pbar:
                optical = optical.to(device, non_blocking=True)
                dem = dem.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                outputs = model(optical, dem)

                if hasattr(criterion, '__code__') and criterion.__code__.co_argcount > 2:
                    loss = criterion(outputs, mask, dem)
                else:
                    loss = criterion(outputs, mask)

                val_loss += loss.item()

                pred_probs = torch.sigmoid(outputs)
                preds = (pred_probs > 0.7).float()

                # 收集所有GPU的预测
                if num_gpus > 1:
                    preds = torch.cat([pred for pred in preds], dim=0)
                    mask = torch.cat([m for m in mask], dim=0)

                tp = ((preds == 1) & (mask == 1)).sum().item()
                fp = ((preds == 1) & (mask == 0)).sum().item()
                fn = ((preds == 0) & (mask == 1)).sum().item()
                tn = ((preds == 0) & (mask == 0)).sum().item()

                all_tp += tp
                all_fp += fp
                all_fn += fn
                all_tn += tn

        # 计算指标
        avg_val_loss = val_loss / len(val_loader)

        precision = all_tp / max(all_tp + all_fp, 1)
        recall = all_tp / max(all_tp + all_fn, 1)
        accuracy = (all_tp + all_tn) / max(all_tp + all_fp + all_fn + all_tn, 1)
        iou = all_tp / max(all_tp + all_fp + all_fn, 1)

        # 记录历史
        history['val_loss'].append(avg_val_loss)
        history['val_iou'].append(iou)
        history['val_precision'].append(precision)
        history['val_recall'].append(recall)
        history['learning_rate'].append(optimizer.param_groups[0]["lr"])

        print(f'Val Loss: {avg_val_loss:.4f}')
        print(f'Val IoU: {iou:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
        print(f'学习率: {optimizer.param_groups[0]["lr"]:.6f}')

        # 学习率调度
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        # 保存最佳模型
        if iou > best_iou:
            best_iou = iou
            model_to_save = model.module if num_gpus > 1 else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_iou': best_iou,
                'history': history,
            }, 'best_U2NET_checkpoint.pth')
            print(f'✓ 保存最佳模型检查点，IoU: {best_iou:.4f}')

    return model, history


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



class LandslideAttentionSimple(nn.Module):
    """简化的滑坡注意力模块，避免通道错误"""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.in_channels = in_channels

        # 1. 多尺度感受野提取（简化为并行空洞卷积）
        self.multi_scale = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels // 4, 3, dilation=1, padding=1),
            nn.Conv2d(in_channels, in_channels // 4, 3, dilation=2, padding=2),
            nn.Conv2d(in_channels, in_channels // 4, 3, dilation=4, padding=4),
            nn.Conv2d(in_channels, in_channels // 4, 3, dilation=8, padding=8)
        ])

        # 2. 滑坡特征注意力（学习滑坡的空间分布）
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 1, 1),
            nn.Sigmoid()
        )

        # 3. 通道注意力（强化滑坡相关特征）
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

        # 4. 特征融合卷积（确保输出通道正确）
        self.fusion_conv = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        # 原始特征
        identity = x

        # 1. 多尺度特征提取
        scale_features = []
        for conv in self.multi_scale:
            scale_features.append(conv(x))

        # 拼接多尺度特征
        multi_scale_feat = torch.cat(scale_features, dim=1)

        # 2. 通道注意力
        channel_weights = self.channel_attention(multi_scale_feat)

        # 3. 空间注意力
        spatial_weights = self.spatial_attention(multi_scale_feat)

        # 4. 应用注意力
        attended_feat = multi_scale_feat * channel_weights * spatial_weights

        # 5. 融合回原始通道数
        fused_feat = self.fusion_conv(attended_feat)

        # 6. 残差连接
        output = identity + fused_feat

        return output


# 修复的RSU5增强版（使用简化注意力）
class RSU5_LandslideEnhanced(RSU5):
    """使用简化注意力模块的RSU5增强版"""

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__(in_ch, mid_ch, out_ch)

        # 添加简化的滑坡注意力（更稳定）
        self.landslide_attention = LandslideAttentionSimple(mid_ch)

        # 可选：在conv5后添加注意力
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
        #     nn.BatchNorm2d(mid_ch),
        #     nn.ReLU(inplace=True),
        #     LandslideAttentionSimple(mid_ch)  # 添加注意力
        # )

    def forward(self, x):
        # 调用原始RSU5的前向传播
        hx = self.conv0(x)
        hx_in = hx

        hx1 = self.relu(self.bn1(self.conv1(hx)))
        hx1 = self.landslide_attention(hx1)  # 在第一层后添加注意力
        hx = self.pool1(hx1)

        hx2 = self.relu(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.relu(self.bn4(self.conv4(hx)))

        # 注意：不要在这里添加注意力，避免改变原始结构
        hx5 = self.relu(self.bn5(self.conv5(hx4)))

        # 解码路径保持不变
        hx4d = self.relu(self.bn4d(self.conv4d(torch.cat((hx4, hx5), 1))))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=True)

        hx3d = self.relu(self.bn3d(self.conv3d(torch.cat((hx3, hx4dup), 1))))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=True)

        hx2d = self.relu(self.bn2d(self.conv2d(torch.cat((hx2, hx3dup), 1))))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=True)

        hx1d = self.relu(self.bn1d(self.conv1d(torch.cat((hx1, hx2dup), 1))))

        # 残差连接
        return hx1d + hx_in


class U2NET(nn.Module):
    """U^2-Net模型 - 早期融合版本，加入滑坡增强"""

    def __init__(self, in_ch=4, n_classes=1):
        super(U2NET, self).__init__()

        # ========== 编码器 (RSU模块) ==========
        # 浅层：保持原始RSU1-3（细节很重要）
        self.stage1 = RSU7(in_ch, 32, 64)
        self.stage2 = RSU6(64, 32, 128)
        self.stage3 = RSU5_LandslideEnhanced(128, 64, 256)  # 增强！

        # 中层：可以部分增强
        self.stage4 = RSU4(256, 128, 512)  # 原始或增强

        # 深层：重点增强区域
        self.stage5 = RSU5_LandslideEnhanced(512, 256, 512)  # 增强！再加一个

        # 最深层：原始或增强
        self.stage6 = RSU6(512, 256, 512)  # 保持原始或增强

        self.stage7 = RSU5(512, 256, 512)  # 保持一个原始RSU作为基准

        # ========== 下采样层 ==========
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # ========== 解码器 ==========
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5_LandslideEnhanced(512, 64, 128)  # 解码器也增强！
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

        # ========== 编码路径 ==========
        hx1 = self.stage1(x)
        hx = self.pool12(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        hx3 = self.stage3(hx)  # 增强的RSU5
        hx = self.pool34(hx3)

        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        hx5 = self.stage5(hx)  # 增强的RSU5
        hx = self.pool56(hx5)

        hx6 = self.stage6(hx)
        hx = self.pool56(hx6)  # 额外下采样

        hx7 = self.stage7(hx)
        hx7up = F.interpolate(hx7, size=hx6.shape[2:], mode='bilinear', align_corners=True)

        # ========== 解码路径 ==========
        hx6d = self.stage6d(torch.cat((hx7up, hx6), 1))
        hx6dup = F.interpolate(hx6d, size=hx5.shape[2:], mode='bilinear', align_corners=True)

        hx5d = self.stage5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:], mode='bilinear', align_corners=True)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=True)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))  # 增强的解码器
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=True)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=True)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # ========== 侧边输出 ==========
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


class FixedU2NET(nn.Module):
    """完整修复的U2NET，所有层都已定义"""

    def __init__(self, in_ch=4, n_classes=1, enhancement_type='minimal'):
        """
        Args:
            enhancement_type:
                - 'none': 原始U2NET
                - 'minimal': 最小增强（推荐）
                - 'moderate': 中等增强
        """
        super().__init__()

        self.enhancement_type = enhancement_type

        # ========== 编码器 ==========
        self.stage1 = RSU7(in_ch, 32, 64)
        self.stage2 = RSU6(64, 32, 128)

        # 选择增强类型
        if enhancement_type == 'none':
            self.stage3 = RSU5(128, 64, 256)
            self.stage4 = RSU4(256, 128, 512)
            self.stage5 = RSU5(512, 256, 512)
            self.stage6 = RSU6(512, 256, 512)
        elif enhancement_type == 'minimal':
            self.stage3 = RSU5_LandslideEnhanced(128, 64, 256)
            self.stage4 = RSU4(256, 128, 512)
            self.stage5 = RSU5_LandslideEnhanced(512, 256, 512)
            self.stage6 = RSU6(512, 256, 512)
        else:  # 'moderate'
            self.stage3 = RSU5_LandslideEnhanced(128, 64, 256)
            self.stage4 = RSU4(256, 128, 512)
            self.stage5 = RSU5_LandslideEnhanced(512, 256, 512)
            self.stage6 = RSU6(512, 256, 512)

        # U2NET标准结构有stage7
        self.stage7 = RSU5(512, 256, 512)

        # ========== 下采样层 ==========
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # ========== 解码器 ==========
        # 注意：U2NET标准解码器有stage6d，不是stage5d开始
        self.stage6d = RSU4F(1024, 256, 512)  # 修复：添加stage6d
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)

        # 解码器增强（对称）
        if enhancement_type != 'none':
            self.stage3d = RSU5_LandslideEnhanced(512, 64, 128)
        else:
            self.stage3d = RSU5(512, 64, 128)

        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        # ========== 侧边输出 ==========
        self.side1 = nn.Conv2d(64, n_classes, 3, padding=1)
        self.side2 = nn.Conv2d(64, n_classes, 3, padding=1)
        self.side3 = nn.Conv2d(128, n_classes, 3, padding=1)
        self.side4 = nn.Conv2d(256, n_classes, 3, padding=1)
        self.side5 = nn.Conv2d(512, n_classes, 3, padding=1)
        self.side6 = nn.Conv2d(512, n_classes, 3, padding=1)

        # ========== 最终融合层 ==========
        self.outconv = nn.Conv2d(6 * n_classes, n_classes, 1)

    def forward(self, optical, dem):
        # 早期融合
        x = torch.cat([optical, dem], dim=1)

        # ========== 编码路径 ==========
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
        hx = self.pool56(hx6)  # 对hx6也下采样

        hx7 = self.stage7(hx)

        # ========== 解码路径 ==========
        # stage7 -> stage6d
        hx7up = F.interpolate(hx7, size=hx6.shape[2:], mode='bilinear', align_corners=True)
        hx6d = self.stage6d(torch.cat((hx7up, hx6), 1))  # 使用stage6d

        # stage6d -> stage5d
        hx6dup = F.interpolate(hx6d, size=hx5.shape[2:], mode='bilinear', align_corners=True)
        hx5d = self.stage5d(torch.cat((hx6dup, hx5), 1))

        # stage5d -> stage4d
        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:], mode='bilinear', align_corners=True)
        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))

        # stage4d -> stage3d
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=True)
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))

        # stage3d -> stage2d
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=True)
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))

        # stage2d -> stage1d
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=True)
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # ========== 侧边输出 ==========
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

        # ========== 融合输出 ==========
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return d0


def extract_boundary_simple(mask, dilation_iters=1):
    """
    简单边界提取：通过膨胀和腐蚀的差值得到边界
    Args:
        mask: [B, 1, H, W] 二值分割掩码 (0/1)
        dilation_iters: 膨胀迭代次数，控制边界宽度
    Returns:
        boundary: [B, 1, H, W] 边界区域掩码
    """
    # 确保是二值掩码
    if mask.max() > 1:
        mask = (mask > 0.5).float()

    # 定义结构元素（3x3十字形，对滑坡长条形特征更友好）
    kernel = torch.tensor([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=torch.float32, device=mask.device).view(1, 1, 3, 3)

    # 膨胀操作
    dilated = mask
    for _ in range(dilation_iters):
        dilated = F.conv2d(dilated, kernel, padding=1)
        dilated = (dilated > 0).float()  # 保持二值

    # 腐蚀操作
    eroded = mask
    for _ in range(dilation_iters):
        # 需要确保所有像素都有足够的邻居
        eroded = F.conv2d(eroded, kernel, padding=1)
        eroded = (eroded >= kernel.sum()).float()  # 只有完全匹配才保留

    # 边界 = 膨胀 - 腐蚀
    boundary = dilated - eroded
    boundary = torch.clamp(boundary, 0, 1)

    return boundary


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BoundarySensitiveDiceLoss(nn.Module):
    """
    边界敏感Dice Loss：专门针对滑坡模糊边界优化
    """

    def __init__(self, boundary_width=2, alpha=1.0, beta=2.0, eps=1e-6):
        """
        Args:
            boundary_width: 边界像素宽度
            alpha: 内部区域权重
            beta: 边界区域权重（beta > alpha表示更关注边界）
            eps: 数值稳定性
        """
        super().__init__()
        self.boundary_width = boundary_width
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1, H, W] 网络预测的概率图或logits
            target: [B, 1, H, W] 真实二值掩码
        """
        # 确保值范围在[0, 1]
        if pred.min() < 0 or pred.max() > 1:
            pred_sigmoid = torch.sigmoid(pred)
        else:
            pred_sigmoid = pred

        if target.max() > 1:
            target_binary = (target > 0.5).float()
        else:
            target_binary = target

        # 提取边界区域
        boundary_mask = self.extract_boundary_with_weights(target_binary)

        # 计算加权Dice Loss
        # 边界区域使用beta权重，内部区域使用alpha权重
        weight_map = self.alpha + (self.beta - self.alpha) * boundary_mask

        # 加权交集和并集
        intersection = (weight_map * pred_sigmoid * target_binary).sum(dim=(2, 3))
        union = (weight_map * (pred_sigmoid + target_binary)).sum(dim=(2, 3))

        # Dice系数
        dice = (2. * intersection + self.eps) / (union + self.eps)
        loss = 1 - dice.mean()

        return loss

    def extract_boundary_with_weights(self, mask):
        """
        提取带权重的边界区域
        """
        B, C, H, W = mask.shape

        # 高斯模糊模拟边界不确定性
        blurred = F.avg_pool2d(mask, kernel_size=5, stride=1, padding=2)

        # 计算梯度幅度
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                device=mask.device).view(1, 1, 3, 3).float()
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                device=mask.device).view(1, 1, 3, 3).float()

        grad_x = F.conv2d(blurred, kernel_x, padding=1)
        grad_y = F.conv2d(blurred, kernel_y, padding=1)
        gradient = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

        # 归一化
        gradient_norm = gradient / (gradient.max() + 1e-6)

        # 非线性增强：让边界区域更突出
        boundary_weights = 2 * torch.sigmoid(5 * gradient_norm - 2.5) - 0.5
        boundary_weights = torch.clamp(boundary_weights, 0, 1)

        return boundary_weights


class ConnectivityLoss(nn.Module):
    """
    连接性损失：强制滑坡区域保持空间连续性
    """

    def __init__(self, temperature=0.1, connectivity_weight=2.0):
        super().__init__()
        self.temperature = temperature
        self.connectivity_weight = connectivity_weight

        # 4邻域卷积核
        self.register_buffer('neighbor_kernel',
                             torch.tensor([[[[0, 1, 0],
                                             [1, 0, 1],
                                             [0, 1, 0]]]], dtype=torch.float32))

    def forward(self, pred, target):
        # 二值化
        pred_binary = (torch.sigmoid(pred) > 0.5).float()
        target_binary = (target > 0.5).float()

        B, C, H, W = pred.shape

        # 计算连通组件数量
        pred_components = self.count_connected_components(pred_binary)
        target_components = self.count_connected_components(target_binary)

        # 组件数量差异损失
        component_loss = F.l1_loss(pred_components, target_components)

        # 边界连续性损失
        boundary_continuity_loss = self.boundary_continuity_loss(pred_binary, target_binary)

        # 总连接性损失
        total_loss = (component_loss * 0.6 + boundary_continuity_loss * 0.4)

        return total_loss * self.connectivity_weight

    def count_connected_components(self, binary_mask):
        """快速估算连通组件数量"""
        B = binary_mask.shape[0]
        components = []

        # 简化的组件计数（不需要scipy）
        for b in range(B):
            mask = binary_mask[b, 0]
            # 使用形态学操作近似
            kernel = torch.ones(1, 1, 3, 3, device=binary_mask.device)
            # 膨胀后计算连通区域
            dilated = F.conv2d(mask.unsqueeze(0).unsqueeze(0), kernel, padding=1) > 0
            # 简单估算：通过局部极大值数量
            local_max = F.max_pool2d(dilated.float(), 3, stride=1, padding=1)
            num_components = (local_max == dilated.float()).sum().item()
            components.append(num_components)

        return torch.tensor(components, dtype=torch.float32, device=binary_mask.device)

    def boundary_continuity_loss(self, pred, target):
        """边界连续性损失"""
        # 提取边界
        kernel = torch.ones(1, 1, 3, 3, device=pred.device)
        pred_dilated = F.conv2d(pred, kernel, padding=1) > 0
        pred_eroded = F.conv2d(pred, kernel, padding=1) < kernel.sum()
        pred_boundary = (pred_dilated.float() - pred_eroded.float()).abs()

        # 检查边界连续性
        neighbor_conv = F.conv2d(pred_boundary, self.neighbor_kernel, padding=1)
        has_neighbor = (neighbor_conv > 0).float()

        # 连续性得分
        pred_score = (has_neighbor * pred_boundary).sum() / (pred_boundary.sum() + 1e-6)

        # 对target做相同操作
        target_dilated = F.conv2d(target, kernel, padding=1) > 0
        target_eroded = F.conv2d(target, kernel, padding=1) < kernel.sum()
        target_boundary = (target_dilated.float() - target_eroded.float()).abs()

        neighbor_conv_target = F.conv2d(target_boundary, self.neighbor_kernel, padding=1)
        has_neighbor_target = (neighbor_conv_target > 0).float()
        target_score = (has_neighbor_target * target_boundary).sum() / (target_boundary.sum() + 1e-6)

        return F.l1_loss(pred_score, target_score)


class MultiScaleIoULoss(nn.Module):
    """
    多尺度IoU损失：同时优化不同大小的滑坡区域
    """

    def __init__(self, scales=[0.5, 1.0, 2.0], weights=None):
        super().__init__()
        self.scales = scales

        if weights is None:
            self.weights = torch.tensor([0.4, 0.3, 0.3])
        else:
            self.weights = torch.tensor(weights)

        self.weights = self.weights / self.weights.sum()

    def forward(self, pred, target):
        B, C, H, W = pred.shape

        total_loss = 0.0

        for i, scale in enumerate(self.scales):
            if scale != 1.0:
                new_H, new_W = int(H * scale), int(W * scale)
                pred_scaled = F.interpolate(pred, size=(new_H, new_W),
                                            mode='bilinear', align_corners=True)
                target_scaled = F.interpolate(target, size=(new_H, new_W),
                                              mode='nearest')
            else:
                pred_scaled = pred
                target_scaled = target

            scale_loss = self.single_scale_iou_loss(pred_scaled, target_scaled)
            total_loss += scale_loss * self.weights[i]

        return total_loss

    def single_scale_iou_loss(self, pred, target):
        """单尺度IoU损失"""
        pred_prob = torch.sigmoid(pred)

        intersection = (pred_prob * target).sum(dim=(2, 3))
        union = pred_prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection

        iou = (intersection + 1e-6) / (union + 1e-6)
        iou_loss = 1 - iou.mean()

        return iou_loss


class EnhancedFocalLoss(nn.Module):
    """
    增强版Focal Loss：专门针对滑坡样本不平衡和困难样本
    """

    def __init__(self, alpha=0.75, gamma=3.0, landslide_weight=2.0,
                 hard_sample_threshold=0.3):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.landslide_weight = landslide_weight
        self.hard_sample_threshold = hard_sample_threshold
        self.hard_sample_ratio = 0.0

    def forward(self, pred, target):
        if target.max() > 1:
            target = (target > 0.5).float()

        pred_prob = torch.sigmoid(pred)

        # 基础交叉熵
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        # Focal调制因子
        p_t = target * pred_prob + (1 - target) * (1 - pred_prob)
        focal_weight = (1 - p_t) ** self.gamma

        # 类别平衡权重
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)

        # 滑坡区域额外权重
        with torch.no_grad():
            kernel = torch.ones(1, 1, 3, 3, device=target.device)
            dilated = F.conv2d(target, kernel, padding=1) > 0
            eroded = F.conv2d(target, kernel, padding=1) < kernel.sum()
            boundary_mask = (dilated.float() - eroded.float()).abs()
            landslide_weight_map = 1.0 + (self.landslide_weight - 1.0) * (target + boundary_mask * 0.5)

        # 困难样本识别
        hard_sample_mask = self.identify_hard_samples(pred_prob, target)
        hard_sample_weight = 1.0 + 2.0 * hard_sample_mask

        # 组合权重
        total_weight = focal_weight * alpha_weight * landslide_weight_map * hard_sample_weight

        # 加权损失
        weighted_loss = total_weight * bce_loss

        # 记录困难样本比例
        self.hard_sample_ratio = hard_sample_mask.float().mean().item()

        return weighted_loss.mean()

    def identify_hard_samples(self, pred_prob, target):
        """识别困难样本"""
        pred_error = torch.abs(pred_prob - target)
        hard_low = pred_prob > self.hard_sample_threshold
        hard_high = pred_prob < (1 - self.hard_sample_threshold)
        hard_sample_mask = (hard_low & hard_high).float()

        with torch.no_grad():
            kernel = torch.ones(1, 1, 3, 3, device=target.device)
            target_dilated = F.conv2d(target, kernel, padding=1) > 0
            target_eroded = F.conv2d(target, kernel, padding=1) < kernel.sum()
            boundary_mask = (target_dilated.float() - target_eroded.float()).abs()

        hard_sample_mask = torch.max(hard_sample_mask, boundary_mask)

        return hard_sample_mask


class LandslideOptimizedLoss(nn.Module):
    """完整版的滑坡专用损失函数组合"""

    def __init__(self,
                 boundary_weight=0.4,
                 connect_weight=0.2,
                 multiscale_weight=0.2,
                 focal_weight=0.2,
                 adaptive_weights=True):
        super().__init__()

        # 各组件损失
        self.boundary_dice = BoundarySensitiveDiceLoss(beta=3.0)  # 修复：添加beta参数
        self.shape_loss = ConnectivityLoss(connectivity_weight=2.0)
        self.multiscale_iou = MultiScaleIoULoss(scales=[0.5, 1.0, 2.0])
        self.hard_example_focal = EnhancedFocalLoss(alpha=0.75, gamma=3.0)

        # 初始权重
        self.boundary_weight = boundary_weight
        self.connect_weight = connect_weight
        self.multiscale_weight = multiscale_weight
        self.focal_weight = focal_weight

        # 是否使用自适应权重
        self.adaptive_weights = adaptive_weights

        if adaptive_weights:
            self.learnable_weights = nn.Parameter(torch.ones(4) / 4)

    def forward(self, pred, target):
        # 计算各项损失
        l_boundary = self.boundary_dice(pred, target)
        l_shape = self.shape_loss(pred, target)
        l_multiscale = self.multiscale_iou(pred, target)
        l_focal = self.hard_example_focal(pred, target)

        if self.adaptive_weights:
            weights = F.softmax(self.learnable_weights, dim=0)
            total_loss = (weights[0] * l_boundary +
                          weights[1] * l_shape +
                          weights[2] * l_multiscale +
                          weights[3] * l_focal)

            self.current_weights = weights.detach().cpu().numpy()
        else:
            total_loss = (self.boundary_weight * l_boundary +
                          self.connect_weight * l_shape +
                          self.multiscale_weight * l_multiscale +
                          self.focal_weight * l_focal)

        return total_loss

    def get_loss_breakdown(self, pred, target):
        """获取各项损失的详细数值"""
        with torch.no_grad():
            l_boundary = self.boundary_dice(pred, target).item()
            l_shape = self.shape_loss(pred, target).item()
            l_multiscale = self.multiscale_iou(pred, target).item()
            l_focal = self.hard_example_focal(pred, target).item()

            hard_ratio = self.hard_example_focal.hard_sample_ratio

        return {
            'boundary_loss': l_boundary,
            'connectivity_loss': l_shape,
            'multiscale_iou_loss': l_multiscale,
            'focal_loss': l_focal,
            'hard_sample_ratio': hard_ratio,
            'current_weights': getattr(self, 'current_weights', None)
        }

def get_simple_training_config():
    """获取简单训练配置"""

    # 1. 创建简单模型
    model = FixedU2NET(in_ch=4, n_classes=1).to('cuda')

    # 2. 使用滑坡专用损失函数（直接替换这里！）
    # ============ 替换开始 ============
    # 原来的：
    # def combined_loss(pred, target):
    #     bce = nn.BCEWithLogitsLoss()(pred, target)
    #     ...
    #     return bce + dice_loss

    # 替换为：
    criterion = LandslideOptimizedLoss(
        boundary_weight=0.4,  # 边界损失权重（重点优化边界）
        connect_weight=0.2,  # 连接性损失权重
        multiscale_weight=0.2,  # 多尺度损失权重
        focal_weight=0.2,  # 困难样本损失权重
        adaptive_weights=True  # 让网络自己学习最佳权重组合
    ).to('cuda')  # 重要：必须放到GPU上

    # 包装成函数形式，保持接口一致
    def combined_loss(pred, target):
        """滑坡专用损失函数"""
        return criterion(pred, target)

    # ============ 替换结束 ============

    # 3. 优化器（建议调整为更激进的学习率）
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,  # 可以尝试增加到2e-4
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
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2)

    # model, criterion, optimizer, scheduler, gradient_config = get_simple_optimized_config(len(train_loader))

    print(f"模型架构: {model.__class__.__name__}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    device_ids = list(range(torch.cuda.device_count()))
    print(f"可用的GPU: {device_ids}")


    # 训练模型
    train_model, history = train_model_multigpu_optimized(
        model=model,
        train_loader=train_loader,  # 你的训练数据加载器
        val_loader=val_loader,  # 你的验证数据加载器
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=100,  # 可以增加epoch
        device_ids=device_ids
    )

    # 保存最终模型
    if device_ids and len(device_ids) > 1:
        # 多GPU训练时，保存module
        torch.save(train_model.module.state_dict(), '/kaggle/working/final_U2NET_model.pth')
    else:
        torch.save(train_model.state_dict(), '/kaggle/working/final_U2NET_model.pth')
    print("最终模型已保存为 'final_U2NET_model.pth'")
    print("训练完成!")
    # 1. 加载训练好的模型

    model.load_state_dict(torch.load('/kaggle/working/final_U2NET_model.pth'))
    model.eval()

    # 2. 运行评估
    results = predict_and_evaluate(
        model=model,
        test_loader=test_loader,  # 你的测试数据加载器
        device='cuda',
        save_dir='/kaggle/working/',
        multigpu=True
    )


# 运行调试
if __name__ == "__main__":
    main()

