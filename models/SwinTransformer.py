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
            }, 'best_SwinTransformer_checkpoint.pth')
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


import warnings
warnings.filterwarnings('ignore')
# 导入timm库的Swin Transformer
import timm


class PatchEmbed(nn.Module):
    """图像分块嵌入"""

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class WindowAttention(nn.Module):
    """窗口注意力机制（修改版）"""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 相对位置偏置表 - 动态计算大小
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        # 生成相对位置索引（一次性计算）
        coords = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords, coords, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 获取相对位置偏置
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

        # 确保形状匹配
        if attn.shape[2:] != relative_position_bias.shape[1:]:
            # 动态调整窗口大小（如果实际窗口与预设不同）
            actual_window_size = int(N ** 0.5)
            if actual_window_size != self.window_size:
                # 重新计算相对位置索引
                coords = torch.arange(actual_window_size)
                coords = torch.stack(torch.meshgrid(coords, coords, indexing='ij'))
                coords_flatten = torch.flatten(coords, 1)
                relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
                relative_coords = relative_coords.permute(1, 2, 0).contiguous()
                relative_coords[:, :, 0] += actual_window_size - 1
                relative_coords[:, :, 1] += actual_window_size - 1
                relative_coords[:, :, 0] *= 2 * actual_window_size - 1
                relative_position_index = relative_coords.sum(-1)

                # 重新计算偏置
                relative_position_bias = self.relative_position_bias_table[
                    relative_position_index.view(-1)].view(
                    actual_window_size * actual_window_size,
                    actual_window_size * actual_window_size, -1)
                relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

        attn = attn + relative_position_bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer块"""

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)

        # 如果需要shift窗口
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # 分区窗口
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)  # nW*B, window_size*window_size, C

        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # 反向shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class Mlp(nn.Module):
    """MLP模块"""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    将特征图划分为窗口（支持任意尺寸）
    Args:
        x: (B, H, W, C)
        window_size (int): 窗口大小
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape

    # 如果尺寸不能被窗口大小整除，进行填充
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))  # 在H和W维度上填充
        H, W = H + pad_h, W + pad_w

    # 计算窗口数量
    num_windows_h = H // window_size
    num_windows_w = W // window_size

    # 重新塑造为窗口
    x = x.view(B, num_windows_h, window_size, num_windows_w, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

    return windows, (H, W), (pad_h, pad_w)


def window_reverse(windows, window_size, H, W, pad_h, pad_w):
    """
    将窗口恢复为特征图（处理填充）
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): 窗口大小
        H (int): 原始高度
        W (int): 原始宽度
        pad_h (int): 高度方向填充
        pad_w (int): 宽度方向填充
    Returns:
        x: (B, H, W, C)
    """
    H_padded = H + pad_h
    W_padded = W + pad_w
    num_windows_h = H_padded // window_size
    num_windows_w = W_padded // window_size

    B = int(windows.shape[0] / (num_windows_h * num_windows_w))

    x = windows.view(B, num_windows_h, num_windows_w, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_padded, W_padded, -1)

    # 去除填充
    if pad_h > 0 or pad_w > 0:
        x = x[:, :H, :W, :].contiguous()

    return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth)"""

    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class PatchMerging(nn.Module):
    """下采样：2倍降采样"""

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H, W, C
        """
        B, H, W, C = x.shape

        # 将2x2邻域的特征拼接在一起
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


# 纯Swin Transformer分割模型（完全离线）
class PureSwinSegmentation(nn.Module):
    """纯Swin Transformer语义分割模型 - 完全离线版"""

    def __init__(self, n_channels=4, n_classes=1, swin_type='tiny'):
        """
        参数:
            n_channels: 输入通道数
            n_classes: 输出类别数
            swin_type: 模型类型 ('tiny', 'small', 'base')
        """
        super().__init__()

        # 配置参数
        configs = {
            'tiny': {
                'embed_dim': 96,
                'depths': [2, 2, 6, 2],
                'num_heads': [3, 6, 12, 24],
                'window_size': 7,
                'drop_path_rate': 0.2
            },
            'small': {
                'embed_dim': 96,
                'depths': [2, 2, 18, 2],
                'num_heads': [3, 6, 12, 24],
                'window_size': 7,
                'drop_path_rate': 0.3
            },
            'base': {
                'embed_dim': 128,
                'depths': [2, 2, 18, 2],
                'num_heads': [4, 8, 16, 32],
                'window_size': 7,
                'drop_path_rate': 0.5
            }
        }

        config = configs[swin_type]
        self.embed_dim = config['embed_dim']
        self.depths = config['depths']
        self.num_heads = config['num_heads']
        self.window_size = config['window_size']

        # 存储各阶段输出通道数（重要！）
        self.stage_channels = [self.embed_dim * (2 ** i) for i in range(len(self.depths))]
        print(f"各阶段通道数: {self.stage_channels}")

        # 1. 输入适配层
        self.input_adapter = nn.Sequential(
            nn.Conv2d(n_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        # 2. Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=224,
            patch_size=4,
            in_chans=3,
            embed_dim=self.embed_dim,
            norm_layer=nn.LayerNorm
        )

        # 3. Swin Transformer阶段
        self.num_layers = len(self.depths)
        dpr = [x.item() for x in torch.linspace(0, config['drop_path_rate'], sum(self.depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(self.embed_dim * 2 ** i_layer),
                depth=self.depths[i_layer],
                num_heads=self.num_heads[i_layer],
                window_size=self.window_size,
                drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None
            )
            self.layers.append(layer)

        # 4. 特征金字塔网络 - 使用正确的通道数
        self.fpn = FPNModule(
            in_channels=self.stage_channels,  # 传递正确的通道数
            out_channels=256
        )

        # 5. 分割头
        self.seg_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, n_classes, 1)
        )

        self.apply(self._init_weights)

        print(f"初始化离线Swin Transformer模型 ({swin_type}):")
        print(f"  输入通道: {n_channels}")
        print(f"  输出类别: {n_classes}")
        print(f"  嵌入维度: {self.embed_dim}")
        print(f"  网络深度: {self.depths}")
        print(f"  各阶段通道数: {self.stage_channels}")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, optical, dem):
        # 1. 融合输入
        x = torch.cat([optical, dem], dim=1)
        input_size = x.shape[2:]

        # 2. 输入适配
        x = self.input_adapter(x)

        # 3. Patch Embedding
        x = self.patch_embed(x)  # B, N, C

        # 4. Swin Transformer层
        features = []
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.view(B, H, W, C)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            # 将特征保存为[B, C, H, W]格式
            B, H, W, C = x.shape
            features.append(x.permute(0, 3, 1, 2).contiguous())

        # 5. FPN融合特征
        fused = self.fpn(features)

        # 6. 上采样到输入尺寸
        output = F.interpolate(fused, size=input_size,
                               mode='bilinear', align_corners=True)

        # 7. 分割头
        output = self.seg_head(output)

        return output


class BasicLayer(nn.Module):
    """Swin Transformer基本层"""

    def __init__(self, dim, depth, num_heads, window_size,
                 drop_path=0., downsample=None):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # 构建块
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path
            )
            for i in range(depth)
        ])

        # 下采样层
        self.downsample = downsample(dim=dim) if downsample is not None else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)
        return x


class FPNModule(nn.Module):
    """特征金字塔网络（修复版）"""

    def __init__(self, in_channels, out_channels=256):
        """
        参数:
            in_channels: 各层输入通道数列表，例如 [96, 192, 384, 768]
            out_channels: 输出通道数
        """
        super().__init__()

        self.in_channels = in_channels

        # 横向连接：将各层特征映射到统一维度
        self.lateral_convs = nn.ModuleList()
        for in_channel in in_channels:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

        # 融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        print(f"FPN初始化 - 输入通道: {in_channels}, 输出通道: {out_channels}")

    def forward(self, features):
        """
        前向传播

        参数:
            features: 特征列表 [feat1, feat2, feat3, feat4]
                    每个特征的形状: [B, C_i, H_i, W_i]
        """
        # 检查特征数量是否与通道数匹配
        assert len(features) == len(self.in_channels), \
            f"特征数量({len(features)})与通道数({len(self.in_channels)})不匹配"

        # 检查每个特征的通道数是否正确
        for i, (feat, expected_channels) in enumerate(zip(features, self.in_channels)):
            actual_channels = feat.shape[1]
            assert actual_channels == expected_channels, \
                f"第{i}层特征: 期望通道数{expected_channels}, 实际通道数{actual_channels}"

        # 处理最后一层特征（最高层）
        last_idx = len(features) - 1
        fused_feature = self.lateral_convs[last_idx](features[last_idx])

        # 自顶向下的特征融合
        for i in range(len(features) - 2, -1, -1):
            # 上采样到与当前层相同的分辨率
            target_size = features[i].shape[2:]
            fused_feature = F.interpolate(
                fused_feature,
                size=target_size,
                mode='bilinear',
                align_corners=True
            )

            # 横向连接
            lateral_feature = self.lateral_convs[i](features[i])

            # 特征融合（逐元素相加）
            fused_feature = fused_feature + lateral_feature

        # 最终的融合卷积
        fused_feature = self.fusion_conv(fused_feature)

        return fused_feature

class SwinTransformerBlock(nn.Module):
    """Swin Transformer块（修改版）"""

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)

        # 计算实际窗口大小（适应输入尺寸）
        actual_window_size = min(self.window_size, H, W)

        # 如果需要shift窗口
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # 分区窗口（带填充处理）
        windows, padded_size, padding = window_partition(shifted_x, actual_window_size)
        windows = windows.view(-1, actual_window_size * actual_window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(windows)

        # 合并窗口（考虑填充）
        attn_windows = attn_windows.view(-1, actual_window_size, actual_window_size, C)
        shifted_x = window_reverse(attn_windows, actual_window_size, H, W, *padding)

        # 反向shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

def get_simple_training_config():
    """获取简单训练配置"""

    # 1. 创建简单模型
    model = PureSwinSegmentation(
        n_channels=4,
        n_classes=1,
        swin_type='base'  # 或 'small', 'base'
    )

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
        torch.save(train_model.module.state_dict(), '/kaggle/working/final_SwinTransformer_model.pth')
    else:
        torch.save(train_model.state_dict(), '/kaggle/working/final_SwinTransformer_model.pth')
    print("最终模型已保存为 'final_SwinTransformer_model.pth'")
    print("训练完成!")
    # 1. 加载训练好的模型

    model.load_state_dict(torch.load('/kaggle/working/final_SwinTransformer_model.pth'))
    model.eval()

    # 2. 运行评估
    results = predict_and_evaluate(
        model=model,
        test_loader=test_loader,  # 你的测试数据加载器
        device='cuda',
        save_dir='predictions_results',
        multigpu=True
    )


# 运行调试
if __name__ == "__main__":
    main()

