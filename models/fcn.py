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
    5. 早停机制（防止梯度爆炸导致的持续训练）
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

    # 添加早停机制相关变量
    best_iou = 0.0
    patience = 10  # 容忍多少个epoch没有改善
    no_improve_epochs = 0
    early_stop = False

    history = {
        'train_loss': [], 'val_loss': [], 'val_iou': [],
        'val_precision': [], 'val_recall': [], 'learning_rate': []
    }

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 40)

        # 检查是否需要早停
        if early_stop:
            print(f"⚠️  早停机制触发，停止训练")
            break

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
            no_improve_epochs = 0  # 重置无改善计数
            model_to_save = model.module if num_gpus > 1 else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_iou': best_iou,
                'history': history,
            }, 'best_FCN_checkpoint.pth')
            print(f'✓ 保存最佳模型检查点，IoU: {best_iou:.4f}')
        else:
            no_improve_epochs += 1
            print(f'⚠️  连续 {no_improve_epochs} 个epoch没有改善 (最佳IoU: {best_iou:.4f})')

            # 检查是否应该早停
            if no_improve_epochs >= patience:
                early_stop = True
                print(f'🚨 早停：连续 {patience} 个epoch没有改善')

        # 检查训练损失是否异常（梯度爆炸的迹象）
        if avg_train_loss > 1e5:  # 训练损失异常大
            print(f'🚨 训练损失异常 ({avg_train_loss:.2f})，可能是梯度爆炸，停止训练')
            early_stop = True
        elif avg_train_loss != avg_train_loss:  # 检查NaN
            print(f'🚨 训练损失为NaN，可能是梯度爆炸，停止训练')
            early_stop = True

    # 保存最终模型（无论是否早停）
    model_to_save = model.module if num_gpus > 1 else model
    final_checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_iou': best_iou,
        'history': history,
        'early_stopped': early_stop,
        'final_epoch': epoch
    }

    torch.save(final_checkpoint, 'final_FCN_model.pth')
    print(f'最终模型已保存，最佳IoU: {best_iou:.4f}')
    if early_stop:
        print(f'训练提前停止于第 {epoch + 1} 个epoch')

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


class FCN(nn.Module):
    """全卷积网络FCN-8s版本 - 早期融合适配"""

    def __init__(self, n_channels=4, n_classes=1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # 编码器部分 (基于VGG16结构，但简化以适应遥感数据)
        # Block 1
        self.conv1_1 = nn.Conv2d(n_channels, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # Block 2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # Block 3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # Block 4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # Block 5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # FCN-8s特定层
        # 分类器 (1x1卷积替换全连接层)
        self.fc6 = nn.Conv2d(512, 4096, 7, padding=3)
        self.drop6 = nn.Dropout2d()

        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_classes, 1)  # 1/32预测

        # 跳层连接分数
        self.score_pool4 = nn.Conv2d(512, n_classes, 1)  # 1/16特征
        self.score_pool3 = nn.Conv2d(256, n_classes, 1)  # 1/8特征

        # 上采样层
        self.upscore2 = nn.ConvTranspose2d(
            n_classes, n_classes, 4, stride=2, bias=False)  # 1/16 -> 1/8
        self.upscore4 = nn.ConvTranspose2d(
            n_classes, n_classes, 4, stride=2, bias=False)  # 1/8 -> 1/4
        self.upscore8 = nn.ConvTranspose2d(
            n_classes, n_classes, 16, stride=8, bias=False)  # 1/4 -> 原尺寸

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
            elif isinstance(m, nn.ConvTranspose2d):
                # 使用双线性插值初始化转置卷积
                nn.init.constant_(m.weight, 0)
                if m.kernel_size[0] == 16:
                    # 8x上采样
                    self._make_bilinear_weights_16x(m)
                elif m.kernel_size[0] == 4:
                    # 2x上采样
                    self._make_bilinear_weights_4x(m)

    def _make_bilinear_weights_4x(self, layer):
        """创建4x4双线性插值权重"""
        factor = 2
        c = layer.weight.size(0)

        weight = torch.zeros(c, c, 4, 4)
        for i in range(c):
            weight[i, i] = self._bilinear_kernel(4, factor)

        layer.weight.data = weight

    def _make_bilinear_weights_16x(self, layer):
        """创建16x16双线性插值权重"""
        factor = 8
        c = layer.weight.size(0)

        weight = torch.zeros(c, c, 16, 16)
        for i in range(c):
            weight[i, i] = self._bilinear_kernel(16, factor)

        layer.weight.data = weight

    def _bilinear_kernel(self, kernel_size, factor):
        """生成双线性插值核"""
        center = (kernel_size - 1) / 2.0

        og = torch.arange(kernel_size)
        og = og.unsqueeze(0).repeat(kernel_size, 1)

        u = (og - center) / factor
        v = (og.t() - center) / factor

        kernel = (1 - torch.abs(u)) * (1 - torch.abs(v))
        kernel = kernel / kernel.sum()

        return kernel

    def forward(self, optical, dem):
        """
        前向传播

        参数:
            optical: 光学影像 [B, C_optical, H, W]
            dem: 高程数据 [B, C_dem, H, W]

        返回:
            分割结果 [B, n_classes, H, W]
        """
        # 早期融合: 在通道维度拼接
        x = torch.cat([optical, dem], dim=1)

        # 编码路径
        # Block 1
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = self.pool1(h)  # 1/2

        # Block 2
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = self.pool2(h)  # 1/4

        # Block 3
        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        pool3 = h  # 保存pool3特征用于跳层连接
        h = self.pool3(h)  # 1/8

        # Block 4
        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        pool4 = h  # 保存pool4特征用于跳层连接
        h = self.pool4(h)  # 1/16

        # Block 5
        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = self.pool5(h)  # 1/32

        # FCN分类器
        h = F.relu(self.fc6(h))
        h = self.drop6(h)

        h = F.relu(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)  # 1/32预测

        # FCN-8s跳层连接融合
        # 第一次上采样 (1/32 -> 1/16)
        h = self.upscore2(h)
        upscore2 = h  # 1/16尺寸

        # 添加pool4的预测
        score_pool4 = self.score_pool4(pool4)
        h = score_pool4[:, :, 5:5 + upscore2.size(2), 5:5 + upscore2.size(3)]
        h = h + upscore2

        # 第二次上采样 (1/16 -> 1/8)
        upscore4 = self.upscore4(h)  # 1/8尺寸

        # 添加pool3的预测
        score_pool3 = self.score_pool3(pool3)
        h = score_pool3[:, :, 9:9 + upscore4.size(2), 9:9 + upscore4.size(3)]
        h = h + upscore4

        # 最终上采样到原尺寸 (1/8 -> 原尺寸)
        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size(2), 31:31 + x.size(3)].contiguous()

        return h


def get_simple_training_config():
    """获取简单训练配置"""

    # 1. 创建简单模型
    model = FCN(n_channels=4, n_classes=1).to('cuda')

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
        torch.save(train_model.module.state_dict(), '/kaggle/working/final_FCN_model.pth')
    else:
        torch.save(train_model.state_dict(), '/kaggle/working/final_FCN_model.pth')
    print("最终模型已保存为 'final_FCN_model.pth'")
    print("训练完成!")
    # 1. 加载训练好的模型

    model.load_state_dict(torch.load('/kaggle/working/final_FCN_model.pth'))
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

