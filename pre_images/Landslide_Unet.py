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

                # 前向传播 - 现在输出是元组
                outputs_tuple = model(optical, dem)

                # 关键修复：验证阶段也需要取第一个输出
                if isinstance(outputs_tuple, tuple):
                    outputs = outputs_tuple[0]  # 只取final_output
                else:
                    outputs = outputs_tuple

                # 计算损失
                loss = criterion(outputs, mask)
                val_loss += loss.item()

                # 关键修复：现在outputs是张量，可以sigmoid了
                pred_probs = torch.sigmoid(outputs)
                preds = (pred_probs > 0.7).float()

                # 计算指标
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
            }, 'best_Landslide_Unet_checkpoint.pth')
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


class EdgeEnhancementModule(nn.Module):
    """边缘增强模块 - 提取并强化边界特征"""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # Sobel-like 可学习边缘检测
        self.edge_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.init_sobel_weights()

        # 边界特征处理
        self.edge_processing = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 残差连接
        self.res_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def init_sobel_weights(self):
        """初始化类似Sobel算子的权重"""
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [0., 0., 0.],
                                [1., 2., 1.]], dtype=torch.float32)

        # 创建深度可分离卷积的权重
        weight = torch.zeros(self.channels, 1, 3, 3)
        for i in range(self.channels):
            # 组合x和y方向梯度
            weight[i, 0, :, :] = (sobel_x + sobel_y) / 2.0

        self.edge_conv.weight = nn.Parameter(weight)
        self.edge_conv.weight.requires_grad = True  # 允许微调

    def forward(self, x):
        identity = x

        # 提取边缘特征
        edge_feat = self.edge_conv(x)
        edge_feat = torch.abs(edge_feat)  # 梯度幅度

        # 生成注意力权重
        edge_attention = self.edge_processing(edge_feat)

        # 增强边界区域
        enhanced = x * (1 + edge_attention)

        # 残差连接
        res = self.res_conv(enhanced)

        return F.relu(res + identity)


class MultiScaleFusion(nn.Module):
    """多尺度特征融合模块"""

    def __init__(self, channels_list):
        super().__init__()
        self.convs = nn.ModuleList()
        for channels in channels_list:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(channels, 64, kernel_size=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                )
            )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(64 * len(channels_list), 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=1)
        )

    def forward(self, features):
        # 统一分辨率到最小尺寸
        target_size = features[-1].shape[2:]
        resized_features = []

        for i, feat in enumerate(features):
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=True)
            feat = self.convs[i](feat)
            resized_features.append(feat)

        # 拼接并融合
        fused = torch.cat(resized_features, dim=1)
        return self.fusion_conv(fused)


class LandslideUNet(nn.Module):
    """边界感知的U-Net网络 - 专门针对滑坡边界优化"""

    def __init__(self, n_channels=4, n_classes=1):
        super().__init__()

        # 编码器 (下采样)
        self.inc = DoubleConv(n_channels, 64)
        self.edge1 = EdgeEnhancementModule(64)  # 第一层边缘增强

        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128)
        )
        self.edge2 = EdgeEnhancementModule(128)  # 第二层边缘增强

        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256)
        )
        self.edge3 = EdgeEnhancementModule(256)  # 第三层边缘增强

        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(256, 512)
        )

        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(512, 1024)
        )

        # 多尺度边界特征融合
        self.multi_scale_fusion = MultiScaleFusion([64, 128, 256, 512])

        # 解码器 (上采样) - 加入边界注意力
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.boundary_att1 = BoundaryAttentionModule(512)
        self.conv1 = DoubleConv(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.boundary_att2 = BoundaryAttentionModule(256)
        self.conv2 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.boundary_att3 = BoundaryAttentionModule(128)
        self.conv3 = DoubleConv(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        # 边界细化头
        self.boundary_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

        # 主分割头
        self.seg_head = nn.Conv2d(64, n_classes, kernel_size=1)

        # 融合卷积
        self.final_conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, n_classes, kernel_size=1)
        )

    def forward(self, optical, dem):
        # 早期融合: 在通道维度拼接
        x = torch.cat([optical, dem], dim=1)

        # 编码路径 with edge enhancement
        x1 = self.inc(x)
        x1_edge = self.edge1(x1)  # 增强边界

        x2 = self.down1(x1_edge)
        x2_edge = self.edge2(x2)

        x3 = self.down2(x2_edge)
        x3_edge = self.edge3(x3)

        x4 = self.down3(x3_edge)
        x5 = self.down4(x4)

        # 多尺度边界特征融合
        boundary_features = self.multi_scale_fusion([x1_edge, x2_edge, x3_edge, x4])

        # 解码路径 with boundary attention
        x = self.up1(x5)
        x = self.boundary_att1(x, boundary_features)  # 加入边界注意力
        x = torch.cat([x, x4], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = self.boundary_att2(x, boundary_features)
        x = torch.cat([x, x3_edge], dim=1)  # 使用边缘增强的特征
        x = self.conv2(x)

        x = self.up3(x)
        x = self.boundary_att3(x, boundary_features)
        x = torch.cat([x, x2_edge], dim=1)  # 使用边缘增强的特征
        x = self.conv3(x)

        x = self.up4(x)
        x = torch.cat([x, x1_edge], dim=1)  # 使用边缘增强的特征
        x = self.conv4(x)

        # 双头输出: 边界 + 分割
        boundary_map = self.boundary_head(x)
        seg_map = self.seg_head(x)

        # 融合边界信息到分割结果
        if boundary_map.shape != seg_map.shape:
            boundary_map = F.interpolate(boundary_map, size=seg_map.shape[2:], mode='bilinear', align_corners=True)

        fused = torch.cat([seg_map, boundary_map], dim=1)
        final_output = self.final_conv(fused)

        return final_output, boundary_map, seg_map


class BoundaryAttentionModule(nn.Module):
    """边界注意力模块 - 引导网络关注边界区域"""

    def __init__(self, channels):
        super().__init__()
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(64, channels, kernel_size=1),  # 64来自multi_scale_fusion
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, boundary_feat):
        # 处理边界特征
        boundary = self.boundary_conv(boundary_feat)
        if boundary.shape[2:] != x.shape[2:]:
            boundary = F.interpolate(boundary, size=x.shape[2:], mode='bilinear', align_corners=True)

        # 生成注意力图
        attention_map = self.attention(torch.cat([x, boundary], dim=1))

        # 应用注意力
        return x * attention_map


# 辅助损失函数
class BoundaryAwareLoss(nn.Module):
    """边界感知的混合损失函数 - 兼容元组输入"""

    def __init__(self, alpha=0.7, beta=0.3, gamma=0.5):
        super().__init__()
        self.alpha = alpha  # 分割损失权重
        self.beta = beta  # 边界损失权重
        self.gamma = gamma  # Dice损失权重

        self.bce_loss = nn.BCEWithLogitsLoss()

    def dice_coefficient(self, pred, target, smooth=1e-5):
        """计算Dice系数"""
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

    def dice_loss(self, pred, target):
        """Dice损失"""
        return 1 - self.dice_coefficient(pred, target)

    def forward(self, pred, target, boundary_pred=None, boundary_target=None):
        """
        支持多种输入格式：
        1. target是张量: 仅分割mask
        2. target是元组: (mask, boundary_mask)
        3. 单独提供boundary_pred和boundary_target
        """
        # 处理target输入 - 检查是否是元组
        if isinstance(target, tuple):
            # 如果target是元组，假设格式为(mask, boundary_mask)
            seg_target, bound_target = target
            # 检查pred是否是元组
            if isinstance(pred, tuple):
                seg_pred, bound_pred = pred
            else:
                seg_pred = pred
                bound_pred = None
        else:
            # target是单个张量
            seg_target = target
            bound_target = boundary_target
            seg_pred = pred
            bound_pred = boundary_pred

        # 主分割损失
        seg_bce = self.bce_loss(seg_pred, seg_target)
        seg_dice = self.dice_loss(seg_pred, seg_target)
        seg_loss = seg_bce + self.gamma * seg_dice

        if bound_pred is not None and bound_target is not None:
            # 边界损失
            bound_bce = self.bce_loss(bound_pred, bound_target)
            bound_dice = self.dice_loss(bound_pred, bound_target)
            bound_loss = bound_bce + self.gamma * bound_dice

            # 总损失
            total_loss = self.alpha * seg_loss + self.beta * bound_loss
            return total_loss, seg_loss, bound_loss

        # 如果没有边界监督，只返回分割损失
        return seg_loss


def get_simple_training_config():
    """获取简单训练配置 - 适配新模型"""

    # 1. 创建模型
    model = LandslideUNet(n_channels=4, n_classes=1)

    # 2. 修改损失函数，适配新模型的3输出格式
    def combined_loss(pred, target):
        """BCE + Dice损失 - 适配3输出元组"""
        # 关键：pred现在是一个元组 (final_output, boundary_map, seg_map)
        # 我们只需要第一个final_output作为分割结果
        if isinstance(pred, tuple):
            final_output = pred[0]  # 只取第一个分割结果
        else:
            final_output = pred

        # 计算损失（保持原有逻辑）
        bce = nn.BCEWithLogitsLoss()(final_output, target)

        # Dice损失
        probs = torch.sigmoid(final_output)
        smooth = 1e-6
        intersection = (probs * target).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice.mean()

        return bce + dice_loss

    # 3. 优化器（保持原样）
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    # 4. 学习率调度器（保持原样）
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

    # 设置
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据准备（使用新函数）
    data_dir = r"F:\zx\datasets\Bijie-landslide-dataset"
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


    model.load_state_dict(torch.load(r'F:\zx\模型结果及参数\final_Landslide_Unet_model.pth', map_location=torch.device('cpu')))

    model.eval()

    # 2. 运行评估
    results = predict_and_evaluate(
        model=model,
        test_loader=test_loader,  # 你的测试数据加载器
        device='cpu',
        save_dir=r'F:\zx\predictions_results\predictions_results_Landslide_Unet',
        multigpu=True
    )


# 运行调试
if __name__ == "__main__":
    main()

