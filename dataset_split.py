import os
import pandas as pd
import numpy as np
from pathlib import Path
import random
from sklearn.model_selection import train_test_split
import warnings
import shutil
from PIL import Image
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


def create_dataset_csv(root_path, output_csv='dataset.csv'):
    """
    创建数据集CSV文件，按7:2:1比例分割训练集、验证集、测试集
    同时保存绝对路径和相对路径
    """
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)

    # 使用Path对象，自动处理路径分隔符
    base_path = Path(root_path)

    print(f"正在扫描目录: {base_path}")
    print("=" * 60)

    # 检查目录结构
    print("目录结构检查:")

    # 尝试找到数据集目录
    possible_paths = [
        base_path / "Bijie-landslide-dataset",
        base_path / "landslide-dataset",
        base_path,
    ]

    dataset_path = None
    for path in possible_paths:
        if path.exists():
            print(f"找到目录: {path}")
            dataset_path = path
            break

    if dataset_path is None:
        print("错误: 未找到数据集目录")
        # 显示当前目录内容
        print(f"当前目录 {base_path} 的内容:")
        for item in base_path.iterdir():
            print(f"  {item.name}")
        return None

    print(f"\n使用数据集路径: {dataset_path}")

    # 检查目录内容
    print("\n目录内容:")
    for item in dataset_path.iterdir():
        if item.is_dir():
            print(f"  [目录] {item.name}")

    # 定义路径
    landslide_path = dataset_path / "landslide"
    non_landslide_path = dataset_path / "non-landslide"

    # 检查是否存在这些目录
    print(f"\n检查滑坡目录: {landslide_path.exists()}")
    print(f"检查非滑坡目录: {non_landslide_path.exists()}")

    # 收集滑坡数据
    landslide_images = []
    if landslide_path.exists():
        image_dir = landslide_path / "image"
        dem_dir = landslide_path / "dem"
        mask_dir = landslide_path / "mask"

        print(f"\n扫描滑坡目录:")
        print(f"  图像目录: {image_dir.exists()}")
        print(f"  DEM目录: {dem_dir.exists()}")
        print(f"  掩码目录: {mask_dir.exists()}")

        if image_dir.exists():
            # 查找所有图片文件
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
                for img_file in image_dir.glob(ext):
                    img_name = img_file.stem
                    dem_file = dem_dir / f"{img_name}{img_file.suffix}"
                    mask_file = mask_dir / f"{img_name}{img_file.suffix}" if mask_dir.exists() else None

                    # 检查文件是否存在
                    dem_exists = dem_file.exists() if dem_dir.exists() else False
                    mask_exists = mask_file.exists() if mask_file else False

                    if dem_exists and (not mask_dir.exists() or mask_exists):
                        record = {
                            'image_path': str(img_file.absolute()),  # 保存绝对路径
                            'dem_path': str(dem_file.absolute()) if dem_exists else '',
                            'mask_path': str(mask_file.absolute()) if mask_exists else '',
                            'label': 1,
                            'type': 'landslide',
                            'filename': img_file.name
                        }
                        landslide_images.append(record)

            print(f"  找到有效的滑坡图像: {len(landslide_images)} 个")

    # 收集非滑坡数据
    non_landslide_images = []
    if non_landslide_path.exists():
        image_dir = non_landslide_path / "image"
        dem_dir = non_landslide_path / "dem"

        print(f"\n扫描非滑坡目录:")
        print(f"  图像目录: {image_dir.exists()}")
        print(f"  DEM目录: {dem_dir.exists()}")

        if image_dir.exists():
            # 查找所有图片文件
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
                for img_file in image_dir.glob(ext):
                    img_name = img_file.stem
                    dem_file = dem_dir / f"{img_name}{img_file.suffix}" if dem_dir.exists() else None

                    # 非滑坡可能没有mask
                    dem_exists = dem_file.exists() if dem_file else False

                    record = {
                        'image_path': str(img_file.absolute()),  # 保存绝对路径
                        'dem_path': str(dem_file.absolute()) if dem_exists else '',
                        'mask_path': '',
                        'label': 0,
                        'type': 'non_landslide',
                        'filename': img_file.name
                    }
                    non_landslide_images.append(record)

            print(f"  找到非滑坡图像: {len(non_landslide_images)} 个")

    print(f"\n" + "=" * 60)
    print(f"数据统计:")
    print(f"  滑坡数据: {len(landslide_images)} 个")
    print(f"  非滑坡数据: {len(non_landslide_images)} 个")
    print(f"  总计: {len(landslide_images) + len(non_landslide_images)} 个")

    if len(landslide_images) == 0 and len(non_landslide_images) == 0:
        print("错误: 未找到任何数据!")
        return None

    # 转换为DataFrame
    landslide_df = pd.DataFrame(landslide_images)
    non_landslide_df = pd.DataFrame(non_landslide_images)

    # 如果没有滑坡数据，直接分割非滑坡数据
    if len(landslide_images) == 0:
        print("\n只有非滑坡数据，直接分割...")
        all_data = non_landslide_df.copy()
        train, temp = train_test_split(all_data, test_size=0.3, random_state=42)
        val, test = train_test_split(temp, test_size=1 / 3, random_state=42)

        train['split'] = 'train'
        val['split'] = 'val'
        test['split'] = 'test'

        all_data = pd.concat([train, val, test], ignore_index=True)

    # 如果没有非滑坡数据，直接分割滑坡数据
    elif len(non_landslide_images) == 0:
        print("\n只有滑坡数据，直接分割...")
        all_data = landslide_df.copy()
        train, temp = train_test_split(all_data, test_size=0.3, random_state=42)
        val, test = train_test_split(temp, test_size=1 / 3, random_state=42)

        train['split'] = 'train'
        val['split'] = 'val'
        test['split'] = 'test'

        all_data = pd.concat([train, val, test], ignore_index=True)

    # 如果有两类数据，分别处理以保持比例
    else:
        print("\n分别处理两类数据以保持比例...")

        # 滑坡数据分割
        landslide_train, landslide_temp = train_test_split(
            landslide_df, test_size=0.3, random_state=42
        )
        landslide_val, landslide_test = train_test_split(
            landslide_temp, test_size=1 / 3, random_state=42
        )

        landslide_train['split'] = 'train'
        landslide_val['split'] = 'val'
        landslide_test['split'] = 'test'

        # 非滑坡数据分割
        non_landslide_train, non_landslide_temp = train_test_split(
            non_landslide_df, test_size=0.3, random_state=42
        )
        non_landslide_val, non_landslide_test = train_test_split(
            non_landslide_temp, test_size=1 / 3, random_state=42
        )

        non_landslide_train['split'] = 'train'
        non_landslide_val['split'] = 'val'
        non_landslide_test['split'] = 'test'

        # 合并所有数据
        all_parts = [
            landslide_train, landslide_val, landslide_test,
            non_landslide_train, non_landslide_val, non_landslide_test
        ]

        all_data = pd.concat(all_parts, ignore_index=True)

    # 添加相对路径（相对于数据集根目录）
    def get_relative_path(abs_path, base_path):
        if not abs_path:
            return ''
        try:
            return str(Path(abs_path).relative_to(base_path))
        except ValueError:
            return abs_path

    all_data['image_relative'] = all_data['image_path'].apply(
        lambda x: get_relative_path(x, dataset_path)
    )
    all_data['dem_relative'] = all_data['dem_path'].apply(
        lambda x: get_relative_path(x, dataset_path)
    )
    all_data['mask_relative'] = all_data['mask_path'].apply(
        lambda x: get_relative_path(x, dataset_path)
    )

    # 重新排列列顺序
    column_order = [
        'split', 'label', 'type', 'filename',
        'image_path', 'dem_path', 'mask_path',
        'image_relative', 'dem_relative', 'mask_relative'
    ]

    all_data = all_data[[col for col in column_order if col in all_data.columns]]

    # 显示统计信息
    print("\n" + "=" * 60)
    print("数据集分割统计:")
    print("=" * 60)

    for split in ['train', 'val', 'test']:
        split_data = all_data[all_data['split'] == split]
        total = len(split_data)

        if total > 0:
            landslide_count = len(split_data[split_data['label'] == 1])
            non_landslide_count = len(split_data[split_data['label'] == 0])

            print(f"\n{split.upper()}集:")
            print(f"  总数: {total}")
            print(f"  滑坡数据: {landslide_count} ({landslide_count / total * 100:.1f}%)")
            print(f"  非滑坡数据: {non_landslide_count} ({non_landslide_count / total * 100:.1f}%)")

    # 保存为CSV
    all_data.to_csv(output_csv, index=False)
    print(f"\n数据集已保存到: {output_csv}")

    # 创建分割文件
    create_split_files(all_data, 'splits', dataset_path)

    return all_data, dataset_path


def create_split_files(df, output_dir='splits', base_path=None):
    """创建独立的训练集、验证集、测试集文件"""
    os.makedirs(output_dir, exist_ok=True)

    for split_name in ['train', 'val', 'test']:
        split_df = df[df['split'] == split_name]

        if len(split_df) > 0:
            # 保存为CSV
            split_df.to_csv(os.path.join(output_dir, f'{split_name}.csv'), index=False)

            # 保存为txt文件（每行一个图像路径）
            with open(os.path.join(output_dir, f'{split_name}_paths.txt'), 'w', encoding='utf-8') as f:
                for _, row in split_df.iterrows():
                    f.write(f"{row['image_path']}\n")

            # 保存为带标签的txt文件
            with open(os.path.join(output_dir, f'{split_name}_labels.txt'), 'w', encoding='utf-8') as f:
                for _, row in split_df.iterrows():
                    f.write(f"{row['image_path']} {row['label']}\n")

            print(f"{split_name}集文件已保存到: {output_dir}/{split_name}.*")

    print(f"\n所有分割文件已保存到: {output_dir}/ 目录")


def create_test_visualization(dataset_csv, output_dir='test_visualization'):
    """
    创建测试集可视化图像，将原始图像和mask拼接在一起

    Args:
        dataset_csv: 数据集CSV文件路径或DataFrame
        output_dir: 输出目录
    """
    print("\n" + "=" * 60)
    print("创建测试集可视化图像")
    print("=" * 60)

    # 读取数据
    if isinstance(dataset_csv, pd.DataFrame):
        df = dataset_csv
    else:
        df = pd.read_csv(dataset_csv)

    # 获取测试集数据（只处理有mask的滑坡数据）
    test_df = df[(df['split'] == 'test') & (df['mask_path'].notna()) & (df['mask_path'] != '')]

    if len(test_df) == 0:
        print("测试集中没有找到带mask的数据！")
        print("尝试查找所有测试集数据...")
        test_df = df[df['split'] == 'test']

    if len(test_df) == 0:
        print("测试集为空！")
        return

    print(f"找到 {len(test_df)} 个测试集样本")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    processed_count = 0
    error_count = 0

    for idx, row in test_df.iterrows():
        try:
            # 读取原始图像
            if not os.path.exists(row['image_path']):
                print(f"警告: 图像文件不存在 - {row['image_path']}")
                continue

            img = Image.open(row['image_path']).convert('RGB')

            # 读取mask（如果有）
            mask_img = None
            if pd.notna(row['mask_path']) and row['mask_path'] != '' and os.path.exists(row['mask_path']):
                mask_img = Image.open(row['mask_path'])

                # 如果mask是二值图像，转换为彩色以便可视化
                if mask_img.mode != 'RGB':
                    mask_img = mask_img.convert('RGB')

            # 读取DEM（如果有）
            dem_img = None
            if pd.notna(row['dem_path']) and row['dem_path'] != '' and os.path.exists(row['dem_path']):
                dem_img = Image.open(row['dem_path'])
                if dem_img.mode != 'RGB':
                    dem_img = dem_img.convert('RGB')

            # 确定输出图像的大小
            # 基本思路：将所有可用的图像水平拼接
            images_to_concat = [img]
            img_widths = [img.width]
            img_heights = [img.height]

            if dem_img:
                images_to_concat.append(dem_img)
                img_widths.append(dem_img.width)
                img_heights.append(dem_img.height)

            if mask_img:
                images_to_concat.append(mask_img)
                img_widths.append(mask_img.width)
                img_heights.append(mask_img.height)

            # 调整所有图像到相同高度
            target_height = max(img_heights)
            resized_images = []

            for image in images_to_concat:
                # 计算缩放比例
                ratio = target_height / image.height
                new_width = int(image.width * ratio)
                resized_img = image.resize((new_width, target_height), Image.LANCZOS)
                resized_images.append(resized_img)

            # 创建拼接后的图像
            total_width = sum(img.width for img in resized_images)
            combined_img = Image.new('RGB', (total_width, target_height))

            # 粘贴所有图像
            x_offset = 0
            labels = ['Original', 'DEM', 'Mask'][:len(resized_images)]

            for i, resized_img in enumerate(resized_images):
                combined_img.paste(resized_img, (x_offset, 0))
                x_offset += resized_img.width

            # 保存拼接后的图像
            filename = f"test_{row['label']}_{Path(row['image_path']).stem}"
            output_path = os.path.join(output_dir, f"{filename}_combined.png")
            combined_img.save(output_path)

            processed_count += 1

            # 每处理10个样本打印一次进度
            if processed_count % 10 == 0:
                print(f"已处理 {processed_count} 个样本...")

        except Exception as e:
            print(f"处理样本时出错 (行 {idx}): {e}")
            error_count += 1
            continue

    print("\n" + "=" * 60)
    print("可视化图像创建完成!")
    print(f"成功处理: {processed_count} 个样本")
    print(f"处理失败: {error_count} 个样本")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    # 创建HTML预览文件
    create_html_preview(output_dir)


def create_html_preview(image_dir):
    """创建HTML预览文件"""
    html_file = os.path.join(image_dir, 'preview.html')

    # 获取所有PNG文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    if not image_files:
        return

    with open(html_file, 'w', encoding='utf-8') as f:
        f.write('''<!DOCTYPE html>
<html>
<head>
    <title>测试集可视化预览</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(600px, 1fr));
            gap: 20px;
        }
        .image-item {
            background: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .image-item img {
            width: 100%;
            height: auto;
            border: 1px solid #ddd;
        }
        .image-info {
            margin-top: 5px;
            font-size: 14px;
            color: #666;
            text-align: center;
        }
        .legend {
            background: white;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .legend h3 {
            margin-top: 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>测试集可视化预览</h1>

        <div class="legend">
            <h3>图例说明：</h3>
            <p>从左到右：原始图像 - DEM数据 - Mask掩码（滑坡数据）</p>
            <p>非滑坡数据只有原始图像和DEM数据</p>
            <p>文件名格式: test_标签_原文件名_combined.png</p>
        </div>

        <div class="image-grid">
''')

        for img_file in sorted(image_files):
            # 从文件名解析信息
            name_parts = img_file.replace('_combined.png', '').split('_')
            label = name_parts[1] if len(name_parts) > 1 else '未知'
            label_text = '滑坡' if label == '1' else '非滑坡'

            f.write(f'''
            <div class="image-item">
                <img src="{img_file}" alt="{img_file}">
                <div class="image-info">
                    文件: {img_file}<br>
                    标签: {label_text} ({label})
                </div>
            </div>
''')

        f.write('''
        </div>
    </div>
</body>
</html>''')

    print(f"HTML预览文件已创建: {html_file}")
    print(f"用浏览器打开此文件查看所有可视化图像")


def main():
    """主函数"""
    print("=" * 70)
    print("滑坡数据集处理工具")
    print("功能:")
    print("  1. 按7:2:1比例分割数据集")
    print("  2. 保存绝对路径和相对路径")
    print("  3. 生成测试集可视化图像")
    print("=" * 70)

    # 获取当前工作目录
    current_dir = os.getcwd()
    print(f"当前工作目录: {current_dir}")

    # 显示当前目录内容
    print("\n当前目录内容:")
    for item in Path(current_dir).iterdir():
        if item.is_dir():
            print(f"  [目录] {item.name}")
        else:
            if item.suffix in ['.py', '.csv', '.txt']:
                print(f"  [文件] {item.name}")

    # 询问用户数据集路径
    print("\n" + "=" * 70)
    user_input = input("请输入数据集路径（直接回车使用当前目录）: ").strip()

    if user_input:
        root_path = user_input
    else:
        root_path = current_dir

    print(f"\n使用路径: {root_path}")

    # 创建数据集CSV
    print("\n" + "=" * 70)
    print("开始创建数据集...")
    result = create_dataset_csv(root_path, 'landslide_dataset.csv')

    if result is None:
        print("数据集创建失败！")
        return

    df, dataset_path = result

    # 创建测试集可视化图像
    print("\n" + "=" * 70)
    print("开始创建测试集可视化图像...")
    create_test_visualization(df, 'test_visualization')

    print("\n" + "=" * 70)
    print("处理完成!")
    print("=" * 70)

    # 显示生成的目录结构
    print("\n生成的目录结构:")
    for root, dirs, files in os.walk('.'):
        level = root.replace('.', '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            if file.endswith(('.csv', '.txt', '.html')) or 'visualization' in root:
                print(f"{subindent}{file}")


if __name__ == "__main__":
    # main()
    df, dataset_path = create_dataset_csv("F:\zx\datasets\landslide", "my_dataset.csv")

    # 创建测试集可视化
    create_test_visualization(df, "test_results")