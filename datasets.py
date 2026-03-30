import json
import os
import glob
from os.path import join
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
# 引入数据增强库
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Data3Dataset(Dataset):
    def __init__(self, data_root, target_size=(256, 256), num_channels=130, transform=None):
        """
        Args:
            data_root: 数据根目录 (e.g. /root/new_dataset)
            target_size: (H, W)
            num_channels: 序列长度，默认130（忽略第一张静态图后）
            transform: albumentations 的增强管道
        """
        self.data_root = data_root
        self.target_size = target_size
        self.num_channels = num_channels
        self.transform = transform
        self.samples = []
        
        def _collect_dynamic_images(search_dir: str):
            """
            只收集编号 >= 1001 的动态序列帧图片（如 1001.jpg ~ 1130.jpg）。
            忽略第一张静态图（1.jpg），因为它不是动态帧。
            """
            exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
            candidates = []
            for fn in os.listdir(search_dir):
                fp = join(search_dir, fn)
                if not os.path.isfile(fp):
                    continue
                lower = fn.lower()
                if not lower.endswith(exts):
                    continue
                stem = os.path.splitext(fn)[0]
                # 只保留编号 >= 1001 的动态帧，忽略 1.jpg 等静态图
                if stem.isdigit() and int(stem) >= 1001:
                    candidates.append((int(stem), fp))
            candidates.sort(key=lambda x: x[0])
            return [fp for _, fp in candidates]

        # 扫描 new_dataset 下所有病例文件夹
        all_folders = sorted([
            join(data_root, d) for d in os.listdir(data_root)
            if os.path.isdir(join(data_root, d))
        ])

        for folder_path in all_folders:
            # 图片在 folder/images/ 子目录
            images_dir = join(folder_path, "images")
            search_dir = images_dir if os.path.isdir(images_dir) else folder_path

            image_files = _collect_dynamic_images(search_dir)
            if len(image_files) != self.num_channels:
                print(f"跳过 {os.path.basename(folder_path)}: 找到 {len(image_files)} 张动态帧，期望 {self.num_channels} 张")
                continue

            # 掩码 json 在 folder/labels/ 子目录
            labels_dir = join(folder_path, "labels")
            if os.path.isdir(labels_dir):
                json_files = glob.glob(join(labels_dir, "*.json"))
            else:
                json_files = glob.glob(join(folder_path, "*.json"))
            target_json = json_files[0] if len(json_files) > 0 else None

            self.samples.append((image_files, target_json))

        print(f"成功加载 {len(self.samples)} 个样本")

    def _json_to_mask(self, json_path, original_h, original_w, target_h, target_w):
        # 从 JSON 静态标注生成掩码
        mask = Image.new('L', (original_w, original_h), 0)
        if json_path is not None and os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                draw = ImageDraw.Draw(mask)
                for shape in data.get('shapes', []):
                    points = shape.get('points', [])
                    if len(points) < 3:
                        continue
                    polygon = tuple(map(tuple, points))
                    draw.polygon(polygon, outline=1, fill=1)
            except Exception:
                pass
        # 缩放到目标尺寸，使用最近邻插值保持边缘清晰
        if (original_w, original_h) != (target_w, target_h):
            mask = mask.resize((target_w, target_h), Image.NEAREST)
        return np.array(mask)

    def __getitem__(self, index):
        image_paths, json_path = self.samples[index]

        # --- A. 读取图片 ---
        img_stack = []
        with Image.open(image_paths[0]) as ref_img:
            original_w, original_h = ref_img.size

        h_target, w_target = (self.target_size if self.target_size else (original_h, original_w))

        for p in image_paths:
            with Image.open(p) as img_obj:
                if self.target_size:
                    img_obj = img_obj.resize((w_target, h_target), Image.BILINEAR)
                img_np = np.array(img_obj)
                if img_np.ndim == 3:
                    img_np = np.dot(img_np[..., :3], [0.299, 0.587, 0.114])
                img_stack.append(img_np)

        # --- B. 堆叠为 (H, W, C) ---
        images_np = np.stack(img_stack, axis=2)  # Shape: (H, W, 130)

        # --- C. 读取 Mask ---
        mask_np = self._json_to_mask(json_path, original_h, original_w, h_target, w_target)

        # --- D. 数据增强 ---
        if self.transform is not None:
            augmented = self.transform(image=images_np, mask=mask_np)
            images_np = augmented['image']
            mask_np = augmented['mask']

        # --- E. 转 Tensor 并归一化 ---
        if isinstance(images_np, np.ndarray):
            images_tensor = torch.from_numpy(images_np).float()
            images_tensor = images_tensor.permute(2, 0, 1)  # (H, W, 130) -> (130, H, W)
        else:
            images_tensor = images_np.float()

        mask_tensor = torch.from_numpy(mask_np).float() if isinstance(mask_np, np.ndarray) else mask_np.float()

        # 归一化
        max_val = images_tensor.max().item()
        if max_val > 255:
            images_tensor = images_tensor / 65535.0
        else:
            images_tensor = images_tensor / 255.0

        # Mask 增加通道维度 (1, H, W)
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        return images_tensor, mask_tensor

    def __len__(self):
        return len(self.samples)


# --- 测试代码 ---
if __name__ == '__main__':
    ROOT_DIR = '/root/new_dataset'

    print("=" * 50)
    print("测试 Data3Dataset (new_dataset, 130通道)")
    print("=" * 50)

    train_transform = A.Compose([
        A.Rotate(limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ElasticTransform(p=0.3, alpha=1, sigma=50),
    ])

    dataset = Data3Dataset(
        data_root=ROOT_DIR,
        target_size=(256, 256),
        num_channels=130,
        transform=train_transform
    )

    if len(dataset) > 0:
        img, mask = dataset[0]
        print(f"Image Shape: {img.shape}")   # 期望: (130, 256, 256)
        print(f"Mask Shape : {mask.shape}")  # 期望: (1, 256, 256)
        print(f"Image max  : {img.max():.4f}")
        print(f"Mask unique: {mask.unique()}")
        print("测试通过！")
    else:
        print("没有找到数据，无法测试。")

