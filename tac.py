import os
import sys
import json
import glob
import torch
import numpy as np
import cv2  # 🚀 必须引入 cv2 处理形态学操作
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from sam2.build_sam import build_sam2

# ==========================================
# ⚙️ 全局配置 (请确认路径是否正确)
# ==========================================
MEDSAM2_DIR = "/root/medsam/MedSAM2"
DATA_ROOT   = "/root/new_dataset"
CKPT_SAM2   = "/root/medsam/sam2.1_hiera_tiny.pt"
CFG_SAM2    = "configs/sam2.1_hiera_t512.yaml"
TRAINED_WEIGHTS = "/root/medsam/checkpoints/best_model.pt"
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE  = 512
NUM_FRAMES  = 130

sys.path.insert(0, MEDSAM2_DIR)
os.chdir(MEDSAM2_DIR)

# ==========================================
# 🧠 1. 加载我们训练好的神级网络结构
# ==========================================
class TemporalAdapter(nn.Module):
    def __init__(self, image_size=512):
        super().__init__()
        self.image_size = image_size
        self.scorer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=4, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=4, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(32, 1)
        )
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        T, C, H, W = x.shape
        scores = self.scorer(x)
        w = F.softmax(scores / self.temperature, dim=0)
        x_fused = (x * w.view(T, 1, 1, 1)).sum(dim=0, keepdim=True)
        if H != self.image_size or W != self.image_size:
            x_fused = F.interpolate(x_fused, size=(self.image_size, self.image_size),
                                    mode="bilinear", align_corners=False)
        rgb = x_fused.repeat(1, 3, 1, 1)
        return (rgb - self.mean) / self.std, w

class FPNDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.GroupNorm(16, 128), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.Conv2d(128+64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(inplace=True))
        self.up3 = nn.Sequential(nn.Conv2d(64+32, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.ReLU(inplace=True))
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, fpn):
        x = F.interpolate(self.up1(fpn[2]), scale_factor=2, mode="bilinear", align_corners=False)
        x = F.interpolate(self.up2(torch.cat([x, fpn[1]], 1)), scale_factor=2, mode="bilinear", align_corners=False)
        x = F.interpolate(self.up3(torch.cat([x, fpn[0]], 1)), scale_factor=4, mode="bilinear", align_corners=False)
        return self.out(x)

# ==========================================
# 📂 2. 数据读取与 JSON 解析
# ==========================================
def load_kidney_masks(folder_path: str, target_h: int, target_w: int):
    labels_dir = os.path.join(folder_path, 'labels')
    json_files = glob.glob(os.path.join(labels_dir, '*.json'))
    if not json_files:
        return None, None

    with open(json_files[0], 'r', encoding='utf-8') as f:
        data = json.load(f)

    orig_h = data.get('imageHeight', target_h)
    orig_w = data.get('imageWidth',  target_w)
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h

    mask_l = Image.new('L', (target_w, target_h), 0)
    mask_r = Image.new('L', (target_w, target_h), 0)

    for shape in data.get('shapes', []):
        label  = shape['label'].lower().strip()
        points = shape['points']
        if len(points) < 3: continue
        scaled = [(p[0] * scale_x, p[1] * scale_y) for p in points]
        
        if label == 'l':
            ImageDraw.Draw(mask_l).polygon(scaled, outline=1, fill=1)
        elif label == 'r':
            ImageDraw.Draw(mask_r).polygon(scaled, outline=1, fill=1)

    return np.array(mask_l, dtype=np.float32), np.array(mask_r, dtype=np.float32)

def extract_tacs():
    print("🚀 正在加载 SAM2 与训练好的提取模型...")
    sam2 = build_sam2(CFG_SAM2, CKPT_SAM2, device=DEVICE, mode="eval", apply_postprocessing=False)
    for p in sam2.parameters(): p.requires_grad_(False)
    sam2.eval()

    adapter = TemporalAdapter(image_size=IMAGE_SIZE).to(DEVICE)
    decoder = FPNDecoder().to(DEVICE)
    
    checkpoint = torch.load(TRAINED_WEIGHTS, map_location=DEVICE, weights_only=True)
    adapter.load_state_dict(checkpoint["adapter"])
    decoder.load_state_dict(checkpoint["decoder"])
    adapter.eval()
    decoder.eval()

    import re
    def natural_key(s): return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', s)]
    
    folders = sorted(
        [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))],
        key=natural_key
    )

    all_left_tacs  = []
    all_right_tacs = []

    print("\n🔥 开始批量提取高精度 TAC 曲线...")
    with torch.no_grad():
        for folder_name in tqdm(folders, desc="Extracting TACs"):
            folder_path = os.path.join(DATA_ROOT, folder_name)
            img_dir = os.path.join(folder_path, "images")
            if not os.path.isdir(img_dir): img_dir = folder_path
            
            exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
            cands = [(int(os.path.splitext(fn)[0]), os.path.join(img_dir, fn))
                     for fn in os.listdir(img_dir)
                     if os.path.isfile(os.path.join(img_dir, fn)) and fn.lower().endswith(exts)
                     and os.path.splitext(fn)[0].isdigit()
                     and int(os.path.splitext(fn)[0]) >= 1001]
            fps = [fp for _, fp in sorted(cands)]
            
            if len(fps) != NUM_FRAMES:
                print(f" ⚠️跳过 {folder_name}: 帧数不足 {NUM_FRAMES}")
                continue

            frames = []
            for fp in fps:
                with Image.open(fp) as img:
                    arr = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR), dtype=np.float32)
                    if arr.ndim == 3: arr = arr[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
                    frames.append(arr)
            
            frames_np = np.stack(frames) # (130, 512, 512)
            mx = frames_np.max()
            if mx > 255: frames_np /= 65535.0
            elif mx > 1.0: frames_np /= 255.0
            
            frames_t = torch.from_numpy(frames_np).unsqueeze(1).to(DEVICE)
            
            # 1. SAM2 生成全局预测掩码
            rgb, _ = adapter(frames_t)
            feats = sam2.forward_image(rgb)
            logits = decoder(feats["backbone_fpn"])
            logits_512 = F.interpolate(logits, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
            
            # 🚀 狠狠修改：转换为 Numpy 进行膨胀处理
            pred_mask_np = (torch.sigmoid(logits_512.squeeze()) > 0.5).cpu().numpy().astype(np.uint8)

            # 🛠️ 膨胀操作：给 ROI “加点肉”，容错呼吸位移并平滑信号
            # 使用 3x3 卷积核进行一次迭代膨胀
            kernel = np.ones((3, 3), np.uint8) 
            pred_mask_np = cv2.dilate(pred_mask_np, kernel, iterations=1) 
            
            # 转回 Tensor
            pred_mask = torch.from_numpy(pred_mask_np).to(DEVICE).float()

            # 2. 读取 JSON 粗略划分
            mask_l_np, mask_r_np = load_kidney_masks(folder_path, IMAGE_SIZE, IMAGE_SIZE)
            if mask_l_np is None or mask_l_np.sum() == 0 or mask_r_np.sum() == 0:
                print(f" ⚠️跳过 {folder_name}: JSON 标注不全")
                continue

            mask_l_t = torch.from_numpy(mask_l_np).to(DEVICE)
            mask_r_t = torch.from_numpy(mask_r_np).to(DEVICE)

            # 3. 灵魂一步：精准且略微膨胀的掩码 ∩ 左右划分 = 鲁棒的最终 ROI
            roi_l = pred_mask * mask_l_t
            roi_r = pred_mask * mask_r_t

            area_l = roi_l.sum() + 1e-8
            area_r = roi_r.sum() + 1e-8

            # 4. 提取 TAC
            frames_raw = torch.from_numpy(frames_np).to(DEVICE)
            tac_l = (frames_raw * roi_l).sum(dim=(1, 2)) / area_l
            tac_r = (frames_raw * roi_r).sum(dim=(1, 2)) / area_r

            all_left_tacs.append(tac_l.cpu().numpy())
            all_right_tacs.append(tac_r.cpu().numpy())

    all_tacs = []
    for l, r in zip(all_left_tacs, all_right_tacs):
        all_tacs.append(l)
        all_tacs.append(r)

    all_tacs = np.array(all_tacs)
    save_path = '/root/medsam/extracted_tacs.npy'
    np.save(save_path, all_tacs)
    
    print("\n===========================================")
    print(f"✅ 提取完成！膨胀处理后的曲线已保存。")
    print(f"📏 数据维度: {all_tacs.shape}")
    print(f"💾 保存路径: {save_path}")
    print("===========================================")

if __name__ == '__main__':
    extract_tacs()
