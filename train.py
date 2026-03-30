import os, sys, json, glob
from os.path import join
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image, ImageDraw
from tqdm import tqdm
from sam2.build_sam import build_sam2
import albumentations as A
import matplotlib.pyplot as plt # 新增：用于画出 Attention 曲线

## 1. 全局控制台
MEDSAM2_DIR = "/root/medsam/MedSAM2"
DATA_ROOT   = "/root/new_dataset"
CKPT_SAM2   = "/root/medsam/sam2.1_hiera_tiny.pt"
CFG_SAM2    = "configs/sam2.1_hiera_t512.yaml"
OUT_DIR     = "/root/medsam/checkpoints"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE  = 512
NUM_EPOCHS  = 50   
LR          = 3e-4   
NUM_FRAMES  = 130
BATCH_SIZE  = 1
os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, MEDSAM2_DIR)
os.chdir(MEDSAM2_DIR)

## 2. 狠狠修改的 MIL 时空适配器 (完美解决注意力塌陷)
class TemporalAdapter(nn.Module):
    def __init__(self, image_size=512):
        super().__init__()
        self.image_size = image_size
        
        # 真正的 MIL 注意力打分器
        self.scorer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=4, bias=False), # 512->128
            nn.BatchNorm2d(16), 
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=4, bias=False), # 128->32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), # -> (130, 32, 1, 1)
            nn.Flatten(),            # -> (130, 32)
            nn.Dropout(0.3),         # 🛡️ 极强防过拟合
            nn.Linear(32, 1)         # -> (130, 1) 输出原始得分
        )
        
        # 可学习的温度系数，逼迫 Softmax 形成尖锐山峰
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.attn_weights = None # 用于外接可视化

    def forward(self, x):
        T, C, H, W = x.shape
        scores = self.scorer(x) # (130, 1)
        
        # 核心：Softmax 拉开差距
        w = F.softmax(scores / self.temperature, dim=0) 
        self.attn_weights = w.detach().cpu().squeeze().numpy()

        # 核心：加权求和，绝对不要求平均！
        x_fused = (x * w.view(T, 1, 1, 1)).sum(dim=0, keepdim=True) 

        if H != self.image_size or W != self.image_size:
            x_fused = F.interpolate(x_fused, size=(self.image_size, self.image_size),
                                    mode="bilinear", align_corners=False)
                                    
        # 暴力转 3 通道 RGB 送给 SAM2
        rgb = x_fused.repeat(1, 3, 1, 1)
        return (rgb - self.mean) / self.std

## 3. 防崩溃的 FPNDecoder (剔除 BatchNorm)
class FPNDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Batch=1 时必须用 GroupNorm，加上 Dropout 防死记硬背
        self.up1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), 
            nn.GroupNorm(16, 128), 
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(128+64, 64, 3, padding=1), 
            nn.GroupNorm(8, 64), 
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(64+32, 32, 3, padding=1), 
            nn.GroupNorm(4, 32), 
            nn.ReLU(inplace=True)
        )
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, fpn):
        x = F.interpolate(self.up1(fpn[2]), scale_factor=2, mode="bilinear", align_corners=False)
        x = F.interpolate(self.up2(torch.cat([x, fpn[1]], 1)), scale_factor=2, mode="bilinear", align_corners=False)
        x = F.interpolate(self.up3(torch.cat([x, fpn[0]], 1)), scale_factor=4, mode="bilinear", align_corners=False)
        return self.out(x)

## (数据集和 Loss 保持你原来的神仙逻辑，这里略写以省篇幅，你需要把原来的贴在这里)
# ---> 请在这里粘贴原有的 KidneyVideoDataset, BceDiceLoss, compute_dice 类 <---
class KidneyVideoDataset(Dataset):
    def __init__(self, data_root, target_size=(512, 512), transform=None):
        self.data_root = data_root
        self.target_size = target_size
        self.transform = transform
        self.samples = []

        def _collect(d):
            exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
            cands = [(int(os.path.splitext(fn)[0]), join(d, fn))
                     for fn in os.listdir(d)
                     if os.path.isfile(join(d, fn)) and fn.lower().endswith(exts)
                     and os.path.splitext(fn)[0].isdigit()
                     and int(os.path.splitext(fn)[0]) >= 1001]
            return [fp for _, fp in sorted(cands)]

        for name in sorted(os.listdir(data_root)):
            folder = join(data_root, name)
            if not os.path.isdir(folder):
                continue
            img_dir = join(folder, "images")
            fps = _collect(img_dir if os.path.isdir(img_dir) else folder)
            if len(fps) != NUM_FRAMES:
                print(f"skip {name}: {len(fps)} frames")
                continue
            lbl_dir = join(folder, "labels")
            jsons = glob.glob(join(lbl_dir if os.path.isdir(lbl_dir) else folder, "*.json"))
            if jsons:
                self.samples.append((fps, jsons[0], name))
        print(f"loaded {len(self.samples)} samples")

    def _json_mask(self, json_path, orig_h, orig_w):
        th, tw = self.target_size
        mask = Image.new("L", (orig_w, orig_h), 0)
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            draw = ImageDraw.Draw(mask)
            for shape in data.get("shapes", []):
                pts = shape.get("points", [])
                if len(pts) >= 3:
                    draw.polygon([tuple(p) for p in pts], outline=1, fill=1)
        except Exception:
            pass
        if (orig_w, orig_h) != (tw, th):
            mask = mask.resize((tw, th), Image.NEAREST)
        return np.array(mask, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fps, json_path, name = self.samples[idx]
        th, tw = self.target_size
        with Image.open(fps[0]) as ref:
            orig_w, orig_h = ref.size
        frames = []
        for fp in fps:
            with Image.open(fp) as img:
                arr = np.array(img.resize((tw, th), Image.BILINEAR), dtype=np.float32)
                if arr.ndim == 3:
                    arr = arr[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
                frames.append(arr)
        
        frames_np = np.stack(frames) 
        mask_np = self._json_mask(json_path, orig_h, orig_w) 

        if self.transform is not None:
            frames_np = frames_np.transpose(1, 2, 0) 
            augmented = self.transform(image=frames_np, mask=mask_np)
            frames_np = augmented['image']
            mask_np = augmented['mask']
            frames_np = frames_np.transpose(2, 0, 1) 

        mx = frames_np.max()
        if mx > 255:
            frames_np /= 65535.0
        elif mx > 1.0:
            frames_np /= 255.0
            
        frames_t = torch.from_numpy(frames_np).unsqueeze(1)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0)
        return frames_t, mask_t, name

class BceDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth
    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        p = torch.sigmoid(logits)
        pf = p.view(p.size(0), -1)
        tf = targets.view(targets.size(0), -1)
        inter = (pf * tf).sum(1)
        dice = 1 - (2 * inter + self.smooth) / (pf.sum(1) + tf.sum(1) + self.smooth)
        return 0.5 * bce + 0.5 * dice.mean()

def compute_dice(pred_logits, gt_mask, eps=1e-5):
    pred = (torch.sigmoid(pred_logits) > 0.5).float().view(-1)
    gt = gt_mask.float().view(-1)
    inter = (pred * gt).sum()
    return (2 * inter + eps) / (pred.sum() + gt.sum() + eps)
# ---> 结束粘贴 <---

def main():
    print("Building SAM2 model ...")
    sam2 = build_sam2(
        config_file=CFG_SAM2,
        ckpt_path=CKPT_SAM2,
        device=DEVICE,
        mode="eval",
        apply_postprocessing=False,
    )
    for p in sam2.parameters():
        p.requires_grad_(False)
    sam2.eval()

    # 初始化 (注意不要再传 num_frames 了)
    adapter = TemporalAdapter(image_size=IMAGE_SIZE).to(DEVICE)
    decoder = FPNDecoder().to(DEVICE)

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.6),
        A.ElasticTransform(alpha=50, sigma=5, alpha_affine=25, p=0.3)
    ])

    full_dataset_train = KidneyVideoDataset(DATA_ROOT, target_size=(IMAGE_SIZE, IMAGE_SIZE), transform=train_transform)
    full_dataset_val   = KidneyVideoDataset(DATA_ROOT, target_size=(IMAGE_SIZE, IMAGE_SIZE), transform=None)

    n_val = max(1, int(len(full_dataset_train) * 0.1))
    indices = torch.randperm(len(full_dataset_train), generator=torch.Generator().manual_seed(42)).tolist()
    
    train_ds = torch.utils.data.Subset(full_dataset_train, indices[:-n_val])
    val_ds   = torch.utils.data.Subset(full_dataset_val, indices[-n_val:])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 剔除了你代码里重复的 optimizer，保留带 1e-4 Weight Decay 的版本防过拟合
    optimizer = optim.AdamW(
        list(adapter.parameters()) + list(decoder.parameters()),
        lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    criterion = BceDiceLoss()

    best_dice = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        adapter.train()
        decoder.train()
        train_loss = 0.0
        
        # 记录训练进度
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [train]")
        for step, (frames, masks, names) in enumerate(pbar):
            frames = frames.squeeze(0).to(DEVICE)
            masks = masks.to(DEVICE).float()
            
            optimizer.zero_grad()
            rgb = adapter(frames)
            
            # ---> 🚨 核心打印区域：监控 Attention 权重 <---
            # 每个 epoch 只打印第一个病人的权重，防止刷屏
            if step == 0:
                weights = adapter.attn_weights
                max_idx = np.argmax(weights)
                max_val = np.max(weights)
                min_val = np.min(weights)
                pbar.write(f"\n🔥 [实时监控] {names[0]} | 最强注意力帧: {max_idx} (权重: {max_val:.4f}) | 最弱帧: (权重: {min_val:.4f})")
                
                # 如果你想每次自动画一张折线图保存下来，可以取消下面这两行的注释：
                # plt.plot(weights); plt.title(f"Epoch {epoch} Attention"); plt.savefig(join(OUT_DIR, f"attn_ep{epoch}.png")); plt.close()
            
            with torch.no_grad():
                feats = sam2.forward_image(rgb)
            fpn = feats["backbone_fpn"]
            logits = decoder(fpn)
            gt = F.interpolate(masks, size=logits.shape[-2:], mode="nearest")
            
            loss = criterion(logits, gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(adapter.parameters()) + list(decoder.parameters()), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # ================= 验证环节 =================
        adapter.eval()
        decoder.eval()
        val_dice = 0.0
        with torch.no_grad():
            for frames, masks, _ in val_loader:
                frames = frames.squeeze(0).to(DEVICE)
                masks = masks.to(DEVICE).float()
                rgb = adapter(frames)
                feats = sam2.forward_image(rgb)
                fpn = feats["backbone_fpn"]
                logits = decoder(fpn)
                gt = F.interpolate(masks, size=logits.shape[-2:], mode="nearest")
                val_dice += compute_dice(logits, gt).item()
        val_dice /= len(val_loader)

        print(f"Epoch {epoch:02d} | loss={train_loss:.4f} | val_dice={val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                "epoch": epoch,
                "adapter": adapter.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_dice": val_dice,
            }, join(OUT_DIR, "best_model.pt"))
            print(f"  -> saved best model (dice={best_dice:.4f})")

    print(f"Training done. Best val dice: {best_dice:.4f}")

if __name__ == "__main__":
    main()
