import os, sys, json, glob
from os.path import join
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import cv2
from scipy.interpolate import splprep, splev
from sam2.build_sam import build_sam2

MEDSAM2_DIR     = "/root/medsam/MedSAM2"
DATA_ROOT       = "/root/new_dataset"
CKPT_SAM2       = "/root/medsam/sam2.1_hiera_tiny.pt"
CFG_SAM2        = "configs/sam2.1_hiera_t512.yaml"
TRAINED_WEIGHTS = "/root/medsam/checkpoints/best_model.pt"
OUTPUT_DIR      = "/root/medsam/test_results2"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE      = 512
NUM_FRAMES      = 130

os.makedirs(OUTPUT_DIR, exist_ok=True)
sys.path.insert(0, MEDSAM2_DIR)
os.chdir(MEDSAM2_DIR)


class TemporalAdapter(nn.Module):
    def __init__(self, num_frames=130, reduction=16, image_size=512):
        super().__init__()
        self.image_size = image_size
        mid = max(num_frames // reduction, 8)
        self.se_fc1 = nn.Linear(num_frames, mid, bias=False)
        self.se_fc2 = nn.Linear(mid, num_frames, bias=False)
        self.frame_proj = nn.Sequential(
            nn.Conv2d(1, 8, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Conv2d(8, 3, 1, bias=True), nn.Sigmoid())
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        T, C, H, W = x.shape
        gap = x.mean(dim=[1, 2, 3])
        w = torch.sigmoid(self.se_fc2(F.relu(self.se_fc1(gap))))
        x = x * w.view(T, 1, 1, 1)
        x = x.mean(dim=0, keepdim=True)
        if H != self.image_size or W != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size),
                              mode="bilinear", align_corners=False)
        return (self.frame_proj(x) - self.mean) / self.std


class FPNDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.Conv2d(128+64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.up3 = nn.Sequential(nn.Conv2d(64+32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, fpn):
        x = F.interpolate(self.up1(fpn[2]), scale_factor=2, mode="bilinear", align_corners=False)
        x = F.interpolate(self.up2(torch.cat([x, fpn[1]], 1)), scale_factor=2, mode="bilinear", align_corners=False)
        x = F.interpolate(self.up3(torch.cat([x, fpn[0]], 1)), scale_factor=4, mode="bilinear", align_corners=False)
        return self.out(x)


class TestKidneyDataset(Dataset):
    def __init__(self, data_root, target_size=(512, 512)):
        self.target_size = target_size
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
                continue
            lbl_dir = join(folder, "labels")
            jsons = glob.glob(join(lbl_dir if os.path.isdir(lbl_dir) else folder, "*.json"))
            json_path = jsons[0] if jsons else None
            self.samples.append((fps, json_path, name))
        print(f"loaded {len(self.samples)} test samples")

    def _json_mask(self, json_path, orig_h, orig_w):
        mask = Image.new("L", (orig_w, orig_h), 0)
        if json_path is not None:
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
        mx = frames_np.max()
        if mx > 255:
            frames_np /= 65535.0
        elif mx > 1.0:
            frames_np /= 255.0
        frames_t = torch.from_numpy(frames_np).unsqueeze(1)
        mask_np  = self._json_mask(json_path, orig_h, orig_w)
        mask_t   = torch.from_numpy(mask_np).unsqueeze(0)
        return frames_t, mask_t, name, orig_h, orig_w


def compute_dice(pred_mask, gt_mask, eps=1e-5):
    pred  = pred_mask.float().view(-1)
    gt    = gt_mask.float().view(-1)
    inter = (pred * gt).sum()
    return (2 * inter + eps) / (pred.sum() + gt.sum() + eps)


def smooth_contour_overlay(base_np, mask_np, color, n_interp=600, line_width=2):
    """
    对掩码每个连通区域用 B-spline 拟合光滑轮廓，绘制在原图上。
    base_np : (H,W,3) uint8
    mask_np : (H,W)   float 0/1
    color   : (R,G,B)
    """
    result = Image.fromarray(base_np.copy())
    draw   = ImageDraw.Draw(result)
    H, W   = mask_np.shape

    mask_u8 = (mask_np > 0.5).astype(np.uint8) * 255
    # 放大 4 倍后提取轮廓，像素更密，样条更平滑
    scale    = 4
    mask_big = cv2.resize(mask_u8, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask_big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 计算最大轮廓面积，过滤掉小于 5% 的噪声连通域
    areas = [cv2.contourArea(c) for c in contours]
    max_area = max(areas) if areas else 1
    min_area_thresh = max(max_area * 0.05, (scale * 3) ** 2)

    for cnt, area in zip(contours, areas):
        if area < min_area_thresh:
            continue
        pts = cnt.squeeze()
        if pts.ndim < 2 or len(pts) < 8:
            continue
        x_pts = pts[:, 0].astype(float)
        y_pts = pts[:, 1].astype(float)
        # 增大平滑因子，减少毛刺
        s_val = max(len(pts) * 12.0, 50.0)
        try:
            tck, _ = splprep([x_pts, y_pts], s=s_val, k=3, per=True)
            xi, yi = splev(np.linspace(0, 1, n_interp), tck)
        except Exception:
            xi, yi = x_pts, y_pts
        # 缩回原始坐标系
        xi = xi / scale
        yi = yi / scale
        poly = list(zip(xi.tolist(), yi.tolist()))
        if len(poly) >= 2:
            draw.line(poly + [poly[0]], fill=color, width=line_width)

    return np.array(result)


def make_comparison(fps, gt_np, pred_np, orig_h, orig_w, dice, name):
    mid = len(fps) // 2
    with Image.open(fps[mid]) as img:
        base = np.array(img.convert("RGB").resize((orig_w, orig_h), Image.BILINEAR))

    # 放大到至少 320px
    scale = max(1, 320 // max(orig_h, orig_w))
    dh, dw = orig_h * scale, orig_w * scale
    base   = np.array(Image.fromarray(base).resize((dw, dh), Image.NEAREST))
    gt_r   = np.array(Image.fromarray((gt_np * 255).astype(np.uint8)).resize((dw, dh), Image.NEAREST)) / 255.0
    pr_r   = np.array(Image.fromarray((pred_np * 255).astype(np.uint8)).resize((dw, dh), Image.NEAREST)) / 255.0

    col_gt   = smooth_contour_overlay(base, gt_r,  (0, 210, 90),  line_width=2)
    col_raw  = base.copy()
    col_pred = smooth_contour_overlay(base, pr_r, (220, 40, 40), line_width=2)

    pad, hdr  = 6, 32
    W_total   = dw * 3 + pad * 4
    H_total   = dh + hdr + pad * 2
    canvas    = Image.new("RGB", (W_total, H_total), (20, 20, 20))
    draw      = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    except Exception:
        font = ImageFont.load_default()

    labels = [
        ("GT  manual",             (0, 210, 90)),
        (f"{name}  frame {mid+1}", (210, 210, 210)),
        (f"Pred  Dice={dice:.3f}", (220, 80, 80)),
    ]
    panels = [col_gt, col_raw, col_pred]
    for i, (img_np, (lbl, lc)) in enumerate(zip(panels, labels)):
        x0 = pad + i * (dw + pad)
        canvas.paste(Image.fromarray(img_np), (x0, hdr))
        draw.text((x0 + 4, 7), lbl, fill=lc, font=font)

    return canvas


def main():
    print("1. Building SAM2 encoder...")
    sam2 = build_sam2(CFG_SAM2, CKPT_SAM2, device=DEVICE, mode="eval", apply_postprocessing=False)
    for p in sam2.parameters():
        p.requires_grad_(False)
    sam2.eval()

    print("2. Loading trained weights...")
    adapter = TemporalAdapter(NUM_FRAMES, image_size=IMAGE_SIZE).to(DEVICE)
    decoder = FPNDecoder().to(DEVICE)
    if not os.path.exists(TRAINED_WEIGHTS):
        print(f"Weight not found: {TRAINED_WEIGHTS}")
        return
    ckpt = torch.load(TRAINED_WEIGHTS, map_location=DEVICE, weights_only=True)
    adapter.load_state_dict(ckpt["adapter"])
    decoder.load_state_dict(ckpt["decoder"])
    adapter.eval()
    decoder.eval()
    print(f"Loaded epoch={ckpt.get('epoch', '?')}  val_dice={ckpt.get('val_dice', 0):.4f}")

    dataset = TestKidneyDataset(DATA_ROOT, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    fps_map = {s[2]: s[0] for s in dataset.samples}
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    total_dice, valid_n = 0.0, 0
    print("\n3. Inference + smooth spline contour...")
    with torch.no_grad():
        for frames, masks, names, orig_h_t, orig_w_t in tqdm(loader, desc="Testing"):
            name   = names[0]
            orig_h = orig_h_t.item()
            orig_w = orig_w_t.item()

            rgb    = adapter(frames.squeeze(0).to(DEVICE))
            feats  = sam2.forward_image(rgb)
            logits = decoder(feats["backbone_fpn"])
            logits_orig = F.interpolate(logits, size=(orig_h, orig_w),
                                        mode="bilinear", align_corners=False)
            pred_np = (torch.sigmoid(logits_orig) > 0.5).squeeze().cpu().numpy().astype(np.float32)
            gt_np   = masks.squeeze().cpu().numpy().astype(np.float32)

            dice = 0.0
            if gt_np.max() > 0:
                dice = compute_dice(torch.tensor(pred_np), torch.tensor(gt_np)).item()
                total_dice += dice
                valid_n += 1
            tqdm.write(f"{name}  Dice={dice:.4f}")

            cmp = make_comparison(fps_map[name], gt_np, pred_np, orig_h, orig_w, dice, name)
            cmp.save(join(OUTPUT_DIR, f"{name}_compare.png"))

    if valid_n > 0:
        print(f"\nAvg Dice ({valid_n} cases): {total_dice / valid_n:.4f}")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

