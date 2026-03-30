"""extract_tac.py
用训练好的模型预测掩码，分别提取左肾/右肾的 TAC 曲线，
输出与 extracted_tacs_left_right.npy 格式一致的新文件。
"""
import os, sys, json, glob
from os.path import join
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
import pandas as pd
from tqdm import tqdm
from sam2.build_sam import build_sam2

MEDSAM2_DIR     = "/root/medsam/MedSAM2"
DATA_ROOT       = "/root/new_dataset"
CKPT_SAM2       = "/root/medsam/sam2.1_hiera_tiny.pt"
CFG_SAM2        = "configs/sam2.1_hiera_t512.yaml"
TRAINED_WEIGHTS = "/root/medsam/checkpoints/best_model.pt"
CSV_PATH        = "/root/medsam/clinical_labels_left_right.csv"
OUT_NPY         = "/root/medsam/extracted_tacs_medsam.npy"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE      = 512
NUM_FRAMES      = 130

sys.path.insert(0, MEDSAM2_DIR)
os.chdir(MEDSAM2_DIR)


# ── 模型定义（与 train.py 一致）──────────────────────────────────────────────
class TemporalAdapter(nn.Module):
    def __init__(self, image_size=512):
        super().__init__()
        self.image_size = image_size
        self.scorer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(32, 1)
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
        return (x_fused.repeat(1, 3, 1, 1) - self.mean) / self.std


class FPNDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.GroupNorm(16, 128), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.Conv2d(192, 64, 3, padding=1),  nn.GroupNorm(8, 64),   nn.ReLU(inplace=True))
        self.up3 = nn.Sequential(nn.Conv2d(96, 32, 3, padding=1),   nn.GroupNorm(4, 32),   nn.ReLU(inplace=True))
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, fpn):
        x = F.interpolate(self.up1(fpn[2]), scale_factor=2, mode="bilinear", align_corners=False)
        x = F.interpolate(self.up2(torch.cat([x, fpn[1]], 1)), scale_factor=2, mode="bilinear", align_corners=False)
        x = F.interpolate(self.up3(torch.cat([x, fpn[0]], 1)), scale_factor=4, mode="bilinear", align_corners=False)
        return self.out(x)


# ── 工具函数 ──────────────────────────────────────────────────────────────────
def collect_frames(folder):
    """收集病例文件夹里的 130 帧图像路径，按文件名数字排序。"""
    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    img_dir = join(folder, "images") if os.path.isdir(join(folder, "images")) else folder
    cands = [(int(os.path.splitext(fn)[0]), join(img_dir, fn))
             for fn in os.listdir(img_dir)
             if os.path.isfile(join(img_dir, fn)) and fn.lower().endswith(exts)
             and os.path.splitext(fn)[0].isdigit()
             and int(os.path.splitext(fn)[0]) >= 1001]
    return [fp for _, fp in sorted(cands)]


def predict_mask(sam2, adapter, decoder, fps, orig_h, orig_w):
    """对一个病例的 130 帧预测整体分割掩码，返回 (orig_h, orig_w) bool array。"""
    frames = []
    for fp in fps:
        with Image.open(fp) as img:
            arr = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR), dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        frames.append(arr)
    frames_np = np.stack(frames)
    mx = frames_np.max()
    if mx > 255:  frames_np /= 65535.0
    elif mx > 1.0: frames_np /= 255.0
    frames_t = torch.from_numpy(frames_np).unsqueeze(1).to(DEVICE)  # (T,1,H,W)

    with torch.no_grad():
        rgb    = adapter(frames_t)
        feats  = sam2.forward_image(rgb)
        logits = decoder(feats["backbone_fpn"])
        logits = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    return (torch.sigmoid(logits) > 0.5).squeeze().cpu().numpy()  # (orig_h, orig_w)


def extract_single_kidney_mask(full_mask, json_path, side, orig_h, orig_w):
    """
    从 JSON 标注里取出单侧肾脏的多边形，生成该侧的 GT 区域掩码，
    然后与模型预测掩码取交集，得到该侧肾脏的预测区域。
    side: 'l' 或 'r'
    """
    # 构建单侧 GT 区域掩码
    roi = Image.new("L", (orig_w, orig_h), 0)
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        draw = ImageDraw.Draw(roi)
        for shape in data.get("shapes", []):
            if shape.get("label", "").lower() == side:
                pts = shape.get("points", [])
                if len(pts) >= 3:
                    draw.polygon([tuple(p) for p in pts], outline=1, fill=1)
    except Exception:
        pass
    roi_np = np.array(roi, dtype=bool)
    # 预测掩码 & 单侧 ROI 的交集
    return full_mask & roi_np


def compute_tac(fps, mask, orig_h, orig_w):
    """
    在给定掩码区域内，对每帧原图计算像素均值，返回长度=130 的 TAC 曲线。
    """
    tac = np.zeros(NUM_FRAMES, dtype=np.float32)
    if mask.sum() == 0:
        return tac
    for i, fp in enumerate(fps):
        with Image.open(fp) as img:
            arr = np.array(img.resize((orig_w, orig_h), Image.NEAREST), dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        # 归一化到 0-1
        mx = arr.max()
        if mx > 255:  arr /= 65535.0
        elif mx > 1.0: arr /= 255.0
        tac[i] = arr[mask].mean()
    return tac


# ── 主流程 ───────────────────────────────────────────────────────────────────
def main():
    print("Building SAM2...")
    sam2 = build_sam2(CFG_SAM2, CKPT_SAM2, device=DEVICE, mode="eval", apply_postprocessing=False)
    for p in sam2.parameters(): p.requires_grad_(False)
    sam2.eval()

    adapter = TemporalAdapter(image_size=IMAGE_SIZE).to(DEVICE)
    decoder = FPNDecoder().to(DEVICE)
    ckpt = torch.load(TRAINED_WEIGHTS, map_location=DEVICE, weights_only=True)
    adapter.load_state_dict(ckpt["adapter"])
    decoder.load_state_dict(ckpt["decoder"])
    adapter.eval(); decoder.eval()
    print(f"Loaded epoch={ckpt.get('epoch','?')}  val_dice={ckpt.get('val_dice',0):.4f}")

    # 读取 CSV，按 sample_index 排序构建输出数组
    df = pd.read_csv(CSV_PATH)
    n_samples = len(df)
    tacs_out = np.zeros((n_samples, NUM_FRAMES), dtype=np.float32)

    # 建立 patient_id -> 数据集文件夹名的映射
    # CSV 里是 P001~P052，数据集文件夹名是 功能_1~功能_20 / 机械_1~20 / 混合_1~12
    # 按字母顺序排列的文件夹与 P001 顺序对应
    case_folders = sorted([join(DATA_ROOT, n) for n in sorted(os.listdir(DATA_ROOT))
                           if os.path.isdir(join(DATA_ROOT, n))])
    patient_ids  = sorted(df["patient_id"].unique())
    assert len(case_folders) == len(patient_ids), \
        f"病例数不匹配: 文件夹 {len(case_folders)} vs CSV {len(patient_ids)}"
    pid2folder = {pid: folder for pid, folder in zip(patient_ids, case_folders)}
    print(f"共 {n_samples} 条记录，{len(patient_ids)} 个病人")

    for _, row in tqdm(df.iterrows(), total=n_samples, desc="Extracting TAC"):
        pid   = row["patient_id"]
        side  = row["kidney_side"]   # 'left' or 'right'
        idx   = int(row["sample_index"])
        folder = pid2folder[pid]

        fps = collect_frames(folder)
        if len(fps) != NUM_FRAMES:
            print(f"  skip {pid}: {len(fps)} frames")
            continue

        # 找 JSON 标注
        lbl_dir = join(folder, "labels")
        jsons = glob.glob(join(lbl_dir if os.path.isdir(lbl_dir) else folder, "*.json"))
        if not jsons:
            print(f"  skip {pid}: no json")
            continue
        json_path = jsons[0]

        # 获取原图尺寸
        with Image.open(fps[0]) as ref:
            orig_w, orig_h = ref.size

        # 预测整体掩码
        full_mask = predict_mask(sam2, adapter, decoder, fps, orig_h, orig_w)

        # 提取单侧掩码
        side_char = 'l' if side == 'left' else 'r'
        kidney_mask = extract_single_kidney_mask(full_mask, json_path, side_char, orig_h, orig_w)

        # 如果交集为空（模型漏检），退回到 GT ROI 本身
        if kidney_mask.sum() == 0:
            print(f"  {pid} {side}: pred empty, falling back to GT ROI")
            kidney_mask = extract_single_kidney_mask(
                np.ones((orig_h, orig_w), dtype=bool), json_path, side_char, orig_h, orig_w)

        tac = compute_tac(fps, kidney_mask, orig_h, orig_w)
        tacs_out[idx] = tac

    np.save(OUT_NPY, tacs_out)
    print(f"\nTAC saved: {OUT_NPY}  shape={tacs_out.shape}")
    print("Done. Now run pic1.py with TAC_PATH pointing to this file.")


if __name__ == "__main__":
    main()

