
import os, sys, json, glob
from os.path import join
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from PIL import Image, ImageDraw

MEDSAM2_DIR = "/root/medsam/MedSAM2"
DATA_ROOT   = "/root/new_dataset"
CKPT_SAM2   = "/root/medsam/sam2.1_hiera_tiny.pt"
CFG_SAM2    = "configs/sam2.1_hiera_t512.yaml"
CKPT_DIR    = "/root/medsam/checkpoints"
OUT_PLOT    = "/root/medsam/overfit_check.png"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE  = 512
NUM_FRAMES  = 130

sys.path.insert(0, MEDSAM2_DIR)
os.chdir(MEDSAM2_DIR)

from sam2.build_sam import build_sam2


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
        self.register_buffer("mean", torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))
    def forward(self, x):
        T, C, H, W = x.shape
        gap = x.mean(dim=[1,2,3])
        w = torch.sigmoid(self.se_fc2(F.relu(self.se_fc1(gap))))
        x = x * w.view(T,1,1,1)
        x = x.mean(dim=0, keepdim=True)
        if H != self.image_size or W != self.image_size:
            x = F.interpolate(x, size=(self.image_size,self.image_size), mode="bilinear", align_corners=False)
        return (self.frame_proj(x) - self.mean) / self.std


class FPNDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = nn.Sequential(nn.Conv2d(256,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.Conv2d(192,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.up3 = nn.Sequential(nn.Conv2d(96,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.out = nn.Conv2d(32,1,1)
    def forward(self, fpn):
        x = F.interpolate(self.up1(fpn[2]), scale_factor=2, mode="bilinear", align_corners=False)
        x = F.interpolate(self.up2(torch.cat([x,fpn[1]],1)), scale_factor=2, mode="bilinear", align_corners=False)
        x = F.interpolate(self.up3(torch.cat([x,fpn[0]],1)), scale_factor=4, mode="bilinear", align_corners=False)
        return self.out(x)


class KidneyDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, target_size=(512,512)):
        self.target_size = target_size
        self.samples = []
        def _collect(d):
            exts = (".jpg",".jpeg",".png",".tif",".tiff")
            cands = [(int(os.path.splitext(fn)[0]), join(d,fn))
                     for fn in os.listdir(d)
                     if os.path.isfile(join(d,fn)) and fn.lower().endswith(exts)
                     and os.path.splitext(fn)[0].isdigit() and int(os.path.splitext(fn)[0])>=1001]
            return [fp for _,fp in sorted(cands)]
        for name in sorted(os.listdir(data_root)):
            folder = join(data_root, name)
            if not os.path.isdir(folder): continue
            img_dir = join(folder,"images")
            fps = _collect(img_dir if os.path.isdir(img_dir) else folder)
            if len(fps) != NUM_FRAMES: continue
            lbl_dir = join(folder,"labels")
            jsons = glob.glob(join(lbl_dir if os.path.isdir(lbl_dir) else folder,"*.json"))
            if jsons: self.samples.append((fps, jsons[0]))
    def _json_mask(self, jp, oh, ow):
        th,tw = self.target_size
        mask = Image.new("L",(ow,oh),0)
        try:
            with open(jp,encoding="utf-8") as f: data=json.load(f)
            draw=ImageDraw.Draw(mask)
            for s in data.get("shapes",[]):
                pts=s.get("points",[])
                if len(pts)>=3: draw.polygon([tuple(p) for p in pts],outline=1,fill=1)
        except: pass
        if (ow,oh)!=(tw,th): mask=mask.resize((tw,th),Image.NEAREST)
        return np.array(mask,dtype=np.float32)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        fps,jp = self.samples[idx]
        th,tw = self.target_size
        with Image.open(fps[0]) as r: ow,oh=r.size
        frames=[]
        for fp in fps:
            with Image.open(fp) as img:
                arr=np.array(img.resize((tw,th),Image.BILINEAR),dtype=np.float32)
                if arr.ndim==3: arr=arr[...,:3]@np.array([0.299,0.587,0.114],dtype=np.float32)
            frames.append(arr)
        fn=np.stack(frames); mx=fn.max()
        if mx>255: fn/=65535.0
        elif mx>1.0: fn/=255.0
        return torch.from_numpy(fn).unsqueeze(1), torch.from_numpy(self._json_mask(jp,oh,ow)).unsqueeze(0)


def eval_dice(sam2, adapter, decoder, loader):
    adapter.eval(); decoder.eval()
    total=0.0
    with torch.no_grad():
        for frames,masks in loader:
            frames=frames.squeeze(0).to(DEVICE)
            masks=masks.to(DEVICE).float()
            rgb=adapter(frames)
            fpn=sam2.forward_image(rgb)["backbone_fpn"]
            logits=decoder(fpn)
            gt=F.interpolate(masks,size=logits.shape[-2:],mode="nearest")
            pred=(torch.sigmoid(logits)>0.5).float()
            pf=pred.view(-1); gf=gt.view(-1)
            inter=(pf*gf).sum()
            total+=((2*inter+1e-5)/(pf.sum()+gf.sum()+1e-5)).item()
    return total/len(loader)


print("Building SAM2...")
sam2=build_sam2(CFG_SAM2,CKPT_SAM2,device=DEVICE,mode="eval",apply_postprocessing=False)
for p in sam2.parameters(): p.requires_grad_(False)
sam2.eval()

dataset=KidneyDataset(DATA_ROOT)
n_val=max(1,int(len(dataset)*0.1))
n_train=len(dataset)-n_val
train_ds,val_ds=random_split(dataset,[n_train,n_val],generator=torch.Generator().manual_seed(42))
train_loader=DataLoader(train_ds,batch_size=1,shuffle=False,num_workers=2)
val_loader  =DataLoader(val_ds,  batch_size=1,shuffle=False,num_workers=2)

# 收集所有 epoch checkpoint
ckpt_files=[]
for f in sorted(os.listdir(CKPT_DIR)):
    if f.startswith("epoch_") and f.endswith(".pt"):
        ep=int(f.split("_")[1].split(".")[0])
        ckpt_files.append((ep, join(CKPT_DIR,f)))
# 加入 best
best_path=join(CKPT_DIR,"best_model.pt")
bk=torch.load(best_path,map_location="cpu",weights_only=True)
best_ep=bk["epoch"]

epochs_list, train_dices, val_dices=[],[],[]

for ep,fpath in ckpt_files:
    print(f"Evaluating epoch {ep}...")
    adapter=TemporalAdapter(NUM_FRAMES,image_size=IMAGE_SIZE).to(DEVICE)
    decoder=FPNDecoder().to(DEVICE)
    ck=torch.load(fpath,map_location=DEVICE,weights_only=True)
    adapter.load_state_dict(ck["adapter"])
    decoder.load_state_dict(ck["decoder"])
    td=eval_dice(sam2,adapter,decoder,train_loader)
    vd=eval_dice(sam2,adapter,decoder,val_loader)
    print(f"  epoch {ep}: train_dice={td:.4f}  val_dice={vd:.4f}")
    epochs_list.append(ep)
    train_dices.append(td)
    val_dices.append(vd)

# 画图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(10,6))
ax.plot(epochs_list,train_dices,"o-",color="#2196F3",lw=2,ms=6,label="Train Dice")
ax.plot(epochs_list,val_dices,  "s--",color="#F44336",lw=2,ms=6,label="Val Dice")
ax.axvline(best_ep,color="gold",ls=":",lw=1.5,label=f"Best epoch ({best_ep})")
gap=[t-v for t,v in zip(train_dices,val_dices)]
max_gap=max(gap); max_gap_ep=epochs_list[gap.index(max_gap)]
ax.annotate(f"max gap={max_gap:.3f}",xy=(max_gap_ep,val_dices[gap.index(max_gap)]),
            xytext=(max_gap_ep+1,val_dices[gap.index(max_gap)]-0.03),
            arrowprops=dict(arrowstyle="->",color="gray"),fontsize=10,color="gray")
ax.set_xlabel("Epoch",fontsize=13)
ax.set_ylabel("Dice Score",fontsize=13)
ax.set_title("过拟合分析: Train vs Val Dice",fontsize=15)
ax.legend(fontsize=12); ax.grid(True,alpha=0.3)
ax.set_ylim(0.5,1.0)
plt.tight_layout()
plt.savefig(OUT_PLOT,dpi=150)
print(f"图已保存: {OUT_PLOT}")
print(f"Train-Val gap per epoch: {dict(zip(epochs_list,[f+str(round(g,4)) for f,g in zip([chr(43)]*len(gap) if [x>=0 for x in gap] else [''],gap)]))}") 
