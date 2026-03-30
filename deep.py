import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# ==========================================
# 0. 环境与配置
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ==========================================
# 1. 动力学潜空间编码器
# ==========================================
class PharmacokineticLatentEncoder(nn.Module):
    def __init__(self, input_dim=130, latent_dim=64, num_classes=2):
        super().__init__()
        self.injection_frame = 83
        self.pre_net = nn.Sequential(
            nn.Linear(self.injection_frame, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True)
        )
        self.post_net = nn.Sequential(
            nn.Linear(input_dim - self.injection_frame, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True)
        )
        self.latent_mapping = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, latent_dim),
            nn.BatchNorm1d(latent_dim)
        )
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        pre_feat  = self.pre_net(x[:, :self.injection_frame])
        post_feat = self.post_net(x[:, self.injection_frame:])
        combined  = torch.cat([pre_feat, post_feat], dim=1)
        latent    = self.latent_mapping(combined)
        logits    = self.classifier(latent)
        return logits, latent

# ==========================================
# 2. 损失函数
# ==========================================
class CenterLoss(nn.Module):
    def __init__(self, num_classes=2, feat_dim=64):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, x, labels):
        batch_size = x.size(0)
        centers = self.centers
        distmat = (torch.pow(x, 2).sum(1, keepdim=True).expand(batch_size, 2) +
                   torch.pow(centers, 2).sum(1, keepdim=True).t().expand(batch_size, 2))
        distmat.addmm_(x, centers.t(), beta=1.0, alpha=-2.0)
        mask = labels.view(-1, 1).eq(
            torch.arange(2, device=x.device).expand(batch_size, 2)
        )
        dist = distmat * mask.float()
        return (dist.sum(0) / mask.sum(0).clamp(min=1)).mean()


class ManifoldOrderLoss(nn.Module):
    """
    核心约束：强制潜空间沿「功能型→机械型」方向形成
    有序的一维流形轨迹。
    
    做法：给每个训练样本分配一个「进度标签」t ∈ [0, 1]
    （0=功能型中心，1=机械型中心），然后要求投影在主轴
    上的排列与 t 单调一致，同时惩罚离轴偏差。
    """
    def __init__(self):
        super().__init__()

    def forward(self, latent, labels, centers):
        """
        latent:  (N, 64)
        labels:  (N,)  0 或 1
        centers: (2, 64)  class centers from CenterLoss
        """
        c0 = centers[0]          # 功能型中心
        c1 = centers[1]          # 机械型中心
        axis = c1 - c0           # 主轴方向向量
        axis_norm = axis / (axis.norm() + 1e-8)

        # 每个样本在主轴上的投影坐标 t_proj
        t_proj = torch.mv(latent - c0.unsqueeze(0), axis_norm)  # (N,)

        # 1. 【排序损失】label=0 的 t 应 < label=1 的 t
        #    用 margin=0: max(0, t_0 - t_1 + margin) for all cross pairs
        t0 = t_proj[labels == 0]   # (N0,)
        t1 = t_proj[labels == 1]   # (N1,)
        # 广播: (N0, N1)
        order_loss = torch.clamp(t0.unsqueeze(1) - t1.unsqueeze(0) + 1.0, min=0).mean()

        # 2. 【离轴损失】惩罚偏离主轴方向的分量，促进流形紧凑
        proj_on_axis = t_proj.unsqueeze(1) * axis_norm.unsqueeze(0)  # (N, 64)
        residual = (latent - c0.unsqueeze(0)) - proj_on_axis         # (N, 64)
        off_axis_loss = residual.pow(2).sum(dim=1).mean()

        return order_loss + 0.1 * off_axis_loss


# ==========================================
# 3. 全维度主路径投影
# ==========================================
def compute_full_dim_principal_projection(latent_feats, labels):
    c0 = latent_feats[labels == 0].mean(axis=0)
    c1 = latent_feats[labels == 1].mean(axis=0)
    main_axis = c1 - c0
    t = np.dot(latent_feats - c0, main_axis) / (np.dot(main_axis, main_axis) + 1e-8)

    t_poly = PolynomialFeatures(degree=2).fit_transform(t.reshape(-1, 1))
    reg = LinearRegression().fit(t_poly, latent_feats)
    project_points = reg.predict(t_poly)
    residuals = latent_feats - project_points
    p_mech_scores = t * 100
    return p_mech_scores, residuals, project_points


# ==========================================
# 4. 训练
# ==========================================
def main():
    set_seed(42)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    print(f"使用设备: {DEVICE}")

    # 加载数据
    tacs = np.load(os.path.join(BASE_DIR, 'extracted_tacs_left_right.npy'))
    df   = pd.read_csv(os.path.join(BASE_DIR, 'clinical_labels_left_right.csv'))
    tacs = (tacs - tacs.min(axis=1, keepdims=True)) / \
           (tacs.max(axis=1, keepdims=True) - tacs.min(axis=1, keepdims=True) + 1e-8)
    class_labels = df['label'].values.astype(np.int64)

    # 仅用纯功能(0)和纯机械(1)训练
    train_mask  = (class_labels == 0) | (class_labels == 1)
    pure_tacs   = torch.tensor(tacs[train_mask],         dtype=torch.float32).to(DEVICE)
    pure_labels = torch.tensor(class_labels[train_mask], dtype=torch.long).to(DEVICE)

    model          = PharmacokineticLatentEncoder().to(DEVICE)
    criterion_cls  = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_cent = CenterLoss().to(DEVICE)
    criterion_mani = ManifoldOrderLoss().to(DEVICE)

    optimizer      = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    optimizer_cent = optim.SGD(criterion_cent.parameters(), lr=0.5)

    print("\n开始带流形约束的潜空间训练...")
    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        optimizer_cent.zero_grad()

        logits, latent = model(pure_tacs)
        l_cls  = criterion_cls(logits, pure_labels)
        l_cent = criterion_cent(latent, pure_labels)
        l_mani = criterion_mani(latent, pure_labels, criterion_cent.centers)

        # 权重：分类 1.0 + 中心聚类 0.5 + 流形排序 2.0
        loss = l_cls + 0.5 * l_cent + 2.0 * l_mani
        loss.backward()
        optimizer.step()
        optimizer_cent.step()

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch [{epoch+1:03d}/300]  Loss={loss.item():.4f}  "
                  f"Cls={l_cls.item():.4f}  Cent={l_cent.item():.4f}  "
                  f"Mani={l_mani.item():.4f}")

    # 推理：全部 104 个样本
    model.eval()
    all_tacs   = torch.tensor(tacs, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        _, all_latent = model(all_tacs)
    all_latent = all_latent.cpu().numpy()
    labels_np  = class_labels

    # 用训练集两极中心做参考
    train_latent = all_latent[train_mask]
    train_labels = class_labels[train_mask]
    p_mech, residuals, proj_pts = compute_full_dim_principal_projection(
        train_latent, train_labels
    )

    # 保存
    torch.save(model.state_dict(), os.path.join(BASE_DIR, 'principal_encoder.pth'))
    np.save(os.path.join(BASE_DIR, 'all_latent_64d.npy'), all_latent)
    np.save(os.path.join(BASE_DIR, 'all_labels.npy'), labels_np)
    print("\n训练完毕！")
    print(f"  模型权重已保存: principal_encoder.pth")
    print(f"  全量潜特征已保存: all_latent_64d.npy  shape={all_latent.shape}")

if __name__ == "__main__":
    main()
