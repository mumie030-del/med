import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

# ==========================================
# 0. 严谨性设置：固定随机种子与设备
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. 核心架构：1D 时序 Gated Attention MIL
# ==========================================
class TemporalGatedAttentionMIL(nn.Module):
    def __init__(self, seq_len=130, instance_dim=32, latent_dim=64):
        super(TemporalGatedAttentionMIL, self).__init__()
        
        # [实例特征提取器] 
        # 使用 1D 卷积在时间轴上滑动，提取局部上下文特征，天然抑制泊松毛刺
        # 输入: (B, 1, 130) -> 输出: (B, instance_dim, 130)
        self.instance_encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=16, out_channels=instance_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # [Gated Attention 注意力机制]
        # 评估130个帧中，哪些帧对判断"机械型梗阻"最有价值
        self.attention_V = nn.Sequential(
            nn.Linear(instance_dim, latent_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(instance_dim, latent_dim),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(latent_dim, 1)
        
        # [包级别分类器]
        # 输入聚合后的单一特征向量，输出二分类 Logits
        self.classifier = nn.Sequential(
            nn.Linear(instance_dim, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # 防止小样本过拟合
            nn.Linear(16, 1)  # 输出层不加 Sigmoid，配合 BCEWithLogitsLoss
        )

    def forward(self, x):
        # x shape: (B, 130)
        x = x.unsqueeze(1)  # 转为 (B, 1, 130) 适配 Conv1d
        
        # 1. 提取实例特征
        H = self.instance_encoder(x)  # (B, 32, 130)
        H = H.permute(0, 2, 1)        # 转为 (B, 130, 32) 视作 130 个实例
        
        # 2. 计算门控注意力权重
        A_V = self.attention_V(H)   # (B, 130, 64)
        A_U = self.attention_U(H)   # (B, 130, 64)
        A_score = self.attention_weights(A_V * A_U)  # (B, 130, 1)
        
        A = F.softmax(A_score, dim=1)  # (B, 130, 1) 沿时间维度归一化
        
        # 3. 实例特征加权聚合 (Bag Representation)
        # H: (B, 130, 32), A: (B, 130, 1) -> 广播相乘后在维度1求和
        M = torch.sum(A * H, dim=1)  # (B, 32)
        
        # 4. 最终分类
        logits = self.classifier(M)  # (B, 1)
        
        return logits, A


# ==========================================
# 2. 数据处理与加载
# ==========================================
def load_and_prepare_data(npy_path, csv_path):
    tacs = np.load(npy_path)
    df = pd.read_csv(csv_path)
    
    labels = df['label'].values.astype(np.int64)
    patient_ids = df['patient_id'].values
    kidney_sides = df['kidney_side'].values
    
    # Min-Max 归一化 (每一条曲线独立归一化到 0-1)
    tacs_min = tacs.min(axis=1, keepdims=True)
    tacs_max = tacs.max(axis=1, keepdims=True)
    tacs = (tacs - tacs_min) / (tacs_max - tacs_min + 1e-8)
    
    return tacs, labels, patient_ids, kidney_sides


# ==========================================
# 3. 主干流程：训练与正交验证
# ==========================================
def run_mil_orthogonal_validation():
    print('=' * 50)
    print('启动 Attention-MIL 跨算法共识验证')
    print('训练集: 纯极性 8 例 | 验证集: 混合型 4 例')
    print('=' * 50)
    
    TACS_PATH = 'extracted_tacs_left_right.npy'
    CSV_PATH  = 'clinical_labels_left_right.csv'
    
    if not os.path.exists(TACS_PATH):
        raise FileNotFoundError('找不到文件 ' + TACS_PATH + '，请确认路径。')

    # 1. 加载数据
    tacs, labels, pids, sides = load_and_prepare_data(TACS_PATH, CSV_PATH)
    
    # ==========================================
    # 2. 严格划分数据集
    # 训练集：从纯功能型(0)随机抽4个，从纯机械型(1)随机抽4个，共8个
    # 验证集：从混合型(2)随机抽4个
    # ==========================================
    set_seed(42)  # 保证可复现

    idx_func = np.where(labels == 0)[0]
    idx_mech = np.where(labels == 1)[0]
    idx_mixed = np.where(labels == 2)[0]

    idx_func_sel  = np.random.choice(idx_func,  size=4, replace=False)
    idx_mech_sel  = np.random.choice(idx_mech,  size=4, replace=False)
    idx_mixed_sel = np.sort(np.random.choice(idx_mixed, size=4, replace=False))

    train_idx = np.concatenate([idx_func_sel, idx_mech_sel])

    train_tacs      = torch.tensor(tacs[train_idx], dtype=torch.float32)
    train_labels_np = np.where(labels[train_idx] == 0, 0.0, 1.0)
    train_labels    = torch.tensor(train_labels_np, dtype=torch.float32).unsqueeze(1)
    
    mixed_tacs  = torch.tensor(tacs[idx_mixed_sel], dtype=torch.float32)
    mixed_pids  = pids[idx_mixed_sel]
    mixed_sides = sides[idx_mixed_sel]
    
    print('\n训练集选取:')
    print('  功能型(0) 索引: ' + str(sorted(idx_func_sel.tolist())))
    print('  机械型(1) 索引: ' + str(sorted(idx_mech_sel.tolist())))
    print('验证集选取:')
    print('  混合型(2) 病人: ' + str(list(zip(mixed_pids, mixed_sides))))

    print('\n数据总览:')
    print('  训练集 (纯极性 8例): 功能型4例 + 机械型4例')
    print('  验证集 (混合型 4例): ' + str(list(mixed_pids)))

    train_loader = DataLoader(
        TensorDataset(train_tacs, train_labels),
        batch_size=4, shuffle=True
    )
    mixed_loader = DataLoader(mixed_tacs, batch_size=1, shuffle=False)

    # 3. 初始化模型、损失函数与优化器
    model     = TemporalGatedAttentionMIL().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()  # 内部自带 Sigmoid，数值更稳定
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 4. 训练模型
    print('\n开始训练 MIL 网络 (构建二元判定基准)...')
    model.train()
    epochs = 200
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            preds = torch.sigmoid(logits) > 0.5
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
            
        if (epoch + 1) % 40 == 0:
            acc = correct / total * 100
            print('  Epoch [' + str(epoch+1).zfill(3) + '/' + str(epochs) + ']'
                  + ' | Loss: ' + str(round(epoch_loss / len(train_loader), 4))
                  + ' | Acc: ' + str(round(acc, 1)) + '%')

    # 5. 推理阶段：对选出的4个混合型样本打分
    print('\n训练完成。开始对 [混合型4例] 输出 MIL 概率打分...')
    model.eval()
    
    results = []
    with torch.no_grad():
        for i, tac in enumerate(mixed_loader):
            tac = tac.to(DEVICE)
            logits, attention_weights = model(tac)
            
            p_mech_mil = torch.sigmoid(logits).item()
            
            # 打药后(第83帧之后)的注意力权重之和
            post_diuretic_attn = attention_weights[0, 83:, 0].sum().item()
            
            results.append({
                'patient_id': mixed_pids[i],
                'kidney_side': mixed_sides[i],
                'P_mech_MIL': round(p_mech_mil, 4),
                'Post_Diuretic_Attention_Sum': round(post_diuretic_attn, 4)
            })
            
            print('  [' + mixed_pids[i] + ' - ' + mixed_sides[i] + ']'
                  + '  P_mech_MIL = ' + str(round(p_mech_mil * 100, 1)) + '%')

    # 6. 保存结果
    df_res = pd.DataFrame(results)
    out_file = 'orthogonal_baseline_MIL.csv'
    df_res.to_csv(out_file, index=False)
    print('\n跨算法共识验证完成！')
    print('结果已保存至: ' + out_file)


if __name__ == '__main__':
    run_mil_orthogonal_validation()
