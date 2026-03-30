import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, spearmanr
import os
from deep import PharmacokineticLatentEncoder

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = '/root/medsam'

TAC_PATH    = os.path.join(BASE_DIR, 'extracted_tacs_left_right.npy')
CSV_PATH    = os.path.join(BASE_DIR, 'clinical_labels_left_right.csv')
CLIN_PATH   = os.path.join(BASE_DIR, 'clinical_anchors_peak.csv')
MODEL_PATH  = os.path.join(BASE_DIR, 'latent_encoder.pth')
CENTER_PATH = os.path.join(BASE_DIR, 'manifold_centers.pth')
OUT_CSV     = os.path.join(BASE_DIR, 'centerloss_results.csv')
OUT_PNG     = os.path.join(BASE_DIR, 'final_clinical_validation.png')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def manifold_projection(latent_feats, c0, c1):
    axis_vec   = c1 - c0
    sample_vec = latent_feats - c0
    dot        = torch.sum(sample_vec * axis_vec, dim=1)
    axis_sq    = torch.sum(axis_vec ** 2) + 1e-8
    return dot / axis_sq
    

def run_prediction():
    # A. 加载模型与极点中心
    model = PharmacokineticLatentEncoder(
        input_dim=130, latent_dim=64, num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    centers = torch.load(CENTER_PATH, map_location=DEVICE)
    c0, c1  = centers[0], centers[1]
    print('模型与极点中心加载完毕。设备:', DEVICE)

    # B. 加载 TAC 并归一化
    tacs_raw  = np.load(TAC_PATH)
    tacs_min  = tacs_raw.min(axis=1, keepdims=True)
    tacs_max  = tacs_raw.max(axis=1, keepdims=True)
    tacs_norm = (tacs_raw - tacs_min) / (tacs_max - tacs_min + 1e-8)
    tacs_t    = torch.tensor(tacs_norm, dtype=torch.float32).to(DEVICE)
    
    # C. 读取标签与临床数据
    df_label = pd.read_csv(CSV_PATH)
    df_clin  = pd.read_csv(CLIN_PATH)
    df_clin['T_half'] = pd.to_numeric(df_clin['T_half'], errors='coerce').fillna(120.0)

    # D. 计算全部104条 TAC 的投影得分
    with torch.no_grad():
        _, latent_feats = model(tacs_t)
        scores = manifold_projection(latent_feats, c0, c1)

    df_label['P_mech'] = scores.cpu().numpy()

    # E. 筛选混合型 (label == 2)
    df_mixed = df_label[df_label['label'] == 2].copy().reset_index(drop=True)
    df_mixed = df_mixed.merge(
        df_clin[['patient_id', 'kidney_side', 'T_half', 'SRF']],
        on=['patient_id', 'kidney_side'], how='left'
    )
    df_mixed['T_half_capped'] = df_mixed['T_half'].clip(upper=120)
    df_mixed['P_mech_pct']    = df_mixed['P_mech'] * 100.0

    print('\n混合型梗阻样本数:', len(df_mixed))

    # F. 打印梗阻占比清单
    print('\n=== 交叉型梗阻 (混合型) 机械性梗阻占比 ===')
    header = '{:<8} {:<6} {:>12} {:>8} {:>11}'.format(
        '病人ID', '肾侧', 'T_half(min)', 'SRF(%)', 'P_mech(%)')
    print(header)
    print('-' * 50)
    ANOMALIES = ['P048', 'P052']
    for _, r in df_mixed.sort_values('T_half').iterrows():
        flag = '  <- 无功能肾' if r['patient_id'] in ANOMALIES else ''
        row_str = '{:<8} {:<6} {:>12.1f} {:>8.1f} {:>11.1f}%{}'.format(
            r['patient_id'], r['kidney_side'],
            r['T_half'], r['SRF'], r['P_mech_pct'], flag)
        print(row_str)

    # G. 保存结果 CSV
    df_mixed[['patient_id', 'kidney_side', 'P_mech']].to_csv(OUT_CSV, index=False)
    print('\nP_mech 结果已保存至:', OUT_CSV)

    # H. 临床分级
    def assign_grade(t):
        if t < 10:   return 'Grade 0\nNon-obstructed\n(T1/2 < 10 min)'
        if t <= 20:  return 'Grade 1\nEquivocal\n(10 <= T1/2 <= 20 min)'
        return               'Grade 2\nObstruction\n(T1/2 > 20 min)'

    grade_order = [
        'Grade 0\nNon-obstructed\n(T1/2 < 10 min)',
        'Grade 1\nEquivocal\n(10 <= T1/2 <= 20 min)',
        'Grade 2\nObstruction\n(T1/2 > 20 min)',
    ]
    df_mixed['Grade'] = df_mixed['T_half_capped'].apply(assign_grade)

    # 统计检验
    groups = [df_mixed[df_mixed['Grade'] == g]['P_mech_pct'].dropna() for g in grade_order]
    valid_groups = [g for g in groups if len(g) > 0]
    if len(valid_groups) >= 2:
        stat, p_kw = kruskal(*valid_groups)
    else:
        stat, p_kw = 0.0, float('nan')
    rho, p_sp = spearmanr(df_mixed['T_half_capped'], df_mixed['P_mech_pct'])

    print('\nKruskal-Wallis: H={:.3f}, p={:.4f}'.format(stat, p_kw))
    print('Spearman r={:.3f}, p={:.4f}'.format(rho, p_sp))

    # I. 绘图
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.set_theme(style='ticks', font_scale=1.1)

    sns.boxplot(x='Grade', y='P_mech_pct', data=df_mixed, order=grade_order,
                width=0.45, ax=ax, color='lightgray', showfliers=False,
                boxprops=dict(alpha=0.6, edgecolor='black'))

    for _, row in df_mixed.iterrows():
        if row['Grade'] not in grade_order:
            continue
        xpos    = grade_order.index(row['Grade'])
        is_anom = row['patient_id'] in ANOMALIES
        color   = '#C44E52' if is_anom else '#4C72B0'
        marker  = 'X'       if is_anom else 'o'
        size    = 180       if is_anom else 130
        ax.scatter(xpos, row['P_mech_pct'], color=color, marker=marker,
                   s=size, edgecolors='black', linewidths=1.2, zorder=5)
        ax.annotate(row['patient_id'], (xpos, row['P_mech_pct']),
                    textcoords='offset points', xytext=(8, -5),
                    fontsize=9, fontweight='bold', zorder=6)

    if not pd.isna(p_kw):
        stars = ('****' if p_kw < 0.0001 else '***' if p_kw < 0.001
                 else '**' if p_kw < 0.01 else '*' if p_kw < 0.05 else 'n.s.')
        pkw_txt = 'Kruskal-Wallis p = {:.4f} {}'.format(p_kw, stars)
    else:
        pkw_txt = 'Kruskal-Wallis p = N/A'
    sp_txt = 'Spearman r = {:.3f},  p = {:.4f}'.format(rho, p_sp)

    ax.text(0.04, 0.97, pkw_txt, transform=ax.transAxes, fontsize=11,
            fontweight='bold', va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#AAAAAA', alpha=0.9))
    ax.text(0.04, 0.87, sp_txt, transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#EEF4FF',
                      edgecolor='#6699CC', alpha=0.9))

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0], marker='o', color='w', label='Standard case',
               markerfacecolor='#4C72B0', markersize=10, markeredgecolor='k'),
        Line2D([0],[0], marker='X', color='w',
               label='Loss-of-Function kidney (P048/P052)',
               markerfacecolor='#C44E52', markersize=12, markeredgecolor='k'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    ax.set_title(
        'Center Loss P_mech vs. SNMMI Clinical Grading\n'
        '(Mixed-type Obstructed Kidneys, n=12)',
        fontsize=14, fontweight='bold', pad=14)
    ax.set_xlabel('Clinical Severity Grade (SNMMI / EAU, based on T1/2)',
                  fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel('Mechanical Obstruction Score P_mech (%)',
                  fontsize=12, fontweight='bold')
    ax.set_ylim(-15, 120)
    ax.grid(axis='y', linestyle='--', alpha=0.55)
    sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    print('图表已保存至:', OUT_PNG)
    plt.show()


if __name__ == '__main__':
    run_prediction()
