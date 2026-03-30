import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 读取 CSV 并合并
# ==========================================
df_clinical   = pd.read_csv('clinical_anchors_peak.csv')  # T_half, SRF
df_centerloss = pd.read_csv('centerloss_results.csv')     # P_mech
df_mil        = pd.read_csv('orthogonal_baseline_MIL.csv')# P_mech_MIL

df = df_clinical.merge(df_centerloss, on=['patient_id', 'kidney_side']) \
                .merge(df_mil,        on=['patient_id', 'kidney_side'])

# 肾积水等级
hydro_map = {
    'P041': 3, 'P042': 2, 'P043': 3, 'P044': 2, 'P045': 3, 'P046': 3,
    'P047': 3, 'P048': 3, 'P049': 1, 'P050': 1, 'P051': 1, 'P052': 3
}
df['Hydro_Level'] = df['patient_id'].map(hydro_map)

# 确保 T_half 是数字类型，防呆处理（防止有文本 'NA' 混入）
df['T_half'] = pd.to_numeric(df['T_half'], errors='coerce').fillna(120.0)

# 单位转换与截断
df['AI_Score']      = df['P_mech']     * 100
df['AI_Score_MIL']  = df['P_mech_MIL'] * 100
df['T_half_capped'] = df['T_half'].clip(upper=120)

# ==========================================
# 🌟 2. 核心：基于 SNMMI 指南的严格临床分级
# 注意：修复了 LaTeX 字符串转义的问题 (使用 \\leq 代替 \le)
# ==========================================
def assign_grade(t_half):
    if t_half < 10:
        return 'Grade 0\nNon-obstructed\n($T_{1/2} < 10$)'
    elif 10 <= t_half <= 20:
        return 'Grade 1\nEquivocal / Gray Zone\n($10 \\leq T_{1/2} \\leq 20$)'
    else:
        return 'Grade 2\nSevere Obstruction\n($T_{1/2} > 20$)'

df['Clinical_Grade'] = df['T_half_capped'].apply(assign_grade)

# 确保箱线图按照从轻到重的顺序排列
grade_order = [
    'Grade 0\nNon-obstructed\n($T_{1/2} < 10$)',
    'Grade 1\nEquivocal / Gray Zone\n($10 \\leq T_{1/2} \\leq 20$)',
    'Grade 2\nSevere Obstruction\n($T_{1/2} > 20$)'
]

# ==========================================
# 3. 统计学检验：Kruskal-Wallis H-test
# ==========================================
def get_kruskal_p(model_col):
    g0 = df[df['Clinical_Grade'] == grade_order[0]][model_col].dropna()
    g1 = df[df['Clinical_Grade'] == grade_order[1]][model_col].dropna()
    g2 = df[df['Clinical_Grade'] == grade_order[2]][model_col].dropna()
    
    # 防止因为某个组别没数据导致 scipy 崩溃
    if len(g0) > 0 and len(g1) > 0 and len(g2) > 0:
        stat, p = kruskal(g0, g1, g2)
        return p
    else:
        return np.nan

p_cl = get_kruskal_p('AI_Score')
p_mil = get_kruskal_p('AI_Score_MIL')

# ==========================================
# 4. 可视化：箱线图 + 散点图
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
sns.set_theme(style='ticks', font_scale=1.1)

# 高亮显示由于肾功能极度衰竭（SRF极低）导致 T1/2 失效的“无功能肾”
anomalies = ['P048', 'P052'] 

for ax, y_col, p_val, title in [
    (axes[0], 'AI_Score',     p_cl,  'Center Loss $P_{mech}$'),
    (axes[1], 'AI_Score_MIL', p_mil, 'MIL Baseline $P_{mech}$')
]:
    # 修复了 Seaborn 的颜色参数：去掉 palette，改用统一的 color
    sns.boxplot(x='Clinical_Grade', y=y_col, data=df, order=grade_order,
                width=0.5, ax=ax, color='lightgray', showfliers=False, 
                boxprops=dict(alpha=0.6, edgecolor='black'))

    # 绘制表层的散点图
    for idx, row in df.iterrows():
        is_anomaly = row['patient_id'] in anomalies
        # 红色 X 代表假阳性异常点，蓝色圆圈代表正常点
        color = '#C44E52' if is_anomaly else '#4C72B0'
        marker = 'X' if is_anomaly else 'o'
        size = 180 if is_anomaly else 120
        
        # 确保 Clinical_Grade 在 grade_order 中，获取正确的 X 轴位置
        if row['Clinical_Grade'] in grade_order:
            x_pos = grade_order.index(row['Clinical_Grade'])
            ax.scatter(x_pos, row[y_col], color=color, marker=marker, 
                       s=size, edgecolors='black', linewidths=1.2, zorder=5)
            
            # 标注病人编号
            ax.annotate(row['patient_id'],
                        (x_pos, row[y_col]),
                        textcoords='offset points', xytext=(8, -5),
                        fontsize=9, color='black', fontweight='bold', zorder=6)

    # 添加显著性 p 值标识
    if pd.notna(p_val):
        p_str = f'Kruskal-Wallis $p = {p_val:.4f}$' + (' **' if p_val < 0.05 else ' (Marginal)')
    else:
        p_str = 'Kruskal-Wallis $p = N/A$ (Data missing)'
        
    ax.text(0.05, 0.95, p_str, transform=ax.transAxes, fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#AAAAAA', alpha=0.9))

    # 在图表右上角添加图例说明
    ax.scatter([], [], color='#4C72B0', marker='o', s=100, edgecolors='black', label='Standard Case')
    ax.scatter([], [], color='#C44E52', marker='X', s=100, edgecolors='black', label='Atypical / Loss of Function\n(e.g., SNMMI False Positive)')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)

    ax.set_title(f'{title}\nStratified by Clinical Guidelines', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Clinical Severity Guideline (SNMMI / EAU)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('AI Prediction $P_{mech}$ Output (%)', fontsize=12, fontweight='bold')
    ax.set_ylim(-10, 115)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    sns.despine(ax=ax)

fig.suptitle('AI $P_{mech}$ Validation via Independent Clinical Grading\n(Overcoming Linear Constraints in Equivocal Cohorts)', 
             fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('clinical_guideline_boxplot.png', dpi=300, bbox_inches='tight')
print('\n✅ 图表已保存为 clinical_guideline_boxplot.png (300 DPI)')
plt.show()
