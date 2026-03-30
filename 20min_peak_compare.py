import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def plot_clinical_anchor():
    print('=' * 50)
    print('Clinical Anchor Validation: AI P_mech vs T_half')
    print('=' * 50)

    AI_RESULT_FILE = 'centerloss_results.csv'
    CLINICAL_FILE  = 'clinical_anchors_peak.csv'

    df_ai       = pd.read_csv(AI_RESULT_FILE)
    df_clinical = pd.read_csv(CLINICAL_FILE)

    df_merged = pd.merge(df_ai, df_clinical, on=['patient_id', 'kidney_side'], how='inner')
    df_merged = df_merged.dropna(subset=['T_half', 'P_mech'])

    n = len(df_merged)
    print(f'Aligned samples: {n}\n')

    if n < 3:
        print('Not enough data for correlation analysis (need at least 3 samples).')
        print('Please fill in T_half values in clinical_anchors.csv first.')
        return

    x = df_merged['P_mech'].values * 100   # AI 机械性占比 (%)
    y = df_merged['T_half'].values          # 临床半排期 (min)

    pearson_r,  pearson_p  = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)

    print('Statistical correlation:')
    print(f'  Pearson  r = {pearson_r:.3f}  (p = {pearson_p:.4f})')
    print(f'  Spearman r = {spearman_r:.3f}  (p = {spearman_p:.4f})')
    print()

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    fig.patch.set_facecolor('white')

    sns.regplot(
        x=x, y=y, ax=ax,
        scatter_kws={'s': 80, 'alpha': 0.85, 'edgecolor': 'white', 'linewidths': 0.8},
        line_kws={'color': '#C44E52', 'linewidth': 2, 'linestyle': '--'}
    )

    # 标注每个病例
    for _, row in df_merged.iterrows():
        ax.annotate(row['patient_id'],
                    (row['P_mech'] * 100, row['T_half']),
                    textcoords='offset points', xytext=(6, 4),
                    fontsize=8, color='#444444')

    # 相关系数文本框
    stats_text = (f'Pearson $r$ = {pearson_r:.3f} ($p$ = {pearson_p:.3f})\n'
                  f'Spearman $\\rho$ = {spearman_r:.3f} ($p$ = {spearman_p:.3f})')
    ax.text(0.05, 0.95, stats_text,
            transform=ax.transAxes, fontsize=11, va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      alpha=0.85, edgecolor='#AAAAAA'))

    # 临床参考线
    ax.axhline(y=10, color='#4C72B0', linestyle='-.', alpha=0.5,
               label='10 min (normal upper limit)')
    ax.axhline(y=20, color='#F5964F', linestyle='-.', alpha=0.5,
               label='20 min (mechanical lower limit)')

    ax.set_xlabel('AI Center Loss $P_{mech}$ (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Clinical $T_{1/2}$ (min)', fontsize=13, fontweight='bold')
    ax.set_title(f'Clinical Anchor Validation\nAI $P_{{mech}}$ vs Diuretic $T_{{1/2}}$ (N={n})',
                 fontsize=14, fontweight='bold', pad=12)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = 'clinical_anchor_validation1.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.savefig('clinical_anchor_validation.pdf', dpi=300,
                format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Figure saved: {out}')


if __name__ == '__main__':
    plot_clinical_anchor()
