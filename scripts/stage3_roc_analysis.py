"""Prognostic value of Age Gap for 7-day mortality: ROC analysis and cumulative occlusion strata.

Section 3 compares Age Gap, FaceAge, chronological age and APACHE II between survivors and
non-survivors; section 3b computes each one's AUC against 7-day mortality and draws the ROC
curves (Table 5, Figure 3); sections 6 and 7 repeat the analysis after the quadratic bias
correction of Age Gap and across cumulative occlusion strata (Table 6).

Relationship to other scripts
-----------------------------
Superseded in part by fig6_bias_corrected_roc.py. Section 6 writes two files, and they have
different status:

  roc_analysis_corrected.csv    live — the source of the bias-corrected AUCs quoted for Age Gap
                                and FaceAge
  roc_curves_corrected.png      superseded — the six-curve figure was replaced by
                                fig6_bias_corrected_roc.py, because two of its six curves
                                duplicated Figure 3 and the argument only needed the other four

The correction is a quadratic regression of Age Gap on chronological age, cross-fitted in 10
folds stratified by outcome, so that no patient contributes to their own correction.

Input  : data/merged_data.xlsx   (analysis set, occlusion grade <= 2)
Output : results/stage3_roc_analysis/death_survival_comparison.csv
         results/stage3_roc_analysis/roc_analysis.csv
         results/stage3_roc_analysis/roc_curves.png
         results/stage3_roc_analysis/roc_analysis_corrected.csv
         results/stage3_roc_analysis/roc_curves_corrected.png     (superseded; see above)
         results/stage3_roc_analysis/cumulative_roc_analysis.csv
         results/stage3_roc_analysis/roc_curves_age_gap_cumulative.png"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold

# 添加utils到路径
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    load_from_merged_xlsx,
    print_data_summary,
    roc_auc_analysis,
    plot_roc_curves,
    RESULTS_DIR
)
from utils.statistical_tests import calculate_roc_auc, format_significance
from utils.config import OCCLUSION_LEVELS

print("=" * 80)
print("Stage 3: Age Gap 预后预测价值评估（ROC分析 + 偏差校正 + 累计遮挡分层）")
print("=" * 80)
print()

# ============================================================================
# 1. 加载数据
# ============================================================================

print("[1/8] 加载数据...")
df = load_from_merged_xlsx(max_occlusion_level=2)  # 排除重度遮挡
print_data_summary(df, "完整数据集（遮挡 <= 2）")

# ============================================================================
# 2. 回归均值偏差校正（二次多项式回归）
# ============================================================================

print("[2/8] 回归均值偏差校正（交叉拟合，二次多项式）...")

# 校正曲线按死亡结局分层做 10 折交叉拟合：每折只用训练行拟合 Age_Gap ~ 实际年龄，
# 再对验证行取残差。这样每位患者的校正值都来自一条"没见过自己"的曲线，
# 避免在同一批数据上既拟合又评估（样本内）造成乐观偏倚。
# 注：不用 sklearn.LinearRegression —— 本环境 scipy.linalg.lstsq 会使进程静默崩溃，
#     numpy.polyfit 数学等价且稳定。
valid = df['Age_Gap'].notna() & df['实际年龄'].notna()
if not valid.all():
    raise ValueError("Age_Gap 存在缺失，与手稿的完整例分析口径不一致")

age_arr = df['实际年龄'].values.astype(float)
gap_arr = df['Age_Gap'].values.astype(float)
y_arr = df['7天死亡'].values.astype(int)
DEGREE = 2
N_FOLDS = 10
SEED = 42

gap_corrected = np.full(len(df), np.nan, dtype=float)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
for tr, te in skf.split(age_arr, y_arr):
    coef = np.polyfit(age_arr[tr], gap_arr[tr], DEGREE)
    gap_corrected[te] = gap_arr[te] - np.polyval(coef, age_arr[te])
if np.isnan(gap_corrected).any():
    raise RuntimeError("交叉拟合未覆盖全部样本")

df['Age_Gap_corrected'] = gap_corrected
df['FaceAge_corrected'] = df['实际年龄'] + df['Age_Gap_corrected']

coef_full = np.polyfit(age_arr, gap_arr, DEGREE)
print(f"  全队列二次拟合系数: {np.round(coef_full, 6)}")
print(f"  校正前 Age Gap: {df['Age_Gap'].mean():+.2f} ± {df['Age_Gap'].std():.2f}")
print(f"  校正后 Age Gap: {df['Age_Gap_corrected'].mean():+.2f} ± {df['Age_Gap_corrected'].std():.2f}")
print(f"  校正前 FaceAge: {df['生物学年龄'].mean():.2f} ± {df['生物学年龄'].std():.2f}")
print(f"  校正后 FaceAge: {df['FaceAge_corrected'].mean():.2f} ± {df['FaceAge_corrected'].std():.2f}")
print()

# ============================================================================
# 3. 死亡/生存组间比较（Age Gap、FaceAge、实际年龄、APACHE）
# ============================================================================

print("[3/8] 死亡/生存组间比较...")

output_dir = RESULTS_DIR / "stage3_roc_analysis"
output_dir.mkdir(parents=True, exist_ok=True)

y_true = df['7天死亡']
surv_mask = (y_true == 0)
dead_mask = (y_true == 1)

comparison_vars = {
    'Age Gap': 'Age_Gap',
    'Age Gap (corrected)': 'Age_Gap_corrected',
    'FaceAge': '生物学年龄',
    'FaceAge (corrected)': 'FaceAge_corrected',
    'Chronological Age': '实际年龄',
    'APACHE II': 'APACHE评分'
}

comparison_results = []
for label, col in comparison_vars.items():
    surv_vals = df.loc[surv_mask, col].dropna()
    dead_vals = df.loc[dead_mask, col].dropna()

    n_surv, n_dead = len(surv_vals), len(dead_vals)
    mean_surv, std_surv = surv_vals.mean(), surv_vals.std()
    mean_dead, std_dead = dead_vals.mean(), dead_vals.std()

    u_stat, p_val = stats.mannwhitneyu(surv_vals, dead_vals, alternative='two-sided')

    comparison_results.append({
        '变量': label,
        '生存组_n': n_surv,
        '生存组_均值': round(mean_surv, 2),
        '生存组_标准差': round(std_surv, 2),
        '死亡组_n': n_dead,
        '死亡组_均值': round(mean_dead, 2),
        '死亡组_标准差': round(std_dead, 2),
        'U统计量': int(u_stat),
        'p值': p_val,
        '显著性': '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
    })

comparison_df = pd.DataFrame(comparison_results)
comparison_df.to_csv(output_dir / "death_survival_comparison.csv", index=False, encoding='utf-8-sig')
print(f"  [OK] 保存结果: {output_dir / 'death_survival_comparison.csv'}")

print("\n死亡/生存组间比较结果:")
print("-" * 85)
print(f"{'变量':<22} {'生存组 (n={})'.format(int(df['7天死亡'].value_counts().get(0,0))):<22} "
      f"{'死亡组 (n={})'.format(int(df['7天死亡'].value_counts().get(1,0))):<22} {'p值':>8} {'显著性':>4}")
print("-" * 85)
for _, row in comparison_df.iterrows():
    surv_str = f"{row['生存组_均值']:.2f} ± {row['生存组_标准差']:.2f}"
    dead_str = f"{row['死亡组_均值']:.2f} ± {row['死亡组_标准差']:.2f}"
    print(f"{row['变量']:<22} {surv_str:<22} {dead_str:<22} {row['p值']:>8.4f} {row['显著性']:>4}")
print("-" * 85)

# 特别提示 Age Gap 方向
ag_row = comparison_df[comparison_df['变量'] == 'Age Gap'].iloc[0]
if ag_row['死亡组_均值'] < ag_row['生存组_均值']:
    print(f"\n  *** 注意: Age Gap 在死亡组中反而更低 ({ag_row['死亡组_均值']:.2f} vs {ag_row['生存组_均值']:.2f})，")
    print(f"      与直觉相反，提示可能存在年龄分层混杂。")
print()

# ============================================================================
# 3. ROC 分析（主要）
# ============================================================================

print("[4/8] ROC 分析（原始4变量对比）...")

y_true = df['7天死亡']

y_scores_dict = {
    'Age Gap': df['Age_Gap'],
    'Chronological Age': df['实际年龄'],
    'FaceAge': df['生物学年龄'],
    'APACHE II': df['APACHE评分']
}

# 计算AUC
results_df = roc_auc_analysis(y_true, y_scores_dict)

# 保存结果
results_df.to_csv(output_dir / "roc_analysis.csv", index=False, encoding='utf-8-sig')
print(f"  [OK] 保存结果: {output_dir / 'roc_analysis.csv'}")

# 显示结果
print("\nROC 分析结果:")
print(results_df.to_string(index=False))
print()

# ============================================================================
# 4. 绘制 ROC 曲线（主要）
# ============================================================================

print("[5/8] 绘制主 ROC 曲线...")

plot_roc_curves(
    y_true,
    y_scores_dict,
    title='ROC Curves for 7-Day Mortality Prediction (N = 1,470)',
    save_path=output_dir / "roc_curves.png"
)

# ============================================================================
# 6. 偏差校正后 ROC 分析（原始 vs 校正 Age Gap / FaceAge 单变量预测）
# ============================================================================

import matplotlib.pyplot as mpl_plt
from sklearn.metrics import roc_curve, roc_auc_score

print("\n[6/8] 偏差校正后 ROC 分析（原始 vs 校正 Age Gap / FaceAge）...")

# 构建校正后的预测分数字典
y_scores_corrected_dict = {
    'Age Gap (raw)': df['Age_Gap'],
    'Bias-corrected Age Gap': df['Age_Gap_corrected'],
    'FaceAge (raw)': df['生物学年龄'],
    'Bias-corrected FaceAge': df['FaceAge_corrected'],
    'Chronological Age': df['实际年龄'],
    'APACHE II': df['APACHE评分']
}

# 计算 AUC
corrected_results_df = roc_auc_analysis(y_true, y_scores_corrected_dict)

# 保存
corrected_results_df.to_csv(output_dir / "roc_analysis_corrected.csv", index=False, encoding='utf-8-sig')
print(f"  [OK] 保存结果: {output_dir / 'roc_analysis_corrected.csv'}")

# 显示结果
print("\n偏差校正后 ROC 分析结果（Age Gap / FaceAge 原始 vs 校正 对比）:")
print(corrected_results_df.to_string(index=False))
print()

# 绘制校正后 ROC 曲线对比图
# Figure 6 只保留校正论证需要的对比：Age Gap 与 FaceAge 各自的原始 vs 校正版本。
# Chronological Age 与 APACHE II 属 Figure 3 的内容，不在此图重复绘制。
# 如需重画该图，可直接运行 scripts/fig6_bias_corrected_roc.py。
print("  绘制校正后 ROC 曲线对比图（Age Gap 与 FaceAge 原始 vs 校正）...")
plot_roc_curves(
    y_true,
    {k: y_scores_corrected_dict[k] for k in
     ('Age Gap (raw)', 'Bias-corrected Age Gap', 'FaceAge (raw)', 'Bias-corrected FaceAge')},
    title='ROC Curves Before and After Bias Correction\nAge Gap and FaceAge, 7-Day Mortality Prediction (N = 1,470)',
    save_path=output_dir / "roc_curves_corrected.png"
)
print(f"  [OK] 保存图表: {output_dir / 'roc_curves_corrected.png'}")

# ============================================================================
# 7. 累计遮挡分层 ROC 分析
# ============================================================================

print("\n[7/8] 累计遮挡分层 ROC 分析（Age Gap）...")

occlusion_levels_for_strat = [0, 1, 2, 3]
cumulative_results = []

for max_level in occlusion_levels_for_strat:
    # 加载对应遮挡等级的数据
    df_level = load_from_merged_xlsx(max_occlusion_level=max_level)
    level_name = OCCLUSION_LEVELS.get(max_level, f'Level {max_level}')

    n_total = len(df_level)
    n_death = int(df_level['7天死亡'].sum())
    death_rate = n_death / n_total * 100 if n_total > 0 else np.nan

    # 计算 Age Gap 的 AUC（含 bootstrap CI）
    result = calculate_roc_auc(df_level['7天死亡'], df_level['Age_Gap'])
    result['累计遮挡'] = f'≤{max_level} ({level_name})'
    result['样本数'] = n_total
    result['死亡数'] = n_death
    result['死亡率%'] = death_rate
    cumulative_results.append(result)

    sig_mark = format_significance(result['p_value'])
    print(f"  遮挡 <= {max_level} ({level_name:　<4}): "
          f"N={n_total:4d}, 死亡={n_death:3d}, 死亡率={death_rate:5.1f}%, "
          f"AUC={result['AUC']:.4f} [{result['CI_lower']:.4f}, {result['CI_upper']:.4f}], "
          f"p={result['p_value']:.4f} {sig_mark}")

# 构建结果 DataFrame
cumulative_df = pd.DataFrame(cumulative_results)
cumulative_df = cumulative_df[['累计遮挡', '样本数', '死亡数', '死亡率%',
                                'AUC', 'CI_lower', 'CI_upper',
                                'p_value', 'significant_vs_0.5',
                                'n_positive', 'n_negative']]

# 保存
cumulative_df.to_csv(output_dir / "cumulative_roc_analysis.csv", index=False, encoding='utf-8-sig')
print(f"\n  [OK] 保存结果: {output_dir / 'cumulative_roc_analysis.csv'}")

# 格式化显示
print("\n累计遮挡分层 ROC 分析汇总（Age Gap 预测7天死亡率）:")
print("-" * 85)
print(f"{'累计遮挡':<16} {'样本数':>5} {'死亡数':>4} {'死亡率%':>7} "
      f"{'AUC':>7} {'95% CI':>18} {'p_vs_0.5':>9} {'显著性':>4}")
print("-" * 85)
for _, row in cumulative_df.iterrows():
    sig = format_significance(row['p_value'])
    print(f"{row['累计遮挡']:<16} {int(row['样本数']):>5} {int(row['死亡数']):>4} "
          f"{row['死亡率%']:>6.1f}%  {row['AUC']:>6.4f}  "
          f"[{row['CI_lower']:.4f}, {row['CI_upper']:.4f}]  "
          f"{row['p_value']:>8.4f}  {sig:>4}")
print("-" * 85)
print()

# ============================================================================
# 8. 绘制累计遮挡分层 ROC 曲线
# ============================================================================

print("[8/8] 绘制累计遮挡分层 ROC 曲线...")

fig, ax = mpl_plt.subplots(figsize=(8, 8))

# 对角线
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random (AUC = 0.50)')

# 颜色
colors = mpl_plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(occlusion_levels_for_strat)))

for max_level, color in zip(occlusion_levels_for_strat, colors):
    df_level = load_from_merged_xlsx(max_occlusion_level=max_level)
    level_name = OCCLUSION_LEVELS.get(max_level, f'Level {max_level}')

    y_true_l = df_level['7天死亡']
    y_score_l = df_level['Age_Gap']

    mask = ~(np.isnan(y_true_l) | np.isnan(y_score_l))
    y_true_clean = np.array(y_true_l)[mask]
    y_score_clean = np.array(y_score_l)[mask]

    if len(np.unique(y_true_clean)) < 2:
        print(f"  跳过硬 {max_level} ({level_name}): 只有一个类别")
        continue

    fpr, tpr, _ = roc_curve(y_true_clean, y_score_clean)
    auc_val = roc_auc_score(y_true_clean, y_score_clean)

    ax.plot(fpr, tpr, linewidth=2, color=color,
            label=f'≤{max_level} ({level_name})  AUC={auc_val:.4f}, N={len(y_true_clean)}')

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('Age Gap ROC by Cumulative Occlusion Threshold', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.3)

mpl_plt.tight_layout()
mpl_plt.savefig(output_dir / "roc_curves_age_gap_cumulative.png", dpi=300, bbox_inches='tight')
print(f"  [OK] 保存图表: {output_dir / 'roc_curves_age_gap_cumulative.png'}")
mpl_plt.close('all')

print()
print("=" * 80)
print("[OK] Stage 3 分析完成（含累计遮挡分层）！")
print("=" * 80)
print(f"\n结果保存在: {output_dir}")
print()
