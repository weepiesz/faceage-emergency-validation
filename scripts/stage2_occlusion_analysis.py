"""Occlusion grade and FaceAge error: stratified metrics, pairwise tests and typical cases.

Quantifies how much each occlusion grade shifts the age-estimation error, tests the grades
against one another (Kruskal-Wallis across all grades, then Mann-Whitney U pairwise) and pulls
out representative high-error cases for inspection. The pairwise comparisons are Table 4.

Relationship to other scripts
-----------------------------
The Kruskal-Wallis test on occlusion grade and the per-grade metrics are computed again from a
different angle in stage1_performance.py and stage1b_bland_altman.py; and
stage2_age_adjusted_error.py revisits the same comparison with age entered as a covariate.

Input  : data/merged_data.xlsx
Output : results/stage2_occlusion_analysis/performance_by_occlusion.csv
         results/stage2_occlusion_analysis/pairwise_comparisons.csv
         results/stage2_occlusion_analysis/typical_cases.csv
         results/stage2_occlusion_analysis/occlusion_performance_summary.png
         results/stage2_occlusion_analysis/scatter_by_occlusion.png
         results/stage2_occlusion_analysis/error_distribution_by_occlusion.png
         results/stage2_occlusion_analysis/analysis_report.md"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Use default English font (DejaVu Sans / Arial)
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 配置
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "stage2_occlusion_analysis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Stage 2: 遮挡对 FaceAge 预测误差的深度分析")
print("=" * 80)
print(f"结果目录: {RESULTS_DIR}")
print()

# ============================================================================
# 1. 加载数据
# ============================================================================

print("[1/6] 加载数据...")

# 添加utils到路径
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_from_merged_xlsx

# 从 merged_data.xlsx 读取完整数据
df = load_from_merged_xlsx(max_occlusion_level=3)

# 重命名列以匹配脚本期望
df = df.rename(columns={'生物学年龄': '预测年龄', 'image_filename': '文件名'})

print(f"  - 加载 {len(df)} 个样本")
print(f"  - 有效样本数: {len(df)}")

# 计算预测误差
df['绝对误差'] = np.abs(df['预测年龄'] - df['实际年龄'])
df['误差'] = df['预测年龄'] - df['实际年龄']

print(f"  - 遮挡等级分布:")
for level in sorted(df['遮挡等级'].unique()):
    count = (df['遮挡等级'] == level).sum()
    print(f"    Level {level}: {count} ({count/len(df)*100:.1f}%)")
print()

# ============================================================================
# 2. 按遮挡等级分层分析
# ============================================================================

print("[2/6] 按遮挡等级分层分析性能指标...")

occlusion_levels = sorted(df['遮挡等级'].unique())
results = []

for level in occlusion_levels:
    subset = df[df['遮挡等级'] == level]

    # 基本统计
    n_samples = len(subset)

    # 性能指标
    mae = subset['绝对误差'].mean()
    rmse = np.sqrt((subset['误差'] ** 2).mean())
    r2 = r2_score(subset['实际年龄'], subset['预测年龄'])
    pearson_r, pearson_p = stats.pearsonr(subset['实际年龄'], subset['预测年龄'])

    # 误差分布
    error_mean = subset['误差'].mean()
    error_std = subset['误差'].std()
    error_median = subset['误差'].median()

    # 误差范围
    within_5 = (subset['绝对误差'] <= 5).sum()
    within_10 = (subset['绝对误差'] <= 10).sum()

    results.append({
        '遮挡等级': f'Level {level}',
        '样本数': n_samples,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Pearson_r': pearson_r,
        'Pearson_p': pearson_p,
        '误差均值': error_mean,
        '误差标准差': error_std,
        '误差中位数': error_median,
        '误差<=5年': within_5,
        '误差<=5年比例': within_5 / n_samples,
        '误差<=10年': within_10,
        '误差<=10年比例': within_10 / n_samples,
    })

    print(f"  Level {level} ({n_samples} 样本):")
    print(f"    MAE = {mae:.2f} 年")
    print(f"    R2 = {r2:.4f}")
    print(f"    误差 = {error_mean:.2f} +/- {error_std:.2f} 年")
    print(f"    误差<=5年: {within_5}/{n_samples} ({within_5/n_samples*100:.1f}%)")

results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_DIR / "performance_by_occlusion.csv", index=False, encoding='utf-8-sig')
print(f"\n  [OK] 保存结果: performance_by_occlusion.csv")
print()

# ============================================================================
# 3. 统计学检验
# ============================================================================

print("[3/6] 执行统计学检验...")

# 3.1 Kruskal-Wallis H 检验（非参数，检验多组间差异）
groups = [df[df['遮挡等级'] == level]['绝对误差'].values for level in occlusion_levels]
h_stat, h_p = stats.kruskal(*groups)
print(f"  Kruskal-Wallis H 检验:")
print(f"    H = {h_stat:.4f}, p = {h_p:.6f}")
if h_p < 0.05:
    print(f"    [OK] 不同遮挡等级的误差存在显著差异 (p < 0.05)")
else:
    print(f"    [X] 不同遮挡等级的误差无显著差异 (p >= 0.05)")

# 3.2 两两比较（Mann-Whitney U 检验）
print(f"\n  两两比较 (Mann-Whitney U 检验):")
pairwise_results = []

for i, level1 in enumerate(occlusion_levels):
    for level2 in occlusion_levels[i+1:]:
        group1 = df[df['遮挡等级'] == level1]['绝对误差'].values
        group2 = df[df['遮挡等级'] == level2]['绝对误差'].values

        u_stat, u_p = stats.mannwhitneyu(group1, group2, alternative='two-sided')

        pairwise_results.append({
            '比较': f'Level {level1} vs Level {level2}',
            'U统计量': u_stat,
            'p值': u_p,
            '是否显著': '是' if u_p < 0.05 else '否'
        })

        sig_mark = "***" if u_p < 0.001 else ("**" if u_p < 0.01 else ("*" if u_p < 0.05 else "ns"))
        print(f"    Level {level1} vs Level {level2}: U = {u_stat:.1f}, p = {u_p:.6f} {sig_mark}")

pairwise_df = pd.DataFrame(pairwise_results)
pairwise_df.to_csv(RESULTS_DIR / "pairwise_comparisons.csv", index=False, encoding='utf-8-sig')
print(f"\n  [OK] 保存结果: pairwise_comparisons.csv")
print()

# ============================================================================
# 4. 可视化分析
# ============================================================================

print("[4/6] 生成可视化图表...")

# 设置配色
colors = sns.color_palette("Set2", len(occlusion_levels))

# 4.1 误差箱线图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 绝对误差箱线图
ax = axes[0, 0]
box_data = [df[df['遮挡等级'] == level]['绝对误差'].values for level in occlusion_levels]
bp = ax.boxplot(box_data, labels=[f'Level {l}' for l in occlusion_levels],
                patch_artist=True, showmeans=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax.set_ylabel('Absolute Error (years)', fontsize=12)
ax.set_xlabel('Occlusion Level', fontsize=12)
ax.set_title('Absolute Error Distribution by Occlusion Level', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# MAE 柱状图
ax = axes[0, 1]
mae_values = [results_df[results_df['遮挡等级'] == f'Level {l}']['MAE'].values[0] for l in occlusion_levels]
bars = ax.bar([f'Level {l}' for l in occlusion_levels], mae_values, color=colors)
ax.set_ylabel('MAE (years)', fontsize=12)
ax.set_xlabel('Occlusion Level', fontsize=12)
ax.set_title('Mean Absolute Error (MAE) by Occlusion Level', fontsize=14, fontweight='bold')
for bar, val in zip(bars, mae_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{val:.2f}', ha='center', va='bottom', fontsize=10)
ax.grid(axis='y', alpha=0.3)

# R2 柱状图
ax = axes[1, 0]
r2_values = [results_df[results_df['遮挡等级'] == f'Level {l}']['R2'].values[0] for l in occlusion_levels]
bars = ax.bar([f'Level {l}' for l in occlusion_levels], r2_values, color=colors)
ax.set_ylabel('R²', fontsize=12)
ax.set_xlabel('Occlusion Level', fontsize=12)
ax.set_title('Prediction Quality (R²) by Occlusion Level', fontsize=14, fontweight='bold')
ax.set_ylim([0, 1])
for bar, val in zip(bars, r2_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10)
ax.grid(axis='y', alpha=0.3)

# 误差<=5年比例
ax = axes[1, 1]
acc5_values = [results_df[results_df['遮挡等级'] == f'Level {l}']['误差<=5年比例'].values[0] * 100
               for l in occlusion_levels]
bars = ax.bar([f'Level {l}' for l in occlusion_levels], acc5_values, color=colors)
ax.set_ylabel('Proportion (%)', fontsize=12)
ax.set_xlabel('Occlusion Level', fontsize=12)
ax.set_title('Proportion of Predictions within \u00b15 Years by Occlusion Level', fontsize=14, fontweight='bold')
ax.set_ylim([0, 100])
for bar, val in zip(bars, acc5_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "occlusion_performance_summary.png", dpi=300, bbox_inches='tight')
print(f"  [OK] 保存图表: occlusion_performance_summary.png")

# 4.2 散点图：预测年龄 vs 实际年龄（按遮挡等级分组）
plt.figure(figsize=(14, 4))

for idx, level in enumerate(occlusion_levels):
    ax = plt.subplot(1, len(occlusion_levels), idx + 1)
    subset = df[df['遮挡等级'] == level]

    ax.scatter(subset['实际年龄'], subset['预测年龄'],
               alpha=0.5, s=20, color=colors[idx], edgecolors='none')

    # 添加理想线
    min_age = min(subset['实际年龄'].min(), subset['预测年龄'].min())
    max_age = max(subset['实际年龄'].max(), subset['预测年龄'].max())
    ax.plot([min_age, max_age], [min_age, max_age], 'k--', linewidth=1, alpha=0.5)

    # 添加回归线
    z = np.polyfit(subset['实际年龄'], subset['预测年龄'], 1)
    p = np.poly1d(z)
    ax.plot(subset['实际年龄'].sort_values(),
            p(subset['实际年龄'].sort_values()),
            'r-', linewidth=2, alpha=0.7)

    r2 = results_df[results_df['遮挡等级'] == f'Level {level}']['R2'].values[0]
    mae = results_df[results_df['遮挡等级'] == f'Level {level}']['MAE'].values[0]

    ax.set_xlabel('Chronological Age (years)', fontsize=10)
    ax.set_ylabel('Predicted Age (years)', fontsize=10)
    ax.set_title(f'Level {level}\n(n={len(subset)}, MAE={mae:.2f}, R2={r2:.3f})',
                 fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig(RESULTS_DIR / "scatter_by_occlusion.png", dpi=300, bbox_inches='tight')
print(f"  [OK] 保存图表: scatter_by_occlusion.png")

# 4.3 误差分布直方图
fig, axes = plt.subplots(1, len(occlusion_levels), figsize=(14, 4))

for idx, level in enumerate(occlusion_levels):
    ax = axes[idx] if len(occlusion_levels) > 1 else axes
    subset = df[df['遮挡等级'] == level]

    ax.hist(subset['误差'], bins=30, color=colors[idx], alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax.axvline(subset['误差'].mean(), color='blue', linestyle='-', linewidth=2, label='Mean Error')

    ax.set_xlabel('Error (Predicted \u2212 Chronological, years)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(f'Level {level} Error Distribution', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "error_distribution_by_occlusion.png", dpi=300, bbox_inches='tight')
print(f"  [OK] 保存图表: error_distribution_by_occlusion.png")

plt.close('all')
print()

# ============================================================================
# 5. 典型案例分析
# ============================================================================

print("[5/6] 识别典型案例...")

typical_cases = []

for level in occlusion_levels:
    subset = df[df['遮挡等级'] == level].copy()

    if len(subset) == 0:
        continue

    # 最好的预测（误差最小）
    best_idx = subset['绝对误差'].idxmin()
    best = subset.loc[best_idx]

    # 最差的预测（误差最大）
    worst_idx = subset['绝对误差'].idxmax()
    worst = subset.loc[worst_idx]

    # 中位数预测
    median_error = subset['绝对误差'].median()
    median_idx = (subset['绝对误差'] - median_error).abs().idxmin()
    median = subset.loc[median_idx]

    typical_cases.extend([
        {
            '遮挡等级': level,
            '案例类型': '最佳预测',
            '文件名': best.get('文件名', 'N/A'),
            '实际年龄': best['实际年龄'],
            '预测年龄': best['预测年龄'],
            '绝对误差': best['绝对误差']
        },
        {
            '遮挡等级': level,
            '案例类型': '中位预测',
            '文件名': median.get('文件名', 'N/A'),
            '实际年龄': median['实际年龄'],
            '预测年龄': median['预测年龄'],
            '绝对误差': median['绝对误差']
        },
        {
            '遮挡等级': level,
            '案例类型': '最差预测',
            '文件名': worst.get('文件名', 'N/A'),
            '实际年龄': worst['实际年龄'],
            '预测年龄': worst['预测年龄'],
            '绝对误差': worst['绝对误差']
        }
    ])

typical_df = pd.DataFrame(typical_cases)
typical_df.to_csv(RESULTS_DIR / "typical_cases.csv", index=False, encoding='utf-8-sig')
print(f"  [OK] 保存结果: typical_cases.csv")
print(f"  - 每个遮挡等级选取 3 个典型案例（最佳/中位/最差）")
print()

# ============================================================================
# 6. 生成分析报告
# ============================================================================

print("[6/6] 生成分析报告...")

report_lines = [
    "# Stage 2: 遮挡对 FaceAge 预测误差的影响分析报告\n",
    f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
    "---\n",
    "\n## 1. 数据概览\n",
    f"- **总样本数**: {len(df)}\n",
    f"- **遮挡等级分布**:\n"
]

for level in occlusion_levels:
    count = (df['遮挡等级'] == level).sum()
    report_lines.append(f"  - Level {level}: {count} ({count/len(df)*100:.1f}%)\n")

report_lines.extend([
    "\n## 2. 分层性能分析\n",
    "\n### 各遮挡等级性能指标\n",
    "\n| 遮挡等级 | 样本数 | MAE (年) | RMSE (年) | R2 | Pearson r | 误差<=5年比例 |\n",
    "|---------|-------|---------|-----------|-----|-----------|-------------|\n"
])

for _, row in results_df.iterrows():
    report_lines.append(
        f"| {row['遮挡等级']} | {row['样本数']} | {row['MAE']:.2f} | {row['RMSE']:.2f} | "
        f"{row['R2']:.4f} | {row['Pearson_r']:.4f} | {row['误差<=5年比例']*100:.1f}% |\n"
    )

# 计算相对变化
if len(occlusion_levels) >= 2:
    base_mae = results_df[results_df['遮挡等级'] == 'Level 0']['MAE'].values[0]
    report_lines.append("\n### 相对 Level 0 的性能变化\n\n")

    for level in occlusion_levels[1:]:
        level_row = results_df[results_df['遮挡等级'] == f'Level {level}'].iloc[0]
        mae_change = ((level_row['MAE'] - base_mae) / base_mae) * 100
        r2_change = ((level_row['R2'] - results_df[results_df['遮挡等级'] == 'Level 0']['R2'].values[0]) /
                     results_df[results_df['遮挡等级'] == 'Level 0']['R2'].values[0]) * 100

        report_lines.append(f"- **Level {level}**:\n")
        report_lines.append(f"  - MAE 增加 {mae_change:+.1f}%\n")
        report_lines.append(f"  - R2 变化 {r2_change:+.1f}%\n")

report_lines.extend([
    "\n## 3. 统计学检验结果\n",
    f"\n### Kruskal-Wallis H 检验\n",
    f"- H 统计量: {h_stat:.4f}\n",
    f"- p 值: {h_p:.6f}\n",
    f"- 结论: {'不同遮挡等级的预测误差存在显著差异' if h_p < 0.05 else '不同遮挡等级的预测误差无显著差异'}\n",
    "\n### 两两比较 (Mann-Whitney U 检验)\n",
    "\n| 比较 | U 统计量 | p 值 | 显著性 |\n",
    "|------|---------|------|--------|\n"
])

for _, row in pairwise_df.iterrows():
    report_lines.append(
        f"| {row['比较']} | {row['U统计量']:.1f} | {row['p值']:.6f} | {row['是否显著']} |\n"
    )

report_lines.extend([
    "\n## 4. 主要发现\n",
    "\n### 4.1 遮挡对预测误差的影响\n"
])

# 自动生成主要发现
if h_p < 0.05:
    report_lines.append("- [OK] **统计学显著**: 遮挡等级对 FaceAge 预测误差有显著影响\n")
else:
    report_lines.append("- [X] **统计学不显著**: 遮挡等级对 FaceAge 预测误差无显著影响\n")

if len(occlusion_levels) >= 2:
    max_level = occlusion_levels[-1]
    max_mae = results_df[results_df['遮挡等级'] == f'Level {max_level}']['MAE'].values[0]
    min_mae = results_df[results_df['遮挡等级'] == 'Level 0']['MAE'].values[0]
    mae_increase = ((max_mae - min_mae) / min_mae) * 100

    report_lines.append(f"- Level {max_level} 相比 Level 0，MAE 增加 {mae_increase:.1f}%\n")

report_lines.extend([
    "\n### 4.2 临床意义\n",
    "- FaceAge 在 ICU 场景下的预测误差受遮挡程度影响\n",
    "- 遮挡严重的患者需要谨慎解读生物学年龄预测结果\n",
    "- 建议记录面部遮挡情况作为预测质量的参考指标\n",
    "\n## 5. 可视化结果\n",
    "- `occlusion_performance_summary.png`: 各遮挡等级性能对比\n",
    "- `scatter_by_occlusion.png`: 预测 vs 实际年龄散点图\n",
    "- `error_distribution_by_occlusion.png`: 误差分布直方图\n",
    "\n## 6. 数据文件\n",
    "- `performance_by_occlusion.csv`: 分层性能指标\n",
    "- `pairwise_comparisons.csv`: 两两比较统计检验结果\n",
    "- `typical_cases.csv`: 典型案例\n",
    "\n---\n",
    "*本报告由 Stage 2 分析脚本自动生成*\n"
])

report_path = RESULTS_DIR / "analysis_report.md"
with open(report_path, 'w', encoding='utf-8') as f:
    f.writelines(report_lines)

print(f"  [OK] 保存报告: analysis_report.md")
print()

# ============================================================================
# 完成
# ============================================================================

print("=" * 80)
print("[OK] Stage 2 分析完成!")
print("=" * 80)
print(f"\n结果保存在: {RESULTS_DIR}")
print("\n生成文件:")
print("  1. performance_by_occlusion.csv - 分层性能指标")
print("  2. pairwise_comparisons.csv - 统计检验结果")
print("  3. typical_cases.csv - 典型案例")
print("  4. occlusion_performance_summary.png - 性能对比图")
print("  5. scatter_by_occlusion.png - 散点图")
print("  6. error_distribution_by_occlusion.png - 误差分布图")
print("  7. analysis_report.md - 分析报告")
print()
