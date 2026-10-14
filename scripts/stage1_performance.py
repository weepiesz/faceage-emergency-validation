"""FaceAge age-estimation accuracy: overall metrics and stratification by occlusion grade.

Section 2 computes the headline accuracy of FaceAge against chronological age (MAE, RMSE,
R-squared, Pearson r) and the error distribution; section 3 repeats those metrics within each
occlusion grade; section 4 runs the Kruskal-Wallis test across grades. These are the numbers
behind Table 3.

Cohort note
-----------
Section 1 loads every occlusion grade (n = 1,485) so that grade 3 can be described. Every other
analysis in the repository uses the grade <= 2 analysis set (n = 1,470).

Relationship to other scripts
-----------------------------
Only the metrics are live. The three figures written by section 7 are superseded by
figure2_agreement_composite.py, which is the manuscript Figure 2. The per-grade metrics and the
Kruskal-Wallis test are computed again from a different angle in stage1b_bland_altman.py and
stage2_occlusion_analysis.py.

Input  : data/merged_data.xlsx
Output : results/stage1_performance/overall_metrics.csv
         results/stage1_performance/stratified_metrics.csv
         results/figures/stage1_scatter.png
         results/figures/stage1_bland_altman.png
         results/figures/stage1_stratified.png"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 设置工作目录为项目根目录（ClinFace-ICU）
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent  # faceage_emergency_study -> ClinFace-ICU
os.chdir(project_root)

# 添加utils到路径
sys.path.insert(0, str(script_dir))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("阶段一：FaceAge 性能评估")
print("=" * 80)
print(f"工作目录: {os.getcwd()}")
print()

# ============================================================================
# 1. 加载数据
# ============================================================================
print("正在加载数据...")

from utils import load_from_merged_xlsx, RESULTS_DIR

df = load_from_merged_xlsx(max_occlusion_level=3)  # 加载所有遮挡等级

biological_ages = df['生物学年龄'].values.astype(float)
actual_ages = df['实际年龄'].values.astype(float)
occlusion_levels = df['遮挡等级'].values.astype(int)

# 图像文件名（用于调试）
image_filenames = []
if 'image_filename' in df.columns:
    image_filenames = df['image_filename'].tolist()

print(f"[OK] 成功加载 {len(biological_ages)} 个样本")
print(f"  实际年龄范围: [{actual_ages.min():.1f}, {actual_ages.max():.1f}]")
print(f"  预测年龄范围: [{np.nanmin(biological_ages):.1f}, {np.nanmax(biological_ages):.1f}]")
print(f"  遮挡等级分布: {np.bincount(occlusion_levels[occlusion_levels >= 0])}")
print()

# ============================================================================
# 2. 整体性能评估
# ============================================================================
print("=" * 80)
print("整体性能评估")
print("=" * 80)

# 过滤有效数据（去除 NaN）
valid_mask = ~np.isnan(biological_ages)
bio_ages_valid = biological_ages[valid_mask]
actual_ages_valid = actual_ages[valid_mask]

# 计算指标
mae = mean_absolute_error(actual_ages_valid, bio_ages_valid)
rmse = np.sqrt(mean_squared_error(actual_ages_valid, bio_ages_valid))
r2 = r2_score(actual_ages_valid, bio_ages_valid)
pearson_r, pearson_p = stats.pearsonr(actual_ages_valid, bio_ages_valid)

age_diff = bio_ages_valid - actual_ages_valid
mean_diff = age_diff.mean()
std_diff = age_diff.std()

print(f"样本数: {len(bio_ages_valid)}")
print(f"MAE (平均绝对误差): {mae:.2f} 岁")
print(f"RMSE (均方根误差): {rmse:.2f} 岁")
print(f"R2 (决定系数): {r2:.4f}")
print(f"Pearson 相关系数: {pearson_r:.4f} (p={pearson_p:.2e})")
print(f"年龄差异: {mean_diff:.2f} ± {std_diff:.2f} 岁")
print()

# 误差分布
within_5 = np.sum(np.abs(age_diff) <= 5)
within_10 = np.sum(np.abs(age_diff) <= 10)
within_15 = np.sum(np.abs(age_diff) <= 15)

print("误差分布:")
print(f"  ≤5 岁: {within_5} ({within_5/len(bio_ages_valid)*100:.1f}%)")
print(f"  ≤10 岁: {within_10} ({within_10/len(bio_ages_valid)*100:.1f}%)")
print(f"  ≤15 岁: {within_15} ({within_15/len(bio_ages_valid)*100:.1f}%)")
print()

# 保存整体指标
overall_metrics = pd.DataFrame({
    '指标': ['样本数', 'MAE', 'RMSE', 'R2', 'Pearson r', 'Pearson p', '平均差异', '差异标准差',
             'Error ≤ 5 yr, n (%)', 'Error ≤ 10 yr, n (%)'],
    '值': [len(bio_ages_valid), mae, rmse, r2, pearson_r, pearson_p, mean_diff, std_diff,
           f'{within_5} ({within_5/len(bio_ages_valid)*100:.1f}%)',
           f'{within_10} ({within_10/len(bio_ages_valid)*100:.1f}%)']
})

# ============================================================================
# 3. 按遮挡等级分层分析
# ============================================================================
print("=" * 80)
print("按遮挡等级分层分析")
print("=" * 80)

stratified_results = []

for level in sorted(np.unique(occlusion_levels[occlusion_levels >= 0])):
    level_mask = (occlusion_levels == level) & valid_mask

    if level_mask.sum() == 0:
        continue

    bio_ages_level = biological_ages[level_mask]
    actual_ages_level = actual_ages[level_mask]

    mae_level = mean_absolute_error(actual_ages_level, bio_ages_level)
    rmse_level = np.sqrt(mean_squared_error(actual_ages_level, bio_ages_level))
    r2_level = r2_score(actual_ages_level, bio_ages_level)
    pearson_r_level, pearson_p_level = stats.pearsonr(actual_ages_level, bio_ages_level)

    age_diff_level = bio_ages_level - actual_ages_level
    mean_diff_level = age_diff_level.mean()
    std_diff_level = age_diff_level.std()
    abs_diff_level = np.abs(age_diff_level)
    within_5_level = np.sum(abs_diff_level <= 5)
    within_10_level = np.sum(abs_diff_level <= 10)
    n_level = int(level_mask.sum())

    # 死亡人数
    deaths_in_level = int(df.loc[level_mask, '7天死亡'].sum())
    mortality_rate_level = deaths_in_level / n_level * 100

    print(f"\n遮挡等级 {int(level)}:")
    print(f"  样本数: {n_level} (死亡: {deaths_in_level}, {mortality_rate_level:.1f}%)")
    print(f"  MAE: {mae_level:.2f} 岁")
    print(f"  RMSE: {rmse_level:.2f} 岁")
    print(f"  R2: {r2_level:.4f}")
    print(f"  Pearson r: {pearson_r_level:.4f} (p={pearson_p_level:.2e})")
    print(f"  年龄差异: {mean_diff_level:.2f} ± {std_diff_level:.2f} 岁")
    print(f"  ≤5 岁: {within_5_level} ({within_5_level/n_level*100:.1f}%)")
    print(f"  ≤10 岁: {within_10_level} ({within_10_level/n_level*100:.1f}%)")

    stratified_results.append({
        '遮挡等级': int(level),
        '样本数': n_level,
        '死亡人数, n (%)': f'{deaths_in_level} ({mortality_rate_level:.1f}%)',
        'MAE': mae_level,
        'RMSE': rmse_level,
        'R2': r2_level,
        'Pearson_r': pearson_r_level,
        'Pearson_p': pearson_p_level,
        '平均差异': mean_diff_level,
        '差异标准差': std_diff_level,
        'Error ≤ 5 yr, n (%)': f'{within_5_level} ({within_5_level/n_level*100:.1f}%)',
        'Error ≤ 10 yr, n (%)': f'{within_10_level} ({within_10_level/n_level*100:.1f}%)'
    })

stratified_df = pd.DataFrame(stratified_results)
print()

# ============================================================================
# 4. 统计检验（Kruskal-Wallis）
# ============================================================================
print("=" * 80)
print("统计检验：不同遮挡等级的误差差异")
print("=" * 80)

# 准备各组的绝对误差
groups = []
for level in sorted(np.unique(occlusion_levels[occlusion_levels >= 0])):
    level_mask = (occlusion_levels == level) & valid_mask
    if level_mask.sum() > 0:
        bio_ages_level = biological_ages[level_mask]
        actual_ages_level = actual_ages[level_mask]
        abs_errors = np.abs(bio_ages_level - actual_ages_level)
        groups.append(abs_errors)

# Kruskal-Wallis 检验
if len(groups) > 1:
    h_stat, p_value = stats.kruskal(*groups)
    print(f"Kruskal-Wallis H 统计量: {h_stat:.4f}")
    print(f"p 值: {p_value:.4e}")

    if p_value < 0.05:
        print("结论: 不同遮挡等级的预测误差存在显著差异 (p < 0.05)")
    else:
        print("结论: 不同遮挡等级的预测误差无显著差异 (p ≥ 0.05)")
else:
    print("警告: 遮挡等级数量不足，无法进行统计检验")

print()

# ============================================================================
# 5. 创建输出目录
# ============================================================================
results_dir = RESULTS_DIR / "stage1_performance"
figures_dir = RESULTS_DIR / "figures"

results_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 6. 保存结果
# ============================================================================
print("正在保存结果...")

overall_metrics.to_csv(results_dir / "_overall_metrics_temp.csv", index=False, encoding='utf-8-sig')
stratified_df.to_csv(results_dir / "_stratified_metrics_temp.csv", index=False, encoding='utf-8-sig')

# 覆盖原文件
import shutil, time
for src, dst in [
    ("_overall_metrics_temp.csv", "overall_metrics.csv"),
    ("_stratified_metrics_temp.csv", "stratified_metrics.csv")
]:
    for attempt in range(5):
        try:
            shutil.copy(results_dir / src, results_dir / dst)
            break
        except PermissionError:
            time.sleep(1)
    os.remove(results_dir / src)

print(f"[OK] 整体指标已保存: {results_dir / 'overall_metrics.csv'}")
print(f"[OK] 分层指标已保存: {results_dir / 'stratified_metrics.csv'}")
print()

# ============================================================================
# 7. 可视化
# ============================================================================
print("正在生成可视化图表...")

# 图1: 整体散点图
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(actual_ages_valid, bio_ages_valid, alpha=0.5, s=20)
ax.plot([actual_ages_valid.min(), actual_ages_valid.max()],
        [actual_ages_valid.min(), actual_ages_valid.max()],
        'r--', lw=2, label='理想预测线')

ax.set_xlabel('实际年龄 (岁)', fontsize=12)
ax.set_ylabel('FaceAge 预测年龄 (岁)', fontsize=12)
ax.set_title(f'FaceAge 预测性能\nMAE={mae:.2f}, R2={r2:.4f}, Pearson r={pearson_r:.4f}',
             fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / "stage1_scatter.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"[OK] 散点图已保存: {figures_dir / 'stage1_scatter.png'}")

# 图2: Bland-Altman 图
fig, ax = plt.subplots(figsize=(10, 6))

mean_ages = (actual_ages_valid + bio_ages_valid) / 2
diff_ages = bio_ages_valid - actual_ages_valid

ax.scatter(mean_ages, diff_ages, alpha=0.5, s=20)
ax.axhline(mean_diff, color='red', linestyle='-', lw=2, label=f'平均差异: {mean_diff:.2f}')
ax.axhline(mean_diff + 1.96*std_diff, color='red', linestyle='--', lw=1.5,
           label=f'95% 一致性界限: ±{1.96*std_diff:.2f}')
ax.axhline(mean_diff - 1.96*std_diff, color='red', linestyle='--', lw=1.5)
ax.axhline(0, color='gray', linestyle=':', lw=1)

ax.set_xlabel('平均年龄 (岁)', fontsize=12)
ax.set_ylabel('年龄差异 (预测 - 实际, 岁)', fontsize=12)
ax.set_title('Bland-Altman 图：FaceAge 预测一致性分析', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / "stage1_bland_altman.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"[OK] Bland-Altman 图已保存: {figures_dir / 'stage1_bland_altman.png'}")

# 图3: 按遮挡等级分层的散点图
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for i, level in enumerate(sorted(np.unique(occlusion_levels[occlusion_levels >= 0]))):
    if i >= 4:
        break

    level_mask = (occlusion_levels == level) & valid_mask
    bio_ages_level = biological_ages[level_mask]
    actual_ages_level = actual_ages[level_mask]

    # 获取该等级的指标
    level_metrics = stratified_df[stratified_df['遮挡等级'] == level].iloc[0]

    axes[i].scatter(actual_ages_level, bio_ages_level, alpha=0.5, s=20)
    axes[i].plot([actual_ages_level.min(), actual_ages_level.max()],
                 [actual_ages_level.min(), actual_ages_level.max()],
                 'r--', lw=2)

    axes[i].set_xlabel('实际年龄 (岁)', fontsize=11)
    axes[i].set_ylabel('FaceAge 预测年龄 (岁)', fontsize=11)
    axes[i].set_title(f'遮挡等级 {int(level)} (n={int(level_metrics["样本数"])})\n'
                      f'MAE={level_metrics["MAE"]:.2f}, R2={level_metrics["R2"]:.4f}',
                      fontsize=12)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / "stage1_stratified.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"[OK] 分层散点图已保存: {figures_dir / 'stage1_stratified.png'}")
print()

# ============================================================================
# 8. 总结
# ============================================================================
print("=" * 80)
print("阶段一分析完成")
print("=" * 80)
print(f"结果文件:")
print(f"  - {results_dir / 'overall_metrics.csv'}")
print(f"  - {results_dir / 'stratified_metrics.csv'}")
print(f"\n图表文件:")
print(f"  - {figures_dir / 'stage1_scatter.png'}")
print(f"  - {figures_dir / 'stage1_bland_altman.png'}")
print(f"  - {figures_dir / 'stage1_stratified.png'}")
print("=" * 80)
