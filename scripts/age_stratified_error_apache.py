"""Within age group: does Age Gap track illness severity, and do the markers differ by outcome?

Two questions, both asked inside each age group rather than across the cohort, so that the age
dependence of Age Gap cannot manufacture the answer:

  1. the correlation of Age Gap with APACHE II and with the inflammatory markers
     (interleukin-6, procalcitonin, C-reactive protein, white-cell count, neutrophils)
  2. whether the estimation error and the inflammatory markers differ between survivors and
     non-survivors

These are Tables 11 and 12; the script also contributes to Table 7.

Relationship to other scripts
-----------------------------
Shares the four-group age stratification with age_stratified_bias.py but asks a different
question: that script describes Age Gap within each group, this one relates error to severity and
outcome. The inflammatory markers are missing for a large share of patients and are analysed
complete-case, with the missingness stated beside every result.

Input  : data/merged_data.xlsx
Output : results/age_stratified_error/age_stratified_mae.csv
         results/age_stratified_error/error_apache_correlation.csv
         results/age_stratified_error/error_apache_scatter.png
         results/age_stratified_error/error_inflamm_correlation.csv
         results/age_stratified_error/error_inflamm_scatter.png
         results/age_stratified_error/error_by_outcome.csv
         results/age_stratified_error/error_by_outcome_boxplot.png
         results/age_stratified_error/apache_by_outcome.csv
         results/age_stratified_error/apache_by_outcome_boxplot.png
         results/age_stratified_error/inflamm_by_outcome.csv
         results/age_stratified_error/inflamm_by_outcome_boxplot.png
         results/age_stratified_error/analysis_report.md"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 添加utils到路径
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    load_from_merged_xlsx,
    split_by_age_group,
    split_by_outcome,
    pearson_correlation_test,
    mann_whitney_test,
    format_significance,
    RESULTS_DIR,
    COLORS_AGE_GROUPS,
    COLORS_OUTCOME,
    INFLAMMATORY_MARKERS,
    FIGSIZE_GRID_2X2,
    DPI_DEFAULT,
    FONT_CHINESE
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = FONT_CHINESE
plt.rcParams['axes.unicode_minus'] = False

# English label mappings
AGE_GROUP_EN = {
    '青年': 'Young (<45)',
    '中年': 'Middle-aged (45\u201364)',
    '老年': 'Older (65\u201384)',
    '极高龄': 'Oldest-old (\u226585)'
}
AGE_GROUPS_ORDER_EN = ['Young (<45)', 'Middle-aged (45\u201364)', 'Older (65\u201384)', 'Oldest-old (\u226585)']
AGE_GROUPS_ORDER_CN = ['青年', '中年', '老年', '极高龄']  # internal keys from split_by_age_group

print("=" * 80)
print("方向1：年龄分层 + 预测误差与 APACHE/炎症标志物/预后关系的分析")
print("=" * 80)
print()

# ============================================================================
# 1. 加载数据
# ============================================================================

print("[1/9] 加载数据...")
df = load_from_merged_xlsx(max_occlusion_level=2)  # 排除重度遮挡
print(f"  总样本数: {len(df)}")
print(f"  死亡率: {df['7天死亡'].mean()*100:.1f}%")

# 炎症标志物缺失率
print(f"\n  炎症标志物缺失率:")
for marker in INFLAMMATORY_MARKERS:
    missing = df[marker].isna().sum()
    missing_pct = missing / len(df) * 100
    print(f"    {marker}: {missing}/{len(df)} ({missing_pct:.1f}%)")
print()

# 定义所有需要与 Age_Gap 做相关分析的变量
ALL_CORRELATION_VARS = ['APACHE评分'] + INFLAMMATORY_MARKERS
VAR_LABELS = {
    'APACHE评分': 'APACHE-II',
    '白细胞介素-6': 'IL-6',
    'PCT降钙素原': 'PCT',
    'C反应蛋白': 'CRP',
    'WBC': 'WBC',
    '中性粒细胞': 'Neutrophils',
}

# ============================================================================
# 2. 年龄分层
# ============================================================================

print("[2/9] 按年龄分层（4组）...")
groups_dict = split_by_age_group(df, age_groups='4groups')

for group_name, group_df in groups_dict.items():
    print(f"  {group_name}: n={len(group_df)}, 死亡率={group_df['7天死亡'].mean()*100:.1f}%")
print()

# 创建输出目录
output_dir = RESULTS_DIR / "age_stratified_error"
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 3. Age Gap MAE by Age Group (with 95% CI)
# ============================================================================
print("[3/9] Computing Age Gap MAE by age group...")

mae_results = []
from scipy import stats as sp_stats

def compute_ci(values, confidence=0.95):
    """Compute 95% CI for mean using t-distribution (more conservative)."""
    n = len(values)
    se = np.std(values, ddof=1) / np.sqrt(n)
    ci = sp_stats.t.interval(confidence, df=n-1, loc=np.mean(values), scale=se)
    return ci[0], ci[1]

# Overall (all ages)
overall_age_gap = df['Age_Gap'].values
overall_abs = np.abs(overall_age_gap)
n_overall = len(df)
mae_low, mae_high = compute_ci(overall_abs)
me_low, me_high = compute_ci(overall_age_gap)
mae_results.append({
    'Age_Group': 'All Ages',
    'N': n_overall,
    'MAE': np.mean(overall_abs),
    'MAE_95CI_Lower': mae_low,
    'MAE_95CI_Upper': mae_high,
    'RMSE': np.sqrt(np.mean(overall_age_gap ** 2)),
    'ME (Mean Error)': np.mean(overall_age_gap),
    'ME_95CI_Lower': me_low,
    'ME_95CI_Upper': me_high,
    'Std': np.std(overall_age_gap),
})

for group_name in AGE_GROUPS_ORDER_CN:
    if group_name not in groups_dict:
        continue
    group_df = groups_dict[group_name]
    age_gap_vals = group_df['Age_Gap'].values
    abs_vals = np.abs(age_gap_vals)
    n_g = len(group_df)
    en_name = AGE_GROUP_EN.get(group_name, group_name)
    mae_low, mae_high = compute_ci(abs_vals)
    me_low, me_high = compute_ci(age_gap_vals)
    mae_results.append({
        'Age_Group': en_name,
        'N': n_g,
        'MAE': np.mean(abs_vals),
        'MAE_95CI_Lower': mae_low,
        'MAE_95CI_Upper': mae_high,
        'RMSE': np.sqrt(np.mean(age_gap_vals ** 2)),
        'ME (Mean Error)': np.mean(age_gap_vals),
        'ME_95CI_Lower': me_low,
        'ME_95CI_Upper': me_high,
        'Std': np.std(age_gap_vals),
    })
    print(f"  {en_name} (n={n_g}): MAE={mae_results[-1]['MAE']:.2f} "
          f"[95% CI: {mae_low:.2f}, {mae_high:.2f}], "
          f"ME={mae_results[-1]['ME (Mean Error)']:+.2f} "
          f"[95% CI: {me_low:+.2f}, {me_high:+.2f}]")

mae_df = pd.DataFrame(mae_results)
mae_df.to_csv(output_dir / "age_stratified_mae.csv", index=False, encoding='utf-8-sig')
print(f"  [OK] Saved: {output_dir / 'age_stratified_mae.csv'}")
print()

# ============================================================================
# 4. 分析1.1：全年龄段及各年龄层内 Age Gap 与 APACHE 的相关性
# ============================================================================

print("[3/9] 分析1.1：各年龄段内 Age Gap 与 APACHE 的相关性...")

correlation_results = []

# 全年龄段（不分层）
all_result = pearson_correlation_test(df['Age_Gap'], df['APACHE评分'])
correlation_results.append({
    '年龄组': '全年龄段',
    '样本数': all_result['n_samples'],
    'Pearson_r': all_result['correlation'],
    'p值': all_result['p_value'],
    '显著性': '是' if all_result['significant'] else '否',
    'Sig_mark': format_significance(all_result['p_value'])
})
print(f"  全年龄段: r={all_result['correlation']:.4f}, p={all_result['p_value']:.4f} "
      f"({format_significance(all_result['p_value'])})")

for group_name, group_df in groups_dict.items():
    result = pearson_correlation_test(group_df['Age_Gap'], group_df['APACHE评分'])

    correlation_results.append({
        '年龄组': group_name,
        '样本数': result['n_samples'],
        'Pearson_r': result['correlation'],
        'p值': result['p_value'],
        '显著性': '是' if result['significant'] else '否',
        'Sig_mark': format_significance(result['p_value'])
    })

    print(f"  {group_name}: r={result['correlation']:.4f}, p={result['p_value']:.4f} "
          f"({format_significance(result['p_value'])})")

correlation_df = pd.DataFrame(correlation_results)
correlation_df.to_csv(output_dir / "error_apache_correlation.csv", index=False, encoding='utf-8-sig')
print(f"  [OK] 保存结果: {output_dir / 'error_apache_correlation.csv'}")
print()

# ============================================================================
# 4. 可视化1.1：Age Gap vs APACHE 散点图（4组 × 2×2）
# ============================================================================

print("[4/9] 绘制散点图（Age Gap vs APACHE）...")

fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_GRID_2X2)
axes = axes.flatten()

for i, group_name in enumerate(AGE_GROUPS_ORDER_CN):
    ax = axes[i]

    if group_name not in groups_dict:
        ax.text(0.5, 0.5, f'{AGE_GROUP_EN.get(group_name, group_name)}\nNo data', ha='center', va='center',
                fontsize=14, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        continue

    group_df = groups_dict[group_name]

    # Scatter by outcome
    survived = group_df[group_df['7天死亡'] == 0]
    died = group_df[group_df['7天死亡'] == 1]

    ax.scatter(survived['APACHE评分'], survived['Age_Gap'], alpha=0.5, s=30,
               color=COLORS_OUTCOME['survived'], label='Survived', edgecolors='none')
    ax.scatter(died['APACHE评分'], died['Age_Gap'], alpha=0.5, s=30,
               color=COLORS_OUTCOME['died'], label='Died', edgecolors='none')

    # Trend line (overall)
    x = group_df['APACHE评分'].dropna().values
    y = group_df['Age_Gap'].loc[group_df['APACHE评分'].notna()].values

    if len(x) >= 10:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(x.min(), x.max(), 100)

        r = correlation_df[correlation_df['年龄组'] == group_name]['Pearson_r'].values[0]
        p_val = correlation_df[correlation_df['年龄组'] == group_name]['p值'].values[0]
        sig = format_significance(p_val)

        ax.plot(x_trend, p(x_trend), "b--", linewidth=2, alpha=0.7,
                label=f'r={r:.3f} {sig}')

    # Reference line
    ax.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)

    en_name = AGE_GROUP_EN.get(group_name, group_name)
    ax.set_xlabel('APACHE-II Score', fontsize=11)
    ax.set_ylabel('Age Gap (years)', fontsize=11)
    ax.set_title(f'{en_name} (n={len(group_df)})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "error_apache_scatter.png", dpi=DPI_DEFAULT, bbox_inches='tight')
print(f"  [OK] 保存图表: {output_dir / 'error_apache_scatter.png'}")
plt.close()
print()

# ============================================================================
# 5. 分析1.1b：各年龄段内 Age Gap 与炎症标志物的相关性
# ============================================================================

print("[5/9] 分析1.1b：各年龄段内 Age Gap 与炎症标志物的相关性...")

inflamm_correlation_results = []

# 全年龄段（不分层）
print(f"  全年龄段:")
for marker in INFLAMMATORY_MARKERS:
    valid_mask = df[marker].notna() & df['Age_Gap'].notna()
    n_valid = valid_mask.sum()

    if n_valid < 10:
        inflamm_correlation_results.append({
            '年龄组': '全年龄段',
            '指标': marker,
            '有效样本数': n_valid,
            'Pearson_r': np.nan,
            'p值': np.nan,
            '显著性': '样本不足',
            'Sig_mark': 'N/A'
        })
        print(f"    {marker}: 有效样本不足 (n={n_valid})")
        continue

    result = pearson_correlation_test(
        df.loc[valid_mask, 'Age_Gap'],
        df.loc[valid_mask, marker]
    )
    inflamm_correlation_results.append({
        '年龄组': '全年龄段',
        '指标': marker,
        '有效样本数': result['n_samples'],
        'Pearson_r': result['correlation'],
        'p值': result['p_value'],
        '显著性': '是' if result['significant'] else '否',
        'Sig_mark': format_significance(result['p_value'])
    })
    print(f"    {marker}: n={n_valid}, r={result['correlation']:.4f}, "
          f"p={result['p_value']:.4f} ({format_significance(result['p_value'])})")

for group_name, group_df in groups_dict.items():
    for marker in INFLAMMATORY_MARKERS:
        # 去除该 marker 为 NaN 的样本
        valid_mask = group_df[marker].notna() & group_df['Age_Gap'].notna()
        n_valid = valid_mask.sum()

        if n_valid < 10:
            inflamm_correlation_results.append({
                '年龄组': group_name,
                '指标': marker,
                '有效样本数': n_valid,
                'Pearson_r': np.nan,
                'p值': np.nan,
                '显著性': '样本不足',
                'Sig_mark': 'N/A'
            })
            print(f"  {group_name} - {marker}: 有效样本不足 (n={n_valid})")
            continue

        result = pearson_correlation_test(
            group_df.loc[valid_mask, 'Age_Gap'],
            group_df.loc[valid_mask, marker]
        )

        inflamm_correlation_results.append({
            '年龄组': group_name,
            '指标': marker,
            '有效样本数': result['n_samples'],
            'Pearson_r': result['correlation'],
            'p值': result['p_value'],
            '显著性': '是' if result['significant'] else '否',
            'Sig_mark': format_significance(result['p_value'])
        })

        print(f"  {group_name} - {marker}: n={n_valid}, r={result['correlation']:.4f}, "
              f"p={result['p_value']:.4f} ({format_significance(result['p_value'])})")

inflamm_correlation_df = pd.DataFrame(inflamm_correlation_results)

# 按指标优先、年龄组其次排序（便于对比同一指标在不同年龄段的规律）
inflamm_correlation_df['_age_sort'] = inflamm_correlation_df['年龄组'].map(
    lambda x: {'全年龄段': 0, '青年': 1, '中年': 2, '老年': 3, '极高龄': 4}.get(x, 5)
)
inflamm_correlation_df['_marker_sort'] = inflamm_correlation_df['指标'].map(
    lambda x: INFLAMMATORY_MARKERS.index(x) if x in INFLAMMATORY_MARKERS else 99
)
inflamm_correlation_df = inflamm_correlation_df.sort_values(['_marker_sort', '_age_sort']).drop(
    columns=['_age_sort', '_marker_sort']
)

inflamm_correlation_df.to_csv(output_dir / "error_inflamm_correlation.csv", index=False, encoding='utf-8-sig')
print(f"  [OK] 保存结果: {output_dir / 'error_inflamm_correlation.csv'}")
print()

# ============================================================================
# 6. 可视化1.1b：Age Gap vs 炎症标志物散点图
# ============================================================================

print("[6/9] 绘制散点图（Age Gap vs 炎症标志物）...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, marker in enumerate(INFLAMMATORY_MARKERS):
    ax = axes[i]

    # Filter valid data
    valid_data = df[[marker, 'Age_Gap', '实际年龄']].dropna()
    if len(valid_data) < 10:
        ax.text(0.5, 0.5, f'{VAR_LABELS.get(marker, marker)}\nInsufficient samples',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        continue

    # Scatter colored by age group
    age_group_bins = {
        '青年': (0, 45),
        '中年': (45, 65),
        '老年': (65, 85),
        '极高龄': (85, 200)
    }
    age_group_colors = {
        '青年': COLORS_AGE_GROUPS[0],
        '中年': COLORS_AGE_GROUPS[1],
        '老年': COLORS_AGE_GROUPS[2],
        '极高龄': COLORS_AGE_GROUPS[3],
    }

    for ag_key, (lo, hi) in age_group_bins.items():
        mask = (valid_data['实际年龄'] >= lo) & (valid_data['实际年龄'] < hi)
        if mask.sum() > 0:
            en_label = AGE_GROUP_EN.get(ag_key, ag_key)
            ax.scatter(valid_data.loc[mask, marker], valid_data.loc[mask, 'Age_Gap'],
                       alpha=0.4, s=25, color=age_group_colors[ag_key],
                       label=en_label, edgecolors='none')

    # Overall trend line
    x = valid_data[marker].values
    y = valid_data['Age_Gap'].values
    if len(x) >= 10:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(x.min(), x.max(), 100)

        # Overall Pearson r
        r_result = pearson_correlation_test(y, x)
        sig = format_significance(r_result['p_value'])

        ax.plot(x_trend, p(x_trend), "k--", linewidth=2, alpha=0.7,
                label=f'r={r_result["correlation"]:.3f} {sig} (n={len(x)})')

    ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.4)
    ax.set_xlabel(VAR_LABELS.get(marker, marker), fontsize=10)
    ax.set_ylabel('Age Gap (years)', fontsize=10)
    ax.set_title(f'Age Gap vs {VAR_LABELS.get(marker, marker)}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, loc='best')
    ax.grid(alpha=0.3)

# 隐藏第6个子图（如果有的话）
for j in range(len(INFLAMMATORY_MARKERS), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig(output_dir / "error_inflamm_scatter.png", dpi=DPI_DEFAULT, bbox_inches='tight')
print(f"  [OK] 保存图表: {output_dir / 'error_inflamm_scatter.png'}")
plt.close()
print()

# 额外：青年组 炎症标志物 vs Age Gap 单独散点图
print("[6b/9] 绘制青年组炎症标志物 vs Age Gap 单独散点图...")

young_df = groups_dict.get('青年', None)

if young_df is not None and len(young_df) >= 10:
    for marker in INFLAMMATORY_MARKERS:
        valid = young_df[[marker, 'Age_Gap', '7天死亡']].dropna()
        if len(valid) < 10:
            print(f"  [SKIP] 青年组 {VAR_LABELS.get(marker, marker)}: 有效样本不足 (n={len(valid)})")
            continue

        fig, ax = plt.subplots(figsize=(6, 5.5))
        survived = valid[valid['7天死亡'] == 0]
        died = valid[valid['7天死亡'] == 1]

        ax.scatter(survived[marker], survived['Age_Gap'],
                   alpha=0.6, s=40, color=COLORS_OUTCOME['survived'],
                   label='Survived', edgecolors='none')
        ax.scatter(died[marker], died['Age_Gap'],
                   alpha=0.6, s=40, color=COLORS_OUTCOME['died'],
                   label='Died', edgecolors='none')

        r_result = pearson_correlation_test(valid['Age_Gap'].values, valid[marker].values)
        z = np.polyfit(valid[marker].values, valid['Age_Gap'].values, 1)
        p_fit = np.poly1d(z)
        x_trend = np.linspace(valid[marker].min(), valid[marker].max(), 100)
        sig = format_significance(r_result['p_value'])
        ax.plot(x_trend, p_fit(x_trend), 'k--', linewidth=2, alpha=0.7,
                label=f"r={r_result['correlation']:.3f} {sig} (n={r_result['n_samples']})")

        ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.4)
        label = VAR_LABELS.get(marker, marker)
        ax.set_xlabel(f'{label}', fontsize=12)
        ax.set_ylabel('Age Gap (years)', fontsize=12)
        ax.set_title(f'Young Group (<45 yr): Age Gap vs {label} (n={len(valid)})',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(alpha=0.3)

        fname = f"young_{label.lower().replace(' ','_')}_agegap_scatter.png"
        plt.tight_layout()
        plt.savefig(output_dir / fname, dpi=DPI_DEFAULT, bbox_inches='tight')
        print(f"  [OK] Saved: {output_dir / fname}")
        plt.close()
else:
    print("  [SKIP] 青年组数据不足")
print()

# 额外：青年+中年组 炎症标志物 vs Age Gap 合并散点图
print("[6c/9] 绘制青年+中年组炎症标志物 vs Age Gap 合并散点图...")

young_df = groups_dict.get('青年', None)
mid_df = groups_dict.get('中年', None)

if young_df is not None and mid_df is not None:
    combined = pd.concat([young_df, mid_df], ignore_index=True)

    for marker in INFLAMMATORY_MARKERS:
        valid = combined[[marker, 'Age_Gap', '7天死亡', '实际年龄']].dropna()
        if len(valid) < 10:
            print(f"  [SKIP] 青年+中年 {VAR_LABELS.get(marker, marker)}: 有效样本不足 (n={len(valid)})")
            continue

        # --- IL-6 special handling: log-scale x-axis, combined groups, single trend line ---
        if marker == '白细胞介素-6':
            fig, ax = plt.subplots(figsize=(6.5, 5.5))

            valid['IL6_log'] = np.log10(valid[marker])
            surv = valid[valid['7天死亡'] == 0]
            died = valid[valid['7天死亡'] == 1]

            ax.scatter(surv[marker], surv['Age_Gap'],
                       alpha=0.7, s=50, color='#2ecc71',
                       marker='o', label=f'Survived (n={len(surv)})', edgecolors='black', linewidth=0.5)
            ax.scatter(died[marker], died['Age_Gap'],
                       alpha=0.7, s=60, color='#e74c3c',
                       marker='X', label=f'Died (n={len(died)})', edgecolors='black', linewidth=0.5)

            # Single trend line (combined <65 yr)
            r_result = pearson_correlation_test(valid['Age_Gap'].values, valid['IL6_log'].values)
            z = np.polyfit(valid['IL6_log'].values, valid['Age_Gap'].values, 1)
            p_fit = np.poly1d(z)
            x_vals = np.linspace(valid['IL6_log'].min(), valid['IL6_log'].max(), 100)
            sig = format_significance(r_result['p_value'])
            ax.plot(10**x_vals, p_fit(x_vals), '-', color='#2c3e50', linewidth=2.5, alpha=0.9,
                    label=f'r={r_result["correlation"]:.3f} {sig}')

            ax.set_xscale('log')
            ax.set_ylim(-40, 50)
            ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.4)
            label = VAR_LABELS.get(marker, marker)
            ax.set_xlabel(f'{label} (log scale)', fontsize=12)
            ax.set_ylabel('Age Gap (years)', fontsize=12)
            ax.set_title(f'Age <65 yr: Age Gap vs {label} (n={len(valid)})',
                         fontsize=13, fontweight='bold')
            ax.legend(fontsize=9, loc='best')
            ax.grid(alpha=0.3)

            fname = f"young_mid_{label.lower().replace(' ','_')}_agegap_scatter.png"
            plt.tight_layout()
            plt.savefig(output_dir / fname, dpi=DPI_DEFAULT, bbox_inches='tight')
            print(f"  [OK] Saved (log-scale, combined): {output_dir / fname}")
            plt.close()

            # --- Additional plot: linear axis + logarithmic fit curve (combined) ---
            surv_all = valid[valid['7天死亡'] == 0]
            died_all = valid[valid['7天死亡'] == 1]

            fig2, ax2 = plt.subplots(figsize=(6.5, 5.5))

            ax2.scatter(surv_all[marker], surv_all['Age_Gap'],
                        alpha=0.7, s=50, color='#2ecc71',
                        marker='o', label=f'Survived (n={len(surv_all)})', edgecolors='black', linewidth=0.5)
            ax2.scatter(died_all[marker], died_all['Age_Gap'],
                        alpha=0.7, s=60, color='#e74c3c',
                        marker='X', label=f'Died (n={len(died_all)})', edgecolors='black', linewidth=0.5)

            # Logarithmic curve fit: Age Gap = a * ln(IL-6) + b
            log_x = np.log(valid[marker].values)
            z2 = np.polyfit(log_x, valid['Age_Gap'].values, 1)
            p_fit2 = np.poly1d(z2)
            x_curve = np.linspace(valid[marker].min(), valid[marker].max(), 200)
            y_curve = p_fit2(np.log(x_curve))
            r_all = pearson_correlation_test(valid['Age_Gap'].values, valid['IL6_log'].values)
            sig_all = format_significance(r_all['p_value'])
            ax2.plot(x_curve, y_curve, '-', color='#2c3e50', linewidth=2.5, alpha=0.9,
                     label=f'r={r_all["correlation"]:.3f} {sig_all}')

            ax2.set_ylim(-40, 50)
            ax2.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.4)
            ax2.set_xlabel(f'{label}', fontsize=12)
            ax2.set_ylabel('Age Gap (years)', fontsize=12)
            ax2.set_title(f'Age <65 yr: Age Gap vs {label} (log fit, n={len(valid)})',
                          fontsize=13, fontweight='bold')
            ax2.legend(fontsize=8, loc='best')
            ax2.grid(alpha=0.3)

            fname2 = "young_mid_il6_agegap_logfit.png"
            plt.tight_layout()
            plt.savefig(output_dir / fname2, dpi=DPI_DEFAULT, bbox_inches='tight')
            print(f"  [OK] Saved (linear axis + log-fit curve): {output_dir / fname2}")
            plt.close()

            continue

        # --- Other markers: keep original two-group layout ---
        fig, ax = plt.subplots(figsize=(6.5, 5.5))

        young_mask = valid['实际年龄'] < 45
        mid_mask = valid['实际年龄'] >= 45

        # Young group
        y_s = valid[young_mask & (valid['7天死亡'] == 0)]
        y_d = valid[young_mask & (valid['7天死亡'] == 1)]
        ax.scatter(y_s[marker], y_s['Age_Gap'],
                   alpha=0.7, s=50, color=COLORS_AGE_GROUPS[0],
                   marker='o', label='Young - Survived', edgecolors='black', linewidth=0.5)
        ax.scatter(y_d[marker], y_d['Age_Gap'],
                   alpha=0.7, s=60, color=COLORS_AGE_GROUPS[0],
                   marker='X', label='Young - Died', edgecolors='black', linewidth=0.5)

        # Middle-aged group
        m_s = valid[mid_mask & (valid['7天死亡'] == 0)]
        m_d = valid[mid_mask & (valid['7天死亡'] == 1)]
        ax.scatter(m_s[marker], m_s['Age_Gap'],
                   alpha=0.7, s=50, color=COLORS_AGE_GROUPS[1],
                   marker='o', label='Middle-aged - Survived', edgecolors='black', linewidth=0.5)
        ax.scatter(m_d[marker], m_d['Age_Gap'],
                   alpha=0.7, s=60, color=COLORS_AGE_GROUPS[1],
                   marker='X', label='Middle-aged - Died', edgecolors='black', linewidth=0.5)

        # Trend lines per age group
        for ag_mask, ag_label, ag_color in [
            (young_mask, 'Young', COLORS_AGE_GROUPS[0]),
            (mid_mask, 'Middle-aged', COLORS_AGE_GROUPS[1])
        ]:
            sub = valid[ag_mask]
            if len(sub) >= 10:
                r_result = pearson_correlation_test(sub['Age_Gap'].values, sub[marker].values)
                z = np.polyfit(sub[marker].values, sub['Age_Gap'].values, 1)
                p_fit = np.poly1d(z)
                x_vals = np.linspace(sub[marker].min(), sub[marker].max(), 100)
                sig = format_significance(r_result['p_value'])
                ax.plot(x_vals, p_fit(x_vals), '--', color=ag_color, linewidth=2.5, alpha=0.8,
                        label=f"{ag_label}: r={r_result['correlation']:.3f} {sig}")

        ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.4)
        label = VAR_LABELS.get(marker, marker)
        ax.set_xlabel(f'{label}', fontsize=12)
        ax.set_ylabel('Age Gap (years)', fontsize=12)
        ax.set_title(f'Young (<45 yr) & Middle-aged (45\u201364 yr): Age Gap vs {label}',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(alpha=0.3)

        fname = f"young_mid_{label.lower().replace(' ','_')}_agegap_scatter.png"
        plt.tight_layout()
        plt.savefig(output_dir / fname, dpi=DPI_DEFAULT, bbox_inches='tight')
        print(f"  [OK] Saved: {output_dir / fname}")
        plt.close()
else:
    print("  [SKIP] 青年或中年组数据不足")
print()

# ============================================================================
# 7. 分析1.2：各年龄段内，死亡组 vs 生存组的预测误差比较
# ============================================================================

print("[7/9] 分析1.2：各年龄段内死亡组 vs 生存组的 Age Gap...")

error_outcome_results = []

for group_name, group_df in groups_dict.items():
    survived = group_df[group_df['7天死亡'] == 0]
    died = group_df[group_df['7天死亡'] == 1]

    # 计算 Cohen's d 和 pooled SD
    s_vals = survived['Age_Gap'].values
    d_vals = died['Age_Gap'].values
    n1, n2 = len(s_vals), len(d_vals)
    s1, s2 = np.std(s_vals, ddof=1), np.std(d_vals, ddof=1)
    m1, m2 = np.mean(s_vals), np.mean(d_vals)

    if n1 >= 2 and n2 >= 2:
        pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        cohens_d = (m1 - m2) / pooled_sd
        # Hedges' g correction for small sample bias
        g = cohens_d * (1 - 3 / (4 * (n1 + n2) - 9))
        # 95% CI for Cohen's d (approximation)
        se_d = np.sqrt((n1 + n2) / (n1 * n2) + cohens_d**2 / (2 * (n1 + n2)))
        d_ci_low = cohens_d - 1.96 * se_d
        d_ci_high = cohens_d + 1.96 * se_d
    else:
        cohens_d, g, d_ci_low, d_ci_high = np.nan, np.nan, np.nan, np.nan

    # Mann-Whitney U 检验
    if len(died) >= 5:  # 至少5个死亡样本才进行检验
        mw_result = mann_whitney_test(s_vals, d_vals)

        error_outcome_results.append({
            '年龄组': group_name,
            '生存组_n': n1,
            '生存组_Age_Gap_均值': m1,
            '生存组_Age_Gap_标准差': s1,
            '生存组_Age_Gap_中位数': np.median(s_vals),
            '死亡组_n': n2,
            '死亡组_Age_Gap_均值': m2,
            '死亡组_Age_Gap_标准差': s2,
            '死亡组_Age_Gap_中位数': np.median(d_vals),
            "Cohen's_d": cohens_d,
            "Hedges_g": g,
            "d_95CI_lower": d_ci_low,
            "d_95CI_upper": d_ci_high,
            'p值': mw_result['p_value'],
            '显著性': '是' if mw_result['significant'] else '否',
            'Sig_mark': format_significance(mw_result['p_value'])
        })

        print(f"  {group_name}: 生存={m1:.2f}±{s1:.2f}, "
              f"死亡={m2:.2f}±{s2:.2f}, "
              f"d={cohens_d:.3f} [95% CI: {d_ci_low:.3f}, {d_ci_high:.3f}], "
              f"p={mw_result['p_value']:.4f} ({format_significance(mw_result['p_value'])})")
    else:
        print(f"  {group_name}: 死亡样本过少 (n={len(died)})，跳过检验")

error_outcome_df = pd.DataFrame(error_outcome_results)
error_outcome_df.to_csv(output_dir / "error_by_outcome.csv", index=False, encoding='utf-8-sig')
print(f"  [OK] 保存结果: {output_dir / 'error_by_outcome.csv'}")
print()

# ============================================================================
# 8. 可视化1.2：Age Gap by 预后箱线图（4组 × 2类）
# ============================================================================

print("[8/9] 绘制箱线图（Age Gap by 预后 + 炎症指标 by 预后）...")

# 8a. Age Gap by outcome
fig, axes = plt.subplots(1, 4, figsize=(16, 5))

for i, group_name in enumerate(AGE_GROUPS_ORDER_CN):
    ax = axes[i]

    if group_name not in groups_dict:
        en_name = AGE_GROUP_EN.get(group_name, group_name)
        ax.text(0.5, 0.5, f'{en_name}\nNo data', ha='center', va='center',
                fontsize=12, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        continue

    group_df = groups_dict[group_name]
    survived = group_df[group_df['7天死亡'] == 0]
    died = group_df[group_df['7天死亡'] == 1]

    data_to_plot = [survived['Age_Gap'].values, died['Age_Gap'].values]
    bp = ax.boxplot(data_to_plot, tick_labels=['Survived', 'Died'], patch_artist=True,
                     showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=6))

    colors = [COLORS_OUTCOME['survived'], COLORS_OUTCOME['died']]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    if len(died) >= 5:
        p_val = error_outcome_df[error_outcome_df['年龄组'] == group_name]['p值'].values
        if len(p_val) > 0:
            sig = format_significance(p_val[0])
            y_max = max(survived['Age_Gap'].max(), died['Age_Gap'].max())
            y_min = min(survived['Age_Gap'].min(), died['Age_Gap'].min())
            y_range = y_max - y_min
            y_pos = y_max + 0.1 * y_range

            ax.plot([1, 2], [y_pos, y_pos], 'k-', linewidth=1)
            ax.text(1.5, y_pos, f'p={p_val[0]:.3f} {sig}', ha='center', va='bottom', fontsize=9)

    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    en_name = AGE_GROUP_EN.get(group_name, group_name)
    ax.set_ylabel('Age Gap (years)', fontsize=11)
    ax.set_title(f'{en_name} (n={len(group_df)})', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "error_by_outcome_boxplot.png", dpi=DPI_DEFAULT, bbox_inches='tight')
print(f"  [OK] 保存图表: {output_dir / 'error_by_outcome_boxplot.png'}")
plt.close()

# 8b. APACHE by outcome
apache_outcome_results = []

for group_name, group_df in groups_dict.items():
    survived = group_df[group_df['7天死亡'] == 0]
    died = group_df[group_df['7天死亡'] == 1]

    if len(died) >= 5:
        mw_result = mann_whitney_test(
            survived['APACHE评分'].dropna().values,
            died['APACHE评分'].dropna().values
        )

        apache_outcome_results.append({
            '年龄组': group_name,
            '生存组_n': len(survived),
            '生存组_均值': survived['APACHE评分'].mean(),
            '生存组_标准差': survived['APACHE评分'].std(),
            '死亡组_n': len(died),
            '死亡组_均值': died['APACHE评分'].mean(),
            '死亡组_标准差': died['APACHE评分'].std(),
            'p值': mw_result['p_value'],
            '显著性': '是' if mw_result['significant'] else '否',
            'Sig_mark': format_significance(mw_result['p_value'])
        })
    else:
        apache_outcome_results.append({
            '年龄组': group_name,
            '生存组_n': len(survived),
            '生存组_均值': survived['APACHE评分'].mean(),
            '生存组_标准差': survived['APACHE评分'].std(),
            '死亡组_n': len(died),
            '死亡组_均值': np.nan,
            '死亡组_标准差': np.nan,
            'p值': np.nan,
            '显著性': '样本不足',
            'Sig_mark': 'N/A'
        })

apache_outcome_df = pd.DataFrame(apache_outcome_results)

# APACHE by outcome boxplot
fig, axes = plt.subplots(1, 4, figsize=(16, 5))

for i, group_name in enumerate(AGE_GROUPS_ORDER_CN):
    ax = axes[i]

    if group_name not in groups_dict:
        en_name = AGE_GROUP_EN.get(group_name, group_name)
        ax.text(0.5, 0.5, f'{en_name}\nNo data', ha='center', va='center',
                fontsize=12, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        continue

    group_df = groups_dict[group_name]
    survived = group_df[group_df['7天死亡'] == 0]
    died = group_df[group_df['7天死亡'] == 1]

    data_to_plot = [survived['APACHE评分'].dropna().values, died['APACHE评分'].dropna().values]
    bp = ax.boxplot(data_to_plot, tick_labels=['Survived', 'Died'], patch_artist=True,
                     showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=6))

    colors = [COLORS_OUTCOME['survived'], COLORS_OUTCOME['died']]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    if len(died) >= 5:
        p_val = apache_outcome_df[apache_outcome_df['年龄组'] == group_name]['p值'].values
        if len(p_val) > 0 and not pd.isna(p_val[0]):
            sig = format_significance(p_val[0])
            y_max = max(survived['APACHE评分'].max(), died['APACHE评分'].max())
            y_min = min(survived['APACHE评分'].min(), died['APACHE评分'].min())
            y_range = y_max - y_min
            y_pos = y_max + 0.1 * y_range

            ax.plot([1, 2], [y_pos, y_pos], 'k-', linewidth=1)
            ax.text(1.5, y_pos, f'p={p_val[0]:.3f} {sig}', ha='center', va='bottom', fontsize=9)

    en_name = AGE_GROUP_EN.get(group_name, group_name)
    ax.set_ylabel('APACHE-II Score', fontsize=11)
    ax.set_title(f'{en_name} (n={len(group_df)})', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "apache_by_outcome_boxplot.png", dpi=DPI_DEFAULT, bbox_inches='tight')
print(f"  [OK] 保存图表: {output_dir / 'apache_by_outcome_boxplot.png'}")
plt.close()

apache_outcome_df.to_csv(output_dir / "apache_by_outcome.csv", index=False, encoding='utf-8-sig')
print(f"  [OK] 保存结果: {output_dir / 'apache_by_outcome.csv'}")

# 8c. 炎症标志物 by 预后
inflamm_outcome_results = []

# 全年龄段（不分层）
survived_all = df[df['7天死亡'] == 0]
died_all = df[df['7天死亡'] == 1]

for marker in INFLAMMATORY_MARKERS:
    survived_vals = survived_all[marker].dropna().values
    died_vals = died_all[marker].dropna().values

    if len(died_vals) >= 3 and len(survived_vals) >= 10:
        mw_result = mann_whitney_test(survived_vals, died_vals)
        inflamm_outcome_results.append({
            '年龄组': '全年龄段',
            '指标': marker,
            '生存组_n': len(survived_vals),
            '生存组_均值': survived_vals.mean(),
            '生存组_标准差': survived_vals.std(),
            '生存组_中位数': np.median(survived_vals),
            '生存组_Q1': np.percentile(survived_vals, 25),
            '生存组_Q3': np.percentile(survived_vals, 75),
            '死亡组_n': len(died_vals),
            '死亡组_均值': died_vals.mean(),
            '死亡组_标准差': died_vals.std(),
            '死亡组_中位数': np.median(died_vals),
            '死亡组_Q1': np.percentile(died_vals, 25),
            '死亡组_Q3': np.percentile(died_vals, 75),
            'p值': mw_result['p_value'],
            '显著性': '是' if mw_result['significant'] else '否',
            'Sig_mark': format_significance(mw_result['p_value'])
        })
    else:
        inflamm_outcome_results.append({
            '年龄组': '全年龄段',
            '指标': marker,
            '生存组_n': len(survived_vals),
            '生存组_均值': np.nan if len(survived_vals) == 0 else survived_vals.mean(),
            '生存组_标准差': np.nan if len(survived_vals) < 2 else survived_vals.std(),
            '生存组_中位数': np.nan if len(survived_vals) == 0 else np.median(survived_vals),
            '生存组_Q1': np.nan if len(survived_vals) == 0 else np.percentile(survived_vals, 25),
            '生存组_Q3': np.nan if len(survived_vals) == 0 else np.percentile(survived_vals, 75),
            '死亡组_n': len(died_vals),
            '死亡组_均值': np.nan if len(died_vals) == 0 else died_vals.mean(),
            '死亡组_标准差': np.nan if len(died_vals) < 2 else died_vals.std(),
            '死亡组_中位数': np.nan if len(died_vals) == 0 else np.median(died_vals),
            '死亡组_Q1': np.nan if len(died_vals) == 0 else np.percentile(died_vals, 25),
            '死亡组_Q3': np.nan if len(died_vals) == 0 else np.percentile(died_vals, 75),
            'p值': np.nan,
            '显著性': '样本不足',
            'Sig_mark': 'N/A'
        })

# 年龄分层
for group_name, group_df in groups_dict.items():
    survived = group_df[group_df['7天死亡'] == 0]
    died = group_df[group_df['7天死亡'] == 1]

    for marker in INFLAMMATORY_MARKERS:
        survived_vals = survived[marker].dropna().values
        died_vals = died[marker].dropna().values

        if len(died_vals) >= 3 and len(survived_vals) >= 10:
            mw_result = mann_whitney_test(survived_vals, died_vals)

            inflamm_outcome_results.append({
                '年龄组': group_name,
                '指标': marker,
                '生存组_n': len(survived_vals),
                '生存组_均值': survived_vals.mean(),
                '生存组_标准差': survived_vals.std(),
                '生存组_中位数': np.median(survived_vals),
                '生存组_Q1': np.percentile(survived_vals, 25),
                '生存组_Q3': np.percentile(survived_vals, 75),
                '死亡组_n': len(died_vals),
                '死亡组_均值': died_vals.mean(),
                '死亡组_标准差': died_vals.std(),
                '死亡组_中位数': np.median(died_vals),
                '死亡组_Q1': np.percentile(died_vals, 25),
                '死亡组_Q3': np.percentile(died_vals, 75),
                'p值': mw_result['p_value'],
                '显著性': '是' if mw_result['significant'] else '否',
                'Sig_mark': format_significance(mw_result['p_value'])
            })
        else:
            inflamm_outcome_results.append({
                '年龄组': group_name,
                '指标': marker,
                '生存组_n': len(survived_vals),
                '生存组_均值': np.nan if len(survived_vals) == 0 else survived_vals.mean(),
                '生存组_标准差': np.nan if len(survived_vals) < 2 else survived_vals.std(),
                '生存组_中位数': np.nan if len(survived_vals) == 0 else np.median(survived_vals),
                '生存组_Q1': np.nan if len(survived_vals) == 0 else np.percentile(survived_vals, 25),
                '生存组_Q3': np.nan if len(survived_vals) == 0 else np.percentile(survived_vals, 75),
                '死亡组_n': len(died_vals),
                '死亡组_均值': np.nan if len(died_vals) == 0 else died_vals.mean(),
                '死亡组_标准差': np.nan if len(died_vals) < 2 else died_vals.std(),
                '死亡组_中位数': np.nan if len(died_vals) == 0 else np.median(died_vals),
                '死亡组_Q1': np.nan if len(died_vals) == 0 else np.percentile(died_vals, 25),
                '死亡组_Q3': np.nan if len(died_vals) == 0 else np.percentile(died_vals, 75),
                'p值': np.nan,
                '显著性': '样本不足',
                'Sig_mark': 'N/A'
            })

inflamm_outcome_df = pd.DataFrame(inflamm_outcome_results)
inflamm_outcome_df.to_csv(output_dir / "inflamm_by_outcome.csv", index=False, encoding='utf-8-sig')
print(f"  [OK] 保存结果: {output_dir / 'inflamm_by_outcome.csv'}")

# 炎症标志物 by 预后箱线图（按年龄组分面板）
n_markers = len(INFLAMMATORY_MARKERS)
n_age_groups = len(AGE_GROUPS_ORDER_CN)
fig, axes = plt.subplots(n_markers, n_age_groups, figsize=(4 * n_age_groups, 4 * n_markers))

# 处理单行或单列的情况
if n_markers == 1 and n_age_groups == 1:
    axes = np.array([[axes]])
elif n_markers == 1:
    axes = np.array([axes])
elif n_age_groups == 1:
    axes = axes.reshape(-1, 1)

for row_i, marker in enumerate(INFLAMMATORY_MARKERS):
    for col_i, group_name in enumerate(AGE_GROUPS_ORDER_CN):
        ax = axes[row_i, col_i]

        if group_name not in groups_dict:
            ax.text(0.5, 0.5, f'{AGE_GROUP_EN.get(group_name, group_name)}\nNo data', ha='center', va='center',
                    fontsize=10, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        group_df = groups_dict[group_name]
        survived_vals = group_df[group_df['7天死亡'] == 0][marker].dropna().values
        died_vals = group_df[group_df['7天死亡'] == 1][marker].dropna().values

        if len(survived_vals) == 0 and len(died_vals) == 0:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center',
                    fontsize=10, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        data_to_plot = [survived_vals, died_vals]
        bp = ax.boxplot(data_to_plot, tick_labels=['Survived', 'Died'], patch_artist=True,
                         showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=5))

        colors = [COLORS_OUTCOME['survived'], COLORS_OUTCOME['died']]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # 添加p值标注
        p_row = inflamm_outcome_df[
            (inflamm_outcome_df['年龄组'] == group_name) &
            (inflamm_outcome_df['指标'] == marker)
        ]
        if len(p_row) > 0 and not pd.isna(p_row['p值'].values[0]):
            p_val = p_row['p值'].values[0]
            sig = format_significance(p_val)
            all_vals = np.concatenate([survived_vals, died_vals])
            if len(all_vals) > 0:
                y_pos = all_vals.max() + 0.1 * (all_vals.max() - all_vals.min())
                ax.plot([1, 2], [y_pos, y_pos], 'k-', linewidth=1)
                ax.text(1.5, y_pos, f'p={p_val:.3f} {sig}', ha='center', va='bottom', fontsize=7)

        ax.set_ylabel(VAR_LABELS.get(marker, marker), fontsize=9)
        if row_i == 0:
            ax.set_title(AGE_GROUP_EN.get(group_name, group_name), fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "inflamm_by_outcome_boxplot.png", dpi=DPI_DEFAULT, bbox_inches='tight')
print(f"  [OK] 保存图表: {output_dir / 'inflamm_by_outcome_boxplot.png'}")
plt.close()
print()

# ============================================================================
# 9. 生成分析报告
# ============================================================================

print("[9/9] 生成分析报告...")

report_lines = [
    "# 方向1：年龄分层 + 预测误差与 APACHE/炎症标志物/预后关系分析报告\n",
    f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
    "---\n",
    "\n## 1. 研究目标\n",
    "在控制年龄的情况下，探索：\n",
    "1. FaceAge 的 Age Gap 是否与疾病严重程度（APACHE-II）及炎症标志物相关\n",
    "2. 死亡组和生存组的 Age Gap 及炎症标志物是否不同\n",
    "\n## 2. 分析指标\n",
    "- **疾病严重度**: APACHE-II 评分\n",
    "- **炎症标志物**: 白细胞介素-6 (IL-6), PCT降钙素原, C反应蛋白 (CRP), WBC, 中性粒细胞\n",
    "\n## 3. 年龄分组\n",
    "- 青年组：<45岁\n",
    "- 中年组：45-64岁\n",
    "- 老年组：65-84岁\n",
    "- 极高龄组：≥85岁\n",
    "\n---\n",
    "\n## 4. Age Gap 预测误差（MAE）按年龄分层\n",
    "\n| 年龄组 | 样本数 | MAE [95% CI] | ME [95% CI] | RMSE | Std |\n",
    "|--------|--------|-------------|-------------|------|-----|\n"
]
for _, row in mae_df.iterrows():
    report_lines.append(
        f"| {row['Age_Group']} | {row['N']} | "
        f"{row['MAE']:.2f} [{row['MAE_95CI_Lower']:.2f}, {row['MAE_95CI_Upper']:.2f}] | "
        f"{row['ME (Mean Error)']:+.2f} [{row['ME_95CI_Lower']:+.2f}, {row['ME_95CI_Upper']:+.2f}] | "
        f"{row['RMSE']:.2f} | {row['Std']:.2f} |\n"
    )
report_lines.extend([
    "\n### 主要发现\n",
    f"- 全年龄组的 MAE = {mae_df.iloc[0]['MAE']:.2f} 年 "
    f"(95% CI: {mae_df.iloc[0]['MAE_95CI_Lower']:.2f}–{mae_df.iloc[0]['MAE_95CI_Upper']:.2f})\n",
    f"- 极高龄组的 MAE 最低 ({mae_df[mae_df['Age_Group'].str.contains('Oldest')]['MAE'].values[0]:.2f})，"
    f"但低估严重 (ME={mae_df[mae_df['Age_Group'].str.contains('Oldest')]['ME (Mean Error)'].values[0]:+.2f})\n",
    "\n---\n",
    "\n## 5. 分析1.1：全年龄段及各年龄层内 Age Gap 与 APACHE 的相关性\n",
    "\n| 年龄组 | 样本数 | Pearson r | p值 | 显著性 |\n",
    "|--------|--------|-----------|-----|--------|\n"
])

for _, row in correlation_df.iterrows():
    report_lines.append(
        f"| {row['年龄组']} | {row['样本数']} | {row['Pearson_r']:+.4f} | {row['p值']:.4f} | {row['显著性']} |\n"
    )

report_lines.extend(["\n### 主要发现\n"])

sig_count = correlation_df[correlation_df['显著性'] == '是'].shape[0]
if sig_count == 0:
    report_lines.append("- 在所有年龄组内，Age Gap 与 APACHE 评分**均无显著相关**\n")
    report_lines.append("- 结论：控制年龄后，预测误差与疾病严重程度**无关**\n")
else:
    report_lines.append(f"- 在 {sig_count}/{len(correlation_df)} 个年龄组中发现显著相关\n")
    for _, row in correlation_df[correlation_df['显著性'] == '是'].iterrows():
        if row['Pearson_r'] < 0:
            report_lines.append(f"  - **{row['年龄组']}组**: r={row['Pearson_r']:.3f} (负相关) - APACHE 越高，Age Gap 越低\n")
        else:
            report_lines.append(f"  - **{row['年龄组']}组**: r={row['Pearson_r']:.3f} (正相关) - APACHE 越高，Age Gap 越高\n")

# --- 分析1.1b：Age Gap vs 炎症标志物 ---
report_lines.extend([
    "\n---\n",
    "\n## 6. 分析1.1b：全年龄段及各年龄层内 Age Gap 与炎症标志物的相关性\n",
    "\n| 年龄组 | 指标 | 有效样本数 | Pearson r | p值 | 显著性 |\n",
    "|--------|------|-----------|-----------|-----|--------|\n"
])

for _, row in inflamm_correlation_df.iterrows():
    if pd.isna(row['Pearson_r']):
        report_lines.append(
            f"| {row['年龄组']} | {row['指标']} | {row['有效样本数']} | N/A | N/A | {row['显著性']} |\n"
        )
    else:
        report_lines.append(
            f"| {row['年龄组']} | {row['指标']} | {row['有效样本数']} | "
            f"{row['Pearson_r']:+.4f} | {row['p值']:.4f} | {row['显著性']} |\n"
        )

report_lines.extend(["\n### 主要发现\n"])

inflamm_sig = inflamm_correlation_df[inflamm_correlation_df['显著性'] == '是']
if len(inflamm_sig) == 0:
    report_lines.append("- 在所有年龄组 × 指标组合中，Age Gap 与炎症标志物**均无显著相关**\n")
    report_lines.append("- 结论：预测误差与炎症状态**无关**\n")
else:
    report_lines.append(f"- 在 {len(inflamm_sig)}/{len(inflamm_correlation_df)} 个组合中发现显著相关：\n")
    for _, row in inflamm_sig.iterrows():
        direction = '负' if row['Pearson_r'] < 0 else '正'
        report_lines.append(
            f"  - **{row['年龄组']} × {row['指标']}**: r={row['Pearson_r']:.3f} ({direction}相关), "
            f"p={row['p值']:.4f}\n"
        )

# --- 分析1.2：Age Gap by 预后 ---
report_lines.extend([
    "\n---\n",
    "\n## 7. 分析1.2：各年龄段内死亡组 vs 生存组的 Age Gap\n",
    "\n| 年龄组 | 生存组 Age Gap | 死亡组 Age Gap | p值 | 显著性 |\n",
    "|--------|---------------|---------------|-----|--------|\n"
])

for _, row in error_outcome_df.iterrows():
    report_lines.append(
        f"| {row['年龄组']} | {row['生存组_Age_Gap_均值']:.2f}±{row['生存组_Age_Gap_标准差']:.2f} | "
        f"{row['死亡组_Age_Gap_均值']:.2f}±{row['死亡组_Age_Gap_标准差']:.2f} | "
        f"{row['p值']:.4f} | {row['显著性']} |\n"
    )

report_lines.extend(["\n### 主要发现\n"])

sig_count_outcome = error_outcome_df[error_outcome_df['显著性'] == '是'].shape[0]
if sig_count_outcome == 0:
    report_lines.append("- 在所有年龄组内，死亡组和生存组的 Age Gap **均无显著差异**\n")
    report_lines.append("- 结论：控制年龄后，预测误差与预后**无关**\n")
else:
    report_lines.append(f"- 在 {sig_count_outcome}/{len(error_outcome_df)} 个年龄组中发现显著差异\n")
    for _, row in error_outcome_df[error_outcome_df['显著性'] == '是'].iterrows():
        diff = row['死亡组_Age_Gap_均值'] - row['生存组_Age_Gap_均值']
        if diff > 0:
            report_lines.append(f"  - **{row['年龄组']}组**: 死亡组 Age Gap 更高 (差异={diff:.2f}年) - 死亡组\"看起来更老\"\n")
        else:
            report_lines.append(f"  - **{row['年龄组']}组**: 死亡组 Age Gap 更低 (差异={diff:.2f}年) - 死亡组\"看起来更年轻\"\n")

# --- 分析1.3：APACHE by 预后 ---
report_lines.extend([
    "\n---\n",
    "\n## 8. 分析1.3：各年龄段内死亡组 vs 生存组的 APACHE 评分\n",
    "\n| 年龄组 | 生存组 APACHE | 死亡组 APACHE | p值 | 显著性 |\n",
    "|--------|--------------|--------------|-----|--------|\n"
])

for _, row in apache_outcome_df.iterrows():
    if pd.isna(row['死亡组_均值']):
        report_lines.append(
            f"| {row['年龄组']} | {row['生存组_均值']:.1f}±{row['生存组_标准差']:.1f} | N/A | N/A | {row['显著性']} |\n"
        )
    else:
        report_lines.append(
            f"| {row['年龄组']} | {row['生存组_均值']:.1f}±{row['生存组_标准差']:.1f} | "
            f"{row['死亡组_均值']:.1f}±{row['死亡组_标准差']:.1f} | "
            f"{row['p值']:.4f} | {row['显著性']} |\n"
        )

report_lines.extend(["\n### 主要发现\n"])

all_sig = apache_outcome_df[apache_outcome_df['显著性'] == '是']
if len(all_sig) == len(apache_outcome_df):
    report_lines.append("- **所有年龄组**：死亡组 APACHE 评分显著高于生存组\n")
elif len(all_sig) > 0:
    report_lines.append(f"- {len(all_sig)}/{len(apache_outcome_df)} 个年龄组：死亡组 APACHE 评分显著高于生存组\n")
else:
    report_lines.append("- 各年龄组内 APACHE 评分与预后关系不显著\n")

# --- 分析1.3b：炎症标志物 by 预后 ---
report_lines.extend([
    "\n---\n",
    "\n## 9. 分析1.3b：各年龄段内死亡组 vs 生存组的炎症标志物\n",
    "\n| 年龄组 | 指标 | 生存组 | 死亡组 | p值 | 显著性 |\n",
    "|--------|------|--------|--------|-----|--------|\n"
])

for _, row in inflamm_outcome_df.iterrows():
    if pd.isna(row['p值']):
        report_lines.append(
            f"| {row['年龄组']} | {row['指标']} | {row['生存组_均值']:.2f}±{row['生存组_标准差']:.2f} | "
            f"N/A | N/A | {row['显著性']} |\n"
        )
    else:
        report_lines.append(
            f"| {row['年龄组']} | {row['指标']} | "
            f"{row['生存组_均值']:.2f}±{row['生存组_标准差']:.2f} | "
            f"{row['死亡组_均值']:.2f}±{row['死亡组_标准差']:.2f} | "
            f"{row['p值']:.4f} | {row['显著性']} |\n"
        )

report_lines.extend(["\n### 主要发现\n"])

inflamm_outcome_sig = inflamm_outcome_df[inflamm_outcome_df['显著性'] == '是']
if len(inflamm_outcome_sig) == 0:
    report_lines.append("- 在所有年龄组 × 指标组合中，死亡组与生存组的炎症标志物**均无显著差异**\n")
else:
    report_lines.append(f"- 在 {len(inflamm_outcome_sig)}/{len(inflamm_outcome_df)} 个组合中发现显著差异\n")
    for _, row in inflamm_outcome_sig.iterrows():
        diff = row['死亡组_均值'] - row['生存组_均值']
        direction = '更高' if diff > 0 else '更低'
        report_lines.append(
            f"  - **{row['年龄组']} × {row['指标']}**: 死亡组 {direction} "
            f"(差异={diff:+.2f}, p={row['p值']:.4f})\n"
        )

# --- 综合结论 ---
report_lines.extend([
    "\n---\n",
    "\n## 10. 综合结论\n",
    "\n### 10.1 核心发现\n"
])

if sig_count == 0 and sig_count_outcome == 0:
    report_lines.extend([
        "1. **Age Gap 与 APACHE 无关**：在控制年龄后，预测误差与疾病严重程度无显著相关\n",
        "2. **Age Gap 与预后无关**：在控制年龄后，死亡组和生存组的预测误差无显著差异\n",
    ])
else:
    report_lines.extend([
        "1. **在部分年龄组发现显著关系**\n",
        "2. **需要进一步探索机制**\n"
    ])

if len(inflamm_sig) == 0:
    report_lines.append("3. **Age Gap 与炎症标志物无关**：预测误差与炎症状态无显著相关\n")
else:
    report_lines.append("3. **在部分年龄组发现 Age Gap 与炎症标志物相关**，需进一步验证生物学机制\n")

report_lines.extend([
    "\n### 10.2 临床意义\n",
    "- 炎症标志物的加入为理解 Age Gap 的生物学意义提供了新的证据维度\n",
    "- 尽管 Age Gap 与各临床指标的关系有限，APACHE 评分仍是预后预测的核心指标\n",
    "\n## 11. 数据文件\n",
    "- `age_stratified_mae.csv`: Age Gap MAE by age group\n",
    "- `error_apache_correlation.csv`: Age Gap-APACHE 相关性\n",
    "- `error_inflamm_correlation.csv`: Age Gap-炎症标志物相关性\n",
    "- `error_by_outcome.csv`: Age Gap by 预后比较\n",
    "- `apache_by_outcome.csv`: APACHE by 预后比较\n",
    "- `inflamm_by_outcome.csv`: 炎症标志物 by 预后比较\n",
    "\n## 12. 可视化结果\n",
    "- `error_apache_scatter.png`: Age Gap vs APACHE 散点图（4组）\n",
    "- `error_inflamm_scatter.png`: Age Gap vs 炎症标志物散点图（5指标）\n",
    "- `error_by_outcome_boxplot.png`: Age Gap by 预后箱线图（4组）\n",
    "- `apache_by_outcome_boxplot.png`: APACHE by 预后箱线图（4组）\n",
    "- `inflamm_by_outcome_boxplot.png`: 炎症标志物 by 预后箱线图（5指标 × 4组）\n",
    "\n---\n",
    "*本报告由分析脚本自动生成*\n"
])

report_path = output_dir / "analysis_report.md"
with open(report_path, 'w', encoding='utf-8') as f:
    f.writelines(report_lines)

print(f"  [OK] 保存报告: {report_path}")
print()

# ============================================================================
# 完成
# ============================================================================

print("=" * 80)
print("[OK] 方向1分析完成！")
print("=" * 80)
print(f"\n结果保存在: {output_dir}")
print("\n数据文件:")
print("  1. error_apache_correlation.csv  - Age Gap vs APACHE 相关性")
print("  2. error_inflamm_correlation.csv - Age Gap vs 炎症标志物相关性")
print("  3. error_by_outcome.csv          - Age Gap by 预后")
print("  4. apache_by_outcome.csv         - APACHE by 预后")
print("  5. inflamm_by_outcome.csv        - 炎症标志物 by 预后")
print("\n可视化:")
print("  6. error_apache_scatter.png       - Age Gap vs APACHE 散点图")
print("  7. error_inflamm_scatter.png      - Age Gap vs 炎症标志物 散点图")
print("  8. error_by_outcome_boxplot.png   - Age Gap by 预后箱线图")
print("  9. apache_by_outcome_boxplot.png  - APACHE by 预后箱线图")
print("  10. inflamm_by_outcome_boxplot.png - 炎症标志物 by 预后箱线图")
print("\n报告:")
print("  11. analysis_report.md")
print()
