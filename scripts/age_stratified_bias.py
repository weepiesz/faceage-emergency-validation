"""Systematic FaceAge bias across the four age groups.

Age-stratified Age Gap: the mean Age Gap in each group (< 45, 45-64, 65-84, >= 85), the
Kruskal-Wallis test across groups, the pairwise comparisons, and the boxplot and scatterplot used
in the manuscript. The monotone fall of Age Gap from the youngest group to the oldest is the
mechanism behind the inverse association of raw Age Gap with mortality. These are the numbers in
Tables 8-10 and Figure 4.

Input  : data/merged_data.xlsx   (analysis set, occlusion grade <= 2, n = 1,470)
Output : results/age_stratified_bias/age_gap_by_group.csv
         results/age_stratified_bias/pairwise_comparisons.csv
         results/age_stratified_bias/age_gap_boxplot.png
         results/age_stratified_bias/age_gap_scatter.png
         results/age_stratified_bias/analysis_report.md"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 添加utils到路径
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    load_from_merged_xlsx,
    split_by_age_group,
    kruskal_wallis_test,
    pairwise_comparisons,
    plot_boxplot_by_group,
    plot_scatter_with_trend,
    RESULTS_DIR,
    COLORS_AGE_GROUPS
)

print("=" * 80)
print("前置分析：年龄分层的系统性偏差分析（4个年龄组）")
print("=" * 80)
print()

# ============================================================================
# 1. 加载数据
# ============================================================================

print("[1/5] 加载数据...")
df = load_from_merged_xlsx(max_occlusion_level=2)  # 排除重度遮挡(Level 3)，分析集N=1470
print(f"  总样本数: {len(df)}")
print()

# ============================================================================
# 2. 年龄分层
# ============================================================================

print("[2/5] 按年龄分层（4组）...")
groups_dict = split_by_age_group(df, age_groups='4groups')

# 统计各组数据
results = []
for group_name, group_df in groups_dict.items():
    results.append({
        '年龄组': group_name,
        '样本数': len(group_df),
        '样本比例(%)': len(group_df) / len(df) * 100,
        '实际年龄_均值': group_df['实际年龄'].mean(),
        '实际年龄_标准差': group_df['实际年龄'].std(),
        '生物学年龄_均值': group_df['生物学年龄'].mean(),
        '生物学年龄_标准差': group_df['生物学年龄'].std(),
        'Age_Gap_均值': group_df['Age_Gap'].mean(),
        'Age_Gap_标准差': group_df['Age_Gap'].std(),
        'Age_Gap_中位数': group_df['Age_Gap'].median(),
        'Age_Gap_Q1': group_df['Age_Gap'].quantile(0.25),
        'Age_Gap_Q3': group_df['Age_Gap'].quantile(0.75)
    })

results_df = pd.DataFrame(results)

# 显示结果
print("\n各年龄组的 Age Gap 统计:")
print(results_df.to_string(index=False))
print()

# 保存结果
output_dir = RESULTS_DIR / "age_stratified_bias"
output_dir.mkdir(parents=True, exist_ok=True)

results_df.to_csv(output_dir / "age_gap_by_group.csv", index=False, encoding='utf-8-sig')
print(f"  [OK] 保存结果: {output_dir / 'age_gap_by_group.csv'}")
print()

# ============================================================================
# 3. 统计检验
# ============================================================================

print("[3/5] 统计检验（Kruskal-Wallis H）...")

age_gap_groups = [group_df['Age_Gap'].values for group_df in groups_dict.values()]
kw_result = kruskal_wallis_test(*age_gap_groups)

print(f"  Kruskal-Wallis H 检验:")
print(f"    H = {kw_result['statistic']:.4f}")
print(f"    p = {kw_result['p_value']:.6f}")
print(f"    显著性: {'是' if kw_result['significant'] else '否'} (α = {kw_result['alpha']})")
print()

# 两两比较
if kw_result['significant']:
    print("  两两比较（Mann-Whitney U）...")
    age_gap_dict = {name: df['Age_Gap'].values for name, df in groups_dict.items()}
    pairwise_results = pairwise_comparisons(age_gap_dict, test='mann_whitney')
    print(pairwise_results.to_string(index=False))
    pairwise_results.to_csv(output_dir / "pairwise_comparisons.csv", index=False, encoding='utf-8-sig')
    print(f"  [OK] 保存两两比较结果: {output_dir / 'pairwise_comparisons.csv'}")
print()

# ============================================================================
# 4. 可视化 - 箱线图
# ============================================================================

print("[4/5] 绘制箱线图...")

# 为df添加年龄组列（用于绘图）
def assign_age_group(age):
    if age < 45:
        return 'Young (<45)'
    elif age < 65:
        return 'Middle-aged (45\u201364)'
    elif age < 85:
        return 'Older (65\u201384)'
    else:
        return 'Oldest-old (\u226585)'

df['年龄组'] = df['实际年龄'].apply(assign_age_group)

AGE_GROUP_ORDER = [
    'Young (<45)',
    'Middle-aged (45\u201364)',
    'Older (65\u201384)',
    'Oldest-old (\u226585)'
]

# 绘制箱线图
plot_boxplot_by_group(
    df,
    group_col='年龄组',
    value_col='Age_Gap',
    title='Age Gap Distribution by Age Group',
    xlabel='Age Group',
    ylabel='Age Gap (FaceAge \u2212 Chronological Age, years)',
    colors=COLORS_AGE_GROUPS,
    group_order=AGE_GROUP_ORDER,
    save_path=output_dir / "age_gap_boxplot.png"
)

print()

# ============================================================================
# 5. 可视化 - 散点图
# ============================================================================

print("[5/5] 绘制散点图...")

plot_scatter_with_trend(
    x=df['实际年龄'],
    y=df['Age_Gap'],
    title='Age Gap vs Chronological Age',
    xlabel='Chronological Age (years)',
    ylabel='Age Gap (FaceAge \u2212 Chronological Age, years)',
    show_correlation=True,
    save_path=output_dir / "age_gap_scatter.png"
)

print()

# ============================================================================
# 6. 生成分析报告
# ============================================================================

print("生成分析报告...")

report_lines = [
    "# 年龄分层的系统性偏差分析报告\n",
    f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
    "---\n",
    "\n## 1. 研究目标\n",
    "评估 FaceAge 在不同年龄段（4组）的系统性偏差\n",
    "\n## 2. 年龄分组\n",
    "- 青年组：<45岁\n",
    "- 中年组：45-64岁\n",
    "- 老年组：65-84岁\n",
    "- 极高龄组：≥85岁\n",
    "\n## 3. 各年龄组 Age Gap 统计\n",
    "\n| 年龄组 | 样本数 | 比例(%) | Age Gap均值 | Age Gap标准差 | Age Gap中位数 |\n",
    "|-------|--------|---------|------------|--------------|-------------|\n"
]

for _, row in results_df.iterrows():
    report_lines.append(
        f"| {row['年龄组']} | {row['样本数']} | {row['样本比例(%)']:.1f} | "
        f"{row['Age_Gap_均值']:.2f} | {row['Age_Gap_标准差']:.2f} | {row['Age_Gap_中位数']:.2f} |\n"
    )

report_lines.extend([
    f"\n## 4. Kruskal-Wallis H 检验\n",
    f"- H统计量 = {kw_result['statistic']:.4f}\n",
    f"- p值 = {kw_result['p_value']:.6f}\n",
    f"- 结论: {'不同年龄组的 Age Gap 存在显著差异' if kw_result['significant'] else '不同年龄组的 Age Gap 无显著差异'}\n",
    "\n## 5. 主要发现\n",
    "\n### 5.1 系统性偏差模式\n"
])

# 分析偏差模式
for _, row in results_df.iterrows():
    if row['Age_Gap_均值'] > 2:
        report_lines.append(f"- **{row['年龄组']}组**: Age Gap = +{row['Age_Gap_均值']:.2f} 年（**系统性高估**）\n")
    elif row['Age_Gap_均值'] < -2:
        report_lines.append(f"- **{row['年龄组']}组**: Age Gap = {row['Age_Gap_均值']:.2f} 年（**系统性低估**）\n")
    else:
        report_lines.append(f"- **{row['年龄组']}组**: Age Gap = {row['Age_Gap_均值']:.2f} 年（接近实际）\n")

report_lines.extend([
    "\n### 5.2 临床意义\n",
    "- FaceAge 模型对年轻患者倾向于**高估**年龄\n",
    "- 对老年和极高龄患者倾向于**低估**年龄\n",
    "- 这种系统性偏差可能影响 Age Gap 的临床解读\n",
    "\n## 6. 可视化结果\n",
    "- `age_gap_boxplot.png`: 各年龄组 Age Gap 箱线图\n",
    "- `age_gap_scatter.png`: Age Gap vs 实际年龄散点图\n",
    "\n## 7. 数据文件\n",
    "- `age_gap_by_group.csv`: 各组统计数据\n",
    "- `pairwise_comparisons.csv`: 两两比较结果（如有）\n",
    "\n---\n",
    "*本报告由分析脚本自动生成*\n"
])

report_path = output_dir / "analysis_report.md"
with open(report_path, 'w', encoding='utf-8') as f:
    f.writelines(report_lines)

print(f"  [OK] 保存报告: {report_path}")
print()

print("=" * 80)
print("[OK] 年龄分层偏差分析完成！")
print("=" * 80)
print(f"\n结果保存在: {output_dir}")
print()
