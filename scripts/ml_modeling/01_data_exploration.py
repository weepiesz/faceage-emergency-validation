"""Missing-value survey of the bedside indicators.

The source of the missingness figures quoted in the Methods. Surveys how much of each candidate
bedside field is missing in the merged workbook — including the GCS score, which is not one of
the eight indicators used in the final models — and profiles the feature distributions and their
correlations, to fix the pre-processing strategy for the machine-learning pipeline.

Input  : data/merged_data.xlsx   (all occlusion grades, including grade 3)
Output : results/ml_modeling/missing_values_analysis.csv
         results/ml_modeling/feature_distributions.png
         results/ml_modeling/feature_correlation_heatmap.png
         results/ml_modeling/data_exploration_report.md"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# 添加utils到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    CLINICAL_DATASET_DIR,
    RESULTS_DIR,
    FONT_CHINESE,
    DPI_DEFAULT
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = FONT_CHINESE
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("方向2 - 机器学习建模：数据探索")
print("=" * 80)
print()

# 创建输出目录
output_dir = RESULTS_DIR / "ml_modeling"
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. 加载数据并查看所有列名
# ============================================================================

print("[1/6] 加载数据并查看列名...")

from utils import load_from_merged_xlsx

df_all = load_from_merged_xlsx(max_occlusion_level=3)
# 仅保留数值列用于探索（排除元数据列）
exclude_cols = ['image_filename', 'group_name', 'excel_file', 'error', 'patient_id']
numeric_cols = [c for c in df_all.columns if c not in exclude_cols and df_all[c].dtype in ('int64', 'float64')]

print(f"  总样本数: {len(df_all)}")
print(f"  数值特征数: {len(numeric_cols)}")
print()

print("所有可用的数值列名:")
for i, col in enumerate(numeric_cols):
    print(f"  {i:2d}. {col}")
print()

# ============================================================================
# 2. 识别快速评估指标的列名
# ============================================================================

print("[2/6] 识别快速评估指标...")

quick_features = ['性别','体温', '心率', '呼吸频率', '收缩压', '舒张压', '血氧饱和度', 'GCS评分', '血糖']
# xlsx 列名与特征名一致，直接映射
feature_mapping = {name: name for name in quick_features}

found = [c for c in quick_features if c in df_all.columns]
missing = [c for c in quick_features if c not in df_all.columns]
for f in found:
    print(f"  {f} -> [OK]")
for f in missing:
    print(f"  {f} -> [WARN] not found")
print()

if missing:
    print(f"[WARN] Warning: features not found: {', '.join(missing)}")

# ============================================================================
# 3. 创建分析用DataFrame（直接从xlsx选取列）
# ============================================================================

print("[3/6] 创建DataFrame...")

# merged_data.xlsx 中的列已完全匹配，直接选取
analysis_cols = found + ['实际年龄', '生物学年龄', 'Age_Gap', '7天死亡', 'APACHE评分', '遮挡等级']
df = df_all[analysis_cols].copy()

print(f"  DataFrame形状: {df.shape}")
print(f"  列名: {list(df.columns)}")
print()

# 过滤：排除重度遮挡 + 生物学年龄有效
valid_mask = (df['遮挡等级'] <= 2) & (~df['生物学年龄'].isna())
df_filtered = df[valid_mask].copy()

print(f"  过滤后样本数: {len(df_filtered)} (排除重度遮挡和无效生物学年龄)")
print(f"  死亡率: {df_filtered['7天死亡'].mean()*100:.1f}%")
print()

# ============================================================================
# 4. 缺失值分析
# ============================================================================

print("[4/6] 缺失值分析...")

# 仅分析快速评估指标
quick_features = list(feature_mapping.keys())

missing_stats = []
for feature in quick_features:
    if feature in df_filtered.columns:
        total = len(df_filtered)
        missing = df_filtered[feature].isna().sum()
        missing_pct = missing / total * 100

        missing_stats.append({
            '特征': feature,
            '缺失数': missing,
            '缺失率(%)': missing_pct,
            '有效样本数': total - missing,
            '均值': df_filtered[feature].mean() if missing < total else np.nan,
            '标准差': df_filtered[feature].std() if missing < total else np.nan,
            '最小值': df_filtered[feature].min() if missing < total else np.nan,
            '最大值': df_filtered[feature].max() if missing < total else np.nan
        })

        status = ""
        if missing_pct < 5:
            status = "[OK] 优秀"
        elif missing_pct < 20:
            status = "[!] 可接受"
        else:
            status = "[X] 不可用"

        print(f"  {feature:12s}: 缺失 {missing:4d} ({missing_pct:5.1f}%) {status}")

missing_df = pd.DataFrame(missing_stats)
missing_df.to_csv(output_dir / "missing_values_analysis.csv", index=False, encoding='utf-8-sig')
print(f"\n  [OK] 保存结果: {output_dir / 'missing_values_analysis.csv'}")
print()

# 数据预处理策略建议
high_missing = missing_df[missing_df['缺失率(%)'] > 20]['特征'].tolist()
medium_missing = missing_df[(missing_df['缺失率(%)'] >= 5) & (missing_df['缺失率(%)'] <= 20)]['特征'].tolist()
low_missing = missing_df[missing_df['缺失率(%)'] < 5]['特征'].tolist()

print("数据预处理策略建议:")
if low_missing:
    print(f"  [OK] 缺失率<5%（删除缺失样本）: {', '.join(low_missing)}")
if medium_missing:
    print(f"  [!] 缺失率5-20%（中位数填充）: {', '.join(medium_missing)}")
if high_missing:
    print(f"  [X] 缺失率>20%（考虑删除特征）: {', '.join(high_missing)}")
print()

# ============================================================================
# 5. 特征分布可视化
# ============================================================================

print("[5/6] 绘制特征分布图...")

# 计算需要的子图数量
n_features = len(quick_features)
n_cols = 3
n_rows = int(np.ceil(n_features / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
axes = axes.flatten() if n_features > 1 else [axes]

for i, feature in enumerate(quick_features):
    ax = axes[i]

    if feature not in df_filtered.columns:
        ax.text(0.5, 0.5, f'{feature}\n未找到', ha='center', va='center', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        continue

    data = df_filtered[feature].dropna()

    if len(data) == 0:
        ax.text(0.5, 0.5, f'{feature}\n无数据', ha='center', va='center', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        continue

    # 绘制直方图
    ax.hist(data, bins=30, alpha=0.7, color='steelblue', edgecolor='black')

    # 添加均值和中位数线
    mean_val = data.mean()
    median_val = data.median()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'均值={mean_val:.1f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'中位数={median_val:.1f}')

    ax.set_xlabel('值', fontsize=10)
    ax.set_ylabel('频数', fontsize=10)
    ax.set_title(f'{feature}\n(n={len(data)}, 缺失={df_filtered[feature].isna().sum()})',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

# 隐藏多余的子图
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig(output_dir / "feature_distributions.png", dpi=DPI_DEFAULT, bbox_inches='tight')
print(f"  [OK] 保存图表: {output_dir / 'feature_distributions.png'}")
plt.close()
print()

# ============================================================================
# 6. 特征相关性分析
# ============================================================================

print("[6/6] 绘制相关性热图...")

# 选择有效特征（缺失率<20%）
valid_features = missing_df[missing_df['缺失率(%)'] < 20]['特征'].tolist()
valid_features_with_target = valid_features + ['实际年龄', '生物学年龄', 'Age_Gap', 'APACHE评分', '7天死亡']

# 计算相关矩阵
corr_data = df_filtered[valid_features_with_target].corr()

# 绘制热图
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax,
            vmin=-1, vmax=1)

ax.set_title('快速评估指标相关性热图', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(output_dir / "feature_correlation_heatmap.png", dpi=DPI_DEFAULT, bbox_inches='tight')
print(f"  [OK] 保存图表: {output_dir / 'feature_correlation_heatmap.png'}")
plt.close()
print()

# ============================================================================
# 7. 生成数据探索报告
# ============================================================================

print("生成数据探索报告...")

report_lines = [
    "# 方向2 - 机器学习建模：数据探索报告\n",
    f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
    "---\n",
    "\n## 1. 研究目标\n",
    "评估在快速评估场景下（无需采血化验），FaceAge 相对于真实年龄的增量预测价值。\n",
    "\n## 2. 快速评估指标（9个）\n",
]

for i, (feature_name, col_name) in enumerate(feature_mapping.items(), 1):
    report_lines.append(f"{i}. {feature_name}（字段名: `{col_name}`）\n")

report_lines.extend([
    "\n## 3. 数据概况\n",
    f"- 原始样本数: {len(df)}\n",
    f"- 过滤后样本数: {len(df_filtered)} (排除重度遮挡和无效生物学年龄)\n",
    f"- 7天死亡率: {df_filtered['7天死亡'].mean()*100:.1f}%\n",
    f"- 死亡人数: {int(df_filtered['7天死亡'].sum())}\n",
    f"- 生存人数: {int((1-df_filtered['7天死亡']).sum())}\n",
    "\n## 4. 缺失值分析\n",
    "\n| 特征 | 缺失数 | 缺失率(%) | 有效样本数 | 状态 |\n",
    "|------|--------|-----------|-----------|------|\n"
])

for _, row in missing_df.iterrows():
    status = ""
    if row['缺失率(%)'] < 5:
        status = "✅ 优秀"
    elif row['缺失率(%)'] < 20:
        status = "⚠️ 可接受"
    else:
        status = "❌ 不可用"

    report_lines.append(
        f"| {row['特征']} | {row['缺失数']:.0f} | {row['缺失率(%)']:.1f} | {row['有效样本数']:.0f} | {status} |\n"
    )

report_lines.extend([
    "\n### 数据预处理策略\n"
])

if low_missing:
    report_lines.append(f"- **缺失率 < 5%**（删除缺失样本）: {', '.join(low_missing)}\n")
if medium_missing:
    report_lines.append(f"- **缺失率 5-20%**（中位数/均值填充）: {', '.join(medium_missing)}\n")
if high_missing:
    report_lines.append(f"- **缺失率 > 20%**（考虑删除特征）: {', '.join(high_missing)}\n")

report_lines.extend([
    "\n## 5. 特征描述性统计\n",
    "\n| 特征 | 均值 | 标准差 | 最小值 | 最大值 |\n",
    "|------|------|--------|--------|--------|\n"
])

for _, row in missing_df.iterrows():
    if not pd.isna(row['均值']):
        report_lines.append(
            f"| {row['特征']} | {row['均值']:.2f} | {row['标准差']:.2f} | {row['最小值']:.2f} | {row['最大值']:.2f} |\n"
        )
    else:
        report_lines.append(f"| {row['特征']} | - | - | - | - |\n")

report_lines.extend([
    "\n## 6. 与目标变量（7天死亡）的相关性\n",
    "\n| 特征 | Pearson 相关系数 |\n",
    "|------|------------------|\n"
])

for feature in valid_features:
    if feature in df_filtered.columns:
        corr_val = df_filtered[[feature, '7天死亡']].corr().iloc[0, 1]
        report_lines.append(f"| {feature} | {corr_val:+.4f} |\n")

report_lines.append(f"| 实际年龄 | {df_filtered[['实际年龄', '7天死亡']].corr().iloc[0, 1]:+.4f} |\n")
report_lines.append(f"| 生物学年龄 | {df_filtered[['生物学年龄', '7天死亡']].corr().iloc[0, 1]:+.4f} |\n")
report_lines.append(f"| Age Gap | {df_filtered[['Age_Gap', '7天死亡']].corr().iloc[0, 1]:+.4f} |\n")
report_lines.append(f"| APACHE评分 | {df_filtered[['APACHE评分', '7天死亡']].corr().iloc[0, 1]:+.4f} |\n")

report_lines.extend([
    "\n## 7. 主要发现\n"
])

# 评估数据质量
usable_features = [f for f in quick_features if f in valid_features]
unusable_features = [f for f in quick_features if f not in valid_features]

if len(usable_features) >= 6:
    report_lines.append(f"- [OK] **数据质量良好**: {len(usable_features)}/{len(quick_features)} 个特征可用\n")
elif len(usable_features) >= 4:
    report_lines.append(f"- [!] **数据质量中等**: {len(usable_features)}/{len(quick_features)} 个特征可用\n")
else:
    report_lines.append(f"- [X] **数据质量较差**: 仅 {len(usable_features)}/{len(quick_features)} 个特征可用\n")

if unusable_features:
    report_lines.append(f"- **不可用特征**: {', '.join(unusable_features)}\n")

# 样本量评估
if len(df_filtered) >= 1000:
    report_lines.append(f"- ✅ **样本量充足**: {len(df_filtered)} 例\n")
elif len(df_filtered) >= 500:
    report_lines.append(f"- ⚠️ **样本量中等**: {len(df_filtered)} 例\n")
else:
    report_lines.append(f"- ❌ **样本量不足**: {len(df_filtered)} 例\n")

# 类别平衡评估
death_rate = df_filtered['7天死亡'].mean() * 100
if 10 <= death_rate <= 30:
    report_lines.append(f"- ✅ **类别平衡良好**: 死亡率 {death_rate:.1f}%\n")
elif 5 <= death_rate < 10 or 30 < death_rate <= 40:
    report_lines.append(f"- ⚠️ **轻度类别不平衡**: 死亡率 {death_rate:.1f}%（可能需要SMOTE或加权）\n")
else:
    report_lines.append(f"- ❌ **严重类别不平衡**: 死亡率 {death_rate:.1f}%（需要特殊处理）\n")

report_lines.extend([
    "\n## 8. 下一步建议\n",
    "\n### 数据预处理\n"
])

if high_missing:
    report_lines.append(f"1. **删除高缺失特征**: {', '.join(high_missing)}\n")
if medium_missing:
    report_lines.append(f"2. **填充中等缺失特征**: {', '.join(medium_missing)}（使用中位数）\n")
if low_missing:
    report_lines.append(f"3. **删除低缺失样本**: {', '.join(low_missing)}\n")

report_lines.extend([
    "4. **特征标准化**: 使用 StandardScaler\n",
    "\n### 建模策略\n",
    "1. 使用分层5折交叉验证（确保每折死亡率一致）\n",
    "2. 算法选择: Random Forest 或 XGBoost\n",
    "3. 超参数调优: GridSearchCV 或 RandomizedSearchCV\n"
])

if death_rate < 10 or death_rate > 30:
    report_lines.append("4. 处理类别不平衡: 考虑SMOTE过采样或类别权重\n")

report_lines.extend([
    "\n## 9. 可视化结果\n",
    "- `feature_distributions.png`: 特征分布直方图\n",
    "- `feature_correlation_heatmap.png`: 特征相关性热图\n",
    "\n## 10. 数据文件\n",
    "- `missing_values_analysis.csv`: 缺失值详细分析\n",
    "\n---\n",
    "*本报告由数据探索脚本自动生成*\n"
])

report_path = output_dir / "data_exploration_report.md"
with open(report_path, 'w', encoding='utf-8') as f:
    f.writelines(report_lines)

print(f"  [OK] 保存报告: {report_path}")
print()

# ============================================================================
# 完成
# ============================================================================

print("=" * 80)
print("[OK] 数据探索完成！")
print("=" * 80)
print(f"\n结果保存在: {output_dir}")
print("\n生成文件:")
print("  1. missing_values_analysis.csv - 缺失值分析")
print("  2. feature_distributions.png - 特征分布图")
print("  3. feature_correlation_heatmap.png - 相关性热图")
print("  4. data_exploration_report.md - 数据探索报告")
print()

# 打印关键信息
print("关键信息:")
print(f"  可用特征数: {len(usable_features)}/{len(quick_features)}")
print(f"  有效样本数: {len(df_filtered)}")
print(f"  死亡率: {death_rate:.1f}%")

if len(usable_features) >= 4 and len(df_filtered) >= 500:
    print("\n  [OK] 数据质量足够，可以继续建模！")
else:
    print("\n  [!] 数据质量可能不足，请仔细检查报告")
print()
