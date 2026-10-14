"""Bland-Altman agreement between FaceAge and chronological age.

The agreement statistics the manuscript reports in place of r and R-squared are computed here:
the overall mean bias and 95% limits of agreement with their confidence intervals, the
proportional-bias regression, and the same quantities within each occlusion grade and each age
band. Section 1 re-derives the key numbers of the manuscript and compares them item by item, so
that a mismatch becomes visible immediately if the input data change.

Statistical methods
-------------------
  mean bias 95% CI   bias +/- t(0.975, n-1) * SD / sqrt(n)
  LoA 95% CI         LoA +/- t(0.975, n-1) * sqrt(3 * SD^2 / n)     (Bland & Altman 1986)
  proportional bias  difference on chronological age is the primary test, because the slope is
                     then directly interpretable as the change in FaceAge error per year of age.
                     difference on mean is reported as a secondary check, with a note on why it
                     is not trustworthy on its own: when the reference standard varies little
                     within a stratum, the mean and the difference are both driven by FaceAge's
                     measurement error and correlate spuriously.

Relationship to other scripts
-----------------------------
Sections 4.1 and 4.2 recompute metrics that stage1_performance.py already produces; they are
kept here so that the agreement and accuracy figures come from a single pass and cannot drift
apart. Section 4.4 (>= 60 years) overlaps stage1c_age_band_alignment.py, which uses the age
bands of the original FaceAge paper rather than bands chosen here.

Input  : data/merged_data.xlsx
Output : results/bland_altman/overall_agreement.csv
         results/bland_altman/by_occlusion.csv
         results/bland_altman/by_age_group.csv
         results/bland_altman/subgroup_ge60.csv
         results/bland_altman/consistency_check.csv
         results/bland_altman/agreement_report.md
         results/figures/stage1b_bland_altman_overall.png
         results/figures/stage1b_bland_altman_by_occlusion.png
         results/figures/stage1b_bland_altman_by_age_group.png
         results/figures/stage1b_bland_altman_ge60.png"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# 设置工作目录为项目根目录（ClinFace-ICU）
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
os.chdir(project_root)

# 添加 utils 到路径
sys.path.insert(0, str(script_dir))

from utils import load_from_merged_xlsx, RESULTS_DIR

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 0.8

print("=" * 80)
print("阶段一（补充）：FaceAge 一致性分析（Bland-Altman）")
print("=" * 80)
print(f"工作目录: {os.getcwd()}")
print()

# ============================================================================
# 1. 加载数据
# ============================================================================
print("[1/6] 加载数据...")

df_full = load_from_merged_xlsx(max_occlusion_level=3)   # 全分析集 N=1485
df_main = load_from_merged_xlsx(max_occlusion_level=2)   # 主分析集 N=1470

# 统一列名：实际年龄 / 生物学年龄 / Age_Gap / 遮挡等级 / 7天死亡
df_full = df_full.rename(columns={'实际年龄': 'chronological_age',
                                  '生物学年龄': 'faceage',
                                  'Age_Gap': 'age_gap',
                                  '遮挡等级': 'occlusion',
                                  '7天死亡': 'death_7d'})
df_main = df_main.rename(columns={'实际年龄': 'chronological_age',
                                  '生物学年龄': 'faceage',
                                  'Age_Gap': 'age_gap',
                                  '遮挡等级': 'occlusion',
                                  '7天死亡': 'death_7d'})

print(f"  全分析集 (<= Level 3): N = {len(df_full)}")
print(f"  主分析集 (<= Level 2): N = {len(df_main)}")
print()

# ============================================================================
# 2. 数据一致性校验（与手稿关键数字比对）
# ============================================================================
print("[2/6] 数据一致性校验（与手稿比对）...")

checks = []


def add_check(item, observed, expected, note=''):
    """记录一项校验结果"""
    ok = (observed == expected)
    checks.append({
        '校验项': item,
        '脚本计算值': observed,
        '手稿值': expected,
        '是否一致': 'PASS' if ok else 'FAIL',
        '备注': note
    })
    return ok


add_check('全分析集 N', len(df_full), 1485, 'Table 3 表头')
add_check('主分析集 N', len(df_main), 1470, 'Table 2 表头')
for lv, exp in [(0, 1244), (1, 198), (2, 28), (3, 15)]:
    add_check(f'遮挡 Level {lv} n', int((df_full['occlusion'] == lv).sum()), exp, 'Table 3')
add_check('全分析集死亡数', int(df_full['death_7d'].sum()), 155, 'Table 3 表头')
add_check('主分析集死亡数', int(df_main['death_7d'].sum()), 150, 'Table 5 / Table 7')
add_check('全分析集 MAE (yr)',
          round(float(df_full['age_gap'].abs().mean()), 2), 7.60, 'Table 3')
add_check('Level 0 MAE (yr)',
          round(float(df_full.loc[df_full['occlusion'] == 0, 'age_gap'].abs().mean()), 2),
          7.48, 'Table 3')
add_check('误差 <=5 yr 例数', int((df_full['age_gap'].abs() <= 5).sum()), 595, 'Table 3')
add_check('误差 <=10 yr 例数', int((df_full['age_gap'].abs() <= 10).sum()), 1064, 'Table 3')
add_check('年龄组 <45 n (主分析集)', int((df_main['chronological_age'] < 45).sum()),
          177, 'Table 8 (N=1470)')
add_check('年龄组 45-64 n (主分析集)',
          int(((df_main['chronological_age'] >= 45) & (df_main['chronological_age'] < 65)).sum()),
          463, 'Table 8 (N=1470)')
add_check('年龄组 65-84 n (主分析集)',
          int(((df_main['chronological_age'] >= 65) & (df_main['chronological_age'] < 85)).sum()),
          628, 'Table 8 (N=1470)')
add_check('年龄组 >=85 n (主分析集)', int((df_main['chronological_age'] >= 85).sum()),
          202, 'Table 8 (N=1470)')

check_df = pd.DataFrame(checks)
n_fail = int((check_df['是否一致'] == 'FAIL').sum())
for _, row in check_df.iterrows():
    flag = '[OK]' if row['是否一致'] == 'PASS' else '[!!]'
    print(f"  {flag} {row['校验项']}: 计算={row['脚本计算值']}, 手稿={row['手稿值']}"
          f"  ({row['备注']})")
print(f"  校验结果: {len(check_df) - n_fail}/{len(check_df)} 项一致")
if n_fail > 0:
    print(f"  [警告] 有 {n_fail} 项与手稿不一致，请核查后再使用本结果！")
print()

# ============================================================================
# 3. 一致性统计函数
# ============================================================================


def bland_altman_stats(chronological_age, faceage, label=''):
    """
    计算 Bland-Altman 一致性统计量

    返回 dict:
        n, bias, sd_diff, loa_lower, loa_upper, loa_lower_ci, loa_upper_ci,
        bias_ci_lower, bias_ci_upper, mae, rmse, within_loa_pct,
        prop_slope, prop_slope_ci, prop_p, prop_r2
    """
    x = np.asarray(chronological_age, dtype=float)
    y = np.asarray(faceage, dtype=float)
    n = len(x)
    if n < 3:
        return None

    diff = y - x                 # 差值 = FaceAge - 时间年龄
    mean_vals = (y + x) / 2      # 均值

    bias = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))
    se_bias = sd_diff / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)

    loa_lower = bias - 1.96 * sd_diff
    loa_upper = bias + 1.96 * sd_diff

    # LoA 的置信区间（Bland & Altman 1986: Var(LoA) ≈ 3 * SD^2 / n）
    se_loa = np.sqrt(3 * sd_diff ** 2 / n)
    loa_lower_ci = (loa_lower - t_crit * se_loa, loa_lower + t_crit * se_loa)
    loa_upper_ci = (loa_upper - t_crit * se_loa, loa_upper + t_crit * se_loa)

    # 比例偏倚（主要检验）：差值对时间年龄的 OLS 回归
    slope_a, intercept_a, r_a, p_a, se_a = stats.linregress(x, diff)
    slope_a_ci = (slope_a - t_crit * se_a, slope_a + t_crit * se_a)

    # 比例偏倚（次要）：差值对均值的 OLS 回归（受测量误差影响，需谨慎解释）
    slope, intercept, r_val, p_val, se_slope = stats.linregress(mean_vals, diff)
    slope_ci = (slope - t_crit * se_slope, slope + t_crit * se_slope)

    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    within = float(np.mean((diff >= loa_lower) & (diff <= loa_upper)) * 100)

    return {
        'group': label,
        'n': n,
        'bias': bias,
        'sd_diff': sd_diff,
        'bias_ci_lower': bias - t_crit * se_bias,
        'bias_ci_upper': bias + t_crit * se_bias,
        'loa_lower': loa_lower,
        'loa_upper': loa_upper,
        'loa_lower_ci_lower': loa_lower_ci[0],
        'loa_lower_ci_upper': loa_lower_ci[1],
        'loa_upper_ci_lower': loa_upper_ci[0],
        'loa_upper_ci_upper': loa_upper_ci[1],
        'mae': mae,
        'rmse': rmse,
        'within_loa_pct': within,
        'prop_slope': float(slope),
        'prop_slope_ci_lower': float(slope_ci[0]),
        'prop_slope_ci_upper': float(slope_ci[1]),
        'prop_p': float(p_val),
        'prop_r2': float(r_val ** 2),
        'prop_slope_age': float(slope_a),
        'prop_slope_age_ci_lower': float(slope_a_ci[0]),
        'prop_slope_age_ci_upper': float(slope_a_ci[1]),
        'prop_p_age': float(p_a),
        'prop_r2_age': float(r_a ** 2),
    }


def fmt_agreement_row(r):
    """打印一行一致性结果"""
    return (f"  {r['group']:<22} n={r['n']:>4}  "
            f"bias={r['bias']:+.2f} ({r['bias_ci_lower']:+.2f} to {r['bias_ci_upper']:+.2f})  "
            f"95% LoA={r['loa_lower']:+.2f} to {r['loa_upper']:+.2f}  "
            f"MAE={r['mae']:.2f}  "
            f"prop.bias(age) slope={r['prop_slope_age']:+.3f} (P={r['prop_p_age']:.3g})")


# ============================================================================
# 4. 计算各层一致性统计量
# ============================================================================
print("[3/6] 计算 Bland-Altman 一致性统计量...")

# 4.1 整体（全分析集 N=1485，对应手稿 Table 3）
overall_rows = [
    bland_altman_stats(df_full['chronological_age'], df_full['faceage'],
                       'Overall (full set)'),
    bland_altman_stats(df_main['chronological_age'], df_main['faceage'],
                       'Main analysis set'),
]
print("\n整体：")
for r in overall_rows:
    print(fmt_agreement_row(r))

# 4.2 按遮挡等级（全分析集，对应手稿 Table 3）
occ_labels = {0: 'Level 0 (none)', 1: 'Level 1 (mild)',
              2: 'Level 2 (moderate)', 3: 'Level 3 (severe)'}
occlusion_rows = []
for lv in [0, 1, 2, 3]:
    sub = df_full[df_full['occlusion'] == lv]
    r = bland_altman_stats(sub['chronological_age'], sub['faceage'], occ_labels[lv])
    if r is not None:
        occlusion_rows.append(r)
# 累积分层（对应手稿 Table 6 的口径）
for max_lv in [0, 1, 2, 3]:
    sub = df_full[df_full['occlusion'] <= max_lv]
    r = bland_altman_stats(sub['chronological_age'], sub['faceage'],
                           f'Cumulative <= Level {max_lv}')
    if r is not None:
        occlusion_rows.append(r)
print("\n按遮挡等级：")
for r in occlusion_rows:
    print(fmt_agreement_row(r))

# 4.3 按年龄组（主分析集 N=1470，对应手稿 Table 8）
age_bins = [(-np.inf, 45, '<45 (young)'),
            (45, 65, '45-64 (middle)'),
            (65, 85, '65-84 (old)'),
            (85, np.inf, '>=85 (very old)')]
age_rows = []
for lo, hi, label in age_bins:
    sub = df_main[(df_main['chronological_age'] >= lo) &
                  (df_main['chronological_age'] < hi)]
    r = bland_altman_stats(sub['chronological_age'], sub['faceage'], label)
    if r is not None:
        age_rows.append(r)
print("\n按年龄组（主分析集）：")
for r in age_rows:
    print(fmt_agreement_row(r))

# 4.4 >=60 岁适用性人群（对应 Comment 3）
ge60_rows = []
sub = df_full[df_full['chronological_age'] >= 60]
ge60_rows.append(bland_altman_stats(sub['chronological_age'], sub['faceage'],
                                    '>=60 yr (full set)'))
sub = df_main[df_main['chronological_age'] >= 60]
ge60_rows.append(bland_altman_stats(sub['chronological_age'], sub['faceage'],
                                    '>=60 yr (main set)'))
sub = df_full[df_full['chronological_age'] < 60]
ge60_rows.append(bland_altman_stats(sub['chronological_age'], sub['faceage'],
                                    '<60 yr (full set)'))
print("\n年龄适用性亚组（Comment 3）：")
for r in ge60_rows:
    print(fmt_agreement_row(r))

# ============================================================================
# 5. 保存结果表
# ============================================================================
print("\n[4/6] 保存结果表...")

results_dir = RESULTS_DIR / "bland_altman"
figures_dir = RESULTS_DIR / "figures"
results_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

cols = ['group', 'n', 'bias', 'sd_diff', 'bias_ci_lower', 'bias_ci_upper',
        'loa_lower', 'loa_upper', 'loa_lower_ci_lower', 'loa_lower_ci_upper',
        'loa_upper_ci_lower', 'loa_upper_ci_upper', 'mae', 'rmse',
        'within_loa_pct',
        'prop_slope_age', 'prop_slope_age_ci_lower', 'prop_slope_age_ci_upper',
        'prop_p_age', 'prop_r2_age',
        'prop_slope', 'prop_slope_ci_lower', 'prop_slope_ci_upper',
        'prop_p', 'prop_r2']

pd.DataFrame(overall_rows)[cols].to_csv(
    results_dir / "overall_agreement.csv", index=False, encoding='utf-8-sig')
pd.DataFrame(occlusion_rows)[cols].to_csv(
    results_dir / "by_occlusion.csv", index=False, encoding='utf-8-sig')
pd.DataFrame(age_rows)[cols].to_csv(
    results_dir / "by_age_group.csv", index=False, encoding='utf-8-sig')
pd.DataFrame(ge60_rows)[cols].to_csv(
    results_dir / "subgroup_ge60.csv", index=False, encoding='utf-8-sig')
check_df.to_csv(results_dir / "consistency_check.csv", index=False,
                encoding='utf-8-sig')
print(f"  [OK] {results_dir / 'overall_agreement.csv'}")
print(f"  [OK] {results_dir / 'by_occlusion.csv'}")
print(f"  [OK] {results_dir / 'by_age_group.csv'}")
print(f"  [OK] {results_dir / 'subgroup_ge60.csv'}")
print(f"  [OK] {results_dir / 'consistency_check.csv'}")

# ============================================================================
# 6. 绘图
# ============================================================================
print("\n[5/6] 生成 Bland-Altman 图...")

C_SCATTER = '#4C72B0'
C_BIAS = '#C44E52'
C_LOA = '#DD8452'
C_TREND = '#55A868'


def draw_bland_altman(ax, x, y, title=''):
    """在指定 axes 上绘制 Bland-Altman 图（出版级英文标注）"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diff = y - x
    mean_vals = (y + x) / 2
    st = bland_altman_stats(x, y)

    ax.scatter(mean_vals, diff, s=12, alpha=0.45, color=C_SCATTER,
               edgecolors='none', zorder=2)
    ax.axhline(0, color='0.6', linestyle=':', linewidth=1, zorder=1)
    ax.axhline(st['bias'], color=C_BIAS, linestyle='-', linewidth=1.8,
               label=f"Mean bias = {st['bias']:+.2f}", zorder=3)
    ax.axhline(st['loa_upper'], color=C_LOA, linestyle='--', linewidth=1.5,
               label=f"+1.96 SD = {st['loa_upper']:+.2f}", zorder=3)
    ax.axhline(st['loa_lower'], color=C_LOA, linestyle='--', linewidth=1.5,
               label=f"-1.96 SD = {st['loa_lower']:+.2f}", zorder=3)

    ax.set_xlabel('Average of FaceAge and chronological age (years)')
    ax.set_ylabel('FaceAge - chronological age (years)')
    ax.set_title(f"{title} (n = {st['n']})", fontsize=11)
    ax.legend(loc='upper right', fontsize=7.5, framealpha=0.9)
    ax.grid(alpha=0.25, linewidth=0.6)
    return st


def draw_diff_vs_age(ax, x, y, title=''):
    """绘制差值 vs 时间年龄图（比例偏倚的主要检验）"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diff = y - x
    st = bland_altman_stats(x, y)

    ax.scatter(x, diff, s=12, alpha=0.45, color=C_SCATTER,
               edgecolors='none', zorder=2)
    ax.axhline(0, color='0.6', linestyle=':', linewidth=1, zorder=1)

    slope, intercept, _, _, _ = stats.linregress(x, diff)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, intercept + slope * xs, color=C_TREND, linewidth=2,
            label=f"Slope = {slope:+.3f} "
                  f"({st['prop_slope_age_ci_lower']:+.3f} to "
                  f"{st['prop_slope_age_ci_upper']:+.3f}), "
                  f"P = {st['prop_p_age']:.3g}", zorder=4)
    ax.axhline(st['bias'], color=C_BIAS, linestyle='-', linewidth=1.5,
               label=f"Mean bias = {st['bias']:+.2f}", zorder=3)

    ax.set_xlabel('Chronological age (years)')
    ax.set_ylabel('FaceAge - chronological age (years)')
    ax.set_title(f"{title} (n = {st['n']})", fontsize=11)
    ax.legend(loc='upper right', fontsize=7.5, framealpha=0.9)
    ax.grid(alpha=0.25, linewidth=0.6)
    return st


# 图1：整体（双面板：标准 BA 图 + 差值 vs 时间年龄）
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
draw_bland_altman(axes[0], df_full['chronological_age'], df_full['faceage'],
                  'A. Bland-Altman plot')
draw_diff_vs_age(axes[1], df_full['chronological_age'], df_full['faceage'],
                 'B. Difference vs chronological age')
fig.suptitle('FaceAge agreement with chronological age (full analysis set)',
             fontsize=13)
fig.tight_layout()
fig.savefig(figures_dir / "stage1b_bland_altman_overall.png", dpi=300,
            bbox_inches='tight')
plt.close(fig)
print(f"  [OK] {figures_dir / 'stage1b_bland_altman_overall.png'}")

# 图2：按遮挡等级（2x2）
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
axes = axes.flatten()
for i, lv in enumerate([0, 1, 2, 3]):
    sub = df_full[df_full['occlusion'] == lv]
    draw_bland_altman(axes[i], sub['chronological_age'], sub['faceage'],
                      occ_labels[lv])
fig.suptitle('Bland-Altman analysis stratified by facial occlusion level',
             fontsize=13)
fig.tight_layout()
fig.savefig(figures_dir / "stage1b_bland_altman_by_occlusion.png", dpi=300,
            bbox_inches='tight')
plt.close(fig)
print(f"  [OK] {figures_dir / 'stage1b_bland_altman_by_occlusion.png'}")

# 图3：按年龄组（2x2）
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
axes = axes.flatten()
for i, (lo, hi, label) in enumerate(age_bins):
    sub = df_main[(df_main['chronological_age'] >= lo) &
                  (df_main['chronological_age'] < hi)]
    draw_bland_altman(axes[i], sub['chronological_age'], sub['faceage'], label)
fig.suptitle('Bland-Altman analysis stratified by chronological age group',
             fontsize=13)
fig.tight_layout()
fig.savefig(figures_dir / "stage1b_bland_altman_by_age_group.png", dpi=300,
            bbox_inches='tight')
plt.close(fig)
print(f"  [OK] {figures_dir / 'stage1b_bland_altman_by_age_group.png'}")

# 图4：>=60 岁 vs <60 岁（Comment 3）
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
sub_ge = df_full[df_full['chronological_age'] >= 60]
sub_lt = df_full[df_full['chronological_age'] < 60]
draw_bland_altman(axes[0], sub_ge['chronological_age'], sub_ge['faceage'],
                  'Age >= 60 years (development population)')
draw_bland_altman(axes[1], sub_lt['chronological_age'], sub_lt['faceage'],
                  'Age < 60 years (outside development range)')
fig.tight_layout()
fig.savefig(figures_dir / "stage1b_bland_altman_ge60.png", dpi=300,
            bbox_inches='tight')
plt.close(fig)
print(f"  [OK] {figures_dir / 'stage1b_bland_altman_ge60.png'}")

# ============================================================================
# 7. 生成文本报告（供回复信与手稿引用）
# ============================================================================
print("\n[6/6] 生成一致性分析报告...")

ov = overall_rows[0]
lines = []
lines.append("# Bland-Altman 一致性分析报告\n")
lines.append("对应审稿意见：Reviewer 1 - Comment 1（以 Bland-Altman 替代 r / R²）\n")
lines.append(f"数据一致性校验：{len(check_df) - n_fail}/{len(check_df)} 项与手稿一致\n")
lines.append("## 整体一致性（全分析集，N = %d）\n" % ov['n'])
lines.append(f"- Mean bias: {ov['bias']:+.2f} years "
             f"(95% CI {ov['bias_ci_lower']:+.2f} to {ov['bias_ci_upper']:+.2f})\n")
lines.append(f"- 95% limits of agreement: {ov['loa_lower']:+.2f} to "
             f"{ov['loa_upper']:+.2f} years\n")
lines.append(f"- LoA 95% CI: 下限 {ov['loa_lower_ci_lower']:+.2f} to "
             f"{ov['loa_lower_ci_upper']:+.2f}; 上限 {ov['loa_upper_ci_lower']:+.2f} "
             f"to {ov['loa_upper_ci_upper']:+.2f}\n")
lines.append(f"- MAE {ov['mae']:.2f} years; RMSE {ov['rmse']:.2f} years\n")
lines.append(f"- 差值落在 LoA 内比例: {ov['within_loa_pct']:.1f}%\n")
lines.append(f"- 比例偏倚（差值 vs 时间年龄，主要检验）: slope = "
             f"{ov['prop_slope_age']:+.3f} (95% CI "
             f"{ov['prop_slope_age_ci_lower']:+.3f} to "
             f"{ov['prop_slope_age_ci_upper']:+.3f}), P = {ov['prop_p_age']:.3g}, "
             f"R² = {ov['prop_r2_age']:.3f}\n")
lines.append(f"- 比例偏倚（差值 vs 均值，次要，仅供参考）: slope = "
             f"{ov['prop_slope']:+.3f}, P = {ov['prop_p']:.3g}\n")
lines.append("  说明：当参照标准（时间年龄）在层内变异较小时，均值与差值同时受 "
             "FaceAge 测量误差驱动，差值-均值回归会产生强正相关假象，"
             "故本报告以差值-时间年龄回归作为比例偏倚的主要依据。\n")

lines.append("\n## 按遮挡等级分层\n")
for r in occlusion_rows:
    lines.append(f"- {r['group']}: n = {r['n']}, bias {r['bias']:+.2f} "
                 f"({r['bias_ci_lower']:+.2f} to {r['bias_ci_upper']:+.2f}), "
                 f"95% LoA {r['loa_lower']:+.2f} to {r['loa_upper']:+.2f}, "
                 f"MAE {r['mae']:.2f}\n")

lines.append("\n## 按年龄组分层（主分析集）\n")
for r in age_rows:
    lines.append(f"- {r['group']}: n = {r['n']}, bias {r['bias']:+.2f} "
                 f"({r['bias_ci_lower']:+.2f} to {r['bias_ci_upper']:+.2f}), "
                 f"95% LoA {r['loa_lower']:+.2f} to {r['loa_upper']:+.2f}, "
                 f"MAE {r['mae']:.2f}\n")

lines.append("\n## 年龄适用性亚组（Comment 3）\n")
for r in ge60_rows:
    lines.append(f"- {r['group']}: n = {r['n']}, bias {r['bias']:+.2f}, "
                 f"95% LoA {r['loa_lower']:+.2f} to {r['loa_upper']:+.2f}, "
                 f"MAE {r['mae']:.2f}\n")

lines.append("\n## 手稿修改建议\n")
lines.append("- Methods：将 'Using chronological age as the reference standard... "
             "MAE, RMSE, R², and Pearson correlation coefficient' 改为 "
             "agreement 框架，报告 mean bias 与 95% LoA。\n")
lines.append("- Table 3：删除 R² 列，新增 Mean bias (95% LoA) 列。\n")
lines.append("- Figure 2：新增 Bland-Altman 面板（差值-均值图）。\n")
lines.append("- Results/Discussion：以 LoA 与比例偏倚表述替代 r / R²。\n")

report_path = results_dir / "agreement_report.md"
with open(report_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)
print(f"  [OK] {report_path}")

print()
print("=" * 80)
print("一致性分析完成")
print("=" * 80)
print(f"结果表: {results_dir}")
print(f"图    : {figures_dir / 'stage1b_bland_altman_*.png'}")
print(f"一致性校验: {len(check_df) - n_fail}/{len(check_df)} 项通过")
print("=" * 80)
