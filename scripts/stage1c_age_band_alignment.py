"""FaceAge agreement in the age bands used by the original FaceAge paper.

FaceAge was developed on a presumed-healthy population aged 60 and over, so its published MAE is
not comparable with a whole-cohort figure. This script re-stratifies the cohort into the bands
the original report uses and recomputes, with the same definitions as Table 3, the quantities
that can be set directly beside the published values:

  original report, technical validation set    < 40 : MAE 7.80    40-60 : MAE 5.15
                                               > 60 : MAE 4.09    all   : MAE 5.87

Reported here: MAE, RMSE, the number and share of estimates within 5 and within 10 years, mean
bias and the 95% limits of agreement. The manuscript quotes the >= 60-year band.

Input  : data/merged_data.xlsx
Output : results/age_band_alignment/age_band_main_set.csv
         results/age_band_alignment/age_band_full_set.csv
         results/age_band_alignment/age_band_report.md"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# 设置工作目录为项目根目录（ClinFace-ICU）
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(script_dir))

from utils import load_from_merged_xlsx, RESULTS_DIR

print("=" * 78)
print("按 FaceAge 原文年龄区间对齐的分层一致性分析")
print("=" * 78)
print(f"工作目录: {os.getcwd()}")
print()

# ============================================================================
# 1. 加载数据
# ============================================================================
print("[1/4] 加载数据...")
df_full = load_from_merged_xlsx(max_occlusion_level=3)   # 全分析集 N=1485
df_main = load_from_merged_xlsx(max_occlusion_level=2)   # 主分析集 N=1470

col_map = {'实际年龄': 'chronological_age',
           '生物学年龄': 'faceage',
           'Age_Gap': 'age_gap',
           '遮挡等级': 'occlusion',
           '7天死亡': 'death_7d'}
df_full = df_full.rename(columns=col_map)
df_main = df_main.rename(columns=col_map)

print(f"  全分析集 (<= Level 3): N = {len(df_full)}")
print(f"  主分析集 (<= Level 2): N = {len(df_main)}")
print()

# ============================================================================
# 2. 区间定义（与原文 Supplementary Figure 14 一致）
# ============================================================================
# <40 / 40-60 / >60；边界归属：40-60 为闭区间 [40, 60]
AGE_BANDS = [
    (-np.inf, 40, '<40',  7.80),
    (40,      60, '40-60', 5.15),
    (60, np.inf, '>60',   4.09),
]
ALL_AGE_REF = 5.87


def band_stats(df, lo, hi, label, ref_mae):
    """计算单个年龄区间的 Table-3 口径指标"""
    # 边界约定：<40 为 (-inf, 40)；40-60 为 [40, 60]；>60 为 (60, +inf)
    if hi == np.inf:
        sub = df[df['chronological_age'] > lo]
    elif lo == -np.inf:
        sub = df[df['chronological_age'] < hi]
    else:
        sub = df[(df['chronological_age'] >= lo) & (df['chronological_age'] <= hi)]

    diff = sub['faceage'].to_numpy(dtype=float) - \
        sub['chronological_age'].to_numpy(dtype=float)
    n = len(diff)
    if n < 3:
        return None

    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    t_crit = stats.t.ppf(0.975, df=n - 1)

    return {
        'age_band': label,
        'n': n,
        'deaths_n': int(sub['death_7d'].sum()),
        'deaths_pct': float(sub['death_7d'].mean() * 100),
        'mae': float(np.mean(np.abs(diff))),
        'rmse': float(np.sqrt(np.mean(diff ** 2))),
        'err_le5_n': int((np.abs(diff) <= 5).sum()),
        'err_le5_pct': float((np.abs(diff) <= 5).mean() * 100),
        'err_le10_n': int((np.abs(diff) <= 10).sum()),
        'err_le10_pct': float((np.abs(diff) <= 10).mean() * 100),
        'bias': bias,
        'bias_ci_lower': bias - t_crit * sd / np.sqrt(n),
        'bias_ci_upper': bias + t_crit * sd / np.sqrt(n),
        'loa_lower': bias - 1.96 * sd,
        'loa_upper': bias + 1.96 * sd,
        'ref_mae': ref_mae,
        'mae_diff': float(np.mean(np.abs(diff))) - ref_mae,
    }


def overall_stats(df, label, ref_mae):
    diff = df['faceage'].to_numpy(dtype=float) - \
        df['chronological_age'].to_numpy(dtype=float)
    n = len(diff)
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return {
        'age_band': label,
        'n': n,
        'deaths_n': int(df['death_7d'].sum()),
        'deaths_pct': float(df['death_7d'].mean() * 100),
        'mae': float(np.mean(np.abs(diff))),
        'rmse': float(np.sqrt(np.mean(diff ** 2))),
        'err_le5_n': int((np.abs(diff) <= 5).sum()),
        'err_le5_pct': float((np.abs(diff) <= 5).mean() * 100),
        'err_le10_n': int((np.abs(diff) <= 10).sum()),
        'err_le10_pct': float((np.abs(diff) <= 10).mean() * 100),
        'bias': bias,
        'bias_ci_lower': bias - t_crit * sd / np.sqrt(n),
        'bias_ci_upper': bias + t_crit * sd / np.sqrt(n),
        'loa_lower': bias - 1.96 * sd,
        'loa_upper': bias + 1.96 * sd,
        'ref_mae': ref_mae,
        'mae_diff': float(np.mean(np.abs(diff))) - ref_mae,
    }


# ============================================================================
# 3. 计算
# ============================================================================
print("[2/4] 计算各年龄区间指标...")

results = {}
for set_name, df in [('main_set', df_main), ('full_set', df_full)]:
    rows = []
    print(f"\n  --- {'主分析集' if set_name == 'main_set' else '全分析集'} "
          f"(N = {len(df)}) ---")
    for i, (lo, hi, label, ref) in enumerate(AGE_BANDS, 1):
        print(f"  [{i}/{len(AGE_BANDS)}] 正在处理区间 {label} 岁 ...")
        r = band_stats(df, lo, hi, label, ref)
        if r is not None:
            rows.append(r)
            print(f"        n={r['n']:>4}  MAE={r['mae']:.2f} "
                  f"(原文 {r['ref_mae']:.2f}, 差 {r['mae_diff']:+.2f})  "
                  f"bias={r['bias']:+.2f}")
    ov = overall_stats(df, 'All ages', ALL_AGE_REF)
    rows.append(ov)
    print(f"  [4/4] 全年龄      n={ov['n']:>4}  MAE={ov['mae']:.2f} "
          f"(原文 {ov['ref_mae']:.2f}, 差 {ov['mae_diff']:+.2f})  "
          f"bias={ov['bias']:+.2f}")
    results[set_name] = pd.DataFrame(rows)

# ============================================================================
# 4. 保存
# ============================================================================
print("\n[3/4] 保存结果...")
out_dir = RESULTS_DIR / "age_band_alignment"
out_dir.mkdir(parents=True, exist_ok=True)

results['main_set'].to_csv(out_dir / "age_band_main_set.csv",
                           index=False, encoding='utf-8-sig')
results['full_set'].to_csv(out_dir / "age_band_full_set.csv",
                           index=False, encoding='utf-8-sig')
print(f"  [OK] {out_dir / 'age_band_main_set.csv'}")
print(f"  [OK] {out_dir / 'age_band_full_set.csv'}")

# 交叉校验：全分析集数值应与既往结果一致
print("\n[4/4] 交叉校验（全分析集）...")
expect = {'<40': 7.81, '40-60': 9.07, '>60': 6.98, 'All ages': 7.60}
ok = 0
for _, r in results['full_set'].iterrows():
    exp = expect.get(r['age_band'])
    got = round(r['mae'], 2)
    flag = '[OK]' if exp is not None and abs(got - exp) < 0.005 else '[!!]'
    if flag == '[OK]':
        ok += 1
    print(f"  {flag} {r['age_band']:<10} 计算 MAE={got:.2f}  既往值={exp}")
print(f"  校验: {ok}/4 项一致")

# ============================================================================
# 5. 报告
# ============================================================================
lines = []
lines.append("# 按 FaceAge 原文年龄区间对齐的分层一致性分析\n")
lines.append("对应审稿意见：Reviewer 1 - Comment 3\n")
lines.append("参照值来源：FaceAge 原文 Supplementary Figure 14"
             "（技术验证集年龄亚组 MAE）\n")

for set_name, title in [('main_set', '主分析集（N = 1,470，遮挡 Level 0–2）'),
                        ('full_set', '全分析集（N = 1,485，遮挡 Level 0–3）')]:
    lines.append(f"\n## {title}\n")
    lines.append("| Age band | n | Deaths, n (%) | MAE | RMSE | "
                 "Error <=5 yr, n (%) | Error <=10 yr, n (%) | "
                 "Mean bias (95% LoA) | Original MAE | Difference |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|---|\n")
    for _, r in results[set_name].iterrows():
        lines.append(
            f"| {r['age_band']} | {r['n']} | {r['deaths_n']} "
            f"({r['deaths_pct']:.1f}) | {r['mae']:.2f} | {r['rmse']:.2f} | "
            f"{r['err_le5_n']} ({r['err_le5_pct']:.1f}) | "
            f"{r['err_le10_n']} ({r['err_le10_pct']:.1f}) | "
            f"{r['bias']:+.2f} ({r['loa_lower']:+.2f} to {r['loa_upper']:+.2f}) | "
            f"{r['ref_mae']:.2f} | {r['mae_diff']:+.2f} |\n")

lines.append("\n## 说明\n")
lines.append("- 原文年龄亚组参照值取自 FaceAge 原文补充材料 "
             "Supplementary Figure 14，仅为其**技术验证集**结果；"
             "原文未报告其临床验证队列的年龄分层表现。\n")
lines.append("- 边界归属：40-60 为闭区间 [40, 60]，>60 为 (60, +inf)。\n")
lines.append("- 主分析集排除 Level 3（n=15）重症遮挡病例，"
             "因其 FaceAge 估计严重偏移，不宜纳入一致性比较。\n")

report_path = out_dir / "age_band_report.md"
with open(report_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)
print(f"  [OK] {report_path}")

print()
print("=" * 78)
print("完成")
print("=" * 78)
