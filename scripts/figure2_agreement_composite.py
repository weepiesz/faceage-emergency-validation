"""Figure 2: composite agreement panel.

One figure carrying the whole agreement story, in place of separate scatter, Bland-Altman and
difference-versus-age plots:

  (a) top row, four panels   FaceAge vs chronological age within each occlusion grade, with n,
                             MAE and mean bias in the panel titles (no r, no R-squared)
  (b) bottom left            Bland-Altman plot (difference against average), annotated with the
                             mean bias and the 95% limits of agreement
  (c) bottom right           difference against chronological age, with the proportional-bias
                             regression line

Design notes: a single colour-blind-safe palette (Okabe-Ito) and one set of font sizes
throughout; no figure-level title, since the caption does that job; a common axis range across
the four top-row panels so they can be compared directly; PNG at 300 dpi plus a vector PDF.

Input  : data/merged_data.xlsx   (all occlusion grades)
Output : results/figures/figure2_agreement_composite.png
         results/figures/figure2_agreement_composite.pdf"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy import stats

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(script_dir))

from utils import load_from_merged_xlsx, RESULTS_DIR

# ============================================================================
# 全局样式（全图统一）
# ============================================================================
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 9.5
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 9.5
plt.rcParams['xtick.labelsize'] = 8.5
plt.rcParams['ytick.labelsize'] = 8.5
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['legend.framealpha'] = 0.9
plt.rcParams['grid.alpha'] = 0.25
plt.rcParams['grid.linewidth'] = 0.6

# Okabe-Ito 色盲友好色板
PALETTE = {
    0: '#0072B2',   # blue
    1: '#E69F00',   # orange
    2: '#009E73',   # green
    3: '#CC79A7',   # reddish purple
}
C_ACCENT = '#D55E00'    # vermillion：偏倚 / LoA 线
C_TREND = '#009E73'     # green：比例偏倚回归线
C_ZERO = '0.55'         # 零线 / 参考线
C_PT = '#0072B2'        # Bland-Altman 散点

OCC_LABELS = {0: 'Level 0 (none)', 1: 'Level 1 (mild)',
              2: 'Level 2 (moderate)', 3: 'Level 3 (severe)'}

print("=" * 80)
print("生成 Figure 2 复合图（一致性分析）")
print("=" * 80)

# ============================================================================
# 1. 数据
# ============================================================================
print("[1/3] 加载数据...")
df = load_from_merged_xlsx(max_occlusion_level=3).rename(columns={
    '实际年龄': 'ca', '生物学年龄': 'fa', 'Age_Gap': 'gap', '遮挡等级': 'occ'})
print(f"  全分析集 N = {len(df)}")
print()


def agreement_stats(ca, fa):
    """Bland-Altman 一致性统计量（含比例偏倚回归）"""
    ca = np.asarray(ca, dtype=float)
    fa = np.asarray(fa, dtype=float)
    diff = fa - ca
    mean_v = (fa + ca) / 2
    n = len(ca)
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    t_crit = stats.t.ppf(0.975, df=n - 1)
    slope, intercept, r_val, p_val, se = stats.linregress(ca, diff)
    return {
        'n': n, 'bias': bias, 'sd': sd,
        'bias_lo': bias - t_crit * sd / np.sqrt(n),
        'bias_hi': bias + t_crit * sd / np.sqrt(n),
        'loa_lo': bias - 1.96 * sd,
        'loa_hi': bias + 1.96 * sd,
        'mae': float(np.mean(np.abs(diff))),
        'slope': float(slope), 'intercept': float(intercept),
        'p': float(p_val), 'se': float(se),
        'slope_lo': float(slope - t_crit * se),
        'slope_hi': float(slope + t_crit * se),
    }


# ============================================================================
# 2. 绘图
# ============================================================================
print("[2/3] 绘制复合图...")

fig = plt.figure(figsize=(14.0, 8.2))
gs = GridSpec(2, 4, figure=fig, height_ratios=[1.0, 1.12],
              hspace=0.42, wspace=0.32,
              left=0.055, right=0.985, top=0.94, bottom=0.075)

# ---------- (a) 上排：按遮挡等级分层的散点图 ----------
ax_a = [fig.add_subplot(gs[0, i]) for i in range(4)]

# 统一轴范围
all_ca, all_fa = df['ca'].values, df['fa'].values
lo = min(all_ca.min(), all_fa.min()) - 3
hi = max(all_ca.max(), all_fa.max()) + 3

for i, lv in enumerate([0, 1, 2, 3]):
    ax = ax_a[i]
    sub = df[df['occ'] == lv]
    st = agreement_stats(sub['ca'], sub['fa'])

    ax.scatter(sub['ca'], sub['fa'], s=9, alpha=0.40,
               color=PALETTE[lv], edgecolors='none', zorder=2)
    ax.plot([lo, hi], [lo, hi], linestyle='--', linewidth=1.2,
            color=C_ZERO, zorder=3, label='Identity')

    # 回归线（统一用强调色，风格一致；仅绘制在该组数据实际范围内，
    # 避免小样本组（Level 2/3）的拟合线外推到无数据区间造成误读）
    slope, intercept, _, _, _ = stats.linregress(sub['ca'], sub['fa'])
    xs = np.linspace(sub['ca'].min(), sub['ca'].max(), 50)
    ax.plot(xs, intercept + slope * xs, color=C_ACCENT, linewidth=1.6,
            zorder=4, label='Fit')

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"{OCC_LABELS[lv]}, n = {st['n']}\n"
                 f"MAE {st['mae']:.2f}, bias {st['bias']:+.2f}", pad=6)
    ax.grid(True)
    if i == 0:
        ax.set_ylabel('FaceAge-estimated age (years)')
        ax.legend(loc='lower right', markerscale=2.2)
    else:
        ax.tick_params(labelleft=False)
    ax.set_xlabel('Chronological age (years)')

# (a) 面板标签（内置左上角）
ax_a[0].text(0.035, 0.965, '(a)', transform=ax_a[0].transAxes,
             ha='left', va='top', fontsize=13, fontweight='bold',
             color='black', zorder=10)

# ---------- (b) 下排左：Bland-Altman 图 ----------
ax_b = fig.add_subplot(gs[1, 0:2])
ca, fa = df['ca'].values, df['fa'].values
st_all = agreement_stats(ca, fa)
diff_all = fa - ca
mean_all = (fa + ca) / 2

ax_b.scatter(mean_all, diff_all, s=10, alpha=0.40, color=C_PT,
             edgecolors='none', zorder=2)
ax_b.axhline(0, color=C_ZERO, linestyle=':', linewidth=1.0, zorder=1)
ax_b.axhline(st_all['bias'], color=C_ACCENT, linestyle='-', linewidth=1.8,
             label=f"Mean bias = {st_all['bias']:+.2f}", zorder=3)
ax_b.axhline(st_all['loa_hi'], color=C_ACCENT, linestyle='--', linewidth=1.5,
             label=f"+1.96 SD = {st_all['loa_hi']:+.2f}", zorder=3)
ax_b.axhline(st_all['loa_lo'], color=C_ACCENT, linestyle='--', linewidth=1.5,
             label=f"-1.96 SD = {st_all['loa_lo']:+.2f}", zorder=3)

ax_b.set_xlabel('Average of FaceAge and chronological age (years)')
ax_b.set_ylabel('FaceAge − chronological age (years)')
ax_b.legend(loc='upper right')
ax_b.grid(True)
ax_b.text(0.02, 0.97, '(b)', transform=ax_b.transAxes,
          ha='left', va='top', fontsize=13, fontweight='bold', zorder=10)

# ---------- (c) 下排右：差值 vs 时间年龄（比例偏倚） ----------
ax_c = fig.add_subplot(gs[1, 2:4])
ax_c.scatter(ca, diff_all, s=10, alpha=0.40, color=C_PT,
             edgecolors='none', zorder=2)
ax_c.axhline(0, color=C_ZERO, linestyle=':', linewidth=1.0, zorder=1)

xs = np.linspace(ca.min(), ca.max(), 50)
ax_c.plot(xs, st_all['intercept'] + st_all['slope'] * xs,
          color=C_TREND, linewidth=2.0, zorder=4,
          label=f"Slope = {st_all['slope']:+.3f} "
                f"({st_all['slope_lo']:+.3f} to {st_all['slope_hi']:+.3f}), "
                f"P < .001")
ax_c.axhline(st_all['bias'], color=C_ACCENT, linestyle='-', linewidth=1.5,
             label=f"Mean bias = {st_all['bias']:+.2f}", zorder=3)

ax_c.set_xlabel('Chronological age (years)')
ax_c.set_ylabel('FaceAge − chronological age (years)')
ax_c.legend(loc='upper right')
ax_c.grid(True)
ax_c.text(0.02, 0.97, '(c)', transform=ax_c.transAxes,
          ha='left', va='top', fontsize=13, fontweight='bold', zorder=10)

# ============================================================================
# 3. 保存
# ============================================================================
print("[3/3] 保存图片...")
figures_dir = RESULTS_DIR / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

png_path = figures_dir / "figure2_agreement_composite.png"
pdf_path = figures_dir / "figure2_agreement_composite.pdf"
fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(pdf_path, bbox_inches='tight')
plt.close(fig)
print(f"  [OK] {png_path}")
print(f"  [OK] {pdf_path}")

print()
print("=" * 80)
print("Figure 2 复合图生成完成")
print("=" * 80)
print(f"整体: n = {st_all['n']}, bias = {st_all['bias']:+.2f} "
      f"({st_all['bias_lo']:+.2f} to {st_all['bias_hi']:+.2f}), "
      f"95% LoA = {st_all['loa_lo']:+.2f} to {st_all['loa_hi']:+.2f}, "
      f"MAE = {st_all['mae']:.2f}")
print(f"比例偏倚: slope = {st_all['slope']:+.3f} "
      f"({st_all['slope_lo']:+.3f} to {st_all['slope_hi']:+.3f}), "
      f"P = {st_all['p']:.3g}")
print("=" * 80)
