"""Figure 7: ROC curves for 7-day mortality before and after bias correction.

Redraws the bias-correction ROC figure. The earlier version (section 6 of
stage3_roc_analysis.py) plotted six curves, two of which — chronological age and APACHE II —
belong to Figure 3 and were duplicated. This script narrows the figure to the comparison the
correction argument actually needs:

    Age Gap / Bias-corrected Age Gap
    FaceAge / Bias-corrected FaceAge
    + a random reference line

FaceAge is kept alongside Age Gap because the correction applies to both, and showing only Age
Gap would read as selective reporting. FaceAge barely moves (0.589 -> 0.597) while Age Gap moves
markedly (0.441 -> 0.488), which is the evidence that the correction removes the age dependence
of Age Gap without arbitrarily altering an unrelated marker.

Naming follows the manuscript and Table 5 — "Age Gap", "Bias-corrected Age Gap", "FaceAge",
"Bias-corrected FaceAge" — with no "(corrected)" suffix.

The correction is a quadratic fit by numpy.polyfit, cross-fitted in 10 folds stratified by
outcome, exactly as in stage3_roc_analysis.py, so every patient is corrected by a curve that was
not fitted on them.

Input  : data/merged_data.xlsx
Output : results/stage3_roc_analysis/roc_curves_corrected.png   (overwrites the superseded figure)"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))

from utils import load_from_merged_xlsx, RESULTS_DIR

print("=" * 80)
print("重绘 Figure 6：Age Gap 与 FaceAge 的偏差校正前后 ROC 曲线")
print("=" * 80)

output_dir = RESULTS_DIR / "stage3_roc_analysis"
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. 加载数据并做二次多项式偏差校正
# ============================================================================

print("\n[1/2] 加载数据并做二次多项式偏差校正（10 折交叉拟合）...")
df = load_from_merged_xlsx(max_occlusion_level=2)

valid = df['Age_Gap'].notna() & df['实际年龄'].notna()
if not valid.all():
    raise ValueError("Age_Gap 存在缺失，与手稿的完整例分析口径不一致")

age = df['实际年龄'].values.astype(float)
gap = df['Age_Gap'].values.astype(float)
y_lab = df['7天死亡'].values.astype(int)

# Age Gap ~ 实际年龄 的二次拟合，残差即校正后的 Age Gap。
# 校正曲线按死亡结局分层做 10 折交叉拟合：每折只用训练行拟合，
# 再对验证行取残差，使每位患者的校正值都来自一条"没见过自己"的曲线。
DEGREE, N_FOLDS, SEED = 2, 10, 42
gap_corrected = np.full(len(df), np.nan, dtype=float)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
for tr, te in skf.split(age, y_lab):
    coef = np.polyfit(age[tr], gap[tr], DEGREE)
    gap_corrected[te] = gap[te] - np.polyval(coef, age[te])
if np.isnan(gap_corrected).any():
    raise RuntimeError("交叉拟合未覆盖全部样本")

df['Age_Gap_corrected'] = gap_corrected
# FaceAge = 实际年龄 + Age Gap，故校正后的 FaceAge 由校正后的 Age Gap 推出
df['FaceAge_corrected'] = df['实际年龄'] + df['Age_Gap_corrected']

print(f"  校正前 Age Gap: {df['Age_Gap'].mean():+.2f} ± {df['Age_Gap'].std():.2f}")
print(f"  校正后 Age Gap: {df['Age_Gap_corrected'].mean():+.2f} ± {df['Age_Gap_corrected'].std():.2f}")
print(f"  校正前 FaceAge: {df['生物学年龄'].mean():.2f} ± {df['生物学年龄'].std():.2f}")
print(f"  校正后 FaceAge: {df['FaceAge_corrected'].mean():.2f} ± {df['FaceAge_corrected'].std():.2f}")

# ============================================================================
# 2. 绘制 Figure 6
# ============================================================================

print("\n[2/2] 绘制 Figure 6（Age Gap 与 FaceAge 原始 vs 校正）...")

y_true = df['7天死亡'].values

series = {
    'Age Gap': (df['Age_Gap'].values, '#1f77b4'),
    'Bias-corrected Age Gap': (df['Age_Gap_corrected'].values, '#2ca02c'),
    'FaceAge': (df['生物学年龄'].values, '#d62728'),
    'Bias-corrected FaceAge': (df['FaceAge_corrected'].values, '#9467bd'),
}

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random (AUC = 0.50)')

for name, (scores, color) in series.items():
    mask = ~(np.isnan(y_true) | np.isnan(scores))
    fpr, tpr, _ = roc_curve(y_true[mask], scores[mask])
    auc = roc_auc_score(y_true[mask], scores[mask])
    print(f"  {name}: AUC = {auc:.4f}")
    ax.plot(fpr, tpr, linewidth=2, color=color, label=f'{name} (AUC = {auc:.3f})')

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves Before and After Bias Correction\nAge Gap and FaceAge, 7-Day Mortality Prediction (N = 1,470)',
             fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()

save_path = output_dir / "roc_curves_corrected.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  [OK] 保存图表: {save_path}")
