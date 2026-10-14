"""Paired-bootstrap 95% confidence intervals for the incremental value of the facial markers.

Resamples the out-of-fold predictions saved by final_table13.py and reports, for each of three
imputation schemes, the delta AUC of one feature set against another on the same resampled
patients (2,000 resamples, seed RandomState(42)).

Contrasts: chronological age vs bedside baseline; FaceAge vs bedside baseline; age + Age Gap vs
age; FaceAge vs age; Age Gap vs baseline; Age Gap vs FaceAge. The point of the exercise is to
show how tight the null is around the increments — that is, how much of an increment the data
can exclude.

Relationship to other scripts
-----------------------------
Superseded for the reported confidence intervals by bootstrap_ci.py, which works on the ml_final.py
pipeline (two schemes) rather than final_table13.py (three schemes). Kept because it is the
three-scheme version of the same robustness check.

Input  : results/ml_modeling/final_table13.npz     (run final_table13.py first)
Output : printed to stdout"""
from pathlib import Path

# 仓库根目录（scripts/<sub>/x.py -> parents[2]）
STUDY = Path(__file__).resolve().parents[2]

import numpy as np
from sklearn.metrics import roc_auc_score

Z = np.load(str(STUDY / 'results' / 'ml_modeling' / 'final_table13.npz'))
y = Z['y'].astype(int)
SCHEMES = ['S0_asIs', 'S1_foldMed', 'S2_foldMICE']
CLFS = ['RF', 'LR']
PAIRS = [('B_real_age', 'A_baseline', '年龄 vs 基线'),
         ('C_faceage', 'A_baseline', 'FaceAge vs 基线'),
         ('D_age_gap', 'B_real_age', '(+)年龄+AgeGap vs (+)年龄'),
         ('C_faceage', 'B_real_age', 'FaceAge vs 年龄'),
         ('E_age_gap_only', 'A_baseline', 'Age Gap vs 基线'),
         ('E_age_gap_only', 'C_faceage', 'Age Gap vs FaceAge')]
N_BOOT = 2000

pos = np.where(y == 1)[0]
neg = np.where(y == 0)[0]
rng = np.random.RandomState(42)
IDX = [np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
       for _ in range(N_BOOT)]

print('=' * 104)
print('ΔAUC 的配对 bootstrap 95% CI （正 = 前一个模型更好；区间跨 0 = 无证据）')
print('  "可排除的增量" = 区间上界，即我们能把大于该值的增量排除掉')
print('=' * 104)
for s in SCHEMES:
    print()
    print('-' * 104)
    print(f'-- {s}')
    print('-' * 104)
    hdr = (f'{"Clf":>4s} {"对比":<28s} {"AUC_新":>7s} {"AUC_基":>7s} {"ΔAUC":>8s} '
           f'{"95% CI":>20s} {"是否跨0":>8s} {"可排除的增量>":>14s}')
    print(hdr)
    for cn in CLFS:
        for a, b, name in PAIRS:
            pa = Z[f'pred|{s}|{cn}|{a}']
            pb = Z[f'pred|{s}|{cn}|{b}']
            A = roc_auc_score(y, pa); B = roc_auc_score(y, pb)
            d = np.array([roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i])
                          for i in IDX])
            lo, hi = np.percentile(d, [2.5, 97.5])
            span = '是' if lo <= 0 <= hi else '否'
            print(f'{cn:>4s} {name:<28s} {A:>7.3f} {B:>7.3f} {A - B:>+8.3f} '
                  f'[{lo:+.3f}, {hi:+.3f}]'.rjust(22) + f' {span:>8s} {hi:>+14.3f}')
