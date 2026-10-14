"""Paired-bootstrap 95% confidence intervals for the key AUC contrasts.

Reads the out-of-fold predictions saved by ml_final.py. For each contrast the patients are
resampled with replacement 2,000 times and the difference in AUC is recomputed within each
resample on the same patients (paired), giving a percentile interval. Seed: RandomState(20251013).

Eight contrasts are reported: the four feature-set contrasts quoted in the Results (chronological
age vs bedside baseline; FaceAge vs chronological age; APACHE II vs bedside baseline; FaceAge,
Age Gap and both markers vs APACHE II) and two bias-corrected versions of FaceAge vs
chronological age.

Input  : results/ml_modeling/ml_final.npz     (run ml_final.py first)
Output : printed to stdout — one line per contrast with the delta AUC and its 95% CI"""
from pathlib import Path

# 仓库根目录（scripts/<sub>/x.py -> parents[2]）
STUDY = Path(__file__).resolve().parents[2]

import numpy as np
from sklearn.metrics import roc_auc_score

d = np.load(str(STUDY / 'results' / 'ml_modeling' / 'ml_final.npz'), allow_pickle=True)
y = d['y_all']
N = len(y)
rng = np.random.RandomState(20251013)


def oof(mode, clf, s):
    return d['pred|MICE|%s|%s|%s' % (mode, clf, s)]


def dci(mode_a, clf, sa, sb):
    """ΔAUC(new - base) 的点估计与配对 bootstrap CI。"""
    A = oof(mode_a, clf, sa)
    B = oof(mode_a, clf, sb)
    pt = roc_auc_score(y, A) - roc_auc_score(y, B)
    bs = np.empty(2000)
    for i in range(2000):
        idx = rng.randint(0, N, N)
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == N:
            bs[i] = np.nan
            continue
        bs[i] = roc_auc_score(yy, A[idx]) - roc_auc_score(yy, B[idx])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return pt, lo, hi


CMP = [
    ('RF  B vs A', 'raw', 'RF', 'B_real_age', 'A_baseline'),
    ('RF  C vs B', 'raw', 'RF', 'C_faceage', 'B_real_age'),
    ('LR  F vs A', 'raw', 'LR', 'F_apache', 'A_baseline'),
    ('LR  G vs F', 'raw', 'LR', 'G_apache_face', 'F_apache'),
    ('LR  H vs F', 'raw', 'LR', 'H_apache_gap', 'F_apache'),
    ('LR  I vs F', 'raw', 'LR', 'I_apache_face_gap', 'F_apache'),
    ('RF  C vs B (校正后)', 'corrected', 'RF', 'C_faceage', 'B_real_age'),
    ('LR  C vs B (校正后)', 'corrected', 'LR', 'C_faceage', 'B_real_age'),
]
print('%-24s %9s   %s' % ('对比', 'dAUC', '95% CI (配对 bootstrap)'))
for nm, mode, clf, sa, sb in CMP:
    pt, lo, hi = dci(mode, clf, sa, sb)
    print('%-24s %+9.4f   %+.4f to %+.4f' % (nm, pt, lo, hi))
