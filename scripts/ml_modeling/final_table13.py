"""Three-scheme imputation comparison for Table 13 (superseded as the Table 13 source).

Written to decide how the eight bedside indicators should be imputed. It reproduces Table 13
under three schemes and reports the six metrics at a fixed 90%-specificity operating point,
with calibration intercept/slope derived by regressing the logit of the out-of-fold predictions
and with bootstrap confidence intervals.

  S0_asIs     merged_data.xlsx exactly as distributed (already median-filled across the full cohort)
  S1_foldMed  original blanks restored, then median-filled inside each training fold
  S2_foldMICE original blanks restored, then single conditional imputation inside each training
              fold (IterativeImputer with random-forest base learners)

Relationship to other scripts
-----------------------------
Superseded as the Table 13 producer by ml_final.py, which is the pipeline actually reported.
What remains live here is the S0 -> S1 comparison itself: the effect of moving the median from
the full cohort into the training folds, reported in Additional file 4, Table 3. The S2 column of
that table is the MICE result from ml_final.py (m = 5), which reproduces Table 13; it is *not*
this script's S2_foldMICE, whose single conditional imputation is a separate experiment.

This script's output also feeds bootstrap_ci_three_scheme.py (results/ml_modeling/final_table13.npz).

Input
-----
  data/merged_data.xlsx
  data/raw_groups/*.xlsx   (needed to restore the original blanks)

Output
------
  results/ml_modeling/final_table13.txt
  results/ml_modeling/final_table13.npz"""
from pathlib import Path

# 仓库根目录（scripts/<sub>/x.py -> parents[2]）
STUDY = Path(__file__).resolve().parents[2]

import os
import sys
import time

import numpy as np
import pandas as pd

os.chdir(str(STUDY))
sys.path.insert(0, 'scripts')
from utils import load_from_merged_xlsx  # noqa: E402

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa: E402
from sklearn.experimental import enable_iterative_imputer  # noqa: E402,F401
from sklearn.impute import IterativeImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (average_precision_score, confusion_matrix,  # noqa: E402
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from scipy.stats import norm  # noqa: E402

import warnings
warnings.filterwarnings('ignore')

QUICK = ['性别', '体温', '心率', '呼吸频率', '收缩压', '舒张压', '血氧饱和度', '血糖']
FEATURE_SETS = ['A_baseline', 'B_real_age', 'C_faceage', 'D_age_gap', 'E_age_gap_only']
LABELS = {'A_baseline': 'A (baseline)', 'B_real_age': 'B (+ age)',
          'C_faceage': 'C (+ FaceAge)', 'D_age_gap': 'D (+ age + Age Gap)',
          'E_age_gap_only': 'E (+ Age Gap)'}
PAIRS = [('B_real_age', 'A_baseline'), ('C_faceage', 'A_baseline'),
         ('D_age_gap', 'B_real_age'), ('C_faceage', 'B_real_age'),
         ('E_age_gap_only', 'A_baseline'), ('E_age_gap_only', 'C_faceage')]
N_REPEATS, N_FOLDS, N_BOOT, SEED = 5, 10, 2000, 42
SCHEMES = ['S0_asIs', 'S1_foldMed', 'S2_foldMICE']
if os.environ.get('SCHEME_FILTER'):
    SCHEMES = [s for s in SCHEMES if s in os.environ['SCHEME_FILTER'].split(',')]
METRICS = ['AUC', 'AUPRC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1']


# ------------------------------------------------------------------ helpers
def compute_midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float); T2[J] = T + 1
    return T2


def fast_delong(preds, m):
    n = preds.shape[1] - m; k = preds.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = compute_midrank(preds[r, :m])
        ty[r, :] = compute_midrank(preds[r, m:])
        tz[r, :] = compute_midrank(preds[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    return aucs, np.atleast_2d(cov)


def delong_test(y_true, s1, s2):
    order = np.argsort(-y_true); m = int(y_true.sum())
    aucs, cov = fast_delong(np.vstack([s1[order], s2[order]]), m)
    l = np.array([[1.0, -1.0]])
    var = float((l @ cov @ l.T)[0, 0])
    z = abs(aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    return float(aucs[0]), float(aucs[1]), float(2 * norm.sf(z))


def calib_ab(y_true, p):
    e = 1e-6
    lg = np.log(np.clip(p, e, 1 - e) / (1 - np.clip(p, e, 1 - e)))
    m = LogisticRegression(penalty='l2', C=1e6, solver='liblinear',
                           max_iter=1000).fit(lg.reshape(-1, 1), y_true)
    return float(m.intercept_[0]), float(m.coef_[0][0])


def boot_ci(y_true, p, fn, n_boot=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    pos = np.where(y_true == 1)[0]; neg = np.where(y_true == 0)[0]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), True),
                              rng.choice(neg, len(neg), True)])
        try:
            vals.append(fn(y_true[idx], p[idx]))
        except Exception:
            pass
    return np.percentile(vals, [2.5, 97.5])


def op_metrics(y_true, p, thr):
    tn, fp, fn, tp = confusion_matrix(y_true, (p >= thr).astype(int)).ravel()
    acc = (tp + tn) / len(y_true)
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return acc, sens, spec, f1


def build(name, quick, age, face, gap):
    if name == 'A_baseline':
        return quick
    if name == 'B_real_age':
        return np.column_stack([quick, age])
    if name == 'C_faceage':
        return np.column_stack([quick, face])
    if name == 'D_age_gap':
        return np.column_stack([quick, age, gap])
    if name == 'E_age_gap_only':
        return np.column_stack([quick, gap])
    raise ValueError(name)


def make_clf():
    cw = None                       # <== 当前配置：unweighted
    return {'RF': RandomForestClassifier(n_estimators=100, max_depth=10,
                                         min_samples_split=10, min_samples_leaf=5,
                                         class_weight=cw, random_state=42, n_jobs=-1),
            'LR': LogisticRegression(penalty='l2', C=1.0, class_weight=cw,
                                     solver='liblinear', max_iter=1000, random_state=42)}


def impute(scheme, Xtr, Xte):
    if scheme == 'S0_asIs':
        return Xtr, Xte
    if scheme == 'S1_foldMed':
        Ztr, Zte = Xtr.copy(), Xte.copy()
        for j in range(Xtr.shape[1]):
            v = np.nanmedian(Xtr[:, j])
            if np.isnan(v):
                v = 0.0
            Ztr[np.isnan(Ztr[:, j]), j] = v
            Zte[np.isnan(Zte[:, j]), j] = v
        return Ztr, Zte
    imp = IterativeImputer(estimator=RandomForestRegressor(n_estimators=50, random_state=0),
                           max_iter=10, random_state=0, sample_posterior=False)
    return imp.fit_transform(Xtr), imp.transform(Xte)


# ------------------------------------------------------------------ 数据
print('=' * 92)
print('当前配置（unweighted + raw + 90% 特异度工作点）下的插补敏感性')
print('=' * 92)
df = load_from_merged_xlsx(max_occlusion_level=2)
n = len(df)
y = df['7天死亡'].values.astype(int)
age = df['实际年龄'].values.astype(float)
gap_raw = df['Age_Gap'].values.astype(float)
face_raw = df['生物学年龄'].values.astype(float)
X = df[QUICK].values.astype(float)
print(f'  N = {n}，死亡 {int(y.sum())}')

# 还原被合并阶段中位数填充掉的原始缺失
raw = {}
XDIR = os.environ.get('FACEAGE_RAW_GROUPS_DIR', str(STUDY / 'data' / 'raw_groups'))
if not os.path.isdir(XDIR):
    raise SystemExit(
        f'未找到原始分组目录 {XDIR}。S1/S2 方案依赖它还原被中位数填充掉的缺失；'
        f'缺了它三个方案会完全相同，脚本没有意义。见 README 第 7 节。')
for f in sorted(os.listdir(XDIR)):
    if not f.endswith('.xlsx') or f.startswith('~'):
        continue
    d = pd.read_excel(os.path.join(XDIR, f))
    if '编号' not in d.columns:
        continue
    d = d.dropna(subset=['编号']).copy()
    d['_id'] = d['编号'].apply(lambda x: str(int(x)) if isinstance(x, float) else str(x).strip())
    raw[os.path.splitext(f)[0]] = d.set_index('_id')

Xr = X.copy()
n_restored = 0
for i, r in enumerate(df.itertuples(index=False)):
    grp = str(getattr(r, 'excel_file')).replace('.xlsx', '')
    pv = getattr(r, '编号')
    pid = str(int(pv)) if isinstance(pv, float) else str(pv)
    if grp in raw and pid in raw[grp].index:
        row = raw[grp].loc[pid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        for j, c in enumerate(QUICK):
            if c in row.index and pd.isna(row[c]):
                Xr[i, j] = np.nan
                n_restored += 1
print(f'  还原原始缺失格 {n_restored} 个（涉及患者 {int(np.isnan(Xr).any(axis=1).sum())} 人）')

# ------------------------------------------------------------------ 主循环
res = {s: {c: {m: {k: [] for k in METRICS} for m in FEATURE_SETS} for c in ('RF', 'LR')}
       for s in SCHEMES}
pred = {s: {c: {m: np.full(n, np.nan) for m in FEATURE_SETS} for c in ('RF', 'LR')}
        for s in SCHEMES}
cnt = {s: {c: {m: np.zeros(n) for m in FEATURE_SETS} for c in ('RF', 'LR')} for s in SCHEMES}
foldpred = {s: {c: {m: [] for m in FEATURE_SETS} for c in ('RF', 'LR')} for s in SCHEMES}

t0 = time.time()
for rep in range(N_REPEATS):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + rep)
    cur = {s: {c: {m: np.full(n, np.nan) for m in FEATURE_SETS} for c in ('RF', 'LR')}
           for s in SCHEMES}
    for k, (tr, te) in enumerate(skf.split(X, y), 1):
        gap_c, face_c = gap_raw, face_raw          # raw 模式：不做偏倚校正
        mats = {}
        if 'S0_asIs' in SCHEMES:
            mats['S0_asIs'] = (X[tr], X[te])
        if 'S1_foldMed' in SCHEMES:
            mats['S1_foldMed'] = impute('S1_foldMed', Xr[tr], Xr[te])
        if 'S2_foldMICE' in SCHEMES:
            mats['S2_foldMICE'] = impute('S2_foldMICE', Xr[tr], Xr[te])
        for m in FEATURE_SETS:
            for s in SCHEMES:
                Xtr, Xte = mats[s]
                A = build(m, Xtr, age[tr], face_c[tr], gap_c[tr])
                B = build(m, Xte, age[te], face_c[te], gap_c[te])
                sc = StandardScaler()
                A, B = sc.fit_transform(A), sc.transform(B)
                for cn, clf in make_clf().items():
                    clf.fit(A, y[tr])
                    p = clf.predict_proba(B)[:, 1]
                    cur[s][cn][m][te] = p
                    res[s][cn][m]['AUC'].append(roc_auc_score(y[te], p))
                    res[s][cn][m]['AUPRC'].append(average_precision_score(y[te], p))
                    foldpred[s][cn][m].append((y[te], p))
        print(f'    rep {rep + 1}/{N_REPEATS} fold {k}/{N_FOLDS}  ({time.time() - t0:.0f}s)',
              end='\r', flush=True)
    for s in SCHEMES:
        for cn in ('RF', 'LR'):
            for m in FEATURE_SETS:
                a = cur[s][cn][m]
                pred[s][cn][m] = np.where(np.isnan(pred[s][cn][m]), a, pred[s][cn][m] + a)
                cnt[s][cn][m] += (~np.isnan(a)).astype(float)
    print(f'  repeat {rep + 1}/{N_REPEATS} 完成  ({time.time() - t0:.0f}s)', flush=True)

# ------------------------------------------------------------------ 工作点与校准
summ = {s: {c: {m: {} for m in FEATURE_SETS} for c in ('RF', 'LR')} for s in SCHEMES}
for s in SCHEMES:
    print(f'  汇总 {s} ...', flush=True)
    for cn in ('RF', 'LR'):
        for m in FEATURE_SETS:
            pred[s][cn][m] = pred[s][cn][m] / cnt[s][cn][m]   # 5 次重复的 OOF 均值
            p = pred[s][cn][m]
            fpr_c, tpr_c, thr_c = roc_curve(y, p)
            ok = np.isfinite(thr_c) & (thr_c > 0) & (thr_c <= 1)
            fpr_c, tpr_c, thr_c = fpr_c[ok], tpr_c[ok], thr_c[ok]
            ok90 = (1 - fpr_c) >= 0.90
            thr_sp90 = float(thr_c[np.argmax(np.where(ok90, tpr_c, -1))])
            # 当前配置做法：阈值取自 OOF，但逐折施加、再对折取均值±SD
            mets = list(zip(*[op_metrics(yy, pp, thr_sp90) for yy, pp in foldpred[s][cn][m]]))
            ci, sl = calib_ab(y, p)
            ci_ci = boot_ci(y, p, lambda yy, pp: calib_ab(yy, pp)[0])
            sl_ci = boot_ci(y, p, lambda yy, pp: calib_ab(yy, pp)[1])
            d = summ[s][cn][m]
            d['AUC'] = (np.mean(res[s][cn][m]['AUC']), np.std(res[s][cn][m]['AUC']))
            d['AUPRC'] = (np.mean(res[s][cn][m]['AUPRC']), np.std(res[s][cn][m]['AUPRC']))
            d['Accuracy'], d['Accuracy_sd'] = np.mean(mets[0]), np.std(mets[0])
            d['Sensitivity'], d['Sensitivity_sd'] = np.mean(mets[1]), np.std(mets[1])
            d['Specificity'], d['Specificity_sd'] = np.mean(mets[2]), np.std(mets[2])
            d['F1'], d['F1_sd'] = np.mean(mets[3]), np.std(mets[3])
            d['CalInt'] = (ci, ci_ci); d['CalSlope'] = (sl, sl_ci)
            d['thr'] = thr_sp90

# ------------------------------------------------------------------ 输出
_OUT_DIR = STUDY / 'results' / 'ml_modeling'
_OUT_DIR.mkdir(parents=True, exist_ok=True)
out = open(str(_OUT_DIR / 'final_table13.txt'), 'w', encoding='utf-8')


def P(*a):
    print(*a)
    print(*a, file=out)


P()
P('=' * 120)
P('【A】当前配置 Table 13 复现核对（S0_asIs = merged_data.xlsx 原样，unweighted + raw + Sp90）')
P('=' * 120)
P(f'{"Clf":>4s} {"Set":<16s} {"AUC":>15s} {"AUPRC":>15s} {"Sens":>15s} {"Spec":>15s} '
  f'{"Acc":>15s} {"F1":>15s} {"Cal.Int (95%CI)":>26s} {"Cal.Slope (95%CI)":>26s}')


def row(d, cn, m):
    return (f'{cn:>4s} {LABELS[m]:<16s} '
            f'{d["AUC"][0]:.3f} ± {d["AUC"][1]:.3f} {d["AUPRC"][0]:.3f} ± {d["AUPRC"][1]:.3f} '
            f'{d["Sensitivity"]:.3f} ± {d["Sensitivity_sd"]:.3f}'.rjust(16)
            + f' {d["Specificity"]:.3f} ± {d["Specificity_sd"]:.3f}'.rjust(16)
            + f' {d["Accuracy"]:.3f} ± {d["Accuracy_sd"]:.3f}'.rjust(16)
            + f' {d["F1"]:.3f} ± {d["F1_sd"]:.3f}'.rjust(16)
            + f' {d["CalInt"][0]:+.2f} ({d["CalInt"][1][0]:+.2f}, {d["CalInt"][1][1]:+.2f})'.rjust(27)
            + f' {d["CalSlope"][0]:.2f} ({d["CalSlope"][1][0]:.2f}, {d["CalSlope"][1][1]:.2f})'.rjust(27))


for cn in ('RF', 'LR'):
    for m in FEATURE_SETS:
        P(row(summ['S0_asIs'][cn][m], cn, m))

for s, title in [('S1_foldMed', '【B】改为折内中位数插补'),
                 ('S2_foldMICE', '【C】改为折内 MICE（单次条件插补）')]:
    if s not in SCHEMES:
        continue
    P()
    P('=' * 120)
    P(f'{title}（unweighted + raw + Sp90）')
    P('=' * 120)
    P(f'{"Clf":>4s} {"Set":<16s} {"AUC":>15s} {"AUPRC":>15s} {"Sens":>15s} {"Spec":>15s} '
      f'{"Acc":>15s} {"F1":>15s} {"Cal.Int (95%CI)":>26s} {"Cal.Slope (95%CI)":>26s}')
    for cn in ('RF', 'LR'):
        for m in FEATURE_SETS:
            P(row(summ[s][cn][m], cn, m))

P()
P('=' * 120)
P('【D】ΔAUC：折内中位数 → 折内 MICE（S1 - S2），当前配置')
P('=' * 120)
if {'S1_foldMed', 'S2_foldMICE'} <= set(SCHEMES):
    for cn in ('RF', 'LR'):
        for m in FEATURE_SETS:
            a = summ['S1_foldMed'][cn][m]['AUC'][0]
            b = summ['S2_foldMICE'][cn][m]['AUC'][0]
            P(f'  {cn:>3s} {LABELS[m]:<20s} {a:.3f} -> {b:.3f}  Δ = {b - a:+.3f}')

P()
P('=' * 120)
P('【E】DeLong（对 5 次重复的 OOF 平均预测）—— 对应 Table 14')
P('=' * 120)
for s in SCHEMES:
    P(f'-- {s}')
    for cn in ('RF', 'LR'):
        for a, b in PAIRS:
            A, B, pv = delong_test(y, pred[s][cn][a], pred[s][cn][b])
            P(f'   {cn:>3s} {a.split("_")[0]} vs {b.split("_")[0]}: '
              f'{A:.3f} vs {B:.3f}  Δ {A - B:+.3f}  P = {pv:.3f}')

out.close()
print('\n完成，结果已写入 <repo>/results/ml_modeling/final_table13.txt')

# 落盘原始数组，便于复核
np.savez_compressed(
    str(STUDY / 'results' / 'ml_modeling' / 'final_table13.npz'),
    y=y,
    **{f'pred|{s}|{c}|{m}': pred[s][c][m] for s in SCHEMES for c in ('RF', 'LR')
       for m in FEATURE_SETS},
    **{f'met|{s}|{c}|{m}|{k}': np.array(res[s][c][m][k])
       for s in SCHEMES for c in ('RF', 'LR') for m in FEATURE_SETS for k in ('AUC', 'AUPRC')})
