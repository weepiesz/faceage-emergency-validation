"""Primary machine-learning pipeline: nine feature sets under repeated stratified cross-validation.

This is the authoritative script for the incremental-value analysis. It produces the performance
of feature sets A-I (Table 13), the pairwise AUC comparisons (Table 14), the inputs to the
calibration and decision-curve panels (Figure 6), and the achieved-precision figures (Table 19).

Design
------
  class_weight=None (unweighted); six metrics computed at a fixed 90%-specificity operating point
  5 x 10 repeated stratified cross-validation (seed 42)
  calibration intercept/slope with 2,000 stratified bootstrap resamples

Imputation scheme
-----------------
  MICE : primary     multiple imputation by chained equations (m = 5, random-forest base
                     learners) fitted inside the training folds; the m predicted probabilities
                     are averaged for each patient
  CC   : sensitivity complete cases only (all eight bedside indicators observed)

Variable mode
-------------
  raw       FaceAge / Age Gap as measured
  corrected quadratic bias correction refitted inside each training fold;
            FaceAge_corrected = chronological age + residual(FaceAge ~ age)

Feature sets
------------
  A bedside (8) | B A+chronological age | C A+FaceAge | D A+age+Age Gap | E A+Age Gap
  F APACHE II   | G F+FaceAge          | H F+Age Gap  | I F+FaceAge+Age Gap

Input
-----
  data/merged_data.xlsx
  data/raw_groups/*.xlsx   (optional; see "Recovering the original blanks" below)

Output
------
  results/ml_modeling/ml_final.txt    human-readable metric tables
  results/ml_modeling/ml_final.log    full console log
  results/ml_modeling/ml_final.npz    out-of-fold predictions (read by bootstrap_ci.py)

Recovering the original blanks
------------------------------
The MICE scheme needs the pre-imputation missingness, but merged_data.xlsx already carries
full-cohort medians in the eight bedside columns. The script therefore tries to recover the
original blanks from the per-patient source workbooks in FACEAGE_RAW_GROUPS_DIR (default
<repo>/data/raw_groups). If that directory does not exist the script still runs, but MICE then
has nothing to impute and its results equal the fold-internal median reference, not the MICE
scheme reported in the manuscript; the console warning says so. See README section 7.

Run order
---------
Run this script before 10_figures_mice.py and bootstrap_ci.py: both consume its output."""
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
SETS = ['A_baseline', 'B_real_age', 'C_faceage', 'D_age_gap', 'E_age_gap_only',
        'F_apache', 'G_apache_face', 'H_apache_gap', 'I_apache_face_gap']
LABELS = {'A_baseline': 'A (baseline 8 indicators)', 'B_real_age': 'B (A + age)',
          'C_faceage': 'C (A + FaceAge)', 'D_age_gap': 'D (A + age + Age Gap)',
          'E_age_gap_only': 'E (A + Age Gap)',
          'F_apache': 'F (APACHE II)', 'G_apache_face': 'G (APACHE II + FaceAge)',
          'H_apache_gap': 'H (APACHE II + Age Gap)',
          'I_apache_face_gap': 'I (APACHE II + FaceAge + Age Gap)'}
PAIRS = [('B_real_age', 'A_baseline'), ('C_faceage', 'A_baseline'),
         ('D_age_gap', 'B_real_age'), ('C_faceage', 'B_real_age'),
         ('E_age_gap_only', 'A_baseline'), ('E_age_gap_only', 'C_faceage'),
         ('F_apache', 'A_baseline'), ('G_apache_face', 'F_apache'),
         ('H_apache_gap', 'F_apache'), ('I_apache_face_gap', 'F_apache')]

N_REPEATS = int(os.environ.get('NREP', 5))
N_FOLDS, N_BOOT, SEED = 10, 2000, 42
M_IMP = int(os.environ.get('MIMP', 5))
SCHEMES = ['MICE', 'CC']
MODES = ['raw', 'corrected']
CLFS = ['RF', 'LR']

OUT_TXT = str(STUDY / 'results' / 'ml_modeling' / 'ml_final.txt')
OUT_NPZ = str(STUDY / 'results' / 'ml_modeling' / 'ml_final.npz')
_OUT_DIR = STUDY / 'results' / 'ml_modeling'
_OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG = open(str(_OUT_DIR / 'ml_final.log'), 'w', encoding='utf-8')


def L(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    print(s, file=LOG, flush=True)


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


def build(name, quick, age, face, gap, apache):
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
    if name == 'F_apache':
        return apache.reshape(-1, 1)
    if name == 'G_apache_face':
        return np.column_stack([apache, face])
    if name == 'H_apache_gap':
        return np.column_stack([apache, gap])
    if name == 'I_apache_face_gap':
        return np.column_stack([apache, face, gap])
    raise ValueError(name)


def make_rf():
    return RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=10,
                                  min_samples_leaf=5, class_weight=None,
                                  random_state=42, n_jobs=-1)


def make_lr():
    return LogisticRegression(penalty='l2', C=1.0, class_weight=None,
                              solver='liblinear', max_iter=1000, random_state=42)


def mice_mats(Xtr, Xte, m):
    """折内多重插补：m 个独立随机种子的 chained equations(RF)"""
    out = []
    for s in range(m):
        imp = IterativeImputer(estimator=RandomForestRegressor(n_estimators=30, max_depth=12,
                                                               random_state=s, n_jobs=-1),
                               max_iter=5, tol=1e-3, random_state=s, sample_posterior=False)
        out.append((imp.fit_transform(Xtr), imp.transform(Xte)))
    return out


# ------------------------------------------------------------------ 数据
L('=' * 100)
L('意见 6/7 ML 主流水线   unweighted + 90% 特异度工作点 + 折内多重插补')
L('=' * 100)
df = load_from_merged_xlsx(max_occlusion_level=2)
n_all = len(df)
y_all = df['7天死亡'].values.astype(int)
age_all = df['实际年龄'].values.astype(float)
gap_all = df['Age_Gap'].values.astype(float)
face_all = df['生物学年龄'].values.astype(float)
apa_all = df['APACHE评分'].values.astype(float)
X_all = df[QUICK].values.astype(float)
L(f'  分析集 N = {n_all}，死亡 {int(y_all.sum())}，死亡率 {y_all.mean():.3f}')
L(f'  APACHE II 缺失 {int(df["APACHE评分"].isna().sum())}')

# 还原被合并阶段中位数填充掉的原始缺失
raw = {}
XDIR = os.environ.get('FACEAGE_RAW_GROUPS_DIR', str(STUDY / 'data' / 'raw_groups'))
if os.path.isdir(XDIR):
    for f in sorted(os.listdir(XDIR)):
        if not f.endswith('.xlsx') or f.startswith('~'):
            continue
        d = pd.read_excel(os.path.join(XDIR, f))
        if '编号' not in d.columns:
            continue
        d = d.dropna(subset=['编号']).copy()
        d['_id'] = d['编号'].apply(lambda x: str(int(x)) if isinstance(x, float) else str(x).strip())
        raw[os.path.splitext(f)[0]] = d.set_index('_id')
    L(f'  载入原始 Excel 分组 {len(raw)} 个')
else:
    L(f'  !! 未找到原始分组目录 {XDIR}')
    L('  !! 无法还原合并阶段被中位数填充掉的缺失：MICE 将运行在已填充的数据上，')
    L('  !! 其结果等同于折内中位数方案，而非正文报告的 MICE。见 README 第 7 节。')

Xr = X_all.copy()
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
miss_mask = np.isnan(Xr).any(axis=1)
L(f'  还原原始缺失格 {n_restored} 个（涉及患者 {int(miss_mask.sum())} 人）')
L('  逐列缺失：' + '  '.join(f'{c}={int(np.isnan(Xr[:, j]).sum())}' for j, c in enumerate(QUICK)))
cc_mask = ~miss_mask
L(f'  完整病例 N = {int(cc_mask.sum())}（死亡 {int(y_all[cc_mask].sum())}）')
L('')

# ------------------------------------------------------------------ 主循环
store = {}                     # store[(scheme,mode,clf,set)] = dict(pred=,auc_folds=,fold_pred=)
t0 = time.time()

for scheme in SCHEMES:
    sel = cc_mask if scheme == 'CC' else np.ones(n_all, bool)
    n = int(sel.sum())
    y = y_all[sel]; age = age_all[sel]; gap = gap_all[sel]; face = face_all[sel]
    apa = apa_all[sel]; Xr_s = Xr[sel]
    L('-' * 100)
    L(f'【{scheme}】 N = {n}，死亡 {int(y.sum())}')
    L('-' * 100)

    pred = {(mo, cn, st): np.full(n, np.nan) for mo in MODES for cn in CLFS for st in SETS}
    cnt = {(mo, cn, st): np.zeros(n) for mo in MODES for cn in CLFS for st in SETS}
    foldpred = {(mo, cn, st): [] for mo in MODES for cn in CLFS for st in SETS}
    aucfolds = {(mo, cn, st): [] for mo in MODES for cn in CLFS for st in SETS}
    apfolds = {(mo, cn, st): [] for mo in MODES for cn in CLFS for st in SETS}
    aucfolds_m = {(mo, cn, st): [] for mo in MODES for cn in CLFS for st in SETS}

    for rep in range(N_REPEATS):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + rep)
        cur = {(mo, cn, st): np.full(n, np.nan) for mo in MODES for cn in CLFS for st in SETS}
        for k, (tr, te) in enumerate(skf.split(Xr_s, y), 1):
            nm = M_IMP if scheme == 'MICE' else 1
            mats = mice_mats(Xr_s[tr], Xr_s[te], nm) if scheme == 'MICE' else [(Xr_s[tr], Xr_s[te])]
            for mo in MODES:
                # accp 必须每个 mo 重新分配，且只含当前 mo 的键。
                # 若含全部键，后一个 mo 的迭代会把前一个 mo 的 cur 覆盖成全零。
                keys_m = [(mo, cn, st) for cn in CLFS for st in SETS]
                accp = {k: np.zeros(len(te)) for k in keys_m}
                if mo == 'raw':
                    gap_tr, gap_te, face_tr, face_te = gap[tr], gap[te], face[tr], face[te]
                else:
                    cg = np.polyfit(age[tr], gap[tr], 2)
                    gap_tr = gap[tr] - np.polyval(cg, age[tr])
                    gap_te = gap[te] - np.polyval(cg, age[te])
                    cf = np.polyfit(age[tr], face[tr], 2)
                    face_tr = age[tr] + (face[tr] - np.polyval(cf, age[tr]))
                    face_te = age[te] + (face[te] - np.polyval(cf, age[te]))

                for mm, (Xtr_i, Xte_i) in enumerate(mats):
                    for st in SETS:
                        A = build(st, Xtr_i, age[tr], face_tr, gap_tr, apa[tr])
                        B = build(st, Xte_i, age[te], face_te, gap_te, apa[te])
                        sc = StandardScaler()
                        A, B = sc.fit_transform(A), sc.transform(B)
                        for cn, clf in (('RF', make_rf()), ('LR', make_lr())):
                            clf.fit(A, y[tr])
                            p = clf.predict_proba(B)[:, 1]
                            accp[(mo, cn, st)] += p
                            if nm > 1:
                                aucfolds_m[(mo, cn, st)].append(roc_auc_score(y[te], p))
                for key in keys_m:
                    accp[key] /= nm
                    cur[key][te] = accp[key]
                    foldpred[key].append((y[te], accp[key]))

            if k % 5 == 0 or k == N_FOLDS:
                L(f'    {scheme} rep {rep+1}/{N_REPEATS} fold {k}/{N_FOLDS}  ({time.time()-t0:.0f}s)')

        for key in pred:
            a = cur[key]
            pred[key] = np.where(np.isnan(pred[key]), a, pred[key] + a)
            cnt[key] += (~np.isnan(a)).astype(float)
        # 逐折 AUC / AUPRC（取本次 repeat 的最后 N_FOLDS 折）
        for mo in MODES:
            for cn in CLFS:
                for st in SETS:
                    for j in range(N_FOLDS):
                        yy, pp = foldpred[(mo, cn, st)][-(N_FOLDS - j)]
                        aucfolds[(mo, cn, st)].append(roc_auc_score(yy, pp))
                        apfolds[(mo, cn, st)].append(average_precision_score(yy, pp))
        L(f'  {scheme} repeat {rep+1}/{N_REPEATS} 完成  ({time.time()-t0:.0f}s)')

    for mo in MODES:
        for cn in CLFS:
            for st in SETS:
                key = (mo, cn, st)
                pred[key] = pred[key] / cnt[key]
                store[(scheme, mo, cn, st)] = {
                    'pred': pred[key],
                    'auc_folds': np.array(aucfolds[key]),
                    'ap_folds': np.array(apfolds[key]),
                    'auc_m': np.array(aucfolds_m[key]) if aucfolds_m[key] else np.array([]),
                    'fold_pred': foldpred[key],
                }
    L(f'  【{scheme}】汇总完成  ({time.time()-t0:.0f}s)')

L('')
L(f'总耗时 {time.time()-t0:.0f}s')

# ------------------------------------------------------------------ 先存档
# CV 结果先落盘，避免后面汇总阶段的任何异常导致整轮重跑
np.savez_compressed(
    OUT_NPZ, y_all=y_all, cc_mask=cc_mask, Xr=Xr,
    **{f'pred|{k[0]}|{k[1]}|{k[2]}|{k[3]}': v['pred'] for k, v in store.items()})
L(f'已写出 {OUT_NPZ}（汇总前先存）')

# ------------------------------------------------------------------ 汇总指标
summary = {}
L('')
L('=' * 130)
L('六项指标 @ 固定 90% 特异度工作点（阈值取自 OOF，逐折施加后取均值±SD）')
L('=' * 130)
hdr = (f'{"Sch":>5s} {"Mode":>10s} {"Clf":>4s} {"Set":<26s} {"AUC":>15s} {"AUPRC":>15s} '
       f'{"Sens":>13s} {"Spec":>13s} {"Acc":>13s} {"F1":>13s} {"CalInt (95%CI)":>24s} {"CalSlope (95%CI)":>24s}')
for scheme in SCHEMES:
    L('')
    L(hdr)
    for mo in MODES:
        for cn in CLFS:
            for st in SETS:
                d = store[(scheme, mo, cn, st)]
                n = len(d['pred'])
                sel = cc_mask if scheme == 'CC' else np.ones(n_all, bool)
                y = y_all[sel]
                p = d['pred']
                fpr_c, tpr_c, thr_c = roc_curve(y, p)
                ok = np.isfinite(thr_c) & (thr_c > 0) & (thr_c <= 1)
                fpr_c, tpr_c, thr_c = fpr_c[ok], tpr_c[ok], thr_c[ok]
                ok90 = (1 - fpr_c) >= 0.90
                if not ok90.any():
                    L(f'!! {scheme}/{mo}/{cn}/{st} 无特异度≥90% 的截点，跳过')
                    continue
                thr = float(thr_c[np.argmax(np.where(ok90, tpr_c, -1))])
                mets = list(zip(*[op_metrics(yy, pp, thr) for yy, pp in d['fold_pred']]))
                ci, sl = calib_ab(y, p)
                ci_ci = boot_ci(y, p, lambda yy, pp: calib_ab(yy, pp)[0])
                sl_ci = boot_ci(y, p, lambda yy, pp: calib_ab(yy, pp)[1])
                summary[(scheme, mo, cn, st)] = dict(
                    auc=(d['auc_folds'].mean(), d['auc_folds'].std()),
                    auprc=(d['ap_folds'].mean(), d['ap_folds'].std()),
                    sens=(np.mean(mets[1]), np.std(mets[1])),
                    spec=(np.mean(mets[2]), np.std(mets[2])),
                    acc=(np.mean(mets[0]), np.std(mets[0])),
                    f1=(np.mean(mets[3]), np.std(mets[3])),
                    thr=thr, ci=ci, ci_ci=ci_ci, sl=sl, sl_ci=sl_ci)
                s = summary[(scheme, mo, cn, st)]
                L(f'{scheme:>5s} {mo:>10s} {cn:>4s} {LABELS[st]:<26s} '
                  f'{s["auc"][0]:>7.3f}±{s["auc"][1]:.3f} {s["auprc"][0]:>7.3f}±{s["auprc"][1]:.3f} '
                  f'{s["sens"][0]:>6.3f}±{s["sens"][1]:.3f} {s["spec"][0]:>6.3f}±{s["spec"][1]:.3f} '
                  f'{s["acc"][0]:>6.3f}±{s["acc"][1]:.3f} {s["f1"][0]:>6.3f}±{s["f1"][1]:.3f} '
                  f'{s["ci"]:>+6.2f} ({s["ci_ci"][0]:+.2f}, {s["ci_ci"][1]:+.2f}) '
                  f'{s["sl"]:>6.2f} ({s["sl_ci"][0]:+.2f}, {s["sl_ci"][1]:+.2f})')

# ------------------------------------------------------------------ DeLong
L('')
L('=' * 130)
L('DeLong 配对检验（OOF 预测）')
L('=' * 130)
delong = {}
for scheme in SCHEMES:
    sel = cc_mask if scheme == 'CC' else np.ones(n_all, bool)
    y = y_all[sel]
    for mo in MODES:
        L('')
        L(f'-- {scheme} / {mo}')
        L(f'{"Clf":>4s} {"对比":<34s} {"AUC_新":>7s} {"AUC_基":>7s} {"dAUC":>8s} {"P(DeLong)":>11s}')
        for cn in CLFS:
            for a, b in PAIRS:
                pa = store[(scheme, mo, cn, a)]['pred']
                pb = store[(scheme, mo, cn, b)]['pred']
                A_, B_, pv = delong_test(y, pa, pb)
                delong[(scheme, mo, cn, a, b)] = (A_ - B_, pv)
                L(f'{cn:>4s} {LABELS[a]+" vs "+LABELS[b]:<34s} {A_:>7.3f} {B_:>7.3f} '
                  f'{A_-B_:>+8.3f} {pv:>11.4f}')

# ------------------------------------------------------------------ 存盘
np.savez_compressed(
    OUT_NPZ, y_all=y_all, cc_mask=cc_mask, Xr=Xr,
    **{f'pred|{k[0]}|{k[1]}|{k[2]}|{k[3]}': v['pred'] for k, v in store.items()})
buf = []
for scheme in SCHEMES:
    for mo in MODES:
        for cn in CLFS:
            for st in SETS:
                s = summary[(scheme, mo, cn, st)]
                buf.append(f'{scheme}|{mo}|{cn}|{st}|AUC={s["auc"][0]:.4f}|SD={s["auc"][1]:.4f}')
open(OUT_TXT, 'w', encoding='utf-8').write('\n'.join(buf))
L('')
L(f'已写出 {OUT_NPZ}')
L(f'已写出 {OUT_TXT}')
L('DONE')
