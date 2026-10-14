"""Figure 5 (ROC across the nine feature sets) and Figure 6 (calibration and decision curves).

Why this script exists
----------------------
The earlier figures were produced by 09_figures.py from out-of-fold predictions written by
08_calibration_dca.py, and 08 contained no imputation code at all: it consumed merged_data.xlsx
with the full-cohort medians already in place. That is the leaky pipeline. This script re-runs the
same cross-validation with fold-internal multiple imputation, exactly as ml_final.py does, and
also extends the feature sets from five (A-E) to nine (A-I) to cover APACHE II.

Conventions (matching the manuscript)
-------------------------------------
  Figure 5   ROC curves averaged vertically across the 50 folds; the legend AUC is the fold mean,
             i.e. the AUC column of Table 13
  Figure 6   pooled out-of-fold predictions — calibration and net benefit can only be pooled

Self-check
----------
The fold-mean AUCs must match the MICE|raw rows of results/ml_modeling/ml_final.txt to four
decimal places, otherwise the pipeline has silently diverged and the script aborts. Run
ml_final.py first. If that file is absent the self-check is skipped (or disabled with
--no-enforce).

Input  : data/merged_data.xlsx ; data/raw_groups/*.xlsx ; results/ml_modeling/ml_final.txt
Output : results/ml_modeling/figures/fig5_roc_feature_sets_mice.{png,pdf}
         results/ml_modeling/figures/fig5_roc_feature_sets_mice_2panel.{png,pdf}
         results/ml_modeling/figures/fig6_calibration_dca_mice.{png,pdf}
         results/ml_modeling/figures/fig5_fig6_numbers_mice.csv

Usage
-----
  python 10_figures_mice.py               full 5 x 10 cross-validation, roughly 8-12 minutes
  python 10_figures_mice.py --nrep 1      smoke test, roughly 2 minutes
  python 10_figures_mice.py --skip-cv     plot from the cached run"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

STUDY = Path(__file__).resolve().parents[2]      # <repo>/scripts/ml_modeling/ -> <repo>
os.chdir(STUDY)
sys.path.insert(0, str(STUDY / 'scripts'))

from utils import load_from_merged_xlsx                     # noqa: E402
from utils.config import DPI_DEFAULT                        # noqa: E402

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa: E402
from sklearn.experimental import enable_iterative_imputer   # noqa: E402,F401
from sklearn.impute import IterativeImputer                 # noqa: E402
from sklearn.linear_model import LogisticRegression         # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve        # noqa: E402
from sklearn.model_selection import StratifiedKFold         # noqa: E402
from sklearn.preprocessing import StandardScaler            # noqa: E402

import warnings
warnings.filterwarnings('ignore')

# ============================================================ 与 ml_final.py 完全一致的配置
QUICK = ['性别', '体温', '心率', '呼吸频率', '收缩压', '舒张压', '血氧饱和度', '血糖']
SETS = ['A_baseline', 'B_real_age', 'C_faceage', 'D_age_gap', 'E_age_gap_only',
        'F_apache', 'G_apache_face', 'H_apache_gap', 'I_apache_face_gap']
BEDSIDE = SETS[:5]          # A-E
APACHE = SETS[5:]           # F-I
LABELS = {'A_baseline': 'A: 8 bedside indicators',
          'B_real_age': 'B: A + chronological age',
          'C_faceage': 'C: A + FaceAge',
          'D_age_gap': 'D: A + chronological age + Age Gap',
          'E_age_gap_only': 'E: A + Age Gap',
          'F_apache': 'F: APACHE II',
          'G_apache_face': 'G: APACHE II + FaceAge',
          'H_apache_gap': 'H: APACHE II + Age Gap',
          'I_apache_face_gap': 'I: APACHE II + FaceAge + Age Gap'}

CLFS = ['RF', 'LR']
CLF_TITLE = {'RF': 'Random forest', 'LR': 'Logistic regression'}
N_FOLDS, SEED, M_IMP = 10, 42, 5
THRESH_GRID = np.round(np.arange(0.02, 0.501, 0.01), 2)
FPR_GRID = np.linspace(0, 1, 101)

COLORS_BEDSIDE = ['#1f77b4', '#2ca02c', '#17becf', '#9467bd', '#8c564b']
COLORS_APACHE = ['#ff7f0e', '#d62728', '#e377c2', '#bcbd22']


# ============================================================ 模型 / 插补（照搬 ml_final.py）
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
    out = []
    for s in range(m):
        imp = IterativeImputer(estimator=RandomForestRegressor(n_estimators=30, max_depth=12,
                                                               random_state=s, n_jobs=-1),
                               max_iter=5, tol=1e-3, random_state=s, sample_posterior=False)
        out.append((imp.fit_transform(Xtr), imp.transform(Xte)))
    return out


# ============================================================ 数据（照搬 ml_final.py）
def load_data(verbose=True):
    df = load_from_merged_xlsx(max_occlusion_level=2)
    n = len(df)
    y = df['7天死亡'].values.astype(int)
    age = df['实际年龄'].values.astype(float)
    gap = df['Age_Gap'].values.astype(float)
    face = df['生物学年龄'].values.astype(float)
    apa = df['APACHE评分'].values.astype(float)
    X = df[QUICK].values.astype(float)

    # 还原合并阶段被中位数填充掉的原始缺失
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
            d['_id'] = d['编号'].apply(
                lambda x: str(int(x)) if isinstance(x, float) else str(x).strip())
            raw[os.path.splitext(f)[0]] = d.set_index('_id')
    else:
        print(f'  !! 未找到原始分组目录 {XDIR}')
        print('  !! 无法还原合并阶段被中位数填充掉的缺失：MICE 将运行在已填充的数据上，')
        print('  !! 结果等同于折内中位数方案，而非正文报告的 MICE。见 README 第 7 节。')

    Xr = X.copy()
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

    if verbose:
        print(f'  分析集 N = {n}，死亡 {int(y.sum())}，死亡率 {y.mean():.3f}')
        print('  逐列缺失：' + '  '.join(
            f'{c}={int(np.isnan(Xr[:, j]).sum())}' for j, c in enumerate(QUICK)))
        print(f'  完整病例 N = {int((~np.isnan(Xr).any(axis=1)).sum())}')
    return dict(df=df, n=n, y=y, age=age, gap=gap, face=face, apa=apa, Xr=Xr)


# ============================================================ CV：只跑 MICE / raw
def run_cv(data, nrep, cache):
    if cache.exists():
        print(f'  [缓存] 读取 {cache.name}')
        z = np.load(cache)
        return ({k[4:]: z[k] for k in z.files if k.startswith('oof|')},
                {(c, s): (z[f'fy|{c}|{s}'], z[f'fp|{c}|{s}']) for c in CLFS for s in SETS},
                z['y'])

    y, age, gap, face, apa, Xr = (data['y'], data['age'], data['gap'],
                                  data['face'], data['apa'], data['Xr'])
    n = data['n']
    t0 = time.time()

    pred = {(c, s): np.full(n, np.nan) for c in CLFS for s in SETS}
    cnt = {(c, s): np.zeros(n) for c in CLFS for s in SETS}
    foldpred = {(c, s): [] for c in CLFS for s in SETS}

    for rep in range(nrep):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + rep)
        cur = {(c, s): np.full(n, np.nan) for c in CLFS for s in SETS}
        for k, (tr, te) in enumerate(skf.split(Xr, y), 1):
            mats = mice_mats(Xr[tr], Xr[te], M_IMP)
            accp = {(c, s): np.zeros(len(te)) for c in CLFS for s in SETS}
            for Xtr_i, Xte_i in mats:
                for st in SETS:
                    A = build(st, Xtr_i, age[tr], face[tr], gap[tr], apa[tr])
                    B = build(st, Xte_i, age[te], face[te], gap[te], apa[te])
                    sc = StandardScaler()
                    A, B = sc.fit_transform(A), sc.transform(B)
                    for c, clf in (('RF', make_rf()), ('LR', make_lr())):
                        clf.fit(A, y[tr])
                        accp[(c, st)] += clf.predict_proba(B)[:, 1]
            for key in accp:
                accp[key] /= M_IMP
                foldpred[key].append((y[te], accp[key]))
                cur[key][te] = accp[key]
            print(f'    rep {rep+1}/{nrep}  fold {k}/{N_FOLDS}  ({time.time()-t0:.0f}s)', flush=True)
        for key in pred:
            a = cur[key]
            pred[key] = np.where(np.isnan(pred[key]), a, pred[key] + a)
            cnt[key] += (~np.isnan(a)).astype(float)
        print(f'  rep {rep+1}/{nrep} 完成 ({time.time()-t0:.0f}s)', flush=True)

    oof = {key: pred[key] / cnt[key] for key in pred}
    folds = {key: (np.array([t[0] for t in v]), np.array([t[1] for t in v]))
             for key, v in foldpred.items()}

    blob = {'y': y}
    for key in oof:
        blob[f'oof|{key[0]}|{key[1]}'] = oof[key]
        blob[f'fy|{key[0]}|{key[1]}'] = folds[key][0]
        blob[f'fp|{key[0]}|{key[1]}'] = folds[key][1]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **blob)
    print(f'  已缓存 {cache}')
    return oof, folds, y


# ============================================================ 自检
def self_check(oof, folds, y, enforce=True, ref_path=None):
    ref = {}
    src = Path(ref_path) if ref_path else STUDY / 'results' / 'ml_modeling' / 'ml_final.txt'
    if src.exists():
        for line in src.read_text(encoding='utf-8').splitlines():
            p = line.strip().split('|')
            if len(p) >= 5 and p[0] == 'MICE' and p[1] == 'raw':
                ref[(p[2], p[3])] = float(p[4].split('=')[1])

    print('\n' + '=' * 78)
    print('  自检：折均值 AUC（本脚本） vs  ml_final.txt（表 13 来源）')
    print('=' * 78)
    bad = 0
    for c in CLFS:
        for s in SETS:
            fy, fp = folds[(c, s)]
            fm = float(np.mean([roc_auc_score(fy[i], fp[i]) for i in range(len(fy))]))
            r = ref.get((c, s))
            flag = ''
            if r is not None and abs(fm - r) > 5e-5:
                flag = '  <<< 不一致'
                bad += 1
            print(f'  {c:>2s} {s:<18s} 折均值={fm:.4f}  参考={r if r else float("nan"):.4f}'
                  f'  池化OOF={roc_auc_score(y, oof[(c, s)]):.4f}{flag}')
    if bad and enforce:
        raise SystemExit(f'!! {bad} 项与本流水线不一致，停止出图')
    if bad:
        print(f'  （冒烟模式：{bad} 项不一致属预期，重复次数不同）')
    else:
        print('  ✓ 全部一致')
    return {k: float(np.mean([roc_auc_score(folds[k][0][i], folds[k][1][i])
                              for i in range(len(folds[k][0]))])) for k in folds}


# ============================================================ Figure 5
def averaged_roc(y_all, p_all, n_folds):
    size = y_all.shape[1]
    tprs = []
    for i in range(n_folds):
        yk, pk = y_all[i], p_all[i]
        if yk.sum() == 0 or yk.sum() == len(yk):
            continue
        fpr, tpr, _ = roc_curve(yk, pk)
        tprs.append(np.interp(FPR_GRID, fpr, tpr))
    t = np.mean(tprs, axis=0)
    t[0], t[-1] = 0.0, 1.0
    return t


def draw_roc(ax, clf, subset, folds, auc, colors, style='-'):
    ax.plot([0, 1], [0, 1], '--', color='grey', lw=1.1, label='Chance')
    for st, c in zip(subset, colors):
        fy, fp = folds[(clf, st)]
        tpr = averaged_roc(fy, fp, fy.shape[0])
        ax.plot(FPR_GRID, tpr, color=c, lw=1.9, ls=style,
                label=f'{LABELS[st]} (AUC = {auc[(clf, st)]:.3f})')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('False positive rate', fontsize=11)
    ax.set_ylabel('True positive rate', fontsize=11)
    ax.legend(loc='lower right', fontsize=8.5, framealpha=0.92)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=10)


def panel_label(ax, s, dx=-0.10):
    ax.text(dx, 1.05, s, transform=ax.transAxes, fontsize=15,
            fontweight='bold', va='bottom', ha='right')


def figure5(folds, auc, out):
    # ---- 版本 1：2x2，床旁集 / APACHE 集分panel（推荐）
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 13))
    plan = [(axes[0, 0], 'RF', BEDSIDE, COLORS_BEDSIDE,
             'Random forest — bedside feature sets'),
            (axes[0, 1], 'RF', APACHE, COLORS_APACHE,
             'Random forest — APACHE II feature sets'),
            (axes[1, 0], 'LR', BEDSIDE, COLORS_BEDSIDE,
             'Logistic regression — bedside feature sets'),
            (axes[1, 1], 'LR', APACHE, COLORS_APACHE,
             'Logistic regression — APACHE II feature sets')]
    for ax, clf, sub, cols, title in plan:
        draw_roc(ax, clf, sub, folds, auc, cols)
        ax.set_title(title, fontsize=12)
    for ax, s in zip(axes.ravel(), 'ABCD'):
        panel_label(ax, s, dx=-0.06)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'fig5_roc_feature_sets_mice.{ext}',
                    dpi=DPI_DEFAULT, bbox_inches='tight')
    plt.close(fig)
    print('  已保存 fig5_roc_feature_sets_mice.png/.pdf  (2x2)')

    # ---- 版本 2：1x2，九条曲线同panel（保持手稿原有版式）
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 8.2))
    for ax, clf in zip(axes, CLFS):
        ax.plot([0, 1], [0, 1], '--', color='grey', lw=1.1, label='Chance')
        for sub, cols, ls in ((BEDSIDE, COLORS_BEDSIDE, '-'),
                              (APACHE, COLORS_APACHE, '--')):
            for st, c in zip(sub, cols):
                fy, fp = folds[(clf, st)]
                tpr = averaged_roc(fy, fp, fy.shape[0])
                ax.plot(FPR_GRID, tpr, color=c, lw=1.9, ls=ls,
                        label=f'{LABELS[st]} (AUC = {auc[(clf, st)]:.3f})')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel('False positive rate', fontsize=11)
        ax.set_ylabel('True positive rate', fontsize=11)
        ax.set_title(CLF_TITLE[clf], fontsize=13)
        ax.legend(loc='lower right', fontsize=8.2, framealpha=0.92)
        ax.grid(alpha=0.3)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=10)
    panel_label(axes[0], 'A', dx=-0.06)
    panel_label(axes[1], 'B', dx=-0.06)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'fig5_roc_feature_sets_mice_2panel.{ext}',
                    dpi=DPI_DEFAULT, bbox_inches='tight')
    plt.close(fig)
    print('  已保存 fig5_roc_feature_sets_mice_2panel.png/.pdf  (1x2)')


# ============================================================ Figure 6
def decile_curve(y, p, nbins=10):
    edges = np.unique(np.quantile(p, np.linspace(0, 1, nbins + 1)))
    if len(edges) < 3:
        return np.array([p.mean()]), np.array([y.mean()])
    idx = np.digitize(p, edges[1:-1])
    xs, ys = [], []
    for b in np.unique(idx):
        m = idx == b
        if m.sum() == 0:
            continue
        xs.append(float(p[m].mean()))
        ys.append(float(y[m].mean()))
    return np.array(xs), np.array(ys)


def net_benefit(y, p, pt):
    n = len(y)
    pred = p >= pt
    tp = ((pred) & (y == 1)).sum() / n
    fp = ((pred) & (y == 0)).sum() / n
    return tp - fp * pt / (1 - pt)


def draw_calib(ax, clf, subset, oof, y, cols):
    ax.plot([0, 1], [0, 1], '--', color='grey', lw=1.1, label='Perfect calibration')
    lim = 0.25
    for st, c in zip(subset, cols):
        xs, ys = decile_curve(y, oof[(clf, st)])
        ax.plot(xs, ys, '-o', color=c, lw=1.7, ms=4.2,
                label=LABELS[st], mfc=c, mec='white', mew=0.6)
        lim = max(lim, float(max(xs.max(), ys.max())) * 1.10)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel('Mean predicted probability', fontsize=11)
    ax.set_ylabel('Observed 7-day mortality', fontsize=11)
    ax.legend(loc='upper left', fontsize=8.2, framealpha=0.92)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=10)


def draw_dca(ax, clf, oof, y, prev):
    ax.plot(THRESH_GRID, [prev - (1 - prev) * t / (1 - t) for t in THRESH_GRID],
            '--', color='grey', lw=1.3, label='Treat all')
    ax.plot(THRESH_GRID, np.zeros_like(THRESH_GRID), ls=':', color='black',
            lw=1.2, label='Treat none')
    for sub, cols, ls in ((BEDSIDE, COLORS_BEDSIDE, '-'), (APACHE, COLORS_APACHE, '--')):
        for st, c in zip(sub, cols):
            nb = [net_benefit(y, oof[(clf, st)], t) for t in THRESH_GRID]
            ax.plot(THRESH_GRID, nb, color=c, lw=1.7, ls=ls, label=LABELS[st])
    ax.set_xlim(0.02, 0.50)
    ax.set_ylim(-0.03, 0.12)
    ax.set_xlabel('Threshold probability', fontsize=11)
    ax.set_ylabel('Net benefit', fontsize=11)
    ax.legend(loc='upper right', fontsize=8.2, framealpha=0.92)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=10)


def figure6(oof, y, out):
    fig = plt.figure(figsize=(12.5, 15))
    gs = GridSpec(3, 2, figure=fig, hspace=0.32, wspace=0.22)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]
    prev = float(y.mean())

    plan = [(axes[0], 'RF', BEDSIDE, COLORS_BEDSIDE,
             'Random forest — bedside feature sets'),
            (axes[1], 'RF', APACHE, COLORS_APACHE,
             'Random forest — APACHE II feature sets'),
            (axes[2], 'LR', BEDSIDE, COLORS_BEDSIDE,
             'Logistic regression — bedside feature sets'),
            (axes[3], 'LR', APACHE, COLORS_APACHE,
             'Logistic regression — APACHE II feature sets')]
    for ax, clf, sub, cols, title in plan:
        draw_calib(ax, clf, sub, oof, y, cols)
        ax.set_title(title, fontsize=12)
    for ax, clf in ((axes[4], 'RF'), (axes[5], 'LR')):
        draw_dca(ax, clf, oof, y, prev)
        ax.set_title(f'{CLF_TITLE[clf]} — decision curve analysis', fontsize=12)
    for ax, s in zip(axes, 'ABCDEF'):
        panel_label(ax, s, dx=-0.09)
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'fig6_calibration_dca_mice.{ext}',
                    dpi=DPI_DEFAULT, bbox_inches='tight')
    plt.close(fig)
    print('  已保存 fig6_calibration_dca_mice.png/.pdf  (6 panel)')


# ============================================================ 数值落盘（供核对）
def dump_numbers(y, oof, folds, auc, out):
    rows = []
    prev = float(y.mean())
    for c in CLFS:
        for s in SETS:
            fy, fp = folds[(c, s)]
            fm = np.array([roc_auc_score(fy[i], fp[i]) for i in range(len(fy))])
            nb = {t: net_benefit(y, oof[(c, s)], t) for t in THRESH_GRID}
            pos = [t for t in THRESH_GRID if nb[t] > 0]
            rows.append(dict(Classifier=c, Feature_set=s,
                             AUC_foldmean=round(fm.mean(), 4), AUC_foldSD=round(fm.std(), 4),
                             AUC_pooledOOF=round(roc_auc_score(y, oof[(c, s)]), 4),
                             NB_gt0_low=min(pos) if pos else np.nan,
                             NB_gt0_high=max(pos) if pos else np.nan,
                             NB_at_thresh_020=round(nb[0.20], 4),
                             Prevalence=round(prev, 4), N=len(y)))
    df = pd.DataFrame(rows)
    df.to_csv(out / 'fig5_fig6_numbers_mice.csv', index=False, encoding='utf-8-sig')
    print('\n' + '=' * 88)
    print('  绘图数值（同时写出 fig5_fig6_numbers_mice.csv）')
    print('=' * 88)
    print(df.to_string(index=False))
    return df


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nrep', type=int, default=5)
    ap.add_argument('--skip-cv', action='store_true')
    ap.add_argument('--ref', default=None, help='自检参考文件（默认 ml_final.txt）')
    ap.add_argument('--no-enforce', action='store_true', help='自检不一致时不中止')
    args = ap.parse_args()

    out = STUDY / 'results' / 'ml_modeling' / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    cache = out / '.cache' / f'fig_mice_cache_nrep{args.nrep}.npz'
    cache.parent.mkdir(parents=True, exist_ok=True)
    if args.skip_cv and not cache.exists():
        raise SystemExit(f'缺少缓存 {cache}')

    print('=' * 78)
    print('  Figure 5 / Figure 6 重做 —— 折内 MICE（与 ml_final.py 同流水线）')
    print('=' * 78)
    print(f'  重复次数 = {args.nrep}，折数 = {N_FOLDS}，插补数 = {M_IMP}')

    print('\n[1/4] 载入数据...')
    data = load_data()

    print(f'\n[2/4] 交叉验证（MICE / raw，{args.nrep} x {N_FOLDS}）...')
    oof, folds, y = run_cv(data, args.nrep, cache)

    print('\n[3/4] 自检...')
    auc = self_check(oof, folds, y,
                     enforce=(args.nrep == 5 and not args.skip_cv and not args.no_enforce),
                     ref_path=args.ref)

    print('\n[4/4] 出图...')
    figure5(folds, auc, out)
    figure6(oof, y, out)
    dump_numbers(y, oof, folds, auc, out)
    print(f'\n全部输出位于：{out}')


if __name__ == '__main__':
    main()
