"""Feature sets F-I: APACHE II as the baseline, and what FaceAge or Age Gap adds to it.

The bedside indicators are a weak baseline, so an increment over them is easy to obtain. This
script rebuilds the incremental analysis on top of APACHE II instead — the strongest single
predictor in the cohort, reported but never entered into a feature set in the original
submission. It produces, for every feature set:

  - AUC +/- SD over 5 x 10 repeated stratified cross-validation
  - DeLong paired tests against the APACHE II baseline
  - calibration intercept and slope (logistic recalibration of the out-of-fold predictions)
  - net benefit across decision thresholds
  - sensitivity and specificity at the 0.5 threshold

Environment note: as in 06, scipy.linalg.lstsq is unusable here, so no sklearn LinearRegression
and logistic regression is always solver='liblinear'. The corrected Age Gap is fitted inside the
training folds of each split.

Input  : data/merged_data.xlsx
Output : results/ml_modeling/apache_incremental/
           apache_auc_results.csv, apache_delong.csv, apache_calibration_netbenefit.csv"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             confusion_matrix)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import RESULTS_DIR, load_from_merged_xlsx

OUT_DIR = RESULTS_DIR / "ml_modeling" / "apache_incremental"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_REPEATS, N_FOLDS = 5, 10
DEGREE = 2
QUICK = ['性别', '体温', '心率', '呼吸频率', '收缩压', '舒张压', '血氧饱和度', '血糖']

print("=" * 78)
print("APACHE II 基线增量价值 / 校准 / 决策曲线")
print("=" * 78)


def delong_roc_test(y_true, scores1, scores2):
    """DeLong 配对 AUC 检验（与 05/06 脚本同一实现）"""
    y_true = np.asarray(y_true, dtype=int)
    s1 = np.asarray(scores1, dtype=float)
    s2 = np.asarray(scores2, dtype=float)
    pos, neg = y_true == 1, y_true == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos < 2 or n_neg < 2:
        return np.nan, 1.0, np.nan, np.nan
    p1, n1 = s1[pos], s1[neg]
    p2, n2 = s2[pos], s2[neg]

    def V(ps, ns):
        V10 = np.array([np.mean(ps[i] > ns) + 0.5 * np.mean(ps[i] == ns)
                        for i in range(n_pos)])
        V01 = np.array([np.mean(ps > ns[j]) + 0.5 * np.mean(ps == ns[j])
                        for j in range(n_neg)])
        return V10, V01

    V10_1, V01_1 = V(p1, n1)
    V10_2, V01_2 = V(p2, n2)
    auc1, auc2 = float(np.mean(V10_1)), float(np.mean(V10_2))

    def cov(a10, a01, aa, b10, b01, ba):
        S10 = np.sum((a10 - aa) * (b10 - ba)) / (n_pos - 1) if n_pos > 1 else 0
        S01 = np.sum((a01 - aa) * (b01 - ba)) / (n_neg - 1) if n_neg > 1 else 0
        return S10 / n_pos + S01 / n_neg

    var1 = cov(V10_1, V01_1, auc1, V10_1, V01_1, auc1)
    var2 = cov(V10_2, V01_2, auc2, V10_2, V01_2, auc2)
    c12 = cov(V10_1, V01_1, auc1, V10_2, V01_2, auc2)
    se = np.sqrt(max(var1 + var2 - 2 * c12, 0))
    if se == 0:
        return 0.0, 1.0, auc1, auc2
    z = (auc1 - auc2) / se
    return float(z), float(2 * stats.norm.sf(abs(z))), auc1, auc2


def make_classifiers():
    return {
        'RF': RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=10,
            min_samples_leaf=5, class_weight='balanced',
            random_state=42, n_jobs=-1),
        'LR': LogisticRegression(
            penalty='l2', C=1.0, class_weight='balanced',
            solver='liblinear', max_iter=1000, random_state=42),
    }


# ============================================================================
# 1. 数据
# ============================================================================
print("\n[1/3] 加载数据...")
df = load_from_merged_xlsx(max_occlusion_level=2)
n, n_death = len(df), int(df['7天死亡'].sum())
print(f"  分析集 N = {n}，死亡 {n_death} 例（{n_death / n:.1%}）")
print(f"  APACHE II：均值 {df['APACHE评分'].mean():.1f} ± {df['APACHE评分'].std():.1f}，"
      f"范围 {df['APACHE评分'].min()}–{df['APACHE评分'].max()}，无缺失")

y = df['7天死亡'].values.astype(int)
age = df['实际年龄'].values.astype(float)
gap_raw = df['Age_Gap'].values.astype(float)
face_raw = df['生物学年龄'].values.astype(float)
apache = df['APACHE评分'].values.astype(float).reshape(-1, 1)
quick = df[QUICK].values.astype(float)

# 无偏校正（折内拟合时逐折重算；此处仅用于单变量参考）
coef_full = np.polyfit(age, gap_raw, DEGREE)
gap_corr_full = gap_raw - np.polyval(coef_full, age)


def build(name, gap_c, face_c):
    """按名称构造特征矩阵；gap_c/face_c 为该折的校正值"""
    if name == 'quick_only':            return quick
    if name == 'apache_only':           return apache
    if name == 'apache_age':            return np.column_stack([apache, age])
    if name == 'apache_faceage':        return np.column_stack([apache, face_c])
    if name == 'apache_agegap':         return np.column_stack([apache, gap_c])
    if name == 'apache_agegap_raw':     return np.column_stack([apache, gap_raw])
    if name == 'apache_quick':          return np.column_stack([apache, quick])
    if name == 'apache_quick_faceage':  return np.column_stack([apache, quick, face_c])
    if name == 'apache_quick_agegap':   return np.column_stack([apache, quick, gap_c])
    if name == 'apache_quick_agegap_c': return np.column_stack([apache, quick, age, gap_c])
    raise ValueError(name)


FEATURE_SETS = ['quick_only', 'apache_only', 'apache_age', 'apache_faceage',
                'apache_agegap_raw', 'apache_agegap', 'apache_quick',
                'apache_quick_faceage', 'apache_quick_agegap',
                'apache_quick_agegap_c']

LABELS = {
    'quick_only': 'Quick (8 vitals) [旧基线A]',
    'apache_only': 'APACHE II',
    'apache_age': 'APACHE II + 实际年龄',
    'apache_faceage': 'APACHE II + FaceAge(校正)',
    'apache_agegap_raw': 'APACHE II + 原始 Age Gap',
    'apache_agegap': 'APACHE II + 校正 Age Gap',
    'apache_quick': 'APACHE II + 8体征',
    'apache_quick_faceage': 'APACHE II + 8体征 + FaceAge(校正)',
    'apache_quick_agegap': 'APACHE II + 8体征 + 校正 Age Gap',
    'apache_quick_agegap_c': 'APACHE II + 8体征 + 实际年龄 + 校正 Age Gap',
}

# DeLong 配对（对比方向：前 vs 后，检验"后者相对前者"的增量）
COMPARISONS = [
    ('apache_only', 'quick_only', 'APACHE II vs 旧基线'),
    ('apache_age', 'apache_only', 'APACHE II + 年龄 vs APACHE II'),
    ('apache_faceage', 'apache_only', 'APACHE II + FaceAge vs APACHE II'),
    ('apache_agegap_raw', 'apache_only', 'APACHE II + 原始AgeGap vs APACHE II'),
    ('apache_agegap', 'apache_only', 'APACHE II + 校正AgeGap vs APACHE II'),
    ('apache_quick', 'apache_only', 'APACHE II + 8体征 vs APACHE II'),
    ('apache_quick_faceage', 'apache_quick', '再加 FaceAge vs APACHE II + 8体征'),
    ('apache_quick_agegap', 'apache_quick', '再加 校正AgeGap vs APACHE II + 8体征'),
    ('apache_quick_agegap_c', 'apache_quick', '再加 年龄+校正AgeGap vs APACHE II+8体征'),
    ('apache_agegap', 'apache_age', '校正AgeGap 在 年龄 之上的增量'),
]

# ============================================================================
# 2. CV
# ============================================================================
print("\n[2/3] 运行 5×10 重复分层交叉验证（折内交叉拟合校正）...")

metrics = {c: {m: {'auc': [], 'sens': [], 'spec': []} for m in FEATURE_SETS}
           for c in ('RF', 'LR')}
repeat_preds = {c: {m: [] for m in FEATURE_SETS} for c in ('RF', 'LR')}

for repeat in range(N_REPEATS):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42 + repeat)
    cur = {c: {m: np.full(n, np.nan) for m in FEATURE_SETS} for c in ('RF', 'LR')}
    for tr, te in skf.split(quick, y):
        c_fold = np.polyfit(age[tr], gap_raw[tr], DEGREE)     # 折内拟合
        gap_c = gap_raw - np.polyval(c_fold, age)
        face_c = age + gap_c
        for m in FEATURE_SETS:
            X = build(m, gap_c, face_c)
            sc = StandardScaler()
            Xtr, Xte = sc.fit_transform(X[tr]), sc.transform(X[te])
            for cn, clf in make_classifiers().items():
                clf.fit(Xtr, y[tr])
                p = clf.predict_proba(Xte)[:, 1]
                pred = (p >= 0.5).astype(int)
                r = metrics[cn][m]
                r['auc'].append(roc_auc_score(y[te], p))
                tn, fp, fn, tp = confusion_matrix(y[te], pred).ravel()
                r['sens'].append(tp / (tp + fn) if (tp + fn) else 0)
                r['spec'].append(tn / (tn + fp) if (tn + fp) else 0)
                cur[cn][m][te] = p
    for cn in ('RF', 'LR'):
        for m in FEATURE_SETS:
            repeat_preds[cn][m].append(cur[cn][m])
    print(f"    repeat {repeat + 1}/{N_REPEATS} 完成", flush=True)

avg = {cn: {m: np.nanmean(np.vstack(repeat_preds[cn][m]), axis=0)
            for m in FEATURE_SETS} for cn in ('RF', 'LR')}

# ============================================================================
# 3. 汇总：AUC、DeLong、校准、net benefit
# ============================================================================
print("\n[3/3] 汇总 AUC / DeLong / 校准 / 决策曲线...")

rows = []
for cn in ('RF', 'LR'):
    for m in FEATURE_SETS:
        r = metrics[cn][m]
        rows.append({'Classifier': cn, 'Model': LABELS[m], 'key': m,
                     'AUC_mean': np.mean(r['auc']), 'AUC_sd': np.std(r['auc']),
                     'Sensitivity@0.5': np.mean(r['sens']),
                     'Specificity@0.5': np.mean(r['spec'])})
res = pd.DataFrame(rows)
res.to_csv(OUT_DIR / "apache_auc_results.csv", index=False, encoding='utf-8-sig')

# DeLong
dl = []
for cn in ('RF', 'LR'):
    for m1, m2, desc in COMPARISONS:
        z, p, a1, a2 = delong_roc_test(y, avg[cn][m1], avg[cn][m2])
        sig = ('***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else 'ns')
        dl.append({'Classifier': cn, 'Comparison': desc,
                   'AUC_new': round(a1, 4), 'AUC_base': round(a2, 4),
                   'Delta_AUC': round(a1 - a2, 4), 'p_value': p, 'sig': sig})
dl = pd.DataFrame(dl)
dl.to_csv(OUT_DIR / "apache_delong.csv", index=False, encoding='utf-8-sig')


def calibration(y_true, p):
    """对折外预测做 logistic 重校准：logit(y) = a + b*logit(p)"""
    eps = 1e-6
    lg = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    cal = LogisticRegression(penalty='l2', C=1e6, solver='liblinear',
                             max_iter=1000)
    cal.fit(lg.reshape(-1, 1), y_true)
    return float(cal.intercept_[0]), float(cal.coef_[0][0])


def net_benefit(y_true, p, pt):
    pred = (p >= pt).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    N = len(y_true)
    return tp / N - fp / N * (pt / (1 - pt))


cal_rows = []
for cn in ('RF', 'LR'):
    for m in FEATURE_SETS:
        a, b = calibration(y, avg[cn][m])
        row = {'Classifier': cn, 'Model': LABELS[m],
               'Calib_intercept': round(a, 3), 'Calib_slope': round(b, 3)}
        for pt in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
            row[f'NB@{pt:.2f}'] = round(net_benefit(y, avg[cn][m], pt), 4)
        cal_rows.append(row)
cal = pd.DataFrame(cal_rows)
cal.to_csv(OUT_DIR / "apache_calibration_netbenefit.csv",
           index=False, encoding='utf-8-sig')

# 参考：单变量 AUC 与 treat-all 的 net benefit
prev = y.mean()
print("\n" + "=" * 78)
print("AUC（5×10 CV，折内交叉拟合校正）")
print("=" * 78)
print(f"{'Classifier':>10s}  {'Model':<34s} {'AUC':>16s} {'Sens@0.5':>9s} {'Spec@0.5':>9s}")
print("-" * 78)
for cn in ('RF', 'LR'):
    for m in FEATURE_SETS:
        r = res[(res['Classifier'] == cn) & (res['key'] == m)].iloc[0]
        print(f"{cn:>10s}  {LABELS[m]:<34s} "
              f"{r['AUC_mean']:.4f} ± {r['AUC_sd']:.4f}  "
              f"{r['Sensitivity@0.5']:>9.3f} {r['Specificity@0.5']:>9.3f}")
    print("-" * 78)

print("\n" + "=" * 78)
print("DeLong 配对检验（折外预测跨 repeat 平均）")
print("=" * 78)
print(f"{'Classifier':>10s}  {'Comparison':<40s} {'ΔAUC':>8s} {'P':>8s} {'sig':>4s}")
print("-" * 78)
for _, r in dl.iterrows():
    print(f"{r['Classifier']:>10s}  {r['Comparison']:<40s} "
          f"{r['Delta_AUC']:>+8.4f} {r['p_value']:>8.3f} {r['sig']:>4s}")

print("\n" + "=" * 78)
print(f"校准与 net benefit（死亡率为 {prev:.3f}）")
print("=" * 78)
print(cal.to_string(index=False))
print(f"\n\ntreat-all 参考：NBs = "
      + ", ".join(f"{pt:.2f}:{prev - (1-prev)*pt/(1-pt):+.3f}"
                  for pt in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)))
print(f"\n结果已保存至：{OUT_DIR}")
