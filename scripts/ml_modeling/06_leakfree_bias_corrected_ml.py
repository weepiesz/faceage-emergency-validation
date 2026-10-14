"""Bias correction of Age Gap fitted inside the training folds, with the leaky version for comparison.

Background: the earlier script 05_bias_corrected_ml.py fitted the quadratic correction of
Age Gap on all 1,470 rows and only then entered the cross-validation loop, so every held-out
patient had contributed to the correction coefficients. That is a leak, and it runs in the
flattering direction (the corrected AUC comes out too high).

Two variants are reported side by side:

  (a) leaky      correction fitted on the full cohort before cross-validation
                 (reproduces the earlier Table 16 and quantifies the effect)
  (b) leakfree   correction refitted on the training rows of each fold only, then applied to
                 both training and held-out rows — the reported version

Environment note: scipy.linalg.lstsq silently kills the process in this environment, which takes
out sklearn's LinearRegression and LogisticRegression(solver='lbfgs'). Polynomial fitting
therefore uses numpy.polyfit (mathematically equivalent) and logistic regression uses
solver='liblinear' (same objective, L2 penalty). The script re-derives the original feature sets
and checks them against the existing Table 16 to confirm the substitutions change nothing.

Input  : data/merged_data.xlsx
         results/ml_modeling/cumulative_occlusion/   (reference table for the verification)
Output : results/ml_modeling/leakfree_bias_corrected/
           leakfree_comparison.csv, leakfree_model_results.csv, leaky_model_results.csv,
           leakfree_delong_tests.csv, verification_vs_table16.csv, report.md"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, accuracy_score, confusion_matrix,
                             f1_score, average_precision_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import RESULTS_DIR, load_from_merged_xlsx

OUT_DIR = RESULTS_DIR / "ml_modeling" / "leakfree_bias_corrected"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_REPEATS, N_FOLDS = 5, 10
DEGREE = 2
QUICK_FEATURES = ['性别', '体温', '心率', '呼吸频率',
                  '收缩压', '舒张压', '血氧饱和度', '血糖']
FEATURE_SET_NAMES = ['A_baseline', 'B_real_age', 'C_faceage',
                     'D_age_gap', 'E_age_gap_only']
FEATURE_SET_LABELS = {
    'A_baseline': 'A (baseline)', 'B_real_age': 'B (+ age)',
    'C_faceage': 'C (+ FaceAge)', 'D_age_gap': 'D (+ age + Age Gap)',
    'E_age_gap_only': 'E (+ Age Gap)',
}

# DeLong 配对检验的对比组（与 05 脚本一致）
COMPARISONS = [
    ('B_real_age', 'A_baseline', 'Age vs baseline'),
    ('C_faceage', 'A_baseline', 'FaceAge vs baseline'),
    ('D_age_gap', 'B_real_age', 'Age Gap incremental over age'),
    ('C_faceage', 'B_real_age', 'FaceAge vs age'),
    ('E_age_gap_only', 'A_baseline', 'Age Gap alone vs baseline'),
]

print("=" * 78)
print("Part C：无泄漏的偏倚校正 ML 建模（含 DeLong 检验）")
print("=" * 78)


def delong_roc_test(y_true, scores1, scores2):
    """DeLong 配对 AUC 检验（与 05_bias_corrected_ml.py 同一实现）"""
    y_true = np.asarray(y_true, dtype=int)
    scores1 = np.asarray(scores1, dtype=float)
    scores2 = np.asarray(scores2, dtype=float)

    pos_idx = y_true == 1
    neg_idx = y_true == 0
    n_pos = int(pos_idx.sum())
    n_neg = int(neg_idx.sum())

    if n_pos < 2 or n_neg < 2:
        return np.nan, 1.0, np.nan, np.nan

    s1_pos, s1_neg = scores1[pos_idx], scores1[neg_idx]
    s2_pos, s2_neg = scores2[pos_idx], scores2[neg_idx]

    def compute_V(pos_scores, neg_scores):
        V10 = np.array([np.mean(pos_scores[i] > neg_scores) +
                        0.5 * np.mean(pos_scores[i] == neg_scores)
                        for i in range(n_pos)])
        V01 = np.array([np.mean(pos_scores > neg_scores[j]) +
                        0.5 * np.mean(pos_scores == neg_scores[j])
                        for j in range(n_neg)])
        return V10, V01

    V10_1, V01_1 = compute_V(s1_pos, s1_neg)
    V10_2, V01_2 = compute_V(s2_pos, s2_neg)

    auc1 = float(np.mean(V10_1))
    auc2 = float(np.mean(V10_2))

    def compute_cov(V10_a, V01_a, auc_a, V10_b, V01_b, auc_b):
        S10 = (np.sum((V10_a - auc_a) * (V10_b - auc_b)) / (n_pos - 1)
               if n_pos > 1 else 0)
        S01 = (np.sum((V01_a - auc_a) * (V01_b - auc_b)) / (n_neg - 1)
               if n_neg > 1 else 0)
        return S10 / n_pos + S01 / n_neg

    var1 = compute_cov(V10_1, V01_1, auc1, V10_1, V01_1, auc1)
    var2 = compute_cov(V10_2, V01_2, auc2, V10_2, V01_2, auc2)
    cov12 = compute_cov(V10_1, V01_1, auc1, V10_2, V01_2, auc2)

    se_diff = np.sqrt(max(var1 + var2 - 2 * cov12, 0))
    if se_diff == 0:
        return 0.0, 1.0, auc1, auc2

    z = (auc1 - auc2) / se_diff
    p_value = 2 * stats.norm.sf(abs(z))
    return float(z), float(p_value), auc1, auc2


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


def build_features(name, quick, age_col, faceage_col, gap_col):
    """按原脚本的列顺序构造特征矩阵"""
    if name == 'A_baseline':
        return quick
    if name == 'B_real_age':
        return np.column_stack([quick, age_col])
    if name == 'C_faceage':
        return np.column_stack([quick, faceage_col])
    if name == 'D_age_gap':
        return np.column_stack([quick, age_col, gap_col])
    if name == 'E_age_gap_only':
        return np.column_stack([quick, gap_col])
    raise ValueError(name)


# ============================================================================
# 1. 加载
# ============================================================================
print("\n[1/4] 加载数据...")

df = load_from_merged_xlsx(max_occlusion_level=2)
n, n_death = len(df), int(df['7天死亡'].sum())
print(f"  分析集 N = {n}，死亡 {n_death} 例（{n_death / n:.1%}）")

y = df['7天死亡'].values.astype(int)
age = df['实际年龄'].values.astype(float)
gap = df['Age_Gap'].values.astype(float)
quick = df[QUICK_FEATURES].values.astype(float)

# 全队列校正（leaky 变体使用；也用于核对）
coef_full = np.polyfit(age, gap, DEGREE)
gap_corr_full = gap - np.polyval(coef_full, age)
faceage_corr_full = age + gap_corr_full


# ============================================================================
# 2. CV 主流程
# ============================================================================
def run_cv(mode, feature_sets):
    """
    mode: 'raw'      不做校正，用原始 生物学年龄 / Age_Gap
          'leaky'    校正曲线在 CV 之前用全队列拟合
          'leakfree' 校正曲线在每折训练集内拟合
    """
    metrics = {c: {m: {'auc': [], 'auprc': [], 'acc': [],
                       'sens': [], 'spec': [], 'f1': []}
                   for m in feature_sets} for c in ('RF', 'LR')}
    # 每次 repeat 的折外预测（用于 DeLong 配对检验）
    repeat_preds = {c: {m: [] for m in feature_sets} for c in ('RF', 'LR')}

    for repeat in range(N_REPEATS):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                              random_state=42 + repeat)
        cur_preds = {c: {m: np.full(len(y), np.nan) for m in feature_sets}
                     for c in ('RF', 'LR')}

        for tr, te in skf.split(quick, y):
            if mode == 'leakfree':
                c_fold = np.polyfit(age[tr], gap[tr], DEGREE)
                gap_c = gap - np.polyval(c_fold, age)
                face_c = age + gap_c
            elif mode == 'leaky':
                gap_c, face_c = gap_corr_full, faceage_corr_full
            else:  # raw
                gap_c, face_c = gap, df['生物学年龄'].values.astype(float)

            for m in feature_sets:
                X = build_features(m, quick, age, face_c, gap_c)
                sc = StandardScaler()
                X_tr = sc.fit_transform(X[tr])
                X_te = sc.transform(X[te])

                for clf_name, clf in make_classifiers().items():
                    clf.fit(X_tr, y[tr])
                    proba = clf.predict_proba(X_te)[:, 1]
                    pred = (proba >= 0.5).astype(int)
                    r = metrics[clf_name][m]
                    r['auc'].append(roc_auc_score(y[te], proba))
                    r['auprc'].append(average_precision_score(y[te], proba))
                    r['acc'].append(accuracy_score(y[te], pred))
                    tn, fp, fn, tp = confusion_matrix(y[te], pred).ravel()
                    r['sens'].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
                    r['spec'].append(tn / (tn + fp) if (tn + fp) > 0 else 0)
                    r['f1'].append(f1_score(y[te], pred, zero_division=0))
                    cur_preds[clf_name][m][te] = proba

        for clf_name in ('RF', 'LR'):
            for m in feature_sets:
                repeat_preds[clf_name][m].append(cur_preds[clf_name][m])

        print(f"    repeat {repeat + 1}/{N_REPEATS} 完成", flush=True)

    # 跨 repeat 平均折外预测
    avg_preds = {}
    for clf_name in ('RF', 'LR'):
        avg_preds[clf_name] = {}
        for m in feature_sets:
            stacked = np.vstack(repeat_preds[clf_name][m])
            avg_preds[clf_name][m] = np.nanmean(stacked, axis=0)

    rows = []
    for clf_name in ('RF', 'LR'):
        for m in feature_sets:
            r = metrics[clf_name][m]
            rows.append({
                'Classifier': clf_name, 'Model': FEATURE_SET_LABELS[m],
                'feature_set': m,
                'AUC_mean': np.mean(r['auc']), 'AUC_sd': np.std(r['auc']),
                'AUPRC_mean': np.mean(r['auprc']), 'AUPRC_sd': np.std(r['auprc']),
                'Accuracy_mean': np.mean(r['acc']),
                'Sensitivity_mean': np.mean(r['sens']),
                'Specificity_mean': np.mean(r['spec']),
                'F1_mean': np.mean(r['f1']),
            })
    return pd.DataFrame(rows), avg_preds


def run_delong(y_true, avg_preds, label):
    """对跨 repeat 平均的折外预测做 DeLong 配对检验"""
    rows = []
    for clf_name in ('RF', 'LR'):
        for m1, m2, desc in COMPARISONS:
            z, p, auc1, auc2 = delong_roc_test(
                y_true, avg_preds[clf_name][m1], avg_preds[clf_name][m2])
            sig = ('***' if p < 0.01 else '**' if p < 0.05
                   else '*' if p < 0.10 else 'ns')
            rows.append({
                'Classifier': clf_name, 'Feature_set': label,
                'Comparison': f'{m1} vs {m2}', 'Description': desc,
                'AUC_m1': round(auc1, 4), 'AUC_m2': round(auc2, 4),
                'Delta_AUC': round(auc1 - auc2, 4),
                'p_value': p, 'significance': sig,
            })
    return pd.DataFrame(rows)


# ============================================================================
# 3. 三个变体
# ============================================================================
print("\n[2/4] 运行 CV：原始（核对用）...")
raw, raw_preds = run_cv('raw', FEATURE_SET_NAMES)

print("\n[3/4] 运行 CV：leaky（复现现有表16）...")
leaky, _ = run_cv('leaky', FEATURE_SET_NAMES)

print("\n[4/4] 运行 CV：leakfree（无泄漏，最终报告版本）...")
leakfree, leakfree_preds = run_cv('leakfree', FEATURE_SET_NAMES)

# DeLong：原始 vs 无泄漏校正
delong_raw = run_delong(y, raw_preds, 'raw')
delong_lf = run_delong(y, leakfree_preds, 'leakfree')
delong_all = pd.concat([delong_raw, delong_lf], ignore_index=True)
delong_all.to_csv(OUT_DIR / "leakfree_delong_tests.csv",
                  index=False, encoding='utf-8-sig')


# ============================================================================
# 4. 核对与输出
# ============================================================================
target_file = (RESULTS_DIR / "ml_modeling" / "cumulative_occlusion"
               / "cumulative_model_paper_table.csv")
verif_rows = []
if target_file.exists():
    tgt = pd.read_csv(target_file)
    tgt = tgt[tgt['Classifier'].isin(['RF', 'LR'])]
    for _, t in tgt.iterrows():
        match = raw[(raw['Classifier'] == t['Classifier']) &
                    (raw['Model'] == t['Model'])]
        if match.empty:
            continue
        got = float(match.iloc[0]['AUC_mean'])
        want = float(str(t['AUC']).split('±')[0].strip())
        verif_rows.append({
            'Classifier': t['Classifier'], 'Model': t['Model'],
            '现有表AUC': want, '本次复算AUC': round(got, 4),
            '差值': round(got - want, 4),
        })
verif = pd.DataFrame(verif_rows)
verif.to_csv(OUT_DIR / "verification_vs_table16.csv",
             index=False, encoding='utf-8-sig')

# 两变体对照
comp_rows = []
for clf_name in ('RF', 'LR'):
    for m in FEATURE_SET_NAMES:
        lk = leaky[(leaky['Classifier'] == clf_name) &
                   (leaky['feature_set'] == m)].iloc[0]
        lf = leakfree[(leakfree['Classifier'] == clf_name) &
                      (leakfree['feature_set'] == m)].iloc[0]
        comp_rows.append({
            'Classifier': clf_name, 'Model': FEATURE_SET_LABELS[m],
            'leaky_AUC': round(lk['AUC_mean'], 4), 'leaky_SD': round(lk['AUC_sd'], 4),
            'leakfree_AUC': round(lf['AUC_mean'], 4), 'leakfree_SD': round(lf['AUC_sd'], 4),
            'ΔAUC(leaky-leakfree)': round(lk['AUC_mean'] - lf['AUC_mean'], 4),
        })
comp = pd.DataFrame(comp_rows)
comp.to_csv(OUT_DIR / "leakfree_comparison.csv", index=False, encoding='utf-8-sig')

leakfree.to_csv(OUT_DIR / "leakfree_model_results.csv",
                index=False, encoding='utf-8-sig')
leaky.to_csv(OUT_DIR / "leaky_model_results.csv",
             index=False, encoding='utf-8-sig')

# 报告
max_delta = comp['ΔAUC(leaky-leakfree)'].abs().max()
lines = [
    "# Part C 无泄漏偏倚校正 ML 建模",
    "",
    f"分析集 N = {n}，死亡 {n_death} 例（{n_death / n:.1%}）；"
    f"5×10 重复分层交叉验证，二次多项式校正",
    "",
    "## 泄漏影响：leaky vs leakfree",
    "",
    "| 分类器 | 特征集 | leaky AUC | leakfree AUC | ΔAUC |",
    "|---|---|---|---|---|",
]
for _, r in comp.iterrows():
    lines.append(f"| {r['Classifier']} | {r['Model']} | "
                 f"{r['leaky_AUC']:.4f} ± {r['leaky_SD']:.4f} | "
                 f"{r['leakfree_AUC']:.4f} ± {r['leakfree_SD']:.4f} | "
                 f"{r['ΔAUC(leaky-leakfree)']:+.4f} |")

lines += [
    "",
    f"**最大 ΔAUC = {max_delta:.4f}**",
    "",
    "注：特征集 A、B 不含校正变量，两变体数值完全相同。",
    "",
]
if not verif.empty:
    lines += [
        "## 与现有表 AUC 的核对（原始特征集）",
        "",
        "| 分类器 | 特征集 | 现有表 | 本次复算 | 差值 |",
        "|---|---|---|---|---|",
    ]
    for _, r in verif.iterrows():
        lines.append(f"| {r['Classifier']} | {r['Model']} | {r['现有表AUC']:.3f} | "
                     f"{r['本次复算AUC']:.4f} | {r['差值']:+.4f} |")
    lines.append("")

# DeLong 对照
lines += [
    "## DeLong 配对检验：原始 vs 无泄漏校正",
    "",
    "| 分类器 | 对比 | 原始 P | 原始 | 无泄漏校正 P | 无泄漏 | 模式一致 |",
    "|---|---|---|---|---|---|---|",
]
delong_pattern_rows = []
for clf_name in ('RF', 'LR'):
    for m1, m2, desc in COMPARISONS:
        o = delong_raw[(delong_raw['Classifier'] == clf_name) &
                       (delong_raw['Comparison'] == f'{m1} vs {m2}')].iloc[0]
        c = delong_lf[(delong_lf['Classifier'] == clf_name) &
                      (delong_lf['Comparison'] == f'{m1} vs {m2}')].iloc[0]
        same = '是' if o['significance'] == c['significance'] else '**否**'
        lines.append(f"| {clf_name} | {m1} vs {m2} | {o['p_value']:.3f} | "
                     f"{o['significance']} | {c['p_value']:.3f} | "
                     f"{c['significance']} | {same} |")
        if same != '是':
            delong_pattern_rows.append(
                f"{clf_name} {m1} vs {m2}: 原始 P = {o['p_value']:.3f} "
                f"({o['significance']}) -> 无泄漏校正 P = {c['p_value']:.3f} "
                f"({c['significance']})")
lines.append("")
if delong_pattern_rows:
    lines += ["**显著性模式发生变化的对比：**", ""]
    lines += [f"- {r}" for r in delong_pattern_rows]
else:
    lines.append("**所有对比的显著性模式在原始与无泄漏校正之间保持一致。**")
lines.append("")

(OUT_DIR / "report.md").write_text("\n".join(lines), encoding='utf-8')

print("\n" + "=" * 78)
print("泄漏影响（leaky - leakfree）")
print("=" * 78)
print(comp.to_string(index=False))
if not verif.empty:
    print("\n与现有表 AUC 的核对（原始特征集，LR 用 liblinear）")
    print(verif.to_string(index=False))

print("\n" + "=" * 78)
print("DeLong 配对检验：原始 vs 无泄漏校正")
print("=" * 78)
print(f"{'分类器':>6s}  {'对比':>22s}  {'原始P':>8s} {'Sig':>4s}  "
      f"{'无泄漏P':>8s} {'Sig':>4s}  模式")
print("-" * 78)
for clf_name in ('RF', 'LR'):
    for m1, m2, desc in COMPARISONS:
        o = delong_raw[(delong_raw['Classifier'] == clf_name) &
                       (delong_raw['Comparison'] == f'{m1} vs {m2}')].iloc[0]
        c = delong_lf[(delong_lf['Classifier'] == clf_name) &
                      (delong_lf['Comparison'] == f'{m1} vs {m2}')].iloc[0]
        same = '一致' if o['significance'] == c['significance'] else '变化'
        print(f"{clf_name:>6s}  {m1 + ' vs ' + m2:>22s}  "
              f"{o['p_value']:>8.3f} {o['significance']:>4s}  "
              f"{c['p_value']:>8.3f} {c['significance']:>4s}  {same}")

print(f"\n结果已保存至：{OUT_DIR}")
