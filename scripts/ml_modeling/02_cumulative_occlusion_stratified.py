"""Does the value of FaceAge fall as occlusion accumulates? Cumulative-stratification modelling.

Instead of testing each occlusion grade separately, the cohort is split into two cumulative
strata and the five bedside-based feature sets (A-E) are refitted within each:

  stratum 1   occlusion <= 1   (none + mild)
  stratum 2   occlusion <= 2   (none + mild + moderate; grade 3 excluded)

The grade-0-only stratum is deliberately not analysed on its own: its mortality is low (9.0%) and
deaths concentrate in the occluded patients, so partitioning it out would compare populations
that differ for reasons unrelated to occlusion. Random forest and logistic regression are run in
both strata.

This script's output is the reference table that 06_leakfree_bias_corrected_ml.py verifies its
own reproduction against.

Input  : data/merged_data.xlsx
Output : results/ml_modeling/cumulative_occlusion/
           cumulative_model_results.csv, cumulative_statistical_tests.csv,
           cumulative_model_paper_table.csv, roc_curves_cumulative.png,
           prc_curves_cumulative.png, cumulative_analysis_report.md"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 添加utils到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    load_from_merged_xlsx,
    RESULTS_DIR,
    FONT_CHINESE,
    DPI_DEFAULT
)

# Use English publication font
plt.rcParams['font.sans-serif'] = FONT_CHINESE
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("方向2 - 机器学习建模：按遮挡累积分层分析")
print("=" * 80)
print()

# ============================================================================
# DeLong检验实现（DeLong, DeLong & Clarke-Pearson, 1988, Biometrics 44:837-845）
# ============================================================================

def delong_roc_test(y_true, scores1, scores2):
    """
    DeLong检验：比较两个相关的ROC曲线的AUC。

    参数:
        y_true: 真实标签
        scores1: 模型1的预测分数
        scores2: 模型2的预测分数

    返回:
        z_stat, p_value, auc1, auc2
    """
    y_true = np.asarray(y_true, dtype=int)
    scores1 = np.asarray(scores1, dtype=float)
    scores2 = np.asarray(scores2, dtype=float)

    pos_idx = y_true == 1
    neg_idx = y_true == 0
    n_pos = pos_idx.sum()
    n_neg = neg_idx.sum()

    if n_pos < 2 or n_neg < 2:
        return 0.0, 1.0, np.nan, np.nan

    s1_pos = scores1[pos_idx]; s1_neg = scores1[neg_idx]
    s2_pos = scores2[pos_idx]; s2_neg = scores2[neg_idx]

    # 结构成分（influence functions）
    def compute_V(pos_scores, neg_scores):
        """V_10 for positives, V_01 for negatives"""
        # 对于每个正例: (1/n_neg) * sum(I(pos > neg))
        V10 = np.array([np.mean(pos_scores[i] > neg_scores) +
                        0.5 * np.mean(pos_scores[i] == neg_scores)
                        for i in range(n_pos)])
        # 对于每个负例: (1/n_pos) * sum(I(pos > neg))
        V01 = np.array([np.mean(pos_scores > neg_scores[j]) +
                        0.5 * np.mean(pos_scores == neg_scores[j])
                        for j in range(n_neg)])
        return V10, V01

    V10_1, V01_1 = compute_V(s1_pos, s1_neg)
    V10_2, V01_2 = compute_V(s2_pos, s2_neg)

    auc1 = np.mean(V10_1)
    auc2 = np.mean(V10_2)

    # 方差和协方差估计
    def compute_cov_components(V10_a, V01_a, auc_a, V10_b, V01_b, auc_b):
        S10 = np.sum((V10_a - auc_a) * (V10_b - auc_b)) / (n_pos - 1) if n_pos > 1 else 0
        S01 = np.sum((V01_a - auc_a) * (V01_b - auc_b)) / (n_neg - 1) if n_neg > 1 else 0
        return S10 / n_pos + S01 / n_neg

    var1 = compute_cov_components(V10_1, V01_1, auc1, V10_1, V01_1, auc1)
    var2 = compute_cov_components(V10_2, V01_2, auc2, V10_2, V01_2, auc2)
    cov12 = compute_cov_components(V10_1, V01_1, auc1, V10_2, V01_2, auc2)

    se_diff = np.sqrt(max(var1 + var2 - 2 * cov12, 0))

    if se_diff == 0:
        return 0.0, 1.0, auc1, auc2

    z = (auc1 - auc2) / se_diff
    p_value = 2 * stats.norm.sf(abs(z))  # 双侧检验

    return z, p_value, auc1, auc2


# ============================================================================
# 1. 加载数据并按遮挡累积分层
# ============================================================================

print("[1/6] Loading data...")
df = load_from_merged_xlsx(max_occlusion_level=2)

# Use all patients with valid FaceAge + occlusion data (N = 1,470)
df_all = df[df['遮挡等级'] <= 2].copy()

print(f"  Total samples: {len(df)}")
print(f"  All patients (occlusion <= 2): {len(df_all)}, deaths {int(df_all['7天死亡'].sum())} ({df_all['7天死亡'].mean()*100:.1f}%)")
print()

# ============================================================================
# 2. 定义特征集和分类器
# ============================================================================

print("[2/6] Defining feature sets and classifiers...")

# 快速评估指标（用于基线模型）
quick_features = ['性别','体温', '心率', '呼吸频率', '收缩压', '舒张压', '血氧饱和度', '血糖']

feature_sets = {
    'A_baseline': quick_features,
    'B_real_age': quick_features + ['实际年龄'],
    'C_faceage':  quick_features + ['生物学年龄'],
    'D_age_gap': quick_features + ['实际年龄', 'Age_Gap'],
    'E_age_gap_only': quick_features + ['Age_Gap']
}

print("  特征集:")
for name, features in feature_sets.items():
    print(f"    {name}: {len(features)} 个特征")

# 定义分类器：随机森林 + 逻辑回归
classifiers = {
    'RandomForest': RandomForestClassifier(
        n_estimators=100, max_depth=10,
        min_samples_split=10, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'LogisticRegression': LogisticRegression(
        penalty='l2', C=1.0, class_weight='balanced',
        solver='lbfgs', max_iter=1000, random_state=42
    )
}
print(f"  分类器: {', '.join(classifiers.keys())}")
print()

# ============================================================================
# 3. 定义建模函数（支持多分类器）
# ============================================================================

def run_model_comparison_for_group(df_group, group_name, feature_sets, classifiers, n_folds=10, n_repeats=5):
    """在单个分组内运行模型对比（重复CV + DeLong检验）"""
    print(f"\n  处理 {group_name} (n={len(df_group)})...")

    y = df_group['7天死亡'].values.astype(int)
    n_samples = len(df_group)

    # 结果存储：50折的每个fold指标
    results = {clf_name: {model_name: {
        'auc_scores': [],
        'auprc_scores': [],
        'accuracy_scores': [],
        'sensitivity_scores': [],
        'specificity_scores': [],
        'precision_scores': [],
        'f1_scores': [],
        # 最终汇聚预测（跨repeat平均，用于ROC/PRC绘图和DeLong检验）
        'y_true_final': None,
        'y_pred_final': None,
    } for model_name in feature_sets.keys()} for clf_name in classifiers.keys()}

    # 每个患者的跨repeat预测追踪
    all_repeat_preds = {clf_name: {model_name: []
        for model_name in feature_sets} for clf_name in classifiers}

    total_folds = n_repeats * n_folds
    fold_counter = 0

    for repeat in range(n_repeats):
        print(f"    Repeat {repeat+1}/{n_repeats}...")
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42 + repeat)

        # 当前repeat的患者级预测
        repeat_preds = {clf_name: {model_name: {
            'y_pred': np.full(n_samples, np.nan)
        } for model_name in feature_sets} for clf_name in classifiers}

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df_group, y)):
            fold_counter += 1
            for model_name, features in feature_sets.items():
                X = df_group[features].values
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled  = scaler.transform(X_test)

                for clf_name, clf in classifiers.items():
                    clf.fit(X_train_scaled, y_train)

                    y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
                    y_pred = (y_pred_proba >= 0.5).astype(int)

                    auc      = roc_auc_score(y_test, y_pred_proba)
                    auprc = average_precision_score(y_test, y_pred_proba)
                    accuracy = accuracy_score(y_test, y_pred)
                    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    precision   = precision_score(y_test, y_pred, zero_division=0)
                    f1          = f1_score(y_test, y_pred, zero_division=0)

                    r = results[clf_name][model_name]
                    r['auc_scores'].append(auc)
                    r['auprc_scores'].append(auprc)
                    r['accuracy_scores'].append(accuracy)
                    r['sensitivity_scores'].append(sensitivity)
                    r['specificity_scores'].append(specificity)
                    r['precision_scores'].append(precision)
                    r['f1_scores'].append(f1)

                    # 记录患者在该fold中的预测
                    repeat_preds[clf_name][model_name]['y_pred'][test_idx] = y_pred_proba

        # 保存当前repeat的预测
        for clf_name in classifiers:
            for model_name in feature_sets:
                all_repeat_preds[clf_name][model_name].append(
                    repeat_preds[clf_name][model_name]['y_pred']
                )
        print(f"      总进度: {fold_counter}/{total_folds} folds")

    # 跨repeat平均预测（每个患者一条记录，用于ROC绘图和DeLong检验）
    for clf_name in classifiers:
        for model_name in feature_sets:
            stacked = np.vstack(all_repeat_preds[clf_name][model_name])
            avg_pred = np.nanmean(stacked, axis=0)
            results[clf_name][model_name]['y_true_final'] = y
            results[clf_name][model_name]['y_pred_final'] = avg_pred

    # 生成汇总表（AUC均值±标准差基于50折）
    summary_data = []
    for clf_name in classifiers.keys():
        for model_name in feature_sets.keys():
            r = results[clf_name][model_name]
            summary_data.append({
                '分类器': clf_name,
                '分组': group_name,
                '模型': model_name,
                'AUC_均值':         np.mean(r['auc_scores']),
                'AUC_标准差':       np.std(r['auc_scores']),
                'AUPRC_均值':       np.mean(r['auprc_scores']),
                'AUPRC_标准差':     np.std(r['auprc_scores']),
                'Accuracy_均值':    np.mean(r['accuracy_scores']),
                'Accuracy_标准差':  np.std(r['accuracy_scores']),
                'Sensitivity_均值': np.mean(r['sensitivity_scores']),
                'Sensitivity_标准差': np.std(r['sensitivity_scores']),
                'Specificity_均值': np.mean(r['specificity_scores']),
                'Specificity_标准差': np.std(r['specificity_scores']),
                'Precision_均值':   np.mean(r['precision_scores']),
                'Precision_标准差': np.std(r['precision_scores']),
                'F1_均值':          np.mean(r['f1_scores']),
                'F1_标准差':        np.std(r['f1_scores'])
            })

    summary_df = pd.DataFrame(summary_data)

    # DeLong检验（在汇聚的跨repeat平均预测上）
    comparisons = [
        ('B_real_age',  'A_baseline', '实际年龄的价值'),
        ('C_faceage',   'A_baseline', 'FaceAge的价值'),
        ('D_age_gap', 'B_real_age', 'Age Gap的增量价值（在有实际年龄时）'),
        ('C_faceage',   'B_real_age', 'FaceAge能否替代实际年龄'),
        ('E_age_gap_only', 'A_baseline', 'Age Gap独立价值（无实际年龄时）'),
        ('E_age_gap_only', 'C_faceage', 'Age Gap vs FaceAge直接对比')
    ]

    comparison_results = []
    for clf_name in classifiers.keys():
        for model1, model2, description in comparisons:
            y_true_final = results[clf_name][model1]['y_true_final']
            y_pred1 = results[clf_name][model1]['y_pred_final']
            y_pred2 = results[clf_name][model2]['y_pred_final']

            z, p_value, auc1, auc2 = delong_roc_test(y_true_final, y_pred1, y_pred2)
            sig = '***' if p_value < 0.01 else '**' if p_value < 0.05 else '*' if p_value < 0.1 else 'ns'
            comparison_results.append({
                '分类器': clf_name,
                '分组':   group_name,
                '对比':   f'{model1} vs {model2}',
                '描述':   description,
                'AUC差异': auc1 - auc2,
                'p值':    p_value,
                '显著性': sig
            })

    comparison_df = pd.DataFrame(comparison_results)
    return results, summary_df, comparison_df


# ============================================================================
# 4. 分别运行两组建模
# ============================================================================

groups = [
    (df_all,     'All patients (N = 1,470)'),
]

all_results  = {}
all_summaries = []
all_comparisons = []

for i, (df_grp, grp_name) in enumerate(groups):
    print(f"\n[{i+3}/6] Running model comparison on {grp_name}...")
    res, summ, comp = run_model_comparison_for_group(df_grp, grp_name, feature_sets, classifiers)
    all_results[grp_name]  = res
    all_summaries.append(summ)
    all_comparisons.append(comp)

print()

# ============================================================================
# 5. 汇总并保存结果
# ============================================================================

print("[5/6] Saving results...")

summary_all    = pd.concat(all_summaries,    ignore_index=True)
comparison_all = pd.concat(all_comparisons,  ignore_index=True)

output_dir = RESULTS_DIR / "ml_modeling" / "cumulative_occlusion"
output_dir.mkdir(parents=True, exist_ok=True)

# 保存CSV（处理文件被IDE锁定的情况）
def _safe_save_csv(df, filepath, encoding='utf-8-sig'):
    """尝试保存CSV，如被锁定则保存到备用文件名"""
    try:
        df.to_csv(filepath, index=False, encoding=encoding)
    except PermissionError:
        alt_path = filepath.parent / f"{filepath.stem}_updated{filepath.suffix}"
        df.to_csv(alt_path, index=False, encoding=encoding)
        print(f"  [!] 原文件被锁定，已保存到: {alt_path}")

_safe_save_csv(summary_all, output_dir / "cumulative_model_results.csv")
_safe_save_csv(comparison_all, output_dir / "cumulative_statistical_tests.csv")

print(f"  [OK] 保存模型结果: {output_dir / 'cumulative_model_results.csv'}")
print(f"  [OK] 保存统计检验: {output_dir / 'cumulative_statistical_tests.csv'}")
print()

# --- 生成论文格式表格（遮挡<=2组，可直接复制到文章）---
# 模型名映射
_MODEL_DISPLAY = {
    'A_baseline': 'A (baseline)',
    'B_real_age': 'B (+ age)',
    'C_faceage': 'C (+ FaceAge)',
    'D_age_gap': 'D (+ age + Age Gap)',
    'E_age_gap_only': 'E (+ Age Gap)'
}
_CLF_DISPLAY = {'RandomForest': 'RF', 'LogisticRegression': 'LR'}

paper_df = summary_all[summary_all['分组'] == 'All patients (N = 1,470)'].copy()
paper_rows = []
for _, row in paper_df.iterrows():
    paper_rows.append({
        'Classifier': _CLF_DISPLAY.get(row['分类器'], row['分类器']),
        'Model': _MODEL_DISPLAY.get(row['模型'], row['模型']),
        'AUC': f"{row['AUC_均值']:.3f} ± {row['AUC_标准差']:.3f}",
        'AUPRC': f"{row['AUPRC_均值']:.3f} ± {row['AUPRC_标准差']:.3f}",
        'Accuracy': f"{row['Accuracy_均值']:.3f} ± {row['Accuracy_标准差']:.3f}",
        'Sensitivity': f"{row['Sensitivity_均值']:.3f} ± {row['Sensitivity_标准差']:.3f}",
        'Specificity': f"{row['Specificity_均值']:.3f} ± {row['Specificity_标准差']:.3f}",
        'F1': f"{row['F1_均值']:.3f} ± {row['F1_标准差']:.3f}",
    })
paper_table = pd.DataFrame(paper_rows)
# 按 RF 先然后 LR，按模型 A→E 排序
_model_order = list(_MODEL_DISPLAY.keys())
paper_table['_clf_sort'] = paper_table['Classifier'].map({'RF': 0, 'LR': 1})
paper_table['_model_sort'] = paper_table['Model'].map(
    {v: i for i, v in enumerate(_MODEL_DISPLAY.values())}
)
paper_table = paper_table.sort_values(['_clf_sort', '_model_sort']).drop(
    columns=['_clf_sort', '_model_sort']
)

paper_path = output_dir / "cumulative_model_paper_table.csv"
try:
    paper_table.to_csv(paper_path, index=False, encoding='utf-8-sig')
    print(f"  [OK] 论文格式表格: {paper_path}")
except PermissionError:
    alt_path = output_dir / "cumulative_model_paper_table_updated.csv"
    paper_table.to_csv(alt_path, index=False, encoding='utf-8-sig')
    print(f"  [!] 原文件被锁定，已保存到: {alt_path}")
print()

# ============================================================================
# 6. 绘制累积分层ROC曲线（每个分类器1行，共2行×3列）
# ============================================================================

print("[6/6] Drawing ROC and PRC curves...")

colors = {
    'A_baseline':  '#1f77b4',
    'B_real_age':  '#ff7f0e',
    'C_faceage':   '#2ca02c',
    'D_age_gap': '#d62728',
    'E_age_gap_only': '#9467bd'
}
labels = {
    'A_baseline':  'A: Baseline',
    'B_real_age':  'B: + Chronological Age',
    'C_faceage':   'C: + FaceAge',
    'D_age_gap': 'D: + Age + Age Gap',
    'E_age_gap_only': 'E: + Age Gap'
}

# Save individual ROC figures per classifier
df_grp, grp_name = groups[0]

for clf_name in classifiers.keys():
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    res = all_results[grp_name][clf_name]

    for model_name in feature_sets.keys():
        y_true_final = res[model_name]['y_true_final']
        y_pred_final = res[model_name]['y_pred_final']
        fpr, tpr, _ = roc_curve(y_true_final, y_pred_final)
        auc = roc_auc_score(y_true_final, y_pred_final)
        ax.plot(fpr, tpr, color=colors[model_name], linewidth=2,
                label=f'{labels[model_name]} (AUC={auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')

    suffix = 'rf' if clf_name == 'RandomForest' else 'lr'
    fpath = output_dir / f"roc_{suffix}.png"
    plt.tight_layout()
    plt.savefig(fpath, dpi=DPI_DEFAULT, bbox_inches='tight')
    print(f"  [OK] Saved: {fpath}")
    plt.close()

print()

# ============================================================================
# 绘制累积分层PRC曲线（每个分类器1行，共2行×2列）
# ============================================================================

print("Drawing PRC curves...")

# Save individual PRC figures per classifier
for clf_name in classifiers.keys():
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    res = all_results[grp_name][clf_name]

    # Baseline precision (positive class proportion)
    y_final = res[list(feature_sets.keys())[0]]['y_true_final']
    baseline_precision = sum(y_final) / len(y_final)

    for model_name in feature_sets.keys():
        y_true_final = res[model_name]['y_true_final']
        y_pred_final = res[model_name]['y_pred_final']
        precision, recall, _ = precision_recall_curve(y_true_final, y_pred_final)
        auprc = average_precision_score(y_true_final, y_pred_final)
        ax.plot(recall, precision, color=colors[model_name], linewidth=2,
                label=f'{labels[model_name]} (AUPRC={auprc:.3f})')

    ax.plot([0, 1], [baseline_precision, baseline_precision], 'k--', linewidth=1, alpha=0.5,
            label=f'Random ({baseline_precision:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_aspect('equal')

    suffix = 'rf' if clf_name == 'RandomForest' else 'lr'
    fpath = output_dir / f"prc_{suffix}.png"
    plt.tight_layout()
    plt.savefig(fpath, dpi=DPI_DEFAULT, bbox_inches='tight')
    print(f"  [OK] Saved: {fpath}")
    plt.close()
print()


# ============================================================================
# 7. 生成分析报告
# ============================================================================

print("[6/6] Generating analysis report...")

report_lines = [
    "# 方向2 - 按遮挡累积分层分析报告\n",
    f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
    "---\n",
    "\n## 1. 研究目标\n",
    "评估 FaceAge 的预测价值是否随着遮挡程度的增加而下降。\n",
    "由于遮挡<=0组（纯无遮挡）死亡率较低（9.0%）且死亡患者集中在有遮挡群体，\n",
    "该组与其他组存在人群异质性，因此仅保留2组累积对比。\n",
    "使用 Random Forest 和 Logistic Regression 两种分类器进行对比，验证结论的稳健性。\n",
    "\n## 2. 分层方案（累积）\n",
]

for df_grp, grp_name in groups:
    report_lines.append(
        f"- **{grp_name}**: {len(df_grp)} 例，死亡 {int(df_grp['7天死亡'].sum())} 例"
        f"（{df_grp['7天死亡'].mean()*100:.1f}%）\n"
    )

report_lines.extend([
    "\n## 3. 模型设计\n",
    "每组内对比5个模型：\n",
    "- A: 基线（9个快速评估指标：性别、体温、心率、呼吸频率、收缩压、舒张压、血氧饱和度、GCS评分、血糖）\n",
    "- B: A + 实际年龄\n",
    "- C: A + FaceAge\n",
    "- D: A + 实际年龄 + Age Gap（生物学年龄 - 实际年龄）\n",
    "- E: A + Age Gap（不依赖实际年龄）\n",
    "\n### 算法\n",
    "- Random Forest (n=100, depth=10, balanced)\n",
    "- Logistic Regression (L2, C=1.0, balanced)\n",
    "- 验证方法: 5次重复×10折分层交叉验证（共50折），取跨repeat平均预测值\n",
    "- 统计检验: DeLong检验（DeLong, DeLong & Clarke-Pearson, 1988），在汇聚预测值上比较AUC\n",
    "- 显著性水平: p < 0.1 标记为*，p < 0.05 标记为**，p < 0.01 标记为***\n",
    "- AUC汇报: 50折的均值±标准差\n",
    "\n## 4. 各组模型性能\n"
])

# Per-classifier tables for each group
for i, (df_grp, grp_name) in enumerate(groups):
    summ = all_summaries[i]
    report_lines.append(f"\n### 4.{i+1} {grp_name}\n")

    for clf_name in classifiers.keys():
        clf_summ = summ[summ['分类器'] == clf_name]
        report_lines.append(f"\n#### {clf_name}\n")
        report_lines.append("\n| 模型 | AUC | AUPRC | Accuracy | Sensitivity | Specificity | F1 |\n")
        report_lines.append("|------|-----|----------|-------------|-------------|----|\n")
        for _, row in clf_summ.iterrows():
            report_lines.append(
                f"| {row['模型']} | {row['AUC_均值']:.3f}±{row['AUC_标准差']:.3f} | "
                f"{row['AUPRC_均值']:.3f}±{row['AUPRC_标准差']:.3f} | "
                f"{row['Accuracy_均值']:.3f} | {row['Sensitivity_均值']:.3f} | "
                f"{row['Specificity_均值']:.3f} | {row['F1_均值']:.3f} |\n"
            )

report_lines.extend([
    "\n## 5. 组内统计检验（DeLong检验, α=0.1: * p<0.1, ** p<0.05, *** p<0.01）\n"
])

for clf_name in classifiers.keys():
    clf_comp = comparison_all[comparison_all['分类器'] == clf_name]
    report_lines.append(f"\n### {clf_name}\n")
    report_lines.append("\n| 分组 | 对比 | AUC差异 | p值 | 显著性 |\n")
    report_lines.append("|------|------|---------|-----|--------|\n")
    for _, row in clf_comp.iterrows():
        report_lines.append(
            f"| {row['分组']} | {row['对比']} | {row['AUC差异']:+.4f} | "
            f"{row['p值']:.4f} | {row['显著性']} |\n"
        )

# 关键趋势：C_faceage vs A_baseline 的 AUC差异 随遮挡增加的变化
report_lines.extend(["\n## 6. FaceAge价值随遮挡增加的趋势\n"])

for clf_name in classifiers.keys():
    clf_summ_all = summary_all[summary_all['分类器'] == clf_name]
    report_lines.append(f"\n### {clf_name}\n")
    report_lines.append("| 分组 | C_faceage AUC | A_baseline AUC | AUC差异 | Sensitivity差异 | F1差异 |\n")
    report_lines.append("|------|--------------|----------------|---------|-----------------|--------|\n")

    for i, (df_grp, grp_name) in enumerate(groups):
        summ = all_summaries[i]
        clf_summ = summ[summ['分类器'] == clf_name]
        auc_A  = clf_summ[clf_summ['模型'] == 'A_baseline']['AUC_均值'].values[0]
        auc_C  = clf_summ[clf_summ['模型'] == 'C_faceage']['AUC_均值'].values[0]
        sens_A = clf_summ[clf_summ['模型'] == 'A_baseline']['Sensitivity_均值'].values[0]
        sens_C = clf_summ[clf_summ['模型'] == 'C_faceage']['Sensitivity_均值'].values[0]
        f1_A   = clf_summ[clf_summ['模型'] == 'A_baseline']['F1_均值'].values[0]
        f1_C   = clf_summ[clf_summ['模型'] == 'C_faceage']['F1_均值'].values[0]
        report_lines.append(
            f"| {grp_name} | {auc_C:.3f} | {auc_A:.3f} | {auc_C-auc_A:+.4f} | "
            f"{sens_C-sens_A:+.4f} | {f1_C-f1_A:+.4f} |\n"
        )

report_lines.extend([
    "\n## 7. 结论\n",
    "- 本分析通过累积纳入遮挡样本，观察FaceAge价值的梯度变化\n",
    "- 若FaceAge价值从组1（排除中度）到组2（含中度）下降，说明遮挡是影响因素\n",
    "- 若FaceAge价值在两组保持稳定，说明遮挡不是主要影响因素\n",
    "- 使用Random Forest和Logistic Regression两种分类器交叉验证结论的稳健性\n",
    "- 采用5次重复×10折CV增加统计稳定性，DeLong检验提供AUC比较的金标准\n",
    "\n## 8. 输出文件\n",
    "- `cumulative_model_results.csv`: 累积分层模型结果（含分类器列）\n",
    "- `cumulative_statistical_tests.csv`: 累积分层统计检验（含分类器列）\n",
    "- `roc_curves_cumulative.png`: 累积分层ROC曲线（2行×2列）\n",
    "- `prc_curves_cumulative.png`: 累积分层PRC曲线（2行×2列）\n",
    "- `cumulative_analysis_report.md`: 本报告\n",
    "\n---\n",
    "*本报告由遮挡累积分层分析脚本自动生成*\n"
])

report_path = output_dir / "cumulative_analysis_report.md"
with open(report_path, 'w', encoding='utf-8') as f:
    f.writelines(report_lines)

print(f"  [OK] 保存报告: {report_path}")
print()

# ============================================================================
# 完成
# ============================================================================

print("=" * 80)
print("[OK] 遮挡累积分层分析完成！")
print("=" * 80)
print(f"\n结果保存在: {output_dir}")
print("\n生成文件:")
print("  1. cumulative_model_results.csv - 累积分层模型结果")
print("  2. cumulative_statistical_tests.csv - 累积分层统计检验")
print("  3. roc_curves_cumulative.png - 累积分层ROC曲线（2行×2列）")
print("  4. prc_curves_cumulative.png - 累积分层PRC曲线（2行×2列）")
print("  5. cumulative_analysis_report.md - 累积分层分析报告")
print()

# 打印关键结论（每个分类器、每个分组）
for clf_name in classifiers.keys():
    print(f"[{clf_name}]")
    for i, (df_grp, grp_name) in enumerate(groups):
        summ = all_summaries[i]
        clf_summ = summ[summ['分类器'] == clf_name]
        auc_A = clf_summ[clf_summ['模型'] == 'A_baseline']['AUC_均值'].values[0]
        auc_B = clf_summ[clf_summ['模型'] == 'B_real_age']['AUC_均值'].values[0]
        auc_C = clf_summ[clf_summ['模型'] == 'C_faceage']['AUC_均值'].values[0]
        print(f"  {grp_name}: A={auc_A:.3f}, B={auc_B:.3f}, C={auc_C:.3f}")
    print()
