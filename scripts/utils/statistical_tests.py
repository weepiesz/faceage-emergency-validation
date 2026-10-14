"""Statistical tests shared across the analysis scripts.

ROC-AUC with bootstrap confidence intervals, Mann-Whitney U, Kruskal-Wallis and the paired
comparisons, all with a fixed seed so that repeated runs of the pipeline give identical intervals.
The analysis scripts call these rather than reimplementing them, which is what keeps the numbers
reported in the manuscript mutually comparable."""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    confusion_matrix
)
from .config import ALPHA


def mann_whitney_test(group1, group2, alternative='two-sided'):
    """
    Mann-Whitney U 检验（非参数检验，用于两组独立样本）

    参数:
        group1, group2: 两组数据（array-like）
        alternative: 'two-sided', 'less', or 'greater'

    返回:
        result: 字典，包含统计量、p值、显著性
    """
    statistic, p_value = stats.mannwhitneyu(group1, group2, alternative=alternative)

    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < ALPHA,
        'alpha': ALPHA
    }


def kruskal_wallis_test(*groups):
    """
    Kruskal-Wallis H 检验（非参数检验，用于多组独立样本）

    参数:
        *groups: 多个组的数据（可变参数）

    返回:
        result: 字典，包含统计量、p值、显著性
    """
    statistic, p_value = stats.kruskal(*groups)

    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < ALPHA,
        'alpha': ALPHA
    }


def pearson_correlation_test(x, y):
    """
    Pearson 相关性检验

    参数:
        x, y: 两个变量（array-like）

    返回:
        result: 字典，包含相关系数、p值、显著性
    """
    # 过滤NaN
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = np.array(x)[mask]
    y_clean = np.array(y)[mask]

    r, p_value = stats.pearsonr(x_clean, y_clean)

    return {
        'correlation': r,
        'p_value': p_value,
        'significant': p_value < ALPHA,
        'n_samples': len(x_clean),
        'alpha': ALPHA
    }


def spearman_correlation_test(x, y):
    """
    Spearman 秩相关检验（非参数）

    参数:
        x, y: 两个变量（array-like）

    返回:
        result: 字典，包含相关系数、p值、显著性
    """
    # 过滤NaN
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = np.array(x)[mask]
    y_clean = np.array(y)[mask]

    rho, p_value = stats.spearmanr(x_clean, y_clean)

    return {
        'correlation': rho,
        'p_value': p_value,
        'significant': p_value < ALPHA,
        'n_samples': len(x_clean),
        'alpha': ALPHA
    }


def calculate_regression_metrics(y_true, y_pred):
    """
    计算回归指标（用于年龄预测）

    参数:
        y_true: 真实值
        y_pred: 预测值

    返回:
        metrics: 字典，包含MAE、RMSE、R²、Pearson相关等
    """
    # 过滤NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = np.array(y_true)[mask]
    y_pred_clean = np.array(y_pred)[mask]

    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    r2 = r2_score(y_true_clean, y_pred_clean)
    pearson_r, pearson_p = stats.pearsonr(y_true_clean, y_pred_clean)

    # 误差分析
    errors = y_pred_clean - y_true_clean
    mae_within_5 = np.mean(np.abs(errors) <= 5) * 100
    mae_within_10 = np.mean(np.abs(errors) <= 10) * 100

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Pearson_r': pearson_r,
        'Pearson_p': pearson_p,
        'n_samples': len(y_true_clean),
        'error_within_5y_%': mae_within_5,
        'error_within_10y_%': mae_within_10
    }


def calculate_roc_auc(y_true, y_score):
    """
    计算ROC AUC及置信区间

    参数:
        y_true: 真实标签（0/1）
        y_score: 预测分数

    返回:
        result: 字典，包含AUC、95%CI、p值等
    """
    # 过滤NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_score))
    y_true_clean = np.array(y_true)[mask]
    y_score_clean = np.array(y_score)[mask]

    # 计算AUC
    try:
        auc = roc_auc_score(y_true_clean, y_score_clean)
    except ValueError:
        # 如果只有一个类别，返回NaN
        return {
            'AUC': np.nan,
            'CI_lower': np.nan,
            'CI_upper': np.nan,
            'p_value': np.nan,
            'n_samples': len(y_true_clean)
        }

    # Bootstrap 计算95%置信区间
    n_bootstraps = 1000
    rng = np.random.RandomState(42)
    bootstrapped_aucs = []

    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_true_clean), len(y_true_clean))
        if len(np.unique(y_true_clean[indices])) < 2:
            continue
        try:
            score = roc_auc_score(y_true_clean[indices], y_score_clean[indices])
            bootstrapped_aucs.append(score)
        except:
            continue

    ci_lower = np.percentile(bootstrapped_aucs, 2.5)
    ci_upper = np.percentile(bootstrapped_aucs, 97.5)

    # 检验AUC是否显著不同于0.5（无预测价值）
    # 使用Wilcoxon符号秩检验
    _, p_value = stats.wilcoxon([x - 0.5 for x in bootstrapped_aucs])

    return {
        'AUC': auc,
        'CI_lower': ci_lower,
        'CI_upper': ci_upper,
        'p_value': p_value,
        'significant_vs_0.5': p_value < ALPHA,
        'n_samples': len(y_true_clean),
        'n_positive': int(y_true_clean.sum()),
        'n_negative': int((1 - y_true_clean).sum())
    }


def roc_auc_analysis(y_true, y_scores_dict):
    """
    多变量ROC分析

    参数:
        y_true: 真实标签（0/1）
        y_scores_dict: 字典，键为变量名，值为预测分数

    返回:
        results_df: DataFrame，包含各变量的AUC、95%CI等
    """
    results = []

    for var_name, y_score in y_scores_dict.items():
        result = calculate_roc_auc(y_true, y_score)
        result['Variable'] = var_name
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df = results_df[['Variable', 'AUC', 'CI_lower', 'CI_upper',
                             'p_value', 'significant_vs_0.5', 'n_samples',
                             'n_positive', 'n_negative']]

    return results_df


def format_significance(p_value, alpha=ALPHA):
    """
    格式化显著性标记

    参数:
        p_value: p值
        alpha: 显著性水平

    返回:
        sig_mark: 显著性标记（"***", "**", "*", "ns"）
    """
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < alpha:
        return "*"
    else:
        return "ns"


def pairwise_comparisons(groups_dict, test='mann_whitney'):
    """
    多组两两比较

    参数:
        groups_dict: 字典，键为组名，值为数据
        test: 'mann_whitney' 或 'ttest'

    返回:
        results_df: DataFrame，包含两两比较的结果
    """
    results = []
    group_names = list(groups_dict.keys())

    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            name1, name2 = group_names[i], group_names[j]
            data1, data2 = groups_dict[name1], groups_dict[name2]

            if test == 'mann_whitney':
                stat, p = stats.mannwhitneyu(data1, data2)
            elif test == 'ttest':
                stat, p = stats.ttest_ind(data1, data2)
            else:
                raise ValueError(f"未知的test类型: {test}")

            results.append({
                'Group1': name1,
                'Group2': name2,
                'Statistic': stat,
                'p_value': p,
                'Significant': p < ALPHA,
                'Sig_mark': format_significance(p)
            })

    results_df = pd.DataFrame(results)
    return results_df
