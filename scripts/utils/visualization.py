"""Plotting building blocks shared across the analysis scripts.

Boxplots by group, scatter with trend, ROC curves, correlation heatmaps, Bland-Altman plots and
the age-stratified grid, all built on the palettes, font sizes and figure sizes defined in
config.py. The manuscript figures themselves are assembled by the individual scripts
(figure2_agreement_composite.py, 10_figures_mice.py, fig6_bias_corrected_roc.py); this module
supplies the parts."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
from .config import (
    FONT_CHINESE,
    COLORS_OCCLUSION,
    COLORS_AGE_GROUPS,
    COLORS_OUTCOME,
    FIGSIZE_SINGLE,
    FIGSIZE_GRID_2X2,
    FIGSIZE_ROC,
    DPI_DEFAULT
)

# Use English publication font
plt.rcParams['font.sans-serif'] = FONT_CHINESE
plt.rcParams['axes.unicode_minus'] = False


def plot_boxplot_by_group(data, group_col, value_col, title='', xlabel='', ylabel='',
                           colors=None, figsize=FIGSIZE_SINGLE, show_pvalues=None,
                           save_path=None, group_order=None):
    """
    通用箱线图

    参数:
        data: DataFrame
        group_col: 分组列名
        value_col: 值列名
        title: 图表标题
        xlabel, ylabel: 轴标签
        colors: 颜色列表
        figsize: 图表尺寸
        show_pvalues: 字典，键为(group1, group2)元组，值为p值
        save_path: 保存路径
        group_order: 自定义组显示顺序（list），为None时按数据出现顺序

    返回:
        fig, ax: matplotlib对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 准备数据
    groups = group_order if group_order is not None else list(data[group_col].unique())
    plot_data = [data[data[group_col] == group][value_col].values for group in groups]

    # 绘制箱线图
    bp = ax.boxplot(plot_data, labels=groups, patch_artist=True,
                     showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=8))

    # 设置颜色
    if colors is None:
        colors = COLORS_OCCLUSION[:len(groups)]

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # 添加p值标注
    if show_pvalues is not None:
        y_max = data[value_col].max()
        y_range = data[value_col].max() - data[value_col].min()
        y_offset = y_max + 0.05 * y_range

        for (g1, g2), p in show_pvalues.items():
            x1, x2 = list(groups).index(g1) + 1, list(groups).index(g2) + 1
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            ax.plot([x1, x2], [y_offset, y_offset], 'k-', linewidth=1)
            ax.text((x1 + x2) / 2, y_offset, sig, ha='center', va='bottom', fontsize=10)
            y_offset += 0.05 * y_range

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DPI_DEFAULT, bbox_inches='tight')
        print(f"  [保存] {save_path}")

    return fig, ax


def plot_scatter_with_trend(x, y, hue=None, hue_labels=None, title='', xlabel='', ylabel='',
                             colors=None, figsize=FIGSIZE_SINGLE, show_correlation=True,
                             save_path=None):
    """
    散点图+趋势线

    参数:
        x, y: 数据（array-like）
        hue: 分组变量（可选）
        hue_labels: 分组标签字典
        title, xlabel, ylabel: 标题和轴标签
        colors: 颜色列表/字典
        figsize: 图表尺寸
        show_correlation: 是否显示相关系数
        save_path: 保存路径

    返回:
        fig, ax: matplotlib对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    if hue is None:
        # 单组散点图
        ax.scatter(x, y, alpha=0.5, s=30, edgecolors='none')

        # 趋势线
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_trend, p(x_trend), "r-", linewidth=2, alpha=0.7, label='Trend line')

        # 相关系数
        if show_correlation:
            from scipy import stats
            r, p_val = stats.pearsonr(x, y)
            sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
            ax.text(0.05, 0.95, f'r = {r:.3f} {sig}', transform=ax.transAxes,
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        # 分组散点图
        unique_hues = np.unique(hue)
        if colors is None:
            colors = COLORS_OUTCOME if len(unique_hues) == 2 else COLORS_AGE_GROUPS

        for i, hue_val in enumerate(unique_hues):
            mask = hue == hue_val
            label = hue_labels.get(hue_val, str(hue_val)) if hue_labels else str(hue_val)
            color = colors.get(hue_val, colors[i]) if isinstance(colors, dict) else colors[i]
            ax.scatter(x[mask], y[mask], alpha=0.5, s=30, color=color,
                       label=label, edgecolors='none')

        # 整体趋势线
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_trend, p(x_trend), "k--", linewidth=2, alpha=0.5, label='Overall trend')

        # 相关系数
        if show_correlation:
            from scipy import stats
            r, p_val = stats.pearsonr(x, y)
            sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
            ax.text(0.05, 0.95, f'r = {r:.3f} {sig}', transform=ax.transAxes,
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.legend(fontsize=10)

    # 添加y=0参考线（如果y包含负值）
    if y.min() < 0:
        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DPI_DEFAULT, bbox_inches='tight')
        print(f"  [保存] {save_path}")

    return fig, ax


def plot_roc_curves(y_true, y_scores_dict, title='ROC Curves Comparison',
                    figsize=FIGSIZE_ROC, save_path=None):
    """
    多条ROC曲线对比

    参数:
        y_true: 真实标签（0/1）
        y_scores_dict: 字典，键为变量名，值为预测分数
        title: 图表标题
        figsize: 图表尺寸
        save_path: 保存路径

    返回:
        fig, ax: matplotlib对象
    """
    from sklearn.metrics import roc_auc_score

    fig, ax = plt.subplots(figsize=figsize)

    # 绘制对角线（随机猜测）
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random (AUC = 0.50)')

    # 颜色
    colors = plt.cm.tab10(np.linspace(0, 1, len(y_scores_dict)))

    # 绘制每条ROC曲线
    for (var_name, y_score), color in zip(y_scores_dict.items(), colors):
        # 过滤NaN
        mask = ~(np.isnan(y_true) | np.isnan(y_score))
        y_true_clean = np.array(y_true)[mask]
        y_score_clean = np.array(y_score)[mask]

        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(y_true_clean, y_score_clean)
        auc = roc_auc_score(y_true_clean, y_score_clean)

        # 绘制
        ax.plot(fpr, tpr, linewidth=2, color=color,
                label=f'{var_name} (AUC = {auc:.4f})')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DPI_DEFAULT, bbox_inches='tight')
        print(f"  [保存] {save_path}")

    return fig, ax


def plot_correlation_heatmap(data, columns, title='Correlation Heatmap',
                              figsize=(10, 8), save_path=None):
    """
    相关性热图

    参数:
        data: DataFrame
        columns: 要显示的列名列表
        title: 图表标题
        figsize: 图表尺寸
        save_path: 保存路径

    返回:
        fig, ax: matplotlib对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 计算相关矩阵
    corr_matrix = data[columns].corr()

    # 绘制热图
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax)

    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DPI_DEFAULT, bbox_inches='tight')
        print(f"  [保存] {save_path}")

    return fig, ax


def plot_bland_altman(y_true, y_pred, title='Bland-Altman Plot',
                      xlabel='Mean', ylabel='Difference',
                      figsize=FIGSIZE_SINGLE, save_path=None):
    """
    Bland-Altman 图（用于一致性分析）

    参数:
        y_true, y_pred: 真实值和预测值
        title, xlabel, ylabel: 标题和轴标签
        figsize: 图表尺寸
        save_path: 保存路径

    返回:
        fig, ax: matplotlib对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 计算均值和差值
    mean_vals = (y_true + y_pred) / 2
    diff_vals = y_pred - y_true

    # 计算统计量
    mean_diff = np.mean(diff_vals)
    std_diff = np.std(diff_vals)
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    # 散点图
    ax.scatter(mean_vals, diff_vals, alpha=0.5, s=30, edgecolors='none')

    # 均值线
    ax.axhline(mean_diff, color='red', linestyle='-', linewidth=2,
               label=f'Mean diff = {mean_diff:.2f}')

    # 一致性界限
    ax.axhline(loa_upper, color='red', linestyle='--', linewidth=2,
               label=f'+1.96 SD = {loa_upper:.2f}')
    ax.axhline(loa_lower, color='red', linestyle='--', linewidth=2,
               label=f'-1.96 SD = {loa_lower:.2f}')

    # 零线
    ax.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DPI_DEFAULT, bbox_inches='tight')
        print(f"  [保存] {save_path}")

    return fig, ax


def plot_age_stratified_grid(data_dict, plot_func, figsize=FIGSIZE_GRID_2X2,
                              title='', save_path=None, **kwargs):
    """
    年龄分层的2×2网格图

    参数:
        data_dict: 字典，键为年龄组名（'青年', '中年', '老年', '极高龄'），值为DataFrame
        plot_func: 绘图函数，接受(ax, data, **kwargs)参数
        figsize: 图表尺寸
        title: 总标题
        save_path: 保存路径
        **kwargs: 传递给plot_func的额外参数

    返回:
        fig, axes: matplotlib对象
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    age_groups = ['青年', '中年', '老年', '极高龄']

    for i, (group_name, ax) in enumerate(zip(age_groups, axes)):
        if group_name in data_dict:
            data = data_dict[group_name]
            plot_func(ax, data, group_name=group_name, **kwargs)
        else:
            ax.text(0.5, 0.5, f'{group_name}\nData missing', ha='center', va='center',
                    fontsize=14, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DPI_DEFAULT, bbox_inches='tight')
        print(f"  [保存] {save_path}")

    return fig, axes
