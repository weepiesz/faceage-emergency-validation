"""Data loading and preprocessing.

load_from_merged_xlsx() is the entry point used by every analysis script. It reads the merged
workbook, renames the Chinese columns to the internal names (实际年龄, 生物学年龄, 7天死亡,
遮挡等级, APACHE评分), derives Age_Gap = 生物学年龄 - 实际年龄, and applies the occlusion filter:
max_occlusion_level=2 gives the analysis set (occlusion grade <= 2, n = 1,470) and
max_occlusion_level=3 the initial dataset (n = 1,485)."""

import h5py
import pandas as pd
import numpy as np
from pathlib import Path
from .config import (
    MERGED_DATA_H5,
    MERGED_DATA_XLSX,
    COMPLETE_SAMPLE_FILE,
    AGE_GROUPS_4,
    AGE_GROUPS_3,
    OCCLUSION_LEVELS,
    AGE_MODELS,
    AGE_MODEL_XLSX_COLS
)


def load_from_h5(max_occlusion_level=3, include_all_features=False, age_model='FaceAge'):
    """
    从 merged_data.h5 加载数据

    参数:
        max_occlusion_level: 最大遮挡等级（0-3），默认3（保留所有）
        include_all_features: 是否包含所有表格特征（用于ML建模），默认False
        age_model: 年龄预测模型，可选 'FaceAge', 'MiVOLO', 'DEX', 'DeepFace'
                   默认 'FaceAge' (向后兼容)

    返回:
        df: pandas DataFrame，包含以下列：
            - 文件名
            - 实际年龄
            - 生物学年龄
            - 7天死亡
            - APACHE评分
            - 遮挡等级
            - Age_Gap（生物学年龄 - 实际年龄）
            - 如果include_all_features=True，还包含所有其他表格特征
    """
    if age_model not in AGE_MODELS:
        raise ValueError(f"未知的age_model: {age_model}, 可选: {list(AGE_MODELS.keys())}")

    bio_age_key = AGE_MODELS[age_model]

    if not MERGED_DATA_H5.exists():
        raise FileNotFoundError(f"找不到数据文件: {MERGED_DATA_H5}")

    with h5py.File(MERGED_DATA_H5, 'r') as f:
        # 检查请求的模型数据是否存在
        available_keys = list(f['metadata'].keys())
        if bio_age_key not in available_keys:
            available_models = [m for m, k in AGE_MODELS.items() if k in available_keys]
            raise KeyError(
                f"HDF5 中找不到 {bio_age_key}, 可用模型: {available_models}. "
                f"请先运行 merge_predictions_to_h5.py 合并新模型预测."
            )

        # 读取生物学年龄
        biological_ages = f['metadata'][bio_age_key][:]

        # 读取表格数据
        numeric_data = f['tabular/numeric_data'][:]
        numeric_columns = [col.decode('utf-8') for col in f['tabular/numeric_columns'][:]]

        # 读取遮挡等级和文件名
        occlusion_levels = f['metadata/occlusion_levels'][:]
        image_filenames = [fn.decode('utf-8') for fn in f['metadata/image_filenames'][:]]

    if include_all_features:
        # 加载所有表格特征
        df = pd.DataFrame(numeric_data, columns=numeric_columns)
        df.insert(0, '文件名', image_filenames)
        df['生物学年龄'] = biological_ages
        df['遮挡等级'] = occlusion_levels

        # 找到年龄列并重命名为"实际年龄"
        age_col = None
        for col in df.columns:
            if '年龄' in col and col != '生物学年龄':
                age_col = col
                break
        if age_col and age_col != '实际年龄':
            df.rename(columns={age_col: '实际年龄'}, inplace=True)

        # 找到死亡列并重命名为"7天死亡"
        death_col = None
        for col in df.columns:
            if '死亡' in col or 'death' in col.lower():
                death_col = col
                break
        if death_col and death_col != '7天死亡':
            df.rename(columns={death_col: '7天死亡'}, inplace=True)

        # 找到APACHE列并重命名
        apache_col = None
        for col in df.columns:
            if 'apache' in col.lower():
                apache_col = col
                break
        if apache_col and apache_col != 'APACHE评分':
            df.rename(columns={apache_col: 'APACHE评分'}, inplace=True)
    else:
        # 只加载基本列（原有逻辑）
        # 提取关键列
        mortality = numeric_data[:, 0]  # 7天死亡
        apache_scores = numeric_data[:, 1]  # APACHE评分

        # 找到年龄列
        age_idx = None
        for i, col in enumerate(numeric_columns):
            if '年龄' in col:
                age_idx = i
                break

        if age_idx is None:
            raise ValueError("未找到年龄列")

        actual_ages = numeric_data[:, age_idx]

        # 创建DataFrame
        df = pd.DataFrame({
            '文件名': image_filenames,
            '实际年龄': actual_ages,
            '生物学年龄': biological_ages,
            '7天死亡': mortality,
            'APACHE评分': apache_scores,
            '遮挡等级': occlusion_levels
        })

    # 过滤有效数据（生物学年龄不是NaN，且遮挡等级在范围内）
    valid_mask = ~np.isnan(df['生物学年龄']) & (df['遮挡等级'] <= max_occlusion_level)
    df = df[valid_mask].reset_index(drop=True)

    # 计算 Age Gap
    df['Age_Gap'] = df['生物学年龄'] - df['实际年龄']

    return df


def load_complete_sample_data():
    """
    从 complete_sample_data.xlsx 加载完整样本数据

    返回:
        df: pandas DataFrame
    """
    if not COMPLETE_SAMPLE_FILE.exists():
        raise FileNotFoundError(f"找不到文件: {COMPLETE_SAMPLE_FILE}")

    df = pd.read_excel(COMPLETE_SAMPLE_FILE)
    return df


def load_from_merged_xlsx(max_occlusion_level=2, age_model='FaceAge'):
    """
    从 Clinical_dataset/merged_data.xlsx 加载数据（替代HDF5加载方式）

    参数:
        max_occlusion_level: 最大遮挡等级（0-3），默认2（排除重度遮挡）
        age_model: 年龄预测模型，可选 'FaceAge', 'MiVOLO', 'DEX', 'DeepFace'
                   默认 'FaceAge'

    返回:
        df: pandas DataFrame，列名与 load_from_h5 输出兼容：
            - 实际年龄
            - 生物学年龄
            - 7天死亡
            - APACHE评分
            - 遮挡等级
            - Age_Gap
            - 性别, 体温, 心率, 呼吸频率, 收缩压, 舒张压, 血氧饱和度, GCS评分, 血糖
            - 以及其他表格特征
    """
    if age_model not in AGE_MODEL_XLSX_COLS:
        available = list(AGE_MODEL_XLSX_COLS.keys())
        raise ValueError(
            f"不支持的 age_model: '{age_model}', 可选: {available}"
        )

    bio_age_col = AGE_MODEL_XLSX_COLS[age_model]

    if not MERGED_DATA_XLSX.exists():
        raise FileNotFoundError(f"找不到文件: {MERGED_DATA_XLSX}")

    df = pd.read_excel(MERGED_DATA_XLSX)

    # 检查请求的模型列是否存在
    if bio_age_col not in df.columns:
        available = [k for k, v in AGE_MODEL_XLSX_COLS.items()
                     if v in df.columns]
        raise KeyError(
            f"merged_data.xlsx 中找不到列 '{bio_age_col}'，"
            f"可用模型: {available}。"
            f"请先运行 merge_multimodel_to_xlsx.py 合并多模型预测。"
        )

    # 列名映射：xlsx → 内部统一命名
    column_map = {
        '年龄': '实际年龄',
        bio_age_col: '生物学年龄',   # 使用当前选中模型的生物学年龄列
        '7天内死亡': '7天死亡',
        'ApacheⅡ评分': 'APACHE评分',
        'occlusion_level': '遮挡等级',
    }
    df = df.rename(columns=column_map)

    # 过滤：生物学年龄有效 且 遮挡等级在范围内
    valid_mask = (
        df['生物学年龄'].notna() &
        (df['遮挡等级'] >= 0) &
        (df['遮挡等级'] <= max_occlusion_level)
    )
    df = df[valid_mask].reset_index(drop=True)

    # 计算 Age Gap
    df['Age_Gap'] = df['生物学年龄'] - df['实际年龄']

    return df


def filter_by_occlusion(df, max_level=2):
    """
    按遮挡等级过滤数据

    参数:
        df: DataFrame
        max_level: 最大遮挡等级（0=无, 1=轻, 2=中, 3=重）

    返回:
        df_filtered: 过滤后的DataFrame
    """
    return df[df['遮挡等级'] <= max_level].copy()


def split_by_age_group(df, age_groups='4groups'):
    """
    按年龄分层

    参数:
        df: DataFrame
        age_groups: '3groups' (旧版) 或 '4groups' (新版，包含≥85岁)

    返回:
        groups_dict: 字典，键为年龄组名称，值为对应的DataFrame
    """
    if age_groups == '4groups':
        groups_config = AGE_GROUPS_4
    elif age_groups == '3groups':
        groups_config = AGE_GROUPS_3
    else:
        raise ValueError(f"未知的age_groups类型: {age_groups}")

    groups_dict = {}
    for group_name, (min_age, max_age) in groups_config.items():
        mask = (df['实际年龄'] >= min_age) & (df['实际年龄'] < max_age)
        groups_dict[group_name] = df[mask].copy()

    return groups_dict


def split_by_outcome(df):
    """
    按预后分组（生存 vs 死亡）

    参数:
        df: DataFrame

    返回:
        survived, died: 两个DataFrame
    """
    survived = df[df['7天死亡'] == 0].copy()
    died = df[df['7天死亡'] == 1].copy()
    return survived, died


def get_sample_summary(df):
    """
    获取样本摘要信息

    参数:
        df: DataFrame

    返回:
        summary: 字典，包含样本量、年龄范围、死亡率等
    """
    summary = {
        '样本数': len(df),
        '实际年龄_均值': df['实际年龄'].mean(),
        '实际年龄_标准差': df['实际年龄'].std(),
        '实际年龄_范围': (df['实际年龄'].min(), df['实际年龄'].max()),
        '生物学年龄_均值': df['生物学年龄'].mean(),
        '生物学年龄_标准差': df['生物学年龄'].std(),
        'Age_Gap_均值': df['Age_Gap'].mean(),
        'Age_Gap_标准差': df['Age_Gap'].std(),
        'APACHE_均值': df['APACHE评分'].mean(),
        'APACHE_标准差': df['APACHE评分'].std(),
        '死亡率': df['7天死亡'].mean(),
        '遮挡分布': df['遮挡等级'].value_counts().to_dict()
    }
    return summary


def print_data_summary(df, title="数据摘要"):
    """
    打印数据摘要信息

    参数:
        df: DataFrame
        title: 标题
    """
    summary = get_sample_summary(df)

    print("=" * 60)
    print(title)
    print("=" * 60)
    print(f"样本数: {summary['样本数']}")
    print(f"实际年龄: {summary['实际年龄_均值']:.1f} ± {summary['实际年龄_标准差']:.1f} "
          f"(范围: {summary['实际年龄_范围'][0]:.1f}-{summary['实际年龄_范围'][1]:.1f})")
    print(f"生物学年龄: {summary['生物学年龄_均值']:.1f} ± {summary['生物学年龄_标准差']:.1f}")
    print(f"Age Gap: {summary['Age_Gap_均值']:.2f} ± {summary['Age_Gap_标准差']:.2f}")
    print(f"APACHE评分: {summary['APACHE_均值']:.1f} ± {summary['APACHE_标准差']:.1f}")
    print(f"7天死亡率: {summary['死亡率']*100:.1f}%")
    print(f"遮挡分布:")
    for level, count in sorted(summary['遮挡分布'].items()):
        level_name = OCCLUSION_LEVELS.get(level, f'Level {level}')
        print(f"  {level_name} (Level {level}): {count} ({count/summary['样本数']*100:.1f}%)")
    print()
