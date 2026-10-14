"""Table 2: baseline characteristics of the cohort.

Demographics, vital signs, laboratory values, severity scores and FaceAge measures, tabulated for
the whole cohort and for the no-occlusion and occluded subgroups. This is the file the
manuscript's baseline-characteristics table is built from.

Note on the file name: this script is named after the table's original number in an early draft.
It produces the manuscript's Table 2.

Input  : data/merged_data.xlsx
Output : results/table1/patient_baseline_characteristics.csv
         results/table1/patient_baseline_characteristics.md"""

import os
import sys
import io
from pathlib import Path

# 修复 Windows 控制台编码问题
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 确保可以导入 utils 模块
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import stats
from utils.data_loader import load_from_merged_xlsx
from utils.config import RESULTS_DIR, QUICK_ASSESSMENT_FEATURES


def format_continuous(mean, std):
    """格式化连续变量: mean ± SD"""
    return f"{mean:.1f} ± {std:.1f}"


def format_categorical(count, total):
    """格式化分类变量: n (%)"""
    if total == 0:
        return "0 (0%)"
    pct = count / total * 100
    return f"{int(count)} ({pct:.1f}%)"


def sig_marker(p_val):
    """显著性标记"""
    if p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    return ""


def generate_table1():
    """生成患者基线特征表"""
    print("=" * 60)
    print("Table 1: 患者基线特征表生成")
    print("=" * 60)

    # ────────────────────────────────────────
    # 1. 加载数据
    # ────────────────────────────────────────
    print("\n[1/3] 加载数据...")
    df_all = load_from_merged_xlsx(max_occlusion_level=2)
    print(f"  全样本: {len(df_all)} 例")

    # 分层
    df_no_occ = df_all[df_all['遮挡等级'] == 0].copy()
    df_with_occ = df_all[df_all['遮挡等级'].isin([1, 2])].copy()
    print(f"  无遮挡组 (Level 0): {len(df_no_occ)} 例")
    print(f"  有遮挡组 (Level 1+2): {len(df_with_occ)} 例")

    N_all = len(df_all)
    N_no = len(df_no_occ)
    N_occ = len(df_with_occ)

    # 死亡率
    print(f"  总体死亡率: {df_all['7天死亡'].mean() * 100:.1f}%")
    print(f"  无遮挡死亡率: {df_no_occ['7天死亡'].mean() * 100:.1f}%")
    print(f"  有遮挡死亡率: {df_with_occ['7天死亡'].mean() * 100:.1f}%")

    # ────────────────────────────────────────
    # 2. 确定性别编码
    # ────────────────────────────────────────
    sex_vals = sorted(df_all['性别'].dropna().unique())
    print(f"\n  性别编码值: {sex_vals}")
    # 常见医学数据编码: 1=男, 0/2=女
    if 1.0 in sex_vals:
        male_val = 1.0
    else:
        male_val = max(sex_vals)
    print(f"  男性编码值: {male_val}")

    # ────────────────────────────────────────
    # 3. 特征定义
    # ────────────────────────────────────────
    feature_categories = [
        ("人口学特征", [
            ("实际年龄", "年龄（岁）", "continuous"),
            ("性别", "性别（男性）", "categorical"),
        ]),
        ("生命体征（快速评估指标）", [
            ("体温", "体温（℃）", "continuous"),
            ("心率", "心率（次/分）", "continuous"),
            ("呼吸频率", "呼吸频率（次/分）", "continuous"),
            ("收缩压", "收缩压（mmHg）", "continuous"),
            ("舒张压", "舒张压（mmHg）", "continuous"),
            ("血氧饱和度", "SpO₂（%）", "continuous"),
            ("GCS评分", "GCS评分", "continuous"),
            ("血糖", "血糖（mmol/L）", "continuous"),
        ]),
        ("实验室检查", [
            ("PH", "pH", "continuous"),
            ("氧分压", "PaO₂（mmHg）", "continuous"),
            ("乳酸Lac", "乳酸（mmol/L）", "continuous"),
        ]),
        ("疾病严重度", [
            ("APACHE评分", "APACHE II评分", "continuous"),
        ]),
        ("FaceAge评估", [
            ("生物学年龄", "FaceAge生物学年龄（岁）", "continuous"),
            ("Age_Gap", "Age Gap（岁）", "continuous"),
        ]),
        ("结局", [
            ("7天死亡", "7天死亡", "binary"),
        ]),
    ]

    # ────────────────────────────────────────
    # 4. 计算统计量
    # ────────────────────────────────────────
    print("\n[2/3] 计算统计量...")
    table_rows = []

    for i, (category, features) in enumerate(feature_categories):
        # 类别标题行
        table_rows.append({
            '类别': category,
            '特征': '',
            f'全样本 (N={N_all})': '',
            f'无遮挡 (N={N_no})': '',
            f'有遮挡 (N={N_occ})': '',
            'P值': '',
        })

        for j, (col_name, display_name, var_type) in enumerate(features):
            v_all = df_all[col_name].dropna()
            v_no = df_no_occ[col_name].dropna()
            v_occ = df_with_occ[col_name].dropna()

            if var_type == 'continuous':
                all_str = format_continuous(v_all.mean(), v_all.std())
                no_str = format_continuous(v_no.mean(), v_no.std())
                occ_str = format_continuous(v_occ.mean(), v_occ.std())

                # 独立样本 t 检验（无遮挡 vs 有遮挡）
                t_stat, p_val = stats.ttest_ind(v_no, v_occ)
                p_str = f"{p_val:.4f} {sig_marker(p_val)}".strip()

            elif var_type == 'categorical':
                # 性别: 报告男性百分比
                n_male_all = int((v_all == male_val).sum())
                n_male_no = int((v_no == male_val).sum())
                n_male_occ = int((v_occ == male_val).sum())

                all_str = format_categorical(n_male_all, len(v_all))
                no_str = format_categorical(n_male_no, len(v_no))
                occ_str = format_categorical(n_male_occ, len(v_occ))

                # 卡方检验
                cont_table = [
                    [n_male_no, len(v_no) - n_male_no],
                    [n_male_occ, len(v_occ) - n_male_occ],
                ]
                chi2, p_val, dof, _ = stats.chi2_contingency(cont_table)
                p_str = f"{p_val:.4f} {sig_marker(p_val)}".strip()

            elif var_type == 'binary':
                # 7天死亡: 报告死亡人数和百分比
                n_pos_all = int(v_all.sum())
                n_pos_no = int(v_no.sum())
                n_pos_occ = int(v_occ.sum())

                all_str = format_categorical(n_pos_all, len(v_all))
                no_str = format_categorical(n_pos_no, len(v_no))
                occ_str = format_categorical(n_pos_occ, len(v_occ))

                # 卡方检验
                cont_table = [
                    [n_pos_no, len(v_no) - n_pos_no],
                    [n_pos_occ, len(v_occ) - n_pos_occ],
                ]
                chi2, p_val, dof, _ = stats.chi2_contingency(cont_table)
                p_str = f"{p_val:.4f} {sig_marker(p_val)}".strip()

            table_rows.append({
                '类别': category if j == 0 else '',
                '特征': display_name,
                f'全样本 (N={N_all})': all_str,
                f'无遮挡 (N={N_no})': no_str,
                f'有遮挡 (N={N_occ})': occ_str,
                'P值': p_str,
            })

    # ────────────────────────────────────────
    # 5. 保存结果
    # ────────────────────────────────────────
    print("\n[3/3] 保存结果...")
    output_dir = RESULTS_DIR / "table1"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建 DataFrame
    table_df = pd.DataFrame(table_rows)
    display_cols = ['类别', '特征',
                    f'全样本 (N={N_all})',
                    f'无遮挡 (N={N_no})',
                    f'有遮挡 (N={N_occ})',
                    'P值']

    # 保存 CSV
    csv_path = output_dir / "patient_baseline_characteristics.csv"
    table_df[display_cols].to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  CSV: {csv_path}")

    # ────────────────────────────────────────
    # 生成 Markdown 报告
    # ────────────────────────────────────────
    md_lines = []
    md_lines.append("# Table 1: 患者基线特征表\n")
    md_lines.append(f"**全样本**: N={N_all} | "
                    f"**无遮挡 (Level 0)**: N={N_no} | "
                    f"**有遮挡 (Level 1+2)**: N={N_occ}\n")
    md_lines.append(f"**总死亡率**: {df_all['7天死亡'].mean() * 100:.1f}% | "
                    f"**无遮挡死亡率**: {df_no_occ['7天死亡'].mean() * 100:.1f}% | "
                    f"**有遮挡死亡率**: {df_with_occ['7天死亡'].mean() * 100:.1f}%\n")
    md_lines.append("\n> P值: 无遮挡组 vs 有遮挡组。\\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001\n")
    md_lines.append("\n| 特征 | 全样本 | 无遮挡 | 有遮挡 | P值 |\n")
    md_lines.append("|------|--------|--------|--------|-----|\n")

    for _, row in table_df.iterrows():
        category = row['类别']
        feature = row['特征']
        all_val = row[f'全样本 (N={N_all})']
        no_val = row[f'无遮挡 (N={N_no})']
        occ_val = row[f'有遮挡 (N={N_occ})']
        p_val = row['P值']

        if feature == '':
            # 类别标题
            md_lines.append(f"| **{category}** | | | | |\n")
        else:
            md_lines.append(
                f"| {feature} | {all_val} | {no_val} | {occ_val} | {p_val} |\n"
            )

    md_content = ''.join(md_lines)

    md_path = output_dir / "patient_baseline_characteristics.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"  Markdown: {md_path}")

    # 打印摘要
    print("\n" + "=" * 60)
    print("Table 1 生成完成!")
    print("=" * 60)

    # 打印关键比较
    print("\n关键组间差异（有遮挡 vs 无遮挡）:")
    print("-" * 50)
    for _, row in table_df.iterrows():
        feature = row['特征']
        p_str = row['P值']
        if feature and '***' in p_str or '**' in p_str or '*' in p_str:
            print(f"  {feature}: {p_str}")


if __name__ == '__main__':
    generate_table1()
