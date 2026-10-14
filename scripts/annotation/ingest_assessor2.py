"""Collect the second assessor's export: validate, reverse the de-identification, merge with assessor 1.

Input  : data/raw/assessor2/occlusion_annotations.csv
         (the file the assessor obtains via "export progress")
Output : results/annotation_agreement/occlusion_annotations_merged.xlsx
             merged table, with an agreement flag per patient
         results/annotation_agreement/ingest_report.txt
             validation report for this intake

This script only collects, validates, merges and previews. The full inter-rater analysis —
weighted kappa, PABAK, Gwet's AC1, bootstrap intervals and the sensitivity analysis — is produced
afterwards by analyze_inter_rater.py.

Usage:
    python scripts/annotation/ingest_assessor2.py
    python scripts/annotation/ingest_assessor2.py --input <dir or csv>
    python scripts/annotation/ingest_assessor2.py --out <dir>"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FACEAGE = PROJECT_ROOT / "faceage_emergency_study"
DROP_DIR = FACEAGE / "data" / "raw" / "assessor2"
KEY_MAPPING = FACEAGE / "scripts" / "annotation" / "key_mapping.csv"
IMG_DIR = FACEAGE / "annotation_package" / "images"
OUT_DIR = FACEAGE / "results" / "annotation_agreement"

CSV_COLS = ["序号", "文件名", "遮挡等级", "耗时秒", "标注时间"]
VALID_LEVELS = {0, 1, 2, 3, 4}
LEVEL_NAMES = {0: "无遮挡或细鼻管", 1: "嘴部插管或粗鼻管",
               2: "呼吸机较大遮挡", 3: "包扎或血污严重遮挡", 4: "无法判读"}

# 报告里同时打印和落盘
_lines = []


def say(msg=""):
    print(msg)
    _lines.append(msg)


def rel(p: Path) -> str:
    """输出目录可能被 --out 指到项目外，此时退回绝对路径。"""
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def read_csv_any(path: Path):
    """标注员的 csv 在 Windows 上可能被 Excel 另存为 GBK，两种都试。"""
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法识别编码: {path}")


def find_input(input_arg):
    """定位待收取的 csv。多个候选时选已标注数量最多的那份。"""
    if input_arg:
        p = Path(input_arg)
        if p.is_file():
            cands = [p]
        elif p.is_dir():
            cands = sorted(p.glob("*.csv"))
        else:
            say(f"[错误] 路径不存在: {p}")
            return None, []
    else:
        if not DROP_DIR.exists():
            say(f"[错误] 收取目录不存在: {DROP_DIR}")
            return None, []
        cands = sorted(DROP_DIR.glob("*.csv"))

    if not cands:
        say(f"[错误] 没有找到任何 .csv")
        say(f"        请把标注员导出的 occlusion_annotations.csv 放到:")
        say(f"        {DROP_DIR}")
        return None, []

    scored = []
    for c in cands:
        try:
            df = read_csv_any(c)
        except Exception as e:
            say(f"        [跳过] {c.name} —— 无法读取（{e}）")
            continue
        if "文件名" not in df.columns or "遮挡等级" not in df.columns:
            say(f"        [跳过] {c.name} —— 缺少必要列")
            continue
        n = int(df["遮挡等级"].notna().sum())
        scored.append((c, n, df))

    if not scored:
        say("[错误] 候选文件都不可用")
        return None, []

    scored.sort(key=lambda t: t[1], reverse=True)
    best = scored[0]
    return best, scored


def cohen_kappa(a, b, labels):
    """未加权 Cohen's kappa（手写，避免依赖 sklearn）。"""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = 0.0
    for lv in labels:
        pe += (sum(1 for x in a if x == lv) / n) * (sum(1 for y in b if y == lv) / n)
    if pe >= 1.0:
        return float("nan")
    return (po - pe) / (1 - pe)


def main() -> int:
    ap = argparse.ArgumentParser(description="收取并合并第二位标注员的结果")
    ap.add_argument("--input", help="csv 文件或所在目录（默认 data/raw/assessor2/）")
    ap.add_argument("--out", help="输出目录（默认 results/annotation_agreement/）")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else OUT_DIR

    say("=" * 72)
    say("收取第二位标注员的结果")
    say("=" * 72)
    say(f"时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    say()

    # --- 定位输入 ---------------------------------------------------------
    best, scored = find_input(args.input)
    if best is None:
        return 1

    if len(scored) > 1:
        say(f"[1] 在目录里发现 {len(scored)} 个候选 csv：")
        for c, n, _ in scored:
            mark = "  <- 选用这份（已标注最多）" if c == best[0] else ""
            say(f"      {c.name}    已标注 {n} 张{mark}")
    else:
        say(f"[1] 收取文件: {best[0].name}")

    chosen_path, chosen_n, df2 = best
    say()
    say(f"    文件: {chosen_path}")
    say(f"    已标注 {chosen_n} 张")
    say()

    # --- 校验 -------------------------------------------------------------
    say("[2] 校验")
    problems = []

    missing_cols = [c for c in CSV_COLS if c not in df2.columns]
    if missing_cols:
        problems.append(f"缺少列: {missing_cols}")
    say(f"    列名: {list(df2.columns)}")

    names2 = df2["文件名"].astype(str).str.strip()
    dup = names2[names2.duplicated()].unique().tolist()
    if dup:
        problems.append(f"文件名重复 {len(dup)} 个: {dup[:5]}")
    say(f"    文件名重复: {len(dup)}")

    disk = sorted(p.name for p in IMG_DIR.glob("*.jpg")) if IMG_DIR.exists() else []
    got, exp = set(names2), set(disk)
    extra, missing = got - exp, exp - got
    if extra:
        problems.append(f"出现不认识的文件名 {len(extra)} 个: {sorted(extra)[:5]}")
    if missing:
        problems.append(f"缺少 {len(missing)} 个文件名: {sorted(missing)[:5]}")
    say(f"    与交付图像的差异: 多出 {len(extra)}，缺少 {len(missing)}（应为 0/0）")

    lv = pd.to_numeric(df2["遮挡等级"], errors="coerce")
    bad_lv = sorted(set(lv.dropna().astype(int)) - VALID_LEVELS)
    if bad_lv:
        problems.append(f"非法等级值: {bad_lv}")
    say(f"    非法等级值: {bad_lv if bad_lv else '无'}")

    say()
    if problems:
        say("    ✗ 发现以下问题：")
        for p in problems:
            say(f"        - {p}")
        say()
        say("    请先确认数据来源无误再继续。下面仍会输出合并表，供排查。")
    else:
        say("    ✓ 校验通过")
    say()

    # --- 还原并合并 -------------------------------------------------------
    say("[3] 还原去标识并合并")
    if not KEY_MAPPING.exists():
        say(f"[错误] 找不到映射表: {KEY_MAPPING}")
        return 1
    key = pd.read_csv(KEY_MAPPING, encoding="utf-8-sig")

    df2 = df2.copy()
    df2["文件名"] = names2
    df2["遮挡等级_标注员2"] = lv
    rename = {"耗时秒": "耗时秒_标注员2", "标注时间": "标注时间_标注员2"}
    df2 = df2.rename(columns=rename)

    merged = key.merge(
        df2[["文件名", "遮挡等级_标注员2", "耗时秒_标注员2", "标注时间_标注员2"]],
        left_on="new_name", right_on="文件名", how="left",
    ).drop(columns=["文件名"])

    a = merged["遮挡等级_标注员1"]
    b = merged["遮挡等级_标注员2"]
    merged["是否一致"] = pd.Series(
        ["" if pd.isna(x) or pd.isna(y) else ("一致" if int(x) == int(y) else "不一致")
         for x, y in zip(a, b)]
    )
    say(f"    合并后 {len(merged)} 行")
    say()

    # --- 预览 -------------------------------------------------------------
    say("[4] 初步预览")
    n_both = int((a.notna() & b.notna()).sum())
    say(f"    标注员 1 已标注: {int(a.notna().sum())} 张")
    say(f"    标注员 2 已标注: {int(b.notna().sum())} 张")
    say(f"    两人都标注了  : {n_both} 张")

    if n_both:
        na = int(merged.loc[a.notna() & b.notna(), "是否一致"].eq("一致").sum())
        say(f"    完全一致      : {na} / {n_both} = {na / n_both * 100:.1f}%")

        aa = a[a.notna() & b.notna()].astype(int).tolist()
        bb = b[a.notna() & b.notna()].astype(int).tolist()
        k = cohen_kappa(aa, bb, sorted(VALID_LEVELS))
        say(f"    未加权 kappa  : {k:.3f}   （仅预览，完整分析另行产出）")
        say()
        say("    混淆矩阵（行=标注员1，列=标注员2）：")
        ct = pd.crosstab(pd.Series(aa, name="标注员1"), pd.Series(bb, name="标注员2"))
        for line in ct.to_string().splitlines():
            say("      " + line)
    say()
    say("    标注员 2 的等级分布:")
    dist = b.value_counts().sort_index()
    for lv_, n_ in dist.items():
        say(f"      {int(lv_)}  {LEVEL_NAMES.get(int(lv_), '?'):<22} {int(n_)}")

    # --- 落盘 -------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    xlsx = out_dir / "occlusion_annotations_merged.xlsx"
    merged.to_excel(xlsx, index=False)
    say()
    say(f"[5] 已写出: {rel(xlsx)}")

    rep = out_dir / "ingest_report.txt"
    rep.write_text("\n".join(_lines), encoding="utf-8-sig")
    say(f"           {rel(rep)}")

    say()
    say("=" * 72)
    say("合并表已生成。接下来做正式的评分者间一致性分析：")
    say("  加权 kappa / PABAK / Gwet's AC1 / 按等级的混淆矩阵 / 敏感性分析")
    say("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
