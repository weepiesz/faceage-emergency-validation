"""Build the de-identified package used for the second assessor's occlusion grading.

Three steps:
  1. take the 1,485 images of the initial dataset out of merged_data.xlsx
  2. rename them IMG_0001.jpg ... IMG_1485.jpg, shuffled with a fixed seed so that grade does not
     cluster along the ordering
  3. copy them into annotation_package/images/ and write the mapping table locally

The mapping table key_mapping.csv links the anonymous file names back to patient identifiers and
must never be placed in the folder handed to the assessor. It stays under scripts/annotation/.

Usage:
    python scripts/annotation/build_annotation_package.py
    python scripts/annotation/build_annotation_package.py --force    rebuild an existing package"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# 路径配置（相对项目根目录）
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MERGED_XLSX = PROJECT_ROOT / "Clinical_dataset" / "merged_data.xlsx"
IMAGE_DIR = PROJECT_ROOT / "Clinical_dataset" / "face_data_complete"
PACKAGE_DIR = PROJECT_ROOT / "faceage_emergency_study" / "annotation_package"
LOCAL_KEY = PROJECT_ROOT / "faceage_emergency_study" / "scripts" / "annotation" / "key_mapping.csv"
HTML_TEMPLATE = PROJECT_ROOT / "faceage_emergency_study" / "scripts" / "annotation" / "annotate_template.html"
HTML_OUT = PACKAGE_DIR / "annotate_occlusion.html"

SHUFFLE_SEED = 20261013  # 固定，保证可复现

# 分析集使用的列
COL_FILENAME = "image_filename"
COL_GROUP = "group_name"
COL_OCCL = "occlusion_level"
COL_AGE = "年龄"
COL_DEATH = "7天内死亡"
COL_APACHE = "ApacheⅡ评分"


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_html() -> int:
    """扫描 images/ 生成网页版标注工具。

    图像清单必须内联进 HTML：交付后对方用 file:// 直接打开，
    此时 fetch()/XHR 会被 CORS 拦住，无法枚举目录，只能由构建期写死。
    """
    images_out = PACKAGE_DIR / "images"
    if not images_out.exists():
        print(f"[错误] 图像目录不存在: {images_out}")
        return 1

    names = sorted(
        p.name for p in images_out.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not names:
        print(f"[错误] {images_out} 下没有图像")
        return 1

    if not HTML_TEMPLATE.exists():
        print(f"[错误] 找不到模板: {HTML_TEMPLATE}")
        return 1
    tpl = HTML_TEMPLATE.read_text(encoding="utf-8")
    if "__IMAGE_LIST__" not in tpl:
        print("[错误] 模板缺少 __IMAGE_LIST__ 占位符")
        return 1

    html = tpl.replace("__IMAGE_LIST__", json.dumps(names, ensure_ascii=False))
    HTML_OUT.write_text(html, encoding="utf-8")
    print(f"[工具] 已生成 {HTML_OUT.name} —— 内联 {len(names)} 个文件名，"
          f"{HTML_OUT.stat().st_size / 1024:.0f} KB")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="构建去标识标注包")
    ap.add_argument("--force", action="store_true", help="目标目录已存在时清空重建")
    ap.add_argument("--tool-only", action="store_true",
                    help="只重新生成网页工具，不重做图像复制（秒级）")
    args = ap.parse_args()

    print("=" * 72)
    print("构建去标识标注包")
    print("=" * 72)

    # 仅重建工具：图像已经是去标识好的，重做一遍纯属浪费时间
    if args.tool_only:
        print("[模式] 仅重建网页工具（不动图像与映射表）")
        rc = build_html()
        print()
        print("完成" if rc == 0 else "失败")
        return rc

    # --- 前置检查 -----------------------------------------------------------
    for p, desc in [(MERGED_XLSX, "merged_data.xlsx"), (IMAGE_DIR, "图像目录")]:
        if not p.exists():
            print(f"[错误] 找不到{desc}: {p}")
            return 1

    # --- 读取分析集 ---------------------------------------------------------
    df = pd.read_excel(MERGED_XLSX)
    print(f"[1/5] 读取 merged_data.xlsx —— {len(df)} 行，{len(df.columns)} 列")

    missing_cols = [c for c in [COL_FILENAME, COL_GROUP, COL_OCCL] if c not in df.columns]
    if missing_cols:
        print(f"[错误] merged_data.xlsx 缺少列: {missing_cols}")
        return 1

    filenames = df[COL_FILENAME].astype(str).tolist()
    if len(set(filenames)) != len(filenames):
        print("[错误] image_filename 存在重复，需先排查数据")
        return 1
    print(f"      image_filename 唯一值 {len(filenames)} 个")

    # --- 确认源图存在 -------------------------------------------------------
    src_paths = []
    missing = []
    for i, fn in enumerate(filenames):
        p = IMAGE_DIR / fn
        if p.exists():
            src_paths.append(p)
        else:
            missing.append(fn)
        if (i + 1) % 500 == 0:
            print(f"      [检查源图] {i + 1}/{len(filenames)} ...")
    print(f"[2/5] 检查源图 —— 找到 {len(src_paths)} 张，缺失 {len(missing)} 张")
    if missing:
        print(f"      [错误] 以下图像缺失，中止：{missing[:10]}")
        return 1

    # --- 打乱顺序 -----------------------------------------------------------
    # 固定 seed 打乱，避免同一遮挡等级/同一批次的图连续出现，
    # 防止标注员从中推断规律、形成先验。
    order = list(range(len(filenames)))
    import random

    random.Random(SHUFFLE_SEED).shuffle(order)
    print(f"[3/5] 打乱顺序（seed={SHUFFLE_SEED}）")

    # --- 准备目标目录 -------------------------------------------------------
    # 只清理 images/ 与 annotations/，不动包内的工具与文档
    images_out = PACKAGE_DIR / "images"
    annotations_out = PACKAGE_DIR / "annotations"
    if images_out.exists() and any(images_out.iterdir()):
        if not args.force:
            print(f"[错误] {images_out} 已存在且非空")
            print("       如需重建请加 --force")
            return 1
        print(f"      清空已存在的图像目录: {images_out}")
        shutil.rmtree(images_out)
    if annotations_out.exists() and any(annotations_out.iterdir()):
        if args.force:
            print(f"      清空已存在的标注目录: {annotations_out}")
            shutil.rmtree(annotations_out)
    images_out.mkdir(parents=True, exist_ok=True)
    annotations_out.mkdir(parents=True, exist_ok=True)

    # --- 复制并重命名 -------------------------------------------------------
    print(f"[4/5] 复制并重命名 {len(order)} 张图像 ...")
    records = []
    for n, src_idx in enumerate(order, start=1):
        src = src_paths[src_idx]
        new_name = f"IMG_{n:04d}.jpg"
        dst = images_out / new_name
        shutil.copy2(src, dst)

        row = df.iloc[src_idx]
        records.append(
            {
                "new_name": new_name,
                "原文件名": src.name,
                "group_name": row.get(COL_GROUP),
                "遮挡等级_标注员1": row.get(COL_OCCL),
                "年龄": row.get(COL_AGE),
                "7天内死亡": row.get(COL_DEATH),
                "ApacheⅡ评分": row.get(COL_APACHE),
            }
        )
        if n % 200 == 0 or n == len(order):
            print(f"      [复制] {n}/{len(order)} ({n / len(order) * 100:.1f}%)")

    # --- 写映射表 -----------------------------------------------------------
    key_df = pd.DataFrame(records)
    key_df.to_csv(LOCAL_KEY, index=False, encoding="utf-8-sig")
    print(f"[5/5] 映射表已写入（本地保存，切勿外发）: {LOCAL_KEY.relative_to(PROJECT_ROOT)}")

    # --- 生成网页标注工具 ---------------------------------------------------
    if build_html() != 0:
        return 1

    # --- 自校验 -------------------------------------------------------------
    print()
    print("=" * 72)
    print("自校验")
    print("=" * 72)

    ok = True

    n_out = len(list(images_out.glob("*.jpg")))
    print(f"  交付包图像数: {n_out} (期望 {len(order)})", "✓" if n_out == len(order) else "✗")
    ok &= n_out == len(order)

    names = [r["new_name"] for r in records]
    uniq = len(set(names)) == len(names)
    print(f"  新文件名唯一: {uniq}", "✓" if uniq else "✗")
    ok &= uniq

    # 随机抽 20 张做 MD5 比对，确认改名后内容与原图一致
    import random as _r

    sample = _r.Random(7).sample(records, 20)
    bad = [
        r
        for r in sample
        if md5_of(images_out / r["new_name"]) != md5_of(IMAGE_DIR / r["原文件名"])
    ]
    print(f"  抽 20 张 MD5 比对: 不一致 {len(bad)} 张", "✓" if not bad else "✗")
    ok &= not bad

    # 遮挡等级分布须与 merged_data 一致
    dist = key_df["遮挡等级_标注员1"].value_counts().sort_index().to_dict()
    print(f"  遮挡等级分布: {dist}")

    print()
    print("=" * 72)
    print("完成" if ok else "完成，但自校验有失败项，请检查")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
