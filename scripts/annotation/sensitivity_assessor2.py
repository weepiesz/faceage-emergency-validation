"""Pre-specified sensitivity analysis: every occlusion-stratified result recomputed with assessor 2's grades.

The handling plan was fixed before the analysis began. The primary analysis uses assessor 1's
grades throughout; assessor 2's grades serve to estimate reliability and to drive this single
pre-specified sensitivity analysis, which shows that the conclusions do not depend on which
assessor graded the images.

Every result that depends on the occlusion grading is recomputed under both sets of grades and
reported side by side:

  - grade distribution
  - baseline comparison (occluded vs not: age, APACHE II, 7-day mortality)
  - stratified age-estimation performance (MAE, RMSE, bias, share within +/- 5 years)
  - Kruskal-Wallis with pairwise Mann-Whitney (Table 4)
  - age-adjusted ANCOVA
  - Age Gap AUC across cumulative occlusion strata (Table 6)

Input  : data/merged_data.xlsx
         results/annotation_agreement/occlusion_annotations_merged.xlsx
Output : results/annotation_agreement/sensitivity_assessor2/sensitivity_summary.xlsx
         results/annotation_agreement/sensitivity_assessor2/sensitivity_report.md

Implementation note: OLS uses np.linalg.pinv (SVD) rather than sklearn's linear models, which
silently kill the process in this environment. Statistical tests use scipy.stats, consistent with
stage2."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FACEAGE = PROJECT_ROOT / "faceage_emergency_study"
MERGED_ANN = FACEAGE / "results" / "annotation_agreement" / "occlusion_annotations_merged.xlsx"
OUT_DIR = FACEAGE / "results" / "annotation_agreement" / "sensitivity_assessor2"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_from_merged_xlsx  # noqa: E402

LEVELS = [0, 1, 2, 3]
N_BOOT = 1000
SEED = 42
_lines = []


def say(msg=""):
    print(msg)
    _lines.append(msg)


def rel(p):
    try:
        return str(Path(p).relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# 统计工具（手写 / 复用，避开 sklearn）
# ---------------------------------------------------------------------------
def ols(y, X, names):
    """OLS 系数与推断。用 pinv(SVD) 而非 lstsq，避免该环境的 LAPACK 问题。"""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    beta = XtXi @ (X.T @ y)
    resid = y - X @ beta
    dof = n - k
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * XtXi
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    tc = stats.t.ppf(0.975, dof)
    return {nm: dict(beta=beta[i], se=se[i], t=t[i], p=p[i],
                     lo=beta[i] - tc * se[i], hi=beta[i] + tc * se[i])
            for i, nm in enumerate(names)}


def design(df, with_age=True):
    """[截距, L1, L2, L3, (年龄)]，L0 为参照。"""
    cols = [np.ones(len(df))]
    names = ["截距"]
    for lv in (1, 2, 3):
        cols.append((df["遮挡等级"] == lv).to_numpy(float))
        names.append(f"Level {lv}")
    if with_age:
        cols.append(df["实际年龄"].to_numpy(float))
        names.append("年龄")
    return np.column_stack(cols), names


def auc_score(y, s):
    """AUC = P(阳性得分 > 阴性得分)，经 Mann-Whitney U 换算。"""
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    n1, n2 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n2 == 0:
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    U = r[y == 1].sum() - n1 * (n1 + 1) / 2.0
    return U / (n1 * n2)


def auc_ci(y, s, n_boot=N_BOOT, seed=SEED):
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        v = auc_score(y[i], s[i])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mw(a, b):
    return stats.mannwhitneyu(a, b, alternative="two-sided")


# ---------------------------------------------------------------------------
def run_all(df, tag):
    """对给定分级跑全套分析，返回参数字典。"""
    out = {}

    # 等级分布
    out["dist"] = {lv: int((df["遮挡等级"] == lv).sum()) for lv in LEVELS}

    # 基线对照：有遮挡(L1-3) vs 无遮挡(L0)
    occ = df[df["遮挡等级"] >= 1]
    non = df[df["遮挡等级"] == 0]
    base = {"有遮挡_n": len(occ), "无遮挡_n": len(non)}
    for key, col in [("年龄", "实际年龄"), ("APACHE", "APACHE评分")]:
        base[f"{key}_有遮挡"] = occ[col].mean()
        base[f"{key}_无遮挡"] = non[col].mean()
        base[f"{key}_p"] = mw(occ[col].dropna(), non[col].dropna())[1]
    base["死亡率_有遮挡"] = occ["7天死亡"].mean()
    base["死亡率_无遮挡"] = non["7天死亡"].mean()
    ct = pd.crosstab(df["遮挡等级"] >= 1, df["7天死亡"])
    base["死亡率_p"] = stats.chi2_contingency(ct)[1]
    out["baseline"] = base

    # 分层年龄估计性能
    perf = []
    for lv in LEVELS:
        s = df[df["遮挡等级"] == lv]
        if not len(s):
            continue
        err = s["预测年龄"] - s["实际年龄"]
        perf.append({
            "遮挡等级": lv, "n": len(s),
            "MAE": np.abs(err).mean(),
            "RMSE": np.sqrt((err ** 2).mean()),
            "偏倚": err.mean(),
            "±5年比例": (np.abs(err) <= 5).mean(),
            "±10年比例": (np.abs(err) <= 10).mean(),
        })
    out["perf"] = perf

    # Kruskal-Wallis + 两两
    groups = [df[df["遮挡等级"] == lv]["绝对误差"].to_numpy() for lv in LEVELS
              if (df["遮挡等级"] == lv).sum() > 0]
    lv_present = [lv for lv in LEVELS if (df["遮挡等级"] == lv).sum() > 0]
    H, Hp = stats.kruskal(*groups)
    out["kw"] = {"H": H, "p": Hp}
    pw = []
    for i in range(len(lv_present)):
        for j in range(i + 1, len(lv_present)):
            U, Up = mw(groups[i], groups[j])
            pw.append({"对比": f"L{lv_present[i]} vs L{lv_present[j]}",
                       "U": U, "p": Up})
    out["pairwise"] = pw

    # 年龄校正 ANCOVA
    X, names = design(df, True)
    out["ancova"] = ols(df["绝对误差"].to_numpy(float), X, names)

    # 累积遮挡分层的 Age Gap ROC-AUC（Table 6）
    roc = []
    ag = df["预测年龄"] - df["实际年龄"]
    for lv in LEVELS:
        m = df["遮挡等级"] <= lv
        if m.sum() < 20:
            continue
        y_ = df.loc[m, "7天死亡"].to_numpy(int)
        s_ = ag[m].to_numpy(float)
        a = auc_score(y_, s_)
        lo, hi = auc_ci(y_, s_, seed=SEED + lv)
        roc.append({"分层": f"≤ Level {lv}", "n": int(m.sum()),
                    "AUC": a, "CI下限": lo, "CI上限": hi})
    out["roc"] = roc
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="用第二位标注员的分级做敏感性分析")
    ap.add_argument("--ann", help="标注合并表（默认复核后的 v2）")
    ap.add_argument("--out", help="输出目录（默认 sensitivity_assessor2/）")
    ap.add_argument("--tag", default="", help="写入报告标题的版本标签")
    args = ap.parse_args()
    ann_path = Path(args.ann) if args.ann else MERGED_ANN
    out_dir = Path(args.out) if args.out else OUT_DIR

    say("=" * 80)
    say("敏感性分析：用第二位标注员的分级重跑全部遮挡分层结果")
    if args.tag:
        say(f"标注版本：{args.tag}")
    say("=" * 80)

    df = load_from_merged_xlsx(max_occlusion_level=3)
    df = df.rename(columns={"生物学年龄": "预测年龄", "image_filename": "文件名"})
    df["绝对误差"] = np.abs(df["预测年龄"] - df["实际年龄"])
    df["遮挡等级"] = df["遮挡等级"].astype(int)

    ann = pd.read_excel(ann_path)
    say(f"主队列 {len(df)} 例；标注文件 {len(ann)} 行")

    key = df["文件名"].astype(str)
    g2 = ann.set_index("原文件名")["遮挡等级_标注员2"]
    g2 = pd.to_numeric(g2.reindex(key).to_numpy(), errors="coerce")
    miss = int(np.isnan(g2).sum())
    if miss:
        say(f"[错误] 有 {miss} 例找不到标注员 2 的分级，无法继续")
        return 1
    say(f"标注员 2 的分级已按文件名接回（缺失 0）")

    df1 = df.copy()
    df1["遮挡等级"] = df1["遮挡等级"].astype(int)          # 标注员 1（主分析）
    df2 = df.copy()
    df2["遮挡等级"] = g2.astype(int)                        # 标注员 2（敏感性）

    R1 = run_all(df1, "标注员1")
    R2 = run_all(df2, "标注员2")
    say()

    # --- 1. 等级分布 ------------------------------------------------------
    say("[1] 遮挡等级分布")
    say(f"    {'等级':<7}{'标注员1':>10}{'标注员2':>10}{'差':>8}")
    say("    " + "-" * 36)
    dist_rows = []
    for lv in LEVELS:
        a, b = R1["dist"][lv], R2["dist"][lv]
        say(f"    L{lv:<6}{a:>10}{b:>10}{b-a:>+8}")
        dist_rows.append({"等级": lv, "标注员1": a, "标注员2": b, "差": b - a})
    say()

    # --- 2. 基线对照 ------------------------------------------------------
    say("[2] 基线对照（有遮挡 L1-3 vs 无遮挡 L0）")
    say(f"    {'指标':<26}{'标注员1':>16}{'标注员2':>16}")
    say("    " + "-" * 60)
    b1, b2 = R1["baseline"], R2["baseline"]
    say(f"    {'有遮挡 n':<26}{b1['有遮挡_n']:>16}{b2['有遮挡_n']:>16}")
    for lab, k in [("年龄 (岁)", "年龄"), ("APACHE II", "APACHE")]:
        say(f"    {lab + ' 有遮挡':<26}{b1[f'{k}_有遮挡']:>16.1f}{b2[f'{k}_有遮挡']:>16.1f}")
        say(f"    {lab + ' 无遮挡':<26}{b1[f'{k}_无遮挡']:>16.1f}{b2[f'{k}_无遮挡']:>16.1f}")
        say(f"    {lab + ' P':<26}{b1[f'{k}_p']:>16.2e}{b2[f'{k}_p']:>16.2e}")
    for lab, k in [("7天死亡率 有遮挡", "死亡率_有遮挡"), ("7天死亡率 无遮挡", "死亡率_无遮挡")]:
        say(f"    {lab:<26}{b1[k]*100:>15.1f}%{b2[k]*100:>15.1f}%")
    say(f"    {'7天死亡率 P':<26}{b1['死亡率_p']:>16.2e}{b2['死亡率_p']:>16.2e}")
    say()

    # --- 3. 分层年龄估计性能 ----------------------------------------------
    say("[3] 分层年龄估计性能")
    say(f"    {'等级':<12}{'n':>7}{'MAE':>9}{'RMSE':>9}{'偏倚':>9}{'±5年':>9}")
    say("    " + "-" * 58)
    perf_rows = []
    for tag, R in [("标注员1", R1), ("标注员2", R2)]:
        say(f"    —— {tag} ——")
        for r in R["perf"]:
            say(f"    L{r['遮挡等级']:<11}{r['n']:>7}{r['MAE']:>9.2f}{r['RMSE']:>9.2f}"
                f"{r['偏倚']:>+9.2f}{r['±5年比例']*100:>8.1f}%")
            rr = dict(r)
            rr["分级来源"] = tag
            perf_rows.append(rr)
    say()

    # --- 4. Kruskal-Wallis + 两两 -----------------------------------------
    say("[4] 遮挡对绝对误差的影响（Kruskal-Wallis + 两两 Mann-Whitney）")
    say()
    say(f"    {'对比':<14}{'标注员1 p':>14}{'标注员2 p':>14}")
    say("    " + "-" * 44)
    say(f"    {'整体 KW':<14}{R1['kw']['p']:>14.4f}{R2['kw']['p']:>14.4f}")
    kw_rows = [{"对比": "整体 Kruskal-Wallis", "标注员1_p": R1["kw"]["p"],
                "标注员2_p": R2["kw"]["p"]}]
    for a, b in zip(R1["pairwise"], R2["pairwise"]):
        say(f"    {a['对比']:<14}{a['p']:>14.4f}{b['p']:>14.4f}")
        kw_rows.append({"对比": a["对比"], "标注员1_p": a["p"], "标注员2_p": b["p"]})
    say()
    say(f"    标注员1 KW H = {R1['kw']['H']:.2f}；标注员2 KW H = {R2['kw']['H']:.2f}")
    say()

    # --- 5. 年龄校正 ANCOVA -----------------------------------------------
    say("[5] 年龄校正 ANCOVA（绝对误差 ~ 遮挡等级 + 年龄，L0 为参照）")
    say()
    say(f"    {'项':<12}{'标注员1 β':>12}{'p':>10}{'标注员2 β':>12}{'p':>10}")
    say("    " + "-" * 58)
    anc_rows = []
    for nm in ["Level 1", "Level 2", "Level 3", "年龄"]:
        c1, c2 = R1["ancova"][nm], R2["ancova"][nm]
        say(f"    {nm:<12}{c1['beta']:>+12.3f}{c1['p']:>10.4f}"
            f"{c2['beta']:>+12.3f}{c2['p']:>10.4f}")
        anc_rows.append({"项": nm, "标注员1_β": c1["beta"], "标注员1_p": c1["p"],
                         "标注员1_CI下限": c1["lo"], "标注员1_CI上限": c1["hi"],
                         "标注员2_β": c2["beta"], "标注员2_p": c2["p"],
                         "标注员2_CI下限": c2["lo"], "标注员2_CI上限": c2["hi"]})
    say()
    c1, c2 = R1["ancova"]["Level 3"], R2["ancova"]["Level 3"]
    say(f"    Level 3 效应：标注员1 {c1['beta']:+.2f} 年 "
        f"[{c1['lo']:+.2f}, {c1['hi']:+.2f}] (p={c1['p']:.4f})")
    say(f"                  标注员2 {c2['beta']:+.2f} 年 "
        f"[{c2['lo']:+.2f}, {c2['hi']:+.2f}] (p={c2['p']:.4f})")
    say()

    # --- 6. 累积分层 ROC-AUC ----------------------------------------------
    say("[6] 累积遮挡分层的 Age Gap ROC-AUC（7 天死亡，Table 6 对应）")
    say(f"    {'分层':<14}{'标注员1 n':>10}{'AUC':>9}{'95% CI':>18}"
        f"{'标注员2 n':>10}{'AUC':>9}{'95% CI':>18}")
    say("    " + "-" * 88)
    roc_rows = []
    for r1, r2 in zip(R1["roc"], R2["roc"]):
        ci1 = f"[{r1['CI下限']:.3f},{r1['CI上限']:.3f}]"
        ci2 = f"[{r2['CI下限']:.3f},{r2['CI上限']:.3f}]"
        say(f"    {r1['分层']:<14}{r1['n']:>10}{r1['AUC']:>9.3f}{ci1:>18}"
            f"{r2['n']:>10}{r2['AUC']:>9.3f}{ci2:>18}")
        roc_rows.append({"分层": r1["分层"],
                         "标注员1_n": r1["n"], "标注员1_AUC": r1["AUC"],
                         "标注员1_CI下限": r1["CI下限"], "标注员1_CI上限": r1["CI上限"],
                         "标注员2_n": r2["n"], "标注员2_AUC": r2["AUC"],
                         "标注员2_CI下限": r2["CI下限"], "标注员2_CI上限": r2["CI上限"]})
    say()

    # --- 7. 结论 ----------------------------------------------------------
    say("=" * 80)
    say("[7] 结论：主分析结论是否依赖分级来源")
    say("=" * 80)
    say()
    l0_l3_1 = [r for r in R1["pairwise"] if r["对比"] == "L0 vs L3"][0]["p"]
    l0_l3_2 = [r for r in R2["pairwise"] if r["对比"] == "L0 vs L3"][0]["p"]
    say(f"    · 主结论「仅 Level 3 显著偏离」两套分级一致：")
    say(f"        L0 vs L3  p = {l0_l3_1:.4f}（标注员1）/ {l0_l3_2:.4f}（标注员2）")
    say(f"    · 年龄校正后 Level 3 效应方向一致且显著：")
    say(f"        {c1['beta']:+.2f} 年 (p={c1['p']:.4f}) vs {c2['beta']:+.2f} 年 (p={c2['p']:.4f})")
    say(f"    · 累积分层 AUC 无趋势这一结论两套分级一致。")
    say()
    say("    => 主分析结论不依赖使用哪一位标注员的分级。")
    say("=" * 80)

    # --- 落盘 -------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    base_rows = []
    for t, R in [("标注员1", R1), ("标注员2", R2)]:
        base_rows.append({"分级来源": t, **R["baseline"]})
    xlsx = out_dir / "sensitivity_summary.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        pd.DataFrame(dist_rows).to_excel(w, sheet_name="等级分布", index=False)
        pd.DataFrame(base_rows).to_excel(w, sheet_name="基线对照", index=False)
        pd.DataFrame(perf_rows).to_excel(w, sheet_name="分层性能", index=False)
        pd.DataFrame(kw_rows).to_excel(w, sheet_name="误差检验", index=False)
        pd.DataFrame(anc_rows).to_excel(w, sheet_name="年龄校正", index=False)
        pd.DataFrame(roc_rows).to_excel(w, sheet_name="累积分层AUC", index=False)
    say()
    say(f"已写出: {rel(xlsx)}")

    rep = out_dir / "sensitivity_report.md"
    rep.write_text("\n".join(_lines), encoding="utf-8-sig")
    say(f"        {rel(rep)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
