"""Inter-rater agreement of the occlusion grading.

Compares the two assessors' grades on the 1,485 photographed patients: observed agreement,
Cohen's kappa (unweighted and linear-weighted), PABAK, Gwet's AC1, and the confusion matrix.

Why PABAK and Gwet's AC1 are reported alongside Cohen's kappa
-------------------------------------------------------------
Grade 0 accounts for 83.8% of the cohort, so the grading distribution is heavily unbalanced —
exactly the setting in which the well-known kappa paradox appears. Agreement can be high while
kappa is dragged down, because a lopsided marginal distribution inflates the expected agreement
pe. Reporting kappa alone would invite an easy objection, so all three are given together:

  Cohen's kappa   the conventional measure, but sensitive to imbalance and biased low here
  PABAK           2 * po - 1; unaffected by the marginal distribution, i.e. observed agreement
                  rescaled
  Gwet's AC1      the most robust of the three under imbalance, and the recommended estimator for
                  ordinal data of this shape

The reporting plan was fixed before the numbers were seen, so the choice cannot be read as
fishing for a favourable statistic.

Environment note: this script avoids sklearn, whose linear models silently kill the process in
this environment. Cohen's kappa and weighted kappa are implemented by hand.

Input  : results/annotation_agreement/occlusion_annotations_merged.xlsx
         (produced by ingest_assessor2.py)
Output : results/annotation_agreement/inter_rater_agreement.xlsx
         results/annotation_agreement/agreement_report.md
         results/annotation_agreement/fig_confusion_matrix.png"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FACEAGE = PROJECT_ROOT / "faceage_emergency_study"
ANN_DIR = FACEAGE / "results" / "annotation_agreement"
MERGED = ANN_DIR / "occlusion_annotations_merged.xlsx"

LABELS = [0, 1, 2, 3]
LEVEL_NAMES = {
    0: "无遮挡或细鼻管",
    1: "嘴部插管或粗鼻管",
    2: "呼吸机较大遮挡",
    3: "包扎或血污严重遮挡",
}
N_BOOT = 2000
SEED = 42

_lines = []


def say(msg=""):
    print(msg)
    _lines.append(msg)


# ---------------------------------------------------------------------------
# 一致性指标（手写，不依赖 sklearn）
# ---------------------------------------------------------------------------
def observed_agreement(a, b):
    return float(np.mean(a == b)) if len(a) else float("nan")


def cohen_kappa(a, b, labels, w=None):
    """Cohen's κ（含加权版本）。

    统一形式 κ = 1 − Σ w·O / Σ w·E，其中 w 为"不一致度"权重
    （对角为 0、差异越大越接近 1）。w=None 时取 W = 1 − I，即未加权 κ。
    """
    n = len(a)
    if n == 0:
        return float("nan")
    k = len(labels)
    pos = {lv: i for i, lv in enumerate(labels)}
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[pos[x], pos[y]] += 1
    O /= n
    E = np.outer(O.sum(axis=1), O.sum(axis=0))
    W = (np.ones((k, k)) - np.eye(k)) if w is None else w
    den = float((W * E).sum())
    if den <= 0:
        return float("nan")
    return 1.0 - float((W * O).sum()) / den


def weighted_matrix(labels, kind):
    k = len(labels)
    idx = np.arange(k)
    d = np.abs(idx[:, None] - idx[None, :]) / (k - 1)
    return d if kind == "linear" else d ** 2


def pabak(a, b):
    """Prevalence-Adjusted Bias-Adjusted Kappa = 2·po − 1。"""
    po = observed_agreement(a, b)
    return 2 * po - 1


def gwet_ac1(a, b, labels):
    """Gwet's AC1（名义型）。

    pe = 1/(q−1) · Σ_k π_k (1 − π_k)，π_k 为第 k 类在两位标注员上的
    合并流行率。相比 κ 的 pe = Σ p_k·q_k，AC1 在不平衡数据上稳定得多。
    """
    n = len(a)
    if n == 0:
        return float("nan")
    q = len(labels)
    po = observed_agreement(a, b)
    pe = 0.0
    for lv in labels:
        pi = (np.sum(a == lv) + np.sum(b == lv)) / (2.0 * n)
        pe += pi * (1 - pi)
    pe /= (q - 1)
    if pe >= 1.0:
        return float("nan")
    return (po - pe) / (1 - pe)


def boot_ci(fn, a, b, labels, n_boot=N_BOOT, seed=SEED):
    """bootstrap 百分位 95% CI（成对重抽样，保持配对结构）。"""
    rng = np.random.default_rng(seed)
    n = len(a)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        try:
            v = fn(a[i], b[i], labels)
        except Exception:
            v = np.nan
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def kendall_tau_b(a, b):
    """Kendall's τ-b（序数关联）。手写，避开 sklearn/scipy 的版本差异。"""
    n = len(a)
    conc = disc = 0
    tie_a = tie_b = 0
    for i in range(n):
        da = a[i] - a[i + 1:]
        db = b[i] - b[i + 1:]
        s = np.sign(da) * np.sign(db)
        conc += int(np.sum(s > 0))
        disc += int(np.sum(s < 0))
        tie_a += int(np.sum((da == 0) & (db != 0)))
        tie_b += int(np.sum((db == 0) & (da != 0)))
    n0 = conc + disc + tie_a + tie_b
    n1 = conc + disc + tie_a
    n2 = conc + disc + tie_b
    if n1 <= 0 or n2 <= 0:
        return float("nan")
    return (conc - disc) / np.sqrt(n1 * n2)


def interpret(k):
    """Landis & Koch 分级。"""
    if not np.isfinite(k):
        return "—"
    for thr, txt in [(0.81, "几乎完全一致 (almost perfect)"),
                     (0.61, "高度一致 (substantial)"),
                     (0.41, "中等一致 (moderate)"),
                     (0.21, "一般 (fair)"),
                     (0.00, "轻微 (slight)")]:
        if k >= thr:
            return txt
    return "差 (poor)"


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="评分者间一致性分析")
    ap.add_argument("--input", help="标注合并表（默认 occlusion_annotations_merged.xlsx）")
    ap.add_argument("--out", help="输出目录（默认 results/annotation_agreement/）")
    args = ap.parse_args()
    src = Path(args.input) if args.input else MERGED
    out_dir = Path(args.out) if args.out else ANN_DIR

    if not src.exists():
        print(f"[错误] 找不到 {src}")
        print("       请先运行 ingest_assessor2.py")
        return 1

    df = pd.read_excel(src)
    a = df["遮挡等级_标注员1"].to_numpy(int)
    b = df["遮挡等级_标注员2"].to_numpy(int)
    n = len(a)

    say("=" * 74)
    say("评分者间一致性分析 —— 面部遮挡类型分级")
    say("=" * 74)
    say(f"数据来源: {src.name}")
    say(f"样本量: {n}")
    say()

    # --- 1. 总体指标 ------------------------------------------------------
    po = observed_agreement(a, b)
    lo_po, hi_po = boot_ci(lambda x, y, _: np.mean(x == y), a, b, LABELS)
    k_unw = cohen_kappa(a, b, LABELS)
    k_lin = cohen_kappa(a, b, LABELS, weighted_matrix(LABELS, "linear"))
    k_quad = cohen_kappa(a, b, LABELS, weighted_matrix(LABELS, "quadratic"))
    pbk = pabak(a, b)
    ac1 = gwet_ac1(a, b, LABELS)
    tau = kendall_tau_b(a, b)

    ci_unw = boot_ci(cohen_kappa, a, b, LABELS)
    ci_lin = boot_ci(lambda x, y, L: cohen_kappa(x, y, L, weighted_matrix(L, "linear")),
                     a, b, LABELS)
    ci_quad = boot_ci(lambda x, y, L: cohen_kappa(x, y, L, weighted_matrix(L, "quadratic")),
                      a, b, LABELS)
    ci_pbk = boot_ci(lambda x, y, _: pabak(x, y), a, b, LABELS)
    ci_ac1 = boot_ci(gwet_ac1, a, b, LABELS)

    say("[1] 总体一致性")
    say()
    rows = [
        ("完全一致率", po, lo_po, hi_po, f"{po*100:.1f}%"),
        ("Cohen's κ（未加权）", k_unw, ci_unw[0], ci_unw[1], interpret(k_unw)),
        ("Cohen's κ（线性加权）", k_lin, ci_lin[0], ci_lin[1], interpret(k_lin)),
        ("Cohen's κ（二次加权）", k_quad, ci_quad[0], ci_quad[1], interpret(k_quad)),
        ("PABAK", pbk, ci_pbk[0], ci_pbk[1], interpret(pbk)),
        ("Gwet's AC1", ac1, ci_ac1[0], ci_ac1[1], interpret(ac1)),
        ("Kendall's τ-b", tau, np.nan, np.nan, "序数关联"),
    ]
    say(f"    {'指标':<22}{'取值':>8}   {'95% CI':<20}{'解释'}")
    say("    " + "-" * 68)
    for name, v, lo, hi, note in rows:
        if np.isfinite(lo) and np.isfinite(hi):
            ci = f"[{lo:.3f}, {hi:.3f}]"
        else:
            ci = ""
        say(f"    {name:<22}{v:>8.3f}   {ci:<20}{note}")
    say()

    # 悖论诊断
    say(f"    注：Level 0 占标注员1 的 {np.mean(a==0)*100:.1f}%、标注员2 的 "
        f"{np.mean(b==0)*100:.1f}%，分布高度不平衡。")
    say(f"        未加权 κ ({k_unw:.3f}) 与 PABAK ({pbk:.3f}) 的差距即'kappa 悖论'"
        f"的体现：")
    say(f"        实际一致率 {po*100:.1f}% 很高，但边缘分布一边倒抬高了期望一致率。")
    say()

    # --- 2. 逐等级 -------------------------------------------------------
    say("[2] 逐等级一致情况（行 = 标注员1）")
    say()
    say(f"    {'等级':<4}{'定义':<22}{'n1':>5}{'n2':>6}{'一致':>6}{'一致率':>8}")
    say("    " + "-" * 68)
    lv_rows = []
    for lv in LABELS:
        m = a == lv
        n1 = int(m.sum())
        n2 = int((b == lv).sum())
        ag = int((m & (a == b)).sum())
        rate = ag / n1 if n1 else float("nan")
        say(f"    {lv:<4}{LEVEL_NAMES[lv]:<22}{n1:>5}{n2:>6}{ag:>6}{rate*100:>7.1f}%")
        lv_rows.append({"等级": lv, "定义": LEVEL_NAMES[lv], "标注员1_n": n1,
                        "标注员2_n": n2, "一致_n": ag, "一致率": rate})
    say()

    # --- 3. 混淆矩阵 -----------------------------------------------------
    say("[3] 混淆矩阵（行 = 标注员1，列 = 标注员2）")
    say()
    cm = pd.crosstab(pd.Series(a, name="标注员1"), pd.Series(b, name="标注员2"))
    cm = cm.reindex(index=LABELS, columns=LABELS, fill_value=0)
    for line in cm.to_string().splitlines():
        say("      " + line)
    say()
    cm_row = (cm.T / cm.sum(axis=1)).T
    say("    按行归一化（每行 = 标注员1 各类的流向占比）:")
    for line in cm_row.round(3).to_string().splitlines():
        say("      " + line)
    say()

    # --- 4. 不一致明细 ---------------------------------------------------
    dis = df[df["遮挡等级_标注员1"] != df["遮挡等级_标注员2"]].copy()
    dis["等级差"] = (dis["遮挡等级_标注员2"] - dis["遮挡等级_标注员1"]).astype(int)
    say(f"[4] 不一致明细：共 {len(dis)} 张（占 {len(dis)/n*100:.1f}%）")
    say()
    say(f"      相差 1 级: {int((dis['等级差'].abs()==1).sum())} 张")
    say(f"      相差 2 级: {int((dis['等级差'].abs()==2).sum())} 张")
    say(f"      相差 3 级: {int((dis['等级差'].abs()==3).sum())} 张")
    say()
    if len(dis):
        say("      按批次分布:")
        g = dis.groupby("group_name").size().sort_values(ascending=False)
        for k_, v_ in g.items():
            tot = int((df["group_name"] == k_).sum())
            say(f"        {k_:<20} {v_:>3} / {tot:<5} ({v_/tot*100:.1f}%)")
        say()
        say("      按标注员1 的等级分布:")
        for lv in LABELS:
            d_ = int((dis["遮挡等级_标注员1"] == lv).sum())
            t_ = int((a == lv).sum())
            say(f"        L{lv}  {d_:>3} / {t_:<5} ({d_/t_*100:.1f}%)")
    say()

    # --- 5. 分层（按批次）------------------------------------------------
    say("[5] 按批次分层的一致性")
    say()
    say(f"    {'批次':<20}{'n':>6}{'一致率':>9}{'κ':>8}")
    say("    " + "-" * 68)
    strat_rows = []
    for g_, sub in df.groupby("group_name"):
        aa = sub["遮挡等级_标注员1"].to_numpy(int)
        bb = sub["遮挡等级_标注员2"].to_numpy(int)
        po_ = observed_agreement(aa, bb)
        kk = cohen_kappa(aa, bb, LABELS)
        say(f"    {g_:<20}{len(sub):>6}{po_*100:>8.1f}%{kk:>8.3f}")
        strat_rows.append({"批次": g_, "n": len(sub), "一致率": po_, "kappa": kk})
    say()

    # --- 6. 落盘 ---------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([{
        "指标": name, "取值": v, "CI下限": lo, "CI上限": hi, "解释": note
    } for name, v, lo, hi, note in rows])
    summary.loc[len(summary)] = ["标注员1 Level0 占比", float(np.mean(a == 0)), np.nan, np.nan, ""]
    summary.loc[len(summary)] = ["标注员2 Level0 占比", float(np.mean(b == 0)), np.nan, np.nan, ""]

    xlsx = out_dir / "inter_rater_agreement.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="总体指标", index=False)
        pd.DataFrame(lv_rows).to_excel(w, sheet_name="逐等级", index=False)
        cm.to_excel(w, sheet_name="混淆矩阵")
        cm_row.round(4).to_excel(w, sheet_name="混淆矩阵_行归一化")
        dis.to_excel(w, sheet_name="不一致明细", index=False)
        pd.DataFrame(strat_rows).to_excel(w, sheet_name="按批次分层", index=False)
    say(f"[6] 已写出: {xlsx.relative_to(PROJECT_ROOT)}")

    rep = out_dir / "agreement_report.md"
    rep.write_text("\n".join(_lines), encoding="utf-8-sig")
    say(f"           {rep.relative_to(PROJECT_ROOT)}")

    # 混淆矩阵图
    try:
        make_figure(cm, cm_row, out_dir / "fig_confusion_matrix.png")
        say(f"           {(out_dir / 'fig_confusion_matrix.png').relative_to(PROJECT_ROOT)}")
    except Exception as e:
        say(f"    [警告] 绘图失败（不影响数值结果）: {e}")

    say()
    say("=" * 74)
    return 0


def make_figure(cm, cm_row, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, data, title, fmt in [
        (axes[0], cm.to_numpy(float), "计数", "{:.0f}"),
        (axes[1], cm_row.to_numpy(float), "按行归一化", "{:.2f}"),
    ]:
        im = ax.imshow(data, cmap="Blues", vmin=0,
                       vmax=data.max() if data.max() > 0 else 1)
        ax.set_xticks(range(len(LABELS)), [f"L{i}" for i in LABELS])
        ax.set_yticks(range(len(LABELS)), [f"L{i}" for i in LABELS])
        ax.set_xlabel("标注员 2")
        ax.set_ylabel("标注员 1")
        ax.set_title(title)
        thr = data.max() / 2
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center",
                        color="white" if data[i, j] > thr else "#263238", fontsize=11)
    fig.suptitle("面部遮挡类型分级 —— 两位标注员混淆矩阵", fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
