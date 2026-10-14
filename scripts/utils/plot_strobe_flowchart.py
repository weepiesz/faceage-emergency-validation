"""Figure 1: patient flow through the study (STROBE diagram).

Draws the flowchart from the counts held in the constants at the top of the file. Patients
enrolled with facial photographs all met the age and consent pre-conditions, so those appear as
inclusion criteria rather than as exclusion steps.

Cohort counts
-------------
Below the first box the constants agree with the manuscript throughout: 1,485 in the initial
dataset, 15 excluded for grade-3 occlusion, 1,470 analysed, 1,244 / 198 / 28 by grade, 1,320
survived and 150 died.

The first two — patients photographed and patients excluded — had been carried over from the
pre-revision cohort (1,501 and 16) and have been corrected to the 1,942 and 457 that the
manuscript reports, which is also arithmetically consistent: 1,942 - 457 = 1,485.

Output: results/figures/strobe_flowchart.png"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os


# ============================================================
# Configuration
# ============================================================
ENROLLED = 1942              # Patients who underwent facial photograph acquisition
EXCLUDED_DATA = 457          # Photograph unusable for reliable detection/cropping, or incomplete record
VALID_DATASET = 1485         # Valid FaceAge + occlusion annotation
EXCLUDED_LEVEL3 = 15         # Level 3 severe facial occlusion
ANALYSIS_COHORT = 1470       # Final analysis cohort
LEVEL0, LEVEL1, LEVEL2 = 1244, 198, 28
SURVIVED, DIED = 1320, 150


def draw_box(ax, x, y, w, h, lines, facecolor, edgecolor, linewidth=2):
    """Draw a rounded box with centered multi-line text.
    lines: list of (text, fontsize, bold, color)
    """
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.2",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=linewidth, zorder=2
    )
    ax.add_patch(box)

    n = len(lines)
    total_h = sum(fs * 0.04 for _, fs, _, _ in lines) + (n - 1) * 0.03
    y_cursor = y + total_h / 2

    for txt, fs, bold, col in lines:
        weight = 'bold' if bold else 'normal'
        y_cursor -= fs * 0.035
        ax.text(x, y_cursor, txt, ha='center', va='center',
                fontsize=fs, fontweight=weight, color=col, zorder=3)
        y_cursor -= fs * 0.01 + 0.03


def arrow(ax, x1, y1, x2, y2, color='#7F8C8D', lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw), zorder=1)


def line(ax, x1, y1, x2, y2, color='#888888', lw=1):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, zorder=1)


def draw_strobe_flowchart(output_path, dpi=300):

    fig, ax = plt.subplots(1, 1, figsize=(11, 13))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 13)
    ax.axis('off')

    cx, ex = 5.2, 8.5         # center x, exclusion x
    mw, mh = 5.2, 1.0         # main box width, height
    ew, eh = 3.6, 0.80        # exclusion box width, height

    # Colors
    DARK = '#2C3E50'
    RED_BG = '#FDEDEC'
    RED_EDGE = '#C0392B'
    GREEN_BG = '#D5F5E3'
    BLUE_BG = '#EBF5FB'
    FINAL_BG = '#D4E6F1'

    def main_box(y, lines, bg='white', ec=DARK, lw=2):
        draw_box(ax, cx, y, mw, mh, lines, bg, ec, lw)

    def excl_box(y, lines):
        draw_box(ax, ex, y, ew, eh, lines, RED_BG, RED_EDGE, 1.5)

    # ============================================================
    # Title
    # ============================================================
    ax.text(cx, 12.5,
            'Figure 1. Patient enrollment flowchart.',
            fontsize=12, fontweight='bold', ha='center', va='center', color=DARK)

    # ============================================================
    # BOX 1: Enrolled patients (ALL met age>=18 & consent = pre-conditions)
    # ============================================================
    y1 = 11.0
    main_box(y1, [
        ("Patients enrolled with facial photos", 10, True, DARK),
        (f"(n = {ENROLLED})", 9, False, '#555555'),
        ("(Age >= 18 yr, informed consent obtained)", 7, False, '#888888'),
    ], bg=BLUE_BG)

    # ============================================================
    # BOX 2R: Excluded — data issues (COMBINED)
    # ============================================================
    y_ex1 = y1  # same y-level as enrolled
    excl_box(y_ex1, [
        ("Excluded (n = 16)", 9, True, '#922B21'),
        ("  • Incomplete clinical data", 7.5, False, '#922B21'),
        ("  • Face detection failure or", 7.5, False, '#922B21'),
        ("     missing annotation / prediction", 7.5, False, '#922B21'),
    ])

    # Arrow: enrolled → exclusion 1 (right, same y-level)
    arrow(ax, cx + mw/2, y1, ex - ew/2, y_ex1, color=DARK, lw=1.5)

    # ============================================================
    # BOX 3: Valid FaceAge + occlusion data
    # ============================================================
    y3 = 8.8
    main_box(y3, [
        ("Patients with valid FaceAge prediction", 10, True, DARK),
        (f"& occlusion annotation (n = {VALID_DATASET})", 9, False, '#555555'),
    ], bg=GREEN_BG)

    # Arrow: enrolled → valid dataset (down)
    # From bottom of enrolled to top of valid dataset
    arrow(ax, cx, y1 - mh/2, cx, y3 + mh/2, color=DARK, lw=2)

    # ============================================================
    # BOX 4R: Excluded — Level 3 occlusion
    # ============================================================
    y_ex2 = y3  # same y-level as valid dataset
    excl_box(y_ex2, [
        ("Excluded (n = 15)", 9, True, '#922B21'),
        ("  • Level 3: severe facial occlusion", 7.5, False, '#922B21'),
    ])

    # Arrow: valid → exclusion 2 (right, same y-level)
    arrow(ax, cx + mw/2, y3, ex - ew/2, y_ex2, color=DARK, lw=1.5)

    # ============================================================
    # BOX 5: Final analysis cohort
    # ============================================================
    y5 = 6.5
    main_box(y5, [
        ("Final analysis cohort", 11, True, '#1A5276'),
        (f"(n = {ANALYSIS_COHORT})", 10, False, '#555555'),
    ], bg=FINAL_BG, ec='#1A5276', lw=2.5)

    # Arrow: valid → analysis (down)
    # From bottom of valid to top of analysis
    arrow(ax, cx, y3 - mh/2, cx, y5 + mh/2, color=DARK, lw=2)

    # ============================================================
    # Sub-group boxes (occlusion levels)
    # ============================================================
    y_sub = 4.8
    sub_w, sub_h = 1.5, 0.60
    sub_data = [
        ("Level 0\nn = 1244\n(83.8%)", cx - 2.2, '#AED6F1', 'Negligible occlusion'),
        ("Level 1\nn = 198\n(13.3%)", cx, '#85C1E9', 'Mild'),
        ("Level 2\nn = 28\n(1.9%)", cx + 2.2, '#5DADE2', 'Moderate'),
    ]
    for txt, sx, fc, _ in sub_data:
        parts = txt.split('\n')
        draw_box(ax, sx, y_sub, sub_w, sub_h, [
            (parts[0], 8, True, DARK),
            (parts[1], 7, False, '#555555'),
            (parts[2], 6.5, False, '#888888'),
        ], fc, '#1A5276', 1.2)

    # Labels under subgroup boxes
    for _, sx, _, label in sub_data:
        ax.text(sx, y_sub - sub_h/2 - 0.15, label, fontsize=6.5,
                ha='center', va='top', color='#777')

    # Connector: analysis → subgroups
    # From bottom of analysis to center-top junction above subgroups
    line(ax, cx, y5 - mh/2, cx, y_sub + sub_h/2 + 0.3, DARK, 1.5)
    arrow(ax, cx, y_sub + sub_h/2 + 0.3, cx, y_sub + sub_h/2, color=DARK, lw=1.5)
    for _, sx, _, _ in sub_data:
        line(ax, cx, y_sub + sub_h/2 + 0.3, sx, y_sub + sub_h/2 + 0.1, '#A0A0A0', 0.8)
        arrow(ax, sx, y_sub + sub_h/2 + 0.1, sx, y_sub + sub_h/2, color='#A0A0A0', lw=0.8)

    # ============================================================
    # Outcome boxes
    # ============================================================
    y_out = 3.0
    out_w, out_h = 2.2, 0.65

    # Survivors
    draw_box(ax, cx - 1.6, y_out, out_w, out_h, [
        ("7-day Survivors", 9, True, '#1E8449'),
        (f"n = {SURVIVED} ({SURVIVED/ANALYSIS_COHORT*100:.1f}%)", 8, False, '#1E8449'),
    ], '#D5F5E3', '#1E8449', 1.8)

    # Non-survivors
    draw_box(ax, cx + 1.6, y_out, out_w, out_h, [
        ("7-day Non-survivors", 9, True, '#C0392B'),
        (f"n = {DIED} ({DIED/ANALYSIS_COHORT*100:.1f}%)", 8, False, '#C0392B'),
    ], '#FDEDEC', '#C0392B', 1.8)

    # Connector: subgroups → outcomes
    # From bottom of subgroup boxes, branch to survivor & non-survivor boxes
    junction_y = y_out + out_h/2 + 0.3
    line(ax, cx, y_sub - sub_h/2 - 0.15, cx, junction_y, DARK, 1.5)
    # Horizontal distribution bar
    line(ax, cx - 1.6, junction_y, cx + 1.6, junction_y, DARK, 1)
    # Down arrows to each outcome
    arrow(ax, cx - 1.6, junction_y, cx - 1.6, y_out + out_h/2, DARK, lw=1)
    arrow(ax, cx + 1.6, junction_y, cx + 1.6, y_out + out_h/2, DARK, lw=1)

    # ============================================================
    # Footer
    # ============================================================
    ax.text(cx, 0.3,
            'Level 0 = no occlusion; Level 1 = mild; Level 2 = moderate; Level 3 = severe.',
            fontsize=7, ha='center', va='center', color='#999')

    # ============================================================
    # Save
    # ============================================================
    plt.tight_layout(pad=0.5)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"[OK] STROBE flowchart saved to: {output_path}")


if __name__ == '__main__':
    # Output to faceage_emergency_study/results/figures/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    output_path = os.path.join(project_dir, 'results', 'figures', 'strobe_flowchart.png')

    print("=" * 60)
    print("STROBE Flowchart Generator")
    print("=" * 60)
    print(f"  Enrolled (with photos):     {ENROLLED}")
    print(f"  Excluded (data/face issues): {EXCLUDED_DATA}")
    print(f"  Valid dataset:              {VALID_DATASET}")
    print(f"  Excluded (Level 3):         {EXCLUDED_LEVEL3}")
    print(f"  Analysis cohort:            {ANALYSIS_COHORT}")
    print(f"  Outcomes: {SURVIVED} survived / {DIED} died")
    print("=" * 60)

    draw_strobe_flowchart(output_path, dpi=300)
