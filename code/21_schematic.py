#!/usr/bin/env python3
"""Study-overview schematic (manuscript Figure 1) + standalone graphical abstract,
built per clinical-ai-figure-design (Omar/Topol LDH template). One contrast: first-line
opioids -> more all-cause ED return vs guideline-preferred dopamine antagonists,
across two health systems. Honest device: the headache-specific-return bar is drawn grey and
crossing the null (RR 1) so the non-significant sensitivity is visible, not asserted in words.
No in-figure title on the manuscript Figure 1 (caption carries it); the graphical-abstract
variant adds a headline strip. Numbers are transcribed constants (no recomputation)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

FIG = os.path.join(os.environ.get("HEADACHE_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")), "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})

OPI = "#D55E00"; DOP = "#0072B2"; INK = "#2B2B2B"; MUT = "#6B7280"
GREY = "#ADB6B6"; BOXF = "#FAFBFD"; EDGE = "#B5BDC8"

# finding block data: (label, rr, lo, hi, color, note)
# Index admission dropped from this panel: it follows the treatment decision and is
# driven by the same triage-acuity signal, so it is not an acute-care outcome of treatment.
#
# These are hard-coded rather than read from results/results_digest_pooled.json and
# results/results_ha_return.json, so they go stale silently whenever the cohort changes.
# Re-check them against those digests before regenerating Figure 1 or the graphical abstract.
# The 72-hour row was 1.79, 1.31, 2.33 until the MC-MED .first() repair in 04_mcmed_analysis.py
# moved the pooled estimate 1.786 -> 1.784, which flips the two-decimal rendering to 1.78.
FINDINGS = [
    ("72-h all-cause return", 1.78, 1.31, 2.32, OPI, ""),
    ("Headache-specific return", 1.31, 0.91, 1.88, GREY, "attenuated"),
]

def block(ax, x, y, w, h, num, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.6",
                                fc=BOXF, ec=EDGE, lw=1.0, zorder=1))
    ax.text(x + 2.2, y + h - 2.6, str(num), fontsize=12, fontweight="bold", color=INK, va="top", zorder=3)
    ax.text(x + 6.8, y + h - 2.8, title, fontsize=9.2, fontweight="bold", color=INK, va="top", zorder=3)

def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11,
                                 lw=1.1, color=MUT, zorder=2, shrinkA=1, shrinkB=1))

def mini_forest(ax, x0, x1, ytop, row_gap):
    # linear RR axis from 0.8 to 3.0 mapped into [xb0, xb1]; xb1 leaves a right margin
    # so the "RR (95% CI)" value column stays clear of the block-4 frame edge
    rr_lo, rr_hi, xb0, xb1 = 0.8, 3.0, x0 + 34, x1 - 22
    xr = lambda rr: xb0 + (rr - rr_lo) / (rr_hi - rr_lo) * (xb1 - xb0)
    xnull = xr(1.0)
    # fixed per-row spacing (rather than stretching to fill ytop..ybot) so the block reads the
    # same whether it holds two rows or three
    rows_y = [ytop - 3 - i * row_gap for i in range(len(FINDINGS))]
    ybot = rows_y[-1]
    ax.plot([xnull, xnull], [ybot, ytop], ls="--", lw=0.8, color=MUT, zorder=2)
    ax.text(xnull, ytop + 1.4, "RR 1", ha="center", fontsize=6.5, color=MUT)
    for (lab, rr, lo, hi, col, note), y in zip(FINDINGS, rows_y):
        ax.text(x0 + 2, y, lab, fontsize=7.6, color=INK, va="center")
        ax.plot([xr(lo), xr(hi)], [y, y], lw=1.8, color=col, zorder=3, solid_capstyle="round")
        ax.plot(xr(rr), y, "o", ms=5.5, color=col, zorder=4)
        txt = f"{rr:.2f} ({lo:.2f} to {hi:.2f})"
        ax.text(xb1 + 2, y, txt, fontsize=7.0, color=(MUT if col == GREY else INK), va="center")
    # directional labels flanking the RR = 1 line, anchored off xnull so they stay correct if
    # rr_lo/rr_hi ever change. RR > 1 = more returns after opioids, so the right side of the
    # line is the side that disfavours opioids. Labels are outward-aligned (ha="right"/"left")
    # rather than centered on a fixed span, since the two strings are very different lengths
    # and centering them collided in the middle.
    arr_y = ybot - 3.2
    lab_y = ybot - 5.0
    gap, arrow_len = 2.0, 9.0
    ax.annotate("", xy=(xnull - gap - arrow_len, arr_y), xytext=(xnull - gap, arr_y),
                arrowprops=dict(arrowstyle="->", lw=0.8, color=MUT))
    ax.annotate("", xy=(xnull + gap + arrow_len, arr_y), xytext=(xnull + gap, arr_y),
                arrowprops=dict(arrowstyle="->", lw=0.8, color=MUT))
    ax.text(xnull - gap, lab_y, "Favors opioid", ha="right", va="top",
            fontsize=6.6, color=MUT)
    ax.text(xnull + gap, lab_y, "Favors dopamine antagonist", ha="left", va="top",
            fontsize=6.6, color=MUT)

def build(headline=None, outstem="Figure1_study_schematic"):
    extra = 0 if headline is None else 8
    fig, ax = plt.subplots(figsize=(7.4, 4.9 + (0.55 if headline is not None else 0)))
    ax.set_xlim(0, 100); ax.set_ylim(0, 66 + extra); ax.axis("off")
    top = 62  # inner border height, identical in both variants (headline stacks ABOVE the border)
    if headline is not None:
        ax.add_patch(FancyBboxPatch((1, 65), 98, 6.6, boxstyle="round,pad=0,rounding_size=1.4",
                                    fc=OPI, ec="none", alpha=0.10, zorder=0))
        ax.text(50, 68.3, headline, ha="center", va="center", fontsize=8.3, fontweight="bold", color=INK)
    ax.add_patch(FancyBboxPatch((1, 1), 98, top, boxstyle="round,pad=0,rounding_size=1.6",
                                fc="none", ec=EDGE, lw=1.2, zorder=0))
    ty, th = top - 27, 25  # top-row block y and height (= 35, identical in both variants)
    # (1) Setting
    block(ax, 4, ty, 28, th, 1, "Adult ED headache")
    ax.text(7, ty + th - 7.5, "13,285 adults", fontsize=8.6, color=INK)
    ax.text(7, ty + th - 11, "2 health systems", fontsize=8.6, color=INK)
    ax.text(7, ty + th - 14.3, "MIMIC-IV-ED (BIDMC)", fontsize=6.6, color=MUT)
    ax.text(7, ty + th - 17.0, "+ MC-MED (Stanford)", fontsize=6.6, color=MUT)
    ax.text(7, ty + th - 20.5, "first treated visit/patient", fontsize=6.8, color=MUT)
    # (2) Comparison
    block(ax, 36, ty, 28, th, 2, "First-line class")
    ax.add_patch(FancyBboxPatch((39, ty + th - 12), 9.0, 5.4, boxstyle="round,pad=0.2,rounding_size=1",
                                fc=OPI, ec="none")); ax.text(43.5, ty + th - 9.3, "Opioid", ha="center", va="center", color="white", fontsize=7.6, fontweight="bold")
    ax.text(50.0, ty + th - 9.3, "vs", ha="center", va="center", fontsize=7.5, color=INK)
    ax.add_patch(FancyBboxPatch((52, ty + th - 12), 10.0, 5.4, boxstyle="round,pad=0.2,rounding_size=1",
                                fc=DOP, ec="none")); ax.text(57.0, ty + th - 9.3, "Dopamine\nantag.", ha="center", va="center", color="white", fontsize=6.6, fontweight="bold")
    ax.text(39, ty + th - 15.5, "guideline-preferred", fontsize=6.6, color=MUT)
    ax.text(39, ty + th - 18.3, "comparator", fontsize=6.6, color=MUT)
    ax.text(39, ty + th - 21.2, "others: NSAID, acetaminophen,", fontsize=5.9, color=MUT)
    ax.text(39, ty + th - 23.2, "triptan, IV magnesium", fontsize=5.9, color=MUT)
    # (3) Measure
    block(ax, 68, ty, 28, th, 3, "All-cause ED return")
    ax.text(71, ty + th - 7.5, "any subsequent", fontsize=8.0, color=INK)
    ax.text(71, ty + th - 10.6, "ED visit", fontsize=8.0, color=INK)
    ax.text(71, ty + th - 14.3, "72 h · 7 d", fontsize=8.0, color=INK)
    ax.text(71, ty + th - 18.0, "overlap-weighted;", fontsize=6.6, color=MUT)
    ax.text(71, ty + th - 20.0, "routinely captured", fontsize=6.6, color=MUT)
    # arrows across the top row
    arrow(ax, 32, ty + th / 2, 36, ty + th / 2)
    arrow(ax, 64, ty + th / 2, 68, ty + th / 2)
    # down-arrow into the finding block (block 4 top is at y=29; ty=35 so this always points down)
    arrow(ax, 50, ty, 50, 30)
    # (4) Finding (full width)
    block(ax, 4, 4, 92, 25, 4, "What we found")
    mini_forest(ax, 6, 96, 22, 6.5)
    fig.savefig(f"{FIG}/{outstem}.pdf", dpi=600, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(f"{FIG}/{outstem}.png", dpi=600, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig); print("wrote", outstem)

build(headline=None, outstem="Figure1_study_schematic")
build(headline="First-line opioids for ED headache were associated with higher all-cause ED return than dopamine antagonists",
      outstem="graphical_abstract")
print("schematic done")
