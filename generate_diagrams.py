"""
UGDR Research — Publication Diagram Generator
=============================================
Generates two publication-quality figures using only matplotlib.

Output:
  results/diagrams/figure1_pipeline.png
  results/diagrams/figure2_architecture.png
"""

import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Output directory ──────────────────────────────────────────────────────────
OUT = pathlib.Path("results/diagrams")
OUT.mkdir(parents=True, exist_ok=True)
print(f"[Setup] Output directory: {OUT.resolve()}\n")

# ── Colour palette ────────────────────────────────────────────────────────────
C = {
    "data"  : "#4A90D9",   # blue    – dataset / split
    "train" : "#43A047",   # green   – training
    "noise" : "#D32F2F",   # red     – noise / corrupted
    "uncert": "#F9A825",   # amber   – uncertainty
    "refine": "#7B1FA2",   # purple  – detection / refinement
    "final" : "#1B5E20",   # dk green – final models
    "eval"  : "#1A237E",   # dk navy – evaluation
}
BG = "#FAFAFA"


# ── Drawing helpers ───────────────────────────────────────────────────────────
def rbox(ax, cx, cy, w, h, label, sub=None,
         color="#4A90D9", tc="white", fs=9.2, sfs=7.4):
    """Rounded rectangle with optional sub-label."""
    p = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle="round,pad=0.07", linewidth=1.2,
                       edgecolor="white", facecolor=color, zorder=3)
    ax.add_patch(p)
    yoff = 0.12 if sub else 0.0
    ax.text(cx, cy + yoff, label, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=tc, zorder=4)
    if sub:
        ax.text(cx, cy - 0.20, sub, ha="center", va="center",
                fontsize=sfs, color=tc, alpha=0.92, zorder=4)


def arr(ax, x1, y1, x2, y2, c="#555", lw=1.6, rad=None):
    """Straight or curved arrow."""
    style = f"arc3,rad={rad}" if rad is not None else "arc3,rad=0"
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=c, lw=lw,
                                connectionstyle=style), zorder=2)


def dline(ax, xs, ys, c="#999", lw=1.2, ls="--"):
    """Polyline (no arrowhead)."""
    ax.plot(xs, ys, color=c, lw=lw, linestyle=ls, zorder=1)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – UGDR Pipeline
# ═════════════════════════════════════════════════════════════════════════════
def figure1():
    fig, ax = plt.subplots(figsize=(14, 19))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 14);  ax.set_ylim(-0.3, 18.5);  ax.axis("off")

    BW, BH = 6.0, 0.72        # shared width / box height
    CW     = 4.8              # column width
    LX, RX = 3.0, 11.0       # left / right column centres

    # Title
    ax.text(7, 18.1,
            "Uncertainty-Guided Data Refinement (UGDR) Pipeline",
            ha="center", va="center", fontsize=13.5,
            fontweight="bold", color="#1A1A2E")

    # ── Shared: Dataset → Split ───────────────────────────────────────────
    rbox(ax, 7, 17.3, BW, BH, "Chest X-Ray Dataset",
         "5,855 images  |  NORMAL 27%  +  PNEUMONIA 73%", color=C["data"])
    rbox(ax, 7, 16.1, BW, BH, "Train / Validation / Test Split",
         "70% / 15% / 15%   stratified,  seed = 42", color=C["data"])
    arr(ax, 7, 16.94, 7, 16.46)

    # ── Branch arrows from Split ──────────────────────────────────────────
    arr(ax, 5.4, 15.74, LX, 15.11, c=C["train"], rad=-0.20)   # → clean
    arr(ax, 8.6, 15.74, RX, 15.11, c=C["noise"],  rad= 0.20)  # → noisy

    # ── LEFT: Clean path ─────────────────────────────────────────────────
    ax.text(0.55, 13.8, "CLEAN\nPATH", ha="center", va="center",
            fontsize=7.5, fontweight="bold", color=C["train"],
            bbox=dict(facecolor="#E8F5E9", edgecolor=C["train"],
                      boxstyle="round,pad=0.3", alpha=0.9))

    rbox(ax, LX, 14.75, CW, BH, "Clean Model Training",
         "ResNet18 + Dropout(0.5)  |  Adam lr=1e-4  |  15 epochs", color=C["train"])
    arr(ax, LX, 14.39, LX, 13.72)
    rbox(ax, LX, 13.35, CW, BH, "CLEAN Model",
         "F1 = 92.9%   (performance ceiling / reference)", color=C["final"])

    # Dashed connector: Clean model → evaluation box (left spine)
    dline(ax, [LX, LX, 4.0], [12.99, 0.75, 0.75], c=C["final"], lw=1.3)
    ax.annotate("", xy=(4.0, 0.75), xytext=(3.9, 0.75),
                arrowprops=dict(arrowstyle="->", color=C["final"], lw=1.4), zorder=2)

    # ── RIGHT: UGDR path ─────────────────────────────────────────────────
    ax.text(13.45, 10.2, "UGDR\nPATH", ha="center", va="center",
            fontsize=7.5, fontweight="bold", color=C["noise"],
            bbox=dict(facecolor="#FFEBEE", edgecolor=C["noise"],
                      boxstyle="round,pad=0.3", alpha=0.9))

    right_steps = [
        (14.75, "Noise Injection  (Synthetic)",
         "20% symmetric flip  |  ~734 labels corrupted  (NORMAL ↔ PNEUMONIA)", C["noise"]),
        (13.35, "Noisy Model Training",
         "Identical architecture + hyperparameters", "#E53935"),
        (11.95, "NOISY Model",
         "F1 = 87.1%   (−5.8 pp vs clean baseline)", "#C62828"),
        (10.55, "MC Dropout Inference",
         "T = 30 passes  |  BatchNorm=eval   Dropout=train", C["uncert"]),
        ( 9.15, "Uncertainty Estimation",
         "μ(x), σ²(x),  u(x) = max σ²_c(x)   per training sample", C["uncert"]),
        ( 7.75, "Dual-Criterion Error Detection",
         "A: u(x) > 0.02      B: ŷ ≠ y  AND  conf > 0.70", C["refine"]),
        ( 6.35, "Remove Suspicious Samples",
         "~726 flagged removed  |  ~2,943 retained  (80% of training set)", C["refine"]),
    ]

    for cy, label, sub, col in right_steps:
        tc = "#1A1A2E" if col == C["uncert"] else "white"
        rbox(ax, RX, cy, CW, BH, label, sub, color=col, tc=tc)

    # Vertical arrows on right column
    for y_top in [14.39, 12.99, 11.59, 10.19, 8.79, 7.39]:
        arr(ax, RX, y_top, RX, y_top - 0.67)

    # ── Merge: Remove Samples → Refined Dataset ───────────────────────────
    arr(ax, RX, 5.99, 7, 5.31, c=C["refine"], rad=0.25)

    # ── Merged column: Refined → Retrain → Cleaned ───────────────────────
    rbox(ax, 7, 4.95, BW, BH, "Refined Dataset  (D_refined)",
         "80% of training set retained  |  No noise labels required", color=C["refine"])
    arr(ax, 7, 4.59, 7, 3.92)

    rbox(ax, 7, 3.55, BW, BH, "Retraining on Refined Data",
         "Same ResNet18 architecture  |  Identical hyperparameters", color=C["train"])
    arr(ax, 7, 3.19, 7, 2.52)

    rbox(ax, 7, 2.15, BW, BH, "CLEANED Model",
         "F1 = 91.9%  |  +4.8 pp vs NOISY  |  83% degradation recovery",
         color=C["final"])
    arr(ax, 7, 1.79, 7, 1.11)

    # ── Evaluation bar ────────────────────────────────────────────────────
    rbox(ax, 7, 0.75, 11, 0.70, "Final Evaluation — Test Set  (n = 878)",
         "CLEAN: F1=92.9%     NOISY: F1=87.1%     CLEANED: F1=91.9%     Recovery: 83%",
         color=C["eval"], fs=9.5, sfs=8.0)

    # ── Legend ────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(facecolor=C["data"],   label="Dataset / Split"),
        mpatches.Patch(facecolor=C["train"],  label="Model Training"),
        mpatches.Patch(facecolor=C["noise"],  label="Noise / Corrupted"),
        mpatches.Patch(facecolor=C["uncert"], label="Uncertainty Estimation"),
        mpatches.Patch(facecolor=C["refine"], label="Detection / Refinement"),
        mpatches.Patch(facecolor=C["final"],  label="Final Models"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, 0.01),
              fontsize=8.5, framealpha=0.92, edgecolor="#BDBDBD")

    path = OUT / "figure1_pipeline.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓ Saved: {path.resolve()}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Model Architecture
# ═════════════════════════════════════════════════════════════════════════════
def figure2():
    fig, ax = plt.subplots(figsize=(16, 6.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 20);  ax.set_ylim(0, 7);  ax.axis("off")

    ax.text(10, 6.55,
            "Model Architecture with MC Dropout for Uncertainty Estimation",
            ha="center", va="center", fontsize=13, fontweight="bold", color="#1A1A2E")

    # Component definitions: (cx, title, subtitle, fill, text_colour)
    CY = 3.5   # vertical centre of all boxes
    BW = 3.0;  BH = 3.2

    components = [
        (2,  "Input Image",
         "224 × 224 × 3\n(RGB)\nImageNet-normalised",
         "#4A90D9", "white"),
        (6,  "ResNet18 Backbone",
         "4 res. stage groups\n~11.18M parameters\nImageNet-1K pretrained",
         "#78909C", "white"),
        (10, "Dropout Layer",
         "p = 0.5\n\u26a1 ACTIVE at inference\n(MC Dropout mode)",
         "#F9A825", "#1A1A2E"),
        (14, "Fully Connected",
         "Linear  512 \u2192 2\n[NORMAL, PNEUMONIA]\n(logits)",
         "#78909C", "white"),
        (18, "Softmax Output",
         "Class probabilities\n\u2192 NORMAL / PNEUMONIA\n+ predicted label",
         "#2E7D32", "white"),
    ]

    for cx, title, sub, col, tc in components:
        p = FancyBboxPatch((cx - BW/2, CY - BH/2), BW, BH,
                           boxstyle="round,pad=0.1", linewidth=1.5,
                           edgecolor="white", facecolor=col, zorder=3)
        ax.add_patch(p)
        ax.text(cx, CY + 0.55, title, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=tc, zorder=4)
        ax.text(cx, CY - 0.40, sub, ha="center", va="center",
                fontsize=8.2, color=tc, alpha=0.93, zorder=4)

    # Arrows between boxes
    gaps = [(2+BW/2, 6-BW/2),
            (6+BW/2, 10-BW/2),
            (10+BW/2, 14-BW/2),
            (14+BW/2, 18-BW/2)]
    for x1, x2 in gaps:
        arr(ax, x1, CY, x2, CY, c="#424242", lw=2.0)

    # 512-d label above backbone → dropout gap
    ax.text(8, CY + 2.1, "512-d feature vector\n(Global Avg Pool output)",
            ha="center", va="center", fontsize=7.8, color="#546E7A",
            bbox=dict(facecolor="#ECEFF1", edgecolor="#90A4AE",
                      boxstyle="round,pad=0.3", alpha=0.9))
    dline(ax, [8, 8], [CY + 1.85, CY + BH/2], c="#90A4AE", lw=1.0, ls="-")

    # MC Dropout annotation below dropout box
    ax.text(10, CY - BH/2 - 0.65,
            "TRAINING: Dropout = regularisation\n"
            "MC INFERENCE: T=30 passes \u2192 predictive variance \u2192 noise-detection signal",
            ha="center", va="center", fontsize=8.2, color="#4A148C",
            bbox=dict(facecolor="#F3E5F5", edgecolor="#7B1FA2",
                      boxstyle="round,pad=0.4", alpha=0.93))
    # Arrow up from annotation to dropout box
    arr(ax, 10, CY - BH/2 - 0.20, 10, CY - BH/2,
        c="#7B1FA2", lw=1.3, rad=0)

    # Mode labels under Training / Inference branches (visual cue)
    for cx, lbl, col in [(6, "Deterministic\ninference", "#546E7A"),
                          (14, "Two logits\nno activation", "#546E7A")]:
        ax.text(cx, CY - BH/2 - 0.50, lbl,
                ha="center", va="top", fontsize=7, color=col, style="italic")

    path = OUT / "figure2_architecture.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓ Saved: {path.resolve()}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 58)
    print("  UGDR Research — Diagram Generator")
    print("=" * 58)

    print("\n[1/2] Generating UGDR Pipeline diagram …")
    figure1()

    print("[2/2] Generating Model Architecture diagram …")
    figure2()

    print("\n" + "=" * 58)
    print("  All diagrams saved:")
    for name in ["figure1_pipeline.png", "figure2_architecture.png"]:
        fp = OUT / name
        tag = "✓" if fp.exists() else "✗  MISSING"
        print(f"  {tag}  {fp.resolve()}")
    print()


if __name__ == "__main__":
    main()
