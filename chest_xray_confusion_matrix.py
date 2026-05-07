"""
Confusion Matrix Visualisation — Chest X-Ray Pneumonia Classifier
=================================================================
Plots side-by-side confusion matrices for:
  • Noisy Baseline
  • Hard Clean 10%

Data loaded from:  results/artifacts/metrics/final_metrics.json
Output saved to:   results/charts/plot_confusion_matrices.png
                   results/charts/plot_confusion_matrix_noisy.png
                   results/charts/plot_confusion_matrix_clean10.png

Usage
-----
  python chest_xray_confusion_matrix.py
"""

import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
_SCRIPT_DIR  = pathlib.Path(__file__).resolve().parent
METRICS_PATH = _SCRIPT_DIR / "results" / "artifacts" / "metrics" / "final_metrics.json"
OUT_DIR      = _SCRIPT_DIR / "results" / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Palette  (matches chest_xray_visualize.py)
# ─────────────────────────────────────────────
PALETTE = {
    "bg"      : "#1A1A2E",
    "panel"   : "#16213E",
    "grid"    : "#2A2A4A",
    "text"    : "#E0E0E0",
    "accent"  : "#F4D03F",
    "correct" : "#4CAF50",   # green  – TP / TN
    "error"   : "#EF5350",   # red    – FP / FN
    "border"  : "#0F3460",
}

plt.rcParams.update({
    "figure.facecolor"  : PALETTE["bg"],
    "axes.facecolor"    : PALETTE["panel"],
    "axes.edgecolor"    : PALETTE["grid"],
    "axes.labelcolor"   : PALETTE["text"],
    "axes.titlecolor"   : PALETTE["text"],
    "axes.titlesize"    : 13,
    "axes.labelsize"    : 11,
    "axes.grid"         : False,
    "xtick.color"       : PALETTE["text"],
    "ytick.color"       : PALETTE["text"],
    "xtick.labelsize"   : 10,
    "ytick.labelsize"   : 10,
    "text.color"        : PALETTE["text"],
    "font.family"       : "DejaVu Sans",
})

# ─────────────────────────────────────────────
# Load real metrics
# ─────────────────────────────────────────────
with open(METRICS_PATH) as f:
    all_metrics = json.load(f)

CONDITIONS = {
    "Noisy Baseline" : "NOISY",
    "Hard Clean 10%" : "CLEANED_10pct",
}

def extract(key):
    m  = all_metrics[key]
    cm = np.array(m["confusion_matrix"])   # [[TN, FP], [FN, TP]]
    return dict(
        cm        = cm,
        tn        = cm[0, 0],
        fp        = cm[0, 1],
        fn        = cm[1, 0],
        tp        = cm[1, 1],
        f1        = m["f1"]        * 100,
        accuracy  = m["accuracy"]  * 100,
        precision = m["precision"] * 100,
        recall    = m["recall"]    * 100,
    )

data = {label: extract(key) for label, key in CONDITIONS.items()}

# ─────────────────────────────────────────────
# Core drawing helper
# ─────────────────────────────────────────────
CELL_LABELS = [["TN", "FP"], ["FN", "TP"]]

def draw_cm(ax, d, title, show_delta=False):
    """Draw one confusion matrix onto *ax*."""
    cm  = d["cm"]
    tot = cm.sum()

    # cell colours  (normalised intensity)
    correct_mask = np.array([[True, False], [False, True]])

    for i in range(2):
        for j in range(2):
            val   = cm[i, j]
            pct   = val / tot * 100
            color = PALETTE["correct"] if correct_mask[i, j] else PALETTE["error"]

            # alpha-modulate so larger counts are more vivid
            intensity = 0.30 + 0.70 * (val / cm.max())
            r, g, b   = matplotlib.colors.to_rgb(color)
            cell_color = (r * intensity, g * intensity, b * intensity)

            rect = plt.Rectangle([j, 1 - i], 1, 1, color=cell_color,
                                  linewidth=2, edgecolor=PALETTE["border"],
                                  transform=ax.transData)
            ax.add_patch(rect)

            # label type  (TN / FP / FN / TP)
            ax.text(j + 0.5, 1 - i + 0.72, CELL_LABELS[i][j],
                    ha="center", va="center", fontsize=11,
                    color="white", fontweight="bold", alpha=0.75)

            # count
            ax.text(j + 0.5, 1 - i + 0.45, str(val),
                    ha="center", va="center", fontsize=26,
                    color="white", fontweight="bold")

            # percentage
            ax.text(j + 0.5, 1 - i + 0.20, f"({pct:.1f}%)",
                    ha="center", va="center", fontsize=9,
                    color="white", alpha=0.80)

    # axes formatting
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Predicted\nNormal", "Predicted\nAbnormal"],
                       fontsize=10, color=PALETTE["text"])
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Actual\nAbnormal", "Actual\nNormal"],
                       fontsize=10, color=PALETTE["text"])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # title
    ax.set_title(title, fontsize=13, fontweight="bold", color=PALETTE["accent"],
                 pad=10)

    # metrics footer inside the axes
    footer = (f"F1 {d['f1']:.2f}%   |   "
              f"Acc {d['accuracy']:.2f}%   |   "
              f"Prec {d['precision']:.2f}%   |   "
              f"Rec {d['recall']:.2f}%")
    ax.text(1.0, -0.08, footer,
            ha="center", va="top", fontsize=8.5,
            color=PALETTE["text"], alpha=0.85,
            transform=ax.transAxes)


# ─────────────────────────────────────────────
# Figure 1 — Side-by-side comparison
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5.4),
                          facecolor=PALETTE["bg"],
                          gridspec_kw={"wspace": 0.14})

for ax, (label, d) in zip(axes, data.items()):
    draw_cm(ax, d, label)

# Delta annotation between the two panels
noisy  = data["Noisy Baseline"]
clean  = data["Hard Clean 10%"]
delta_fp  = noisy["fp"] - clean["fp"]
delta_fn  = clean["fn"] - noisy["fn"]
delta_f1  = clean["f1"] - noisy["f1"]

fig.text(0.5, 0.97,
         "Confusion Matrix Comparison — Noisy Baseline vs Hard Clean 10%",
         ha="center", fontsize=14, fontweight="bold", color=PALETTE["text"])

# Build human-readable delta strings
fp_dir = "+" if clean["fp"] > noisy["fp"] else "-"
fn_dir = "+" if clean["fn"] > noisy["fn"] else "-"
fig.text(0.5, 0.91,
         (f"Hard Clean 10%:  "
          f"FP {noisy['fp']} → {clean['fp']}  ({fp_dir}{abs(delta_fp)})    "
          f"FN {noisy['fn']} → {clean['fn']}  ({fn_dir}{abs(delta_fn)})    "
          f"ΔF1 = +{delta_f1:.2f} pp"),
         ha="center", fontsize=9.5, color=PALETTE["accent"])

out_both = OUT_DIR / "plot_confusion_matrices.png"
fig.savefig(out_both, dpi=180, bbox_inches="tight",
            facecolor=PALETTE["bg"])
plt.close(fig)
print(f"[Saved] {out_both}")

# ─────────────────────────────────────────────
# Figure 2 & 3 — Individual matrices
# ─────────────────────────────────────────────
for label, d in data.items():
    fig2, ax2 = plt.subplots(figsize=(5.6, 5.0), facecolor=PALETTE["bg"])
    draw_cm(ax2, d, label)
    fig2.suptitle(f"Confusion Matrix — {label}",
                  fontsize=12, fontweight="bold",
                  color=PALETTE["text"], y=1.01)
    slug = label.lower().replace(" ", "_").replace("%", "pct")
    out_single = OUT_DIR / f"plot_confusion_matrix_{slug}.png"
    fig2.savefig(out_single, dpi=180, bbox_inches="tight",
                 facecolor=PALETTE["bg"])
    plt.close(fig2)
    print(f"[Saved] {out_single}")

print("\nAll confusion matrices saved to:", OUT_DIR)
