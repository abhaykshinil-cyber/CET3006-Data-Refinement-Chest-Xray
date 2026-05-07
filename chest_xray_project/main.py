"""
main.py - Master pipeline orchestrator.

Stages 1-6  : data loading, clean training, noise injection,
               noisy training, MC Dropout, error detection.
Stages 7-8  : threshold-based multi-experiment cleaning.
Stage  8W   : confidence-weighted loss training (no sample removal).
Stage  9    : full test-set evaluation for all models.
Stage  10   : plots 3-8 + auto paper summary.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import (
    BACKBONE, BASE_DIR, CKPT_CLEAN, CKPT_NOISY,
    CLEANING_LEVELS, CONFIDENCE_THRESH, DATASET_ROOT,
    EARLY_STOP_PAT, ENTROPY_ALPHA, LEARNING_RATE,
    MC_PASSES, NOISE_RATE, NUM_EPOCHS, PLOTS_DIR,
    RANDOM_SEED, UNCERTAINTY_THRESH, WEIGHT_FLOOR,
)
from data_loader import (
    build_dataloaders, build_noisy_dataloaders,
    collect_all_samples, stratified_split,
)
from detection import (
    assign_sample_weights, build_cleaned_loader,
    build_weighted_training_loader, detect_errors,
    print_detection_report,
)
from evaluation import (
    compute_metrics, generate_paper_summary,
    plot_roc_pr_curves, plot_uncertainty_histogram,
    plot_uncertainty_scatter, print_metrics,
    run_inference, run_inference_proba,
    save_confusion_matrices, save_metrics,
)
from model import build_model
from train import run_training, run_training_weighted
from uncertainty import compute_combined_score, mc_dropout_predict


ARTIFACTS_DIR   = BASE_DIR / "results" / "artifacts"
HISTORY_DIR     = ARTIFACTS_DIR / "histories"
METRICS_DIR     = ARTIFACTS_DIR / "metrics"
UNCERTAINTY_DIR = ARTIFACTS_DIR / "uncertainty"

_PAL_BG   = "#1A1A2E"
_PAL_GRID = "#2A2A4A"
_PAL_TEXT = "#E0E0E0"
_MODEL_COLORS = {
    "CLEAN":    "#4FC3F7",
    "NOISY":    "#EF9A9A",
    "WEIGHTED": "#CE93D8",
}
_CLEAN_PALETTE = ["#A5D6A7", "#66BB6A", "#2E7D32"]


def _section(title):
    print("\n" + "=" * 66)
    print(f"  {title}")
    print("=" * 66)


def _set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _history_dict(logger):
    return {
        "train_loss": [e["train_loss"] for e in logger.history],
        "val_loss":   [e["val_loss"]   for e in logger.history],
        "train_acc":  [e["train_acc"]  for e in logger.history],
        "val_acc":    [e["val_acc"]    for e in logger.history],
    }


def _save_uncertainty_audit(mc_result, output_path):
    rows = []
    unc  = mc_result.variance.max(axis=1)
    for idx, sid in enumerate(mc_result.sample_ids):
        rows.append({
            "sample_id":   int(sid),
            "file_path":   mc_result.file_paths[idx],
            "true_label":  int(mc_result.true_labels[idx]),
            "prediction":  int(mc_result.pred_classes[idx]),
            "confidence":  float(mc_result.confidence[idx]),
            "uncertainty": float(unc[idx]),
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)


def _print_threshold_comparison(results, experiment_meta):
    base_keys = ["CLEAN", "NOISY", "WEIGHTED"]
    ordered   = base_keys + [k for k in results if k not in base_keys]
    metrics   = ["accuracy", "precision", "recall", "f1"]
    labels    = {"accuracy": "Accuracy", "precision": "Precision",
                 "recall": "Recall", "f1": "F1-Score"}
    col_w = 11; name_w = 14
    header = (
        f"\n  {'Model':<{name_w}}"
        + "".join(f"  {labels[m]:>{col_w}}" for m in metrics)
        + f"  {'Delta F1 vs Noisy':>19}  {'Removed':>8}  {'Train N':>8}"
    )
    sep = "  " + "-" * (len(header) - 2)
    print("\n" + "=" * 66)
    print("  Threshold Cleaning Comparison -- Test Set")
    print("=" * 66)
    print(header)
    print(sep)
    noisy_f1 = results["NOISY"]["f1"] * 100
    for name in ordered:
        if name not in results:
            continue
        r   = results[name]
        row = f"  {name:<{name_w}}"
        for m in metrics:
            row += f"  {r[m]*100:>{col_w}.2f}%"
        if name in ("CLEAN", "NOISY"):
            delta_str = "  " + " " * 19
        else:
            delta     = r["f1"] * 100 - noisy_f1
            delta_str = f"  {delta:>+18.2f}pp"
        meta      = experiment_meta.get(name, {})
        n_removed = meta.get("n_removed", "--")
        n_train   = meta.get("n_train",   "--")
        row += delta_str + f"  {str(n_removed):>8}  {str(n_train):>8}"
        print(row)
    print(sep)
    print()


def _color_for(key, exp_keys):
    if key in _MODEL_COLORS:
        return _MODEL_COLORS[key]
    idx = exp_keys.index(key) if key in exp_keys else 0
    return _CLEAN_PALETTE[idx % len(_CLEAN_PALETTE)]


def _plot_threshold_f1(results, experiment_meta, save_path):
    base_keys = ["CLEAN", "NOISY", "WEIGHTED"]
    ordered   = [k for k in base_keys + [k for k in results if k not in base_keys] if k in results]
    exp_keys  = [k for k in ordered if k not in base_keys]
    f1_vals   = [results[k]["f1"] * 100 for k in ordered]
    colors    = [_color_for(k, exp_keys) for k in ordered]

    fig, ax = plt.subplots(figsize=(11, 5), facecolor=_PAL_BG)
    ax.set_facecolor("#16213E")
    bars = ax.bar(ordered, f1_vals, color=colors, edgecolor=_PAL_BG, linewidth=0.8, zorder=3)
    for bar, val in zip(bars, f1_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.2f}%", ha="center", va="bottom",
                fontsize=8.5, color=_PAL_TEXT, fontweight="bold")
    ax.set_ylabel("F1-Score (%)", color=_PAL_TEXT)
    ax.set_ylim(max(0, min(f1_vals) - 3), min(100, max(f1_vals) + 3))
    ax.set_xlabel("Model", color=_PAL_TEXT)
    ax.tick_params(colors=_PAL_TEXT, labelsize=9)
    ax.grid(axis="y", color=_PAL_GRID, linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    noisy_f1 = results["NOISY"]["f1"] * 100
    ax.axhline(noisy_f1, color=_MODEL_COLORS["NOISY"], linewidth=1.4,
               linestyle="--", alpha=0.6, label=f"Noisy baseline ({noisy_f1:.2f}%)")
    ax.legend(fontsize=8, facecolor="#0F3460", labelcolor=_PAL_TEXT, edgecolor=_PAL_GRID)
    fig.suptitle("F1-Score vs Cleaning Strategy", fontsize=13,
                 fontweight="bold", color="#F4D03F", y=0.98)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_PAL_BG)
    plt.close(fig)
    print(f"  Saved plot -> {save_path.name}")


def _plot_multi_metrics(results, save_path):
    base_keys = ["CLEAN", "NOISY", "WEIGHTED"]
    ordered   = [k for k in base_keys + [k for k in results if k not in base_keys] if k in results]
    exp_keys  = [k for k in ordered if k not in base_keys]
    metric_keys   = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision\n(macro)", "Recall\n(macro)", "F1-Score\n(macro)"]
    colors = [_color_for(k, exp_keys) for k in ordered]
    width  = 0.55

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor=_PAL_BG)
    fig.suptitle("Multi-Strategy Cleaning -- Test Set Metrics", fontsize=13,
                 fontweight="bold", color="#F4D03F", y=1.00)
    for ax, mk, ml in zip(axes, metric_keys, metric_labels):
        vals = [results[k][mk] * 100 for k in ordered]
        bars = ax.bar(ordered, vals, color=colors, edgecolor=_PAL_BG,
                      linewidth=0.6, width=width, zorder=3)
        ax.set_facecolor("#16213E")
        ax.set_title(ml, color=_PAL_TEXT, fontsize=10, pad=6)
        ax.set_ylim(max(0, min(vals) - 3), min(100, max(vals) + 3))
        ax.tick_params(colors=_PAL_TEXT, labelsize=7.5)
        ax.set_xticklabels(ordered, rotation=30, ha="right", fontsize=7)
        ax.grid(axis="y", color=_PAL_GRID, linewidth=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=7, color=_PAL_TEXT)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_PAL_BG)
    plt.close(fig)
    print(f"  Saved plot -> {save_path.name}")


def _plot_val_accuracy_curves(histories, save_path):
    fig, ax   = plt.subplots(figsize=(11, 5), facecolor=_PAL_BG)
    ax.set_facecolor("#16213E")
    exp_keys  = [k for k in histories if k not in _MODEL_COLORS]
    all_colors = dict(_MODEL_COLORS)
    for i, k in enumerate(exp_keys):
        all_colors[k] = _CLEAN_PALETTE[i % len(_CLEAN_PALETTE)]
    for name, hist in histories.items():
        epochs = range(1, len(hist["val_acc"]) + 1)
        ax.plot(epochs, hist["val_acc"], label=name,
                color=all_colors.get(name, "#CCCCCC"),
                linewidth=2, linestyle="--" if name == "NOISY" else "-")
    ax.set_xlabel("Epoch", color=_PAL_TEXT)
    ax.set_ylabel("Validation Accuracy (%)", color=_PAL_TEXT)
    ax.tick_params(colors=_PAL_TEXT)
    ax.grid(color=_PAL_GRID, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, facecolor="#0F3460", labelcolor=_PAL_TEXT, edgecolor=_PAL_GRID)
    fig.suptitle("Validation Accuracy -- All Models", fontsize=13,
                 fontweight="bold", color="#F4D03F", y=0.99)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_PAL_BG)
    plt.close(fig)
    print(f"  Saved plot -> {save_path.name}")


def _save_threshold_csv(results, meta, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    base_keys = ["CLEAN", "NOISY", "WEIGHTED"]
    ordered   = base_keys + [k for k in results if k not in base_keys]
    metrics   = ["accuracy", "precision", "recall", "f1"]
    with path.open("w", encoding="utf-8") as fh:
        fh.write("model,n_removed,n_train," + ",".join(metrics) + "\n")
        for name in ordered:
            if name not in results:
                continue
            r = results[name]; m = meta.get(name, {})
            row = (f"{name},{m.get('n_removed','?')},{m.get('n_train','?')},"
                   + ",".join(f"{r[k]:.6f}" for k in metrics))
            fh.write(row + "\n")
    print(f"  Saved CSV   -> {path.name}")


def main():
    _set_seeds(RANDOM_SEED)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_ok = torch.cuda.is_available()
    print(f"\n  Device       : {device}")
    print(f"  CUDA         : {cuda_ok}")
    if cuda_ok:
        print(f"  GPU name     : {torch.cuda.get_device_name(0)}")
    print(f"  Seed         : {RANDOM_SEED}")
    print(f"  Backbone     : {BACKBONE}")

    for d in (ARTIFACTS_DIR, HISTORY_DIR, METRICS_DIR, UNCERTAINTY_DIR, PLOTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # Stage 1 -- Data
    _section("Stage 1 - Data Loading & Splitting")
    all_samples = collect_all_samples(DATASET_ROOT)
    train_samples, val_samples, test_samples = stratified_split(all_samples)
    clean_train_loader, _, val_loader, test_loader = build_dataloaders(
        train_samples, val_samples, test_samples)
    print(f"  Train : {len(train_samples):,}  |  Val : {len(val_samples):,}  |  Test : {len(test_samples):,}")

    # Stage 2 -- Clean model
    _section("Stage 2 - Train Model 1 (Clean Dataset)")
    model_clean, _ = build_model(pretrained=True, backbone=BACKBONE)
    model_clean.to(device)
    logger_clean = run_training(
        model_clean, clean_train_loader, val_loader, device,
        save_path=CKPT_CLEAN,
        history_path=HISTORY_DIR / "clean_history.json",
        num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, patience=EARLY_STOP_PAT,
    )

    # Stage 3 -- Noise injection
    _section(f"Stage 3 - Label Noise Injection ({NOISE_RATE * 100:.0f}%)")
    (noisy_train_loader, noisy_analysis_loader,
     val_loader_n, test_loader_n,
     noisy_samples, noisy_sample_ids) = build_noisy_dataloaders(
        train_samples, val_samples, test_samples, noise_rate=NOISE_RATE)
    known_noisy_ids = set(noisy_sample_ids)
    print(f"  Injected noise into {len(noisy_sample_ids):,} training labels.")

    # Stage 4 -- Noisy model
    _section("Stage 4 - Train Model 2 (Noisy Dataset)")
    model_noisy, _ = build_model(pretrained=True, backbone=BACKBONE)
    model_noisy.to(device)
    logger_noisy = run_training(
        model_noisy, noisy_train_loader, val_loader_n, device,
        save_path=CKPT_NOISY,
        history_path=HISTORY_DIR / "noisy_history.json",
        num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, patience=EARLY_STOP_PAT,
    )

    # Stage 5 -- MC Dropout
    _section(f"Stage 5 - MC Dropout Uncertainty (T={MC_PASSES} passes)")
    mc_result = mc_dropout_predict(
        model_noisy, noisy_analysis_loader, device, n_passes=MC_PASSES)
    _save_uncertainty_audit(mc_result, UNCERTAINTY_DIR / "mc_dropout_audit.json")
    print(f"  mean_probs shape : {mc_result.mean_probs.shape}")

    # Stage 6 -- Detection
    _section("Stage 6 - Error Detection")
    flagged = detect_errors(
        mc_result,
        uncertainty_thresh=UNCERTAINTY_THRESH,
        confidence_thresh=CONFIDENCE_THRESH,
        known_noisy_ids=known_noisy_ids,
    )
    print_detection_report(flagged, len(noisy_samples), known_noisy_ids)

    # Stages 7-8 -- Threshold cleaning
    _section(f"Stages 7-8 -- Threshold Cleaning ({[str(int(f*100))+'%' for f in CLEANING_LEVELS]})")
    print(f"  Flagged pool : {len(flagged):,} samples")
    flagged_ranked = sorted(flagged, key=lambda s: s.uncertainty, reverse=True)
    experiment_registry = {}

    for fraction in CLEANING_LEVELS:
        n_remove    = max(1, int(round(len(flagged_ranked) * fraction)))
        removed_ids = {s.sample_id for s in flagged_ranked[:n_remove]}
        clean_samp  = [s for s in noisy_samples if s.sample_id not in removed_ids]
        exp_label   = f"CLEANED_{int(fraction * 100):02d}pct"

        _section(f"  Experiment: {exp_label}")
        print(f"  Fraction removed : {fraction*100:.0f}% of flagged pool ({n_remove:,} samples)")
        print(f"  Training set     : {len(noisy_samples):,} -> {len(clean_samp):,}")

        exp_train_ld, _, exp_val_ld, exp_test_ld = build_cleaned_loader(
            clean_samp, val_samples, test_samples)
        model_exp, _ = build_model(pretrained=True, backbone=BACKBONE)
        model_exp.to(device)
        logger_exp = run_training(
            model_exp, exp_train_ld, exp_val_ld, device,
            save_path=BASE_DIR / f"best_model_{exp_label.lower()}.pth",
            history_path=HISTORY_DIR / f"{exp_label.lower()}_history.json",
            num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, patience=EARLY_STOP_PAT,
        )
        experiment_registry[exp_label] = {
            "model": model_exp, "test_loader": exp_test_ld, "logger": logger_exp,
            "n_removed": n_remove, "n_train": len(clean_samp), "fraction": fraction,
        }

    # Stage 8W -- Weighted training
    _section("Stage 8W - Confidence-Weighted Loss Training (no sample removal)")
    print(f"  Weight floor : {WEIGHT_FLOOR}   |   Dataset size : {len(noisy_samples):,} (retained)")
    weight_map      = assign_sample_weights(noisy_samples, flagged, weight_floor=WEIGHT_FLOOR)
    weighted_loader = build_weighted_training_loader(noisy_samples, weight_map)
    model_weighted, _ = build_model(pretrained=True, backbone=BACKBONE)
    model_weighted.to(device)
    logger_weighted = run_training_weighted(
        model_weighted, weighted_loader, val_loader_n, device,
        save_path=BASE_DIR / "best_model_weighted.pth",
        history_path=HISTORY_DIR / "weighted_history.json",
        num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, patience=EARLY_STOP_PAT,
    )

    # Stage 9 -- Evaluation
    _section("Stage 9 - Test Set Evaluation (All Models)")
    results         = {}
    roc_proba_data  = {}
    experiment_meta = {
        "CLEAN":    {"n_removed": 0,                    "n_train": len(train_samples)},
        "NOISY":    {"n_removed": len(known_noisy_ids), "n_train": len(noisy_samples)},
        "WEIGHTED": {"n_removed": 0,                    "n_train": len(noisy_samples)},
    }
    eval_plan = [
        ("CLEAN",    model_clean,    test_loader),
        ("NOISY",    model_noisy,    test_loader_n),
        ("WEIGHTED", model_weighted, test_loader_n),
    ]
    for label, info in experiment_registry.items():
        eval_plan.append((label, info["model"], info["test_loader"]))
        experiment_meta[label] = {"n_removed": info["n_removed"], "n_train": info["n_train"]}

    for name, model, loader in eval_plan:
        preds, labels_true = run_inference(model, loader, device)
        results[name] = compute_metrics(labels_true, preds)
        print(f"\n  -> {name}")
        print_metrics(results[name], split_name=name)
        probs, _ = run_inference_proba(model, loader, device)
        roc_proba_data[name] = (probs, labels_true)

    _print_threshold_comparison(results, experiment_meta)
    save_metrics(results, METRICS_DIR / "final_metrics.json")
    save_confusion_matrices(results, METRICS_DIR / "confusion_matrices.json")
    _save_threshold_csv(results, experiment_meta, METRICS_DIR / "threshold_comparison.csv")

    # Stage 10 -- Visualisations
    _section("Stage 10 - Visualisations")
    histories = {
        "CLEAN":    _history_dict(logger_clean),
        "NOISY":    _history_dict(logger_noisy),
        "WEIGHTED": _history_dict(logger_weighted),
    }
    for label, info in experiment_registry.items():
        histories[label] = _history_dict(info["logger"])

    if all(histories[k]["train_loss"] for k in histories):
        unc_scores = mc_result.variance.max(axis=1)
        plot_uncertainty_histogram(
            {"NORMAL":    unc_scores[mc_result.true_labels == 0].tolist(),
             "PNEUMONIA": unc_scores[mc_result.true_labels == 1].tolist()},
            save_path=PLOTS_DIR / "plot3_uncertainty_histogram.png",
        )
        _plot_threshold_f1(results, experiment_meta, PLOTS_DIR / "plot4_threshold_f1.png")
        _plot_multi_metrics(results, PLOTS_DIR / "plot5_multi_metrics.png")
        _plot_val_accuracy_curves(histories, PLOTS_DIR / "plot6_val_accuracy_curves.png")
        plot_roc_pr_curves(roc_proba_data, save_path=PLOTS_DIR / "plot7_roc_pr_curves.png")
        combined_scores = compute_combined_score(mc_result, entropy_alpha=ENTROPY_ALPHA)
        plot_uncertainty_scatter(
            combined_scores=combined_scores,
            true_labels=mc_result.true_labels,
            pred_labels=mc_result.pred_classes,
            known_noisy_ids=known_noisy_ids,
            sample_ids=mc_result.sample_ids,
            save_path=PLOTS_DIR / "plot8_uncertainty_scatter.png",
        )
        plt.close("all")
        print(f"  All plots saved to: {PLOTS_DIR}")
    else:
        print("  Skipping plots -- training history unavailable.")

    # Paper summary
    _section("Paper Summary")
    flagged_ids = {s.sample_id for s in flagged}
    true_pos    = flagged_ids & known_noisy_ids
    generate_paper_summary(
        results,
        {
            "total_train": len(noisy_samples),
            "n_flagged":   len(flagged),
            "precision":   len(true_pos) / len(flagged_ids)     if flagged_ids     else 0.0,
            "recall":      len(true_pos) / len(known_noisy_ids) if known_noisy_ids else 0.0,
        },
        output_path=ARTIFACTS_DIR / "paper_summary.txt",
    )

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
