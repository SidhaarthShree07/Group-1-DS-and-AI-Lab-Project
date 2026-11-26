#!/usr/bin/env python3
"""Generate Milestone 5 Evaluation Visualisations from Real Run Files.

Author: Sahilpreet
Role: Evaluation, Calibration & Visualisation Lead

Generates:
1. docs/figures/m5/confusion_matrices_test.png:
   Detection confusion matrices computed directly from real test predictions.
2. docs/figures/m5/precision_recall_curves.png:
   Precision-Recall curves among candidate cells computed using sklearn.metrics.precision_recall_curve
   on real P(error) values and ground truth labels.
3. docs/figures/m5/precision_coverage_curve.png:
   Precision-Coverage curve computed by sweeping confidence thresholds over real
   repair proposals.
4. docs/figures/m5/reliability_diagrams_m5.png:
   10-bin Expected Calibration Error diagrams computed on real candidate probabilities
   (raw logits vs temperature scaled T=0.826).

Usage:
  MPLCONFIGDIR=/tmp/mpl python scripts/generate_m5_plots.py [--runs <run_folders>...]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score

from src.evaluation.calibrate import apply_temperature
from src.evaluation.metrics import calculate_ece
from src.profiling.context import norm


def get_raha_excluded_rows(table: str, total_rows: int, budget: int = 20, seed: int = 42) -> Set[int]:
    """Return the set of 20 row indices excluded for Raha's labeled tuples under Rule D8."""
    rng = random.Random(f"{seed}:{table.lower()}")
    return set(rng.sample(range(total_rows), min(budget, total_rows)))


def load_runs_data(run_dirs: List[Path]) -> Dict[str, Dict[str, Any]]:
    """Load predictions and cells from run folders, excluding Raha's 20 labelled rows per table."""
    tables_data: Dict[str, Dict[str, Any]] = {}

    for rd in run_dirs:
        pred_path = rd / "predictions.jsonl"
        cells_path = rd / "cells.csv"
        if not pred_path.exists() or not cells_path.exists():
            continue

        # Load cell truth
        raw_cell_gt: Dict[str, Dict[Tuple[str, str], Tuple[str, str]]] = {}
        with open(cells_path, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if str(row.get("error_type", "")).strip() == "format_artifact":
                    continue
                t = str(row.get("table", "")).strip().lower()
                r = str(row.get("row", "")).strip()
                c = str(row.get("column", "")).strip()
                raw_cell_gt.setdefault(t, {})[(r, c)] = (str(row.get("dirty_value", "")), str(row.get("clean_value", "")))

        # Compute excluded rows per table (leave out Raha's 20 labelled rows only for benchmark tables)
        benchmark_tables = {"hospital", "flights", "beers", "rayyan", "movies"}
        excluded_by_table: Dict[str, Set[int]] = {}
        for t, cells in raw_cell_gt.items():
            if t in benchmark_tables:
                n_rows = len(set(int(r) for r, _ in cells.keys()))
                excluded_by_table[t] = get_raha_excluded_rows(t, n_rows, budget=20, seed=42)
            else:
                excluded_by_table[t] = set()

        # Load predictions
        with open(pred_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                p = json.loads(line)
                p_raw = p.get("p_error")
                if p_raw is None or not isinstance(p_raw, (int, float)):
                    continue
                p_raw = float(p_raw)
                if math.isnan(p_raw) or math.isinf(p_raw) or not (0.0 <= p_raw <= 1.0):
                    continue

                t = str(p.get("table", "")).strip().lower()
                r = str(p.get("row", "")).strip()
                c = str(p.get("column", "")).strip()

                # Leave out Raha's 20 labelled rows
                if int(r) in excluded_by_table.get(t, set()):
                    continue

                gt = raw_cell_gt.get(t, {}).get((r, c))
                if gt is None:
                    continue
                dirty_val, clean_val = gt
                is_err = norm(dirty_val) != norm(clean_val)

                if t not in tables_data:
                    tables_data[t] = {
                        "probs": [],
                        "labels": [],
                        "predictions": [],
                    }

                tables_data[t]["probs"].append(p_raw)
                tables_data[t]["labels"].append(1 if is_err else 0)
                tables_data[t]["predictions"].append({
                    "p_error": p_raw,
                    "repair": p.get("repair"),
                    "p_repair_min": p.get("p_repair_min"),
                    "clean_val": clean_val,
                    "dirty_val": dirty_val,
                    "is_err": is_err,
                })

    return tables_data


plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Carlito", "Helvetica", "Arial", "DejaVu Sans"],
                     "axes.titleweight": "bold", "figure.facecolor": "white"})
NAVY_CMAP = LinearSegmentedColormap.from_list("contextguard", ["#f4f7fb", "#9db8dd", "#14305c"])

# Theme styling palette (Tailwind-inspired clean design)
COLOR_TITLE = "#14305c"
COLOR_TEXT = "#39424e"
COLOR_MUTED = "#6b7585"
COLOR_GRID = "#e3e8f0"
COLOR_BORDER = "#b8c2d1"

TABLE_PALETTE = {
    # 5 Core Validation Tables (solid lines, distinct rich tones)
    "hospital": {"color": "#1f4a8a", "style": "-", "label": "Hospital [Val]"},
    "flights": {"color": "#d98e04", "style": "-", "label": "Flights [Val]"},
    "beers": {"color": "#0f9d8a", "style": "-", "label": "Beers [Val]"},
    "rayyan": {"color": "#7a4fb5", "style": "-", "label": "Rayyan [Val]"},
    "movies": {"color": "#c0392b", "style": "-", "label": "Movies [Val]"},
    # 4 Out-of-Distribution Unseen Tables (dashed lines, vibrant high-contrast tones)
    "german": {"color": "#4f86c6", "style": "--", "label": "German [Unseen]"},
    "soccer": {"color": "#5bb55b", "style": "--", "label": "Soccer [Unseen]"},
    "billionaire": {"color": "#b46fd0", "style": "--", "label": "Billionaire [Unseen]"},
    "marketing": {"color": "#e8743b", "style": "--", "label": "Marketing [Unseen]"},
    "restaurant": {"color": "#8a6d3b", "style": "--", "label": "Restaurant [Unseen]"},
}


def plot_real_confusion_matrices(tables_data: Dict[str, Dict[str, Any]], out_path: Path, t1: float = 0.50) -> None:
    """Detection confusion matrices among candidate cells: validation tables in the top row, unseen tables below.

    Rows of each matrix: true label (Error, Clean); columns: predicted action (Flagged, Kept).
    Each cell shows the count and its share of the true-label row.
    """
    val_names = ["beers", "flights", "hospital", "movies", "rayyan"]
    val = [t for t in val_names if t in tables_data]
    unseen = sorted(t for t in tables_data if t not in val_names)
    n_cols = max(len(val), len(unseen), 1)

    fig, axes = plt.subplots(2, n_cols, figsize=(3.05 * n_cols + 0.9, 8.3))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.835, bottom=0.115, wspace=0.34, hspace=0.62)
    axes = np.atleast_2d(axes)

    for r_idx, names in enumerate([val, unseen]):
        for c_idx in range(n_cols):
            ax = axes[r_idx, c_idx]
            if c_idx >= len(names):
                ax.axis("off")
                continue
            name = names[c_idx]
            data = tables_data[name]
            y_true = np.array(data["labels"])
            y_pred = (np.array(data["probs"]) >= t1).astype(int)
            cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
            tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
            mat = np.array([[tp, fn], [fp, tn]])
            row_share = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1)
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            f1 = 2 * prec * rec / max(1e-9, prec + rec)

            ax.imshow(row_share, cmap=NAVY_CMAP, vmin=0, vmax=1, interpolation="nearest", aspect="equal")
            for i in range(2):
                for j in range(2):
                    colour = "white" if row_share[i, j] > 0.55 else COLOR_TITLE
                    lab = [["TP", "FN"], ["FP", "TN"]][i][j]
                    ax.text(j, i - 0.10, f"{mat[i, j]:,}", ha="center", va="center", color=colour, fontsize=12.5, fontweight="bold")
                    ax.text(j, i + 0.20, f"{lab}  {row_share[i, j] * 100:.0f}%", ha="center", va="center", color=colour, fontsize=8.6)
            ax.set_title(f"{name.capitalize()}\nF1 {f1:.3f}  |  {len(y_true):,} cells", fontsize=10.5, fontweight="bold",
                         color=COLOR_TITLE, pad=9, linespacing=1.35)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Flagged", "Kept"], fontsize=9, color=COLOR_TEXT)
            ax.set_yticklabels(["Error", "Clean"], fontsize=9, color=COLOR_TEXT)
            ax.tick_params(length=0, pad=5)
            if c_idx == 0:
                ax.set_ylabel("True label", fontsize=9, color=COLOR_MUTED, labelpad=8)
            ax.set_xlabel("Predicted action", fontsize=9, color=COLOR_MUTED, labelpad=7)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks([0.5], minor=True)
            ax.set_yticks([0.5], minor=True)
            ax.grid(which="minor", color="white", linewidth=3)
            ax.tick_params(which="minor", length=0)

    for r_idx, label in enumerate(["Core validation tables", "Unseen tables"]):
        pos = axes[r_idx, 0].get_position()
        fig.text(0.075, pos.y1 + 0.082, label, fontsize=11.5, fontweight="bold", color="#1f4a8a")
    fig.suptitle(f"ContextGuard: detection confusion matrices among candidate cells (flag threshold {t1:.2f})",
                 fontsize=13.5, fontweight="bold", color=COLOR_TITLE, y=0.978)
    fig.text(0.075, 0.022, "Share shown is the fraction of the true-label row. Candidate cells are the cells the detectors proposed, not every cell of the table.",
             fontsize=8.6, color=COLOR_MUTED)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"[SUCCESS] Wrote real confusion matrices -> {out_path}")


def plot_real_pr_curves(tables_data: Dict[str, Dict[str, Any]], out_path: Path) -> None:
    """Plot real Precision-Recall curves among candidate cells using distinct colors for all 9 tables."""
    fig, ax = plt.subplots(figsize=(11.2, 6.2))

    # Sort so Core Validation comes first, then Unseen
    sorted_tables = sorted(tables_data.keys(), key=lambda t: (
        0 if t in {"hospital", "flights", "beers", "rayyan", "movies"} else 1, t
    ))

    skipped = []
    for tbl_name in sorted_tables:
        data = tables_data[tbl_name]
        y_true = np.array(data["labels"])
        y_score = np.array(data["probs"])
        if y_true.sum() < 20:
            skipped.append((tbl_name, int(y_true.sum())))
            continue

        prec, rec, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)

        cfg = TABLE_PALETTE.get(tbl_name, {"color": "#2563eb", "style": "-", "label": tbl_name.capitalize()})
        color = cfg["color"]
        style = cfg["style"]
        label = f"{cfg['label']} (AP = {ap:.3f})"

        ax.plot(rec, prec, label=label, color=color, linestyle=style, linewidth=2.3)

    if skipped:
        note = "; ".join(f"{n.capitalize()} not drawn: only {k} error cells among its candidates" for n, k in skipped)
        fig.text(0.07, 0.01, note, fontsize=8.6, color=COLOR_MUTED)
    ax.set_title("ContextGuard: Precision-Recall Curves (among candidate cells)",
                 fontsize=12.5, fontweight="bold", color=COLOR_TITLE, pad=12)
    ax.set_xlabel("Recall (True Positive Rate)", fontsize=10.5, fontweight="medium", color=COLOR_TEXT)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=10.5, fontweight="medium", color=COLOR_TEXT)
    ax.set_xlim([0.0, 1.01])
    ax.set_ylim([0.25, 1.015])
    ax.grid(True, linestyle="--", alpha=0.5, color=COLOR_GRID)

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9.2, frameon=False, labelspacing=0.9, handlelength=2.6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(COLOR_BORDER)
        ax.spines[spine].set_linewidth(1.0)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=250)
    plt.close()
    print(f"[SUCCESS] Wrote real PR curves (among candidate cells) -> {out_path}")


def plot_real_precision_coverage(tables_data: Dict[str, Dict[str, Any]], out_path: Path) -> None:
    """Plot empirical Precision-Coverage curve showing the full trajectory across all coverage levels."""
    fig, ax = plt.subplots(figsize=(8.5, 5.8))

    all_repairs = []
    for data in tables_data.values():
        for p in data["predictions"]:
            rep = p.get("repair")
            if rep is not None and str(rep).strip() != "":
                p_min = p.get("p_repair_min")
                try:
                    p_min = float(p_min) if p_min is not None else 0.0
                except (ValueError, TypeError):
                    p_min = 0.0
                is_correct_rep = norm(str(rep)) == norm(p["clean_val"])
                all_repairs.append((p_min, is_correct_rep))

    if all_repairs:
        # Sweep threshold from 0.99 down to 0.00 to show complete curve to 100% coverage
        thresholds = np.linspace(0.99, 0.00, 100)
        coverages = []
        precisions = []

        total_proposals = len(all_repairs)
        for t in thresholds:
            accepted = [r for r in all_repairs if r[0] >= t]
            if not accepted:
                continue
            cov = len(accepted) / total_proposals * 100.0
            prec = sum(r[1] for r in accepted) / len(accepted) * 100.0
            coverages.append(cov)
            precisions.append(prec)

        # Plot curve and soft area
        ax.plot(coverages, precisions, color="#2563eb", linewidth=2.6, label="Empirical Precision vs Coverage")
        ax.fill_between(coverages, precisions, 55, color="#3b82f6", alpha=0.08)

        # Annotate operating point at t2 = 0.900
        idx_opt = int(np.argmin(np.abs(thresholds - 0.90)))
        cov_opt = coverages[idx_opt]
        prec_opt = precisions[idx_opt]
        ax.scatter([cov_opt], [prec_opt], color="#dc2626", s=50, zorder=5)
        ax.annotate(f"Production Gate (t2 = 0.900)\nCov = {cov_opt:.1f}%, Prec = {prec_opt:.1f}%",
                    xy=(cov_opt, prec_opt), xytext=(cov_opt + 7, prec_opt - 8),
                    arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.2),
                    fontsize=8.5, fontweight="bold", color="#991b1b",
                    bbox=dict(boxstyle="round,pad=0.4", fc="#fef2f2", ec="#fca5a5", lw=1.0))

    ax.axhline(98.0, color="#dc2626", linestyle=":", alpha=0.85, linewidth=1.6,
               label="2.0% Max Tolerable FRR Safety Cap (Prec ≥ 98.0%)")

    ax.set_title("Milestone 1 Commitment: Empirical Precision-Coverage Curve",
                 fontsize=12.5, fontweight="bold", color=COLOR_TITLE, pad=12)
    ax.set_xlabel("Coverage (% of Candidate Repairs Executed)", fontsize=10.5, fontweight="medium", color=COLOR_TEXT)
    ax.set_ylabel("Applied Repair Precision (%)", fontsize=10.5, fontweight="medium", color=COLOR_TEXT)

    # Full visibility of curve: from 55% up to 102%
    ax.set_xlim([0, 102])
    ax.set_ylim([58, 102])
    ax.grid(True, linestyle="--", alpha=0.5, color=COLOR_GRID)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.96, edgecolor=COLOR_BORDER)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(COLOR_BORDER)
        ax.spines[spine].set_linewidth(1.0)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=250)
    plt.close()
    print(f"[SUCCESS] Wrote real precision-coverage curve -> {out_path}")


def _compute_calibration_bins(probs: List[float], labels: List[bool], n_bins: int = 10) -> List[Dict[str, float]]:
    """Compute empirical accuracy and confidence across equal-width bins."""
    bin_step = 1.0 / n_bins
    bins_data = []
    for b in range(n_bins):
        low = b * bin_step
        high = (b + 1) * bin_step
        if b == 0:
            idx = [i for i, p in enumerate(probs) if low <= p <= high]
        else:
            idx = [i for i, p in enumerate(probs) if low < p <= high]
        if idx:
            mean_conf = sum(probs[i] for i in idx) / len(idx)
            acc = sum(1 for i in idx if labels[i]) / len(idx)
            bins_data.append({"confidence": mean_conf, "accuracy": acc, "count": len(idx)})
    return bins_data


def plot_real_reliability_diagrams(tables_data: Dict[str, Dict[str, Any]], out_path: Path, temp: float = 0.826) -> None:
    """Plot empirical 10-bin reliability diagrams across pooled candidate predictions."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=True)

    all_raw_probs: List[float] = []
    all_temp_probs: List[float] = []
    all_labels: List[bool] = []

    for data in tables_data.values():
        all_raw_probs.extend(data["probs"])
        all_labels.extend([bool(x) for x in data["labels"]])
        all_temp_probs.extend(apply_temperature(data["probs"], temp))

    if all_raw_probs:
        ece_raw = calculate_ece(all_labels, all_raw_probs, n_bins=10)
        ece_temp = calculate_ece(all_labels, all_temp_probs, n_bins=10)

        # Plot 1: Raw
        raw_bins = _compute_calibration_bins(all_raw_probs, all_labels, n_bins=10)
        axes[0].bar([b["confidence"] for b in raw_bins if b["count"] > 0],
                    [b["accuracy"] for b in raw_bins if b["count"] > 0],
                    width=0.08, alpha=0.75, color="#1f4a8a", edgecolor="#14305c", linewidth=1.2,
                    label="Empirical Accuracy")
        axes[0].plot([0, 1], [0, 1], color="#c0392b", linestyle="--", linewidth=1.6, label="Perfect Calibration")
        axes[0].set_title(f"1. Uncalibrated Raw Logits\nECE = {ece_raw:.4f}",
                          fontsize=11, fontweight="bold", color=COLOR_TITLE)
        axes[0].set_xlabel("Mean Predicted Confidence", fontsize=9.5, fontweight="medium", color=COLOR_TEXT)
        axes[0].set_ylabel("Empirical Accuracy", fontsize=9.5, fontweight="medium", color=COLOR_TEXT)
        axes[0].set_xlim([0, 1])
        axes[0].set_ylim([0, 1])
        axes[0].grid(True, linestyle="--", alpha=0.5, color=COLOR_GRID)
        axes[0].legend(loc="upper left", fontsize=8.5, framealpha=0.96, edgecolor=COLOR_BORDER)

        # Plot 2: Temperature Scaled
        temp_bins = _compute_calibration_bins(all_temp_probs, all_labels, n_bins=10)
        axes[1].bar([b["confidence"] for b in temp_bins if b["count"] > 0],
                    [b["accuracy"] for b in temp_bins if b["count"] > 0],
                    width=0.08, alpha=0.75, color="#0f9d8a", edgecolor="#0a7365", linewidth=1.2,
                    label="Empirical Accuracy")
        axes[1].plot([0, 1], [0, 1], color="#c0392b", linestyle="--", linewidth=1.6, label="Perfect Calibration")
        axes[1].set_title(f"2. Temperature Scaled (T = {temp})\nECE = {ece_temp:.4f} (T fitted on core validation)",
                          fontsize=11, fontweight="bold", color=COLOR_TITLE)
        axes[1].set_xlabel("Mean Predicted Confidence", fontsize=9.5, fontweight="medium", color=COLOR_TEXT)
        axes[1].set_xlim([0, 1])
        axes[1].set_ylim([0, 1])
        axes[1].grid(True, linestyle="--", alpha=0.5, color=COLOR_GRID)
        axes[1].legend(loc="upper left", fontsize=8.5, framealpha=0.96, edgecolor=COLOR_BORDER)

        for ax in axes:
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            for spine in ["left", "bottom"]:
                ax.spines[spine].set_color(COLOR_BORDER)
                ax.spines[spine].set_linewidth(1.0)

    n_total = len(all_raw_probs)
    plt.suptitle(f"ContextGuard: Empirical Probability Calibration (10 Bins)\nPooled over {len(tables_data)} tables (5 Core Validation + {len(tables_data) - 5} Unseen, N = {n_total:,} candidates)",
                 fontsize=12.5, fontweight="bold", color=COLOR_TITLE, y=1.02)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Wrote real reliability diagrams -> {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", default=[], help="Paths to run folders containing predictions.jsonl and cells.csv.")
    args = parser.parse_args()

    fig_dir = _REPO_ROOT / "docs" / "figures" / "m5"

    run_paths = [Path(p) for p in args.runs] if args.runs else [
        _REPO_ROOT / "M4_run_outputs" / "m4_final2_test_hfb",
        _REPO_ROOT / "M4_run_outputs" / "m4_final_test_rayyan",
        _REPO_ROOT / "M4_run_outputs" / "m4_final_test_movies",
    ]

    existing_runs = [p for p in run_paths if p.exists() and (p / "predictions.jsonl").exists()]

    if not existing_runs:
        print("[ERROR] No run folders found on disk with predictions.jsonl.")
        return 1

    print(f"[*] Loading prediction records from {len(existing_runs)} run directories (excluding 20 labelled rows per table)...")
    tables_data = load_runs_data(existing_runs)
    if not tables_data:
        print("[ERROR] No valid predictions could be parsed from provided run folders.")
        return 1

    print(f"[*] Found prediction data for {len(tables_data)} tables: {list(tables_data.keys())}")
    plot_real_confusion_matrices(tables_data, fig_dir / "confusion_matrices_test.png")
    plot_real_pr_curves(tables_data, fig_dir / "precision_recall_curves.png")
    plot_real_precision_coverage(tables_data, fig_dir / "precision_coverage_curve.png")
    plot_real_reliability_diagrams(tables_data, fig_dir / "reliability_diagrams_m5.png")

    print("[ALL DONE] Generated empirical report figures in docs/figures/m5/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
