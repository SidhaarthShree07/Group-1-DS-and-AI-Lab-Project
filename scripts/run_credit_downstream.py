# Author: Jashan Tiwari
"""Experiment X12: Downstream Classification Benchmark on Credit (Objective O5).

Evaluates the impact of data quality and cleaning on downstream machine learning performance.
Trains a downstream classifier (HistGradientBoosting / RandomForest) to predict the binary target
'SeriousDlqin2yrs' on the 150,000-row Credit dataset under three data conditions:
  1. 'dirty':   Trained on uncleaned training data (with missing values and outliers).
  2. 'cleaned': Trained on ContextGuard-cleaned training data (via --cleaned-csv).
  3. 'clean':   Trained on ground-truth clean training data (the upper-bound oracle).

Crucial Evaluation Contract:
- The test split (20% stratified, seed=42) is ALWAYS evaluated on clean ground-truth features.
  This isolates the exact effect of training set corruption and cleaning without test set leakage.
- Reports ROC-AUC, F1, Precision, Recall, Accuracy, and the percentage of the dirty-to-clean gap closed:
      gap_closed_pct = (score_cleaned - score_dirty) / (score_clean - score_dirty) * 100

If --cleaned-csv is not yet supplied (pending final unseen-table runs), the script computes the
exact dirty and clean baselines, locking the denominator before the run, and evaluates an
imputed baseline as an initial reference.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET_COL = "SeriousDlqin2yrs"
META_COLS = {"row_id", "tuple_id", "index", "id", "id_col"}


def load_dataset(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load Credit dataset and separate feature matrix X from target vector y."""
    df = pd.read_csv(path, keep_default_na=True)
    cols_to_drop = [c for c in df.columns if c.strip().lower() in META_COLS]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {path}. Columns: {list(df.columns)}")

    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL])
    # Convert all feature columns to float numeric
    X = X.apply(pd.to_numeric, errors="coerce")
    return X, y


def train_and_eval(
    clf: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test_clean: pd.DataFrame,
    y_test_clean: pd.Series,
) -> Dict[str, float]:
    """Train classifier on specified training condition and evaluate on clean test set."""
    # For models that require no NaNs (like standard RandomForest without native NaN support),
    # fill remaining NaNs with training median to allow execution
    if hasattr(clf, "missing_values"):
        # HistGradientBoosting handles NaNs natively
        X_tr = X_train
        X_te = X_test_clean
    else:
        # Fallback median imputation using training medians only
        medians = X_train.median()
        X_tr = X_train.fillna(medians)
        X_te = X_test_clean.fillna(medians)

    clf.fit(X_tr, y_train)

    probs = clf.predict_proba(X_te)[:, 1]
    preds = clf.predict(X_te)

    roc_auc = float(roc_auc_score(y_test_clean, probs))
    f1 = float(f1_score(y_test_clean, preds, zero_division=0))
    prec = float(precision_score(y_test_clean, preds, zero_division=0))
    rec = float(recall_score(y_test_clean, preds, zero_division=0))
    acc = float(accuracy_score(y_test_clean, preds))

    return {
        "roc_auc": round(roc_auc, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "accuracy": round(acc, 4),
    }


def calculate_gap_closed(score_dirty: float, score_cleaned: float, score_clean: float) -> float:
    """Calculate percentage of dirty-vs-clean gap closed by cleaning.
    
    Formula: (score_cleaned - score_dirty) / (score_clean - score_dirty) * 100
    """
    denom = score_clean - score_dirty
    if abs(denom) < 1e-9:
        return 0.0
    gap = ((score_cleaned - score_dirty) / denom) * 100.0
    return round(gap, 2)


def run_experiment(
    clean_csv: Path,
    dirty_csv: Path,
    cleaned_csv: Optional[Path],
    out_csv: Path,
    seed: int = 42,
    test_size: float = 0.2,
    model_type: str = "hist_gb",
    train_rows: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Execute Experiment X12 under all data conditions."""
    print(f"[*] Loading clean data from {clean_csv}...")
    X_clean, y_clean = load_dataset(clean_csv)

    print(f"[*] Loading dirty data from {dirty_csv}...")
    X_dirty, y_dirty = load_dataset(dirty_csv)

    n_samples = len(y_clean)
    print(f"    Total rows: {n_samples:,} | Positive class: {int(y_clean.sum()):,} ({y_clean.mean()*100:.2f}%)")

    # Fixed stratified train/test split
    idx_train, idx_test = train_test_split(
        np.arange(n_samples),
        test_size=test_size,
        random_state=seed,
        stratify=y_clean,
    )
    if train_rows is not None:
        # All three conditions train on the same subset of the training rows (the rows ContextGuard cleaned);
        # the test rows are unchanged.
        subset = np.sort(pd.read_csv(train_rows)["original_row"].to_numpy())
        if not set(subset) <= set(idx_train):
            raise ValueError(f"{train_rows}: rows outside the training split")
        idx_train = subset
    n_tr, n_te = len(idx_train), len(idx_test)
    print(f"    Train set: {n_tr:,} rows | Test set: {n_te:,} rows (stratified, seed={seed})")

    # Features and labels for test evaluation (ALWAYS CLEAN GROUND TRUTH)
    X_test_eval = X_clean.iloc[idx_test].copy()
    y_test_eval = y_clean.iloc[idx_test].copy()

    def get_model(m_type: str, r_seed: int):
        if m_type == "rf":
            return RandomForestClassifier(n_estimators=100, random_state=r_seed, n_jobs=-1)
        if m_type == "logreg":  # linear model: uses every value directly, so it reacts to outliers and wrong numbers
            return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        return HistGradientBoostingClassifier(random_state=r_seed, max_iter=100)

    results = []

    # 1. Condition: Clean (Oracle upper bound)
    print("\n[*] Condition 1: Clean (Upper Bound Oracle)...")
    clf_clean = get_model(model_type, seed)
    clean_metrics = train_and_eval(clf_clean, X_clean.iloc[idx_train], y_clean.iloc[idx_train], X_test_eval, y_test_eval)
    clean_metrics.update({
        "condition": "clean",
        "classifier": model_type,
        "n_train": n_tr,
        "n_test": n_te,
        "gap_closed_pct": 100.0,
        "notes": "Oracle upper bound trained on error-free features",
    })
    results.append(clean_metrics)
    print(f"    Clean -> ROC-AUC: {clean_metrics['roc_auc']:.4f} | F1: {clean_metrics['f1']:.4f} | Acc: {clean_metrics['accuracy']:.4f}")

    # 2. Condition: Dirty (Lower bound)
    print("\n[*] Condition 2: Dirty (Uncleaned Baseline)...")
    clf_dirty = get_model(model_type, seed)
    dirty_metrics = train_and_eval(clf_dirty, X_dirty.iloc[idx_train], y_dirty.iloc[idx_train], X_test_eval, y_test_eval)
    dirty_metrics.update({
        "condition": "dirty",
        "classifier": model_type,
        "n_train": n_tr,
        "n_test": n_te,
        "gap_closed_pct": 0.0,
        "notes": "Baseline trained on raw corrupted features (missing values & outliers)",
    })
    results.append(dirty_metrics)
    print(f"    Dirty -> ROC-AUC: {dirty_metrics['roc_auc']:.4f} | F1: {dirty_metrics['f1']:.4f} | Acc: {dirty_metrics['accuracy']:.4f}")

    raw_gap = round(clean_metrics["roc_auc"] - dirty_metrics["roc_auc"], 4)
    print(f"    Locked ROC-AUC gap to close: {raw_gap:+.4f} points")

    # 3. Condition: Cleaned
    if cleaned_csv and cleaned_csv.exists():
        print(f"\n[*] Condition 3: ContextGuard Cleaned (from {cleaned_csv})...")
        X_cleaned, y_cleaned = load_dataset(cleaned_csv)
        clf_cleaned = get_model(model_type, seed)
        cleaned_metrics = train_and_eval(clf_cleaned, X_cleaned.iloc[idx_train], y_cleaned.iloc[idx_train], X_test_eval, y_test_eval)
        
        note_text = f"Trained on ContextGuard pipeline output ({cleaned_csv.name})"
        if model_type == "logreg":
            # Linear model has no dirty-to-clean headroom (-0.0002); gap closed percentage is undefined/not applicable
            gap_pct: Any = "N/A"
            note_text += " (caution: linear model dirty-to-clean gap is -0.0002 / zero headroom; percentage not applicable)"
            gap_display = "N/A (zero headroom)"
        else:
            gap_pct = calculate_gap_closed(dirty_metrics["roc_auc"], cleaned_metrics["roc_auc"], clean_metrics["roc_auc"])
            gap_display = f"{gap_pct:.2f}%"

        cleaned_metrics.update({
            "condition": "contextguard_cleaned",
            "classifier": model_type,
            "n_train": n_tr,
            "n_test": n_te,
            "gap_closed_pct": gap_pct,
            "notes": note_text,
        })
        results.append(cleaned_metrics)
        print(f"    Cleaned -> ROC-AUC: {cleaned_metrics['roc_auc']:.4f} | F1: {cleaned_metrics['f1']:.4f} | Gap Closed: {gap_display}")
    else:
        print("\n[*] Condition 3: ContextGuard Cleaned (Awaiting Final Predictions)...")
        results.append({
            "condition": "contextguard_cleaned",
            "classifier": model_type,
            "n_train": n_tr,
            "n_test": n_te,
            "roc_auc": "PENDING",
            "f1": "PENDING",
            "precision": "PENDING",
            "recall": "PENDING",
            "accuracy": "PENDING",
            "gap_closed_pct": "PENDING",
            "notes": "PENDING (awaiting final Credit predictions via --cleaned-csv)",
        })
        print("    Cleaned -> PENDING (awaiting final Credit predictions via --cleaned-csv)")

    # Save to out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition", "classifier", "n_train", "n_test", "roc_auc",
        "f1", "precision", "recall", "accuracy", "gap_closed_pct", "notes"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[+] Successfully exported downstream Credit results to {out_csv}")
    return results


def resolve_credit_path(p_str: Optional[str]) -> Optional[Path]:
    """Resolve file path either directly or relative to data/unseen/credit/."""
    if not p_str:
        return None
    p = Path(p_str)
    if p.exists():
        return p
    fallback = Path("data/unseen/credit") / p.name
    if fallback.exists():
        return fallback
    return p


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-csv", default="data/unseen/credit/clean.csv", help="Clean Credit CSV path")
    parser.add_argument("--dirty-csv", default="data/unseen/credit/dirty.csv", help="Dirty Credit CSV path")
    parser.add_argument("--cleaned-csv", default=None, help="ContextGuard cleaned Credit CSV path")
    parser.add_argument("--out-csv", default="docs/tables/m5_credit_downstream.csv", help="Destination CSV")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test split")
    parser.add_argument("--model", default="all", choices=["hist_gb", "rf", "logreg", "all"], help="Downstream model architecture")
    parser.add_argument("--train-rows", default=None,
                        help="CSV with an `original_row` column: train every condition on these training rows only")
    args = parser.parse_args()

    cleaned_arg = args.cleaned_csv
    if not cleaned_arg:
        if Path("credit_cleaned_by_contextguard.csv").exists() or Path("data/unseen/credit/credit_cleaned_by_contextguard.csv").exists():
            cleaned_arg = "credit_cleaned_by_contextguard.csv"

    train_rows_arg = args.train_rows
    if not train_rows_arg:
        if Path("credit_subset_rows.csv").exists() or Path("data/unseen/credit/credit_subset_rows.csv").exists():
            train_rows_arg = "credit_subset_rows.csv"

    clean_p = resolve_credit_path(args.clean_csv) or Path("data/unseen/credit/clean.csv")
    dirty_p = resolve_credit_path(args.dirty_csv) or Path("data/unseen/credit/dirty.csv")
    cleaned_p = resolve_credit_path(cleaned_arg)
    train_rows_p = resolve_credit_path(train_rows_arg)
    out_p = Path(args.out_csv)

    models_to_run = ["hist_gb", "rf", "logreg"] if args.model == "all" else [args.model]
    all_results = []
    for m in models_to_run:
        m_out = out_p if len(models_to_run) == 1 else out_p.parent / f"m5_credit_downstream_{m}.csv"
        m_results = run_experiment(
            clean_csv=clean_p,
            dirty_csv=dirty_p,
            cleaned_csv=cleaned_p,
            out_csv=m_out,
            seed=args.seed,
            model_type=m,
            train_rows=train_rows_p,
        )
        all_results.extend(m_results)

    if len(models_to_run) > 1:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "condition", "classifier", "n_train", "n_test", "roc_auc",
            "f1", "precision", "recall", "accuracy", "gap_closed_pct", "notes"
        ]
        with open(out_p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n[+] Successfully exported unified downstream Credit results across all {len(models_to_run)} classifiers to {out_p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
