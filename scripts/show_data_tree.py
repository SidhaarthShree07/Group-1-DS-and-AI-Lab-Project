# Author: Jashan Tiwari
"""
scripts/show_data_tree.py

ContextGuard Milestone 3: Data Organization Tree & Repository Structure Auditor
Author: Jashan Tiwari (Data Lead) - DSAI Lab, IIT Madras BS
Role: Implements Milestone 3 Task Card (Items 1 & 2):
1. Audits all directories under data/, calculating file counts, sizes, and documenting
   contents and the script responsible for generating each directory.
2. Exports summary table to docs/tables/m3_data_tree.csv and prints formatted tree for Report §8.
3. Generates clean repository tree structure for Report §15 (excluding .git, __pycache__, .venv).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

KNOWN_FOLDERS_METADATA = {
    "data/raw": (
        "Raw pristine data sources fetched from pinned commits and archives",
        "scripts/download_data.py",
    ),
    "data/raw/raha": (
        "Raha benchmark tables (Hospital, Flights, Beers, Rayyan, Movies, Tax)",
        "scripts/download_data.py",
    ),
    "data/raw/BClean": (
        "BClean healthcare benchmark tables (Inpatient, Facilities)",
        "scripts/download_data.py",
    ),
    "data/raw/ZeroED": (
        "ZeroED benchmark table (Billionaire)",
        "scripts/download_data.py",
    ),
    "data/raw/adult": (
        "UCI Adult tabular dataset for synthetic generation",
        "scripts/download_data.py",
    ),
    "data/raw/germany": (
        "Clean-Me-If-You-Can Germany address slice (Zenodo range download)",
        "scripts/download_data.py",
    ),
    "data/processed": (
        "Processed cell-level ground-truth files and instruction examples",
        "Pipeline scripts",
    ),
    "data/processed/core": (
        "Cell-level ground-truth labels for 5 core benchmark tables",
        "src/data/load_core.py",
    ),
    "data/processed/external": (
        "Cell-level ground truth for external training tables (Tax, Inpatient, Facilities)",
        "src/data/load_external.py",
    ),
    "data/processed/germany": (
        "Cell-level ground truth for unseen Germany address evaluation table",
        "src/data/load_external.py",
    ),
    "data/processed/synthetic": (
        "Clean UCI tables (Adult, Bike, Wine) prepared for synthetic noise injection",
        "src/data/synthetic.py",
    ),
    "data/processed/examples": (
        "Full fine-tuning instruction-response JSONL datasets (train, val, test)",
        "src/llm/build_examples.py",
    ),
    "data/splits": (
        "Partitioned clean/dirty table splits and cell ground-truth CSVs",
        "Splitting scripts",
    ),
    "data/splits/core": (
        "Group-aware 70/15/15 train/val/test splits for 5 core tables",
        "src/data/split.py",
    ),
    "data/splits/external": (
        "Entity-aware 70/15/15 train/val/test splits for external training tables",
        "src/data/load_external.py",
    ),
    "data/splits/synthetic": (
        "Train-only synthetic tables with controlled error distributions",
        "src/data/synthetic.py",
    ),
    "data/subsets": (
        "Small-scale validation subsets for rapid pipeline and model verification",
        "src/data/make_subset.py",
    ),
    "data/subsets/m3": (
        "100-row validation subsets (Hospital, Flights, Beers) for M3 end-to-end runs",
        "src/data/make_subset.py",
    ),
    "data/sample": (
        "Versioned minimal sample of instruction examples for repository verification",
        "src/llm/build_examples.py",
    ),
}


def get_dir_metrics(dir_path: Path) -> Tuple[int, int]:
    """Returns (file_count, total_size_bytes) for direct files in directory."""
    files = [f for f in dir_path.iterdir() if f.is_file() and not f.name.startswith(".DS_Store")]
    total_size = sum(f.stat().st_size for f in files)
    return len(files), total_size


def get_recursive_dir_metrics(dir_path: Path) -> Tuple[int, int]:
    """Returns (file_count, total_size_bytes) for all files under directory recursively."""
    total_files = 0
    total_size = 0
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.startswith(".DS_Store"):
                continue
            fp = Path(root) / f
            if not fp.is_symlink():
                total_size += fp.stat().st_size
                total_files += 1
    return total_files, total_size


def format_bytes(num_bytes: int) -> str:
    """Formats bytes into human-readable string."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    elif num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{num_bytes / (1024 * 1024 * 1024):.2f} GB"


def audit_data_tree(repo_root: Path) -> List[Dict[str, Any]]:
    """Scans data/ directory and builds structured audit table."""
    data_dir = repo_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"data/ directory not found at {data_dir}")

    # Discover all unique directories under data/
    discovered_dirs = set()
    for root, dirs, _ in os.walk(data_dir):
        rel = Path(root).relative_to(repo_root).as_posix()
        discovered_dirs.add(rel)

    # Sort topologically/alphabetically
    sorted_dirs = sorted(discovered_dirs, key=lambda p: (p.count("/"), p))

    records: List[Dict[str, Any]] = []

    for rel_path in sorted_dirs:
        p = repo_root / rel_path
        file_count, size_bytes = get_dir_metrics(p)
        rec_files, rec_size = get_recursive_dir_metrics(p)

        desc, script = KNOWN_FOLDERS_METADATA.get(
            rel_path,
            (
                "Table partition folder containing clean, dirty, and cell ground-truth CSVs",
                "src/data/split.py" if "splits" in rel_path else "src/data/make_subset.py" if "subsets" in rel_path else "scripts/download_data.py",
            ),
        )

        records.append({
            "folder": rel_path,
            "direct_files": file_count,
            "direct_size": format_bytes(size_bytes),
            "total_files": rec_files,
            "total_size": format_bytes(rec_size),
            "size_bytes": rec_size,
            "description": desc,
            "written_by": script,
        })

    return records


def print_repository_tree(repo_root: Path) -> str:
    """Generates the clean canonical repository tree for Report §15."""
    tree_lines = [
        "Group-1-DS-and-AI-Lab-Project/",
        "├── configs/",
        "│   └── m3_smoke.yaml              # Milestone 3 run configuration (Decision D20)",
        "├── data/",
        "│   ├── raw/                       # Downloaded raw sources (git-ignored)",
        "│   ├── processed/                 # Ground truth cells & instruction jsonl (git-ignored)",
        "│   │   ├── core/                  # Core table cell ground truth CSVs",
        "│   │   ├── external/              # External table cell ground truth CSVs",
        "│   │   └── examples/              # train.jsonl, val.jsonl, test.jsonl",
        "│   ├── splits/                    # 70/15/15 partitioned CSVs (git-ignored)",
        "│   │   ├── core/                  # Hospital, Flights, Beers, Rayyan, Movies",
        "│   │   ├── external/              # Tax, Inpatient, Facilities",
        "│   │   └── synthetic/             # Adult, Bike, Wine",
        "│   ├── subsets/                   # Small-scale validation subsets (git-ignored)",
        "│   │   └── m3/                    # 100-row subsets (Hospital, Flights, Beers)",
        "│   └── sample/                    # Tracked minimal instruction sample (m2_examples_sample.jsonl)",
        "├── docs/",
        "│   ├── figures/m2/                # Milestone 2 vector SVG figures",
        "│   ├── figures/m3/                # Milestone 3 architecture & pipeline figures",
        "│   ├── milestones/                # Milestone reports, slides, and worklogs",
        "│   └── tables/                    # Summary audit, quality, split, and loader CSVs",
        "│       ├── m2_core_splits.csv",
        "│       ├── m2_core_quality.csv",
        "│       ├── m2_core_dictionary.csv",
        "│       ├── m3_subset_summary.csv",
        "│       ├── m3_loader_stats.csv",
        "│       └── m3_data_tree.csv",
        "├── models/                        # Model configuration and checkpoints (git-ignored)",
        "├── outputs/                       # Experiment run logs and predictions (git-ignored)",
        "│   └── runs/                      # Per-run records (outputs/runs/<run_id>/)",
        "├── scripts/",
        "│   ├── build_all.sh               # One-command full pipeline reproduction",
        "│   ├── download_data.py           # Pinned data source downloader & hash verifier",
        "│   ├── generate_m2_outputs.py     # Milestone 2 quality audit and dictionary builder",
        "│   ├── eda_analysis.py            # Exploratory data analysis & outlier profiling",
        "│   └── show_data_tree.py          # Data tree and repository structure generator",
        "├── src/",
        "│   ├── data/                      # Data loading, entity splitting, and subset sampling",
        "│   │   ├── load_core.py           # Core benchmark table ingestion & Raha normalization",
        "│   │   ├── load_external.py       # External source ingestion & schema alignment",
        "│   │   ├── split.py               # Group-aware 70/15/15 clean-key splitter",
        "│   │   ├── synthetic.py           # Controlled synthetic error generation",
        "│   │   └── make_subset.py         # 100-row validation subset builder (Decision D17)",
        "│   ├── profiling/                 # Column profiling and dependency extraction",
        "│   │   ├── context.py             # Prompt context builder (row, profile, dependencies)",
        "│   │   └── dependencies.py        # TANE / approximate dependency miner",
        "│   ├── detectors/                 # Classical candidate detectors D1-D6 & Stage 0 D7",
        "│   ├── llm/                       # Language model training, data loading, and inference",
        "│   │   ├── build_examples.py      # Instruction-response example generator",
        "│   │   └── dataset.py             # Gemma chat template ExampleDataset & collate (D13, D14)",
        "│   └── evaluation/                # Metrics, JSON parsing, calibration, and scoring",
        "└── tests/                         # Pytest test suite",
        "    ├── test_data_pipeline.py      # Core data ingestion and split tests",
        "    ├── test_context.py            # Context builder and prompt budget tests",
        "    ├── test_synthetic.py          # Synthetic generator tests",
        "    └── test_m3_jashan.py          # Unit tests for ExampleDataset, collate, and make_subset",
    ]
    return "\n".join(tree_lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit data/ tree and display repository structure")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root path",
    )
    args = parser.parse_args()

    repo_root = args.root
    tables_dir = repo_root / "docs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("      MILESTONE 3: AUDITING DATA/ DIRECTORY ORGANIZATION (REPORT §8)           ")
    print("=" * 80)

    records = audit_data_tree(repo_root)

    # Print data tree table
    print(f"{'Folder':<35} | {'Files':<6} | {'Total Size':<10} | {'Written By':<25}")
    print("-" * 82)
    for r in records:
        print(f"{r['folder']:<35} | {r['total_files']:<6} | {r['total_size']:<10} | {r['written_by']:<25}")

    # Export to CSV
    out_csv = tables_dir / "m3_data_tree.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["folder", "direct_files", "direct_size", "total_files", "total_size", "description", "written_by"]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)

    print("=" * 80)
    print(f"[SUCCESS] Exported data tree audit -> {out_csv.relative_to(repo_root)}")
    print("=" * 80)
    print("\n" + "=" * 80)
    print("      CANONICAL REPOSITORY DIRECTORY STRUCTURE FOR REPORT §15                 ")
    print("=" * 80)
    repo_tree_text = print_repository_tree(repo_root)
    print(repo_tree_text)
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())

