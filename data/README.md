# Data

No raw or processed data files are committed (`raw/`, `processed/` and `splits/` stay empty in git; only the 20-record sample in `sample/` is tracked). `scripts/download_data.py` downloads every source from its owner at a pinned version and checks SHA-256; `scripts/build_all.sh` builds everything else. This file lists where each table comes from, under which licence, and how errors are defined.

## Sources and licences

| Role | Table | Source | Licence |
|---|---|---|---|
| Training | Hospital, Flights, Beers, Rayyan, Movies, Tax | [Raha repository](https://github.com/BigDaMa/raha) (`datasets/`) | Apache-2.0 |
| Training | Inpatient, Facilities (10% variants) | [BClean repository](https://github.com/thethe-github/BClean) | No licence file; terms inherited from the benchmarks it bundles (reviewed and approved, see below) |
| Unseen-table test | Restaurant, Marketing, Soccer | [TableEG release](https://github.com/viviancircle/TableEG) (`source/`, `test_dataset/`) | No licence file; terms inherited from the benchmarks it bundles (reviewed and approved, see below) |
| Unseen-table test | Billionaire | [ZeroED repository](https://github.com/ZJU-DataVIP-Lab/ZeroED) | No licence file; terms inherited from the benchmarks it bundles (reviewed and approved, see below) |
| Unseen-table test | Address slice (Germany) | [AddressTable on Zenodo](https://zenodo.org/records/20841898), from Clean Me If You Can | OpenStreetMap-derived ground truth ODbL 1.0 (share-alike, attribution); Zenodo record lists CC BY 4.0; code MIT |
| Downstream test | Credit | Kaggle [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit), as shipped in TableEG | Kaggle competition rules |
| Augmentation | Adult, Bike Sharing, Wine Quality | [UCI Machine Learning Repository](https://archive.ics.uci.edu/) | CC BY 4.0 |

**Licence review (28 Oct 2026).** The team reviewed the licences and approved use for research and sharing with attribution. Attribution for every source is in [`THIRD_PARTY.md`](../THIRD_PARTY.md). Files derived from the Germany slice stay under ODbL 1.0 with attribution to OpenStreetMap contributors and are published separately from the training files (they are test-only). TableEG's outputs fall under the Llama 3.1 Community Licence; we use TableEG only for error-type labels and unseen test tables, never as training data.

## Prepared data (Milestone 2)

| File | Content | Size |
|---|---|---|
| `processed/examples/train.jsonl` | 55,091 instruction–response examples (44,077 benchmark, 11,014 synthetic) | about 102 MB |
| `processed/examples/val.jsonl` | 9,341 examples | about 17 MB |
| `processed/examples/test.jsonl` | 9,547 sampled examples (`--full-test` writes every test cell) | about 18 MB |
| `sample/m2_examples_sample.jsonl` | 20 records from the Apache-2.0 core tables | 40 KB |

The three example files are not in git because of their size; they are published as a release asset of this repository (linked in the release notes of the Milestone 2 submission) and are rebuilt byte for byte by `bash scripts/build_all.sh`. Record format: Section 3 of the Milestone 2 report; table-level counts: `docs/tables/m2_*.csv`.

## How an error is defined

A cell is an error when its dirty value differs from its clean value after Raha's normalisation, in this order: unescape HTML entities, collapse runs of tab, newline and space into one space, strip tab, newline and space from both ends. The clean value is the gold repair. Error types (missing value, pattern violation, rule violation, outlier) come from the annotation files released with TableEG where they exist.

## Known differences between copies

- **Beers:** TableEG's dirty file stores missing values as blanks where Raha's dirty file has `N/A`, so 1,005 errors disappear from that copy. The Raha file is used.
- **Movies:** TableEG's copy re-formatted `RatingValue` (`7.0` became `7`) in both files, losing a net 668 errors. The Raha file is used.
- **University** is not in the TableEG release. The TableEG training files (`dataset/train/*.jsonl`) cover the unseen-test tables, so they are not used for training or for evaluation.

## Checksums, versions and audit counts

Repository commits, file sizes and SHA-256 checksums of the audited external files, and the error counts of the external tables, are in [`download_log.md`](download_log.md). The Tax table is mostly re-formatting (97.6% of its error cells are `3` against `3.0` or a leading zero), which is why format differences are counted separately from true errors (Milestone 1 report, Section 7.7).
