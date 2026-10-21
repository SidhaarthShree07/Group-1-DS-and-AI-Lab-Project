# Data

No data files are committed. `raw/` and `processed/` stay empty in git; download scripts and counts are added with Milestone 2. This file lists where each table comes from, under which licence, and how errors are defined.

## Sources and licences

| Role | Table | Source | Licence |
|---|---|---|---|
| Training | Hospital, Flights, Beers, Rayyan, Movies, Tax | [Raha repository](https://github.com/BigDaMa/raha) (`datasets/`) | Apache-2.0 |
| Training | Inpatient, Facilities (10% variants) | [BClean repository](https://github.com/thethe-github/BClean) | None stated |
| Unseen-table test | Restaurant, Marketing, Soccer | [TableEG release](https://github.com/viviancircle/TableEG) (`source/`, `test_dataset/`) | None stated |
| Unseen-table test | Billionaire | [ZeroED repository](https://github.com/ZJU-DataVIP-Lab/ZeroED) | None stated |
| Unseen-table test | Address slice (Germany) | [AddressTable on Zenodo](https://zenodo.org/records/20841898), from Clean Me If You Can | OpenStreetMap-derived ground truth ODbL 1.0 (share-alike, attribution); Zenodo record lists CC BY 4.0; code MIT |
| Downstream test | Credit | Kaggle [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit), as shipped in TableEG | Kaggle competition rules |
| Augmentation | Adult, Bike Sharing, Wine Quality | [UCI Machine Learning Repository](https://archive.ics.uci.edu/) | CC BY 4.0 |

Tables whose licence is "None stated" are cited and used for research, never redistributed.

## How an error is defined

A cell is an error when its dirty value differs from its clean value after Raha's normalisation, in this order: unescape HTML entities, collapse runs of tab, newline and space into one space, strip tab, newline and space from both ends. The clean value is the gold repair. Error types (missing value, pattern violation, rule violation, outlier) come from the annotation files released with TableEG where they exist.

## Known differences between copies

- **Beers:** TableEG's dirty file stores missing values as blanks where Raha's dirty file has `N/A`, so 1,005 errors disappear from that copy. The Raha file is used.
- **Movies:** TableEG's copy re-formatted `RatingValue` (`7.0` became `7`) in both files, losing a net 668 errors. The Raha file is used.
- **University** is not in the TableEG release. The TableEG training files (`dataset/train/*.jsonl`) cover the unseen-test tables, so they are not used for training or for evaluation.

## Checksums, versions and audit counts

Repository commits, file sizes and SHA-256 checksums of the audited external files, and the error counts of the external tables, are in [`download_log.md`](download_log.md). The Tax table is mostly re-formatting (97.6% of its error cells are `3` against `3.0` or a leading zero), which is why format differences are counted separately from true errors (Milestone 1 report, Section 7.7).
