# External dataset download log (Milestone 1)

Audit date: 20 Oct 2026. Raw data is never committed; this log and `scripts/external_inventory.py` let anyone re-create the audit. Milestone 2 turns the log into a download script.

## 1. Sources, versions and licences

| Source | URL | Version used | Licence actually stated | Note |
|---|---|---|---|---|
| Raha (Tax) | https://github.com/BigDaMa/raha | commit `7be1334b8c7bbdac3f47ef514fb3e1e8c5fc181c` (5 Jun 2025) | Apache-2.0 (LICENSE file) | Canonical Tax pair; similarly named Tax files in ZeroED are different versions |
| BClean (Inpatient, Facilities) | https://github.com/thethe-github/BClean (the BClean paper's own repository) | commit `bb8e09e3119dcc85475184592a311901a6b4c290` (15 Apr 2024, `master`, message "update README.md") | **None stated** (no LICENSE file) | Checked against the GitHub API: the commit exists and its `dataset/` files have exactly the sizes of Sahilpreet's copies. A second copy of the datasets is at https://github.com/yyssl88/BClean (commit `0f94decf…`); it gives the same counts. |
| ZeroED (Billionaire) | https://github.com/ZJU-DataVIP-Lab/ZeroED | commit `2b84266e1d8b2014d36643d343429d9a20bd769e` (28 Feb 2025) | **None stated** | The README points to https://github.com/WelkinNi/ZeroED as the paper's repository |
| Clean Me If You Can (German slice) | https://zenodo.org/records/20841898 (DOI 10.5281/zenodo.20841898) | AddressTable v1.0-2026-06-26, file `AddressTable.zip` (7.3 GB; only the two DE-slice files were extracted) | Zenodo record lists CC BY 4.0; the GitHub README (https://github.com/D2IP-TUB/Clean-Me-If-You-Can, commit `52c934cbcd7e14c40d5df9d90524c2d85361ff64`) says ground truth derived from OpenStreetMap is ODbL 1.0 and the code is MIT | We follow the stricter reading: attribute OpenStreetMap contributors and keep derived databases under ODbL share-alike |
| UCI tables (Adult, Bike Sharing, Wine Quality) | https://archive.ics.uci.edu/ | see Section 6 | CC BY 4.0 (UCI page) | Clean tables only, for training-only augmentation |

## 2. Files audited (checksums of the files we ran the audit on)

| Dataset | File | Bytes | SHA-256 |
|---|---|---:|---|
| Tax | `raha/datasets/tax/clean.csv` | 14,621,003 | `201290927ae92e65b3940d776b3df5b4d953c5dfd9abb231715a2e65ecca87b0` |
| Tax | `raha/datasets/tax/dirty.csv` | 14,633,203 | `8dd3429ec4791b2ed1a688c308a57a9f3d1a94f77d1f4e98294a67273270b973` |
| Billionaire | `ZeroED/data/billionaire_clean.csv` | 500,410 | `9103811393f2c192094f339963c2b0537adf1384ddb2803a5783da41a326b8cf` |
| Billionaire | `ZeroED/data/billionaire_error-01.csv` | 487,843 | `c1c3f3a20b1778603b0655027c64b1e74153d350edf181e164bdf4ae016940d7` |
| Inpatient | `BClean/dataset/Inpatient/Inpatient_clean.csv` | 495,764 | `75bd12ef098de0b7c6f4a5153890ed3ec5af731dc7f5404b741c9509320337c6` |
| Inpatient | `BClean/dataset/Inpatient/Inpatient_dirty_10.csv` | 495,276 | `680da3f6c11ec267dbf47397138568224deede0aeef1fc54a028a87527b9cfab` |
| Facilities | `BClean/dataset/facilities/facilities_clean.csv` | 1,114,686 | `07fa1a78b6fd3a87cc867f093cbb96feea4770cc5f3d5be7b8d1469adaf53198` |
| Facilities | `BClean/dataset/facilities/facilities_dirty10.csv` | 1,112,644 | `3d5fadd2cf96b9e95712e3c839a235cf7989ab9865400a52b64c86adbb59ab39` |
| Clean Me If You Can (DE) | `AddressTable/de_slice/named/slice_de_clean.csv` | 74,380,625 | `78c852e3a5680a7460732b67ab87d82484a718297190fa8852af73a80817a192` |
| Clean Me If You Can (DE) | `AddressTable/de_slice/named/slice_de_dirty.csv` | 65,282,822 | `2ef8ab1424c8c357af1aeaecef12b376cb7f20616437035f2753d0b2f2ac86ef` |

The BClean files are taken from the paper's repository at the commit above, so their sizes match Sahilpreet's copies exactly (the hashes above are of the files from that commit). The other copy (`yyssl88/BClean`) holds the same tables with slightly smaller files (4,018 bytes for Inpatient, 7,993 bytes for Facilities, one byte per row, which fits different line endings); the audit gives identical counts on both.

## 3. Alignment checks (all pass)

| Dataset | Rows | Data columns | Key used to align clean and dirty |
|---|---:|---:|---|
| Tax | 200,000 | 15 | Row position; headers identical |
| Billionaire | 2,614 | 22 | Row position; headers identical |
| Inpatient | 4,017 | 11 | Row position. The dirty file has an extra unnamed first column holding 0, 1, 2, …; it is dropped after checking the header and the sequence |
| Facilities | 7,992 | 10 | Same as Inpatient |
| Clean Me If You Can, DE slice | 374,533 (5,000 drawn) | 9 (plus `id`) | `id`: the clean and dirty files hold the same 374,533 ids |

BClean's example scripts use `Inpatient_dirty_10.csv` and `facilities_dirty5.csv`; we use the 10% file for Facilities as the task requires (`facilities_dirty10.csv`). Observed changed-cell rates of the other variants (5% Facilities 5.17%; Inpatient 15% 15.15%, 20% 20.18%) agree with their names.

## 4. Error counts (Raha rule) and types

Error cells follow Raha's rule (unescape HTML, collapse `[	
 ]+`, strip). Types are our heuristic, since these sources have no annotations: **missing_value** (dirty is empty or a missing token), **format_artifact** (the same number written differently, such as `3` and `3.0`, or a leading zero added or dropped), **rule_violation** (the dirty value is another valid value of the same column, so it breaks dependencies on other columns), **pattern_violation** (any other change). Outliers cannot be told apart from rule or pattern changes by this method and are not assigned here.

| Dataset | Error cells | Rate | Rows with an error | Missing | Pattern | Rule | Format artifact |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tax | 121,219 | 4.04% | 103,875 | 0 | 2,166 | 800 | 118,253 |
| Billionaire | 5,195 | 9.03% | 2,273 | 1,386 | 3,049 | 742 | 18 |
| Inpatient (10%) | 4,465 | 10.10% | 2,720 | 1,800 | 1,684 | 981 | 0 |
| Facilities (10%) | 8,318 | 10.41% | 5,249 | 3,001 | 2,384 | 2,933 | 0 |
| Clean Me If You Can, DE slice (5,000 rows, seed 42) | 18,678 | 41.51% | 5,000 | 12,087 | 6,292 | 295 | 4 |

Per-column counts: `docs/tables/external_column_differences.csv`. Billionaire has 5,195 errors under Raha's rule; plain string comparison gives 5,266 because 71 cells differ only in whitespace or HTML entities.

## 5. Clean Me If You Can: German slice

The Zenodo archive already contains the slice (`AddressTable/de_slice/named/slice_de_clean.csv` and `slice_de_dirty.csv`, 74.4 MB and 65.3 MB), so `prepare_de_slice.py` and the merged source inputs are not needed. The README's path (`data/slices/de/named/`) differs from the archive's (`de_slice/named/`). Both files have the header `id, name, road, house_number, postcode, locality, region, country, country_code, provenance_graph_iri` and the same 374,533 ids. The sample is `random.Random(42).sample(sorted(common ids), 5000)`; it is deterministic and is in `scripts/external_inventory.py`. Every sampled row contains at least one error, 65% of the errors are missing values, and the clean side uses full forms (`Deutschland`, `Baden-Württemberg`) where the dirty side often has `DE` or `SH`.

## 6. UCI clean tables (training-only augmentation)

| Dataset | File | Rows | Columns | Citation |
|---|---|---:|---:|---|
| Adult | `adult.data` | 32,561 | 15 | Becker and Kohavi (1996), https://doi.org/10.24432/C5XW20 |
| Adult | `adult.test` | 16,281 | 15 | same; stays held out |
| Bike Sharing | `day.csv` | 731 | 16 | Fanaee-T (2013), https://doi.org/10.24432/C5W894 |
| Bike Sharing | `hour.csv` | 17,379 | 17 | same |
| Wine Quality | `winequality-red.csv` | 1,599 | 12 | Cortez et al. (2009), https://doi.org/10.24432/C56S3T |
| Wine Quality | `winequality-white.csv` | 4,898 | 12 | same |

UCI lists CC BY 4.0 for all three (checked by Sahilpreet on the UCI pages, 20 Oct 2026); keep the attribution when the data is reused or adapted.

## 7. Overlap checks

- **Inpatient and Facilities against Hospital** (clean files, exact match): no shared identifiers (Facilities CCNs: 0 of 888 equal a Hospital ProviderNumber; no shared phone numbers; Inpatient provider ids and phones: 0). By name and address there is a small overlap with Facilities: 1 of Hospital's 45 hospital names and 1 of its 45 addresses also occur in Facilities, and 6 of its 44 zip codes do. These are the same domain, so both tables stay **training-only**, as planned.
- **Billionaire against the other tables:** no row-level overlap. Exact matches of person names (3: George Lucas, James Moran, Steven Spielberg, found in Movies) and of company names (Tax 15, Beers 2, Facilities 2, Hospital 1, Inpatient 1) are single common words or famous names in different attributes; none is a repeated row.

## 8. Remaining open points

- The original data owner of Billionaire is not stated in the ZeroED repository.
