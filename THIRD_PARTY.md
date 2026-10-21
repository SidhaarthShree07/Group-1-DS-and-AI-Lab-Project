# Third-party data, models and tools

| Resource | Used for | Licence | Notes |
|---|---|---|---|
| Raha benchmark tables (Hospital, Flights, Beers, Rayyan, Movies, Tax) — github.com/BigDaMa/raha | Training and evaluation | Apache-2.0 | Canonical copies of the five core tables |
| TableEG release (annotations, Restaurant, Marketing, Credit, Soccer test files) — github.com/viviancircle/TableEG | Error-type labels, unseen-table tests, downstream test | None stated | Raw files are not redistributed here |
| Kaggle "Give Me Some Credit" (via TableEG) | Downstream experiment (RQ4) | Kaggle competition rules | Not redistributed |
| Clean Me If You Can (Ahmadi et al., PVLDB 2026) — github.com/D2IP-TUB/Clean-Me-If-You-Can, data at zenodo.org/records/20841898 (DOI 10.5281/zenodo.20841898) | Real-error address test slice (Germany) | Ground truth derived from OpenStreetMap: ODbL 1.0; Zenodo record lists CC BY 4.0; code MIT | Share-alike and OpenStreetMap attribution; we follow the stricter ODbL reading |
| Gemma 4 E4B-it — huggingface.co/google/gemma-4-E4B-it | Base model | Apache-2.0 | Fine-tuned with LoRA |
| Qwen3-4B-Instruct-2507, Qwen3.5-9B — huggingface.co/Qwen | Zero-shot baselines in the model comparison | Apache-2.0 | Not part of the final system |
| BClean repository (Inpatient, Facilities) — github.com/thethe-github/BClean (the paper's own repository; a second copy of the data is at github.com/yyssl88/BClean) | Training only | None stated (no LICENSE file) | Cite and use for research; do not redistribute |
| ZeroED repository (Billionaire) — github.com/ZJU-DataVIP-Lab/ZeroED | Unseen-table test | None stated (no LICENSE file) | Cite and use for research; do not redistribute |
| UCI Adult (Becker and Kohavi, 1996, doi 10.24432/C5XW20), Bike Sharing (Fanaee-T, 2013, doi 10.24432/C5W894), Wine Quality (Cortez et al., 2009, doi 10.24432/C56S3T) | Clean tables for training-only synthetic augmentation | CC BY 4.0 | Keep attribution; `adult.test` stays held out |
