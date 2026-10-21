# Contributing

## Team and responsibilities
| Member | GitHub | Area |
|---|---|---|
| Sidhaarth Shree | SidhaarthShree07 | LLM, evaluation, integration, repository owner |
| Jashan | jashantiwari044 | Data (benchmark tables) |
| Sahilpreet | 23f3000115 | Data (external datasets) |
| Aaradhya | adhi1911 | Detectors and baselines |

## Branches and pull requests
- Work on a branch named `<milestone>/<name>-<topic>`, for example `m1/jashan-data-audit`; never commit directly to `main`.
- Open a pull request into `main`; at least one other member reviews it before merging.
- A file is merged only when it is final for the milestone (no placeholders).

## Coding conventions
- Python 3.12, formatted with `black`, imports sorted with `isort`; type hints on public functions.
- Reusable code lives in `src/`; notebooks are for exploration and are cleaned before commit.
- Every script reads paths and seeds from arguments or `configs/`; no hard-coded absolute paths.
- Never commit data files, model weights, credentials or `.env` files.

## Reproducibility
Fix random seeds, record library versions in `requirements.txt`, and log every run (config, seed, data version) so that reported numbers can be reproduced.
