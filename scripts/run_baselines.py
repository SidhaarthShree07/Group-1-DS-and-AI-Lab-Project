import os
import sys
from pathlib import Path

# Add repo root to sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_baselines_full import main as run_full_baselines

def main():
    print("Running the M4 baselines (B1 HoloClean, B2 dBoost, B3 Raha + Baran)")
    run_full_baselines()

if __name__ == "__main__":
    main()
