"""Run the whole pipeline: data -> features -> models -> plots.

Usage:
    python run_all.py              # rebuild everything from the raw files
    python run_all.py --eval-only  # reuse data/processed/features.csv (fast demo)
"""
import os
import subprocess
import sys

RAW_NEEDED = ["data/raw/anom5j.xls", "data/raw/satcat.csv",
              "data/raw/sunspot.csv", "data/raw/mass.csv"]


def run(module):
    print(f"\n=== {module} ===", flush=True)
    subprocess.run([sys.executable, "-m", module], check=True)


def main():
    if "--eval-only" not in sys.argv:
        missing = [f for f in RAW_NEEDED if not os.path.exists(f)]
        if missing:
            sys.exit("Missing raw files (see README, 'Getting the data'): " + ", ".join(missing))
        if not os.path.exists("data/raw/xray_daily.csv"):
            run("src.download_xray")          # slow: downloads ~150 monthly files
        run("src.data_processing")
        run("src.add_space_features")
        run("src.build_features")
    run("src.evaluate")
    run("src.plots")
    print("\nDone. See results/comparison_table.md and results/figures/")


if __name__ == "__main__":
    main()