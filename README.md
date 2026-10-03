# Satellite Anomaly Prediction: Reproduce and Extend

UE24CS352A Machine Learning mini-project.

**Reference:** C. Naughton, "Satellite Anomaly Prediction using Survival Analysis and Machine Learning" (Stanford CS229 project report and poster). Author's code: <https://github.com/cwnaught/CS229-Final-Project>. We used it to understand the pipeline; this repository is an independent re-implementation.

## Problem

Predict the time to event (TTE, in days) until a satellite's next anomaly. Each interval between two consecutive anomalies of one satellite is one example. The model uses six features:

- starting month
- sunspot number
- X-ray flux
- spacecraft mass
- perigee
- orbital inclination

## Team

| Member | GitHub | Contribution |
|---|---|---|
| Divya Gupta | [divyaguptamo](https://github.com/divyaguptamo) | download_xray.py, data_processing.py, add_space_features.py, build_features.py, mass.csv |
| Vanishree | [Vaniprsy](https://github.com/Vaniprsy )| models.py, evaluate.py, plots.py |

## Results

All models use 10-fold CV with seed 42 and the same folds.

The main metric is mean relative error, mean(|ŷ − y| / y), as in the reference. MAE in days is a supplementary metric. "± std" is the standard deviation of test error across the 10 folds.

| Model | Reported train / test | Ours train / test (± std) | Ours test MAE (days) |
|---|---|---|---|
| Linear Regression | 455% / 475% | 462% / 469% (± 90) | 21.4 |
| SVR (RBF, C=10, ε=0.1, gamma='auto') | 65% / 190% | 56% / 187% (± 22) | 17.4 |
| Naïve Bayes (Kaplan-Meier prior) | 100% / 167% | 180% / 184% (± 23) | 17.4 |
| **Random Forest (our addition)** | not in reference | 267% / 396% (± 81) | 20.1 |
| Median baseline (sanity check) | not in reference | 146% / 146% (± 19) | 17.1 |

Supplementary runs are in `results/comparison_table.md` and `results/results.csv`. They cover:

- SVR with the current `gamma='scale'` default
- SVR with standardized features
- SVR with X-ray flux scaled by 1e6
- the reference-style Naïve Bayes density

Figures are in `results/figures/`. The predicted-vs-true plots show TTE from 0 to 40 days, where about 88% of the intervals lie. The longest intervals (up to 346 days) fall outside the plotted range.

### Reproduction

- **Linear Regression** matches closely: 469% vs 475% test.
- **SVR** reproduces the reference's overfitting pattern (low train error, high test error) only with `gamma='auto'`. This was the scikit-learn default before version 0.22, and the reference code sets it explicitly. With the modern `gamma='scale'` default, SVR no longer overfits: 152% / 154%.
- **Naïve Bayes** test error is 184% vs the reported 167%. That is within one fold standard deviation (± 23 points).

### New model: Random Forest

**Why we chose it:** Random Forest handles small tabular data without feature scaling and can capture nonlinear interactions between space-weather and spacecraft features. None of the reference's models can capture these interactions.

**Outcome:** it performs worse than the reference models, and worse than a constant-median baseline, on both metrics. Its training error is far below its test error, so it overfits. Two properties of the data explain this:

- There are only 661 rows.
- The three spacecraft features (mass, perigee, inclination) take just 11 distinct values, one per satellite.

Because the folds are shuffled rows, the forest can memorise per-satellite patterns that do not generalise.

**Overall:** on MAE, every model except Linear Regression and the Random Forest scores between 16.9 and 17.4 days. That is essentially the baseline's 17.1 days, so none of the models predicts anomaly timing usefully. This agrees with the reference's own conclusion.

## Differences from the reference

- **Dataset size:** 661 rows from 11 satellites, versus 726 rows from 10.
  - GOES-02 X-ray data is no longer available from NCEI.
  - There are no X-ray files for 1981 and 1982.
  - Intervals that start on a day without an X-ray reading are dropped. The reference also merges on the exact date. See `data/raw/xray_coverage.csv`.
  - February is skipped, as in the reference's month list.
- **Satellite attributes:**
  - Perigee and inclination come from today's Celestrak SATCAT, so they reflect the latest orbit, not the 1980s orbit.
  - Mass is "mass at launch" from WMO OSCAR and Gunter's Space Page; sources are listed in `data/raw/mass.csv`.
  - The reference's hand-built feature file is not public.
- **Sunspot numbers:** we use the SILSO daily total sunspot number, not the NOAA series used in the reference.
- **Evaluation protocol:** the reference paper describes 10-fold CV, but its published code uses a single train/validation split. Our 10-fold averages are therefore not computed exactly the same way as the reported numbers.
- **Naïve Bayes details:**
  - The paper does not state how it handles variance. We add a small smoothing term: 1e-3 of each feature's variance.
  - The reference code's 100% training error appears to come from a stale loop variable: it indexes `predvect[i]` inside a `for k` loop. We could not confirm this without the reference data.
  - The reference omits the 1/σ factor in its Gaussian density. We use the correct density and report the reference-style variant as a supplementary row.
- **X-ray scaling:** the reference SVR code multiplies X-ray flux by 1e6. This does not change the test error here (187.3% either way).

## Getting the data

Raw files are not committed. Put them in `data/raw/`:

| File | Source |
|---|---|
| `anom5j.xls` | NOAA NCEI, "Space Weather Satellite Anomalies", link "Anomaly table (sans TDRS-1)" |
| `satcat.csv` | <https://celestrak.org/pub/satcat.csv> |
| `sunspot.csv` | SILSO daily total sunspot number: <https://www.sidc.be/SILSO/INFO/sndtotcsv.php> |
| X-ray flux | downloaded automatically by `src/download_xray.py` from the NCEI GOES SEM archives |

`data/raw/mass.csv` and the processed tables in `data/processed/` are committed, so the models can be evaluated without the raw files.

## Setup and run

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows (PowerShell)
# source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt

python run_all.py --eval-only     # fast: models and plots from data/processed/features.csv
python run_all.py                 # full rebuild from the raw files
```

Tested with Python 3.14 on Windows.

## Layout

```
src/download_xray.py        GOES X-ray download and daily averaging
src/check_xray.py           checks which GOES X-ray files exist on NCEI
src/data_processing.py      TTE intervals and filters
src/add_space_features.py   sunspot and X-ray merge
src/build_features.py       satellite attributes and final feature table
src/models.py               survival Naive Bayes (Kaplan-Meier prior)
src/evaluate.py             10-fold CV, relative error and MAE
src/plots.py                figures
run_all.py                  full pipeline
```
