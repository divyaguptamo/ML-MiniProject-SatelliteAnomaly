# \# Satellite Anomaly Prediction: Reproduce and Extend

# 

# UE24CS352A Machine Learning mini-project.

# 

# \*\*Reference:\*\* C. Naughton, "Satellite Anomaly Prediction using Survival Analysis and Machine Learning" (Stanford CS229 project report and poster). Author's code: https://github.com/cwnaught/CS229-Final-Project (used to understand the pipeline; this repository is an independent re-implementation).

# 

# \## Problem

# 

# Predict the time to event (TTE, in days) until the next anomaly of a satellite, from six features: starting month, sunspot number, X-ray flux, spacecraft mass, perigee and orbital inclination. Each interval between consecutive anomalies of one satellite is one example.

# 

# \## Team

# 

# | Member | GitHub | Contribution |

# |---|---|---|

# | TODO name | divyaguptamo | TODO |

# | TODO partner | TODO | TODO |

# 

# \## Results (10-fold CV, seed 42, same folds for every model)

# 

# Metric: mean relative error = mean(|ŷ − y| / y), as in the reference. MAE (days) is a supplementary metric.

# 

# | Model | Reported train / test | Ours train / test | Ours test MAE (days) |

# |---|---|---|---|

# | Linear Regression | 455% / 475% | 463% / 469% | 21.4 |

# | SVR (RBF, C=10, ε=0.1, gamma='auto') | 65% / 190% | 56% / 187% | 17.4 |

# | Naïve Bayes (Kaplan-Meier prior) | 100% / 167% | 180% / 184% | 17.4 |

# | \*\*Random Forest (our addition)\*\* | not in reference | 267% / 396% | 20.1 |

# | Median baseline (sanity check) | not in reference | 146% / 146% | 17.1 |

# 

# Supplementary runs are in `results/comparison\_table.md` and `results/results.csv`; figures are in `results/figures/`.

# 

# \*\*Reproduction.\*\* Linear Regression matches closely. SVR matches the reference's overfitting pattern only with `gamma='auto'`, the scikit-learn default before version 0.22, and the reference code sets it explicitly. Naïve Bayes test error is within about 17 points of the reported value.

# 

# \*\*New model: Random Forest.\*\* Chosen because it handles small tabular data without feature scaling and captures nonlinear interactions. It performs worse than the reference models and than a constant-median baseline on both metrics. Training error is far below test error, so it overfits; the data has only 661 rows and 11 distinct satellites (three features take just 11 values). On MAE, all models except Linear Regression and the Random Forest lie between 16.9 and 17.4 days, essentially the baseline's 17.1, so none of them predicts anomaly timing usefully. This agrees with the reference's own conclusion.

# 

# \## Differences from the reference

# 

# \- \*\*Dataset size:\*\* 661 rows from 11 satellites, versus 726 rows from 10. GOES-02 X-ray data is no longer available from NCEI and 1981 to 1982 have no X-ray files, so intervals starting on days without an X-ray reading are dropped (the reference also merges on the exact date). See `data/raw/xray\_coverage.csv`. February is skipped, as in the reference's month list.

# \- \*\*Satellite attributes:\*\* perigee and inclination come from today's Celestrak SATCAT (latest orbit, not the 1980s orbit). Mass comes from WMO OSCAR and Gunter's Space Page ("mass at launch"); sources are listed in `data/raw/mass.csv`. The reference's hand-built feature file is not public.

# \- \*\*Sunspot numbers:\*\* SILSO daily total sunspot number, not the NOAA series used in the reference.

# \- \*\*Evaluation protocol:\*\* the reference code uses a single split, not 10-fold CV, so our averaged numbers are not computed the same way.

# \- \*\*Naïve Bayes details:\*\* the paper does not state variance handling; we add a small smoothing term (1e-3 of each feature's variance). The reference code's 100% training error appears to result from indexing the training predictions with a stale loop variable (`predvect\[i]` instead of `predvect\[k]`); we could not verify this without the reference data. The reference also omits the 1/σ factor in its Gaussian density; we use the correct density and report the reference-style variant as a supplementary row.

# \- \*\*X-ray scaling:\*\* the reference SVR code multiplies X-ray flux by 1e6; this does not change the test error here (187.3% either way).

# 

# \## Getting the data

# 

# Raw files are not committed. Put them in `data/raw/`:

# 

# | File | Source |

# |---|---|

# | `anom5j.xls` | NOAA NCEI, "Space Weather Satellite Anomalies", link "Anomaly table (sans TDRS-1)" |

# | `satcat.csv` | https://celestrak.org/pub/satcat.csv |

# | `sunspot.csv` | SILSO daily total sunspot number: https://www.sidc.be/SILSO/INFO/sndtotcsv.php |

# | X-ray flux | downloaded automatically by `src/download\_xray.py` from NCEI GOES SEM archives |

# 

# `data/raw/mass.csv` and the processed tables in `data/processed/` are committed, so the models can be evaluated without the raw files.

# 

# \## Setup and run

# 

# ```

# python -m venv .venv

# .venv\\Scripts\\Activate.ps1

# pip install -r requirements.txt

# 

# python run\_all.py --eval-only   # fast: models and plots from data/processed/features.csv

# python run\_all.py               # full rebuild from the raw files

# ```

# 

# Tested with Python 3.14 on Windows.

# 

# \## Layout

# 

# ```

# src/download\_xray.py        GOES X-ray download and daily averaging

# src/data\_processing.py      TTE intervals and filters

# src/add\_space\_features.py   sunspot and X-ray merge

# src/build\_features.py       satellite attributes and final feature table

# src/models.py               survival Naive Bayes (Kaplan-Meier prior)

# src/evaluate.py             10-fold CV, relative error and MAE

# src/plots.py                figures

# run\_all.py                  full pipeline

# ```

