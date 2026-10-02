| Model | Reported train | Reported test | Ours train | Ours test |
|---|---|---|---|---|
| Linear Regression | 455% | 475% | 462% | 469% |
| Support Vector Regression | - | - | 152% | 154% |
| SVR (gamma='auto', 2018 default) | 65% | 190% | 56% | 187% |
| SVR (gamma='auto', X-ray x1e6) | 65% | 190% | 50% | 187% |
| SVR (standardized features) | - | - | 143% | 159% |
| Naive Bayes | 100% | 167% | 180% | 184% |
| Naive Bayes (reference density) | 100% | 167% | 207% | 218% |
| Random Forest (new model) | - | - | 267% | 396% |
| Median baseline (sanity check) | - | - | 146% | 146% |
