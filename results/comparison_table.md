| Model | Reported train | Reported test | Ours train | Ours test | Ours test MAE (days) |
|---|---|---|---|---|---|
| Linear Regression | 455% | 475% | 462% | 469% | 21.4 |
| Support Vector Regression | - | - | 152% | 154% | 17.1 |
| SVR (gamma='auto', 2018 default) | 65% | 190% | 56% | 187% | 17.4 |
| SVR (gamma='auto', X-ray x1e6) | 65% | 190% | 50% | 187% | 17.3 |
| SVR (standardized features) | - | - | 143% | 159% | 16.9 |
| Naive Bayes | 100% | 167% | 180% | 184% | 17.4 |
| Naive Bayes (reference density) | 100% | 167% | 207% | 218% | 18.4 |
| Random Forest (new model) | - | - | 267% | 396% | 20.1 |
| Median baseline (sanity check) | - | - | 146% | 146% | 17.1 |
