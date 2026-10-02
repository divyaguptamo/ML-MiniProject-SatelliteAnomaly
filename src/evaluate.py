"""10-fold CV of all models.

Main metric (as in the reference): mean(|y_hat - y| / y)  -> "relative error".
Supplementary metric: MAE in days, which does not blow up on very short TTEs.
"""
import os

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVR

from src.models import SurvivalNaiveBayes

SEED = 42
FEATURES = ["month", "sunspot", "xray_flux", "mass_kg", "perigee_km", "inclination_deg"]
REPORTED = {  # (train, test) mean relative error from the reference, in %
    "Linear Regression": (455, 475),
    "SVR (gamma='auto', 2018 default)": (65, 190),
    "SVR (gamma='auto', X-ray x1e6)": (65, 190),
    "Naive Bayes": (100, 167),
    "Naive Bayes (reference density)": (100, 167),
}


def rel_err(y, yhat):
    return float(np.mean(np.abs(yhat - y) / y))


def mae(y, yhat):
    return float(np.mean(np.abs(yhat - y)))


def scale_xray(X):
    """Reference svm.py multiplies X-ray flux (column 2) by 1e6."""
    X = np.array(X, dtype=float, copy=True)
    X[:, 2] *= 1e6
    return X


def models():
    return {
        "Linear Regression": LinearRegression(),
        "Support Vector Regression": SVR(kernel="rbf", C=10, epsilon=0.1),
        "SVR (gamma='auto', 2018 default)": SVR(
            kernel="rbf", C=10, epsilon=0.1, gamma="auto"),
        "SVR (gamma='auto', X-ray x1e6)": make_pipeline(
            FunctionTransformer(scale_xray),
            SVR(kernel="rbf", C=10, epsilon=0.1, gamma="auto")),
        "SVR (standardized features)": make_pipeline(
            StandardScaler(), SVR(kernel="rbf", C=10, epsilon=0.1)),
        "Naive Bayes": SurvivalNaiveBayes(),
        "Naive Bayes (reference density)": SurvivalNaiveBayes(normalize_density=False),
        "Random Forest (new model)": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
        "Median baseline (sanity check)": DummyRegressor(strategy="median"),
    }


def main():
    d = pd.read_csv("data/processed/features.csv")
    X, y = d[FEATURES].to_numpy(float), d["tte"].to_numpy(float)
    kf = KFold(n_splits=10, shuffle=True, random_state=SEED)

    rows = []
    oof = pd.DataFrame({"true_tte": y})
    print(f"{'model':34s} {'rel-err train/test':>20s}   {'MAE train/test (days)':>22s}")
    for name in models():
        tr_r, te_r, tr_m, te_m = [], [], [], []
        pred = np.zeros(len(y))
        for train_idx, test_idx in kf.split(X):
            m = models()[name]           # fresh model each fold
            m.fit(X[train_idx], y[train_idx])
            p_tr, p_te = m.predict(X[train_idx]), m.predict(X[test_idx])
            pred[test_idx] = p_te
            tr_r.append(rel_err(y[train_idx], p_tr))
            te_r.append(rel_err(y[test_idx], p_te))
            tr_m.append(mae(y[train_idx], p_tr))
            te_m.append(mae(y[test_idx], p_te))
        oof[name] = pred
        rows.append({"model": name,
                     "train_rel_%": 100 * np.mean(tr_r),
                     "test_rel_%": 100 * np.mean(te_r),
                     "test_rel_std_%": 100 * np.std(te_r),
                     "train_mae_days": np.mean(tr_m),
                     "test_mae_days": np.mean(te_m)})
        r = rows[-1]
        print(f"{name:34s} {r['train_rel_%']:8.1f}% /{r['test_rel_%']:7.1f}%   "
              f"{r['train_mae_days']:9.1f} /{r['test_mae_days']:7.1f}")

    res = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    res.to_csv("results/results.csv", index=False)
    oof.to_csv("results/oof_predictions.csv", index=False)

    lines = ["| Model | Reported train | Reported test | Ours train | Ours test | Ours test MAE (days) |",
             "|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        rep = REPORTED.get(r["model"])
        rt = f"{rep[0]}%" if rep else "-"
        re_ = f"{rep[1]}%" if rep else "-"
        lines.append(f"| {r['model']} | {rt} | {re_} | {r['train_rel_%']:.0f}% | "
                     f"{r['test_rel_%']:.0f}% | {r['test_mae_days']:.1f} |")
    with open("results/comparison_table.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nSaved results/results.csv, results/oof_predictions.csv, results/comparison_table.md")


if __name__ == "__main__":
    main()