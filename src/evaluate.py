"""10-fold CV of all models; metric = mean(|y_hat - y| / y), as in the reference."""
import os

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.models import SurvivalNaiveBayes

SEED = 42
FEATURES = ["month", "sunspot", "xray_flux", "mass_kg", "perigee_km", "inclination_deg"]
REPORTED = {  # (train, test) mean error from the reference, in %
    "Linear Regression": (455, 475),
    "Support Vector Regression": (65, 190),
    "Naive Bayes": (100, 167),
}


def rel_err(y, yhat):
    return float(np.mean(np.abs(yhat - y) / y))


def models():
    return {
        "Linear Regression": LinearRegression(),
        "Support Vector Regression": SVR(kernel="rbf", C=10, epsilon=0.1),
        "SVR (standardized features)": make_pipeline(
            StandardScaler(), SVR(kernel="rbf", C=10, epsilon=0.1)),
        "Naive Bayes": SurvivalNaiveBayes(),
        "Random Forest (new model)": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
        "Median baseline (sanity check)": DummyRegressor(strategy="median"),
    }


def main():
    d = pd.read_csv("data/processed/features.csv")
    X, y = d[FEATURES].to_numpy(float), d["tte"].to_numpy(float)
    kf = KFold(n_splits=10, shuffle=True, random_state=SEED)

    rows = []
    for name in models():
        tr, te = [], []
        for train_idx, test_idx in kf.split(X):
            m = models()[name]           # fresh model each fold
            m.fit(X[train_idx], y[train_idx])
            tr.append(rel_err(y[train_idx], m.predict(X[train_idx])))
            te.append(rel_err(y[test_idx], m.predict(X[test_idx])))
        rows.append({"model": name,
                     "train_%": 100 * np.mean(tr), "test_%": 100 * np.mean(te),
                     "test_std_%": 100 * np.std(te)})
        print(f"{name:34s} train {rows[-1]['train_%']:7.1f}%   test {rows[-1]['test_%']:7.1f}%")

    res = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    res.to_csv("results/results.csv", index=False)

    lines = ["| Model | Reported train | Reported test | Ours train | Ours test |",
             "|---|---|---|---|---|"]
    for _, r in res.iterrows():
        rep = REPORTED.get(r["model"])
        rt = f"{rep[0]}%" if rep else "-"
        re_ = f"{rep[1]}%" if rep else "-"
        lines.append(f"| {r['model']} | {rt} | {re_} | {r['train_%']:.0f}% | {r['test_%']:.0f}% |")
    with open("results/comparison_table.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nSaved results/results.csv and results/comparison_table.md")


if __name__ == "__main__":
    main()