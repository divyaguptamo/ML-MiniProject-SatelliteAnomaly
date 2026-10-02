"""Figures: predicted vs true TTE (like the reference's Figure 2) and an error bar chart."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = "results/figures/"
SCATTER = ["SVR (gamma='auto', 2018 default)", "Naive Bayes",
           "Random Forest (new model)", "Median baseline (sanity check)"]
LIM = 40


def scatter_plot(oof):
    fig, axes = plt.subplots(1, len(SCATTER), figsize=(4 * len(SCATTER), 4), sharey=True)
    for ax, name in zip(axes, SCATTER):
        ax.scatter(oof["true_tte"], oof[name], s=8, alpha=0.5, color="tab:orange")
        ax.plot([0, LIM], [0, LIM], "k--", linewidth=1)
        ax.set_xlim(0, LIM)
        ax.set_ylim(0, LIM)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("True TTE (days)")
    axes[0].set_ylabel("Predicted TTE (days)")
    fig.suptitle("Out-of-fold predictions (10-fold CV); dashed line = perfect prediction")
    fig.tight_layout()
    fig.savefig(OUT + "predicted_vs_true.png", dpi=150)
    plt.close(fig)


def error_bars(res):
    res = res.sort_values("test_rel_%")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    a1.barh(res["model"], res["test_rel_%"], color="tab:blue")
    a1.set_xlabel("Test mean relative error (%)")
    a1.set_title("Reference metric")
    a2.barh(res["model"], res["test_mae_days"], color="tab:green")
    a2.set_xlabel("Test MAE (days)")
    a2.set_title("Supplementary metric")
    fig.tight_layout()
    fig.savefig(OUT + "error_comparison.png", dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    scatter_plot(pd.read_csv("results/oof_predictions.csv"))
    error_bars(pd.read_csv("results/results.csv"))
    print("Saved", OUT + "predicted_vs_true.png", "and", OUT + "error_comparison.png")


if __name__ == "__main__":
    main()