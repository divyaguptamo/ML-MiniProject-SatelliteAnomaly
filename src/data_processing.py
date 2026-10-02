"""Stage 1: build time-to-event (TTE) intervals from the NOAA anomaly list.

Follows the reference pipeline's filters:
  - TTE = days between consecutive anomalies of the same satellite
  - drop TTE == 0 and TTE >= 365 days
  - keep satellites with more than MIN_POINTS intervals
  - drop SCATHA and ECS 1
  - keep only satellites that can be matched to Celestrak by name,
    plus a hand-picked list of GPS satellites (as in the reference)
Deviation from the reference: anomalies are sorted by date within each
satellite before differencing.
"""
import os

import pandas as pd

RAW = "data/raw/"
OUT = "data/processed/"

MIN_POINTS = 20   # keep satellites with MORE than this many intervals
MAX_TTE = 365     # drop intervals of this many days or more
EXCLUDE = {"scatha", "ecs 1"}
GPS_NAMES = ["gps 5111", "gps 5112", "gps 5113", "gps 5114",
             "gps 5118", "gps 9794", "gps svn11"]


def norm(s):
    """Lowercase names, '-' -> space, so 'GOES-5' matches Celestrak 'GOES 5'."""
    return s.astype(str).str.strip().str.replace("-", " ", regex=False).str.lower()


def load_anomalies():
    df = pd.read_excel(RAW + "anom5j.xls")[["BIRD", "ADATE"]].dropna()
    df["bird"] = norm(df["BIRD"])
    df["adate"] = pd.to_datetime(df["ADATE"].astype(str).str[:10], errors="coerce")
    return df.dropna(subset=["adate"])[["bird", "adate"]]


def load_satcat():
    sc = pd.read_csv(RAW + "satcat.csv")
    sc = sc[sc["OBJECT_TYPE"] == "PAY"].copy()
    sc["name"] = norm(sc["OBJECT_NAME"])
    return sc


def build_intervals(df):
    df = df.sort_values(["bird", "adate"]).copy()
    df["next"] = df.groupby("bird")["adate"].shift(-1)
    df["tte"] = (df["next"] - df["adate"]).dt.days
    iv = df.dropna(subset=["tte"]).rename(columns={"adate": "t0"})
    iv = iv[(iv["tte"] > 0) & (iv["tte"] < MAX_TTE)]
    return iv[["bird", "t0", "tte"]].reset_index(drop=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    iv = build_intervals(load_anomalies())
    iv = iv[~iv["bird"].isin(EXCLUDE)]

    names_sc = set(load_satcat()["name"])
    summary = iv["bird"].value_counts().rename("n_intervals").to_frame()
    summary["in_celestrak"] = summary.index.isin(names_sc)
    summary["in_gps_list"] = summary.index.isin(GPS_NAMES)

    big = summary[summary["n_intervals"] > MIN_POINTS]
    print("Satellites with more than", MIN_POINTS, "intervals:\n")
    print(big.to_string())

    keep = big[big["in_celestrak"] | big["in_gps_list"]]
    iv = iv[iv["bird"].isin(keep.index)]
    iv.to_csv(OUT + "intervals.csv", index=False)
    print(f"\nKept {len(iv)} intervals from {iv['bird'].nunique()} satellites")
    print("(reference paper: 726 rows, 10 satellites, before the X-ray merge)")


if __name__ == "__main__":
    main()