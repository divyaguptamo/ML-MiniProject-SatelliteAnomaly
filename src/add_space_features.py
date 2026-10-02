"""Stage 2: merge sunspot number and X-ray flux onto each interval's start date."""
import pandas as pd

RAW = "data/raw/"
OUT = "data/processed/"


def main():
    iv = pd.read_csv(OUT + "intervals.csv", parse_dates=["t0"])
    n0 = len(iv)

    ss = pd.read_csv(RAW + "sunspot.csv", sep=";", header=None,
                     names=["year", "month", "day", "frac", "sunspot",
                            "std", "nobs", "prov"])
    ss["t0"] = pd.to_datetime(ss[["year", "month", "day"]])
    ss = ss[ss["sunspot"] >= 0][["t0", "sunspot"]]  # -1 means missing

    xr = pd.read_csv(RAW + "xray_daily.csv", parse_dates=["date"])
    xr = xr.rename(columns={"date": "t0"})

    d = iv.merge(ss, on="t0", how="inner")
    n1 = len(d)
    d = d.merge(xr, on="t0", how="inner")
    n2 = len(d)
    d["month"] = d["t0"].dt.month

    print(f"Intervals before merge:       {n0}")
    print(f"After sunspot merge:          {n1}")
    print(f"After X-ray merge:            {n2}")
    print(f"Dropped (no X-ray that day):  {n1 - n2}")
    print("\nIntervals per satellite:")
    print(d["bird"].value_counts().to_string())
    print("\nDropped intervals by year of start date:")
    lost = iv[~iv["t0"].isin(d["t0"])]
    print(lost["t0"].dt.year.value_counts().sort_index().to_string())

    d.to_csv(OUT + "intervals_with_weather.csv", index=False)
    print(f"\nSaved {len(d)} rows -> {OUT}intervals_with_weather.csv")
    print("(reference paper: 726 rows, 10 satellites)")


if __name__ == "__main__":
    main()