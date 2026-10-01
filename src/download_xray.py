"""Download GOES 1-minute X-ray files from NCEI and build a daily-mean table.

For each month we try the satellite the reference pipeline used first, then any
other GOES satellite that has XRS data. A coverage report is saved so the
differences from the reference can be documented.
"""
import calendar
import io
import os
import urllib.request
import urllib.error

import pandas as pd

BASE = ("https://www.ncei.noaa.gov/data/goes-space-environment-monitor/"
        "access/avg/{year}/{month:02d}/goes{sat}/csv/"
        "g{sat}_xrs_1m_3s_{year}{month:02d}01_{year}{month:02d}{last:02d}.csv")

YEARS = range(1978, 1994)
ALL_SATS = ["02", "03", "05", "06", "07"]

# Satellite the reference pipeline used in each year (tried first)
def reference_sat(year):
    if year <= 1982:
        return "02"
    if year <= 1986:
        return "05"
    return "06"

# The reference code skipped February (its month list omits '02').
# Set to False to include February as well.
SKIP_FEBRUARY = True

CACHE = "data/raw/xray_monthly/"
OUT = "data/raw/xray_daily.csv"
COVERAGE = "data/raw/xray_coverage.csv"
MISSING = -99999


def fetch(url, path):
    """Return file text (from cache or download), or None if not available."""
    if os.path.exists(path):
        return open(path, encoding="utf-8", errors="ignore").read()
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            text = r.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError:
        return None
    except urllib.error.URLError as e:
        print(f"  network problem: {e}")
        return None
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def parse(text):
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip().lower() == "data:")
    df = pd.read_csv(io.StringIO("\n".join(lines[start + 1:])), skipinitialspace=True)
    df = df[["time_tag", "xl"]]
    df["time_tag"] = pd.to_datetime(df["time_tag"])
    df = df[df["xl"] != MISSING].dropna()
    return df.set_index("time_tag").resample("D").mean().dropna()


def main():
    os.makedirs(CACHE, exist_ok=True)
    frames, cover = [], []
    for year in YEARS:
        order = [reference_sat(year)] + [s for s in ALL_SATS if s != reference_sat(year)]
        for month in range(1, 13):
            if SKIP_FEBRUARY and month == 2:
                continue
            last = calendar.monthrange(year, month)[1]
            used = None
            for sat in order:
                url = BASE.format(sat=sat, year=year, month=month, last=last)
                text = fetch(url, f"{CACHE}g{sat}_{year}{month:02d}.csv")
                if text:
                    try:
                        frames.append(parse(text))
                        used = sat
                        break
                    except StopIteration:
                        continue
            cover.append({"year": year, "month": month, "satellite_used": used,
                          "reference_sat": reference_sat(year)})
            print(f"{year}-{month:02d}: {'GOES-' + used if used else 'NO DATA'}")

    daily = pd.concat(frames).sort_index()
    daily = daily[~daily.index.duplicated(keep="first")]
    daily.index.name = "date"
    daily = daily.rename(columns={"xl": "xray_flux"})
    daily.to_csv(OUT)

    cov = pd.DataFrame(cover)
    cov.to_csv(COVERAGE, index=False)
    print("\n--- Coverage summary ---")
    print(f"Months with data: {cov['satellite_used'].notna().sum()} of {len(cov)}")
    print("Months with NO data:")
    print(cov[cov["satellite_used"].isna()].groupby("year").size().to_string())
    print(f"\nSaved {len(daily)} daily rows -> {OUT}")


if __name__ == "__main__":
    main()