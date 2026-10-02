"""Stage 3: attach perigee, inclination (Celestrak) and mass (data/raw/mass.csv)
to every interval and write the final 6-feature table."""
import sys

import pandas as pd

RAW = "data/raw/"
OUT = "data/processed/"

# Anomaly-file name -> Celestrak OBJECT_ID (identified from the satcat output)
CELESTRAK_ID = {
    "goes 4": "1980-074A", "goes 5": "1981-049A", "goes 6": "1983-041A",
    "meteosat 1": "1977-108A", "meteosat 2": "1981-057A",
    "marecs a": "1981-122A", "noaa 11": "1988-089A",
    "gps 5113": "1978-093A", "gps 5114": "1978-112A",
    "gps 5118": "1980-032A", "gps 9794": "1983-072A",
}


def main():
    d = pd.read_csv(OUT + "intervals_with_weather.csv", parse_dates=["t0"])

    sc = pd.read_csv(RAW + "satcat.csv")
    ids = pd.DataFrame({"bird": list(CELESTRAK_ID), "OBJECT_ID": list(CELESTRAK_ID.values())})
    attrs = ids.merge(sc[["OBJECT_ID", "PERIGEE", "INCLINATION"]], on="OBJECT_ID", how="left")
    attrs = attrs.rename(columns={"PERIGEE": "perigee_km", "INCLINATION": "inclination_deg"})

    mass = pd.read_csv(RAW + "mass.csv")
    mass["bird"] = mass["bird"].str.strip().str.lower()
    missing = mass[mass["mass_kg"].isna()]["bird"].tolist()
    if missing:
        sys.exit(f"mass_kg is empty for: {missing}. Fill data/raw/mass.csv first.")
    attrs = attrs.merge(mass[["bird", "mass_kg"]], on="bird", how="left")
    attrs.to_csv(OUT + "satellite_attributes.csv", index=False)

    out = d.merge(attrs, on="bird", how="left")
    cols = ["bird", "t0", "month", "sunspot", "xray_flux",
            "mass_kg", "perigee_km", "inclination_deg", "tte"]
    out = out[cols]
    assert out.notna().all().all(), "NaNs in final table: check satellite names"
    out.to_csv(OUT + "features.csv", index=False)

    print(f"Final table: {out.shape[0]} rows x {out.shape[1] - 3} features, "
          f"{out['bird'].nunique()} satellites")
    print(out.describe().T[["min", "mean", "max"]].to_string())


if __name__ == "__main__":
    main()