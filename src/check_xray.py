"""List which GOES satellites have XRS files for each year (Jan and Jul checked)."""
import re
import urllib.request

ROOT = ("https://www.ncei.noaa.gov/data/goes-space-environment-monitor/"
        "access/avg/{y}/{m:02d}/")


def listing(url):
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            html = r.read().decode("utf-8", errors="ignore")
    except Exception:
        return []
    return re.findall(r'href="([^"?/][^"]*)"', html)


for year in range(1978, 1994):
    found = {}
    for month in (1, 7):
        base = ROOT.format(y=year, m=month)
        for sat in [x for x in listing(base) if x.endswith("/")]:
            if sat.startswith("."):
                continue
            files = listing(base + sat + "csv/")
            xrs = [f for f in files if "_xrs_" in f]
            if xrs:
                found.setdefault(sat.strip("/"), []).append(
                    sorted(set(f.split("_xrs_")[1].split("_19")[0] for f in xrs)))
    print(year, found if found else "NO XRS FILES")