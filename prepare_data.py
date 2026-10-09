"""
prepare_data.py - convert the raw HMIS (Gujarat) long-format file into the
wide, one-row-per-(district, period) schema required by the assignment.

Raw source : Dataful / HMIS "District and Year-wise data ... for Gujarat"
Output     : data/districts.csv with columns
             period, district, anc_coverage, institutional_delivery,
             immunization, high_risk_cases

Indicator definitions (documented in README):
  anc_coverage           = PW with 4+ ANC check-ups / PW registered for ANC * 100
  institutional_delivery = institutional deliveries / total deliveries * 100
                           (total = institutional + home SBA + home non-SBA)
  immunization           = children given Pentavalent-3 / live births (M + F) * 100
  high_risk_cases        = PW hypertension detected + severe anaemia (Hb<7) treated
                           + eclampsia cases managed   (raw count)

Why 2017-18 onward by default: before 2017-18 HMIS measured "3 ANC check-ups"
and later "4+ ANC check-ups". Mixing them would create a fake trend at the
2016-17 -> 2017-18 boundary. Use --start to override at your own risk.

Usage:
  python prepare_data.py --raw path/to/raw.csv --out data/districts.csv
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

TOTAL_FACILITY = "Public and Private"   # prefix of the combined facility label


def pick(df: pd.DataFrame, pattern: str) -> pd.Series:
    """Sum values of all indicators matching regex `pattern`, per (period, district).

    Regex matching (not exact names) makes the script robust to small wording
    changes between financial years, e.g. 'Hb<7' vs 'Hb < 7'.
    """
    m = df[df["indicator"].str.contains(pattern, flags=re.I, regex=True, na=False)]
    if m.empty:
        raise ValueError(f"No rows matched pattern: {pattern!r}")
    return m.groupby(["period", "district"])["value"].sum(min_count=1)


def build(raw_path: str, start: str = "2017-18") -> pd.DataFrame:
    raw = pd.read_csv(raw_path, low_memory=False)
    raw = raw.rename(columns={"fiscal_year": "period",
                              "district_as_per_lgd": "district"})
    raw = raw[raw["period"] >= start]
    # Keep combined (public + private) totals only to avoid double counting.
    raw = raw[raw["facility_category"].str.startswith(TOTAL_FACILITY)
              & (raw["category"] == "Total")]

    registered = pick(raw, r"^Total number of pregnant women registered for ANC$")
    anc4 = pick(raw, r"^Number of PW received 4 or more ANC check ups$")
    inst = pick(raw, r"^Number of Institutional Deliveries conducted")
    home = pick(raw, r"^Number of Home Deliveries attended by")
    penta3 = pick(raw, r"^Child immunisation - Pentavalent 3$")
    births = pick(raw, r"^Live Birth - (?:Male|Female)$")
    htn = pick(raw, r"^New cases of PW with hypertension detected$")
    anaemia = pick(raw, r"^Number of PW having severe anaemia \(Hb ?< ?7\) treated$")
    eclampsia = pick(raw, r"^Number of Eclampsia cases managed during delivery$")

    out = pd.DataFrame({
        "anc_coverage": anc4 / registered * 100,
        "institutional_delivery": inst / (inst + home) * 100,
        "immunization": penta3 / births * 100,
        "high_risk_cases": htn.add(anaemia, fill_value=0).add(eclampsia, fill_value=0),
    }).reset_index()

    out["high_risk_cases"] = out["high_risk_cases"].round().astype("Int64")
    for c in ["anc_coverage", "institutional_delivery", "immunization"]:
        out[c] = out[c].round(1)
    return out.sort_values(["district", "period"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="raw HMIS CSV (long format)")
    ap.add_argument("--out", default="data/districts.csv")
    ap.add_argument("--start", default="2017-18",
                    help="first fiscal year to include (default 2017-18)")
    a = ap.parse_args()

    df = build(a.raw, a.start)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"Wrote {len(df)} rows -> {a.out}")
    print(df.head(), "\n")
    print("Missing values per column:\n", df.isna().sum())
    if a.start < "2017-18":
        print("\nWARNING: ANC definition changes at 2017-18 (3+ -> 4+ check-ups); "
              "trends across that boundary are not comparable.", file=sys.stderr)


if __name__ == "__main__":
    main()
