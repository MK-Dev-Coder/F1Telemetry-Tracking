import pandas as pd

from config import DATA_DIR, OPENF1_DIR, RAW_DIR

KEY = ["Driver", "LapNumber"]
SAME_COLUMNS = ["Team", "Stint", "Compound", "TyreLife", "FreshTyre", "Position", "TrackStatus", "Rainfall"]


def share(mask):
    return round(100 * mask.mean(), 1)                      # % of laps


# Compare one race that we have from both FastF1 and OpenF1, lap by lap
def compare(name):
    ff = pd.read_parquet(RAW_DIR / name)
    of = pd.read_parquet(OPENF1_DIR / name)
    both = ff.merge(of, on=KEY, how="outer", suffixes=("_ff", "_of"), indicator=True)
    m = both[both["_merge"] == "both"].copy()
    timed = m["LapTime_ff"].notna() & m["LapTime_of"].notna()
    gap = (m["LapTime_ff"] - m["LapTime_of"]).abs()[timed]
    offset = (m["Time_of"] - m["Time_ff"]).median()          # the two sources start their clock differently
    neutral = lambda s: s.fillna("").str.replace("[12]", "", regex=True)   # keep 4, 5, 6, 7 only

    result = {
        "Race": f"{ff['Year'].iloc[0]} {ff['EventName'].iloc[0]}",
        "Laps FastF1": len(ff), "Laps OpenF1": len(of), "Laps in both": len(m),
        "LapTime exact (%)": share(gap < 0.001), "LapTime within 0.1 s (%)": share(gap < 0.1),
        "LapTime biggest gap (s)": round(gap.max(), 3),
    }
    for col in SAME_COLUMNS:
        a, b = m[f"{col}_ff"], m[f"{col}_of"]
        result[f"{col} same (%)"] = share((a == b) | (a.isna() & b.isna()))
    result["SC/VSC/red flag same (%)"] = share(neutral(m["TrackStatus_ff"]) == neutral(m["TrackStatus_of"]))
    for col in ["AirTemp", "TrackTemp"]:
        result[f"{col} average gap"] = round((m[f"{col}_ff"] - m[f"{col}_of"]).abs().mean(), 2)
    for col in ["PitInTime", "PitOutTime"]:
        laps_ff = set(map(tuple, m.loc[m[f"{col}_ff"].notna(), KEY].values))
        laps_of = set(map(tuple, m.loc[m[f"{col}_of"].notna(), KEY].values))
        result[f"{col} laps FastF1 / OpenF1 / both"] = f"{len(laps_ff)} / {len(laps_of)} / {len(laps_ff & laps_of)}"
        result[f"{col} average gap (s)"] = round((m[f"{col}_of"] - offset - m[f"{col}_ff"]).abs().mean(), 2)
    result["Time offset (s)"] = round(offset, 1)
    result["Time offset spread (s)"] = round((m["Time_of"] - m["Time_ff"] - offset).abs().quantile(0.95), 3)

    # Every lap where something important is different, for a closer look
    diff = pd.Series(False, index=m.index)
    for col in ["Compound", "Stint", "TyreLife", "Position", "TrackStatus"]:
        diff |= (m[f"{col}_ff"] != m[f"{col}_of"]) & ~(m[f"{col}_ff"].isna() & m[f"{col}_of"].isna())
    diff |= (m["LapTime_ff"] - m["LapTime_of"]).abs() >= 0.1
    cols = KEY + [f"{c}_{s}" for c in ["LapTime", "Compound", "Stint", "TyreLife", "Position", "TrackStatus"]
                  for s in ("ff", "of")]
    laps = m.loc[diff, cols].assign(Race=result["Race"])
    return result, laps


def main():
    names = sorted(p.name for p in OPENF1_DIR.glob("*.parquet") if (RAW_DIR / p.name).exists())
    results, laps = zip(*[compare(n) for n in names])
    summary = pd.DataFrame(results).set_index("Race").T
    print(summary.to_string())
    summary.to_csv(DATA_DIR / "source_comparison.csv")
    pd.concat(laps).to_csv(DATA_DIR / "source_differences.csv", index=False)
    print("\nsaved data/source_comparison.csv and data/source_differences.csv")


if __name__ == "__main__":
    main()
