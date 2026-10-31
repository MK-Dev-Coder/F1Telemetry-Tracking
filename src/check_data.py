import pandas as pd

from config import CACHE_DIR, RAW_DIR

# 1. The races you saved as tables (one file per race in data/raw)
files = sorted(RAW_DIR.glob("*.parquet"))
if not files:
    raise SystemExit(f"No race files found in {RAW_DIR}")
laps = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
races = laps.groupby(["Year", "Round", "EventName"]).size().reset_index(name="Laps")

print("Saved races:", len(races), "-", len(laps), "laps in total")
print(races.groupby("Year").size().to_string(header=False))    # races per year
print()
print(races.to_string(index=False))                            # every race, with its number of laps

# 2. Races FastF1 downloaded into its cache, but that are not in data/raw yet
saved = set(zip(races["Year"], races["EventName"]))
missing = []
for f in sorted(CACHE_DIR.glob("*/*/*_Race/_extended_timing_data.ff1pkl")):   # = lap data of one race
    year = int(f.parts[-4])
    event = f.parts[-3][11:].replace("_", " ")                  # "2023-03-05_Bahrain_Grand_Prix" -> "Bahrain Grand Prix"
    if (year, event) not in saved:
        missing.append(f"{year} {event}")

print()
print("In the FastF1 cache but not saved yet:", len(missing))
for race in missing:
    print("  ", race)
