import gzip
import json
import unicodedata

import fastf1
import pandas as pd

from config import CACHE_DIR, DATA_DIR, OPENF1_DIR, OPENF1_JSON_DIR, RAW_DIR

OUT_FILE = DATA_DIR / "all_laps.parquet"

# FastF1's place names -> OpenF1's circuit names, so every track has one name
CIRCUIT_NAMES = {
    "Barcelona": "Catalunya", "Monaco": "Monte Carlo", "Montreal": "Montreal", "Budapest": "Hungaroring",
    "Marina Bay": "Singapore", "Sao Paulo": "Interlagos", "Yas Island": "Yas Marina Circuit",
    "Yas Marina": "Yas Marina Circuit", "Miami Gardens": "Miami", "Madrid": "Madring",
}


def plain(text):
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()   # "São" -> "Sao"


# Recipe 1: the FastF1 races, with the circuit name from FastF1's calendar
def fastf1_races():
    fastf1.Cache.enable_cache(CACHE_DIR)
    fastf1.set_log_level("WARNING")
    laps = pd.concat([pd.read_parquet(f) for f in sorted(RAW_DIR.glob("*.parquet"))], ignore_index=True)
    places = {}
    for year in laps["Year"].unique():
        schedule = fastf1.get_event_schedule(int(year), include_testing=False)
        for _, event in schedule.iterrows():
            place = plain(event["Location"])
            places[(year, event["RoundNumber"])] = CIRCUIT_NAMES.get(place, place)
    laps["Circuit"] = [places[key] for key in zip(laps["Year"], laps["Round"])]
    laps["Source"] = "FastF1"
    return laps


# Recipe 2: the OpenF1 races, with the circuit name OpenF1 gave the race
def openf1_races():
    tables = []
    for f in sorted(OPENF1_DIR.glob("*.parquet")):
        raw = json.loads(gzip.decompress((OPENF1_JSON_DIR / f"{f.stem}.json.gz").read_bytes()))
        tables.append(pd.read_parquet(f).assign(Circuit=raw["session"]["circuit_short_name"]))
    return pd.concat(tables, ignore_index=True)


# Recipe 3: one table with every lap. A race we have from both sources keeps the FastF1 copy.
def main():
    ff = fastf1_races()
    of = openf1_races()
    have = set(zip(ff["Year"], ff["Round"]))
    duplicate = [key in have for key in zip(of["Year"], of["Round"])]
    print("OpenF1 races also in FastF1 (FastF1 copy kept):",
          sorted({f"{y}_{r:02d}" for (y, r), d in zip(zip(of["Year"], of["Round"]), duplicate) if d}))
    laps = pd.concat([ff, of[[not d for d in duplicate]]], ignore_index=True)

    laps["RaceId"] = laps["Year"].astype(str) + "_" + laps["Round"].map("{:02d}".format)
    laps["TrackStatus"] = laps["TrackStatus"].fillna("").astype(str)
    laps["Compound"] = laps["Compound"].fillna("UNKNOWN").replace({"None": "UNKNOWN", "TEST_UNKNOWN": "UNKNOWN"})
    laps = laps.sort_values(["Year", "Round", "Driver", "LapNumber"]).reset_index(drop=True)
    assert not laps.duplicated(["RaceId", "Driver", "LapNumber"]).any(), "two rows for the same lap"

    laps.to_parquet(OUT_FILE, index=False)
    races = laps.groupby(["Year", "Source"]).agg(Races=("RaceId", "nunique"), Laps=("LapNumber", "size"))
    print(races.to_string())
    print("Total:", laps["RaceId"].nunique(), "races,", len(laps), "laps,", laps["Circuit"].nunique(), "circuits")
    print("saved", OUT_FILE)


if __name__ == "__main__":
    main()
