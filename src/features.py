import pandas as pd

from config import DATA_DIR

IN_FILE = DATA_DIR / "all_laps.parquet"
OUT_FILE = DATA_DIR / "features.parquet"
HORIZON = 3                                   # the target: does the driver pit in the next 3 laps?
DRY = ["SOFT", "MEDIUM", "HARD"]

# What the model sees. Everything is known at the end of the lap - nothing from the future.
CATEGORIES = ["Compound", "Circuit"]
FEATURES = CATEGORIES + [
    "TyreLife", "Stint", "FreshTyre", "LapsInStint", "DryCompoundsUsed", "FieldOnWetTyres",   # tyres
    "LapNumber", "LapsLeft", "RaceProgress",                                                  # race progress
    "PaceVsField", "TyreWear",                                                                # pace
    "Position", "GapAhead", "GapBehind", "GapToLeader",                                       # race situation
    "FieldStopsLast2", "AheadStoppedRecently", "BehindStoppedRecently",
    "SafetyCar", "VSC", "Yellow", "NeutralStarted", "LapsUnderNeutral",                       # track status
    "AirTemp", "TrackTemp", "Humidity", "Rainfall",                                           # weather
]
KEEP = ["RaceId", "Year", "Round", "EventName", "Driver", "Team", "Source"]
# The study's races: 2022-2025 to train and tune, and the first 17 races of 2026 to test (up to Singapore), as in
# the write-up. Races added later (the 2026 United States GP on) are used by the app and the race preview only, so
# running the training and evaluation scripts again gives the same numbers.
TEST_LAST_ROUND = 17


def in_study(rows):
    return (rows["Year"] < 2026) | (rows["Round"] <= TEST_LAST_ROUND)


# Recipe 1: which laps end with a real pit stop?
# Real = the tyres really change (another compound, or the tyre age starts again), the car is
# in the pit lane for less than 90 seconds, and it isn't under a red flag. This leaves out
# drive-through penalties, the field driving through the pit lane behind the safety car
# (Australia 2025, Canada 2025) and cars waiting in the pit lane (Monaco 2026).
def mark_stops(laps, by_driver):
    life, next_life = laps["TyreLife"], by_driver["TyreLife"].shift(-1)
    new_tyres = (by_driver["Compound"].shift(-1) != laps["Compound"]) | (next_life <= life)
    unknown_age = life.isna() | next_life.isna()                       # then trust the stint number
    new_tyres |= unknown_age & (by_driver["Stint"].shift(-1) > laps["Stint"])
    lane_time = by_driver["PitOutTime"].shift(-1) - laps["PitInTime"]
    laps["IsStop"] = (laps["PitInTime"].notna() & by_driver["LapNumber"].shift(-1).notna() & new_tyres
                      & ~(lane_time >= 90) & ~laps["TrackStatus"].str.contains("5"))
    by_driver = laps.groupby(["RaceId", "Driver"])
    target = pd.Series(False, index=laps.index)
    for k in range(1, HORIZON + 1):                          # a stop on lap +1, +2 or +3
        ahead = by_driver["LapNumber"].shift(-k) - laps["LapNumber"] <= HORIZON
        target |= by_driver["IsStop"].shift(-k, fill_value=False) & ahead
    laps["PitSoon"] = target.astype(int)


# Recipe 2: tyre features
def tyre_features(laps, by_driver):
    laps["FreshTyre"] = laps["FreshTyre"].astype(int)
    laps["LapsInStint"] = laps.groupby(["RaceId", "Driver", "Stint"]).cumcount() + 1
    used = [(laps["Compound"] == c).groupby([laps["RaceId"], laps["Driver"]]).cummax() for c in DRY]
    laps["DryCompoundsUsed"] = sum(u.astype(int) for u in used)        # F1 rule: a dry race needs 2
    wet = laps["Compound"].isin(["INTERMEDIATE", "WET"])
    laps["FieldOnWetTyres"] = wet.groupby([laps["RaceId"], laps["LapNumber"]]).transform("mean")


# Recipe 3: race progress and pace
def pace_features(laps):
    laps["LapsLeft"] = laps["TotalLaps"] - laps["LapNumber"]
    laps["RaceProgress"] = laps["LapNumber"] / laps["TotalLaps"]
    field = laps.groupby(["RaceId", "LapNumber"])["LapTime"].transform("median")
    laps["PaceVsField"] = laps["LapTime"] / field - 1                  # +0.02 = 2% slower than the field

    # Tyre wear: the last 3 clean laps compared with the best clean lap of this stint.
    # Clean = green flag, not lap 1, not the lap out of the pits, and within 15% of the field
    # (so a damaged car crawling back to the pits doesn't count).
    clean = ((laps["TrackStatus"] == "1") & (laps["LapNumber"] > 1) & laps["PitOutTime"].isna()
             & (laps["PaceVsField"] < 0.15))
    stint = ["RaceId", "Driver", "Stint"]
    by_stint = laps[clean].groupby(stint)["LapTime"]
    last3 = by_stint.transform(lambda s: s.rolling(3, min_periods=1).mean())
    laps["TyreWear"] = last3 - by_stint.cummin()                        # seconds slower than the stint's best
    laps["TyreWear"] = laps.groupby(stint)["TyreWear"].ffill()          # unclean laps keep the last value


# Recipe 4: gaps and what the cars around you did
def race_features(laps, by_driver):
    order = laps.sort_values(["RaceId", "LapNumber", "Position"])
    by_lap = order.groupby(["RaceId", "LapNumber"])["Time"]
    placed = laps["Position"].notna()
    # Seconds to the car in front / behind. Over 60 s they aren't racing each other, so 60 is the cap.
    laps["GapAhead"] = by_lap.diff().where(placed).clip(0, 60)
    laps["GapBehind"] = (-by_lap.diff(-1)).where(placed).clip(0, 60)
    laps["GapToLeader"] = (laps["Time"] - laps.groupby(["RaceId", "LapNumber"])["Time"].transform("min")).clip(lower=0)

    stops = laps.groupby(["RaceId", "LapNumber"])["IsStop"].sum()
    last2 = stops.add(stops.groupby(level=0).shift(1), fill_value=0)   # this lap + the lap before
    # Did the car just ahead / just behind (2 laps ago) stop in the last 2 laps? (undercut)
    laps["RecentStop"] = laps["IsStop"] | by_driver["IsStop"].shift(1, fill_value=False)
    field = last2.reindex(pd.MultiIndex.from_frame(laps[["RaceId", "LapNumber"]])).values
    laps["FieldStopsLast2"] = field - laps["RecentStop"].astype(int)    # other cars only
    who = laps.dropna(subset=["Position"]).set_index(["RaceId", "LapNumber", "Position"])["Driver"]
    who = who[~who.index.duplicated()]
    recent = laps.set_index(["RaceId", "LapNumber", "Driver"])["RecentStop"]
    position_before = by_driver["Position"].shift(2)
    for name, step in [("AheadStoppedRecently", -1), ("BehindStoppedRecently", 1)]:
        key = pd.MultiIndex.from_arrays([laps["RaceId"], laps["LapNumber"] - 2, position_before + step])
        rival = pd.Series(who.reindex(key).values, index=laps.index)
        key = pd.MultiIndex.from_arrays([laps["RaceId"], laps["LapNumber"], rival])
        laps[name] = recent.reindex(key).eq(True).astype(int).values


# Recipe 5: track status and weather as 0/1 numbers
def status_features(laps):
    status = laps["TrackStatus"]
    laps["SafetyCar"] = status.str.contains("4").astype(int)
    laps["VSC"] = status.str.contains("[67]").astype(int)
    laps["Yellow"] = status.str.contains("2").astype(int)
    laps["Rainfall"] = laps["Rainfall"].astype(int)

    # A safety car or VSC that has just come out is the cheapest moment to pit
    neutral = (laps["SafetyCar"] == 1) | (laps["VSC"] == 1)
    by_driver = neutral.groupby([laps["RaceId"], laps["Driver"]])
    laps["NeutralStarted"] = (neutral & ~by_driver.shift(1, fill_value=False)).astype(int)
    block = (~neutral).groupby([laps["RaceId"], laps["Driver"]]).cumsum()      # new number after each green lap
    laps["LapsUnderNeutral"] = neutral.groupby([laps["RaceId"], laps["Driver"], block]).cumsum().where(neutral, 0)


# Every lap with every feature, in-laps included (the tyre model uses the in-laps).
# The F1 25 part passes its own laps (same columns), so the game gets exactly the same features.
def build(laps=None):
    laps = pd.read_parquet(IN_FILE) if laps is None else laps
    laps = laps.sort_values(["RaceId", "Driver", "LapNumber"]).reset_index(drop=True)
    by_driver = laps.groupby(["RaceId", "Driver"])
    mark_stops(laps, by_driver)
    by_driver = laps.groupby(["RaceId", "Driver"])                     # again, now with IsStop
    tyre_features(laps, by_driver)
    pace_features(laps)
    race_features(laps, by_driver)
    status_features(laps)
    return laps


def main():
    laps = build()
    stops = laps.groupby("RaceId")["IsStop"].sum()
    print("Pit stops:", int(stops.sum()), "- per race: median", stops.median(), ", most", int(stops.max()),
          f"({stops.idxmax()})")

    # The in-lap itself is left out: when it ends, the car is already in the pit lane
    rows = laps[laps["PitInTime"].isna()].copy()
    for col in CATEGORIES:
        rows[col] = rows[col].astype("category")
    rows = rows[KEEP + FEATURES + ["IsStop", "PitSoon"]]
    rows.to_parquet(OUT_FILE, index=False)

    split = rows["Year"].map(lambda y: "test (2026)" if y == 2026 else "tune (2025)" if y == 2025 else "train (2022-24)")
    split = split.where(in_study(rows), "after the study (app only)")
    print(rows.groupby(split).agg(Races=("RaceId", "nunique"), Rows=("PitSoon", "size"),
                                  PitSoon=("PitSoon", "mean")).round(3).to_string())
    empty = rows[FEATURES].isna().mean().mul(100).round(1)
    print("Empty cells (%):", empty[empty > 0].to_dict())
    print("saved", OUT_FILE, "-", len(FEATURES), "features")


if __name__ == "__main__":
    main()
