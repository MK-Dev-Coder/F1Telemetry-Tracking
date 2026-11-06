import gzip
import json
import os
import sys
import time

import fastf1
import pandas as pd
import requests

from collect_data import LAP_COLUMNS, WEATHER_COLUMNS
from config import CACHE_DIR, OPENF1_DIR, OPENF1_JSON_DIR, SEASONS

OPENF1 = "https://api.openf1.org/v1"
ENDPOINTS = ["drivers", "laps", "stints", "pit", "position", "weather", "race_control"]
FIRST_YEAR = 2023                    # OpenF1 has no data before 2023
PAUSE = 2.1                          # seconds between requests (free users: 30 requests a minute)
TYPICAL_LANE = 22.0                  # seconds in the pit lane, when OpenF1 doesn't say
COLUMNS = LAP_COLUMNS + WEATHER_COLUMNS + ["Year", "Round", "EventName", "TotalLaps", "Source"]
QUICK = os.environ.get("F1_OPENF1_QUICK") == "1"   # set by newraces.py: give up and retry next time
                                                    # instead of waiting for hours


# One request to OpenF1, waiting and retrying if it says no
def get(endpoint, **params):
    for attempt in range(3 if QUICK else 30):
        try:
            answer = requests.get(f"{OPENF1}/{endpoint}", params=params, timeout=60)
        except requests.RequestException:
            print("   no answer from OpenF1 - waiting 1 minute...")
            time.sleep(60)
            continue
        time.sleep(PAUSE)
        if answer.status_code == 200:
            return answer.json()
        if answer.status_code == 404:                       # OpenF1 says "nothing found"
            return []
        if answer.status_code == 429:
            print("   too many requests - waiting 30 seconds...")
            time.sleep(30)
        elif answer.status_code == 401:                     # closed to free users during a live session
            if QUICK:
                raise RuntimeError("OpenF1 is closed while an F1 session is live")
            print("   OpenF1 is closed while an F1 session is live - waiting 10 minutes...")
            time.sleep(10 * 60)
        else:
            print("   OpenF1 says", answer.status_code, "- waiting 1 minute...")
            time.sleep(60)
    raise RuntimeError(f"OpenF1 did not answer for {endpoint} {params}")


# Finished races of one year, with FastF1's round number and name
def rounds_by_date(year):
    """FastF1's schedule of one year: race day -> the event (its RoundNumber and EventName)."""
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    return {row["Session5DateUtc"].date(): row for _, row in schedule.iterrows() if pd.notna(row["Session5DateUtc"])}


def round_for(race, by_date):
    """FastF1's round number and name for an OpenF1 race (matched on its race day, in UTC), or None."""
    event = by_date.get(pd.Timestamp(race["date_start"]).tz_convert("UTC").date())
    return None if event is None else (int(event["RoundNumber"]), event["EventName"])


def race_list(year):
    by_date = rounds_by_date(year)
    now = pd.Timestamp.now(tz="UTC")
    races = []
    for race in get("sessions", year=year, session_name="Race"):
        if race.get("is_cancelled") or pd.Timestamp(race["date_end"]) > now:
            continue                                        # cancelled, or not raced yet
        found = round_for(race, by_date)
        if found is None:
            print("   no FastF1 round for", year, race["country_name"], "- skipped")
            continue
        races.append((*found, race))
    return sorted(races, key=lambda r: r[0])


def saved_race(year, rnd):
    """A race whose OpenF1 data is already saved (newraces.py saves the ones the cloud fetched): its round, name
    and session from the file and FastF1's schedule, without asking OpenF1 again. None if it isn't saved."""
    file = OPENF1_JSON_DIR / f"{year}_{rnd:02d}.json.gz"
    if not file.exists():
        return None
    race = json.loads(gzip.decompress(file.read_bytes()))["session"]
    found = round_for(race, rounds_by_date(year))
    return (*found, race) if found and found[0] == rnd else None


def races_of(year, only=None):
    """Races to build: one saved race straight from its file, or else the year's finished races (from OpenF1)."""
    saved = saved_race(*only) if only else None
    return [saved] if saved else race_list(year)


# Download everything for one race (or load it if we already have it)
def download_race(race, json_file):
    if json_file.exists():
        return json.loads(gzip.decompress(json_file.read_bytes()))
    raw = {"session": race}
    for name in ENDPOINTS:
        raw[name] = get(name, session_key=race["session_key"])
    json_file.write_bytes(gzip.compress(json.dumps(raw).encode("utf-8")))
    return raw


def times(column):
    return pd.to_datetime(column, format="ISO8601", utc=True)


# Each driver's tyres. OpenF1 gets the stints wrong for drivers who start from the pit lane, so when the counts match
# we take the stint changes from the pit stops instead.
def tyre_table(stints, pit_in_laps, laps):
    rows = []
    for number, driver_laps in laps.groupby("driver_number"):
        own = stints[stints["driver_number"] == number].sort_values("stint_number")
        pits = sorted(set(pit_in_laps.get(number, [])))
        if len(own) == len(pits) + 1:
            starts = [1] + [p + 1 for p in pits]
            ends = [p for p in pits] + [10_000]
        else:                                               # e.g. a tyre change under a red flag
            starts, ends = list(own["lap_start"]), list(own["lap_end"])
            for k in range(1, len(starts)):                 # a lap in two stints stays in the first one
                if pd.notna(starts[k]) and pd.notna(ends[k - 1]) and starts[k] <= ends[k - 1]:
                    starts[k] = ends[k - 1] + 1
        for k, (_, stint) in enumerate(own.iterrows()):
            first, last = starts[k], ends[k]
            for lap in driver_laps["lap_number"]:
                if pd.notna(first) and pd.notna(last) and first <= lap <= last:
                    age = stint["tyre_age_at_start"] if pd.notna(stint["tyre_age_at_start"]) else 0
                    rows.append({"driver_number": number, "lap_number": lap, "Stint": k + 1,
                                 "Compound": stint["compound"], "TyreLife": age + lap - first + 1,
                                 "FreshTyre": age == 0})
    tyres = pd.DataFrame(rows, columns=["driver_number", "lap_number", "Stint", "Compound",
                                        "TyreLife", "FreshTyre"])
    return tyres.drop_duplicates(["driver_number", "lap_number"])     # one tyre row per lap, always


# Track status code at each moment of the race (FastF1's codes: 1 green, 2 yellow, 4 safety car, 5 red flag, 6 virtual
# safety car, 7 VSC ending)
def status_timeline(race_control):
    timeline, safety, yellows = [], None, set()
    for msg in sorted(race_control, key=lambda m: m["date"]):
        text, flag, scope = (msg.get("message") or "").upper(), msg.get("flag"), msg.get("scope")
        if msg.get("category") == "SafetyCar":
            if "VIRTUAL" in text or "VSC" in text:
                safety = "7" if "ENDING" in text else "6"
            elif "DEPLOYED" in text:
                safety = "4"
        elif flag == "RED" or text.startswith("RED FLAG"):  # in 2026 it's "RED FLAG - RACE SUSPENDED" with no flag
            safety = "5"
        elif flag in ("GREEN", "CLEAR") and scope == "Track":
            safety, yellows = None, set()
        elif flag in ("YELLOW", "DOUBLE YELLOW"):
            yellows.add(msg.get("sector") if scope == "Sector" else "track")
        elif flag == "CLEAR" and scope == "Sector":
            yellows.discard(msg.get("sector"))
        else:
            continue
        timeline.append((pd.Timestamp(msg["date"]), safety or ("2" if yellows else "1")))
    return timeline


def lap_status(start, end, timeline):
    codes = ["1"]
    for moment, code in timeline:
        if pd.notna(start) and moment <= start:
            codes = [code]                                  # the status when the lap started
        elif pd.notna(end) and moment < end:
            codes.append(code)                              # and every change during the lap
    return "".join(dict.fromkeys(codes))                    # e.g. "671" = VSC, ending, green


# Turn one race into a table with the same columns as the FastF1 files
def race_table(raw, year, rnd, event_name):
    t0 = pd.Timestamp(raw["session"]["date_start"])
    drivers = pd.DataFrame(raw["drivers"]).drop_duplicates("driver_number")
    laps = pd.DataFrame(raw["laps"]).drop_duplicates(["driver_number", "lap_number"])
    laps = laps.sort_values(["driver_number", "lap_number"])
    laps["start"] = times(laps["date_start"])
    laps["end"] = laps["start"] + pd.to_timedelta(laps["lap_duration"], unit="s")
    next_start = laps.groupby("driver_number")["start"].shift(-1)
    laps["end"] = laps["end"].fillna(next_start)            # no lap time? use the next lap's start
    laps["start"] = laps["start"].fillna(laps.groupby("driver_number")["end"].shift(1))

    # Pit stops: OpenF1's "date" is when the car leaves the pit lane
    pit = pd.DataFrame(raw["pit"], columns=["driver_number", "lap_number", "date", "lane_duration",
                                            "pit_duration"])
    if pit.empty:                       # no pit data for a few early 2023 races: use the
        outs = laps[laps["is_pit_out_lap"] == True]         # out-laps instead (the stop was the lap before)
        pit = pd.DataFrame({"driver_number": outs["driver_number"], "lap_number": outs["lap_number"] - 1,
                            "date": outs["date_start"], "lane_duration": TYPICAL_LANE, "pit_duration": None})
        pit = pit[(pit["lap_number"] >= 1) & pit["date"].notna()]
    pit["out"] = times(pit["date"])
    lane = pit["lane_duration"].fillna(pit["pit_duration"]).fillna(pit["lane_duration"].median())
    pit["in"] = pit["out"] - pd.to_timedelta(lane, unit="s")
    seconds = lambda t: (t - t0).dt.total_seconds()
    pit_in = pit.assign(PitInTime=seconds(pit["in"]))[["driver_number", "lap_number", "PitInTime"]]
    pit_out = pit.assign(PitOutTime=seconds(pit["out"]), lap_number=pit["lap_number"] + 1)
    pit_in_laps = pit.groupby("driver_number")["lap_number"].apply(list).to_dict()

    tyres = tyre_table(pd.DataFrame(raw["stints"]), pit_in_laps, laps)
    table = (laps.merge(drivers[["driver_number", "name_acronym", "team_name"]], on="driver_number", how="left")
                 .merge(tyres, on=["driver_number", "lap_number"], how="left")
                 .merge(pit_in.drop_duplicates(["driver_number", "lap_number"]), on=["driver_number", "lap_number"], how="left")
                 .merge(pit_out[["driver_number", "lap_number", "PitOutTime"]]
                        .drop_duplicates(["driver_number", "lap_number"]), on=["driver_number", "lap_number"], how="left"))

    # Position at the end of each lap = the last position update before the lap ended
    position = pd.DataFrame(raw["position"], columns=["date", "driver_number", "position"])
    position["end"] = times(position["date"])
    known = table[table["end"].notna()].sort_values("end")
    known = pd.merge_asof(known, position.sort_values("end")[["end", "driver_number", "position"]],
                          on="end", by="driver_number", direction="backward")
    table = table.merge(known[["driver_number", "lap_number", "position"]], on=["driver_number", "lap_number"], how="left")

    # Weather when the lap started (one reading per minute)
    weather = pd.DataFrame(raw["weather"], columns=["date", "air_temperature", "track_temperature",
                                                    "humidity", "rainfall"])
    weather["start"] = times(weather["date"])
    weather = weather.sort_values("start").drop(columns="date")
    known = table[table["start"].notna()].sort_values("start")
    known = pd.merge_asof(known, weather, on="start", direction="nearest")
    table = table.merge(known[["driver_number", "lap_number", "air_temperature", "track_temperature",
                               "humidity", "rainfall"]], on=["driver_number", "lap_number"], how="left")

    timeline = status_timeline(raw["race_control"])
    table["TrackStatus"] = [lap_status(s, e, timeline) for s, e in zip(table["start"], table["end"])]

    table = table.rename(columns={"name_acronym": "Driver", "team_name": "Team", "lap_number": "LapNumber",
                                  "lap_duration": "LapTime", "position": "Position",
                                  "air_temperature": "AirTemp", "track_temperature": "TrackTemp",
                                  "humidity": "Humidity", "rainfall": "Rainfall"})
    table["Time"] = seconds(table["end"])                   # seconds since the race start
    table["Rainfall"] = table["Rainfall"].fillna(0).astype(bool)
    table["FreshTyre"] = table["FreshTyre"].eq(True)
    for col in ["LapNumber", "Stint", "TyreLife", "Position"]:
        table[col] = table[col].astype(float)
    table["Year"] = year
    table["Round"] = rnd
    table["EventName"] = event_name
    timed = table.loc[table["LapTime"].notna(), "LapNumber"]  # OpenF1 doesn't give the planned laps, and some races
    table["TotalLaps"] = int(timed.max() if len(timed) else table["LapNumber"].max())   # have an untimed lap after the flag
    table["Source"] = "OpenF1"
    table = table.drop_duplicates(["driver_number", "LapNumber"])   # never two rows for one lap
    return table[COLUMNS].sort_values(["Driver", "LapNumber"]).reset_index(drop=True)


# Every finished race from 2023 on, skipping the ones already saved.
# Run "python src/collect_openf1.py 2026 17" to (re)build one race, or "python src/collect_openf1.py rebuild" to
# rebuild every race from the saved json files.
def main():
    CACHE_DIR.mkdir(exist_ok=True)
    OPENF1_JSON_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)
    fastf1.set_log_level("WARNING")
    only = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else None
    rebuild = sys.argv[1:] == ["rebuild"]
    years = [only[0]] if only else [y for y in SEASONS if y >= FIRST_YEAR]

    for year in years:
        for rnd, name, race in races_of(year, only):
            out_file = OPENF1_DIR / f"{year}_{rnd:02d}.parquet"
            if only and rnd != only[1]:
                continue
            if out_file.exists() and not (only or rebuild):
                continue
            print(year, "round", rnd, "-", name)
            try:
                raw = download_race(race, OPENF1_JSON_DIR / f"{year}_{rnd:02d}.json.gz")
                if not raw["laps"]:
                    print("   skipped: OpenF1 has no laps for this race")
                    continue
                table = race_table(raw, year, rnd, name)
            except Exception as err:
                print("   skipped:", err)
                continue
            table.to_parquet(out_file, index=False)
            print("   saved", len(table), "laps")

    print("Done:", len(list(OPENF1_DIR.glob("*.parquet"))), "OpenF1 races saved")


if __name__ == "__main__":
    main()
