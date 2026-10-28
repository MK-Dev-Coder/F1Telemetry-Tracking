import gzip
import json

import fastf1
import pandas as pd

from collect_openf1 import get
from config import CACHE_DIR, DATA_DIR, OPENF1_JSON_DIR

LAPS_FILE = DATA_DIR / "all_laps.parquet"
TIMING_FILE = DATA_DIR / "timing.parquet"            # sector times, mini-sectors and speeds, per driver per lap
MESSAGES_FILE = DATA_DIR / "race_control.parquet"    # race control messages
TRACKS_FILE = DATA_DIR / "tracks.json"               # one lap of car positions per circuit, to draw the map
COLOURS_FILE = DATA_DIR / "team_colours.json"        # each driver's team colour, per race
NAMES_FILE = DATA_DIR / "driver_names.json"          # driver code -> surname
NAMES_2022 = {"VET": "Vettel", "LAT": "Latifi", "MSC": "Schumacher"}   # 2022 only, so not in OpenF1
# OpenF1 starts in 2023, so the 2022 team colours are written here
COLOURS_2022 = {"Red Bull Racing": "3671C6", "Ferrari": "ED1C24", "Mercedes": "6CD3BF", "McLaren": "F58020",
                "Alpine": "2293D1", "AlphaTauri": "4E7C9B", "Aston Martin": "2D826D", "Williams": "37BEDD",
                "Alfa Romeo": "B12039", "Haas F1 Team": "B6BABD"}

TIMING_COLUMNS = ["RaceId", "Driver", "LapNumber", "S1", "S2", "S3", "Seg1", "Seg2", "Seg3",
                  "I1", "I2", "ST"]                   # km/h at the two intermediate points and the speed trap
MESSAGE_COLUMNS = ["RaceId", "Seconds", "Lap", "Category", "Flag", "Message"]

# OpenF1's mini-sector codes (the colours on F1's live timing)
SEGMENT = {2048: "y", 2049: "g", 2051: "p", 2064: "pit"}        # yellow, green (personal best), purple (fastest)


def read_json(race_id):
    file = OPENF1_JSON_DIR / f"{race_id}.json.gz"
    return json.loads(gzip.decompress(file.read_bytes())) if file.exists() else None


def times(column):
    return pd.to_datetime(column, format="ISO8601", utc=True)


# Recipe 1: sector times, mini-sectors and speed traps from OpenF1 (2023 onwards)
def openf1_timing(race_id, raw):
    names = {d["driver_number"]: d["name_acronym"] for d in raw["drivers"]}
    rows = []
    for lap in raw["laps"]:
        segments = [[SEGMENT.get(code, "") for code in (lap.get(f"segments_sector_{k}") or [])] for k in (1, 2, 3)]
        rows.append({"RaceId": race_id, "Driver": names.get(lap["driver_number"]), "LapNumber": float(lap["lap_number"]),
                     "S1": lap.get("duration_sector_1"), "S2": lap.get("duration_sector_2"), "S3": lap.get("duration_sector_3"),
                     "Seg1": segments[0], "Seg2": segments[1], "Seg3": segments[2],
                     "I1": lap.get("i1_speed"), "I2": lap.get("i2_speed"), "ST": lap.get("st_speed")})
    return rows


# Recipe 2: sector times and speed traps from the FastF1 cache (2022), with no internet. FastF1 has no mini-sectors.
def fastf1_timing(race_id):
    year, rnd = int(race_id[:4]), int(race_id[5:])
    session = fastf1.get_session(year, rnd, "R")
    session.load(laps=True, telemetry=False, weather=False, messages=False)
    laps = session.laps
    return [{"RaceId": race_id, "Driver": r["Driver"], "LapNumber": float(r["LapNumber"]),
             "S1": r["Sector1Time"].total_seconds() if pd.notna(r["Sector1Time"]) else None,
             "S2": r["Sector2Time"].total_seconds() if pd.notna(r["Sector2Time"]) else None,
             "S3": r["Sector3Time"].total_seconds() if pd.notna(r["Sector3Time"]) else None,
             "Seg1": [], "Seg2": [], "Seg3": [],
             **{k: r[f"Speed{k}"] if pd.notna(r[f"Speed{k}"]) else None for k in ("I1", "I2", "ST")}} for _, r in laps.iterrows()]


# Recipe 3: race control messages, timed from lights out (the leader's lap 1 start)
def openf1_messages(race_id, raw):
    starts = [l["date_start"] for l in raw["laps"] if l["lap_number"] == 1 and l.get("date_start")]
    if not starts:
        return []
    lights_out = times(pd.Series(starts)).min()
    return [{"RaceId": race_id, "Seconds": (times(pd.Series([m["date"]]))[0] - lights_out).total_seconds(),
             "Lap": m.get("lap_number"), "Category": m.get("category"), "Flag": m.get("flag"), "Message": m.get("message")}
            for m in raw["race_control"]]


# Recipe 4: the shape of each circuit - the positions of the car on the race's fastest lap.
# "t" is the seconds since the start of that lap, so the app can place a car that is X seconds behind.
def track_shape(raw):
    laps = [l for l in raw["laps"] if l.get("lap_duration") and l.get("date_start") and not l.get("is_pit_out_lap")]
    if not laps:
        return None
    best = min(laps, key=lambda l: l["lap_duration"])
    start = times(pd.Series([best["date_start"]]))[0]
    end = start + pd.to_timedelta(float(best["lap_duration"]), unit="s")
    points = get("location", session_key=raw["session"]["session_key"], driver_number=best["driver_number"],
                 **{"date>": start.strftime("%Y-%m-%dT%H:%M:%S.%f"), "date<": end.strftime("%Y-%m-%dT%H:%M:%S.%f")})
    points = [p for p in points if p.get("x") is not None]
    if len(points) < 50:
        return None
    t0 = times(pd.Series([points[0]["date"]]))[0]
    return {"x": [p["x"] for p in points], "y": [p["y"] for p in points],
            "t": [round((times(pd.Series([p["date"]]))[0] - t0).total_seconds(), 3) for p in points],
            "lap_time": best["lap_duration"]}


def main():
    fastf1.Cache.enable_cache(CACHE_DIR)
    fastf1.Cache.offline_mode(True)
    fastf1.set_log_level("CRITICAL")
    laps = pd.read_parquet(LAPS_FILE, columns=["RaceId", "Circuit"]).drop_duplicates()
    circuit_of = dict(zip(laps["RaceId"], laps["Circuit"]))

    teams = pd.read_parquet(LAPS_FILE, columns=["RaceId", "Driver", "Team"]).drop_duplicates(["RaceId", "Driver"])
    colours = {race: {d: COLOURS_2022.get(t, "888888") for d, t in zip(r["Driver"], r["Team"])}
               for race, r in teams.groupby("RaceId")}
    timing, messages = [], []
    for race_id in sorted(circuit_of):
        raw = read_json(race_id)
        if raw:
            timing += openf1_timing(race_id, raw)
            messages += openf1_messages(race_id, raw)
            colours[race_id].update({d["name_acronym"]: d["team_colour"] for d in raw["drivers"] if d.get("team_colour")})
        else:
            try:
                timing += fastf1_timing(race_id)
            except Exception as err:
                print("   no sector times for", race_id, "-", err)
    timing = pd.DataFrame(timing, columns=TIMING_COLUMNS).dropna(subset=["Driver"])
    timing = timing.drop_duplicates(["RaceId", "Driver", "LapNumber"])
    timing.to_parquet(TIMING_FILE, index=False)
    pd.DataFrame(messages, columns=MESSAGE_COLUMNS).to_parquet(MESSAGES_FILE, index=False)
    COLOURS_FILE.write_text(json.dumps(colours), encoding="utf-8")
    names = dict(NAMES_2022)
    for race_id in sorted(circuit_of):                                   # later races win (newest spelling)
        raw = read_json(race_id)
        if raw:
            names.update({d["name_acronym"]: str(d.get("last_name") or d["name_acronym"]).title() for d in raw["drivers"]})
    NAMES_FILE.write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")
    print(f"Sector times: {len(timing)} laps in {timing['RaceId'].nunique()} races. "
          f"Race control messages: {len(messages)}")

    # One map per circuit, from its latest race with OpenF1 data (downloaded once)
    tracks = json.loads(TRACKS_FILE.read_text(encoding="utf-8")) if TRACKS_FILE.exists() else {}
    for race_id in sorted(circuit_of, reverse=True):
        circuit = circuit_of[race_id]
        raw = None if circuit in tracks else read_json(race_id)
        if raw:
            print("track map:", circuit, f"({race_id})")
            shape = track_shape(raw)
            if shape:
                tracks[circuit] = shape
                TRACKS_FILE.write_text(json.dumps(tracks), encoding="utf-8")
    print("Track maps:", len(tracks), "circuits. No map for:", sorted(set(circuit_of.values()) - set(tracks)))


if __name__ == "__main__":
    main()
