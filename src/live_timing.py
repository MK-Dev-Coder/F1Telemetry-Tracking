import gzip
import json
import math
from functools import lru_cache

import numpy as np
import pandas as pd

from build_timing import COLOURS_FILE, MESSAGES_FILE, NAMES_FILE, TIMING_FILE, TRACKS_FILE
from config import OPENF1_JSON_DIR

# Made by build_timing.py. Without them the app still runs, just with no sector times,
# mini-sectors, race control messages or track maps.
def by_race(file):
    rows = pd.read_parquet(file) if file.exists() else pd.DataFrame()
    return {race: r for race, r in rows.groupby("RaceId")} if "RaceId" in rows else {}


def read_json(file):
    return json.loads(file.read_text(encoding="utf-8")) if file.exists() else {}


TIMING, MESSAGES = by_race(TIMING_FILE), by_race(MESSAGES_FILE)
TRACKS, COLOURS, NAMES = read_json(TRACKS_FILE), read_json(COLOURS_FILE), read_json(NAMES_FILE)
OPENF1_CATEGORY = {"Flag": "FLAG", "SafetyCar": "SAFETY CAR", "SessionStatus": "FLAG", "Other": "CONTROL"}
TYRE_WORD = {"SOFT": "soft", "MEDIUM": "medium", "HARD": "hard", "INTERMEDIATE": "intermediate", "WET": "wet"}
RACES = {}                                    # filled in by api.py: race id -> every lap of that race


def num(v):
    """A plain float for JSON, or None."""
    return None if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v) else round(float(v), 3)


def median(values):
    """The median as a plain float, or None when every value is missing (and no warning about it)."""
    return num(values.median()) if values.notna().any() else None


def colour(time, personal_best, overall_best):
    """F1 timing colours: purple = fastest of anyone so far, green = the driver's best so far, yellow = slower."""
    if pd.isna(time):
        return ""
    return "p" if time <= overall_best + 1e-9 else "g" if time <= personal_best + 1e-9 else "y"


SPEEDS = ["I1", "I2", "ST"]                    # km/h at the two intermediate points and the speed trap (higher = better)


# Recipe 1: everything that doesn't change with the lap you look at, worked out once per race
@lru_cache(maxsize=40)
def prepare(race_id):
    laps = RACES[race_id].copy()
    timing = TIMING.get(race_id)
    if timing is not None:
        laps = laps.merge(timing.reindex(columns=["Driver", "LapNumber", "S1", "S2", "S3", "Seg1", "Seg2", "Seg3", *SPEEDS]),
                          on=["Driver", "LapNumber"], how="left")      # (older timing files have no speeds)
    else:                                                                # no timing file: no sector times
        for col in ["S1", "S2", "S3", *SPEEDS]:
            laps[col] = np.nan
        for col in ["Seg1", "Seg2", "Seg3"]:
            laps[col] = None
    laps = laps.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)
    # A lap with no end time (OpenF1 often has none for the last lap): estimate it from the lap before
    by_driver = laps.groupby("Driver")
    estimate = by_driver["Time"].shift(1) + laps["LapTime"].fillna(by_driver["LapTime"].shift(1))
    laps["Time"] = laps["Time"].fillna(estimate)
    by_driver = laps.groupby("Driver")
    for col in ["LapTime", "S1", "S2", "S3"]:
        laps[f"PB_{col}"] = by_driver[col].cummin()                       # the driver's best so far,
        laps[f"PB_{col}"] = laps.groupby("Driver")[f"PB_{col}"].ffill()  # kept through laps with no time
        best_by_lap = laps.groupby("LapNumber")[col].min().cummin()      # anyone's best up to this lap
        laps[f"Best_{col}"] = laps["LapNumber"].map(best_by_lap)
        laps[f"Col_{col}"] = [colour(t, pb, best) for t, pb, best in zip(laps[col], laps[f"PB_{col}"], laps[f"Best_{col}"])]
    for col in SPEEDS:                                                     # the same, highest speed first
        laps[col] = pd.to_numeric(laps[col], errors="coerce")
        laps[f"PB_{col}"] = laps.groupby("Driver")[col].cummax()
        laps[f"PB_{col}"] = laps.groupby("Driver")[f"PB_{col}"].ffill()
        laps[f"Best_{col}"] = laps["LapNumber"].map(laps.groupby("LapNumber")[col].max().cummax())
        laps[f"Col_{col}"] = [colour(-v, -pb, -best) for v, pb, best in zip(laps[col], laps[f"PB_{col}"], laps[f"Best_{col}"])]

    zero = lights_out(laps)
    laps["Clock"] = laps["Time"] - zero
    return laps, zero


def lights_out(laps):
    """When the race started, on the Time clock: the start of lap 1, or an estimate when lap 1 has no times
    (a red flag on lap 1, or a gap in the data): the first timed lap minus that many typical laps."""
    first = laps[laps["LapNumber"] == laps["LapNumber"].min()]
    zero = (first["Time"] - first["LapTime"]).min()
    if pd.isna(zero):
        typical = laps.loc[laps["LapNumber"] > 1, "LapTime"].median()
        timed = laps[laps["Time"].notna()]
        if len(timed) and pd.notna(typical):
            n = timed["LapNumber"].min()
            zero = timed.loc[timed["LapNumber"] == n, "Time"].min() - n * typical
    return zero


def laps_down(gap, leader_lap):
    """How many laps behind the leader a gap means (0 if on the lead lap)."""
    return int(gap // leader_lap) if gap is not None and leader_lap and gap >= leader_lap else 0


def lap_text(gap, leader_lap):
    if gap is None:
        return None
    down = laps_down(gap, leader_lap)
    if down:
        return f"+{down} LAP" + ("S" if down > 1 else "")
    return f"{gap:+.3f}"                                                 # a negative gap keeps its sign


# Recipe 2: the messages up to this lap - race control's own (2023 on), plus pit stops,
# overtakes and fastest laps worked out from the timing
@lru_cache(maxsize=40)
def all_messages(race_id):
    laps, zero = prepare(race_id)
    out = []
    real = MESSAGES.get(race_id)
    if real is not None:
        for _, m in real.iterrows():
            if m["Flag"] == "BLUE":                                       # dozens per race: left out
                continue
            out.append({"lap": int(m["Lap"]) if pd.notna(m["Lap"]) else 1, "seconds": num(m["Seconds"]),
                        "cat": OPENF1_CATEGORY.get(m["Category"], "CONTROL"), "text": str(m["Message"])})
    stops = laps[laps["IsStop"]]
    for _, s in stops.iterrows():
        nxt = laps[(laps["Driver"] == s["Driver"]) & (laps["LapNumber"] == s["LapNumber"] + 1)]
        out.append({"lap": int(s["LapNumber"]), "seconds": num(s["Clock"]), "cat": "PIT",
                    "text": f"{s['Driver']} enters the pit lane on {TYRE_WORD.get(s['Compound'], 'unknown')} tyres"})
        if len(nxt):
            n = nxt.iloc[0]
            out.append({"lap": int(n["LapNumber"]), "seconds": num(n["Clock"]), "cat": "PIT",
                        "text": f"{n['Driver']} rejoins on {TYRE_WORD.get(n['Compound'], 'unknown')} tyres"})
    best = math.inf
    for lap, rows in laps.groupby("LapNumber"):
        quick = rows.loc[rows["LapTime"].idxmin()] if rows["LapTime"].notna().any() else None
        if quick is not None and quick["LapTime"] < best - 1e-9:
            best = quick["LapTime"]
            if lap > 1:
                m, s = divmod(best, 60)
                out.append({"lap": int(lap), "seconds": num(quick["Clock"]), "cat": "TIMING",
                            "text": f"Fastest lap: {quick['Driver']} {int(m)}:{s:06.3f} on lap {int(lap)}"})
    # Overtakes: two drivers swap places from one lap to the next, and neither of them pitted
    pos = laps.pivot_table(index="LapNumber", columns="Driver", values="Position")
    pitted = set(zip(laps.loc[laps["IsStop"] | laps["PitOutTime"].notna(), "Driver"],
                     laps.loc[laps["IsStop"] | laps["PitOutTime"].notna(), "LapNumber"]))
    for lap in pos.index[1:]:
        before, now = pos.loc[lap - 1] if lap - 1 in pos.index else None, pos.loc[lap]
        if before is None:
            continue
        for a in now.index:
            if pd.isna(now[a]) or pd.isna(before[a]) or now[a] >= before[a] or (a, lap) in pitted or (a, lap - 1) in pitted:
                continue
            for b in now.index:
                if before[b] == now[a] and now[b] == before[a] and (b, lap) not in pitted and (b, lap - 1) not in pitted:
                    row = laps[(laps["Driver"] == a) & (laps["LapNumber"] == lap)].iloc[0]
                    out.append({"lap": int(lap), "seconds": num(row["Clock"]), "cat": "OVERTAKE",
                                "text": f"{a} passes {b} for P{int(now[a])}"})
    return sorted(out, key=lambda m: (m["lap"], m["seconds"] if m["seconds"] is not None else -1e9), reverse=True)


# Recipe 3: the live timing screen at the end of one lap
def snapshot(race_id, lap):
    laps, _ = prepare(race_id)
    upto = laps[laps["LapNumber"] <= lap]
    now = upto[upto["LapNumber"] == lap].sort_values("Position", na_position="last")
    leader = now.iloc[0]
    ref = leader["Time"] if pd.notna(leader["Time"]) else now["Time"].min()      # when the leader crossed the line
    leader_lap = num(leader["LapTime"]) or median(now["LapTime"]) or median(upto["LapTime"])
    colours = COLOURS.get(race_id, {})

    best = {"lap": None, "sectors": [None, None, None], "speeds": [None, None, None]}
    for k, col in enumerate(["LapTime", "S1", "S2", "S3"]):
        if upto[col].notna().any():
            r = upto.loc[upto[col].idxmin()]
            holder = {"time": num(r[col]), "driver": r["Driver"]}
            if k == 0:
                best["lap"] = holder
            else:
                best["sectors"][k - 1] = holder
    for k, col in enumerate(SPEEDS):                                       # the highest speed so far at each point
        if upto[col].notna().any():
            r = upto.loc[upto[col].idxmax()]
            best["speeds"][k] = {"speed": num(r[col]), "driver": r["Driver"]}

    drivers, ahead_time = [], None
    for _, r in now.iterrows():
        gap = num(r["Time"] - ref) if pd.notna(r["Time"]) and pd.notna(ref) else None
        interval = num(r["Time"] - ahead_time) if ahead_time is not None and pd.notna(r["Time"]) else None
        ahead_time = r["Time"] if pd.notna(r["Time"]) else ahead_time
        mine = upto[upto["Driver"] == r["Driver"]]
        stints = [{"compound": str(s["Compound"].iloc[0]), "from": int(s["LapNumber"].min()), "to": int(s["LapNumber"].max())}
                  for _, s in mine.groupby("Stint", sort=True)]
        out_lap = pd.notna(r["PitOutTime"]) and lap > 1
        status = "PIT" if r["IsStop"] else "OUT LAP" if out_lap else "ON TRACK"
        behind = None if gap is None or not leader_lap else round(gap % leader_lap, 3)
        drivers.append({
            "driver": r["Driver"], "team": r["Team"], "colour": "#" + colours.get(r["Driver"], "888888"),
            "position": num(r["Position"]), "status": status,
            "last_lap": num(r["LapTime"]), "last_col": r["Col_LapTime"], "best_lap": num(r["PB_LapTime"]),
            "best_is_fastest": bool(best["lap"] and num(r["PB_LapTime"]) == best["lap"]["time"]),
            "gap": gap, "gap_text": lap_text(gap, leader_lap) if r["Driver"] != leader["Driver"] else None,
            "interval": interval, "interval_text": lap_text(interval, leader_lap) if interval is not None else None,
            "laps": int(lap) - laps_down(gap, leader_lap),                # lapped cars have done fewer laps
            "pits": int(mine["IsStop"].iloc[:-1].sum()) if len(mine) > 1 else 0,
            "compound": str(r["Compound"]), "tyre_age": num(r["TyreLife"]), "stints": stints,
            "sectors": [{"time": num(r[s]), "col": r[f"Col_{s}"], "best": num(r[f"PB_{s}"])} for s in ["S1", "S2", "S3"]],
            "segments": [list(r[s]) if isinstance(r[s], (list, np.ndarray)) else [] for s in ["Seg1", "Seg2", "Seg3"]],
            "speeds": [{"speed": num(r[s]), "col": r[f"Col_{s}"], "best": num(r[f"PB_{s}"])} for s in SPEEDS],
            "behind_leader_s": behind,                                    # where the car is on the track map
        })
    racing, race_laps = set(now["Driver"]), int(laps["LapNumber"].max())
    distance = laps["TotalLaps"].max() if "TotalLaps" in laps else race_laps
    gone = [(mine["Position"].iloc[-1], code, mine) for code, mine in upto[~upto["Driver"].isin(racing)].groupby("Driver")]
    for final, code, mine in sorted(gone, key=lambda g: (g[0] if pd.notna(g[0]) else 99)):
        last = mine.iloc[-1]
        # Still racing but missing from the data on this lap (the car shows up again later), so don't call
        # it retired. At the flag, F1 classifies anyone who completed 90% of the race distance.
        no_data = ((laps["Driver"] == code) & (laps["LapNumber"] > lap)).any()
        finished = not no_data and lap >= race_laps and last["LapNumber"] >= 0.9 * distance
        status = "NO DATA" if no_data else "FINISHED" if finished else "OUT"
        drivers.append({"driver": code, "team": last["Team"], "colour": "#" + colours.get(code, "888888"),
                        "position": num(final) if status != "OUT" else None, "status": status,
                        "laps": int(last["LapNumber"]), "compound": str(last["Compound"]),
                        "tyre_age": num(last["TyreLife"]), "pits": int(mine["IsStop"].sum()), "stints": [],
                        "sectors": [], "segments": [[], [], []], "speeds": [], "behind_leader_s": None})

    used = [c for c in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"] if (upto["Compound"] == c).any()]
    return {"lap": int(lap), "clock": num(leader["Clock"]), "leader_lap_time": leader_lap,
            "weather": {"air": num(leader["AirTemp"]), "track": num(leader["TrackTemp"]),
                        "humidity": num(leader["Humidity"]), "rain": bool(leader["Rainfall"])},
            "compounds_used": used, "best": best, "drivers": drivers,
            "messages": [m for m in all_messages(race_id) if m["lap"] <= lap][:80],
            "has_segments": any(d["segments"] and any(d["segments"]) for d in drivers)}


def track(circuit, race_id):
    """The circuit's shape and where the sectors end (from the median sector times)."""
    shape = TRACKS.get(circuit)
    if not shape:
        return None
    laps, _ = prepare(race_id)
    s = laps[["S1", "S2", "S3"]].median()
    total = s.sum()
    fractions = [num(s["S1"] / total), num((s["S1"] + s["S2"]) / total)] if total and not pd.isna(total) else [1 / 3, 2 / 3]
    return {**shape, "sector_ends": fractions}


# Each driver's car number, name, team and photo, from OpenF1's driver list of each race (2023 on). The photos are
# F1's own: OpenF1 gives their address, and the page loads them from F1's site (without the internet it shows the
# car number instead). A race without its own list (2022) takes each driver from the nearest race that has one.
@lru_cache(maxsize=1)
def driver_lists():
    out = {}
    for file in sorted(OPENF1_JSON_DIR.glob("*.json.gz")):
        try:
            drivers = json.loads(gzip.decompress(file.read_bytes())).get("drivers") or []
        except (OSError, ValueError):
            continue
        out[file.name.split(".")[0]] = {
            d["name_acronym"]: {"number": d.get("driver_number"), "first": d.get("first_name"), "last": d.get("last_name"),
                                "team": d.get("team_name"), "photo": photo(d.get("headshot_url"))}
            for d in drivers if d.get("name_acronym")}
    return out


def photo(url):
    """F1's driver photo, one size up from OpenF1's link (1col is about 90 pixels wide, 2col about 180)."""
    return url.replace("/1col/", "/2col/") if url else None


def cards(race_id, codes):
    """{code: {number, first, last, team, photo}} for the drivers of one race."""
    lists = driver_lists()
    when = lambda r: int(r[:4]) * 40 + int(r[5:])
    nearest = sorted(lists, key=lambda r: abs(when(r) - when(race_id)))
    out = {}
    for code in codes:
        mine = [lists[r][code] for r in nearest if code in lists[r]]
        if mine:
            out[code] = {**mine[0], "photo": next((c["photo"] for c in mine if c["photo"]), None)}
    return out
