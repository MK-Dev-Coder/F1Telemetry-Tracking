"""F1 25: the pit stop, tyre and race outcome models, live, on the player's own game races.

The models learned from real F1 races (2022-2025). The game's laps are put into the same columns as
data/all_laps.parquet and go through exactly the same feature code (features.py and train_tyre_model.py),
so the models see the game the way they saw real races. The game's AI makes its own strategy, so this
shows the models working live during a race - how accurate they are comes from the real 2026 races.

Game races are often shorter than real ones (25% or 50% distance), and the game wears the tyres out
faster to match. So the lap counts the models use (tyre age, laps on the set, lap number, laps left) are
scaled to the real race at that track: in a 13-lap Jeddah (real race: 50 laps) one lap counts as 3.8.
The race outcome model (each car's chance to win, finish on the podium and score points) keeps the game's
own laps for the gaps and the laps left - catching a car takes real laps - and scales only the tyre ages.
"""
import json
import math
import time
from collections import Counter, defaultdict

import lightgbm as lgb
import pandas as pd

import battle
import features
import game_telemetry
import pit_now
import race_outcome as ro
import train_tyre_model as tyre
from config import MODELS_DIR
from explain_pit_model import as_points, describe, explain, load_model, sigmoid, to_points

HUMIDITY = 55.0                               # the game doesn't send humidity: the typical value of the real races
RAIN = game_telemetry.RAIN
OUT = {4: "DNF", 5: "DSQ", 6: "NOT CLASSIFIED", 7: "RETIRED"}
GAME_NAMES = {"Paul Ricard": "Le Castellet", "Monaco": "Monte Carlo", "Yas Marina": "Yas Marina Circuit",
              "Madrid": "Madring"}                # F1 26 (first raced in 2026, so not in the training data yet)
LAP_COUNTS = ["TyreLife", "LapsInStint", "LapNumber", "LapsLeft"]    # scaled to the real race length
# Counted over the last 2-3 laps: in a shortened race those laps cover several real laps, so the counts are
# brought back to what a real-race window would hold (in a 13-lap Jeddah, 6 or more stops in 2 laps was on
# 21% of laps against 2.4% in the real races)
WINDOW_COUNTS = ["FieldStopsLast2", "AheadStoppedRecently", "BehindStoppedRecently",
                 "FitSoftRecent", "FitMediumRecent", "FitHardRecent", "FitWetRecent"]

_models, _cache = {}, {}


def models():
    """The two saved models, and each track's real race length (loaded the first time they're needed)."""
    if not _models:
        pit, info = load_model()
        races = pd.read_parquet(features.IN_FILE, columns=["RaceId", "Circuit", "TotalLaps"]).drop_duplicates("RaceId")
        _models.update(pit=pit, info=info, tyre=lgb.Booster(model_file=str(MODELS_DIR / "tyre_model.txt")),
                       tyre_info=json.loads((MODELS_DIR / "tyre_model.json").read_text(encoding="utf-8")),
                       real_laps=races.groupby("Circuit")["TotalLaps"].median().round().to_dict(),
                       typical_laps=float(races["TotalLaps"].median()),
                       outcome=ro.load_model() if ro.MODEL_FILE.exists() else None,   # python train_race_model.py
                       pit_now=pit_now.load() if ro.MODEL_FILE.exists() else None,
                       battle=battle.load() if ro.MODEL_FILE.exists() else None)
    return _models


def circuit(track):
    """The game's track under the name the models learned, or None for a track they never saw
    (a track not raced in 2022-2026, or a short or reverse layout)."""
    name = GAME_NAMES.get(track, track)
    return name if name in models()["info"]["categories"]["Circuit"] else None


SHORTEST_SCALED = 0.2                         # races under 20% distance (the game's "5 laps") aren't scaled


def race_scale(track_name, game_laps, sprint=False):
    """How many real-race laps one lap of this game race is worth (1.0 for a full-length race).
    The game wears the tyres out faster in 25%, 35% and 50% races; in its very short races (5 laps) it doesn't,
    and nobody has to stop, so those keep their own laps (in a real 5-lap Miami race nobody pitted). A sprint
    is short by design (about a third of the race, normal tyre wear, no stop), so it isn't scaled either."""
    m = models()
    real = m["real_laps"].get(track_name) or m["typical_laps"]
    scaled = not sprint and SHORTEST_SCALED * real <= game_laps < real
    factor = real / max(game_laps, 1) if scaled else 1.0
    return {"game_laps": int(game_laps), "real_laps": int(real), "factor": factor, "sprint": sprint,
            "very_short": not sprint and game_laps < SHORTEST_SCALED * real,
            "horizon_here": round(features.HORIZON / factor, 1)}


# Recipe 1: when the safety car and the VSC were out, from the game's safety car events
def neutral_periods(events):
    periods, out = [], {}
    for e in events:
        text = e["text"]
        kind = "6" if text.startswith("Virtual") else "4" if text.startswith("Safety car") else None
        if e["code"] != "SCAR" or kind is None:
            continue
        if text.endswith("deployed"):
            out[kind] = e["t"]
        elif text.endswith(("returned", "resume race")) and kind in out:
            periods.append((out.pop(kind), e["t"], kind))
    return periods + [(start, math.inf, kind) for kind, start in out.items()]       # still out


def track_status(start, end, periods, red_flags):
    """F1's track status codes for one lap, as in the real data: "1" green, "4" safety car, "6" VSC, "5" red."""
    codes = {kind for a, b, kind in periods if a < end and b > start}
    codes |= {"5"} if any(start < t <= end for t in red_flags) else set()
    return "".join(sorted(codes)) or "1"


def conditions(rec, info, key, missing=None):
    """The weather when the lap ended (kept with each lap); older sessions only have the session's latest."""
    value = rec.get(key)
    return value if value is not None else info.get(key, missing)


# Recipe 2: the game's laps in the columns of the real races (one row per car and finished lap)
def lap_rows(s):
    periods = neutral_periods(s.events)
    red_flags = [e["t"] for e in s.events if e["code"] == "RDFL"]
    info, rows = s.info, []
    # Sessions recorded from now on mark the laps a car came into and out of the pit lane; older ones only
    # mark pit laps, so there the first of two pit laps in a row is taken as the in-lap
    marked = any("pit_entry" in r or "pit_exit" in r for laps in s.laps.values() for r in laps.values())
    for car, laps in s.laps.items():
        done = [(n, r) for n, r in sorted(laps.items()) if r.get("t") is not None and r.get("position")]
        stint, compound, age_before, fresh, t_before, pit_before = 0, None, None, False, None, False
        for n, r in done:
            c, age = r.get("compound") or "UNKNOWN", r.get("tyre_age")
            if c == "UNKNOWN" and compound:                  # no tyre data from the game yet: still the same set
                c = compound
            changed = c != compound and compound not in (None, "UNKNOWN")
            if stint == 0 or changed or (age is not None and age_before is not None and age < age_before):
                stint, fresh = stint + 1, (age or 0) == 0                     # a new set: was it new?
            compound, age_before = c, age
            end = r["t"]
            start = t_before if t_before is not None else end - (r.get("time") or 0)
            pitted = bool(r.get("pitted"))
            if marked:                                       # the in-lap ends in the pit lane; the out-lap starts there
                pit_in, pit_out = r.get("pit_entry"), r.get("pit_exit")
            else:
                pit_in, pit_out = pitted and not pit_before, pitted and pit_before
            rows.append({"RaceId": str(s.uid), "Driver": str(car), "Team": s.drivers.get(car, {}).get("team", ""),
                         "LapNumber": float(n), "LapTime": r.get("time") or math.nan, "Time": end,
                         "Stint": float(stint), "Compound": c, "TyreLife": math.nan if age is None else age + 1.0,
                         "FreshTyre": fresh, "PitInTime": end - 1 if pit_in else math.nan,
                         "PitOutTime": start + 10 if pit_out else math.nan,
                         "TrackStatus": track_status(start, end, periods, red_flags), "Position": float(r["position"]),
                         "AirTemp": float(conditions(r, info, "air_temp", math.nan)),     # the lap's own weather
                         "TrackTemp": float(conditions(r, info, "track_temp", math.nan)),
                         "Humidity": HUMIDITY, "Rainfall": conditions(r, info, "weather") in RAIN,
                         "Year": 0, "Round": 0, "EventName": f"F1 25 {info.get('track', '')}",
                         "TotalLaps": info.get("total_laps") or max(laps), "Circuit": circuit(info.get("track")),
                         "Source": "F1 25"})
            t_before, pit_before = end, pitted
    return last_lap_times(rows)


def last_lap_times(rows):
    """On the last lap the game gives every car the same end time (the moment the race was over), so all the gaps
    would be 0. Those laps get their end time back from the lap before and the lap time."""
    before, by_lap = {(r["Driver"], r["LapNumber"]): r["Time"] for r in rows}, defaultdict(list)
    for r in rows:
        by_lap[r["LapNumber"]].append(r)
    for lap, here in by_lap.items():
        same, count = Counter(r["Time"] for r in here).most_common(1)[0]
        if len(here) >= 3 and count >= len(here) / 2:
            for r in here:
                prev = before.get((r["Driver"], lap - 1))
                if r["Time"] == same and prev is not None and not math.isnan(r["LapTime"]):
                    r["Time"] = prev + r["LapTime"]
    return rows


# Recipe 3: features and both models for every lap so far (worked out again only when a lap is added).
# The features are made from the game's own laps; only what the models see is scaled to the real race.
def model_all_laps(rows, sprint=False, grid=None):
    m, start = models(), time.perf_counter()
    laps = tyre.add_tyre_features(features.build(pd.DataFrame(rows)))
    for col in features.CATEGORIES:
        laps[col] = laps[col].astype("category")
    scale = race_scale(rows[0]["Circuit"], rows[0]["TotalLaps"], sprint)
    seen = laps.copy()
    seen[LAP_COUNTS] = seen[LAP_COUNTS] * scale["factor"]
    seen[WINDOW_COUNTS] = seen[WINDOW_COUNTS] / scale["factor"]
    chance, shap_values, base = explain(m["pit"], m["info"], seen)
    tyres = m["tyre"].predict(seen[m["tyre_info"]["features"]])
    outcome = outcome_all_laps(laps, scale, grid or {})
    return {"laps": laps, "seen": seen, "scale": scale, "chance": chance, "shap": shap_values, "base": base,
            "tyres": tyres, "outcome": outcome, "ms": round(1000 * (time.perf_counter() - start), 1)}


# Recipe 3b: the race outcome model on every lap so far: each car's chance to win, finish on the podium and score
# points. Its features are made the same way as for the real races; the grid slot comes from the game.
def outcome_all_laps(laps, scale, grid):
    if not models().get("outcome"):
        return None
    booster, info = models()["outcome"]
    first = laps.sort_values("LapNumber").drop_duplicates("Driver")
    starts = {d: float(grid.get(int(d)) or p) for d, p in zip(first["Driver"], first["Position"])}   # (or lap 1)
    results = pd.DataFrame({"RaceId": laps["RaceId"].iloc[0], "Driver": list(starts), "Grid": list(starts.values()),
                            "Final": math.nan, "Status": None})
    rows = ro.add_features(laps, results)
    for col in ["TyreLife", "TyreAgeVsAhead"]:                 # tyres wear faster in a shortened race
        rows[col] = rows[col] * scale["factor"]
    rows = ro.add_pit_adjusted(rows, info["pit_loss"])
    rows = rows[rows["Position"].notna()].reset_index(drop=True)
    # A lap can be shown when every place P1 ... Pn is on it (the finished race's last lap can miss the winner)
    whole = rows.groupby("LapNumber")["Position"].apply(lambda s: sorted(s) == list(range(1, len(s) + 1)))
    return {"rows": rows, **ro.summary(ro.chances(booster, info, rows)), "why": {}, "whole": set(whole[whole].index)}


def outcome_lap(o, status, lap):
    """The last lap (up to `lap`) that every car still racing has finished: the chances need the whole field on one
    lap. While live that's the leader's lap or one before it (a lapped car holds it a lap back)."""
    latest = o["rows"].groupby("Driver")["LapNumber"].max()
    racing = [d for d in latest.index if status.get(int(d)) not in (*OUT, 3)]           # 3 = finished
    full = min(lap, latest[racing].min() if racing else latest.max())
    shown = [n for n in o["whole"] if n <= full]
    return int(max(shown)) if shown else None


def outcome_why(o, lap):
    """The SHAP values for the win, podium and points chances on one lap (worked out once per lap)."""
    if lap not in o["why"]:
        booster, info = models()["outcome"]
        here = o["rows"][o["rows"]["LapNumber"] == lap]
        o["why"][lap] = (here, {target: ro.explain(booster, info, here, k) for target, k in ro.SHOWN.items()})
    return o["why"][lap]


def outcome_text(name, row, effect, k, scale):
    """A race outcome reason in words, with the tyre ages in the game's laps (the model saw them scaled)."""
    if scale["factor"] > 1.05 and name in ("TyreLife", "TyreAgeVsAhead") and pd.notna(row[name]):
        game = row.copy()
        game[name] = row[name] / scale["factor"]
        return f"{ro.describe(name, game, effect, k)} (worn like {abs(row[name]):.0f} laps of the real race)"
    return ro.describe(name, row, effect, k)


def status_name(code):
    """F1's track status code for a lap -> the name the pages use."""
    return "RED FLAG" if "5" in code else "SAFETY CAR" if "4" in code else "VSC" if "6" in code or "7" in code else "GREEN"


def coded(answer, drivers):
    """Car numbers -> driver codes in a pit-now answer (the game's cars are numbered)."""
    code = lambda car: drivers.get(int(car), {}).get("code", f"C{int(car):02d}")
    if not answer:
        return answer
    out = {**answer, "driver": code(answer["driver"]) if "driver" in answer else None}
    for side in ("ahead", "behind"):
        if out.get(side):
            out[side] = {**out[side], "driver": code(out[side]["driver"])}
    if "around" in out:
        out["around"] = [{**a, "driver": code(a["driver"])} for a in out["around"]]
    return out


def rejoins_for(w, at, drivers):
    """Where every car would rejoin if it pitted at the end of lap `at` (the last whole lap): {car: {...}}."""
    o, m = w.get("outcome"), models()
    if o is None or at is None or not m.get("pit_now"):
        return {}
    here = o["rows"][o["rows"]["LapNumber"] == at]
    loss = pit_now.cost(m["pit_now"], m["outcome"][1]["pit_loss"], here["Circuit"].iloc[0], lap_status(here))[0]
    return {int(car): coded(r, drivers) for car, r in pit_now.rejoin_all(here, loss).items()}


def lap_status(here):
    """The track status of a lap: the leader's."""
    return status_name(here.loc[here["Position"].idxmin(), "TrackStatus"])


def battles_for(w, at, drivers):
    """Every car against the car just ahead at the end of lap `at` (the last whole lap): the battle forecast and the
    undercut threat, {car: {...}}."""
    o, m = w.get("outcome"), models()
    if o is None or at is None or not m.get("battle"):
        return {}
    here = o["rows"][o["rows"]["LapNumber"] == at]
    code = lambda car: drivers.get(int(car), {}).get("code", f"C{int(car):02d}")
    return {int(car): {**b, "ahead": code(b["ahead"])}
            for car, b in battle.lap_battles(m["battle"], here, lap_status(here), w["scale"]["factor"]).items()}


def outcome_for(w, status, lap, full=False):
    """Every car's chances at the end of the last whole lap (up to `lap`): {car: {...}}, and that lap.
    full=True: every reason, the chance of each finishing position, and the step that makes the field add up."""
    o = w.get("outcome")
    if o is None or o["rows"].empty:
        return {}, None
    at = outcome_lap(o, status, lap)
    if at is None:
        return {}, None
    here, why = outcome_why(o, at)
    out = {}
    for idx, row in here.iterrows():
        chances = {t: number(o[t][idx]) for t in ro.SHOWN}
        target = next((t for t in ro.SHOWN if chances[t] >= 0.1), "points")              # win, else podium, else points
        entry = {**chances, "likely": int(o["likely"][idx]), "expected": number(o["expected"][idx], 2), "for": target}
        for t, k in ro.SHOWN.items():
            if not full and t != target:
                continue
            values, start, words = why[t]
            points = values.loc[idx]
            order = points.abs().sort_values(ascending=False).index[:None if full else 1]
            reasons = [{"feature": n, "text": outcome_text(n, words.loc[idx], points[n], k, w["scale"]),
                        "points": round(float(points[n]), 4), "label": as_points(points[n])} for n in order]
            if full:
                entry.setdefault("why", {})[t] = {"chance": chances[t], "start": number(start), "model": number(start + points.sum()),
                                                  "fitted": number(chances[t] - start - points.sum()), "reasons": reasons}
            else:
                entry["reason"] = reasons[0]
        if full:
            entry["spot"] = [number(v) for v in o["spot"][idx][:max(len(here), int(here["Position"].max()))]]
        out[int(row["Driver"])] = entry
    return out, at


def worked_out(s, lock):
    """The models' results for every lap of a race (kept until the session's laps or events change)."""
    models()                                  # loaded before taking the lock (only slow the first time)
    sprint = game_telemetry.RECORDER.is_sprint(s)          # looked up once per session (reads the saved list)
    with lock:                                # a quick copy of what's needed, then the work without the lock
        version = (s.token, s.version)
        hit = _cache.get(s.uid)
        same = hit is not None and hit.get("version") == version
        rows = None if same else lap_rows(s) if s.is_race() else []
        status = {car: d["result_status"] for car, d in s.lap_now.items()}
        grid = {car: d.get("grid") for car, d in s.lap_now.items()}
        drivers = {car: dict(d) for car, d in s.drivers.items()}
        player = s.player
    if same:
        return hit, status, drivers, player
    if not rows:
        return None, status, drivers, player
    key = (len(rows), sum(r["Time"] for r in rows), "".join(r["TrackStatus"] for r in rows))
    if hit is None or hit["key"] != key:
        hit = {"key": key, **model_all_laps(rows, sprint, grid)}
    hit["version"] = version
    _cache[s.uid] = hit
    return hit, status, drivers, player


def reason_text(name, seen, game, info, effect, scale):
    """A reason in plain English. In a shortened race, lap counts are told in the game's laps, with what they
    are worth in the real race (which is what the model weighed)."""
    if name == "Circuit" and pd.isna(seen["Circuit"]):
        return "a track the model never saw in real F1"
    if name == "Humidity":
        return "humidity (the game doesn't send it: a typical value is used)"
    if scale["factor"] > 1.05 and name in LAP_COUNTS:
        return f"{describe(name, game, info, effect)} (like {seen[name]:.0f} in the real {scale['real_laps']}-lap race)"
    if scale["factor"] > 1.05 and name == "FieldStopsLast2":
        return f"{describe(name, game, info, effect)} (like {seen[name]:.1f} in 2 laps of the real race)"
    if scale["factor"] > 1.05 and name in WINDOW_COUNTS:
        return describe(name, game, info, effect)
    if scale["factor"] > 1.05 and name == "Circuit":
        usual = info["usual_stint"]["track"].get(f"{seen['Circuit']}|{seen['Compound']}")
        text = f"at {seen['Circuit']} teams pit {'more' if effect > 0 else 'less'} than at most tracks"
        return text + (f" ({str(seen['Compound']).lower()} tyres usually last about {usual:.0f} laps in the real race, "
                       f"about {usual / scale['factor']:.0f} in this one)" if usual else "")
    return describe(name, seen, info, effect)


def reasons(w, k, top=None):
    info, seen, game = models()["info"], w["seen"].iloc[k], w["laps"].iloc[k]
    points = to_points(w["base"], w["shap"].iloc[k])
    order = points.abs().sort_values(ascending=False).index[:top]
    return [{"feature": n, "text": reason_text(n, seen, game, info, points[n], w["scale"]),
             "points": round(float(points[n]), 4), "label": as_points(points[n])} for n in order]


def number(v, digits=3):
    return None if v is None or pd.isna(v) else round(float(v), digits)


# Recipe 4: every car at the end of one lap (the latest lap while live), like the real-race Pit strategy view
def assemble(w, status, drivers, player, lap=None):
    laps, info = w["laps"], models()["info"]
    last = int(laps["LapNumber"].max())
    lap = min(lap or last, last)
    outcome, outcome_at = outcome_for(w, status, lap)
    rejoins = rejoins_for(w, outcome_at, drivers)
    battles = battles_for(w, outcome_at, drivers)
    out = []
    for key, mine in laps.groupby("Driver"):
        seen = mine[mine["LapNumber"] <= lap]                     # lapped cars are on an earlier lap
        if seen.empty:
            continue
        c, k = int(key), laps.index.get_loc(seen.index[-1])
        row, d = laps.iloc[k], drivers.get(int(key), {})
        gone = status.get(c) in OUT and seen.index[-1] == mine.index[-1] and row["LapNumber"] < lap   # retired
        pitting = not pd.isna(row["PitInTime"])                   # the lap ended in the pit lane
        chance = None if pitting or gone else float(w["chance"][k])
        out.append({"car": c, "driver": d.get("code", f"C{c:02d}"), "name": d.get("name", ""),
                    "colour": "#" + d.get("colour", "888888"), "team": d.get("team", ""), "player": c == player,
                    "lap": int(row["LapNumber"]), "position": int(row["Position"]), "compound": str(row["Compound"]),
                    "tyre_age": number(row["TyreLife"], 0), "status": OUT.get(status.get(c)) if gone else "PIT" if pitting else "",
                    "pitting_now": pitting, "pit_chance": number(chance),
                    "alert": chance is not None and chance >= info["threshold"],
                    "reasons": [] if chance is None else reasons(w, k, top=3),
                    "next_tyre": {name: number(p) for name, p in zip(models()["tyre_info"]["classes"], w["tyres"][k])},
                    "outcome": outcome.get(c), "rejoin": None if pitting or gone else rejoins.get(c),
                    "battle": None if gone else battles.get(c)})
    out.sort(key=lambda d: (bool(d["status"]) and d["status"] != "PIT", d["position"]))
    leader = laps[laps["LapNumber"] == lap]
    ts = leader.loc[leader["Position"].idxmin(), "TrackStatus"] if len(leader) else "1"
    return {"available": True, "lap": lap, "last_lap": last, "total_laps": int(laps["TotalLaps"].iloc[0]),
            "track_status": status_name(ts),
            "circuit_known": not pd.isna(laps["Circuit"].iloc[0]), "alert_threshold": number(info["threshold"]),
            "scale": w["scale"], "model_ms": w["ms"], "laps_worked_out": len(laps), "outcome_lap": outcome_at,
            "has_outcome": w.get("outcome") is not None, "drivers": out}


def snapshot(s, lock, lap=None):
    w, status, drivers, player = worked_out(s, lock)
    if not s.is_race():
        return {"available": False, "note": "Pit stop predictions are made during races."}
    if w is None:
        return {"available": False, "note": "The predictions appear when the cars finish their first lap."}
    return assemble(w, status, drivers, player, lap)


# Recipe 5: one car - the full explanation at the end of one lap, and its pit stop chance on every lap
def car_detail(s, lock, car, lap=None):
    w, status, drivers, player = worked_out(s, lock)
    if w is None:
        return None
    laps = w["laps"]
    mine = laps[laps["Driver"] == str(car)]
    if mine.empty:
        return None
    upto = mine[mine["LapNumber"] <= (lap or mine["LapNumber"].max())]
    k = laps.index.get_loc((upto if len(upto) else mine).index[-1])
    row, pitting = laps.iloc[k], not pd.isna(laps.iloc[k]["PitInTime"])
    series = [{"lap": int(r["LapNumber"]), "pit_chance": None if not pd.isna(r["PitInTime"]) else number(w["chance"][laps.index.get_loc(i)]),
               "pitted": bool(r["IsStop"]), "compound": str(r["Compound"])} for i, r in mine.iterrows()]
    outcome, outcome_at = outcome_for(w, status, int(row["LapNumber"]), full=True)
    o, mine_outcome = w.get("outcome"), None
    if outcome_at is not None and car in outcome:
        rows = o["rows"]
        done = rows[(rows["Driver"] == str(car)) & (rows["LapNumber"] <= outcome_at)]
        mine_outcome = {**outcome[car], "lap": outcome_at,
                        "laps": [{"lap": int(r["LapNumber"]), **{t: number(o[t][i]) for t in ro.SHOWN}} for i, r in done.iterrows()]}
    # If they pit now (at the end of the last whole lap): where they'd rejoin, and the chances after
    pit = None
    if mine_outcome is not None and not pitting and models().get("pit_now"):
        here = o["rows"][o["rows"]["LapNumber"] == outcome_at]
        booster, info = models()["outcome"]
        pick = models()["tyre_info"]["classes"][int(w["tyres"][k].argmax())]
        now_status = lap_status(here)
        pit = coded(pit_now.what_if(models()["pit_now"], booster, info, here, str(car), now_status, pick, w["scale"]["factor"]), drivers)
        if pit:
            pit["lap"] = outcome_at
    return {"car": car, "driver": drivers.get(car, {}).get("code", f"C{car:02d}"), "lap": int(row["LapNumber"]),
            "position": int(row["Position"]), "pitting_now": pitting,
            "pit_chance": None if pitting else number(w["chance"][k]), "typical_lap_chance": number(sigmoid(w["base"])),
            "reasons": [] if pitting else reasons(w, k),
            "next_tyre": {name: number(p) for name, p in zip(models()["tyre_info"]["classes"], w["tyres"][k])},
            "alert_threshold": number(models()["info"]["threshold"]), "total_laps": int(row["TotalLaps"]),
            "scale": w["scale"], "laps": series, "outcome": mine_outcome, "pit_now": pit}


# Recipe 6: the strategy battle - two cars side by side up to the last whole lap (up to `lap`), as for real races
def duel(s, lock, a, b, lap=None):
    w, status, drivers, player = worked_out(s, lock)
    o = None if w is None else w.get("outcome")
    if o is None or not models().get("battle"):
        return None
    at = outcome_lap(o, status, lap or int(o["rows"]["LapNumber"].max()))
    if at is None:
        return None
    here = o["rows"][o["rows"]["LapNumber"] == at]
    answer = battle.duel(models()["battle"], o["rows"], str(a), str(b), at, lap_status(here), w["scale"]["factor"])
    if answer is None:
        return None
    code = lambda car: drivers.get(int(car), {}).get("code", f"C{int(car):02d}")
    now = answer["now"] and {**answer["now"], **{k: code(answer["now"][k]) for k in ("front", "back", "ahead")}}
    return {**answer, "lap": at, "cars": [a, b], "a": code(a), "b": code(b), "now": now,
            "stints": {code(k): v for k, v in answer["stints"].items()}, "stops": {code(k): v for k, v in answer["stops"].items()}}
