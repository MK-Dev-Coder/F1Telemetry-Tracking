import asyncio
import json
import math
import os
import sys
import time
from functools import lru_cache

import lightgbm as lgb
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

import battle
import engineer
import features
import game_api
import live_timing
import pit_now
import race_outcome as ro
import train_tyre_model as tyre
from config import DATA_DIR, MODELS_DIR, ROOT
from explain_pit_model import as_points, describe, explain, load_model, sigmoid, to_points

TRAINED_UP_TO = 2025                          # races up to 2025 were used to train the models


# Recipe 1: every lap of every race with all features (built once, then read back from a file).
# Each lap's features only use what was known at the end of that lap, so replaying a race lap by
# lap gives the same answers the app would have given live.
def load_laps():
    laps = tyre.replay_laps()
    for col in features.CATEGORIES:
        laps[col] = laps[col].astype("category")
    if OUTCOME:                                       # the race outcome model's features, and each final result
        laps = ro.add_pit_adjusted(ro.add_features(laps, RESULTS), OUTCOME_INFO["pit_loss"])
    return {race: rows.sort_values(["LapNumber", "Position"]) for race, rows in laps.groupby("RaceId")}


PIT, PIT_INFO = load_model()
TYRE = lgb.Booster(model_file=str(MODELS_DIR / "tyre_model.txt"))
TYRE_INFO = json.loads((MODELS_DIR / "tyre_model.json").read_text(encoding="utf-8"))
OUTCOME, OUTCOME_INFO = ro.load_model() if ro.MODEL_FILE.exists() else (None, None)   # python train_race_model.py
RESULTS = ro.load_results() if OUTCOME else None
PIT_NOW = pit_now.load() if OUTCOME else None                            # what a stop costs (python pit_now.py)
BATTLE = battle.load() if OUTCOME else None                               # what an undercut gains (python battle.py)
ENGINEER = engineer.load() if OUTCOME else None                           # each track's briefing (python engineer.py)
# The race preview (python race_preview.py): the next race, and what it said before each race of 2025 and 2026
PREVIEW_FILE, PREVIEW_TEST_FILE = MODELS_DIR / "preview_next.json", DATA_DIR / "preview_backtest.parquet"
PREVIEW = json.loads(PREVIEW_FILE.read_text(encoding="utf-8")) if PREVIEW_FILE.exists() else None
PREVIEW_TEST = pd.read_parquet(PREVIEW_TEST_FILE) if PREVIEW_TEST_FILE.exists() else None
COLOURS_FILE = DATA_DIR / "team_colours.json"                           # (python build_timing.py; without it, grey)
COLOURS_BY_RACE = json.loads(COLOURS_FILE.read_text(encoding="utf-8")) if COLOURS_FILE.exists() else {}
if RESULTS is not None and RESULTS.empty:              # no final results to work from: no race outcome
    print("No final results found (data/openf1/json): the race outcome model is off.")
    OUTCOME = None
RACES = load_laps()
live_timing.RACES.update(RACES)

app = FastAPI(title="F1 Strategy Predictor",
              description="Pit stop chance (next 3 laps) with SHAP reasons, and the likely next tyre, lap by lap.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])
app.include_router(game_api.router)                                         # F1 25: the player's own sessions
game_api.start()                                                            # listen for the game on port 20777
class WebPages(StaticFiles):
    """The web pages. The browser always checks for a newer version, so an update shows straight away."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


@app.get("/app/f1-26.html", include_in_schema=False)
def f1_26_page():
    """The F1 26 section (the 2026 Season Pack): the same page as F1 25, showing the F1 26 sessions."""
    return FileResponse(ROOT / "web" / "f1-25.html", headers={"Cache-Control": "no-cache"})


app.mount("/app", WebPages(directory=ROOT / "web", html=True), name="web")      # the menu and the pages


def clean(v):
    """Numbers and text that JSON understands (NaN -> null)."""
    if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NA:
        return None
    if hasattr(v, "item"):
        v = v.item()
    return round(v, 3) if isinstance(v, float) else v


def race_rows(race_id):
    if race_id not in RACES:
        raise HTTPException(404, f"No race {race_id}. See /races for the list.")
    return RACES[race_id]


def track_status(rows):
    """The leader's track status on this lap (lapped cars reach the same lap number later)."""
    leader = rows[rows["Position"] == rows["Position"].min()]
    status = (leader if len(leader) else rows)["TrackStatus"].astype(str)
    for code, name in [("5", "RED FLAG"), ("4", "SAFETY CAR"), ("[67]", "VSC"), ("2", "YELLOW")]:
        if status.str.contains(code).any():
            return name
    return "GREEN"


# Recipe 2: everything for one lap - the heart of replay mode
def lap_snapshot(race_id, lap, reveal=False):
    rows = race_rows(race_id)
    rows = rows[rows["LapNumber"] == lap]
    if rows.empty:
        raise HTTPException(404, f"No lap {lap} in {race_id}.")
    start = time.perf_counter()
    chance, shap_values, base = explain(PIT, PIT_INFO, rows)                # pit chance + SHAP
    tyres = TYRE.predict(rows[TYRE_INFO["features"]])                       # chance of each compound
    model_ms = 1000 * (time.perf_counter() - start)

    drivers = []
    for i, (idx, row) in enumerate(rows.iterrows()):
        pitting = bool(row["IsStop"])                                       # coming in at the end of this lap
        points = to_points(base, shap_values.loc[idx])
        top = points.abs().sort_values(ascending=False).index[:3]
        entry = {
            "driver": row["Driver"], "team": row["Team"], "position": clean(row["Position"]),
            "compound": str(row["Compound"]), "tyre_age": clean(row["TyreLife"]),
            "gap_ahead": clean(row["GapAhead"]), "pitting_now": pitting,
            "pit_chance": None if pitting else clean(chance[i]),
            "alert": (not pitting) and bool(chance[i] >= PIT_INFO["threshold"]),
            "reasons": [] if pitting else [{"text": describe(n, row, PIT_INFO, points[n]),
                                            "points": clean(points[n]), "label": as_points(points[n])} for n in top],
            "next_tyre": {c: clean(tyres[i][k]) for k, c in enumerate(TYRE_INFO["classes"])},
        }
        if reveal:                                                          # what really happened (spoilers)
            entry["actual"] = {"pits_in_next_3_laps": bool(row["PitSoon"]),
                               "fitted": row["Next"] if pitting else None}
        drivers.append(entry)
    return {"race_id": race_id, "lap": lap, "total_laps": int(race_rows(race_id)["LapNumber"].max()),
            "track_status": track_status(rows), "alert_threshold": clean(PIT_INFO["threshold"]),
            "model_ms": round(model_ms, 2), "total_ms": round(1000 * (time.perf_counter() - start), 2),
            "drivers": drivers}


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/app/")


@app.get("/races")
def races():
    out = []
    for race_id, rows in RACES.items():
        first = rows.iloc[0]
        out.append({"race_id": race_id, "year": int(first["Year"]), "round": int(first["Round"]),
                    "name": first["EventName"], "circuit": str(first["Circuit"]), "source": first["Source"],
                    "laps": int(rows["LapNumber"].max()), "drivers": int(rows["Driver"].nunique()),
                    "seen_in_training": int(first["Year"]) <= TRAINED_UP_TO})
    return out


@app.get("/races/{race_id}")
def race(race_id: str):
    rows = race_rows(race_id)
    first = rows.iloc[0]
    status = [{"lap": int(lap), "status": track_status(r)} for lap, r in rows.groupby("LapNumber")]
    stops = rows[rows["IsStop"]].sort_values("LapNumber")
    return {"race_id": race_id, "name": first["EventName"], "year": int(first["Year"]),
            "circuit": str(first["Circuit"]), "laps": int(rows["LapNumber"].max()),
            "drivers": sorted(rows["Driver"].unique()), "seen_in_training": int(first["Year"]) <= TRAINED_UP_TO,
            "track_status": [s for s in status if s["status"] != "GREEN"],
            "pit_stops": [{"driver": r["Driver"], "lap": int(r["LapNumber"]), "from": str(r["Compound"]), "to": r["Next"]}
                          for _, r in stops.iterrows()],
            "source": first["Source"], "track": live_timing.track(str(first["Circuit"]), race_id),
            "names": {code: live_timing.NAMES.get(code, code) for code in rows["Driver"].unique()},
            "cards": live_timing.cards(race_id, rows["Driver"].unique()),          # number, name, team, photo
            "start_positions": {r["Driver"]: clean(r["Position"]) for _, r in rows[rows["LapNumber"] == 1].iterrows()},
            "lap_list": lap_numbers(race_id)}


@app.get("/drivers/cards")
def driver_cards(year: int = Query(..., ge=2022, le=2030)):
    """Every driver's car number, name, team and photo as at the end of a season (the game pages: F1 25 has the 2025
    grid, F1 26 the 2026 one)."""
    codes = sorted({code for cards in live_timing.driver_lists().values() for code in cards})
    return live_timing.cards(f"{year}_30", codes)


@app.get("/races/{race_id}/laps/{lap}")
def lap(race_id: str, lap: int, reveal: bool = False):
    """Every driver at the end of this lap: pit chance for the next 3 laps, top 3 reasons, likely next tyre."""
    return lap_snapshot(race_id, lap, reveal)


@app.get("/races/{race_id}/laps/{lap}/drivers/{driver}")
def lap_driver(race_id: str, lap: int, driver: str):
    """The full explanation for one driver on one lap: every feature's effect."""
    rows = race_rows(race_id)
    rows = rows[(rows["LapNumber"] == lap) & (rows["Driver"] == driver.upper())]
    if rows.empty:
        raise HTTPException(404, f"No lap {lap} for {driver} in {race_id}.")
    chance, shap_values, base = explain(PIT, PIT_INFO, rows)
    row, points = rows.iloc[0], to_points(base, shap_values.iloc[0])
    order = points.abs().sort_values(ascending=False).index
    return {"race_id": race_id, "lap": lap, "driver": row["Driver"], "pit_chance": clean(chance[0]),
            "typical_lap_chance": clean(sigmoid(base)),
            "reasons": [{"feature": n, "value": clean(row[n]) if n not in features.CATEGORIES else str(row[n]),
                         "text": describe(n, row, PIT_INFO, points[n]), "points": clean(points[n]),
                         "label": as_points(points[n])} for n in order]}


@app.get("/races/{race_id}/drivers/{driver}")
def driver_race(race_id: str, driver: str, reveal: bool = False):
    """One driver's pit chance on every lap of the race (for a chart)."""
    rows = race_rows(race_id)
    rows = rows[rows["Driver"] == driver.upper()].sort_values("LapNumber")
    if rows.empty:
        raise HTTPException(404, f"No driver {driver} in {race_id}.")
    chance, _, _ = explain(PIT, PIT_INFO, rows)
    laps = []
    for i, (_, row) in enumerate(rows.iterrows()):
        entry = {"lap": int(row["LapNumber"]), "position": clean(row["Position"]), "compound": str(row["Compound"]),
                 "tyre_age": clean(row["TyreLife"]), "pit_chance": None if row["IsStop"] else clean(chance[i])}
        if reveal:
            entry["pitted"] = bool(row["IsStop"])
        laps.append(entry)
    return {"race_id": race_id, "driver": driver.upper(), "alert_threshold": clean(PIT_INFO["threshold"]), "laps": laps}


# Recipe 3: the race outcome - each driver's chance to win, finish on the podium and score points, lap by lap.
# A whole race is worked out once (about 0.1 s), then kept; the SHAP reasons are worked out per lap.
@lru_cache(maxsize=32)
def outcome_race(race_id):
    rows = race_rows(race_id)
    rows = rows[rows["Position"].notna()]
    s = ro.summary(ro.chances(OUTCOME, OUTCOME_INFO, rows))
    return rows[["LapNumber", "Driver"]].assign(win=s["win"], podium=s["podium"], points=s["points"],
                                                 likely=s["likely"], expected=s["expected"], spot=list(s["spot"]))


@lru_cache(maxsize=256)
def outcome_why(race_id, lap):
    """SHAP values for the win, podium and points chances of every driver on one lap."""
    rows = race_rows(race_id)
    rows = rows[(rows["LapNumber"] == lap) & rows["Position"].notna()]
    return rows, {target: ro.explain(OUTCOME, OUTCOME_INFO, rows, k) for target, k in ro.SHOWN.items()}


def outcome_at(race_id, lap):
    """The lap the race outcome is shown for: this one, or the last one before it with positions in the timing
    (the last lap of most OpenF1 races has none). None before the first."""
    laps = outcome_race(race_id)["LapNumber"]
    done = laps[laps <= lap]
    return None if done.empty else int(done.max())


def outcome_lap(race_id, lap, reveal=False, reasons=False):
    """Every driver's chances at the end of one lap (and the lap they are from). With reasons: the main reason
    (SHAP) for the biggest prize still in reach - about 0.2 s, so only when the page shows them."""
    at = outcome_at(race_id, lap) if OUTCOME else None
    if at is None:
        return {}, None
    table = outcome_race(race_id)
    table = table[table["LapNumber"] == at].set_index("Driver")
    rows = race_rows(race_id)
    rows = rows[(rows["LapNumber"] == at) & rows["Position"].notna()]
    why = outcome_why(race_id, at)[1] if reasons else None
    out = {}
    for idx, row in rows.iterrows():
        c = table.loc[row["Driver"]]
        target = next((t for t in ro.SHOWN if c[t] >= 0.1), "points")     # win, else podium, else points
        out[row["Driver"]] = {**{t: clean(c[t]) for t in ro.SHOWN}, "likely": clean(c["likely"]),
                              "expected": clean(c["expected"]), "for": target, "reason": None}
        if why:
            values, _, words = why[target]
            points = values.loc[idx]
            top = points.abs().idxmax()
            out[row["Driver"]]["reason"] = {"text": ro.describe(top, words.loc[idx], points[top], ro.SHOWN[target]),
                                            "points": clean(points[top]), "label": as_points(points[top])}
        if reveal:                                                          # the real result (spoiler)
            out[row["Driver"]]["final"] = clean(row["Final"])
    return out, at


# Recipe 4: the live timing screen - timing for every driver plus the models, at the end of one lap
def live_snapshot(race_id, lap, reveal=False, reasons=False):
    start = time.perf_counter()
    model = lap_snapshot(race_id, lap, reveal)                          # 404 if the race or lap doesn't exist
    timing = live_timing.snapshot(race_id, lap)
    began = time.perf_counter()
    outcome, outcome_from = outcome_lap(race_id, lap, reveal, reasons)
    outcome_ms = 1000 * (time.perf_counter() - began)
    rejoins, pit_cost = pit_rejoins(race_id, lap, model["track_status"])
    battles = battle.lap_battles(BATTLE, lap_cars(race_id, lap), model["track_status"]) if OUTCOME else {}
    by_driver = {d["driver"]: d for d in model["drivers"]}
    for d in timing["drivers"]:
        m = by_driver.get(d["driver"])
        if m:
            d.update({k: m[k] for k in ("pit_chance", "alert", "reasons", "next_tyre", "pitting_now")})
            if reveal:
                d["actual"] = m.get("actual")
        d["outcome"] = outcome.get(d["driver"])
        gone = d.get("status") in ("OUT", "FINISHED", "NO DATA")
        d["rejoin"] = None if d.get("pitting_now") or gone else rejoins.get(d["driver"])
        d["battle"] = None if gone else battles.get(d["driver"])          # against the car just ahead
    return {"race_id": race_id, "pit_cost": pit_cost, "total_laps": model["total_laps"], "track_status": model["track_status"],
            "alert_threshold": model["alert_threshold"], "model_ms": model["model_ms"], "outcome_ms": round(outcome_ms, 2),
            "has_outcome": bool(OUTCOME), "outcome_lap": outcome_from, "outcome_reasons": reasons, "total_ms": round(1000 * (time.perf_counter() - start), 2), **timing}


def lap_cars(race_id, lap):
    """The cars with a place on this lap (with every feature)."""
    rows = race_rows(race_id)
    return rows[(rows["LapNumber"] == lap) & rows["Position"].notna()]


def pit_rejoins(race_id, lap, status):
    """Where every car would rejoin if it pitted at the end of this lap, and what a stop costs now."""
    rows = lap_cars(race_id, lap)
    if not OUTCOME or rows.empty:
        return {}, None
    loss, (low, high), kind = pit_now.cost(PIT_NOW, OUTCOME_INFO["pit_loss"], rows["Circuit"].iloc[0], status)
    return pit_now.rejoin_all(rows, loss), {"seconds": round(loss, 1), "range": [round(low, 1), round(high, 1)], "under": kind}


@app.get("/races/{race_id}/laps/{lap}/drivers/{driver}/pit_now")
def lap_driver_pit_now(race_id: str, lap: int, driver: str):
    """If this driver pitted at the end of this lap: what the stop costs, where they'd rejoin and the gaps to the cars
    around, and their win, podium and points chances now and after (the race outcome model asked again)."""
    need_outcome()
    rows = lap_cars(race_id, lap)
    mine = rows[rows["Driver"] == driver.upper()]
    if mine.empty:
        raise HTTPException(404, f"No lap {lap} for {driver} in {race_id}.")
    tyres = TYRE.predict(mine[TYRE_INFO["features"]])[0]                     # the tyre they'd fit (the tyre model)
    compound = TYRE_INFO["classes"][int(tyres.argmax())]
    answer = pit_now.what_if(PIT_NOW, OUTCOME, OUTCOME_INFO, rows, driver.upper(), track_status(rows), compound)
    if answer is None:
        raise HTTPException(404, f"The places after a stop can't be counted on lap {lap}: cars are missing from the timing.")
    return {"race_id": race_id, "lap": lap, **answer}


@app.get("/races/{race_id}/duel/{a}/{b}")
def duel(race_id: str, a: str, b: str, lap: int = Query(..., ge=1)):
    """The strategy battle: two drivers side by side up to the end of this lap - the gap between them lap by lap,
    their places, tyres and stops, and now: who is ahead, the battle forecast and the undercut threat."""
    need_outcome()
    rows = race_rows(race_id)
    answer = battle.duel(BATTLE, rows, a.upper(), b.upper(), lap, track_status(rows[rows["LapNumber"] == lap]))
    if answer is None:
        raise HTTPException(404, f"No laps for both {a} and {b} in {race_id} up to lap {lap}.")
    return {"race_id": race_id, "lap": lap, **answer}


@app.get("/races/{race_id}/live/{lap}")
def live(race_id: str, lap: int, reveal: bool = False, reasons: bool = False):
    """The live timing screen at the end of a lap: timing, sectors, mini-sectors, stints, messages, pit chance,
    and each driver's chance to win, finish on the podium and score points (reasons=true: with the main reason)."""
    return live_snapshot(race_id, lap, reveal, reasons)


def need_outcome():
    if not OUTCOME:
        raise HTTPException(503, "No race outcome model yet: run python train_race_model.py")


@app.get("/races/{race_id}/outcome")
def race_outcome_chances(race_id: str, reveal: bool = False):
    """Every driver's win, podium and points chance and likely finish on every lap (for the charts)."""
    need_outcome()
    table, laps = outcome_race(race_id), lap_numbers(race_id)
    drivers = {}
    for code, mine in table.groupby("Driver"):
        mine = mine.set_index("LapNumber").reindex(laps)
        drivers[code] = {t: [clean(v) for v in mine[t]] for t in [*ro.SHOWN, "likely", "expected"]}
    answer = {"race_id": race_id, "laps": laps, "drivers": drivers}
    if reveal:
        answer["final"] = {r["Driver"]: clean(r["Final"]) for _, r in race_rows(race_id).drop_duplicates("Driver").iterrows()}
    return answer


@app.get("/races/{race_id}/laps/{lap}/drivers/{driver}/outcome")
def lap_driver_outcome(race_id: str, lap: int, driver: str, reveal: bool = False):
    """One driver on one lap: the win, podium and points chances with every reason (SHAP), and the chance of
    each finishing position."""
    need_outcome()
    table, at = outcome_race(race_id), outcome_at(race_id, lap)
    on_lap = table[table["LapNumber"] == at]
    mine = on_lap[on_lap["Driver"] == driver.upper()]
    if mine.empty:
        raise HTTPException(404, f"No lap {lap} for {driver} in {race_id}.")
    c = mine.iloc[0]
    rows, why = outcome_why(race_id, at)
    i = list(rows["Driver"]).index(c["Driver"])
    row, out = rows.iloc[i], {}
    for target, k in ro.SHOWN.items():
        values, start, words = why[target]
        points = values.iloc[i]
        model = start + points.sum()                                       # the average driver + every reason
        out[target] = {"chance": clean(c[target]), "start": clean(start), "model": clean(model),
                       "fitted": clean(c[target] - model),                 # the step that makes the field add up
                       "reasons": [{"feature": n, "text": ro.describe(n, words.iloc[i], points[n], k), "points": clean(points[n]),
                                    "label": as_points(points[n])} for n in points.abs().sort_values(ascending=False).index]}
    places = int(max(len(on_lap), rows["Position"].max()))                 # (more if the timing misses some cars)
    answer = {"race_id": race_id, "lap": lap, "from_lap": at, "driver": c["Driver"], "position": clean(row["Position"]),
              "likely": clean(c["likely"]), "expected": clean(c["expected"]),
              "spot": [clean(v) for v in c["spot"][:places]], "why": out}
    if reveal:
        answer["final"] = clean(row["Final"])
    return answer


# Recipe 5: the race preview - before a race, each driver's chance to win, finish on the podium and score points
def who(code, race_id=None):
    colours = COLOURS_BY_RACE.get(race_id) or (COLOURS_BY_RACE[max(COLOURS_BY_RACE)] if COLOURS_BY_RACE else {})
    return {"name": live_timing.NAMES.get(code, code), "colour": "#" + str(colours.get(code, "888888"))}


def preview_record():
    """How the preview did before each race of 2025 and 2026: who it favoured, and who won."""
    races, out = [], {}
    for race_id, race in PREVIEW_TEST.groupby("RaceId"):
        won = race[race["Final"] == 1]
        entry = {"race_id": race_id, "name": race["EventName"].iloc[0], "winner": won["Driver"].iloc[0] if len(won) else None}
        for way in ("before qualifying", "after qualifying"):
            fav = race.loc[race[f"{way}|win"].idxmax()]
            entry[way] = {"favourite": fav["Driver"], "chance": clean(fav[f"{way}|win"]),
                          "winner_chance": clean(won[f"{way}|win"].iloc[0]) if len(won) else None}
        races.append(entry)
    for way in ("before qualifying", "after qualifying"):
        right = sum(r[way]["favourite"] == r["winner"] for r in races)
        out[way] = {"favourite_won": right, "races": len(races)}
    return {"summary": out, "races": races}


@app.get("/preview")
def preview():
    """The next race: each driver's chance to win, finish on the podium and score points before qualifying, with
    the reasons (SHAP), and how the preview did before each race of 2025 and 2026."""
    if PREVIEW is None:
        raise HTTPException(503, "No race preview yet: run python race_preview.py")
    drivers = [{**d, **who(d["driver"])} for d in PREVIEW["drivers"]]
    past = pd.Timestamp(PREVIEW["date"]).date() < pd.Timestamp.now(tz="UTC").date()   # run race_preview.py again
    return {**PREVIEW, "drivers": drivers, "is_past": past, **(preview_record() if PREVIEW_TEST is not None else {})}


@app.get("/preview/{race_id}")
def preview_race(race_id: str):
    """What the preview said before one race of 2025 or 2026 (trained on the races before it), and the result."""
    if PREVIEW_TEST is None:
        raise HTTPException(503, "No race preview yet: run python race_preview.py")
    race = PREVIEW_TEST[PREVIEW_TEST["RaceId"] == race_id].sort_values("before qualifying|win", ascending=False)
    if race.empty:
        raise HTTPException(404, f"The preview was tested on the races of 2025 and 2026, not {race_id}.")
    ways = ("before qualifying", "after qualifying")
    return {"race_id": race_id, "name": race["EventName"].iloc[0],
            "drivers": [{"driver": r["Driver"], **who(r["Driver"], race_id), "team": r["Team"], "grid": clean(r["Grid"]),
                         "final": clean(r["Final"]),
                         **{w: {t: clean(r[f"{w}|{t}"]) for t in [*ro.SHOWN, "likely"]} for w in ways}} for _, r in race.iterrows()]}


# Recipe 6: the engineer's briefing for a track - how long each tyre lasts there (on a day this hot, in a race this
# long), the strategies that worked, the track's facts, and the race prediction (the preview for the next race)
def need_engineer():
    if ENGINEER is None:
        raise HTTPException(503, "No engineer briefing yet: run python engineer.py")


def event_name(race_id):
    return RACES[race_id]["EventName"].iloc[0] if race_id in RACES else race_id


@app.get("/engineer")
def engineer_tracks():
    """Every track with real races in the data, and which one is next."""
    need_engineer()
    nxt = PREVIEW["circuit"] if PREVIEW else None
    tracks = [{"circuit": c, "name": PREVIEW["name"] if c == nxt else event_name(t["races"][-1]), "races": len(t["races"]),
               "last": t["races"][-1], "laps": t["race_info"][t["races"][-1]]["laps"], "next": c == nxt}
              for c, t in ENGINEER["tracks"].items()]
    return {"next": nxt, "next_name": PREVIEW["name"] if PREVIEW else None, "next_date": PREVIEW["date"] if PREVIEW else None,
            "tracks": sorted(tracks, key=lambda t: t["name"])}


@app.get("/engineer/{circuit}")
def engineer_track(circuit: str, track_temp: float | None = Query(None, ge=-10, le=70), rain: bool = False,
                   distance: int = Query(100, ge=20, le=100), game_laps: int | None = Query(None, ge=1, le=100),
                   before: str | None = None):
    """The briefing for one track: tyres (how long each lasts on a day this hot, and in a race of `distance` % or of
    `game_laps` laps, the game's shorter races), strategies, the winners, the track's facts and the race prediction.
    before=race id: only the races here before that one (a replay of that race: what was known then, no spoilers),
    and the prediction is what the preview said before it."""
    need_engineer()
    b = engineer.brief(ENGINEER, circuit, track_temp, rain, distance, game_laps, before)
    if b is None:
        raise HTTPException(404, f"No real races at {circuit}. See /engineer for the tracks.")
    nxt = PREVIEW and PREVIEW["circuit"] == circuit and before is None
    b["name"] = PREVIEW["name"] if nxt else event_name(before if before in RACES else ENGINEER["tracks"][circuit]["races"][-1])
    if b.get("empty"):
        b["prediction"] = None
        return b
    b["winners"] = [{**w, **who(w["driver"], w["race_id"])} for w in b["winners"]]
    if before is not None:                                # what the preview said before this race (2025 and 2026)
        tested = PREVIEW_TEST is not None and (PREVIEW_TEST["RaceId"] == before).any()
        b["prediction"] = {"kind": "this", **preview_race(before)} if tested else None
    elif nxt:
        b["prediction"] = {"kind": "next", "name": PREVIEW["name"], "date": PREVIEW["date"], "version": PREVIEW["version"],
                           "drivers": [{"driver": d["driver"], "team": d["team"], **{t: d[t] for t in (*ro.SHOWN, "likely")},
                                        **who(d["driver"])} for d in PREVIEW["drivers"]]}
    elif PREVIEW_TEST is not None and (PREVIEW_TEST["RaceId"] == b["races"][-1]).any():   # what it said before the last race here
        b["prediction"] = {"kind": "past", **preview_race(b["races"][-1])}
    else:
        b["prediction"] = None
    return b


def lap_numbers(race_id):
    """The laps that have data (a few races have whole laps missing, e.g. Miami 2025 laps 2-24)."""
    return [int(n) for n in sorted(race_rows(race_id)["LapNumber"].unique())]


# One lap every few seconds, as if the race were live (Server-Sent Events). Each lap is worked out in a
# worker thread, so a running replay doesn't hold up the server's other requests.
def lap_seconds(race_id, lap):
    """How long the leader took for this lap (for real-time replay): the field's median if the leader's is missing,
    and at most 4 minutes (a red flag lap can last an hour)."""
    rows = race_rows(race_id)
    here = rows[rows["LapNumber"] == lap]
    lead = here.loc[here["Position"] == here["Position"].min(), "LapTime"].dropna()
    secs = lead.iloc[0] if len(lead) else here["LapTime"].median()
    return float(min(max(secs, 30), 240)) if pd.notna(secs) else 90.0


# speed: 1 = real time (each lap takes as long as it took the leader), 2 = twice as fast...; without it, `seconds`
# between laps. Each lap says when the next one comes (next_in), so the page can show it.
def stream(race_id, start, seconds, make_snapshot, speed=None):
    todo = [n for n in lap_numbers(race_id) if n >= start]

    async def laps():
        for i, n in enumerate(todo):
            snap = await run_in_threadpool(make_snapshot, n)
            wait = None if i + 1 == len(todo) else lap_seconds(race_id, todo[i + 1]) / speed if speed else seconds
            snap["next_in"] = None if wait is None else round(wait, 2)
            yield f"data: {json.dumps(snap)}\n\n"
            if wait:
                await asyncio.sleep(wait)

    return StreamingResponse(laps(), media_type="text/event-stream")


@app.get("/races/{race_id}/live_replay")
async def live_replay(race_id: str, start: int = 1, seconds: float = 2.0, reveal: bool = False, reasons: bool = False,
                      speed: float | None = Query(None, gt=0, le=100)):
    """Replay mode for the live timing screen (speed=1: real time, each lap as long as it really took)."""
    return stream(race_id, start, seconds, lambda n: live_snapshot(race_id, n, reveal, reasons), speed)


@app.get("/races/{race_id}/replay")
async def replay(race_id: str, start: int = 1, seconds: float = 2.0, reveal: bool = False):
    """Replay mode for the simple page."""
    return stream(race_id, start, seconds, lambda n: lap_snapshot(race_id, n, reveal))


if __name__ == "__main__":
    # python api.py --lan (start_app_lan.bat): also open to phones and tablets on your home network, for the dash
    lan = "--lan" in sys.argv[1:]
    os.environ["F1_APP_LAN"] = "1" if lan else "0"
    uvicorn.run(app, host="0.0.0.0" if lan else "127.0.0.1", port=8000)   # this app object: not loaded a second time
