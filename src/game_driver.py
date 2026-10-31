"""F1 25 / F1 26: the driver report. A model that learns your driving from every corner of every clean lap you've
driven, and keeps learning as you drive more.

Each row is one corner of one lap against the same corner on your best lap of that session: how much earlier you
braked, how much slower you were in the corner, how much later you were back on full throttle, how much you coasted
and worked the wheel, whether a wheel locked or spun, and the conditions (tyre wear, tyre age, lap, a race or not).
The target is the time that corner cost you (from its braking point to the next corner's). A LightGBM model learns
how your habits turn into lost time; SHAP says how much each habit costs you on a lap. It is tested on sessions it
didn't learn from (in five groups of sessions) against a plain average.

The rows of each session are kept in data/driver/ (made from your recordings, never changing them); a new or longer
session is added, and the model learns again, the next time the report is asked for. So are your race starts: your
reaction time when the lights go out and the places you made on lap 1."""
import json
import threading
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import game_analysis as ga
import game_telemetry as game
from config import DATA_DIR

DRIVER_DIR = DATA_DIR / "driver"             # the report's own files: rows.parquet, starts.json, history.json
VERSION = 2                     # the row format: rows made by an older version are made again
MIN_CORNERS, MIN_SESSIONS = 120, 2
HABITS = {                      # the habits the model weighs, and what each measures
    "brake_early": "Braking point",         # m earlier than on your best lap (+)
    "speed_loss": "Corner speed",           # km/h slower at the slowest point of the corner (+)
    "throttle_late": "Back on the throttle",  # m later to full throttle after the corner (+)
    "coast_more": "Coasting",               # m more with neither throttle nor brake, braking point to exit (+)
    "steer_more": "Steering",               # more wheel work through the corner (+: more corrections)
    "lock": "Lock-ups",                     # a wheel locked into the corner (1)
    "spin": "Wheelspin",                    # a rear wheel spun on the way out (1)
}
CONTEXT = ["corner_speed", "wear", "tyre_age", "lap", "race", "track_temp", "trail"]
FEATURES = list(HABITS) + CONTEXT
MODEL = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.9, verbose=-1)
_lock, _memo = threading.Lock(), {}


# ---- one session's corners ------------------------------------------------------------------------------------
def _arrays(df):
    """One lap's samples along the lap: distance (always going up), speed, pedals, steering, wheel slip, time."""
    df = df[df["distance"].notna() & (df["distance"] >= 0)].sort_values("distance")
    if len(df) < 50:
        return None
    d = df["distance"].to_numpy(float)
    keep = np.concatenate([[True], np.diff(d) > 0])
    col = lambda k: df[k].to_numpy(float)[keep] if k in df and df[k].notna().any() else None
    t = df["lap_time"] if "lap_time" in df and df["lap_time"].notna().all() else df["t"] - df["t"].iloc[0]
    return {"d": d[keep], "v": col("speed"), "brake": np.nan_to_num(col("brake")), "thr": np.nan_to_num(col("throttle")),
            "steer": col("steer"), "t": t.to_numpy(float)[keep],
            "slip": None if col("slip_FL") is None else np.vstack([col(f"slip_{w}") for w in ("RL", "RR", "FL", "FR")])}


def _zone(L, a, b):
    return (L["d"] >= a) & (L["d"] <= b)


def _coast(L, a, b):
    """Metres between a and b with neither throttle nor brake on."""
    z = _zone(L, a, b)
    if z.sum() < 2:
        return 0.0
    d, idle = L["d"][z], (L["thr"][z] < 0.1) & (L["brake"][z] < 0.1)
    return float(np.sum(np.diff(d) * idle[1:]))


def _steer_work(L, a, b):
    """How much the wheel moved between a and b (the sum of every change: more corrections, more work)."""
    if L["steer"] is None:
        return None
    z = _zone(L, a, b)
    return float(np.abs(np.diff(L["steer"][z])).sum()) if z.sum() > 2 else None


def _slipped(L, a, b, kind):
    """Did a wheel lock under braking (kind lock) or a rear wheel spin on the throttle (spin) between a and b, for three
    samples or more?"""
    if L["slip"] is None:
        return None
    z = _zone(L, a, b)
    s = np.nan_to_num(L["slip"][:, z])
    if kind == "lock":
        hit = (s.min(axis=0) < ga.LOCK) & (L["brake"][z] > 0.2)
    else:
        hit = (s[:2].max(axis=0) > ga.SPIN) & (L["thr"][z] > 0.3)
    return int(hit.sum() >= 3)


def session_rows(sid, state, tele):
    """Every corner of every clean lap of a session against your best lap of it. state = the session's saved
    session.json, tele = its telemetry table."""
    laps = {int(n): r for n, r in state["laps"].get(str(state["player"]), {}).items()}
    info = state["info"]
    race = info.get("session_type_id") in (15, 16, 17)
    clean = ga.clean_laps(laps, race)
    if len(clean) < 2 or tele is None or tele.empty:
        return []
    by_lap = {int(n): g for n, g in tele.groupby("lap")}
    ref_n = min(clean, key=lambda n: clean[n]["time"])
    R = _arrays(by_lap[ref_n]) if ref_n in by_lap else None
    if R is None:
        return []
    apexes = ga.find_corners(R)
    if not apexes:
        return []
    starts = [(ga._brake_point(R, x) or x - 150) - 30 for x in apexes]
    clock = lambda L, x: float(np.interp(x, L["d"], L["t"]))
    rows = []
    for n in sorted(clean):
        L = _arrays(by_lap[n]) if n != ref_n and n in by_lap else None
        if L is None or L["d"][-1] < 0.9 * R["d"][-1]:        # (a lap with part of it missing)
            continue
        end = min(L["d"][-1], R["d"][-1])
        wear = clean[n].get("tyre_wear")
        for k, x in enumerate(apexes):
            s0, s1 = max(0.0, starts[k]), (starts[k + 1] if k + 1 < len(apexes) else end)
            if s1 <= s0 + 1:
                continue
            lost = (clock(L, s1) - clock(L, s0)) - (clock(R, s1) - clock(R, s0))
            bL, bR = ga._brake_point(L, x), ga._brake_point(R, x)
            vL, vR = ga._at_corner(L, x), ga._at_corner(R, x)
            tL, tR = ga._throttle_point(L, x), ga._throttle_point(R, x)
            a = bL if bL is not None else x - 150
            wR, wL = _steer_work(R, x - 200, x + 200), _steer_work(L, x - 200, x + 200)
            rows.append({
                "sid": sid, "track": info.get("track"), "started": state["started"], "lap": n, "corner": k + 1,
                "at": round(x), "lost": lost,
                "brake_early": bR - bL if bL is not None and bR is not None and abs(bR - bL) <= 120 else np.nan,
                "speed_loss": vR - vL if vL is not None and vR is not None else np.nan,
                "throttle_late": tL - tR if tL is not None and tR is not None and abs(tL - tR) <= 200 else np.nan,
                "coast_more": _coast(L, a, x + 150) - _coast(R, bR if bR is not None else x - 150, x + 150),
                "steer_more": wL - wR if wL is not None and wR is not None else np.nan,
                "lock": _slipped(L, x - 300, x + 30, "lock"), "spin": _slipped(L, x, x + 250, "spin"),
                "trail": int(L["brake"][_zone(L, x - 40, x)].max(initial=0) > 0.1),
                "corner_speed": vR, "wear": float(np.mean(wear)) if wear else np.nan,
                "tyre_age": clean[n].get("tyre_age"), "race": int(race), "track_temp": clean[n].get("track_temp"),
                "brake_at": bL})
    return rows


# ---- your starts ------------------------------------------------------------------------------------------------
def session_start(sid, state, tele):
    """Your start in a race: your reaction time when the lights went out (the game's own in the 2026 format; F1 25
    doesn't send it, so there it's when your car first moved after lights out, if the recording has the start), your
    grid slot, your place after lap 1 and the places that gained. None for a session that isn't a race."""
    info = state.get("info", {})
    if info.get("session_type_id") not in (15, 16, 17):
        return None
    me = state["player"]
    lap1 = state["laps"].get(str(me), {}).get("1", {}).get("position")
    grid = next((c.get("grid") for c in state.get("classification") or [] if c.get("car") == me and c.get("grid")), None)
    lights = next((e["t"] for e in state.get("events") or [] if e["code"] == "LGOT"), None)    # lights out
    estimate = None
    if tele is not None and lights is not None and "speed" in tele:
        d = tele[tele["lap"] == 1].sort_values("t")
        if len(d) and d["t"].iloc[0] - lights <= 0.2:          # the recording has the moment the lights went out
            moved = d[d["speed"] >= 1]
            v = float(moved["t"].iloc[0]) - lights if len(moved) else None
            estimate = round(v, 3) if v is not None and 0.15 <= v <= 1.5 else None    # (else rolling, or stalled)
    game_value = info.get("start_reaction")
    return {"sid": sid, "track": info.get("track"), "started": state.get("started"), "game": _game_of(info),
            "type": info.get("session_type"), "reaction": game_value or estimate,
            "source": "game" if game_value else "estimate" if estimate else None, "game_value": game_value,
            "estimate": estimate, "grid": grid, "lap1": lap1, "gained": grid - lap1 if grid and lap1 else None}


def _starts(items, game_name):
    """Your starts in one game: your reaction times race by race, and the places you made on lap 1. The estimate
    from your car's first movement is checked against the game's own time on the races that have both."""
    mine = sorted((x for x in items if x["game"] == game_name), key=lambda x: x["started"] or 0)
    timed = [x for x in mine if x["reaction"]]
    pairs = [(x["game_value"], x["estimate"]) for x in items if x["game_value"] and x["estimate"]]
    check = {"pairs": len(pairs), "within": sum(abs(a - b) <= 0.05 for a, b in pairs)} if pairs else None
    if not timed:
        return {"races": len(mine), "timed": 0, "check": check} if mine else None
    r = np.array([x["reaction"] for x in timed])
    third = max(1, len(timed) // 3)
    gained = [x for x in timed if x["gained"] is not None]
    mid = float(np.median(r))
    quick = [x["gained"] for x in gained if x["reaction"] <= mid]
    slow = [x["gained"] for x in gained if x["reaction"] > mid]
    best = min(timed, key=lambda x: x["reaction"])
    return {"races": len(mine), "timed": len(timed), "average": round(float(r.mean()), 3), "spread": round(float(r.std()), 3),
            "best": {"reaction": best["reaction"], "track": best["track"], "started": best["started"]},
            "latest": timed[-1]["reaction"], "median": round(mid, 3),
            "improving": bool(len(timed) >= 4 and r[-third:].mean() < r[:third].mean() - 0.02),
            "worse": bool(len(timed) >= 4 and r[-third:].mean() > r[:third].mean() + 0.02),
            "gained": round(float(np.mean([x["gained"] for x in gained])), 1) if gained else None,
            "quick_gain": round(float(np.mean(quick)), 1) if len(quick) >= 2 else None,
            "slow_gain": round(float(np.mean(slow)), 1) if len(slow) >= 2 else None,
            "estimated": sum(x["source"] == "estimate" for x in timed), "check": check,
            "races_list": [{k: x[k] for k in ("track", "started", "reaction", "source", "grid", "lap1", "gained", "type")} for x in timed]}


# ---- every session, kept and added to ------------------------------------------------------------------------
def _sessions(skip=()):
    """Your saved sessions: (id, the stamp that changes when it grows, game, folder)."""
    out = []
    if not game.GAME_DIR.exists():
        return out
    for folder in game.GAME_DIR.iterdir():
        f = folder / "session.json"
        if folder.name == "0" or folder.name in skip or not f.exists() or not (folder / "telemetry.parquet").exists():
            continue
        out.append((folder.name, f.stat().st_mtime_ns, folder))
    return out


def _file(name):
    """One of the report's own files (in DRIVER_DIR, which the tests point somewhere else)."""
    return DRIVER_DIR / name


def _game_of(info):
    return "F1 26" if info.get("formula") == 13 or info.get("regs_2026") else "F1 25"


def refresh(skip=()):
    """Add the corners of sessions that are new (or longer) since last time; drop the ones whose session is gone.
    Returns all the rows, and whether anything changed."""
    DRIVER_DIR.mkdir(parents=True, exist_ok=True)
    old = pd.read_parquet(_file("rows.parquet")) if _file("rows.parquet").exists() else pd.DataFrame()
    if not old.empty and ("version" not in old or (old["version"] != VERSION).any()):
        old = pd.DataFrame()
    known = old.groupby("sid")["stamp"].first().to_dict() if not old.empty else {}
    starts = json.loads(_file("starts.json").read_text(encoding="utf-8")) if _file("starts.json").exists() else {}
    if starts.get("version") != VERSION:                         # (made by an older version: everything again)
        starts, known = {"version": VERSION, "sessions": {}}, {}
    here, parts, changed = set(), [old] if not old.empty and known else [], False
    for sid, stamp, folder in _sessions(skip):
        here.add(sid)
        if known.get(sid) == stamp and starts["sessions"].get(sid, {}).get("stamp") == stamp:
            continue
        state, rows, start = {}, [], None
        try:
            state = json.loads((folder / "session.json").read_text(encoding="utf-8"))
            if not state.get("test"):                             # (test races from the simulator: not you)
                tele = pd.read_parquet(folder / "telemetry.parquet")
                rows, start = session_rows(sid, state, tele), session_start(sid, state, tele)
        except (OSError, ValueError, KeyError) as err:
            print(f"Driver report: session {sid} skipped ({err})")
        starts["sessions"][sid] = {"stamp": stamp, "start": start}
        parts = [p[p["sid"] != sid] for p in parts]
        # A session with no clean corners still gets a row of its own (lost = NaN), so it isn't read again
        new = pd.DataFrame(rows or [{"sid": sid, "lost": np.nan}])
        new["stamp"], new["game"], new["version"] = stamp, _game_of(state.get("info", {})), VERSION
        new["type"] = state.get("info", {}).get("session_type")
        parts.append(new)
        changed = True
    rows = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not rows.empty and not set(rows["sid"]) <= here:            # sessions deleted since
        rows, changed = rows[rows["sid"].isin(here)].reset_index(drop=True), True
    gone = set(starts["sessions"]) - here - set(skip)
    for sid in gone:
        del starts["sessions"][sid]
    if changed and not rows.empty:
        rows.to_parquet(_file("rows.parquet"), index=False)
    if changed or gone:
        _file("starts.json").write_text(json.dumps(starts), encoding="utf-8")
    return rows, changed


def all_starts():
    """Every race start kept by the last refresh."""
    if not _file("starts.json").exists():
        return []
    return [x["start"] for x in json.loads(_file("starts.json").read_text(encoding="utf-8")).get("sessions", {}).values() if x.get("start")]


# ---- the model --------------------------------------------------------------------------------------------------
def learn(rows):
    """Train the model on the corners, test it on each session left out in turn, and say what each habit costs."""
    X, y, groups = rows[FEATURES].astype(float), rows["lost"].astype(float), rows["sid"]
    errors, base = [], []
    if groups.nunique() >= 3:                                     # sessions predicted by a model that never saw them
        for train, test in GroupKFold(n_splits=min(5, groups.nunique())).split(X, y, groups):
            m = lgb.LGBMRegressor(**MODEL).fit(X.iloc[train], y.iloc[train])
            errors.append(float(np.abs(m.predict(X.iloc[test]) - y.iloc[test]).mean()))
            base.append(float(np.abs(y.iloc[train].mean() - y.iloc[test]).mean()))
    model = lgb.LGBMRegressor(**MODEL).fit(X, y)
    shap = model.booster_.predict(X, pred_contrib=True)[:, :-1]       # each feature's part of each corner's time
    return model, pd.DataFrame(shap, columns=FEATURES, index=rows.index), {
        "mae": float(np.mean(errors)) if errors else None, "base": float(np.mean(base)) if base else None,
        "tested_on": len(errors)}


def report(game_name, track=None, skip=()):
    """The driver report for one game: what the model has learned about your driving."""
    with _lock:
        rows, changed = refresh(skip)
        mine = rows[(rows.get("game") == game_name) & rows["lost"].notna()] if not rows.empty else rows
        key = (game_name, len(mine), float(mine["lost"].sum()) if len(mine) else 0.0)
        if key in _memo:
            out = _memo[key]
        else:
            out = _build(mine, game_name)
            _memo.clear()
            _memo[key] = out
        starts = _starts(all_starts(), game_name)
    return {**out, "starts": starts, "track": _track(out, mine, track) if out.get("ready") else None}


def _build(rows, game_name):
    sessions = rows["sid"].nunique() if len(rows) else 0
    laps = rows.groupby(["sid", "lap"]).ngroups if len(rows) else 0
    base = {"game": game_name, "sessions": sessions, "laps": laps, "corners": len(rows), "need": MIN_CORNERS}
    if len(rows) < MIN_CORNERS or sessions < MIN_SESSIONS:
        return {**base, "ready": False}
    rows = rows.reset_index(drop=True)
    model, shap, test = learn(rows)
    lap_key = [rows["sid"], rows["lap"]]
    per_lap = shap.groupby(lap_key).sum()                          # each habit's part of each lap's time
    order = rows.groupby("sid")["started"].first().sort_values()
    habits = []
    for k, name in HABITS.items():
        cost = per_lap[k].clip(lower=0)
        by_session = per_lap[k].groupby(level=0).mean().reindex(order.index).dropna()
        third = max(1, len(by_session) // 3)
        early, late = by_session.iloc[:third].mean(), by_session.iloc[-third:].mean()
        habits.append({"key": k, "name": name, "cost": round(float(cost.mean()), 3), "often": round(float((per_lap[k] > 0.01).mean()), 2),
                       "trend": [{"sid": s, "started": float(order[s]), "cost": round(float(v), 3)} for s, v in by_session.items()],
                       "improving": bool(len(by_session) >= 3 and late < early - 0.01), "worse": bool(len(by_session) >= 3 and late > early + 0.01),
                       "measure": _measure(rows, k)})
    habits.sort(key=lambda h: -h["cost"])
    kind = pd.cut(rows["corner_speed"], [0, 120, 200, 999], labels=["slow", "medium", "fast"])
    types = [{"type": str(t), "corners": int((kind == t).sum()), "lost": round(float(rows.loc[kind == t, "lost"].mean()), 3)}
             for t in ("slow", "medium", "fast") if (kind == t).sum() >= 10]
    style = _style(rows)
    history = _history(game_name, sessions, laps, len(rows), test)
    lost_lap = rows.groupby(lap_key)["lost"].sum()
    return {**base, "ready": True, "habits": habits, "types": types, "style": style, "model": test, "history": history,
            "lap_lost": round(float(lost_lap.mean()), 3), "updated": time.time(),
            "conditions": {c: round(float(per_lap[c].clip(lower=0).mean()), 3) for c in ("wear", "tyre_age", "lap", "track_temp")}}


def _measure(rows, k):
    """What a habit looks like in your laps, in its own units, against your best laps."""
    v = rows[k].dropna()
    if k in ("lock", "spin"):
        laps = rows.groupby(["sid", "lap"])[k].max().dropna()
        return {"per_10_laps": round(float(laps.sum() / max(len(laps), 1) * 10), 1)}
    if not len(v):
        return {}
    return {"median": round(float(v.median()), 1), "worse_share": round(float((v > {"brake_early": 5, "speed_loss": 2, "throttle_late": 10,
                                                                                       "coast_more": 10, "steer_more": 0.1}[k]).mean()), 2)}


def _style(rows):
    """Your driving style in plain numbers (from every clean lap): how alike your braking points are lap to lap, how
    often you trail-brake into the corner, how much you coast, lock up and spin the wheels."""
    spread = rows.groupby(["sid", "corner"])["brake_at"].std().dropna()
    laps = rows.groupby(["sid", "lap"]).agg(lock=("lock", "max"), spin=("spin", "max"))
    return {"brake_spread": round(float(spread.median()), 1) if len(spread) else None,
            "trail": round(float(rows["trail"].mean()), 2),
            "coast": round(float(rows["coast_more"].clip(lower=0).mean()), 1),
            "locks_per_10": round(float(laps["lock"].mean() * 10), 1) if laps["lock"].notna().any() else None,
            "spins_per_10": round(float(laps["spin"].mean() * 10), 1) if laps["spin"].notna().any() else None}


def _history(game_name, sessions, laps, corners, test):
    """The model's learning log: each time it learned more, how much it knew and how well it guessed."""
    log = json.loads(_file("history.json").read_text(encoding="utf-8")) if _file("history.json").exists() else {}
    mine = log.setdefault(game_name, [])
    if not mine or mine[-1]["corners"] != corners:
        mine.append({"at": time.time(), "sessions": sessions, "laps": laps, "corners": corners,
                     "mae": None if test["mae"] is None else round(test["mae"], 4),
                     "base": None if test["base"] is None else round(test["base"], 4)})
        _file("history.json").write_text(json.dumps(log), encoding="utf-8")
    return mine[-30:]


def _track(out, rows, track):
    """At one track: the corners where you lose the most, and the habit behind each."""
    here = rows[rows["track"] == track] if track else rows.iloc[0:0]
    if len(here) < 20:
        return None
    here = here.assign(spot=(here["at"] / 100).round() * 100)          # (corners within 100 m are one place)
    top = here.groupby("spot").agg(lost=("lost", "mean"), laps=("lap", "count"), **{k: (k, "mean") for k in HABITS})
    top = top[top["laps"] >= 3].sort_values("lost", ascending=False).head(4)
    marks = {"brake_early": 5, "speed_loss": 2, "throttle_late": 10, "coast_more": 10}
    corners = []
    for spot, r in top.iterrows():
        why = max(marks, key=lambda k: (r[k] or 0) / marks[k])
        corners.append({"at": int(spot), "lost": round(float(r["lost"]), 3), "laps": int(r["laps"]), "why": why,
                        "value": round(float(r[why]), 1)})
    return {"track": track, "corners": corners}
