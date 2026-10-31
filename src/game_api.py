"""The F1 25 part of the app: the player's own game sessions, recorded live from the game."""
import asyncio
import json
import math
import os

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

import game_analysis
import game_driver
import game_radio
import game_strategy
import game_telemetry as game
import pit_now

router = APIRouter(prefix="/game", tags=["F1 25 game"])
RECORDER = game.RECORDER


def start():
    """Start listening for the game (the app calls this when it starts)."""
    RECORDER.start()


def session_or_404(sid):
    try:
        s = RECORDER.get(sid)
    except ValueError:
        s = None
    if s is None:
        raise HTTPException(404, f"No game session {sid}. See /game/sessions.")
    return s


def thin(rows, most):
    """Every n-th row, so a chart gets at most `most` points."""
    step = max(1, math.ceil(len(rows) / most))
    return rows[::step]


def session_view(s, latest_events=None):
    """Everything the page shows for one session. The live stream sends only the latest events, to stay small."""
    with RECORDER.lock:
        events = s.events[-latest_events:] if latest_events else s.events
        return {"session": s.summary(), "info": s.info, "race": s.is_race(), "live": RECORDER.is_live(s.uid),
                "leaderboard": s.leaderboard(), "laps": s.player_laps(), "best": s.best_times(),
                "car": s.car_now(), "events": list(reversed(events)), "events_total": len(s.events),
                "classification": s.classification, "setup": s.setup, "tyre_sets": s.tyre_set_list(),
                "lap_chart": s.lap_chart()}


def live_view(s):
    """The live stream's message: the session, and in races the pit stop predictions at the latest lap."""
    view = session_view(s, 200)
    try:
        view["strategy"] = game_strategy.snapshot(s, RECORDER.lock)
    except Exception as err:                  # the live leaderboard and car must keep going even if this fails
        view["strategy"] = {"available": False, "note": f"The pit stop predictions failed on this lap ({type(err).__name__}: {err})."}
    try:                                      # the engineer's own radio calls (the page says each new one once)
        view["radio"] = game_radio.RADIO.calls_for(s, RECORDER.lock, view["strategy"])
    except Exception:
        view["radio"] = []
    return view


class Question(BaseModel):
    text: str
    session: str | None = None               # the session on the page (the live one if not given)


@router.post("/radio/ask")
def radio_ask(q: Question):
    """Team radio: a question to your race engineer (the words you said), answered from your live data -
    in the local AI's words when Ollama is running, else straight from the data."""
    sid = q.session or (str(RECORDER.live_uid) if RECORDER.live_uid else None)
    if not sid:
        raise HTTPException(404, "No session yet: drive in F1 25 first, then ask.")
    return game_radio.RADIO.answer(session_or_404(sid), RECORDER.lock, q.text[:300])


@router.get("/radio/status")
def radio_status():
    """Is the local AI engineer ready (Ollama and its model)? Without it, the engineer answers from the data."""
    return game_radio.ai_status()


@router.get("/status")
def status():
    """Is the app receiving the game? Also shows this PC's address to type into the game's settings, and whether the
    app can be opened from a phone or tablet on your home network (started with start_app_lan.bat)."""
    return {**RECORDER.status(), "lan": os.environ.get("F1_APP_LAN") == "1"}


@router.get("/sessions")
def sessions():
    """Every recorded game session, newest first."""
    out = RECORDER.all_sessions()
    for s in out:
        s["live"] = RECORDER.is_live(s["id"])
    return out


@router.get("/sessions/{sid}")
def session(sid: str):
    """One session: leaderboard, the player's laps, tyres and car, events and the final result."""
    return session_view(session_or_404(sid))


@router.get("/sessions/{sid}/telemetry")
def telemetry(sid: str, lap: int | None = None, compare: int | None = None):
    """The player's speed, throttle, brake, gear and RPM along one lap (and a second lap to compare)."""
    s = session_or_404(sid)
    with RECORDER.lock:
        laps = s.player_laps()
        samples = list(s.samples)
    valid = [l for l in laps if not l["invalid"]] or laps
    if lap is None and valid:
        lap = min(valid, key=lambda l: l["time"])["lap"]           # the fastest lap by default
    elif lap is None and samples:
        lap = samples[-1][1]                                       # no lap done yet: the lap in progress
    # Sessions recorded by older versions also hold the out-lap before the line (negative distances): left out
    trace = lambda n: [dict(zip(game.SAMPLE_COLUMNS, r)) for r in thin([r for r in samples if r[1] == n and r[2] >= 0], 1500)]
    return {"lap": lap, "compare": compare, "laps": laps, "trace": trace(lap) if lap else [],
            "compare_trace": trace(compare) if compare else []}


@router.get("/sessions/{sid}/map")
def track_map(sid: str):
    """The track's shape (from the player's own driving) and where every car is."""
    s = session_or_404(sid)
    with RECORDER.lock:
        outline = s.best_outline()
        latest, seen = [], set()
        racing = sum(1 for d in s.lap_now.values() if d["result_status"] in (2, 3))
        for t, car, x, z in reversed(s.positions):                # the newest position of each car
            if car in seen:
                continue
            seen.add(car)
            d, lap = s.drivers.get(int(car), {}), s.lap_now.get(int(car), {})
            latest.append({"car": int(car), "driver": d.get("code", f"C{int(car):02d}"), "colour": "#" + d.get("colour", "888888"),
                           "x": x, "z": z, "player": int(car) == s.player, "position": lap.get("position"),
                           "in_pit": bool(lap.get("pit_status"))})
            if len(seen) >= max(racing, 1):
                break
        starts = s.info.get("sector_starts") or []
    points = thin(outline, 800)
    marks = []
    for k, start in enumerate(starts[1:], start=2):              # where sectors 2 and 3 begin
        near = min(outline, key=lambda p: abs(p[2] - start), default=None)
        if near:
            marks.append({"sector": k, "x": near[0], "z": near[1]})
    return {"outline": [[p[0], p[1]] for p in points], "cars": latest, "sector_marks": marks}


@router.get("/sessions/{sid}/strategy")
def strategy(sid: str, lap: int | None = Query(None, ge=1)):
    """Races: every car's pit stop chance for the next 3 laps, its top reasons (SHAP) and the likely next tyre,
    from the models trained on real F1, at the end of a lap (the latest one if no lap is given)."""
    return game_strategy.snapshot(session_or_404(sid), RECORDER.lock, lap)


@router.get("/sessions/{sid}/strategy/{car}")
def strategy_car(sid: str, car: int, lap: int | None = Query(None, ge=1)):
    """One car: every reason for its pit stop chance at the end of a lap, and its chance on every lap."""
    detail = game_strategy.car_detail(session_or_404(sid), RECORDER.lock, car, lap)
    if detail is None:
        raise HTTPException(404, f"No finished race laps for car {car} in session {sid}.")
    return detail


@router.get("/sessions/{sid}/duel/{a}/{b}")
def duel(sid: str, a: int, b: int, lap: int | None = Query(None, ge=1)):
    """The strategy battle: two cars side by side up to the end of a lap (the latest whole lap if no lap is given)."""
    answer = game_strategy.duel(session_or_404(sid), RECORDER.lock, a, b, lap)
    if answer is None:
        raise HTTPException(404, f"No finished race laps for both car {a} and car {b} in session {sid}.")
    return answer


@router.get("/sessions/{sid}/engineer")
def engineer(sid: str):
    """The engineer's briefing for a session: the game's own plan (its pit window, where you'd rejoin, your tyre sets
    with the laps each has left, how long the game says each compound lasts and its pace), the real track it is (for
    the real F1 briefing at /engineer/{circuit}) and how this race's length wears the tyres."""
    s = session_or_404(sid)
    sprint = RECORDER.is_sprint(s)
    with RECORDER.lock:
        info, sets, race, wear = dict(s.info), s.tyre_set_list(), s.is_race(), game_analysis.wear_forecast(s)
    track = info.get("track", "")
    circuit = game_strategy.GAME_NAMES.get(track, track)
    scale = game_strategy.race_scale(circuit, info.get("total_laps") or 1, sprint) if race else None
    return {"track": track, "circuit": circuit, "session_type": info.get("session_type"), "race": race, "sprint": sprint,
            "total_laps": info.get("total_laps"), "scale": scale, "track_temp": info.get("track_temp"),
            "weather": info.get("weather"), "rain": info.get("weather") in game.RAIN,
            "pit_window": info.get("pit_window"), "pit_plan": info.get("pit_plan"), "pit_rejoin": info.get("pit_rejoin"), "tyre_sets": sets,
            "forecast": (info.get("forecast") or [])[:4], "wear": wear}


def pit_loss(track):
    """What a stop costs under green at this track (seconds, from the real races), or None for a track not raced."""
    name, m = game_strategy.circuit(track), game_strategy.models()
    if not name or not m.get("pit_now") or not m.get("outcome"):
        return None
    return pit_now.cost(m["pit_now"], m["outcome"][1]["pit_loss"], name, "GREEN")[0]


@router.get("/sessions/{sid}/report")
def report(sid: str):
    """The session's report: your result, places lap by lap, stints and stops, consistency, your pace against the
    cars around you on the same tyre, your theoretical best lap (sectors against pole) and the Time Trial."""
    s = session_or_404(sid)
    with RECORDER.lock:
        return game_analysis.report(s)


@router.get("/sessions/{sid}/corners")
def corners(sid: str, lap: int | None = Query(None, ge=1), compare: int | None = Query(None, ge=1)):
    """Where you lose time: one lap against another (by default your fastest), corner by corner."""
    s = session_or_404(sid)
    with RECORDER.lock:
        laps, samples = s.player_laps(), list(s.samples)
    valid = sorted((l for l in laps if not l["invalid"]), key=lambda l: l["time"])
    lap = lap or (valid[0]["lap"] if valid else None)
    other = compare or next((l["lap"] for l in valid if l["lap"] != lap), None)
    if not lap or not other or other == lap:
        return {"lap": lap, "compare": other, "auto": compare is None, "corners": None}
    rows = lambda n: [dict(zip(game.SAMPLE_COLUMNS, r)) for r in samples if r[1] == n and r[2] is not None and r[2] >= 0]
    found = game_analysis.corners(rows(lap), rows(other)) or {"corners": None}
    return {"lap": lap, "compare": other, "auto": compare is None, **found}


@router.get("/sessions/{sid}/slips")
def slips(sid: str, lap: int | None = Query(None, ge=1)):
    """Lock-ups and wheelspin, on the track map (every lap, or one)."""
    s = session_or_404(sid)
    with RECORDER.lock:
        samples, outline = list(s.samples), s.best_outline()
    return game_analysis.slips(samples, outline, lap)


@router.get("/sessions/{sid}/energy")
def energy(sid: str):
    """Energy on each lap (harvested and used, against the limit in F1 26), where on the lap, and in F1 26 overtake
    mode and active aero."""
    s = session_or_404(sid)
    with RECORDER.lock:
        cs, laps, samples, events = list(s.car_samples), s.player_laps(), list(s.samples), list(s.events)
        length, limit, which = s.info.get("track_length"), s.status_now.get(s.player, {}).get("ers_harvest_limit"), s.game()
    return {"game": which, "energy": game_analysis.energy(cs, laps, length, limit or None),
            "modes": game_analysis.modes(samples, events, s.player)}


@router.get("/sessions/{sid}/switch")
def switch(sid: str):
    """When to switch to the wet tyres (or back to slicks): the game's own pace for your sets, the laps left and
    what a stop costs here."""
    s = session_or_404(sid)
    loss = pit_loss(s.info.get("track", ""))
    with RECORDER.lock:
        return game_analysis.switch(s, loss) or {"call": "none"}


@router.get("/progress")
def progress(game_name: str = Query("F1 25", alias="game")):
    """Your progress: your best lap in each session at each track (one game: F1 25 or F1 26)."""
    return {"game": game_name, "tracks": game_analysis.progress(RECORDER.all_sessions(), game_name)}


@router.get("/setups")
def setups(game_name: str = Query("F1 25", alias="game"), track: str | None = None):
    """Your setups at a track (one game), each with its session's best lap, to compare two."""
    rows = [r for r in RECORDER.setups() if r["game"] == game_name]
    return {"game": game_name, "track": track, "tracks": sorted({r["track"] for r in rows}),
            "sessions": [r for r in rows if track is None or r["track"] == track]}


@router.get("/driver")
def driver(game_name: str = Query("F1 25", alias="game"), track: str | None = None):
    """The driver report: what the model has learned about your driving (one game), and at one track the corners
    where you lose the most. It learns your new sessions each time it's asked (the live one when it ends)."""
    live = {str(RECORDER.live_uid)} if RECORDER.live_uid and RECORDER.is_live(RECORDER.live_uid) else set()
    return game_driver.report(game_name, track, skip=live)


@router.get("/sessions/{sid}/car")
def car(sid: str):
    """The player's car over the session: fuel, ERS and tyre wear (for charts)."""
    s = session_or_404(sid)
    with RECORDER.lock:
        rows = [dict(zip(game.CAR_COLUMNS, r)) for r in thin(list(s.car_samples), 600)]
        return {"now": s.car_now(), "samples": rows}


@router.get("/live")
async def live(seconds: float = 1.0):
    """While you drive: the live session, sent every second (Server-Sent Events)."""
    async def updates():
        while True:
            uid = RECORDER.live_uid if RECORDER.live_uid and RECORDER.is_live(RECORDER.live_uid) else None
            if uid:
                view = await run_in_threadpool(live_view, RECORDER.sessions[uid])
                yield f"data: {json.dumps(view)}\n\n"
            else:
                yield f"data: {json.dumps({'live': False, 'status': RECORDER.status()})}\n\n"
            await asyncio.sleep(max(0.25, seconds))

    return StreamingResponse(updates(), media_type="text/event-stream")
