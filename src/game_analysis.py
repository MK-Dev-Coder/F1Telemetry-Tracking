"""F1 25 / F1 26: what the game page works out from your recorded sessions. The race report (result, places, stints,
consistency, pace against the cars around you), the theoretical best lap, your own tyre wear forecast, where you lose
time corner by corner, lock-ups and wheelspin, energy per lap (F1 26), overtake mode and active aero (F1 26), the
Time Trial against your rival, when to switch to the wet tyres, your progress and your setups.
Plain functions of a session's recorded data: the tests feed them made-up laps."""
import numpy as np

import game_telemetry as game

WORN = 70                  # % tyre wear: from about here the grip falls away (the line of the wear forecast)
SLOW = 1.07                # a lap over 107% of the usual one isn't a racing lap (a safety car, a spin, traffic)
LOCK, SPIN = -0.12, 0.15   # wheel slip: a front or rear wheel locking under braking, a rear spinning on the throttle
DIP = 25                   # km/h: a corner is where the speed dips by at least this much on both sides
WINDOW = 5.0               # seconds after overtake mode is switched on that a pass still counts for it
COLUMN = {name: k for k, name in enumerate(game.SAMPLE_COLUMNS)}
CAR = {name: k for k, name in enumerate(game.CAR_COLUMNS)}


def _median(values):
    return float(np.median(values)) if len(values) else None


# ---- clean laps, the theoretical best and the race report ----------------------------------------------------
def clean_laps(laps, race=True):
    """The laps that show a car's pace: timed, valid, not in or out of the pits, not the start of a race, and not
    slower than 107% of its usual lap (a safety car, a spin). laps = {lap number: lap record}."""
    timed = {n: r for n, r in laps.items() if r.get("time") and not r.get("invalid") and not r.get("pitted")
             and not (race and n == 1)}
    usual = _median([r["time"] for r in timed.values()])
    return {n: r for n, r in timed.items() if usual and r["time"] <= usual * SLOW}


def theoretical(s):
    """Your best lap, your theoretical best (your best three sectors added up) and each sector against pole's lap."""
    mine = s.laps.get(s.player, {})
    valid = {n: r for n, r in mine.items() if r.get("time") and not r.get("invalid")}
    if not valid:
        return None
    n_best = min(valid, key=lambda n: valid[n]["time"])
    best = valid[n_best]
    pole = None                                             # the fastest valid lap in the session, anyone's
    for car, laps in s.laps.items():
        for n, r in laps.items():
            if r.get("time") and not r.get("invalid") and all(r.get(k) for k in ("s1", "s2", "s3")):
                if pole is None or r["time"] < pole[2]["time"]:
                    pole = (car, n, r)
    session = s.best_times()
    sectors = []
    for k in ("s1", "s2", "s3"):
        good = [(r[k], n) for n, r in valid.items() if r.get(k)]
        t, lap = min(good) if good else (None, None)
        p = pole[2][k] if pole else None
        sectors.append({"sector": int(k[1]), "best": t, "lap": lap, "on_best_lap": best.get(k), "pole": p,
                        "to_pole": round(t - p, 3) if t and p else None, "session_best": session.get(k)})
    ideal = round(sum(x["best"] for x in sectors), 3) if all(x["best"] for x in sectors) else None
    drv = s.drivers.get(pole[0], {}) if pole else {}
    return {"best": {"lap": n_best, "time": best["time"], **{k: best.get(k) for k in ("s1", "s2", "s3")}},
            "theoretical": ideal, "gain": round(best["time"] - ideal, 3) if ideal else None, "sectors": sectors,
            "pole": {"driver": drv.get("code", "?"), "name": drv.get("name"), "colour": "#" + drv.get("colour", "888888"),
                     "lap": pole[1], "time": pole[2]["time"], "you": pole[0] == s.player,
                     "gap": round(best["time"] - pole[2]["time"], 3)} if pole else None}


def report(s):
    """The report of a session: your result, your places lap by lap (and the cars that finished around you), your
    stints and stops, how consistent you were, and your pace against the cars around you on the same tyre."""
    me, race = s.player, s.is_race()
    mine = s.laps.get(me, {})
    row = next((c for c in s.classification if c["car"] == me), None)
    board = next((r for r in s.leaderboard() if r["player"]), None)
    final = row["position"] if row else board["position"] if board else None
    first = next((r.get("position") for n, r in sorted(mine.items()) if r.get("position")), None)
    grid = row["grid"] if row and row.get("grid") else first
    cars = len(s.classification) or len(s.lap_now)
    clean = clean_laps(mine, race)
    times = [r["time"] for r in clean.values()]
    usual = _median(times)
    consistency = None if len(times) < 3 else {
        "laps": len(times), "usual": round(usual, 3), "spread": round(float(np.std(times)), 3),
        "within": sum(abs(t - usual) <= 0.5 for t in times), "best": round(min(times), 3)}
    stops = [n for n, r in sorted(mine.items()) if r.get("pit_entry")]
    if not stops:                                               # sessions recorded before pit entry was kept
        before = 0
        for n, r in sorted(mine.items()):
            if (r.get("pit_stops") or 0) > before:
                stops.append(n)
            before = r.get("pit_stops") or before
    # The cars that finished just ahead and behind you (two each way), and your pace against them on each tyre
    order = {c["car"]: c["position"] for c in s.classification} or {i: d["position"] for i, d in s.lap_now.items()}
    near = [c for c, p in sorted(order.items(), key=lambda kv: kv[1]) if c != me and final and p and abs(p - final) <= 2]
    pace = []
    for car in near:
        theirs = clean_laps(s.laps.get(car, {}), race)
        for comp in sorted({r.get("compound") for r in clean.values()} - {None, "UNKNOWN"}):
            a = [r["time"] for r in clean.values() if r.get("compound") == comp]
            b = [r["time"] for r in theirs.values() if r.get("compound") == comp]
            if len(a) >= 2 and len(b) >= 2:
                d = s.drivers.get(car, {})
                pace.append({"car": car, "driver": d.get("code", f"C{car:02d}"), "name": d.get("name", ""),
                             "colour": "#" + d.get("colour", "888888"), "position": order.get(car), "compound": comp,
                             "you": round(_median(a), 3), "them": round(_median(b), 3), "laps": [len(a), len(b)],
                             "delta": round(_median(a) - _median(b), 3)})
    chart = [c for c in s.lap_chart() if c["player"] or c["car"] in near] if race else []
    return {"race": race, "session_type": s.info.get("session_type"), "track": s.info.get("track"),
            "result": {"position": final, "grid": grid, "cars": cars, "points": row.get("points") if row else None,
                       "status": row.get("result") if row else None, "race_time": row.get("race_time") if row else None,
                       "gained": grid - final if race and grid and final else None},
            "laps": len([r for r in mine.values() if r.get("time")]), "stints": s.stints(me), "stops": stops,
            "consistency": consistency, "pace": pace, "chart": chart, "best": theoretical(s),
            "time_trial": time_trial(s)}


# ---- your own tyre wear forecast ----------------------------------------------------------------------------
def wear_forecast(s):
    """When each tyre of your set reaches 70% wear, from how fast it has worn on this set (a straight line through
    the wear at the end of each lap, and now). None until a set has two points."""
    mine, stints = s.laps.get(s.player, {}), s.stints(s.player)
    if not stints:
        return None
    start = stints[-1]["from"]
    pts = [(float(n), r["tyre_wear"]) for n, r in sorted(mine.items()) if n >= start and r.get("tyre_wear")]
    now, lap = s.damage_now.get(s.player, {}).get("wear"), s.lap_now.get(s.player, {})
    length = s.info.get("track_length") or 0
    if now and lap.get("lap") and lap.get("result_status") == 2 and length:      # the lap in progress, while racing
        x = lap["lap"] - 1 + max(0.0, min(1.0, (lap.get("lap_distance") or 0) / length))
        if not pts or x > pts[-1][0] + 0.05:
            pts.append((x, now))
    if len(pts) < 2 or pts[-1][0] - pts[0][0] < 0.9:
        return None
    total = s.info.get("total_laps") if s.is_race() else None
    xs = np.array([p[0] for p in pts])
    wheels = []
    for k, name in enumerate(["RL", "RR", "FL", "FR"]):
        ys = np.array([p[1][k] for p in pts], dtype=float)
        rate = max(0.0, float(np.polyfit(xs, ys, 1)[0]))
        wear = float(ys[-1])
        at = xs[-1] if wear >= WORN else xs[-1] + (WORN - wear) / rate if rate > 0 else None
        wheels.append({"wheel": name, "wear": round(wear, 1), "rate": round(rate, 2),
                       "at": None if at is None else round(float(at), 1),
                       "flag": None if not total else round(min(100.0, wear + rate * max(0.0, total - xs[-1])), 1)})
    soon = [w for w in wheels if w["at"] is not None]
    return {"compound": stints[-1]["compound"], "from": start, "lap": round(float(xs[-1]), 1), "total": total,
            "line": WORN, "wheels": wheels, "first": min(soon, key=lambda w: w["at"]) if soon else None}


# ---- where you lose time, corner by corner ------------------------------------------------------------------
def _lap(rows):
    """One lap's samples as arrays along the lap: distance (always going up), speed, brake, throttle, time."""
    rows = sorted((r for r in rows if r.get("distance") is not None and r["distance"] >= 0), key=lambda r: r["distance"])
    if len(rows) < 50:
        return None
    t0 = rows[0]["t"]
    d = np.array([r["distance"] for r in rows], dtype=float)
    keep = np.concatenate([[True], np.diff(d) > 0])
    pick = lambda k, fill=0.0: np.array([fill if r.get(k) is None else r[k] for r in rows], dtype=float)[keep]
    t = np.array([r["lap_time"] if r.get("lap_time") is not None else r["t"] - t0 for r in rows], dtype=float)[keep]
    return {"d": d[keep], "v": pick("speed"), "brake": pick("brake"), "thr": pick("throttle"), "t": t}


def find_corners(L, step=5.0):
    """The corners of a lap: where the speed bottoms out, dipping at least DIP km/h on both sides within 600 m.
    Corners closer than 150 m are one (a chicane). Returns the distances of the slowest points."""
    grid = np.arange(0, L["d"][-1], step)
    v = np.interp(grid, L["d"], L["v"])
    v = np.convolve(v, np.ones(5) / 5, mode="same")              # a little smoothing
    reach, near = int(600 / step), int(30 / step)
    found = []
    for i in range(near, len(v) - near):
        if v[i] > v[max(0, i - near):i + near + 1].min() + 1e-9:
            continue
        left, right = v[max(0, i - reach):i + 1].max(), v[i:i + reach + 1].max()
        if min(left, right) - v[i] >= DIP:
            if found and grid[i] - found[-1][0] < 150:
                if v[i] < found[-1][1]:
                    found[-1] = (grid[i], v[i])
            else:
                found.append((grid[i], v[i]))
    return [float(x) for x, _ in found]


def _brake_point(L, apex):
    """Where the braking for a corner starts (the run of braking that ends at the corner), or None for a lift."""
    zone = (L["d"] >= apex - 400) & (L["d"] <= apex + 10)
    d, b = L["d"][zone], L["brake"][zone]
    on = np.where(b > 0.1)[0]
    if not len(on):
        return None
    k = on[-1]
    while k > 0 and b[k - 1] > 0.1:
        k -= 1
    return float(d[k])


def _throttle_point(L, apex):
    """Where you're back on full throttle after a corner."""
    after = np.where((L["d"] >= apex) & (L["d"] <= apex + 500) & (L["thr"] >= 0.95))[0]
    return float(L["d"][after[0]]) if len(after) else None


def _at_corner(L, apex, width=80):
    zone = (L["d"] >= apex - width) & (L["d"] <= apex + width)
    return float(L["v"][zone].min()) if zone.any() else None


def corners(rows_a, rows_b):
    """Lap A against lap B, corner by corner: where each brakes, the speed in the corner, when back on full
    throttle, and the time A loses (+) or gains (-) from that braking point to the next corner's."""
    A, B = _lap(rows_a), _lap(rows_b)
    if A is None or B is None:
        return None
    apexes = find_corners(A)
    if not apexes:
        return {"corners": [], "total": None}
    starts = [(_brake_point(A, x) or x - 150) for x in apexes]
    end = min(A["d"][-1], B["d"][-1])
    clock = lambda L, x: float(np.interp(x, L["d"], L["t"]))
    out = []
    for k, apex in enumerate(apexes):
        s0 = max(0.0, starts[k] - 30)
        s1 = max(s0 + 1, (starts[k + 1] - 30) if k + 1 < len(apexes) else end)
        lost = (clock(A, s1) - clock(A, s0)) - (clock(B, s1) - clock(B, s0))
        ba, bb = _brake_point(A, apex), _brake_point(B, apex)
        va, vb = _at_corner(A, apex), _at_corner(B, apex)
        ta, tb = _throttle_point(A, apex), _throttle_point(B, apex)
        why = []
        if ba is not None and bb is not None and 5 <= abs(ba - bb) <= 120:      # (more: not the same braking)
            why.append(f"you brake {abs(ba - bb):.0f} m {'earlier' if ba < bb else 'later'}")
        elif ba is None and bb is not None:
            why.append("you only lift (no braking)")
        elif bb is None and ba is not None:
            why.append("you brake where you could just lift")
        if va is not None and vb is not None and abs(va - vb) >= 2:
            why.append(f"you are {abs(va - vb):.0f} km/h {'slower' if va < vb else 'faster'} in the corner")
        if ta is not None and tb is not None and abs(ta - tb) >= 10:
            why.append(f"full throttle {abs(ta - tb):.0f} m {'later' if ta > tb else 'sooner'}")
        out.append({"corner": k + 1, "at": round(apex), "lost": round(lost, 3), "brake": [ba, bb], "speed": [va, vb],
                    "throttle": [ta, tb], "why": why})
    return {"corners": out, "total": round(sum(c["lost"] for c in out), 3)}


# ---- lock-ups and wheelspin ---------------------------------------------------------------------------------
def slips(samples, outline, lap=None):
    """Lock-ups (a wheel locking under braking) and wheelspin (a rear wheel spinning on the throttle), from the wheel
    slip. Frames in a row make one moment; each is put on the track map by its distance along the lap. A place
    where it happened on two laps or more is a hot spot."""
    c = COLUMN
    track = sorted(([p[2], p[0], p[1]] for p in outline if len(p) > 2), key=lambda p: p[0])
    dist = np.array([p[0] for p in track]) if track else np.array([])
    where = lambda d: (track[int(np.clip(np.searchsorted(dist, d), 0, len(track) - 1))][1:]) if len(track) else (None, None)
    wheels = ["RL", "RR", "FL", "FR"]
    events, current = [], None
    for r in samples:
        if r[c["slip_FL"]] is None or r[c["distance"]] is None or r[c["distance"]] < 0 or (lap and r[c["lap"]] != lap):
            continue
        slip = [r[c[f"slip_{w}"]] or 0.0 for w in wheels]
        speed, brake, thr = r[c["speed"]] or 0, r[c["brake"]] or 0, r[c["throttle"]] or 0
        kind, worst = None, None
        if brake > 0.2 and speed > 40 and min(slip) < LOCK:
            kind, worst = "lock", int(np.argmin(slip))
        elif thr > 0.3 and speed > 20 and max(slip[:2]) > SPIN:
            kind, worst = "spin", int(np.argmax(slip[:2]))
        if kind is None:
            if current and r[c["distance"]] - current["end"] > 20:
                if current["frames"] >= 3:
                    events.append(current)
                current = None
            continue
        if current and current["kind"] == kind and current["lap"] == r[c["lap"]] and r[c["distance"]] - current["end"] <= 20:
            current["frames"] += 1
            current["end"] = r[c["distance"]]
            if abs(slip[worst]) > abs(current["slip"]):
                current.update(slip=slip[worst], wheel=wheels[worst], distance=r[c["distance"]], speed=speed)
        else:
            if current and current["frames"] >= 3:
                events.append(current)
            current = {"kind": kind, "lap": r[c["lap"]], "distance": r[c["distance"]], "end": r[c["distance"]],
                       "wheel": wheels[worst], "slip": slip[worst], "speed": speed, "frames": 1}
    if current and current["frames"] >= 3:
        events.append(current)
    out = []
    for e in events:
        x, z = where(e["distance"])
        out.append({"kind": e["kind"], "lap": e["lap"], "distance": round(e["distance"]), "wheel": e["wheel"],
                    "slip": round(e["slip"], 3), "speed": round(e["speed"]), "x": x, "z": z})
    spots = {}
    for e in out:
        spots.setdefault((e["kind"], round(e["distance"] / 60)), []).append(e)
    hot = [{"kind": k, "distance": round(float(np.mean([e["distance"] for e in v]))), "laps": sorted({e["lap"] for e in v}),
            "x": v[0]["x"], "z": v[0]["z"]} for (k, _), v in spots.items() if len({e["lap"] for e in v}) >= 2]
    return {"events": out, "hot": sorted(hot, key=lambda h: -len(h["laps"])), "lock": LOCK, "spin": SPIN,
            "counts": {"lock": sum(e["kind"] == "lock" for e in out), "spin": sum(e["kind"] == "spin" for e in out)}}


# ---- energy per lap, and overtake mode and active aero (F1 26) ------------------------------------------------
def energy(car_samples, laps, length, limit=None, bins=40):
    """Energy on each complete lap: harvested and used (the game's own counters for the lap, or, in sessions
    recorded before they were kept, from how the battery went up and down), against the most the rules let you
    harvest a lap (F1 26). And where on the lap the battery goes up (harvesting) and down (using it)."""
    c = CAR
    done = {l["lap"] for l in laps if l.get("time")}
    per, along, counts = [], np.zeros(bins), np.zeros(bins)
    for n in sorted(done):
        rows = [r for r in car_samples if r[c["lap"]] == n and r[c["ers"]] is not None and r[c["distance"]] is not None]
        if len(rows) < 10:
            continue
        late = [r for r in rows if r[c["distance"]] > 300]
        h = [r[c["ers_harvested"]] for r in late if len(r) > c["ers_harvested"] and r[c["ers_harvested"]] is not None]
        u = [r[c["ers_deployed"]] for r in late if len(r) > c["ers_deployed"] and r[c["ers_deployed"]] is not None]
        store = np.array([r[c["ers"]] for r in rows], dtype=float)
        step = np.diff(store)
        if h and u:
            harvested, used, how = max(h), max(u), "game"
        else:
            harvested, used, how = float(step[step > 0].sum()), float(-step[step < 0].sum()), "battery"
        per.append({"lap": n, "harvested": round(harvested), "used": round(used), "battery_end": round(float(store[-1])),
                    "how": how, "share": round(harvested / limit, 3) if limit else None})
        if length:
            for r0, r1 in zip(rows, rows[1:]):
                k = int(np.clip(r1[c["distance"]] / length * bins, 0, bins - 1))
                along[k] += r1[c["ers"]] - r0[c["ers"]]
                counts[k] += 1
    if not per:
        return None
    laps_seen = max(1, len(per))
    return {"laps": per, "limit": limit, "bins": [round(float(v / laps_seen)) for v in along], "length": length}


def modes(samples, events, player):
    """F1 26: on each lap, how much of it you spent in straight mode (active aero) and in overtake mode; each time
    you used overtake mode, and whether a pass of yours came while it was on (or within 5 s of it ending)."""
    c = COLUMN
    rows = [r for r in samples if r[c["aero"]] is not None and r[c["distance"]] is not None and r[c["distance"]] >= 0]
    if not rows:
        return None
    laps, uses, on = {}, [], None
    for r0, r1 in zip(rows, rows[1:]):
        dt = r1[c["t"]] - r0[c["t"]]
        if not 0 < dt < 0.5:                                   # a gap in the recording, or a flashback
            dt = 0.0
        x = laps.setdefault(r0[c["lap"]], {"lap": r0[c["lap"]], "time": 0.0, "straight": 0.0, "overtake": 0.0})
        x["time"] += dt
        x["straight"] += dt if r0[c["aero"]] == 1 else 0.0
        x["overtake"] += dt if r0[c["overtake"]] == 1 else 0.0
        if r0[c["overtake"]] == 1 and on is None:
            on = {"lap": r0[c["lap"]], "start": r0[c["t"]], "end": r0[c["t"]]}
        if on is not None:
            on["end"] = r0[c["t"]]
            if r1[c["overtake"]] != 1 or dt == 0.0:
                uses.append(on)
                on = None
    if on is not None:
        uses.append(on)
    passes = [e for e in events if e.get("code") == "OVTK" and (e.get("cars") or [None])[0] == player]
    lost = [e for e in events if e.get("code") == "OVTK" and (e.get("cars") or [None, None])[1] == player]
    for u in uses:
        inside = lambda e: u["start"] <= e["t"] <= u["end"] + WINDOW
        u.update(seconds=round(u["end"] - u["start"], 1), passes=sum(map(inside, passes)), passed_by=sum(map(inside, lost)))
    uses = [u for u in uses if u["seconds"] >= 0.5]
    counted = {id(e) for e in passes for u in uses if u["start"] <= e["t"] <= u["end"] + WINDOW}
    per = [{"lap": x["lap"], "straight": round(x["straight"] / x["time"] * 100) if x["time"] else None,
            "overtake": round(x["overtake"], 1)} for x in sorted(laps.values(), key=lambda x: x["lap"]) if x["time"] > 5]
    return {"laps": per, "uses": [{k: u[k] for k in ("lap", "seconds", "passes", "passed_by")} for u in uses],
            "total": {"uses": len(uses), "with_pass": sum(u["passes"] > 0 for u in uses), "passes": len(passes),
                      "passes_with": len(counted), "passed_by": len(lost)}}


# ---- Time Trial --------------------------------------------------------------------------------------------
def time_trial(s):
    """You against your personal best and your rival: the gap on the lap and in each sector."""
    tt = s.info.get("time_trial")
    if not tt:
        return None
    me = tt["session_best"] if tt["session_best"].get("time") else tt["personal_best"]
    gap = lambda other: {k: round(me[k] - other[k], 3) if me.get(k) and other.get(k) else None for k in ("time", "s1", "s2", "s3")}
    return {**tt, "you": me, "to_personal": gap(tt["personal_best"]), "to_rival": gap(tt["rival"])}


# ---- when to switch tyres ----------------------------------------------------------------------------------
WET = {"INTERMEDIATE", "WET"}


def switch(s, pit_loss=None):
    """Should you switch to the wet tyres (or back to slicks)? The game says how much faster or slower each of your
    sets would be now than the one fitted; over the laps left that gain is set against what a stop costs here."""
    sets = s.tyre_set_list()
    fitted = next((t for t in sets if t["fitted"]), None)
    if not fitted:
        return None
    wet_now = fitted["compound"] in WET
    others = [t for t in sets if t["available"] and not t["fitted"]
              and (t["compound"] not in WET if wet_now else t["compound"] in WET)]
    best = min(others, key=lambda t: t["delta"], default=None)
    lap = s.lap_now.get(s.player, {})
    total = s.info.get("total_laps") if s.is_race() else None
    left = max(0, total - lap["lap"] + 1) if total and lap.get("lap") and lap.get("result_status") == 2 else None
    gain = round(-best["delta"] * left, 1) if best and left is not None else None
    history = [{"lap": n, **r["set_deltas"]} for n, r in sorted(s.laps.get(s.player, {}).items()) if r.get("set_deltas")]
    if best is None:
        call = "none"
    elif best["delta"] >= 0:
        call = "stay"
    elif left is None or pit_loss is None:
        call = "faster"
    else:
        call = "box" if gain > pit_loss else "stay_close"
    return {"fitted": fitted["compound"], "wet_now": wet_now, "best": best, "left": left, "gain": gain,
            "pit_loss": None if pit_loss is None else round(pit_loss, 1), "call": call, "history": history,
            "forecast": s.info.get("forecast") or []}


# ---- your progress and your setups (over all your sessions) -------------------------------------------------
def progress(summaries, game_name):
    """Your best lap in each session at each track, in the order you drove them, and how much faster you got."""
    tracks = {}
    for x in summaries:
        if x.get("game") == game_name and not x.get("test") and x.get("best_lap"):
            tracks.setdefault(x["track"], []).append({"id": x["id"], "started": x["started"], "best_lap": x["best_lap"],
                                                      "type": x.get("type"), "weather": x.get("weather")})
    out = []
    for track, pts in tracks.items():
        pts.sort(key=lambda p: p["started"])
        best = min(pts, key=lambda p: p["best_lap"])
        out.append({"track": track, "sessions": len(pts), "best": best, "first": pts[0],
                    "gain": round(pts[0]["best_lap"] - best["best_lap"], 3), "points": pts, "last": pts[-1]["started"]})
    return sorted(out, key=lambda t: -t["last"])
