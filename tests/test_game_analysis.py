"""game_analysis.py: the game page's report, theoretical best, wear forecast, corners, lock-ups, energy, overtake mode,
Time Trial, when to switch tyres and your progress, on made-up laps."""
import struct

import numpy as np
import pytest

import game_analysis as ga
import game_telemetry as gt


def lap(n, time, compound="MEDIUM", **more):
    return {"lap": n, "time": time, "s1": round(time * 0.3, 3), "s2": round(time * 0.4, 3), "s3": round(time * 0.3, 3),
            "compound": compound, "t": float(n * 100), "position": 1, **more}


def session(laps, others=None, race=True):
    s = gt.Session(1, 2025)
    s.info = {"session_type_id": 15 if race else 7, "session_type": "Race" if race else "Qualifying 3", "track": "Miami",
              "total_laps": 20, "track_length": 5000}
    s.player = 0
    s.drivers = {0: {"code": "YOU", "name": "You", "colour": "FFFFFF"}, 1: {"code": "VER", "name": "Verstappen", "colour": "3671C6"}}
    s.laps = {0: {r["lap"]: r for r in laps}, 1: {r["lap"]: r for r in (others or [])}}
    return s


def test_clean_laps_leave_out_the_start_pit_laps_and_slow_ones():
    laps = {1: lap(1, 95.0), 2: lap(2, 90.0), 3: lap(3, 90.4, pitted=True), 4: lap(4, 90.2), 5: lap(5, 100.0), 6: lap(6, 89.8, invalid=True)}
    assert sorted(ga.clean_laps(laps)) == [2, 4]                    # 100.0 is over 107% of the usual lap
    assert sorted(ga.clean_laps(laps, race=False)) == [1, 2, 4]


def test_the_theoretical_best_and_pole():
    mine = [lap(2, 90.0), lap(3, 90.5)]
    mine[1].update(s2=35.0, s3=26.0)                               # a better middle and last sector on lap 3
    s = session(mine, [lap(2, 89.0)], race=False)
    b = ga.theoretical(s)
    assert b["best"]["lap"] == 2 and b["theoretical"] == pytest.approx(27.0 + 35.0 + 26.0)
    assert b["pole"]["driver"] == "VER" and b["pole"]["gap"] == pytest.approx(1.0)
    assert b["sectors"][1]["lap"] == 3 and b["sectors"][0]["to_pole"] == pytest.approx(27.0 - 26.7)


def test_the_report_result_stops_consistency_and_pace():
    mine = [lap(n, 90 + (n % 2) * 0.2, "SOFT" if n < 5 else "HARD", position=2, pit_entry=n == 4, pitted=n in (4, 5)) for n in range(1, 11)]
    theirs = [lap(n, 90.5, "SOFT" if n < 5 else "HARD", position=1) for n in range(1, 11)]
    s = session(mine, theirs)
    s.classification = [{"car": 1, "position": 1, "grid": 2}, {"car": 0, "position": 2, "grid": 5, "points": 18, "result": "FINISHED", "race_time": 905.0}]
    r = ga.report(s)
    assert r["result"]["position"] == 2 and r["result"]["gained"] == 3 and r["stops"] == [4]
    assert r["consistency"]["laps"] == 7 and r["consistency"]["within"] == 7
    hard = next(p for p in r["pace"] if p["compound"] == "HARD")
    assert hard["driver"] == "VER" and hard["delta"] < 0                 # you were quicker on the hards
    assert {c["car"] for c in r["chart"]} == {0, 1}


def test_the_wear_forecast_reaches_70_percent():
    laps = [lap(n, 90.0, tyre_wear=[10 + 5 * n, 10 + 5 * n, 5 + 3 * n, 5 + 3 * n]) for n in range(1, 6)]
    s = session(laps)
    f = ga.wear_forecast(s)
    rear = next(w for w in f["wheels"] if w["wheel"] == "RL")
    assert rear["rate"] == pytest.approx(5.0) and rear["at"] == pytest.approx(12.0)    # 35% now, +5% a lap
    assert f["first"]["wheel"] in ("RL", "RR") and rear["flag"] == pytest.approx(100.0)
    assert ga.wear_forecast(session([lap(1, 90.0, tyre_wear=[10] * 4)])) is None           # one point: no line yet


def trace(slow=0.0, brake_at=900.0):
    """A made-up lap: a straight, a corner at 1 km (down to 80 km/h, or slower), and a straight."""
    rows, t = [], 0.0
    for d in np.arange(0, 2000, 2.0):
        v = min(300.0, 80 - slow + 0.6 * abs(d - 1000))
        t += 2.0 / (v / 3.6)
        rows.append({"t": t, "distance": float(d), "speed": v, "brake": 1.0 if brake_at <= d < 1000 else 0.0,
                     "throttle": 1.0 if d > 1060 or d < brake_at else 0.0, "lap_time": t})
    return rows


def test_the_corners_and_where_time_is_lost():
    c = ga.corners(trace(slow=10, brake_at=880), trace())
    assert len(c["corners"]) == 1
    k = c["corners"][0]
    assert 950 <= k["at"] <= 1050 and k["lost"] > 0
    assert any("slower in the corner" in w for w in k["why"]) and any("earlier" in w for w in k["why"])
    assert c["total"] == pytest.approx(k["lost"])


def frame(lap_no, d, slip_front=0.0, slip_rear=0.0, brake=0.0, throttle=0.0, speed=150):
    row = [None] * len(gt.SAMPLE_COLUMNS)
    c = ga.COLUMN
    row[c["t"]], row[c["lap"]], row[c["distance"]], row[c["speed"]] = lap_no * 100 + d / 100, lap_no, d, speed
    row[c["brake"]], row[c["throttle"]] = brake, throttle
    row[c["slip_FL"]] = row[c["slip_FR"]] = slip_front
    row[c["slip_RL"]] = row[c["slip_RR"]] = slip_rear
    return row


def test_lock_ups_wheelspin_and_hot_spots():
    samples = []
    for n in (1, 2):
        samples += [frame(n, d) for d in range(0, 500, 10)]
        samples += [frame(n, d, slip_front=-0.3, brake=0.8) for d in range(500, 540, 5)]      # a lock-up into a corner
        samples += [frame(n, d) for d in range(540, 900, 10)]
    samples += [frame(3, d, slip_rear=0.4, throttle=1.0, speed=60) for d in range(100, 130, 5)]  # wheelspin, once
    outline = [[float(d), 0.0, float(d)] for d in range(0, 1000, 5)]
    s = ga.slips(samples, outline)
    assert s["counts"] == {"lock": 2, "spin": 1}
    assert len(s["hot"]) == 1 and s["hot"][0]["laps"] == [1, 2] and s["hot"][0]["kind"] == "lock"
    assert ga.slips(samples, outline, lap=3)["counts"] == {"lock": 0, "spin": 1}


def test_energy_from_the_battery_and_from_the_games_counters():
    c = ga.CAR
    rows = []
    for k, d in enumerate(range(0, 5000, 100)):
        r = [None] * len(gt.CAR_COLUMNS)
        r[c["t"]], r[c["lap"]], r[c["distance"]] = k * 0.5, 1, float(d)
        r[c["ers"]] = 2e6 + (1e5 if k % 2 else 0)                  # up 0.1 MJ, down 0.1 MJ, ...
        rows.append(r)
    e = ga.energy(rows, [{"lap": 1, "time": 90.0}], 5000, limit=8e6)
    assert e["laps"][0]["how"] == "battery" and e["laps"][0]["harvested"] == pytest.approx(25 * 1e5, rel=0.05)
    for r in rows:                                                 # the game's own counters for the lap
        r[c["ers_harvested"]], r[c["ers_deployed"]] = r[c["distance"]] * 1000, r[c["distance"]] * 500
    e = ga.energy(rows, [{"lap": 1, "time": 90.0}], 5000, limit=8e6)
    assert e["laps"][0]["how"] == "game" and e["laps"][0]["harvested"] == 4_900_000 and e["laps"][0]["share"] == pytest.approx(0.6125, abs=0.001)


def test_overtake_mode_and_the_passes_it_brought():
    c, rows = ga.COLUMN, []
    for k in range(400):
        r = [None] * len(gt.SAMPLE_COLUMNS)
        r[c["t"]], r[c["lap"]], r[c["distance"]] = k * 0.1, 1, k * 10.0
        r[c["aero"]] = 1 if k < 100 else 0                          # straight mode for a quarter of the lap
        r[c["overtake"]] = 1 if 200 <= k < 250 else 0               # overtake mode for 5 s
        rows.append(r)
    events = [{"code": "OVTK", "t": 26.0, "cars": [0, 3]}, {"code": "OVTK", "t": 35.0, "cars": [0, 4]}, {"code": "OVTK", "t": 2.0, "cars": [5, 0]}]
    m = ga.modes(rows, events, player=0)
    assert m["laps"][0]["straight"] == 25 and m["laps"][0]["overtake"] == pytest.approx(5.0, abs=0.2)
    assert m["total"] == {"uses": 1, "with_pass": 1, "passes": 2, "passes_with": 1, "passed_by": 1}


def test_the_time_trial_packet():
    r = gt.Recorder(resume=False)
    head = gt.HEADER.pack(2025, 25, 1, 0, 1, 14, 123, 12.5, 100, 100, 0, 255)
    sets = [gt.TIME_TRIAL.pack(0, 1, 90_500, 27_000, 36_000, 27_500, 0, 1, 0, 1, 0, 1),     # you, this session
            gt.TIME_TRIAL.pack(0, 1, 90_200, 26_900, 36_000, 27_300, 0, 1, 0, 1, 0, 1),     # your personal best
            gt.TIME_TRIAL.pack(5, 2, 89_900, 26_800, 35_700, 27_400, 2, 3, 1, 1, 0, 1)]     # your rival
    r.handle(head + b"".join(sets), 0.0, record=False)
    tt = r.sessions[123].info["time_trial"]
    assert tt["rival"]["time"] == pytest.approx(89.9) and tt["rival"]["team"] == "Red Bull Racing" and tt["rival"]["gearbox"] == "automatic"
    assert ga.time_trial(r.sessions[123])["to_rival"]["s2"] == pytest.approx(0.3)


def wet_session(delta, left):
    s = session([lap(n, 95.0) for n in range(1, 21 - left)])
    sets = [(17, 17, 30, 1, 15, 10, 20, 0, 0), (7, 7, 0, 1, 15, 25, 25, int(delta * 1000), 0)]   # mediums (fitted), inters
    s.tyre_sets = {"sets": [struct.unpack("<7BhB", struct.pack("<7BhB", *x)) for x in sets], "fitted": 0}
    s.lap_now = {0: {"lap": 21 - left, "result_status": 2}}
    return s


def test_when_to_switch_to_the_wet_tyres():
    assert ga.switch(wet_session(-1.5, 10), pit_loss=12.0)["call"] == "box"           # 15 s in 10 laps > 12 s
    assert ga.switch(wet_session(-1.5, 5), pit_loss=12.0)["call"] == "stay_close"     # 7.5 s < 12 s
    assert ga.switch(wet_session(2.0, 10), pit_loss=12.0)["call"] == "stay"
    s = wet_session(-1.5, 10)
    assert s.set_deltas() == {"fitted": "MEDIUM", "INTERMEDIATE": -1.5}


def test_your_progress_at_each_track():
    rows = [{"id": "a", "track": "Miami", "started": 1, "best_lap": 91.0, "game": "F1 25", "type": "Race"},
            {"id": "b", "track": "Miami", "started": 2, "best_lap": 89.5, "game": "F1 25", "type": "Qualifying 3"},
            {"id": "c", "track": "Monza", "started": 3, "best_lap": 82.0, "game": "F1 26", "type": "Race"},
            {"id": "d", "track": "Miami", "started": 4, "best_lap": 90.0, "game": "F1 25", "type": "Race", "test": True}]
    p = ga.progress(rows, "F1 25")
    assert len(p) == 1 and p[0]["sessions"] == 2 and p[0]["best"]["id"] == "b" and p[0]["gain"] == pytest.approx(1.5)
