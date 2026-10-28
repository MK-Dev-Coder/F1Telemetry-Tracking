"""game_driver.py: the driver report's model learns which habit costs you time, from every corner against your best
lap, and keeps learning: a new or longer session is added the next time, and nothing else is read again."""
import json
import os

import numpy as np
import pandas as pd
import pytest

import game_driver as gd
import game_telemetry as gt


def trace(lap, slow=0.0, brake_at=900.0):
    """A made-up lap: a straight, a corner at 1 km (down to 80 km/h, or slower), and a straight."""
    rows, t = [], 0.0
    for d in np.arange(0, 2000, 2.0):
        v = min(300.0, 80 - slow + 0.6 * abs(d - 1000))
        t += 2.0 / (v / 3.6)
        rows.append({"t": lap * 1000 + t, "lap": lap, "distance": float(d), "speed": v, "steer": 0.0,
                     "brake": 1.0 if brake_at <= d < 1000 else 0.0, "throttle": 1.0 if d > 1060 or d < brake_at else 0.0, "lap_time": t})
    return rows


def state(times):
    return {"player": 0, "started": 1.0, "test": False, "info": {"session_type_id": 7, "track": "Miami"},
            "laps": {"0": {str(n): {"time": t, "tyre_wear": [20, 20, 10, 10], "tyre_age": n} for n, t in times.items()}}}


def test_a_corner_against_your_best_lap():
    tele = pd.DataFrame(trace(1) + trace(2, slow=10, brake_at=880))
    rows = gd.session_rows("s1", state({1: 60.0, 2: 61.0}), tele)
    assert len(rows) == 1
    r = rows[0]
    assert r["lap"] == 2 and r["brake_early"] == pytest.approx(20, abs=2) and r["speed_loss"] == pytest.approx(10, abs=1)
    assert r["lost"] > 0 and r["trail"] == 1 and r["wear"] == 15


def made_up_rows(n=400, seed=1):
    """Corners where braking early is what costs the time (and the rest is noise)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({k: rng.normal(0, 1, n) for k in gd.FEATURES})
    df["brake_early"] = rng.uniform(-5, 30, n)
    df["lost"] = 0.01 * df["brake_early"] + rng.normal(0, 0.02, n)
    df["sid"] = [f"s{k % 5}" for k in range(n)]
    df["lap"], df["corner"] = np.arange(n) // 8, np.arange(n) % 8
    df["started"] = df["sid"].str[1].astype(float)
    df["corner_speed"], df["brake_at"], df["trail"] = rng.uniform(60, 280, n), rng.uniform(0, 5000, n), 1
    df["lock"], df["spin"], df["track"] = 0, 0, "Miami"
    return df


def test_the_model_finds_the_habit_that_costs_time(tmp_path, monkeypatch):
    monkeypatch.setattr(gd, "DRIVER_DIR", tmp_path)
    out = gd._build(made_up_rows(), "F1 26")
    assert out["ready"] and out["habits"][0]["key"] == "brake_early"
    assert out["model"]["mae"] < out["model"]["base"]                 # it beats a plain average on unseen sessions
    assert json.loads((tmp_path / "history.json").read_text())["F1 26"][-1]["corners"] == 400


def test_too_little_driving_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(gd, "DRIVER_DIR", tmp_path)
    out = gd._build(made_up_rows(50), "F1 25")
    assert not out["ready"] and out["corners"] == 50


def test_it_keeps_learning_new_sessions(tmp_path, monkeypatch):
    monkeypatch.setattr(gt, "GAME_DIR", tmp_path / "f1_25")
    monkeypatch.setattr(gd, "DRIVER_DIR", tmp_path / "driver")

    def session(sid, times):
        folder = gt.GAME_DIR / sid
        folder.mkdir(parents=True)
        (folder / "session.json").write_text(json.dumps(state(times)), encoding="utf-8")
        pd.DataFrame([r for n in times for r in trace(n, slow=0 if n == 1 else 6)]).to_parquet(folder / "telemetry.parquet")
        return folder

    first = session("100", {1: 60.0, 2: 60.6})
    rows, changed = gd.refresh()
    assert changed and set(rows["sid"]) == {"100"}
    assert not gd.refresh()[1]                                        # nothing new: nothing read again
    session("200", {1: 60.0, 2: 60.5, 3: 60.7})
    rows, changed = gd.refresh()
    assert changed and rows.groupby("sid").size().to_dict() == {"100": 1, "200": 2}
    (first / "session.json").write_text(json.dumps(state({1: 60.0, 2: 60.6})), encoding="utf-8")
    os.utime(first / "session.json", ns=(1, 10**18))                  # the session grew: it's read again
    assert gd.refresh()[1]
    assert set(gd.refresh(skip={"200"})[0]["sid"]) == {"100"}         # the live session waits until it ends


def test_your_start_from_the_game_or_from_when_the_car_moves():
    st = state({1: 60.0})
    st["info"]["session_type_id"] = 15
    st["events"] = [{"code": "LGOT", "t": 10.0}]                   # the lights went out 10 s into the session
    st["classification"] = [{"car": 0, "grid": 6}]
    st["laps"]["0"]["1"]["position"] = 3
    tele = pd.DataFrame([{"t": 10.0 + k * 0.05, "lap": 1, "speed": 0 if k < 7 else 5 * k} for k in range(1, 40)])
    x = gd.session_start("s", st, tele)                             # F1 25: when the car first moved
    assert x["source"] == "estimate" and x["reaction"] == pytest.approx(0.35) and x["gained"] == 3
    st["info"]["start_reaction"] = 0.38                             # F1 26: the game's own reaction time
    x = gd.session_start("s", st, tele)
    assert x["source"] == "game" and x["reaction"] == 0.38 and x["estimate"] == pytest.approx(0.35)
    late = tele.assign(t=tele["t"] + 2.0)                           # the recording began after the start
    st["info"].pop("start_reaction")
    assert gd.session_start("s", st, late)["reaction"] is None
    st["info"]["session_type_id"] = 7                               # qualifying: no start
    assert gd.session_start("s", st, tele) is None


def test_your_starts_and_the_places_they_make():
    items = [{"sid": str(k), "track": "Miami", "started": k, "game": "F1 26", "type": "Race", "reaction": r, "source": "game",
              "game_value": r, "estimate": r + 0.02, "grid": 10, "lap1": 10 - g, "gained": g}
             for k, (r, g) in enumerate([(0.5, 0), (0.45, 1), (0.3, 4), (0.28, 5)])]
    s = gd._starts(items, "F1 26")
    assert s["timed"] == 4 and s["best"]["reaction"] == 0.28 and s["improving"] and s["gained"] == 2.5
    assert s["quick_gain"] == 4.5 and s["slow_gain"] == 0.5 and s["check"] == {"pairs": 4, "within": 4}
    assert gd._starts(items, "F1 25") is None
