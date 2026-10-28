"""Tests on the real data and the trained models (about a minute). The data folder is only read, never changed.
Skipped when the data folder isn't there (it isn't on GitHub: see "Get the data" in the README)."""
import numpy as np
import pandas as pd
import pytest

import features
import race_outcome as ro
import train_tyre_model as tyre
from config import DATA_DIR
from explain_pit_model import explain, load_model, sigmoid

SAVED = ["test_predictions", "race_test_predictions", "tyre_test_predictions", "shap_values_2026"]   # the test answers
NEEDED = [tyre.REPLAY_FILE, ro.RESULTS_FILE, features.IN_FILE, features.OUT_FILE] + [DATA_DIR / f"{n}.parquet" for n in SAVED]
pytestmark = [pytest.mark.skipif(not all(f.exists() for f in NEEDED),
                                 reason="needs the data folder and the trained models (see the README)"),
              pytest.mark.filterwarnings("ignore:Using `httpx`")]       # (FastAPI's test client, not the app)
STUDY = [f"2026_{r:02d}" for r in range(1, features.TEST_LAST_ROUND + 1)]
RACE = "2026_17"                                                   # Singapore 2026, the last test race


@pytest.mark.parametrize("name", SAVED)
def test_the_study_is_tested_on_the_17_races_of_2026_only(name):
    """The saved test answers: all 17 test races, nothing the models were trained on, and no race added later."""
    races = pd.read_parquet(DATA_DIR / f"{name}.parquet", columns=["RaceId"])["RaceId"].unique()
    assert sorted(races) == STUDY


def test_a_real_race_never_uses_later_laps():
    race = pd.read_parquet(features.IN_FILE, filters=[("RaceId", "==", RACE)])
    full, part = features.build(race), features.build(race[race["LapNumber"] <= 30].copy())
    known = lambda laps: (laps[laps["LapNumber"] < 30].sort_values(["Driver", "LapNumber"])
                          .reset_index(drop=True)[["Driver"] + features.FEATURES])
    pd.testing.assert_frame_equal(known(part), known(full))


def test_the_pit_models_reasons_add_up():
    """SHAP: the typical lap's chance plus every reason is exactly the model's chance."""
    model, info = load_model()
    rows = pd.read_parquet(features.OUT_FILE, filters=[("RaceId", "==", RACE)])
    chance, values, base = explain(model, info, rows)
    assert sigmoid(base + values.sum(axis=1)).values == pytest.approx(chance, abs=1e-6)
    assert ((chance >= 0) & (chance <= 1)).all()


@pytest.fixture(scope="module")
def outcome():
    """The race outcome model's rows for the 2026 test races (as train_race_model.py makes them)."""
    model, info = ro.load_model()
    laps = pd.read_parquet(tyre.REPLAY_FILE)
    laps = laps[(laps["Year"] == 2026) & features.in_study(laps)]
    rows = ro.add_pit_adjusted(ro.add_features(laps, pd.read_parquet(ro.RESULTS_FILE)), info["pit_loss"])
    return model, info, rows[(rows["RaceId"] == RACE) & rows["Position"].notna()].reset_index(drop=True)


@pytest.mark.parametrize("n", [1, 20, 45, 62])
def test_real_laps_have_one_winner(outcome, n):
    model, info, rows = outcome
    here = rows[rows["LapNumber"] == n].reset_index(drop=True)
    s = ro.summary(ro.chances(model, info, here))
    assert s["win"].sum() == pytest.approx(1, abs=1e-6)
    assert s["podium"].sum() == pytest.approx(3, abs=0.01)
    assert s["points"].sum() == pytest.approx(10, abs=0.01)
    assert ((s["win"] <= s["podium"]) & (s["podium"] <= s["points"])).all()


def test_the_race_models_reasons_add_up(outcome):
    """SHAP against the average driver: the average driver's chance plus every reason is the model's own chance."""
    model, info, rows = outcome
    here = rows[rows["LapNumber"] == 30].reset_index(drop=True)
    for k in ro.SHOWN.values():
        values, start, _ = ro.explain(model, info, here, k)
        own = model.predict(ro.stack(here, info["features"], [k]))
        assert (start + values.sum(axis=1)).values == pytest.approx(own, abs=1e-4)


def test_short_game_races_are_scaled_to_the_real_race():
    import game_strategy as gs
    jeddah = gs.race_scale("Jeddah", 13)                                    # a 25% race: the real one has 50 laps
    assert jeddah["factor"] == pytest.approx(50 / 13)
    assert gs.race_scale("Jeddah", 50)["factor"] == 1
    assert gs.race_scale("Jeddah", 5)["very_short"] and gs.race_scale("Jeddah", 5)["factor"] == 1
    assert gs.race_scale("Jeddah", 17, sprint=True)["factor"] == 1


# -- the app itself: its web answers, as the pages get them -----------------------------------------------------
@pytest.fixture(scope="module")
def app():
    if tyre.REPLAY_FILE.stat().st_mtime <= features.IN_FILE.stat().st_mtime:
        pytest.skip("the replay file is older than the laps: start the app once to build it")
    from fastapi.testclient import TestClient
    import game_api
    game_api.start = lambda: None                                            # don't listen for the game (the app may be running)
    import api
    return TestClient(api.app)


def test_the_app_lists_the_races(app):
    races = {r["race_id"]: r for r in app.get("/races").json()}
    assert set(STUDY) <= set(races) and "2026_18" in races                  # the study's races, and Austin
    assert not any(r["seen_in_training"] for r in races.values() if r["year"] == 2026)


def test_replay_speed_must_make_sense(app):
    assert app.get(f"/races/{RACE}/live_replay?speed=0").status_code == 422
    assert app.get(f"/races/{RACE}/live_replay?speed=101").status_code == 422


def test_unknown_race_or_driver(app):
    assert app.get("/races/1999_01").status_code == 404
    assert app.get(f"/races/{RACE}/laps/30/drivers/XXX").status_code == 404


def test_a_drivers_pit_reasons_add_up(app):
    w = app.get(f"/races/{RACE}/laps/30/drivers/NOR").json()
    assert sum(r["points"] for r in w["reasons"]) == pytest.approx(w["pit_chance"] - w["typical_lap_chance"], abs=0.02)


def test_if_they_pit_now(app):
    w = app.get(f"/races/{RACE}/laps/30/drivers/NOR/pit_now").json()
    assert w["cost"] > 0 and w["cost_range"][0] <= w["cost"] <= w["cost_range"][1]
    assert w["range"][0] <= w["position"] <= w["range"][1]                  # a cheaper stop can only help
    assert all(0 <= v <= 1 for t in ro.SHOWN for v in w["chances"][t])


def test_no_rejoin_places_when_cars_are_missing(app):
    snap = app.get("/races/2025_06/live/25").json()          # Miami 2025: lap 25 has only P10-P19 in the timing
    assert all(d["rejoin"] is None for d in snap["drivers"])
    assert app.get("/races/2025_06/laps/25/drivers/HAM/pit_now").status_code == 404


def test_new_tyres_are_free_under_a_red_flag(app):
    snap = app.get("/races/2023_03/live/9").json()           # Melbourne 2023: the red flag on lap 9
    assert snap["track_status"] == "RED FLAG" and snap["pit_cost"]["under"] == "red"
    w = app.get("/races/2023_03/laps/9/drivers/VER/pit_now").json()
    assert (w["cost"], w["position"]) == (0.0, 2)            # P2 keeps P2


def test_every_car_is_against_the_car_just_ahead(app):
    snap = app.get(f"/races/{RACE}/live/30").json()
    placed = sorted((d for d in snap["drivers"] if d["position"] is not None), key=lambda d: d["position"])
    for front, back in zip(placed, placed[1:]):
        if back["battle"]:
            assert back["battle"]["ahead"] == front["driver"]
            u = back["battle"]["undercut"]
            assert u is None or 0 <= u["chance"] <= 1


def test_no_battle_forecast_or_undercut_under_a_red_flag(app):
    snap = app.get("/races/2023_03/live/9").json()           # Melbourne 2023: the red flag on lap 9
    battles = [d["battle"] for d in snap["drivers"] if d["battle"]]
    assert battles and all(b["closing"] is None and b["undercut"] is None for b in battles)


def test_the_strategy_battle_of_two_real_drivers(app):
    d = app.get(f"/races/{RACE}/duel/NOR/PIA?lap=30").json()
    assert [x["lap"] for x in d["laps"]] == list(range(1, 31))
    assert {d["now"]["front"], d["now"]["back"]} == {"NOR", "PIA"}
    assert app.get(f"/races/{RACE}/duel/NOR/XXX?lap=30").status_code == 404
    assert app.get(f"/races/{RACE}/duel/NOR/PIA").status_code == 422          # the lap is needed


def test_every_driver_has_a_card_for_the_driver_picker(app):
    """The strategy battle's picker: each driver's car number, name, team and photo (OpenF1 from 2023; a 2022 race
    takes each driver from the nearest race, so Verstappen is still number 1)."""
    race = app.get(f"/races/{RACE}").json()
    assert set(race["cards"]) == set(race["drivers"])
    assert race["cards"]["LEC"]["number"] == 16 and race["cards"]["LEC"]["photo"].startswith("https://")
    old = app.get("/races/2022_05").json()["cards"]
    assert old["VER"]["number"] == 1 and "VET" not in old            # (Vettel retired before OpenF1: his number shows)


def test_the_engineer_briefing(app):
    """The next race's track, its tyres (shorter on a hotter day and in a game race) and the race prediction."""
    tracks = app.get("/engineer").json()
    nxt = tracks["next"]
    assert any(t["circuit"] == nxt and t["next"] for t in tracks["tracks"])
    b = app.get(f"/engineer/{nxt}").json()
    assert b["prediction"]["kind"] == "next" and sum(d["win"] for d in b["prediction"]["drivers"]) == pytest.approx(1, abs=0.01)
    used = [t for t in b["tyres"].values() if t["life"] is not None]
    assert used and all(t["range"][0] <= t["life"] <= t["range"][1] for t in used)
    hot = app.get(f"/engineer/{nxt}?track_temp={b['usual_temp'] + 15}&distance=25").json()
    assert hot["tyres"]["MEDIUM"]["life_today"] < b["tyres"]["MEDIUM"]["life_today"]
    assert hot["game_laps"] == round(b["laps"] / 4) and hot["tyres"]["MEDIUM"]["game"]["life"] < hot["tyres"]["MEDIUM"]["life_today"]
    assert sum(p["share"] for p in b["plans"]) <= 1.0001
    assert app.get("/engineer/Nowhere").status_code == 404 and app.get(f"/engineer/{nxt}?distance=5").status_code == 422


def test_a_replays_engineer_knows_only_the_races_before_it(app):
    """The F1 page's Engineer tab: only the races at this track before this one, and what the preview said before it."""
    circuit = app.get(f"/races/{RACE}").json()["circuit"]
    b = app.get(f"/engineer/{circuit}?before={RACE}").json()
    assert b["races"] and all(r < RACE for r in b["races"]) and all(w["race_id"] < RACE for w in b["winners"])
    assert b["prediction"]["kind"] == "this" and b["prediction"]["race_id"] == RACE
    first = app.get("/engineer/Singapore?before=2022_17").json()                # the first race there in the data
    assert first["empty"] and first["prediction"] is None


def test_the_games_engineer(app):
    """The F1 25 / F1 26 Engineer tab: the game's own plan, and the real track for the real briefing."""
    races = [s for s in app.get("/game/sessions").json() if s["type_id"] in (15, 16, 17)]
    if not races:
        pytest.skip("no recorded game race")
    e = app.get(f"/game/sessions/{races[0]['id']}/engineer").json()
    assert e["race"] and e["scale"]["game_laps"] == e["total_laps"] and isinstance(e["tyre_sets"], list)
    if e["scale"]["factor"] > 1 and app.get(f"/engineer/{e['circuit']}").status_code == 200:
        b = app.get(f"/engineer/{e['circuit']}?game_laps={e['total_laps']}").json()
        assert b["game_laps"] == e["total_laps"] and b["tyres"]["MEDIUM"]["game"] is not None
    assert app.get("/game/sessions/nope/engineer").status_code == 404


def test_the_games_report_and_analysis(app):
    """F1 25 / F1 26: the report, corners, lock-ups, energy, the wet-tyre call, progress and setups, on a recorded race."""
    races = [x for x in app.get("/game/sessions").json() if x["type_id"] in (15, 16, 17) and x["laps"] >= 3 and x["game"]]
    if not races:
        pytest.skip("no recorded game race")
    sid, which = races[0]["id"], races[0]["game"]
    r = app.get(f"/game/sessions/{sid}/report").json()
    assert r["race"] and r["result"]["position"] and r["laps"] >= 3 and r["stints"]
    for part in ("corners", "slips", "energy", "switch"):
        assert app.get(f"/game/sessions/{sid}/{part}").status_code == 200
    assert any(t["track"] == races[0]["track"] for t in app.get(f"/game/progress?game={which}").json()["tracks"])
    assert all(x["game"] == which and x["setup"] for x in app.get(f"/game/setups?game={which}").json()["sessions"])


def test_drivers_without_the_team_colours_file(app, monkeypatch):
    """The README says the laps file is enough: without team_colours.json the drivers are grey, not an error."""
    import api
    monkeypatch.setattr(api, "COLOURS_BY_RACE", {})
    assert api.who("VER", "2026_17")["colour"] == "#888888"


def test_the_driver_report(app, tmp_path, monkeypatch):
    """The driver report learns from the recorded sessions (its own files go to a temporary folder here)."""
    import game_driver
    monkeypatch.setattr(game_driver, "DRIVER_DIR", tmp_path)
    monkeypatch.setattr(game_driver, "_memo", {})
    for which in ("F1 25", "F1 26"):
        d = app.get(f"/game/driver?game={which}").json()
        assert d["game"] == which and d["corners"] >= 0
        if d["ready"]:
            assert len(d["habits"]) == len(game_driver.HABITS) and d["laps"] > 0 and d["history"]
        if d["starts"] and d["starts"]["timed"]:
            assert all(0.1 < r["reaction"] < 2 for r in d["starts"]["races_list"])


def test_the_undercut_and_forecast_report_is_frozen_to_the_study():
    """data/battle_evaluation.md: tested on the 33 real undercuts of the 17 test races (python battle.py)."""
    text = (DATA_DIR / "battle_evaluation.md").read_text(encoding="utf-8")
    assert "rounds 1-17" in text and "On the 33 real undercuts" in text


def test_the_race_outcome_chart_adds_up(app):
    drivers = app.get(f"/races/{RACE}/outcome").json()["drivers"]
    wins = np.array([[v if v is not None else 0 for v in d["win"]] for d in drivers.values()])
    assert wins.sum(axis=0) == pytest.approx(np.ones(wins.shape[1]), abs=0.02)   # one winner on every lap


def test_the_race_preview(app):
    w = app.get("/preview").json()
    assert sum(d["win"] for d in w["drivers"]) == pytest.approx(1, abs=0.01)
    assert isinstance(w["is_past"], bool)                    # (true once the race has been run)
    assert app.get("/preview/2026_17").status_code == 200
    assert app.get("/preview/2022_01").status_code == 404                    # tested on 2025 and 2026 only
