"""pit_now.py: if they pit now - what the stop costs and where they'd rejoin."""
import json

import pandas as pd
import pytest

import pit_now

TABLE = {"share": {"green": [0.9, 1.0, 1.1], "SC": [0.35, 0.43, 0.5], "VSC": [0.5, 0.61, 0.7]}}
PIT_LOSS = {"track": {"Austin": 22.0}, "overall": 21.0}


def test_a_stop_costs_less_under_the_safety_car_and_the_vsc():
    green, (low, high), kind = pit_now.cost(TABLE, PIT_LOSS, "Austin", "GREEN")
    assert (green, kind) == (22.0, "green") and low < green < high
    sc, _, sc_kind = pit_now.cost(TABLE, PIT_LOSS, "Austin", "SAFETY CAR")
    vsc, _, vsc_kind = pit_now.cost(TABLE, PIT_LOSS, "Austin", "VSC")
    assert (sc_kind, vsc_kind) == ("SC", "VSC")
    assert sc < vsc < green


def test_a_track_never_seen_uses_the_usual_cost():
    assert pit_now.cost(TABLE, PIT_LOSS, "Madring", "GREEN")[0] == 21.0


def test_new_tyres_are_free_under_a_red_flag():
    assert pit_now.cost(TABLE, PIT_LOSS, "Austin", "RED FLAG") == (0.0, (0.0, 0.0), "red")
    assert pit_now.rejoin(lap(), "BBB", 0.0)["position"] == 2                 # keeps the place


def test_the_saved_table_learned_the_same_order():
    """models/pit_now.json, learned from the 2022-2025 stops: cheapest under the safety car, then the VSC."""
    saved = json.loads(pit_now.TABLE_FILE.read_text(encoding="utf-8"))
    middle = {k: v[1] for k, v in saved["share"].items()}
    assert middle["SC"] < middle["VSC"] < middle["green"] == pytest.approx(1.0, abs=0.05)
    assert saved["trained_on"] == "2022-2025"


def lap():
    return pd.DataFrame({"Driver": ["AAA", "BBB", "CCC", "DDD", "EEE"], "Position": [1.0, 2.0, 3.0, 4.0, 5.0],
                         "GapToLeader": [0.0, 5.0, 12.0, 30.0, 31.0]})


def test_where_the_leader_rejoins():
    w = pit_now.rejoin(lap(), "AAA", 22.0)                                   # 22 s behind where they are now
    assert w["position"] == 3
    assert w["ahead"] == {"driver": "CCC", "gap": 10.0}
    assert w["behind"] == {"driver": "DDD", "gap": 8.0}
    assert [c["driver"] for c in w["around"]] == ["BBB", "CCC", "DDD", "EEE"]
    assert [c["position"] for c in w["around"]] == [1, 2, 4, 5]             # P3 is the driver who pitted


def test_the_last_car_stays_last():
    w = pit_now.rejoin(lap(), "EEE", 22.0)
    assert w["position"] == 5 and w["behind"] is None


def test_no_gap_no_answer():
    rows = lap()
    rows.loc[1, "GapToLeader"] = float("nan")                                # BBB's gap is missing
    assert pit_now.rejoin(rows, "AAA", 22.0) is None
    assert pit_now.rejoin_all(rows, 22.0) == {}


def test_a_lap_with_cars_missing_from_the_timing_has_no_answer():
    """Miami 2025 lap 25 has only P10-P19: the places after a stop can't be counted (P10 isn't "P7 after a stop")."""
    rows = lap().assign(Position=[10.0, 11.0, 12.0, 13.0, 14.0])
    assert pit_now.rejoin_all(rows, 22.0) == {}
    assert pit_now.rejoin(rows, "AAA", 22.0) is None


def test_cars_close_by_keep_their_places():
    """A car 0.04 s away is shown with a gap of 0.0: its place says whether it's ahead or behind."""
    rows = lap().assign(GapToLeader=[0.0, 5.0, 21.96, 22.04, 31.0])
    around = {c["driver"]: c for c in pit_now.rejoin(rows, "AAA", 22.0)["around"]}
    assert (around["CCC"]["position"], around["CCC"]["gap"]) == (2, -0.0)     # just ahead
    assert (around["DDD"]["position"], around["DDD"]["gap"]) == (4, 0.0)      # just behind


@pytest.mark.parametrize("loss", [0.5, 10.0, 22.0, 45.0])
def test_one_driver_and_the_table_agree(loss):
    every = pit_now.rejoin_all(lap(), loss)
    for driver in lap()["Driver"]:
        one = pit_now.rejoin(lap(), driver, loss)
        assert every[driver] == {k: one[k] for k in ("position", "ahead", "behind")}
