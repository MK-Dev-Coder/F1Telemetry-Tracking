"""battle.py: the battle forecast, the undercut threat and the strategy battle (two drivers side by side)."""
import json
import math

import numpy as np
import pandas as pd
import pytest

import battle

# Made-up undercut gains (s) for the 4 tyre-wear groups: more worn tyres ahead, bigger gains
TABLE = {"gains": [[-1.0, 0.0, 0.5, 1.0], [0.0, 0.5, 1.0, 2.0], [0.5, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]],
         "all": [-1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0]}


def car(driver, position, time, **change):
    """One car at the end of a lap, with what battle.py reads."""
    row = {"Driver": driver, "Position": float(position), "Time": time, "Pace": 0.0, "UsualLap": 90.0, "LapsLeft": 20.0,
           "PitInTime": np.nan, "PitOutTime": np.nan, "LapsInStint": 12.0, "Compound": "MEDIUM", "TyreWear": 0.5}
    return pd.Series({**row, **change})


@pytest.mark.parametrize("gap, rate, left, laps", [
    (3.0, 0.5, 20, 4),            # 2 s to find at 0.5 s a lap
    (3.0, 0.3, 20, 7),            # 6.7 laps: the 7th lap gets there
    (0.8, None, 20, 0),           # already within 1 s
    (3.0, 0.5, 3, None),          # not before the end
    (3.0, 0.01, 20, None),        # hardly closing
    (3.0, -0.4, 20, None),        # dropping back
    (3.0, None, 20, None),        # no pace (or not under green)
])
def test_laps_until_within_1_second(gap, rate, left, laps):
    assert battle.laps_to_reach(gap, rate, left) == laps


def test_closing_comes_from_the_pace():
    """Pace is against the field (0.01 = 1% slower): the car ahead 1% slower on a 90 s lap gives 0.9 s a lap."""
    assert battle.closing(car("BBB", 2, 0, Pace=0.0), car("AAA", 1, 0, Pace=0.01)) == pytest.approx(0.9)
    assert battle.closing(car("BBB", 2, 0, Pace=0.01), car("AAA", 1, 0, Pace=0.0)) == pytest.approx(-0.9)
    assert battle.closing(car("BBB", 2, 0, Pace=np.nan), car("AAA", 1, 0)) is None


def test_the_undercut_chance_is_the_share_of_real_undercuts_that_gained_more():
    u = battle.undercut(TABLE, 1.5, 0.5)                                   # wear 0.3-0.7: gains 0, 0.5, 1, 2
    assert u["chance"] == 0.25 and u["gain"] == 0.8 and u["cases"] == 4
    assert battle.undercut(TABLE, 1.5, 2.0)["chance"] == 0.75              # worn tyres ahead: more of them gained more
    assert battle.undercut(TABLE, 1.5, np.nan)["cases"] == len(TABLE["all"])   # wear not known: all of them


def test_a_close_car_on_green_has_both():
    p = battle.pair(TABLE, car("BBB", 2, 3601.5, Pace=0.0), car("AAA", 1, 3600.0, Pace=0.005), "GREEN")
    assert p["ahead"] == "AAA" and p["gap"] == 1.5 and p["closing"] == pytest.approx(0.45)
    assert p["laps"] == 2 and p["undercut"]["chance"] == 0.25             # 1 of the 4 gains is over 1.5 s


@pytest.mark.parametrize("status, back, front", [
    ("SAFETY CAR", {}, {}),                                     # the stops cost less and the gaps close up
    ("GREEN", {"Compound": "INTERMEDIATE"}, {}),                 # learned on dry tyres only
    ("GREEN", {"PitInTime": 3599.0}, {}),                        # the car behind is already coming in
    ("GREEN", {}, {"PitOutTime": 3500.0}),                       # the car ahead has just pitted
    ("GREEN", {"LapsInStint": 3.0}, {}),                         # a new set: no reason to stop again
    ("GREEN", {"LapsLeft": 2.0}, {}),                            # too late to pit
])
def test_no_undercut_when_it_does_not_apply(status, back, front):
    p = battle.pair(TABLE, car("BBB", 2, 3602.0, **back), car("AAA", 1, 3600.0, **front), status)
    assert p["undercut"] is None


def test_no_undercut_too_far_back_and_no_forecast_under_the_safety_car():
    assert battle.pair(TABLE, car("BBB", 2, 3611.0), car("AAA", 1, 3600.0), "GREEN")["undercut"] is None
    p = battle.pair(TABLE, car("BBB", 2, 3602.0), car("AAA", 1, 3600.0), "SAFETY CAR")
    assert p["closing"] is None and p["laps"] is None


def test_a_shortened_game_race_counts_laps_on_the_set_as_real_laps():
    """In F1 25's 25% races one lap is worth about 4 real laps, so 2 laps on a set are 8 real laps: not new."""
    back, front = car("BBB", 2, 3602.0, LapsInStint=2.0), car("AAA", 1, 3600.0, LapsInStint=2.0)
    assert battle.pair(TABLE, back, front, "GREEN")["undercut"] is None
    assert battle.pair(TABLE, back, front, "GREEN", factor=4.0)["undercut"] is not None


def test_every_car_against_the_car_just_ahead():
    lap = pd.DataFrame([car("CCC", 3, 3605.0), car("AAA", 1, 3600.0), car("BBB", 2, 3601.5),
                        car("DDD", np.nan, np.nan)])                        # (no place: left out)
    b = battle.lap_battles(TABLE, lap, "GREEN")
    assert set(b) == {"BBB", "CCC"}
    assert (b["BBB"]["ahead"], b["BBB"]["gap"]) == ("AAA", 1.5)
    assert (b["CCC"]["ahead"], b["CCC"]["gap"]) == ("BBB", 3.5)


def test_no_battle_with_the_car_two_places_ahead():
    """A place missing from the timing (97 laps of 6,562): the car behind it isn't racing the car two places ahead."""
    lap = pd.DataFrame([car("AAA", 1, 3600.0), car("BBB", 2, 3601.5), car("DDD", 4, 3608.0), car("EEE", 5, 3609.0)])
    b = battle.lap_battles(TABLE, lap, "GREEN")
    assert set(b) == {"BBB", "EEE"} and b["EEE"]["ahead"] == "DDD"


def test_no_pace_yet_is_not_the_safety_car():
    """On lap 1 nobody has green laps of pace: no forecast, but still under green (the page says why)."""
    p = battle.pair(TABLE, car("BBB", 2, 3602.0, Pace=np.nan), car("AAA", 1, 3600.0), "GREEN")
    assert p["closing"] is None and p["laps"] is None and p["racing"] is True
    assert battle.pair(TABLE, car("BBB", 2, 3602.0), car("AAA", 1, 3600.0), "VSC")["racing"] is False


def test_an_empty_wear_group_uses_every_undercut():
    table = {**TABLE, "gains": [[], *TABLE["gains"][1:]]}
    u = battle.undercut(table, 1.5, 0.1)                                 # under 0.3 s: no undercuts learned
    assert u["cases"] == len(TABLE["all"]) and 0 <= u["chance"] <= 1


def real_undercut(answer_lap=5):
    """Two cars over 7 laps: BBB 2 s behind AAA, BBB pits at the end of lap 4, AAA at the end of `answer_lap`.
    After both stops BBB is 0.5 s ahead (a 2.5 s gain)."""
    rows = []
    for n in range(1, 8):
        gap = 2.0 if n <= 3 else -0.5 if n >= 6 else 20.0                  # (during the stops the gap jumps)
        for driver, time, pos in [("AAA", 100.0 * n, 1 if gap > 0 else 2), ("BBB", 100.0 * n + gap, 2 if gap > 0 else 1)]:
            rows.append({"RaceId": "2025_01", "Driver": driver, "LapNumber": float(n), "Position": float(pos), "Time": time,
                         "IsStop": (driver, n) in {("BBB", 4), ("AAA", answer_lap)}, "TrackStatus": "1", "Compound": "MEDIUM",
                         "TyreWear": 0.9, "Year": 2025, "Circuit": "Austin"})
    return pd.DataFrame(rows)


def test_the_real_undercuts_are_found():
    found = battle.undercuts(real_undercut())
    assert len(found) == 1
    u = found.iloc[0]
    assert (u["Driver"], u["Ahead"], u["LapNumber"]) == ("BBB", "AAA", 4)
    assert (u["Gap"], u["After"], u["Gain"], u["Wear"]) == (2.0, -0.5, 2.5, 0.9)


def test_only_when_the_car_ahead_answers_next_lap():
    assert battle.undercuts(real_undercut(answer_lap=6)).empty


def duel_race():
    """AAA leads BBB by 1 s on lap 1, 2 s on lap 2 and 3 s on lap 3; BBB pits at the end of lap 2."""
    rows = []
    for n in (1, 2, 3):
        for driver, gap, pos in [("AAA", 0.0, 1), ("BBB", float(n), 2)]:
            stint = 2.0 if driver == "BBB" and n == 3 else 1.0
            rows.append(car(driver, pos, 100.0 * n + gap, LapNumber=float(n), Stint=stint, IsStop=driver == "BBB" and n == 2,
                            Compound="HARD" if stint == 2 else "MEDIUM", TyreLife=float(n)))
    return pd.DataFrame(rows)


def test_the_strategy_battle():
    d = battle.duel(TABLE, duel_race(), "AAA", "BBB", 3, "GREEN")
    assert [x["gap"] for x in d["laps"]] == [1.0, 2.0, 3.0]
    assert d["stops"] == {"AAA": [], "BBB": [2]}
    assert d["stints"]["BBB"] == [{"compound": "MEDIUM", "from": 1, "to": 2}, {"compound": "HARD", "from": 3, "to": 3}]
    assert (d["now"]["front"], d["now"]["back"], d["now"]["between"], d["now"]["gap"]) == ("AAA", "BBB", 0, 3.0)
    assert json.dumps(d)                                                   # (the web answer)


def test_the_strategy_battle_stops_at_the_lap_shown():
    d = battle.duel(TABLE, duel_race(), "BBB", "AAA", 2, "GREEN")
    assert [x["lap"] for x in d["laps"]] == [1, 2]
    assert d["laps"][-1]["gap"] == -2.0                                    # A = BBB, 2 s behind B = AAA
    assert battle.duel(TABLE, duel_race(), "AAA", "XXX", 3, "GREEN") is None


def test_the_saved_table_learned_bigger_gains_on_worn_tyres():
    """models/battle.json, learned from the real undercuts of 2022-2025."""
    saved = json.loads(battle.TABLE_FILE.read_text(encoding="utf-8"))
    medians = [float(np.median(g)) for g in saved["gains"]]
    assert medians == sorted(medians) and medians[-1] > medians[0]
    assert sum(len(g) for g in saved["gains"]) == saved["cases"] == len(saved["all"])
    assert saved["trained_on"] == "2022-2025" and saved["strike"] == 1.0
    assert not math.isnan(battle.undercut(saved, 1.0, 0.8)["chance"])
