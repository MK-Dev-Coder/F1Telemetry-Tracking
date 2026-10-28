"""features.py: the pit stop model's inputs, on a made-up race (see conftest.py)."""
import pandas as pd
import pytest

import features
from conftest import lap


def test_the_study_stops_at_round_17_of_2026():
    rows = pd.DataFrame({"Year": [2022, 2025, 2026, 2026, 2026], "Round": [22, 24, 1, 17, 18]})
    assert features.in_study(rows).tolist() == [True, True, True, True, False]


def test_a_real_stop_is_found(race):
    laps = features.build(race)
    assert lap(laps, "AAA", 4)["IsStop"]
    assert laps["IsStop"].sum() == 1                                  # AAA's, and nothing else


def test_a_drive_through_is_not_a_stop(race):
    assert not lap(features.build(race), "BBB", 6)["IsStop"]


def test_new_tyres_under_a_red_flag_are_not_a_stop(race):
    assert not lap(features.build(race), "CCC", 7)["IsStop"]


def test_pit_soon_is_the_3_laps_before_a_stop(race):
    laps = features.build(race)
    aaa = laps[laps["Driver"] == "AAA"].set_index("LapNumber")["PitSoon"]
    assert aaa.loc[1:3].tolist() == [1, 1, 1]
    assert aaa.loc[4:].sum() == 0                                     # the in-lap itself: the stop is already made


def test_tyre_features(race):
    laps = features.build(race)
    assert lap(laps, "AAA", 4)["DryCompoundsUsed"] == 1
    assert lap(laps, "AAA", 5)["DryCompoundsUsed"] == 2               # medium, then hard
    assert lap(laps, "AAA", 5)["LapsInStint"] == 1
    assert lap(laps, "AAA", 4)["LapsInStint"] == 4


def test_race_progress(race):
    row = lap(features.build(race), "AAA", 3)
    assert row["LapsLeft"] == 7
    assert row["RaceProgress"] == pytest.approx(0.3)


def test_safety_car_features(race):
    laps = features.build(race)
    assert [lap(laps, "BBB", n)["SafetyCar"] for n in (7, 8, 9, 10)] == [0, 1, 1, 0]
    assert [lap(laps, "BBB", n)["NeutralStarted"] for n in (8, 9)] == [1, 0]
    assert [lap(laps, "BBB", n)["LapsUnderNeutral"] for n in (8, 9, 10)] == [1, 2, 0]


def test_gaps(race):
    laps = features.build(race)
    leader = laps[(laps["LapNumber"] == 3) & (laps["Position"] == 1)].iloc[0]
    assert leader["GapToLeader"] == 0
    assert pd.isna(leader["GapAhead"])                                # no car ahead of the leader
    assert lap(laps, "BBB", 3)["GapAhead"] == pytest.approx(1.5)      # 0.5 s a lap slower than AAA
    assert lap(laps, "DDD", 10)["GapAhead"] == 60                     # over 60 s: not racing, so capped


def test_stops_by_other_cars(race):
    laps = features.build(race)
    assert lap(laps, "BBB", 5)["FieldStopsLast2"] == 1                # AAA stopped on lap 4
    assert lap(laps, "AAA", 5)["FieldStopsLast2"] == 0                # its own stop doesn't count


@pytest.mark.parametrize("cut", [3, 5, 6, 9])
def test_features_never_use_later_laps(race, cut):
    """Every feature of a lap is the same whether or not the laps after it exist (the race so far is all the app
    has when it is live). A car that goes into the pits counts as stopping on its in-lap - it is in the pit lane
    when the lap ends - and the lap after shows whether the tyres were really changed (a drive-through isn't a
    stop), so the check is on the laps before the last one."""
    full = features.build(race)
    part = features.build(race[race["LapNumber"] <= cut].copy())
    known = lambda laps: (laps[laps["LapNumber"] < cut].sort_values(["Driver", "LapNumber"])
                          .reset_index(drop=True)[["Driver"] + features.FEATURES])
    pd.testing.assert_frame_equal(known(part), known(full))
