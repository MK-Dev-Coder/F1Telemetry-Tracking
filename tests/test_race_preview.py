"""race_preview.py: before a race, each driver's chances, from what was known before it."""
import json

import numpy as np
import pandas as pd
import pytest

import race_preview as rp
from config import MODELS_DIR

HISTORY = ["DriverForm5", "LastFinish", "Pace3", "SeasonAvg", "SeasonWins", "SeasonPodiums", "TeamForm3",
           "TeamPace3", "TeamDNF5", "TrackDriver"]


def races():
    """3 drivers (A and B in one team) over three races of 2025 and the first of 2026."""
    finishes = {"2025_01": [1, 2, 3], "2025_02": [1, 3, 2], "2025_03": [2, 1, 3], "2026_01": [3, 1, 2]}
    circuit = {"2025_01": "Melbourne", "2025_02": "Shanghai", "2025_03": "Suzuka", "2026_01": "Melbourne"}
    rows = [{"RaceId": r, "Year": int(r[:4]), "Driver": d, "Team": team, "Final": f, "Pace": -0.01 * (4 - f),
             "Retired": 0, "Circuit": circuit[r]}
            for r, fs in finishes.items() for (d, team), f in zip([("A", "T1"), ("B", "T1"), ("C", "T2")], fs)]
    return pd.DataFrame(rows)


def at(rows, race, driver):
    return rows[(rows["RaceId"] == race) & (rows["Driver"] == driver)].iloc[0]


def test_the_first_race_knows_nothing_yet():
    first = rp.add_history(races())
    first = first[first["RaceId"] == "2025_01"]
    assert first[["DriverForm5", "LastFinish", "SeasonAvg", "TrackDriver"]].isna().all().all()
    assert (first["SeasonWins"] == 0).all()


def test_the_season_counts_start_again_each_year():
    rows = rp.add_history(races())
    assert at(rows, "2025_03", "A")["SeasonWins"] == 2                       # won races 1 and 2
    assert at(rows, "2026_01", "A")["SeasonWins"] == 0                       # a new season
    assert pd.isna(at(rows, "2026_01", "A")["SeasonAvg"])
    assert at(rows, "2026_01", "A")["DriverForm5"] == pytest.approx(4 / 3)    # form carries over: P1, P1, P2
    assert at(rows, "2026_01", "A")["TrackDriver"] == 1                      # won here (Melbourne) in 2025


@pytest.mark.parametrize("race", ["2025_02", "2025_03", "2026_01"])
def test_only_races_before_are_used(race):
    """Change a race's result (and everything after it): what was known before that race stays the same."""
    rows = races()
    changed = rows.copy()
    later = changed["RaceId"] >= race
    changed.loc[later, "Final"] = changed.loc[later, "Final"].map({1: 3, 2: 1, 3: 2})
    changed.loc[later, "Pace"] = 0.05
    before, after = rp.add_history(rows), rp.add_history(changed)
    upto = lambda r: r[r["RaceId"] <= race].sort_values(["RaceId", "Driver"]).reset_index(drop=True)[HISTORY]
    pd.testing.assert_frame_equal(upto(before), upto(after))


def test_the_next_race_preview_adds_up():
    """models/preview_next.json (the next race): one winner, three on the podium, ten in the points."""
    preview = json.loads((MODELS_DIR / "preview_next.json").read_text(encoding="utf-8"))
    drivers = preview["drivers"]
    assert sum(d["win"] for d in drivers) == pytest.approx(1, abs=0.01)
    assert sum(d["podium"] for d in drivers) == pytest.approx(3, abs=0.01)
    assert sum(d["points"] for d in drivers) == pytest.approx(10, abs=0.01)
    assert all(d["win"] <= d["podium"] <= d["points"] for d in drivers)
    for d in drivers:                                                        # the 8 biggest reasons, biggest first
        sizes = [abs(r["points"]) for r in d["why"]["win"]["reasons"]]
        assert len(sizes) == 8 and sizes == sorted(sizes, reverse=True)
