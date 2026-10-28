"""Shared bits for the tests. Run them from the F1 App folder:  python -m pytest tests

The tests use small made-up races, so they never read or change the real data. Only test_with_data.py reads the
real data folder (without changing it), and it is skipped when the folder isn't there (it isn't on GitHub).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))      # the app's own files (features.py, ...)

LAPS = 10
CARS = [("AAA", 0.0), ("BBB", 0.5), ("CCC", 1.0), ("DDD", 15.0)]      # each car's seconds a lap slower than AAA
IN_LAPS = {("AAA", 4), ("BBB", 6), ("CCC", 7)}                       # into the pits at the end of the lap
NEW_TYRES = {("AAA", 5), ("CCC", 8)}                                 # out of the pits on hard tyres
OUT_LAPS = NEW_TYRES | {("BBB", 7)}                                  # BBB: a drive-through, same tyres


def make_race():
    """A made-up 10-lap race with 4 cars, in the same columns as data/all_laps.parquet.
    AAA makes a real stop at the end of lap 4 (medium -> hard). BBB takes a drive-through on lap 6 (same tyres, so
    not a stop). CCC changes tyres in the pit lane under the red flag on lap 7 (not a race stop). DDD is 15 s a lap
    slower than everyone. The safety car is out on laps 8 and 9."""
    rows = []
    for driver, slower in CARS:
        start, compound, life, stint = 3600.0, "MEDIUM", 0.0, 1.0       # session time at the start of the lap
        for lap in range(1, LAPS + 1):
            if (driver, lap) in NEW_TYRES:
                compound, life, stint = "HARD", 0.0, stint + 1
            life += 1
            lap_time, pit_in, pit_out = 90.0 + slower, np.nan, np.nan
            if (driver, lap) in OUT_LAPS:
                lap_time, pit_out = lap_time + 18, start + 20
            if (driver, lap) in IN_LAPS:
                lap_time += 5
                pit_in = start + lap_time - 5
            status = "5" if lap == 7 else "4" if lap in (8, 9) else "1"
            rows.append({"Driver": driver, "Team": f"Team {driver[0]}", "LapNumber": float(lap), "LapTime": lap_time,
                         "Time": start + lap_time, "Stint": stint, "Compound": compound, "TyreLife": life,
                         "FreshTyre": True, "PitInTime": pit_in, "PitOutTime": pit_out, "TrackStatus": status,
                         "AirTemp": 25.0, "TrackTemp": 35.0, "Humidity": 50.0, "Rainfall": False, "Year": 2025,
                         "Round": 1, "EventName": "Test Grand Prix", "TotalLaps": LAPS, "Circuit": "Austin",
                         "Source": "test", "RaceId": "2025_01"})
            start += lap_time
    race = pd.DataFrame(rows)
    race["Position"] = race.groupby("LapNumber")["Time"].rank(method="first")
    return race


@pytest.fixture
def race():
    return make_race()


def lap(laps, driver, n):
    """One driver's row on one lap."""
    return laps[(laps["Driver"] == driver) & (laps["LapNumber"] == n)].iloc[0]
