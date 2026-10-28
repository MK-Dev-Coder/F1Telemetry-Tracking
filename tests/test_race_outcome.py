"""race_outcome.py: each driver's chance to win, finish on the podium and score points."""
import numpy as np
import pandas as pd
import pytest

import race_outcome as ro


def field(n, seed=None):
    """The model's own answers for n cars (cars x top 1 ... top n): a field like a real one, car 1 the quickest -
    or, with a seed, random answers that only go up with k."""
    if seed is not None:
        return np.sort(np.random.default_rng(seed).uniform(0.02, 0.98, (n, n)), axis=1)
    skill, ks = np.linspace(2, -2, n)[:, None], np.arange(1, n + 1)[None, :]
    return 1 / (1 + np.exp(-(skill + np.log(ks / (n - ks + 0.5)))))


@pytest.mark.parametrize("n", [3, 10, 20, 22])
def test_one_winner_three_on_the_podium(n):
    raw = field(n)
    q = ro.fit_together(raw)
    assert q.sum(axis=0) == pytest.approx(np.arange(1, n + 1), abs=1e-6)    # the top k has k cars
    for k in range(n - 1):                                                   # the drivers keep their order
        assert (np.argsort(-raw[:, k], kind="stable") == np.argsort(-q[:, k], kind="stable")).all()


@pytest.mark.parametrize("seed", range(20))
def test_chances_fit_together_whatever_the_model_says(seed):
    """Even with random answers (unlike a real field): one winner, the favourite stays the favourite, and a podium
    chance is never below the win chance. Making the chances only go up with k can then lift a top k a little above
    k cars (at most 0.14 of a car in 300 random fields; 0 for a field like a real one, above)."""
    raw = field(20, seed)
    q = ro.fit_together(raw)
    assert q[:, 0].sum() == pytest.approx(1, abs=1e-6)                       # exactly one winner
    assert (np.argsort(-raw[:, 0], kind="stable") == np.argsort(-q[:, 0], kind="stable")).all()
    assert (np.diff(q, axis=1) >= 0).all()                                   # podium >= win, points >= podium
    assert (q.sum(axis=0) >= np.arange(1, 21) - 1e-6).all()
    assert (q.sum(axis=0) <= np.arange(1, 21) + 0.25).all()
    assert (q[:, -1] == 1).all()                                             # every car is in the top 20 of 20


def test_summary():
    q = ro.fit_together(field(3))
    s = ro.summary(np.hstack([q, np.ones((3, ro.MAX_CARS - 3))]))
    assert s["win"].sum() == pytest.approx(1)
    assert s["spot"].sum(axis=1) == pytest.approx(np.ones(3))                # each car finishes somewhere
    assert s["likely"].tolist() == [1, 2, 3]                                 # the quickest car is likeliest to win
    assert (s["expected"] >= 1).all() and (s["expected"] <= 3).all()


def lap_rows(positions, lap=1):
    return pd.DataFrame({"RaceId": "2025_01", "LapNumber": lap, "Position": positions})


def test_a_lap_with_places_missing_keeps_the_models_answers():
    raw = np.tile(np.linspace(0.1, 0.9, ro.MAX_CARS), (3, 1))
    out = ro.by_lap(raw, lap_rows([2.0, 3.0, 5.0]))                          # P1 and P4 missing from the timing
    assert out[:, :-1] == pytest.approx(raw[:, :-1])                         # not forced to add up
    complete = ro.by_lap(raw, lap_rows([1.0, 2.0, 3.0]))
    assert complete[:, 0].sum() == pytest.approx(1)


def cars_on_a_lap():
    return pd.DataFrame({"RaceId": "2025_01", "LapNumber": 30.0, "Position": [1.0, 2.0, 3.0, 4.0],
                         "GapToLeader": [0.0, 2.0, 5.0, 9.0], "AdjustedPosition": [1.0, 2.0, 3.0, 4.0],
                         "AdjustedGap": [0.0, 2.0, 5.0, 9.0], "LapsLeft": 10.0, "Grid": [3.0, 1.0, 2.0, 4.0]})


def test_the_top_k_line():
    f = ro.k_features(cars_on_a_lap(), [1, 3])                               # the win and the podium
    assert f["PlacesToK"][:, 0].tolist() == [0, 1, 2, 3]
    assert f["MarginK"][:, 0].tolist() == [2, -2, -5, -9]                    # leader 2 s clear of P2; others behind P1
    assert f["MarginK"][:, 1].tolist() == [9, 7, 4, -4]                      # P4 is 4 s behind P3
    assert f["MarginKPerLap"][0, 0] == pytest.approx(0.2)                    # 2 s with 10 laps to go
    assert f["GridToK"][:, 0].tolist() == [2, 0, 1, 3]


def test_answers():
    rows = pd.DataFrame({"Final": [2]})
    assert ro.answers(rows, [1, 2, 3]).tolist() == [0, 1, 1]                 # P2: not a win, top 2, podium


def test_a_leader_who_must_still_stop_is_really_behind():
    rows = cars_on_a_lap().assign(Circuit="Austin", MustStillStop=[1, 0, 0, 0], GapToLeader=[0.0, 10.0, 30.0, 40.0])
    rows = ro.add_pit_adjusted(rows, {"track": {"Austin": 22.0}, "overall": 21.0})
    assert rows["AdjustedPosition"].tolist() == [2, 1, 3, 4]                 # 0 + 22 s is behind P2's 10 s
    assert rows["AdjustedGap"].tolist() == [12, 0, 20, 30]


def test_reasons_in_plain_english():
    row = ro.at_k(cars_on_a_lap().assign(Compound="MEDIUM", UsualLap=95.0), 1).iloc[0]
    row["MarginKPerLap"] = 0.6
    assert ro.describe("MarginKPerLap", row, k=1) == "P2 needs 0.6 s a lap to catch them"
    assert ro.describe("PlacesToK", row, k=3) == "in a podium place now"
    assert ro.describe("GridToK", row, k=1) == "started P3"
    row["Pace"] = np.nan
    assert ro.describe("Pace", row, k=1) == "no green laps yet to judge the pace"
