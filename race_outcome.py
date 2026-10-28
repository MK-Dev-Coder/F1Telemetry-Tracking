"""Race outcome: each driver's chance to win, to finish on the podium and in the points, lap by lap.

One LightGBM model answers "will this driver finish in the top k?" for every k from 1 (a win) to 20.
k is one of its inputs, and the model is told that a bigger k can only mean a bigger chance (a monotone
constraint), so the answers always fit together: a podium chance is never below the win chance.
Everything it sees is known at the end of the lap - nothing from the future.
"""
import gzip
import json
import pickle
import re
import sqlite3

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from config import CACHE_DIR, DATA_DIR, MODELS_DIR, OPENF1_JSON_DIR
from explain_pit_model import sigmoid

RESULTS_FILE = DATA_DIR / "race_results.parquet"
MODEL_FILE, INFO_FILE = MODELS_DIR / "race_model.txt", MODELS_DIR / "race_model.json"
BACKGROUND_FILE = MODELS_DIR / "race_background.parquet"     # SHAP's "average driver" (written by train_race_model.py)
MAX_K = 20                                    # top 1 ... top 20 (2026 has 22 cars: 21 and 22 come from the fitting step)
MAX_CARS = 22
SHOWN = {"win": 1, "podium": 3, "points": 10}
RESULT_COLUMNS = ["RaceId", "Driver", "Final", "Grid", "Status"]
# The tyre as a number, softest to hardest, then the wet-weather tyres. (No text features: SHAP can then compare
# with an average driver - its background data doesn't work with LightGBM's category splits.)
TYRES = {"SOFT": 1, "MEDIUM": 2, "HARD": 3, "INTERMEDIATE": 4, "WET": 5}
# The features that depend on k: where the driver is against the top-k line (P3 for the podium, P10 for the points)
K_FEATURES = ["PlacesToK", "MarginK", "MarginKPerLap", "AdjustedPlacesToK", "AdjustedMarginK", "GridToK"]
AHEAD = ["GapAhead", "PaceVsAhead", "TyreAgeVsAhead", "StopsVsAhead"]     # empty for the leader
BEHIND = ["GapBehind", "PaceVsBehind"]                                   # empty for the last car
GROUPS = {                                    # the feature groups, added one at a time (picked on 2025)
    # Each place and gap is given against the top-k line (P3 for the podium), not as a plain position: the
    # model's job is "will they be inside the line at the end?", and one fact told twice splits SHAP's credit
    "Race situation": ["PlacesToK", "MarginK", "MarginKPerLap", "GridToK", "GapAhead", "GapBehind", "LapsLeft",
                       "RaceProgress", "SafetyCar", "VSC"],
    "Pit stops to come": ["AdjustedPlacesToK", "AdjustedMarginK", "StopsMade", "StopsVsAhead", "StopsVsLeader"],
    "Pace and tyres": ["Pace", "PaceVsAhead", "PaceVsBehind", "PaceVsLeader", "Tyre", "TyreLife", "TyreAgeVsAhead",
                       "FieldOnWetTyres"],
    "Form": ["TeamForm", "DriverForm"],
}                                             # (the track's name was tried too: it never helped on 2025)


# Recipe 1: the official result of the races FastF1 has saved (all of 2022 and 2023, and two more), from
# its saved web answers (Jolpica). The cache file is opened read-only: nothing in it changes.
def official_results():
    path = CACHE_DIR / "fastf1_http_cache.sqlite"
    rows = []
    if not path.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)
    con = sqlite3.connect(f"{path.as_uri()}?mode=ro&immutable=1", uri=True)
    for (value,) in con.execute("SELECT value FROM responses"):
        answer = pickle.loads(value)                  # FastF1's own cache of its downloads
        found = re.search(r"/f1/(\d{4})/(\d+)/results\.json", answer.get("request", {}).get("url", ""))
        if not found:
            continue
        for race in json.loads(answer["_content"])["MRData"]["RaceTable"]["Races"]:
            for r in race["Results"]:
                rows.append({"RaceId": f"{found[1]}_{int(found[2]):02d}", "Driver": r["Driver"].get("code"),
                             "Final": int(r["position"]), "Grid": int(r["grid"]), "Status": r["status"]})
    con.close()
    return pd.DataFrame(rows, columns=RESULT_COLUMNS).drop_duplicates(["RaceId", "Driver"])   # (a race saved twice)


# Recipe 2: the order at the flag from OpenF1's position feed (2023 onwards): each car's last position.
# The first position of each car is its grid slot.
def openf1_results():
    rows = []
    for file in sorted(OPENF1_JSON_DIR.glob("*.json.gz")):
        race = json.loads(gzip.open(file).read())
        code = {d["driver_number"]: d["name_acronym"] for d in race["drivers"]}
        position = pd.DataFrame(race["position"]).sort_values("date").groupby("driver_number")["position"]
        for number, last in position.last().items():
            rows.append({"RaceId": file.name[:7], "Driver": code.get(number, str(number)), "Final": int(last),
                         "Grid": int(position.get_group(number).iloc[0]), "Status": None})
    return pd.DataFrame(rows, columns=RESULT_COLUMNS).drop_duplicates(["RaceId", "Driver"])


# The official result where we have it (it includes penalties given after the race), OpenF1 for the rest.
# A pit lane start (grid 0) counts as the back of the grid.
def build_results():
    official, openf1 = official_results(), openf1_results()
    both = official.merge(openf1, on=["RaceId", "Driver"], suffixes=("", "_openf1"))
    results = pd.concat([official.assign(Source="official"),
                         openf1[~openf1["RaceId"].isin(official["RaceId"])].assign(Source="OpenF1")], ignore_index=True)
    cars = results.groupby("RaceId")["Driver"].transform("size")
    results["Grid"] = results["Grid"].where(results["Grid"] > 0, cars)
    results.to_parquet(RESULTS_FILE, index=False)
    return results, both


def load_results():
    return pd.read_parquet(RESULTS_FILE) if RESULTS_FILE.exists() else build_results()[0]


# Recipe 3: the features. laps = every lap of every race (the replay laps, which already have the pit stop
# features), sorted or not. Returns the laps sorted by race, driver and lap, with the new columns.
def add_features(laps, results):
    laps = laps.drop(columns=[c for c in ["Grid", "Final"] if c in laps], errors="ignore")
    laps = laps.merge(results[["RaceId", "Driver", "Grid", "Final"]], on=["RaceId", "Driver"], how="left")
    laps = laps.sort_values(["RaceId", "Driver", "LapNumber"]).reset_index(drop=True)
    by = [laps["RaceId"], laps["Driver"]]

    # Pace: the median of the last 5 green laps against the field (-0.01 = 1% quicker than the field)
    clean = ((laps["TrackStatus"] == "1") & (laps["LapNumber"] > 1) & laps["PitOutTime"].isna()
             & laps["PitInTime"].isna() & (laps["PaceVsField"] < 0.15))
    green = laps.loc[clean, ["RaceId", "Driver", "PaceVsField"]]
    recent = green.groupby(["RaceId", "Driver"])["PaceVsField"].rolling(5, min_periods=1).median()
    laps["Pace"] = recent.reset_index(level=[0, 1], drop=True).reindex(laps.index)
    laps["Pace"] = laps.groupby(by)["Pace"].ffill()
    # A usual lap time for this race so far (only to say the pace in seconds)
    usual = laps[clean].groupby(["RaceId", "LapNumber"])["LapTime"].median()
    usual = usual.groupby(level=0).expanding().median().reset_index(level=0, drop=True)
    laps["UsualLap"] = usual.reindex(pd.MultiIndex.from_frame(laps[["RaceId", "LapNumber"]])).values
    laps["UsualLap"] = laps.groupby(by)["UsualLap"].ffill()

    laps["Tyre"] = laps["Compound"].astype(str).map(TYRES)

    # Pit stops so far, and the F1 rule: a dry race needs two dry compounds (not if wet tyres were used)
    laps["StopsMade"] = laps.groupby(by)["IsStop"].cumsum() - laps["IsStop"].astype(int)   # stops before this lap
    # (IsStop is set from the next lap - new tyres or not - so the stop at the end of this lap counts from the next one)
    wet = laps["Compound"].isin(["INTERMEDIATE", "WET"]).groupby(by).cummax()
    laps["MustStillStop"] = ((laps["DryCompoundsUsed"] < 2) & ~wet).astype(int)

    laps = lap_relations(laps)

    # Form: the average finish of the team's cars and of the driver in their last 5 races this season
    # (only races before this one; empty at the first race of a season, e.g. after a rule change like 2026)
    teams = laps.drop_duplicates(["RaceId", "Driver"])[["RaceId", "Driver", "Team"]]
    done = results.merge(teams, on=["RaceId", "Driver"]).sort_values("RaceId")
    done["Year"] = done["RaceId"].str[:4]

    def form(who):
        avg = done.groupby(["Year", who, "RaceId"], sort=True)["Final"].mean()
        before = avg.groupby(level=[0, 1]).transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        return before.droplevel(0)

    team, driver = form("Team"), form("Driver")
    laps["TeamForm"] = team.reindex(pd.MultiIndex.from_frame(laps[["Team", "RaceId"]])).values
    laps["DriverForm"] = driver.reindex(pd.MultiIndex.from_frame(laps[["Driver", "RaceId"]])).values
    return laps


# Each car against the car just ahead, just behind and the leader on its lap (also used again by pit_now.py, with a
# car moved to where it would rejoin after a stop)
def lap_relations(laps):
    key = ["RaceId", "LapNumber", "Position"]
    placed = laps.dropna(subset=["Position"]).drop_duplicates(key).set_index(key)

    def rival(column, position):
        return placed[column].reindex(pd.MultiIndex.from_arrays([laps["RaceId"], laps["LapNumber"], position])).values

    laps["PaceVsAhead"] = laps["Pace"] - rival("Pace", laps["Position"] - 1)            # below 0 = quicker
    laps["PaceVsBehind"] = laps["Pace"] - rival("Pace", laps["Position"] + 1)
    laps["PaceVsLeader"] = laps["Pace"] - rival("Pace", pd.Series(1.0, index=laps.index))
    laps["TyreAgeVsAhead"] = laps["TyreLife"] - rival("TyreLife", laps["Position"] - 1)  # above 0 = older tyres
    laps["StopsVsAhead"] = laps["StopsMade"] - rival("StopsMade", laps["Position"] - 1)  # above 0 = more stops done
    laps["StopsVsLeader"] = laps["StopsMade"] - rival("StopsMade", pd.Series(1.0, index=laps.index))
    return laps


# The usual time a pit stop costs at each track: the in-lap and the out-lap against two normal laps
# (green flag only). Worked out from the training races, then kept in the model's file.
def pit_loss_table(laps):
    laps = laps.sort_values(["RaceId", "Driver", "LapNumber"])
    by = laps.groupby(["RaceId", "Driver"])
    normal = laps[(laps["TrackStatus"] == "1") & laps["PitInTime"].isna() & laps["PitOutTime"].isna()]
    normal = normal.groupby(["RaceId", "Driver"])["LapTime"].median()
    normal = normal.reindex(pd.MultiIndex.from_frame(laps[["RaceId", "Driver"]])).values
    loss = laps["LapTime"] + by["LapTime"].shift(-1) - 2 * normal
    green = (laps["TrackStatus"] == "1") & (by["TrackStatus"].shift(-1) == "1")
    ok = laps["IsStop"] & green & loss.between(5, 60)
    track = loss[ok].groupby(laps.loc[ok, "Circuit"].astype(str)).median()
    return {"track": track.round(1).to_dict(), "overall": round(float(loss[ok].median()), 1)}


# The race order once the compulsory stop still to come is counted: a leader who still has to pit is
# really about one pit stop (20-30 s) behind where the gap says.
def add_pit_adjusted(laps, table):
    laps["PitLoss"] = laps["Circuit"].astype(str).map(table["track"]).astype(float).fillna(table["overall"])
    behind = (laps["GapToLeader"] + laps["PitLoss"] * laps["MustStillStop"]).where(laps["Position"].notna())
    lap = behind.groupby([laps["RaceId"], laps["LapNumber"]])
    laps["AdjustedGap"] = behind - lap.transform("min")
    laps["AdjustedPosition"] = lap.rank(method="first")
    return laps


# Recipe 4: where each driver is against the top-k line, for each k. rows = whole laps (every car on them).
def on_lap(rows, key, value):
    """For each row: `value` of the car in each place (1 ... 23) on the same lap, in the order of `key`."""
    lap = rows.groupby(["RaceId", "LapNumber"], sort=False).ngroup().values
    place, v = rows[key].values.astype(float), rows[value].values.astype(float)
    table = np.full((lap.max() + 1 if len(lap) else 0, MAX_CARS + 2), np.nan)
    ok = (place >= 1) & (place <= MAX_CARS + 1)                            # (NaN is never >= 1)
    table[lap[ok], place[ok].astype(int)] = v[ok]
    return table[lap, 1:]


def margin(rows, ks, key, value):
    """Seconds clear of the first car outside the top k (above 0), or behind the last car inside it (below 0)."""
    gaps, kk = on_lap(rows, key, value), np.array(ks)
    mine, place = rows[value].values[:, None], rows[key].values[:, None]
    return np.where(place <= kk, gaps[:, kk] - mine, gaps[:, kk - 1] - mine).clip(-120, 120)


def k_features(rows, ks):
    """The k features as rows x len(ks) arrays."""
    kk = np.array(list(ks))[None, :]
    near = margin(rows, ks, "Position", "GapToLeader")
    return {"PlacesToK": rows["Position"].values[:, None] - kk, "MarginK": near,
            "MarginKPerLap": near / rows["LapsLeft"].clip(lower=0.5).values[:, None],
            "AdjustedPlacesToK": rows["AdjustedPosition"].values[:, None] - kk,
            "AdjustedMarginK": margin(rows, ks, "AdjustedPosition", "AdjustedGap"),
            "GridToK": rows["Grid"].values[:, None] - kk}


def at_k(rows, k):
    """The rows with the k features for one k (to put the reasons into words)."""
    out = rows.copy()
    for name, values in k_features(rows, [k]).items():
        out[name] = values[:, 0]
    return out


# One row per lap and per k, with the answer (did the driver finish in the top k?)
def stack(rows, feats, ks=range(1, MAX_K + 1)):
    ks = list(ks)
    X = rows[[f for f in feats if f not in K_FEATURES]].loc[rows.index.repeat(len(ks))].reset_index(drop=True)
    X["K"] = np.tile(ks, len(rows)).astype(float)
    for name, values in k_features(rows, ks).items():
        if name in feats:
            X[name] = values.ravel()                                      # row 1 k=1, k=2, ..., row 2 k=1, ...
    return X[["K"] + feats]


def answers(rows, ks=range(1, MAX_K + 1)):
    ks = np.array(list(ks))
    return (rows["Final"].values[:, None] <= ks[None, :]).astype(int).ravel()


def load_model():
    info = json.loads(INFO_FILE.read_text(encoding="utf-8"))
    return lgb.Booster(model_file=str(MODEL_FILE)), info


# Recipe 5: make the chances add up across the field. p = drivers x k (top 1, top 2, ...). Among the n cars
# on this lap there is exactly one winner, and k cars in the top k (cars out of the race are behind them all).
# Each column is moved on the log-odds scale (the same step for every driver) until it adds up, so the order
# of the drivers never changes and no chance reaches 100% unless it must (every car on the lap is in the top n).
def fit_together(p):
    n, ks = p.shape[0], np.arange(1, p.shape[1] + 1)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z, shift = np.log(p / (1 - p)), np.zeros(p.shape[1])
    target = np.minimum(ks, n).astype(float)
    for _ in range(30):                                                    # Newton's method
        q = sigmoid(z + shift)
        shift -= np.clip((q.sum(axis=0) - target) / np.maximum((q * (1 - q)).sum(axis=0), 1e-9), -5, 5)
    q = sigmoid(z + shift)
    q[:, ks >= n] = 1.0
    return np.maximum.accumulate(q, axis=1)


# The chances lap by lap: raw = rows x k (the model's own answers), rows = the laps of one or more races.
# A lap with n cars in places P1 ... Pn uses k = 1 ... n; beyond that every car is surely in the top k.
# A lap with places missing from the timing (e.g. Miami 2025, lap 25: only P10-P19) keeps the model's own answers.
def by_lap(raw, rows, together=True):
    out, position = np.ones_like(raw), rows["Position"].values
    for ix in rows.groupby(["RaceId", "LapNumber"]).indices.values():
        n = min(len(ix), raw.shape[1])
        if not np.array_equal(np.sort(position[ix]), np.arange(1, len(ix) + 1)):
            out[ix, :-1] = np.maximum.accumulate(raw[ix, :-1], axis=1)       # (the last k: every car)
        else:
            out[ix, :n] = fit_together(raw[ix, :n]) if together else np.maximum.accumulate(raw[ix, :n], axis=1)
    return out


def chances(booster, info, rows):
    """Every driver's chance of each top k (rows x 22) at the end of each lap in rows."""
    raw = booster.predict(stack(rows, info["features"], range(1, MAX_CARS + 1))).reshape(len(rows), MAX_CARS)
    return by_lap(raw, rows, info.get("fit_together", True))


def summary(fitted):
    """Win, podium and points chance, the likely finish (the median: at least an even chance of that place or
    better), the expected (average) finish and the chance of each finishing position."""
    n = fitted.shape[1]
    at_most = np.hstack([np.zeros((len(fitted), 1)), fitted])              # P(top 0) = 0
    spot = np.diff(at_most, axis=1)                                        # P(finish exactly k)
    expected = (1 - at_most[:, :-1]).sum(axis=1)                           # = sum over k of P(behind k)
    pick = lambda k: fitted[:, min(k, n) - 1]
    likely = np.argmax(fitted >= 0.5, axis=1) + 1                             # every car is surely in the top n
    return {"win": pick(1), "podium": pick(3), "points": pick(10), "likely": likely, "expected": expected, "spot": spot}


# Recipe 6: SHAP for one k, in % points. The starting point is the average driver: a fixed sample of 2022-2025
# driver-laps asked the same question (about 5% for a win, 15% for a podium, 50% for points), and each reason moves
# the chance away from it; together they add up exactly to the model's chance (interventional TreeSHAP).
def make_background(rows, feats, laps=60, size=100):
    """100 driver-laps from 60 random laps of the training races, for each k shown."""
    pick = rows[["RaceId", "LapNumber"]].drop_duplicates().sample(laps, random_state=42)
    whole = rows.merge(pick, on=["RaceId", "LapNumber"])                  # whole laps: the k features need every car
    return pd.concat([stack(whole, feats, [k]).sample(size, random_state=42) for k in SHOWN.values()], ignore_index=True)


EXPLAINERS = {}


def explainer(booster, info, k):
    if (id(booster), k) not in EXPLAINERS:
        background = pd.read_parquet(BACKGROUND_FILE)
        EXPLAINERS[(id(booster), k)] = shap.TreeExplainer(
            booster, data=background[background["K"] == k][["K"] + info["features"]],
            feature_perturbation="interventional", model_output="probability")
    return EXPLAINERS[(id(booster), k)]


def explain(booster, info, rows, k):
    """SHAP values (% points, drivers x features), the average driver's chance, and the rows with that k's features
    (to put the reasons into words). k is the same for the driver and the average driver, so it plays no part."""
    ex = explainer(booster, info, k)
    values = ex.shap_values(stack(rows, info["features"], [k]))
    return pd.DataFrame(values[:, 1:], columns=info["features"], index=rows.index), float(ex.expected_value), at_k(rows, k)


INSIDE = {1: "leading now", 3: "in a podium place now", 10: "in the points now"}
OFF = {1: "the lead", 3: "the podium", 10: "the points"}


def unknown(name, row, k):
    """What to say when a value isn't known: lap 1 has no green laps yet (no pace), the timing misses a car..."""
    pos = row["Position"]
    ahead, behind = f"P{pos - 1:.0f}", f"P{pos + 1:.0f}"
    if name in ("MarginK", "MarginKPerLap", "AdjustedMarginK"):
        places = row.get("AdjustedPlacesToK" if name == "AdjustedMarginK" else "PlacesToK")
        return f"gap to P{k} not known" if pd.notna(places) and places > 0 else "no car behind to take the place"
    if name in AHEAD and pos == 1:
        return {"GapAhead": "leading the race, no car ahead", "PaceVsAhead": "no car ahead to compare pace with",
                "TyreAgeVsAhead": "no car ahead to compare tyres with", "StopsVsAhead": "no car ahead to compare stops with"}[name]
    if name in BEHIND and pd.isna(row.get("GapBehind")):
        return "no car close behind" if name == "GapBehind" else "no car behind to compare pace with"
    return {"GapAhead": f"gap to {ahead} not known", "GapBehind": f"gap to {behind} not known",
            "Pace": "no green laps yet to judge the pace", "PaceVsLeader": "no green laps yet to compare pace with the leader",
            "PaceVsAhead": f"no green laps yet to compare pace with {ahead}",
            "PaceVsBehind": f"no green laps yet to compare pace with {behind}",
            "TyreAgeVsAhead": f"{ahead}'s tyre age not known", "StopsVsAhead": f"{ahead}'s pit stops not known",
            "StopsVsLeader": "the leader's pit stops not known", "Tyre": "tyre compound not known",
            "TyreLife": "tyre age not known", "GridToK": "starting place not known",
            "PlacesToK": "place not known", "AdjustedPlacesToK": "place once the stops are counted not known",
            "TeamForm": "no team form yet (first race of the season)", "DriverForm": "no form yet (first race of the season)",
            }.get(name, "not known")


def describe(name, row, effect=0.0, k=None):
    v, pos = row[name], row["Position"]
    if pd.isna(v):
        return unknown(name, row, k)
    if name == "Tyre":
        return f"on {str(row['Compound']).lower()} tyres"
    v = float(v)
    lap = row.get("UsualLap")
    secs = lambda x: f"{abs(x) * lap:.1f} s a lap" if pd.notna(lap) else f"{abs(x):.1%}"
    laps = lambda x: f"{abs(x):.0f} lap" + ("" if round(abs(x)) == 1 else "s")
    stops = lambda x: f"{abs(x):.0f} stop" + ("" if round(abs(x)) == 1 else "s")
    text = {
        "GapAhead": f"{v:.1f} s behind P{pos - 1:.0f}",
        "GapBehind": f"{v:.1f} s ahead of P{pos + 1:.0f}",
        "LapsLeft": f"{laps(v)} to go",
        "RaceProgress": f"{v:.0%} of the race done",
        "SafetyCar": "safety car out: the field is bunched up" if v else "no safety car",
        "VSC": "virtual safety car out" if v else "no virtual safety car",
        "Pace": f"{secs(v)} {'quicker' if v < 0 else 'slower'} than the field (last 5 green laps)",
        "PaceVsAhead": f"{secs(v)} {'quicker' if v < 0 else 'slower'} than P{pos - 1:.0f}",
        "PaceVsBehind": f"{secs(v)} {'quicker' if v < 0 else 'slower'} than P{pos + 1:.0f}",
        "PaceVsLeader": "the leader" if pos == 1 else f"{secs(v)} {'quicker' if v < 0 else 'slower'} than the leader",
        "TyreLife": f"tyres {laps(v)} old",
        "StopsMade": f"{stops(v)} made" if v else "no pit stop yet",
        "TyreAgeVsAhead": (f"tyres {laps(v)} {'older' if v > 0 else 'fresher'} than P{pos - 1:.0f}'s" if v
                           else f"tyres as old as P{pos - 1:.0f}'s"),
        "StopsVsAhead": (f"{stops(v)} {'more' if v > 0 else 'fewer'} than P{pos - 1:.0f}" if v
                         else f"same stops as P{pos - 1:.0f}"),
        "StopsVsLeader": ("the leader" if pos == 1 else f"{stops(v)} {'more' if v > 0 else 'fewer'} than the leader" if v
                          else "same stops as the leader"),
        "FieldOnWetTyres": f"{v:.0%} of cars on wet-weather tyres",
        "PlacesToK": (INSIDE.get(k, f"in the top {k} now") if v <= 0 else
                      f"{v:.0f} place{'' if v == 1 else 's'} off {OFF.get(k, f'the top {k}')}"),
        "MarginK": f"{v:.1f} s clear of P{(k or 0) + 1}" if v >= 0 else f"{-v:.1f} s behind P{k}",
        "MarginKPerLap": (f"P{(k or 0) + 1} needs {abs(v):.{2 if abs(v) < 0.1 else 1}f} s a lap to catch them" if v >= 0
                          else f"needs {abs(v):.{2 if abs(v) < 0.1 else 1}f} s a lap to catch P{k}"),
        "AdjustedPlacesToK": (f"{INSIDE.get(k, f'in the top {k} now')} once the stops still to make are counted" if v <= 0 else
                              f"{v:.0f} place{'' if v == 1 else 's'} off {OFF.get(k, f'the top {k}')} once the stops still "
                              "to make are counted"),
        "AdjustedMarginK": (f"{v:.1f} s clear of P{(k or 0) + 1}" if v >= 0 else f"{-v:.1f} s behind P{k}")
                           + " once the stops still to make are counted",
        "GridToK": f"started P{row['Grid']:.0f}",
        "TeamForm": f"team averaged P{v:.1f} in its last races",
        "DriverForm": f"averaged P{v:.1f} in the last races",
    }
    return text.get(name, f"{name} = {v}")
