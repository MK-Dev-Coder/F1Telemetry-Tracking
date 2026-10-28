"""If they pit now: where a driver would rejoin, the gaps to the cars around, and how their race outcome changes.

The stop costs the track's usual pit stop time (learned from 2022-2025, the same numbers the race outcome model
uses). Under a safety car or a VSC a stop costs less, because the cars that stay out are slow too - how much less
is learned from the same races. Then the race outcome model is asked again with the driver moved to where they'd
rejoin, on new tyres (the tyre model's pick), with the stop done. It assumes nobody else pits on that lap and the
others keep their gaps.
    python pit_now.py   -> models/pit_now.json (what a stop costs under green, safety car and VSC)
"""
import json

import numpy as np
import pandas as pd

import features
import race_outcome as ro
import train_tyre_model as tyre
from config import MODELS_DIR

TABLE_FILE = MODELS_DIR / "pit_now.json"
KINDS = {"SAFETY CAR": "SC", "VSC": "VSC"}                # the track status as the pages have it
DRY = ["SOFT", "MEDIUM", "HARD"]


# Recipe 1: what a stop costs against the cars that stay out - the in-lap and the out-lap against the field's
# laps - under green, the safety car and the VSC, as a share of the track's usual (green) cost
def build():
    laps = tyre.replay_laps()
    laps = laps[features.in_study(laps) & (laps["Year"] <= 2025)].sort_values(["RaceId", "Driver", "LapNumber"])
    laps = laps.reset_index(drop=True)
    by = [laps["RaceId"], laps["Driver"]]
    field = laps.groupby(["RaceId", "LapNumber"])["LapTime"].transform("median")
    loss = laps["LapTime"] + laps["LapTime"].groupby(by).shift(-1) - field - field.groupby(by).shift(-1)
    status = laps["TrackStatus"].astype(str)
    kind = np.select([status.str.contains("4"), status.str.contains("[67]"), status == "1"], ["SC", "VSC", "green"], "other")
    ok = (laps["IsStop"] & loss.between(0, 60)).values
    stops = pd.DataFrame({"loss": loss[ok].values, "kind": kind[ok], "circuit": laps.loc[ok, "Circuit"].astype(str).values})
    usual = stops[stops["kind"] == "green"].groupby("circuit")["loss"].median()
    stops["share"] = stops["loss"] / stops["circuit"].map(usual)
    table = {"share": {k: [round(float(stops.loc[stops["kind"] == k, "share"].quantile(q)), 3) for q in (0.25, 0.5, 0.75)]
                       for k in ("green", "SC", "VSC")},
             "stops": {k: int(v) for k, v in stops["kind"].value_counts().items()}, "trained_on": "2022-2025"}
    TABLE_FILE.write_text(json.dumps(table, indent=1), encoding="utf-8")
    return table


def load():
    return json.loads(TABLE_FILE.read_text(encoding="utf-8")) if TABLE_FILE.exists() else build()


def cost(table, pit_loss, circuit, status):
    """What a stop costs here now (seconds), its usual range (the middle half of the stops) and the track status.
    Under a red flag the cars wait in the pit lane and may change tyres there, so new tyres cost no time."""
    if status == "RED FLAG":
        return 0.0, (0.0, 0.0), "red"
    usual = pit_loss["track"].get(str(circuit), pit_loss["overall"])
    kind = KINDS.get(status, "green")
    low, mid, high = (usual * share for share in table["share"][kind])
    return mid, (low, high), kind


# Recipe 2: where they'd come out. lap = the cars with a place on the lap (Driver, Position, GapToLeader); the stop is
# at the end of this lap, so the car loses `loss` seconds against all the others.
def whole(lap):
    """Can the places after a stop be counted? Only if every place P1 ... Pn is in the timing, each with a gap to
    the leader (a few laps miss cars: Miami 2025 lap 25 has only P10-P19)."""
    return (lap["GapToLeader"].notna().all()
            and sorted(lap["Position"]) == list(range(1, len(lap) + 1)))


def rejoin_all(lap, loss, around=None):
    """Every car's rejoin place if it pitted at the end of this lap: {driver: {position, ahead, behind}}, or {} when
    the places can't be counted. around=driver: only that driver, with the two cars ahead and behind after the stop
    (gap < 0: that car is ahead)."""
    if not whole(lap):
        return {}
    names, gaps, out = lap["Driver"].to_numpy(), lap["GapToLeader"].to_numpy(dtype=float), {}
    order = np.argsort(gaps, kind="stable")
    names, gaps = names[order], gaps[order]
    for k, (name, gap) in enumerate(zip(names, gaps)):
        if around is not None and name != around:
            continue
        after = gap + loss                                                     # behind today's leader
        others = np.delete(np.arange(len(gaps)), k)
        ahead = others[gaps[others] <= after]
        behind = others[gaps[others] > after]
        out[name] = {"position": len(ahead) + 1,
                     "ahead": None if not len(ahead) else {"driver": names[ahead[-1]], "gap": round(float(after - gaps[ahead[-1]]), 1)},
                     "behind": None if not len(behind) else {"driver": names[behind[0]], "gap": round(float(gaps[behind[0]] - after), 1)}}
        if around is not None:
            near = [(j, len(ahead) - len(ahead[-2:]) + i + 1) for i, j in enumerate(ahead[-2:])] + \
                   [(j, len(ahead) + 2 + i) for i, j in enumerate(behind[:2])]
            out[name]["around"] = [{"driver": names[j], "position": p, "gap": round(float(gaps[j] - after), 1)} for j, p in near]
    return out


def rejoin(lap, driver, loss):
    """One driver's rejoin place, with the cars around them after the stop (None when it can't be counted)."""
    return rejoin_all(lap, loss, around=driver).get(driver)


# Recipe 3: the lap again with the driver pitted: their new place and gaps, new tyres (`compound`), the stop done.
# factor: how many real-race laps one lap is worth (F1 25's shortened races scale the tyre ages, as there).
def pitted(lap, driver, loss, compound, info, factor=1.0):
    rows = lap.copy()
    rows["Compound"] = rows["Compound"].astype(str)                       # (a category list may not hold the new tyre)
    i = rows.index[rows["Driver"] == driver][0]
    new = "INTERMEDIATE" if compound == "INTER/WET" else compound if compound in ro.TYRES else rows.at[i, "Compound"]
    used = {c: bool(rows.at[i, f"Used{c.title()}"]) for c in DRY if f"Used{c.title()}" in rows}
    dry_used = rows.at[i, "DryCompoundsUsed"] + (1 if new in DRY and not used.get(new, True) else 0)
    rows.at[i, "DryCompoundsUsed"] = dry_used
    rows.at[i, "MustStillStop"] = int(bool(rows.at[i, "MustStillStop"]) and dry_used < 2 and new in DRY)
    rows.at[i, "Compound"], rows.at[i, "Tyre"], rows.at[i, "TyreLife"] = new, ro.TYRES.get(new), 1.0 * factor
    rows.at[i, "StopsMade"] = rows.at[i, "StopsMade"] + 1
    rows.at[i, "GapToLeader"] = rows.at[i, "GapToLeader"] + loss
    rows = rows.sort_values("GapToLeader").reset_index(drop=True)                 # the new order and gaps
    rows["GapToLeader"] = rows["GapToLeader"] - rows["GapToLeader"].min()
    rows["Position"] = np.arange(1, len(rows) + 1, dtype=float)
    rows["GapAhead"] = rows["GapToLeader"].diff().clip(0, 60)
    rows["GapBehind"] = (-rows["GapToLeader"].diff(-1)).clip(0, 60)
    rows = ro.lap_relations(rows)                                                 # against the new cars around
    return ro.add_pit_adjusted(rows, info["pit_loss"])


def what_if(table, booster, info, lap, driver, status, compound, factor=1.0):
    """Everything about pitting now for one driver: the cost, where they'd rejoin (with its range), and the race
    outcome model's chances now and if they pit. lap = the cars on the lap with the race outcome features."""
    lap = lap[lap["Position"].notna()].reset_index(drop=True)
    loss, (low, high), kind = cost(table, info["pit_loss"], lap["Circuit"].iloc[0], status)
    place = rejoin(lap, driver, loss)
    if place is None:
        return None
    after_rows = pitted(lap, driver, loss, compound, info, factor)
    now, then = ro.summary(ro.chances(booster, info, lap)), ro.summary(ro.chances(booster, info, after_rows))
    i, j = lap.index[lap["Driver"] == driver][0], after_rows.index[after_rows["Driver"] == driver][0]
    change = {t: [round(float(now[t][i]), 3), round(float(then[t][j]), 3)] for t in ro.SHOWN}
    change["likely"] = [int(now["likely"][i]), int(then["likely"][j])]
    return {"driver": driver, "cost": round(loss, 1), "cost_range": [round(low, 1), round(high, 1)], "under": kind,
            "tyre": compound, **place,
            "range": [rejoin(lap, driver, low)["position"], rejoin(lap, driver, high)["position"]],
            "chances": change}


if __name__ == "__main__":
    print(json.dumps(build(), indent=1))
