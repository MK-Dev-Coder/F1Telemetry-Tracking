"""Car against car: the undercut threat, the battle forecast and the strategy battle (two drivers side by side).

Undercut threat: if the car behind pits now and the car just ahead answers on the next lap, does the car behind come
out ahead? What an undercut gains is learned from the real ones of 2022-2025: the car behind pitted one lap before
the car just ahead of it, both under green on dry tyres, and the gap is measured before the first stop and after
the second. It gains more when the car ahead's tyres are worn (TyreWear: its last laps against its best on this
set). The chance is the share of the real undercuts, on tyres worn about as much, that gained more than the gap.

Battle forecast: how many laps until the car behind is within 1 second (striking distance: DRS until 2025, the
overtake mode in 2026), from each car's pace over its last 5 green laps (the race outcome model's pace).

Both are tested on the 17 test races of 2026, like the models.
    python battle.py   -> models/battle.json (the undercut gains) and data/battle_evaluation.md
"""
import json
import math

import numpy as np
import pandas as pd

import features
import race_outcome as ro
import train_tyre_model as tyre
from config import DATA_DIR, MODELS_DIR

TABLE_FILE = MODELS_DIR / "battle.json"
REPORT_FILE = DATA_DIR / "battle_evaluation.md"
DRY = ["SOFT", "MEDIUM", "HARD"]
WEAR = [0.3, 0.7, 1.2]                # the car ahead's tyre wear (s): under 0.3, 0.3-0.7, 0.7-1.2, over 1.2
STRIKE = 1.0                          # striking distance (s)
NEWEST = 5                            # an undercut needs both cars at least 5 laps into their set
AHEAD = 10                            # laps the forecast looks ahead in the test


# Recipe 1: the real undercuts - one row per pair: the car behind (B) pitted at the end of lap n, the car just ahead
# of it on lap n-1 (A) at the end of lap n+1. Gap = B's time - A's time (above 0: B behind); gain = gap before - after.
def undercuts(laps):
    one = laps.set_index(["RaceId", "Driver", "LapNumber"])
    one = one[~one.index.duplicated()]
    seat = laps.dropna(subset=["Position"]).set_index(["RaceId", "LapNumber", "Position"])["Driver"]
    seat = seat[~seat.index.duplicated()]
    get = lambda col, driver, n: one[col].reindex(pd.MultiIndex.from_arrays([race, driver, n])).to_numpy()

    pairs = laps.loc[laps["IsStop"], ["RaceId", "Driver", "LapNumber", "Year", "Circuit"]].reset_index(drop=True)
    race, n, b = pairs["RaceId"].to_numpy(), pairs["LapNumber"].to_numpy(), pairs["Driver"].to_numpy()
    a = seat.reindex(pd.MultiIndex.from_arrays([race, n - 1, get("Position", b, n - 1) - 1])).to_numpy()
    answered = (pd.Series(get("IsStop", a, n + 1)).eq(True) & ~pd.Series(get("IsStop", a, n)).eq(True)).to_numpy()
    green = np.all([get("TrackStatus", car, n + k) == "1" for car in (a, b) for k in (-1, 0, 1, 2)], axis=0)
    dry = np.isin(get("Compound", a, n).astype(str), DRY) & np.isin(get("Compound", b, n).astype(str), DRY)
    before = get("Time", b, n - 1) - get("Time", a, n - 1)
    after = get("Time", b, n + 2) - get("Time", a, n + 2)
    pairs = pairs.assign(Ahead=a, Gap=before, After=after, Gain=before - after, Wear=get("TyreWear", a, n - 1))
    ok = answered & green & dry & (before < 10) & (np.abs(before - after) <= 10)     # (over 10 s: a slow stop)
    return pairs[ok].reset_index(drop=True)


def wear_bin(wear):
    return int(np.searchsorted(WEAR, wear)) if wear is not None and not math.isnan(wear) else None


def build():
    laps = tyre.replay_laps()
    laps = laps[features.in_study(laps)]
    cases = undercuts(laps)
    train = cases[cases["Year"] <= 2025]
    bins = train["Wear"].map(wear_bin)
    table = {"wear_bins": WEAR, "strike": STRIKE, "newest": NEWEST, "trained_on": "2022-2025", "cases": len(train),
             "gains": [sorted(round(float(g), 2) for g in train.loc[bins == k, "Gain"]) for k in range(len(WEAR) + 1)],
             "all": sorted(round(float(g), 2) for g in train["Gain"])}
    TABLE_FILE.write_text(json.dumps(table), encoding="utf-8")
    report(table, train, cases[cases["Year"] == 2026], laps)
    return table


def load():
    return json.loads(TABLE_FILE.read_text(encoding="utf-8")) if TABLE_FILE.exists() else build()


# Recipe 2: one pair now
def undercut(table, gap, wear):
    """If the car behind pits now and the car ahead pits next lap: the chance the car behind comes out ahead, and the
    usual gain (the middle half of the real undercuts on tyres worn about as much)."""
    k = wear_bin(wear)
    gains = np.array(table["gains"][k] if k is not None and table["gains"][k] else table["all"])   # (none on such tyres: all)
    return {"chance": round(float((gains > gap).mean()), 3), "gain": round(float(np.median(gains)), 1),
            "range": [round(float(np.quantile(gains, q)), 1) for q in (0.25, 0.75)], "cases": len(gains)}


def closing(back, front):
    """Seconds a lap the car behind gains on the car ahead (below 0: it's dropping back), from their pace over the
    last 5 green laps."""
    rate = (front["Pace"] - back["Pace"]) * back["UsualLap"]
    return None if pd.isna(rate) else float(rate)


def laps_to_reach(gap, rate, laps_left):
    """Laps until the gap is within striking distance: 0 = already, None = not before the end (or not closing)."""
    if gap <= STRIKE:
        return 0
    if rate is None or rate < 0.05:
        return None
    k = math.ceil((gap - STRIKE) / rate)
    return k if k <= laps_left else None


def pair(table, back, front, status, factor=1.0):
    """The car behind against the car ahead: the gap, the battle forecast and the undercut threat (None when it
    doesn't apply: not green, wet tyres, a car on an in-lap or out-lap, a set under 5 laps old, too far apart).
    None without both cars' times."""
    gap = float(back["Time"] - front["Time"])
    if math.isnan(gap):
        return None
    racing = status in ("GREEN", "YELLOW")
    rate = closing(back, front) if racing else None
    out = {"ahead": front["Driver"], "gap": round(gap, 1), "closing": None if rate is None else round(rate, 2),
           "laps": laps_to_reach(gap, rate, back["LapsLeft"]) if racing else None, "racing": racing, "undercut": None}
    pitting = any(pd.notna(car[c]) for car in (back, front) for c in ("PitInTime", "PitOutTime"))
    fresh = min(back["LapsInStint"], front["LapsInStint"]) * factor < NEWEST
    dry = str(back["Compound"]) in DRY and str(front["Compound"]) in DRY
    if status == "GREEN" and dry and not pitting and not fresh and back["LapsLeft"] >= 3 and gap < 10:
        out["undercut"] = undercut(table, gap, front["TyreWear"])
    return out


def lap_battles(table, lap, status, factor=1.0):
    """Every car against the car just ahead on this lap: {driver: pair(...)}. lap = the cars with a place on the
    lap, with the race outcome features. A car whose place just ahead is missing from the timing (97 laps of 6,562)
    has none: the car two places ahead isn't the one it's racing."""
    cars = lap[lap["Position"].notna() & lap["Time"].notna()].sort_values("Position")
    rows = [r for _, r in cars.iterrows()]
    return {back["Driver"]: pair(table, back, front, status, factor) for front, back in zip(rows, rows[1:])
            if back["Position"] - front["Position"] == 1}


# Recipe 3: the strategy battle - two drivers side by side, lap by lap up to `upto`: the gap between them, their
# places, tyres and stops, and now: who is ahead, the battle forecast and the undercut threat between them.
# rows = every lap of one race, with the race outcome features.
def stints(mine):
    """One driver's stints so far [{compound, from, to}] (mine: their laps, by lap number)."""
    return [{"compound": str(s["Compound"].iloc[0]), "from": int(s.index.min()), "to": int(s.index.max())}
            for _, s in mine.groupby("Stint", sort=True)]


def duel(table, rows, a, b, upto, status, factor=1.0):
    rows = rows[rows["Position"].notna() & (rows["LapNumber"] <= upto)]
    both = {d: rows[rows["Driver"] == d].set_index("LapNumber").sort_index() for d in (a, b)}
    if any(m.empty for m in both.values()):
        return None
    laps = []
    for n in sorted(set(both[a].index) | set(both[b].index)):
        side = {d: both[d].loc[n] if n in both[d].index else None for d in (a, b)}
        gap = side[b]["Time"] - side[a]["Time"] if side[a] is not None and side[b] is not None else math.nan
        laps.append({"lap": int(n), "gap": None if pd.isna(gap) else round(float(gap), 2),
                     **{k: None if side[d] is None else {"position": int(side[d]["Position"]), "compound": str(side[d]["Compound"]),
                                                         "tyre_age": None if pd.isna(side[d]["TyreLife"]) else int(side[d]["TyreLife"]),
                                                         "pitted": bool(side[d]["IsStop"])} for k, d in (("a", a), ("b", b))}})
    shared = sorted(set(both[a].index) & set(both[b].index))
    now = None
    if shared:
        n = shared[-1]
        first, second = sorted((both[a].loc[n], both[b].loc[n]), key=lambda r: r["Position"])
        now = pair(table, second, first, status, factor)
        if now:
            now = {"lap": int(n), "front": first["Driver"], "back": second["Driver"],
                   "between": int(abs(first["Position"] - second["Position"]) - 1), **now}
    return {"a": a, "b": b, "laps": laps, "now": now,
            "stints": {d: stints(both[d]) for d in (a, b)},
            "stops": {d: [int(n) for n in both[d].index[both[d]["IsStop"].astype(bool)]] for d in (a, b)}}


# Recipe 4: the test on the 17 test races of 2026
def forecast_test(laps):
    """For every car within 10 s of the car ahead under green: the forecast (laps to within 1 s) and what really
    happened in the next laps while both stayed out under green (unknown when a stop or a safety car came first).
    The simple way to compare with: the gap's change over the last 3 laps."""
    laps = laps[laps["Position"].notna()].sort_values(["RaceId", "LapNumber", "Position"])
    clean = (laps["TrackStatus"].astype(str) == "1") & laps["PitInTime"].isna() & laps["PitOutTime"].isna()
    laps = laps.assign(Clean=clean.astype(int), Ahead=laps.groupby(["RaceId", "LapNumber"])["Driver"].shift(1))
    out = []
    for race, r in laps.groupby("RaceId"):
        last = int(r["LapNumber"].max())
        names = sorted(r["Driver"].unique())
        col = {d: i for i, d in enumerate(names)}
        grid = np.arange(-3, last + AHEAD + 1)                              # lap numbers (with room both ways)
        T = r.pivot_table("Time", "LapNumber", "Driver").reindex(index=grid, columns=names).to_numpy()
        C = r.pivot_table("Clean", "LapNumber", "Driver", aggfunc="max").reindex(index=grid, columns=names)
        C = C.fillna(0).astype(bool).to_numpy()
        here = r[r["Ahead"].notna() & r["Clean"].astype(bool)]
        m = here["LapNumber"].to_numpy(int)
        bi, ai = here["Driver"].map(col).to_numpy(), here["Ahead"].map(col).to_numpy()
        row = lambda n: n - grid[0]
        gap = T[row(m), bi] - T[row(m), ai]
        steps = m[:, None] + np.arange(1, AHEAD + 1)
        future = T[row(steps), bi[:, None]] - T[row(steps), ai[:, None]]
        fine = C[row(steps), bi[:, None]] & C[row(steps), ai[:, None]]
        broke = ~fine & (steps <= last)                                     # a stop or a safety car (not the race's end)
        reached = (future <= STRIKE) & np.cumprod(fine, axis=1).astype(bool)
        before = T[row(m - 3), bi] - T[row(m - 3), ai]
        steady = C[row(m - 1), bi] & C[row(m - 1), ai] & C[row(m - 2), bi] & C[row(m - 2), ai]
        front = r.drop_duplicates(["LapNumber", "Driver"]).set_index(["LapNumber", "Driver"])["Pace"]
        front = front.reindex(pd.MultiIndex.from_arrays([m, here["Ahead"]])).to_numpy()
        out.append(pd.DataFrame({"RaceId": race, "Gap": gap, "Pace": (front - here["Pace"].to_numpy()) * here["UsualLap"].to_numpy(),
                                 "Trend": np.where(steady, (before - gap) / 3, np.nan), "LapsLeft": here["LapsLeft"].to_numpy(),
                                 "Reached": np.where(reached.any(axis=1), reached.argmax(axis=1) + 1, np.nan),
                                 "Broke": np.where(broke.any(axis=1), broke.argmax(axis=1) + 1, np.inf),
                                 "Ok": C[row(m), ai]}))
    d = pd.concat(out, ignore_index=True)
    return d[d["Ok"] & d["Gap"].between(STRIKE, 10, inclusive="neither") & d["Pace"].notna() & d["Trend"].notna()]


def forecast_scores(d, way, within=5):
    k = np.array([laps_to_reach(g, r, left) if pd.notna(r) else None for g, r, left in zip(d["Gap"], d[way], d["LapsLeft"])],
                 dtype=float)
    reached = d["Reached"].to_numpy()
    known = (reached <= within) | (d["Broke"].to_numpy() > within)       # (unknown: a stop or a safety car came first)
    said = known & (k <= within)
    real = known & (reached <= within)
    both = known & (k <= AHEAD) & d["Reached"].notna().to_numpy()
    error = np.abs(k[both] - d["Reached"].to_numpy()[both])
    return {"said": int(said.sum()), "right": float((said & real).sum() / max(said.sum(), 1)),
            "found": float((said & real).sum() / max(real.sum(), 1)), "real": int(real.sum()),
            "error": float(np.median(error)) if len(error) else math.nan,
            "mean_error": float(error.mean()) if len(error) else math.nan}


def report(table, train, test, laps):
    worked = (test["After"] < 0).to_numpy()
    ours = np.array([undercut(table, g, w)["chance"] for g, w in zip(test["Gap"], test["Wear"])])
    flat = np.array([(np.array(table["all"]) > g).mean() for g in test["Gap"]])
    rate = float((train["After"] < 0).mean())
    brier = lambda p: float(((p - worked) ** 2).mean())
    rows = ro.add_features(laps[laps["Year"] == 2026], ro.load_results())
    d = forecast_test(rows)
    lines = [
        "# Undercut threat and battle forecast: tested on the 2026 test races (rounds 1-17)", "",
        f"Undercut: learned from {table['cases']} real undercuts of 2022-2025 (the car behind pitted one lap before the "
        "car just ahead, both under green on dry tyres). Gains in seconds, by the car ahead's tyre wear:", "",
        "| Car ahead's tyre wear | Undercuts | Usual gain (middle half) |", "|---|---:|---|"]
    names = [f"under {WEAR[0]} s"] + [f"{lo}-{hi} s" for lo, hi in zip(WEAR, WEAR[1:])] + [f"over {WEAR[-1]} s"]
    for name, gains in zip(names, table["gains"]):
        g = np.array(gains)
        lines.append(f"| {name} | {len(g)} | {np.median(g):.1f} s ({np.quantile(g, .25):.1f} to {np.quantile(g, .75):.1f}) |")
    lines += ["", f"On the {len(test)} real undercuts of the 2026 test races (it worked {worked.sum()} times, "
              f"{worked.mean():.0%}):", "",
              "| Way | Brier score (lower is better) | Right at 50% |", "|---|---:|---:|",
              f"| The chance by tyre wear (the app) | {brier(ours):.3f} | {((ours >= 0.5) == worked).mean():.0%} |",
              f"| The chance without tyre wear | {brier(flat):.3f} | {((flat >= 0.5) == worked).mean():.0%} |",
              f"| How often undercuts worked in 2022-2025 ({rate:.0%}) | {brier(np.full(len(test), rate)):.3f} | "
              f"{((rate >= 0.5) == worked).mean():.0%} |", ""]
    pace, trend = forecast_scores(d, "Pace"), forecast_scores(d, "Trend")
    lines += [f"Battle forecast: {len(d):,} laps where a car was 1-10 s behind the car ahead under green. 'Within 5 laps' "
              "is checked only while both cars stayed out under green.", "",
              "| Way | Said within 1 s in 5 laps | It happened | Of those that happened, foreseen | Error in laps (median, mean) |",
              "|---|---:|---:|---:|---|"]
    for name, s in [("Pace over the last 5 green laps (the app)", pace), ("The gap's change over the last 3 laps", trend)]:
        lines.append(f"| {name} | {s['said']:,} | {s['right']:.0%} | {s['found']:.0%} of {s['real']:,} | "
                     f"{s['error']:.0f}, {s['mean_error']:.1f} |")
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    build()
