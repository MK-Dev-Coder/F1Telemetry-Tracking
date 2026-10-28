"""The race engineer's briefing for a track, from the real races there in 2022-2026:

- tyres: how many laps each compound lasts here (stints that ended in a stop under green) and how fast it wears
  (seconds lost a lap, for each lap on the set, with the fuel burn taken out), and how a hotter or cooler track
  changes that (learned from all tracks: on a hotter day than usual the medium and hard wear faster)
- strategies: the plans the finishers used here (soft-medium, medium-hard...), how often, when they stopped, how many
  places they gained, and what the winners did
- the track: the race length, what a stop costs (under green, the safety car and the VSC), how often there was a
  safety car or VSC, how often the pole sitter won, and the weather in past races
- in F1 25 / F1 26: the same for the game's shorter race (the game wears the tyres faster to match)

Each race is kept on its own, so a briefing can use only the races before one (a replay shows what an engineer
would have known then, with no spoilers). What it is: what really happened here, not a simulation. The app's own
view: the study's numbers don't use it.
    python engineer.py   -> models/engineer.json
"""
import json

import numpy as np
import pandas as pd

import pit_now
import race_outcome as ro
import train_tyre_model as tyre
from config import MODELS_DIR

TABLE_FILE = MODELS_DIR / "engineer.json"
DRY = ["SOFT", "MEDIUM", "HARD"]
WET = ["INTERMEDIATE", "WET"]
LETTER = {"SOFT": "S", "MEDIUM": "M", "HARD": "H", "INTERMEDIATE": "I", "WET": "W"}
COMPOUND = {v: k for k, v in LETTER.items()}
FUEL = 0.06                # seconds a lap the car gets quicker as the fuel burns, for each lap (a usual F1 figure)
FEWEST = 5                 # fewer stints than this at a track: no tyre life for that compound (its wear: all tracks)
MOST_CHANGE = 0.4          # a hotter or cooler day changes the tyre life by at most 40% either way


def quantiles(values, qs=(0.25, 0.5, 0.75)):
    v = pd.Series(values, dtype=float).dropna()
    return [round(float(v.quantile(q)), 1) for q in qs] if len(v) else [None] * len(qs)


# Recipe 1: every stint of every race; the ones that ended in a stop under green show how long the tyre lasted
def stints(laps):
    laps = laps.sort_values(["RaceId", "Driver", "LapNumber"])
    st = laps.groupby(["RaceId", "Driver", "Stint"], sort=False).agg(
        Circuit=("Circuit", "first"), Compound=("Compound", "first"), Start=("LapNumber", "min"), End=("LapNumber", "max"),
        Laps=("LapNumber", "size"), Stop=("IsStop", "last"), EndStatus=("TrackStatus", "last"))
    st = st.reset_index().sort_values(["RaceId", "Driver", "Start"])
    st["Compound"] = st["Compound"].astype(str)
    st["Green"] = st["Stop"].astype(bool) & (st["EndStatus"].astype(str) == "1") & (st["Laps"] >= 3)
    return st


# Recipe 2: how fast each stint's tyres wore: the lap time's rise for each lap on the set, on clean laps (green, not
# in or out of the pits, within 7% of the field), with the fuel burn added back
def wear(laps):
    clean = ((laps["TrackStatus"].astype(str) == "1") & laps["PitInTime"].isna() & laps["PitOutTime"].isna()
             & (laps["LapNumber"] > 1) & (laps["PaceVsField"] < 0.07) & laps["Compound"].astype(str).isin(DRY)
             & ~laps["Rainfall"].astype(bool))
    rows = []
    for (race, driver, stint), g in laps[clean].groupby(["RaceId", "Driver", "Stint"]):
        if len(g) >= 8:
            slope = np.polyfit(g["LapsInStint"], g["LapTime"], 1)[0] + FUEL
            rows.append((race, str(g["Circuit"].iloc[0]), str(g["Compound"].iloc[0]), slope, g["TrackTemp"].median()))
    w = pd.DataFrame(rows, columns=["RaceId", "Circuit", "Compound", "Wear", "Temp"])
    return w[w["Wear"].between(-0.3, 0.6)]                     # (a damaged car or a traffic jam)


def temperature_effect(w):
    """For each compound: how much faster it wears for each degree the track is hotter than usual there (s a lap,
    for each lap on the set), from all tracks, each against its own usual temperature and wear."""
    w = w.assign(dT=w["Temp"] - w.groupby("Circuit")["Temp"].transform("median"))
    out = {}
    for compound in DRY:
        x = w[(w["Compound"] == compound) & w["dT"].notna()]
        y = x["Wear"] - x.groupby("Circuit")["Wear"].transform("mean")
        out[compound] = round(float(np.polyfit(x["dT"], y, 1)[0]), 4) if len(x) > 30 else 0.0
    return out


# Recipe 3: each finisher's plan (the compounds in order, the laps they stopped, places gained)
def finishers(st, results, laps):
    total = laps.groupby("RaceId")["TotalLaps"].first()
    done = st.groupby(["RaceId", "Driver"]).agg(Plan=("Compound", lambda c: "-".join(LETTER.get(x, "?") for x in c)),
                                                 Laps=("End", "max"), StopLaps=("End", lambda e: [int(x) for x in e.iloc[:-1]]))
    done = done.reset_index().merge(results[["RaceId", "Driver", "Grid", "Final"]], on=["RaceId", "Driver"], how="left")
    done = done[done["Final"].notna() & (done["Laps"] >= 0.9 * done["RaceId"].map(total))]
    done["Gained"] = done["Grid"].where(done["Grid"] > 0) - done["Final"]           # (grid 0: from the pit lane)
    return done


def build():
    laps = tyre.replay_laps()
    results = ro.load_results()
    info = ro.load_model()[1]
    st, w = stints(laps), wear(laps)
    fin = finishers(st, results, laps)
    won = results[results["Final"] == 1].set_index("RaceId")
    tracks = {}
    for circuit, rows in laps.groupby(laps["Circuit"].astype(str)):
        races = sorted(rows["RaceId"].unique())
        race_info = {}
        for race in races:
            r = rows[rows["RaceId"] == race]
            status = "".join(r["TrackStatus"].astype(str))
            mine = st[st["RaceId"] == race]
            winner = won["Driver"].get(race)
            ws = mine[mine["Driver"] == winner] if winner is not None else mine.iloc[:0]
            race_info[race] = {"laps": int(r["TotalLaps"].iloc[0]), "track": round(float(r["TrackTemp"].median()), 1),
                               "air": round(float(r["AirTemp"].median()), 1), "rain": bool(r["Rainfall"].max()),
                               "wet": bool(mine["Compound"].isin(WET).any()), "sc": "4" in status, "vsc": any(c in status for c in "67"),
                               "winner": winner, "winner_grid": None if winner is None or pd.isna(won["Grid"].get(race)) else int(won["Grid"].get(race)),
                               "winner_stints": [{"compound": c, "from": int(a), "to": int(b)} for c, a, b in zip(ws["Compound"], ws["Start"], ws["End"])]}
        g, ww, f = st[st["Green"] & (st["Circuit"].astype(str) == circuit)], w[w["Circuit"] == circuit], fin[fin["RaceId"].isin(races)]
        tracks[circuit] = {
            "races": races, "race_info": race_info,
            "pit_loss": round(float(info["pit_loss"]["track"].get(circuit, info["pit_loss"]["overall"])), 1),
            "stints": [[r, LETTER.get(c, "?"), int(n)] for r, c, n in zip(g["RaceId"], g["Compound"], g["Laps"])],
            "wear": [[r, LETTER.get(c, "?"), round(float(x), 4)] for r, c, x in zip(ww["RaceId"], ww["Compound"], ww["Wear"])],
            "finishers": [[r, p, s, None if pd.isna(gn) else int(gn), int(fl)] for r, p, s, gn, fl in
                          zip(f["RaceId"], f["Plan"], f["StopLaps"], f["Gained"], f["Final"])]}
    wet = st[st["Compound"].isin(WET) & st["Stop"].astype(bool) & (st["Laps"] >= 3)]
    table = {"tracks": tracks, "temperature_effect": temperature_effect(w), "fuel": FUEL,
             "pit_share": {k: v[1] for k, v in pit_now.load()["share"].items()},       # (a stop under the SC or VSC)
             "all_wear": {c: round(float(w.loc[w["Compound"] == c, "Wear"].median()), 3) for c in DRY},
             "wet_stints": [[r, LETTER[c], int(n)] for r, c, n in zip(wet["RaceId"], wet["Compound"], wet["Laps"])],
             "wet_races": sorted(laps.loc[laps["Compound"].astype(str).isin(WET), "RaceId"].unique())}
    TABLE_FILE.write_text(json.dumps(table), encoding="utf-8")
    return table


def load():
    return json.loads(TABLE_FILE.read_text(encoding="utf-8")) if TABLE_FILE.exists() else build()


# Recipe 4: the briefing for one track, from the races there (only the ones before `before`, a race id, if given),
# on a day as hot as `track_temp` (the usual one if not given), in the rain or not, and for a race of `distance` %
# or of `game_laps` laps (the game's shorter races wear the tyres faster to match, so they last as many laps less)
def brief(table, circuit, track_temp=None, rain=False, distance=100, game_laps=None, before=None):
    t = table["tracks"].get(circuit)
    if t is None:
        return None
    races = [r for r in t["races"] if before is None or r < before]
    if not races:
        return {"circuit": circuit, "races": [], "empty": True, "rain": bool(rain)}
    info = [t["race_info"][r] for r in races]
    keep = set(races)
    laps = int(np.median([i["laps"] for i in info]))
    temps = [i["track"] for i in info]
    usual = round(float(np.median(temps)), 1)
    temp = usual if track_temp is None else round(float(track_temp), 1)
    if game_laps is None and distance < 100:
        game_laps = max(1, round(laps * distance / 100))
    game_laps = game_laps if game_laps is not None and game_laps < laps else None
    factor = laps / game_laps if game_laps else 1.0
    scale = lambda v: None if v is None else round(v / factor, 1)
    tyres = {}
    for c in DRY:
        n = [x[2] for x in t["stints"] if x[0] in keep and COMPOUND[x[1]] == c]
        worn = [x[2] for x in t["wear"] if x[0] in keep and COMPOUND[x[1]] == c]
        life = quantiles(n) if len(n) >= FEWEST else [None] * 3
        base = float(np.median(worn)) if len(worn) >= FEWEST else table["all_wear"][c]
        hot = base + table["temperature_effect"][c] * (temp - usual)              # the wear on this day
        hot = max(hot, 0.3 * base) if base > 0 else hot
        change = 1.0 if base <= 0 else min(1 + MOST_CHANGE, max(1 - MOST_CHANGE, base / hot))
        life_today = None if life[1] is None else life[1] * change
        tyres[c] = {"life": life[1], "range": [life[0], life[2]], "stints": len(n), "wear": round(base, 3), "wear_here": len(worn) >= FEWEST,
                    "wear_today": round(hot, 3), "change": round(change - 1, 3),
                    "life_today": None if life_today is None else round(life_today, 1),
                    "range_today": [None if v is None else round(v * change, 1) for v in (life[0], life[2])],
                    "game": None if factor == 1 else {"life": scale(life_today), "range": [scale(None if v is None else v * change) for v in (life[0], life[2])]}}
    dry_races = {r for r, i in zip(races, info) if not i["wet"]}
    fin = [f for f in t["finishers"] if f[0] in dry_races]
    plans = []
    for plan in sorted({f[1] for f in fin}):
        g = [f for f in fin if f[1] == plan]
        gained = [f[3] for f in g if f[3] is not None]
        plans.append({"plan": plan, "drivers": len(g), "share": round(len(g) / len(fin), 3),
                      "stop_laps": [int(np.median([f[2][k] for f in g if len(f[2]) > k])) for k in range(plan.count("-"))],
                      "gained": round(float(np.mean(gained)), 1) if gained else None,
                      "top5": sum(f[4] <= 5 for f in g), "wins": sum(f[4] == 1 for f in g)})
    plans.sort(key=lambda p: -p["drivers"])
    moved = [abs(f[3]) for f in t["finishers"] if f[0] in keep and f[3] is not None]
    winners = [{"race_id": r, "year": int(r[:4]), "driver": i["winner"], "grid": i["winner_grid"], "wet": i["wet"],
                "plan": "-".join(LETTER.get(s["compound"], "?") for s in i["winner_stints"]), "stints": i["winner_stints"]}
               for r, i in zip(races, info) if i["winner"]]
    wet_keep = [w for w in table["wet_stints"] if before is None or w[0] < before]
    wet = None
    if rain:
        wet = {COMPOUND[c]: {"life": q[1], "range": [q[0], q[2]], "stints": len(n), "game": None if factor == 1 else
                             {"life": scale(q[1]), "range": [scale(q[0]), scale(q[2])]}}
               for c in ("I", "W") for n in [[w[2] for w in wet_keep if w[1] == c]] for q in [quantiles(n)]}
        wet.update(races=len({w for w in table["wet_races"] if before is None or w < before}))
    return {"circuit": circuit, "races": races, "laps": laps, "game_laps": game_laps, "distance": round(100 / factor),
            "track_temp": temp, "usual_temp": usual, "rain": bool(rain),
            "weather": {"track": usual, "air": round(float(np.median([i["air"] for i in info])), 1),
                        "coolest": min(temps), "hottest": max(temps), "rain_races": sum(i["rain"] for i in info)},
            "pit_loss": {"green": t["pit_loss"], "sc": round(t["pit_loss"] * table["pit_share"]["SC"], 1),
                         "vsc": round(t["pit_loss"] * table["pit_share"]["VSC"], 1)},
            "safety_car": {"sc": sum(i["sc"] for i in info), "vsc": sum(i["vsc"] for i in info),
                           "either": sum(i["sc"] or i["vsc"] for i in info), "of": len(info)},
            "history": [{"race_id": r, "year": int(r[:4]), "sc": i["sc"], "vsc": i["vsc"], "rain": i["rain"], "wet": i["wet"],
                         "track": i["track"]} for r, i in zip(races, info)],     # each race here: safety car, rain
            "pole_won": sum(i["winner_grid"] == 1 for i in info), "places_moved": round(float(np.mean(moved)), 1) if moved else None,
            "tyres": tyres, "plans": plans[:6], "winners": winners, "wet": wet,
            "temperature_effect": table["temperature_effect"], "fuel": table["fuel"]}


if __name__ == "__main__":
    tb = build()
    print("tracks:", len(tb["tracks"]), " temperature effect:", tb["temperature_effect"])
    m = brief(tb, "Mexico City")
    print({c: (x["life"], x["range"], x["wear"]) for c, x in m["tyres"].items()}, m["plans"][:2], m["safety_car"])
    print("before 2025_20:", brief(tb, "Mexico City", before="2025_20")["races"])
