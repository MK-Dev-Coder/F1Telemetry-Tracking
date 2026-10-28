"""Race preview: before a race starts, each driver's chance to win, finish on the podium and score points.

One row per driver per race, with only what is known before the start: their recent form and their team's, race
pace from the lap data of the last races (finishing places hide bad luck, lap times don't), how often the team's
cars retire, how they did at this track before, and - once qualifying is done - the grid. The same kind of model
as the race outcome model: "will they finish in the top k?" for every k, with the chances fitted together so the
field has one winner, three podium places and ten points places.

It is tested the way it would be used: before each race of 2025 and 2026 it is trained again on the races before
it, then asked about that race (the settings are picked the same way on 2024). Then the next race is predicted.
    python race_preview.py   -> data/preview_backtest.parquet, data/preview_evaluation.md, figures/preview_*.png,
                                models/preview_next.json (the next race, with the reasons)
"""
import json
import time
import warnings

import fastf1
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import shap

import collect_openf1
import features
import race_outcome as ro
import train_tyre_model as tyre
from config import CACHE_DIR, DATA_DIR, FIGURES_DIR, MODELS_DIR
from evaluate import save, table

matplotlib.use("Agg")                         # save charts to files, don't open windows
import matplotlib.pyplot as plt               # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
RNG = np.random.default_rng(42)
BACKTEST_FILE, NEXT_FILE = DATA_DIR / "preview_backtest.parquet", MODELS_DIR / "preview_next.json"
REPORT = DATA_DIR / "preview_evaluation.md"
MAX_K = 20
POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
# Each place-like feature is given against the top-k line (as in the race outcome model): below 0 = inside it
K_FEATURES = {"FormToK": "DriverForm5", "LastToK": "LastFinish", "SeasonToK": "SeasonAvg", "TeamFormToK": "TeamForm3",
              "TrackToK": "TrackDriver", "GridToK": "Grid"}
BEFORE = ["FormToK", "LastToK", "SeasonToK", "SeasonWins", "SeasonPodiums", "TeamFormToK", "Pace3", "TeamPace3",
          "TeamDNF5", "TrackToK"]
# After qualifying the grid says the most; with every feature it gets watered down, so a few sets are tried on 2024
AFTER_SETS = {"grid and everything": BEFORE + ["GridToK"], "grid, pace and wins": ["GridToK", "Pace3", "TeamPace3", "SeasonWins"],
              "grid, form and pace": ["GridToK", "FormToK", "TeamFormToK", "Pace3", "TeamPace3", "SeasonWins", "TeamDNF5"],
              "grid only": ["GridToK"]}
VERSIONS = {"before qualifying": BEFORE, "after qualifying": AFTER_SETS["grid and everything"]}     # (set in main)
LGBM = dict(objective="binary", learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=42, verbose=-1)
TRIES = [dict(num_leaves=7, min_child_samples=400, n_estimators=150), dict(num_leaves=7, min_child_samples=400, n_estimators=300),
         dict(num_leaves=15, min_child_samples=800, n_estimators=200)]


# Recipe 1: one row per driver per race - the result, and what the lap data says about that race
def driver_races():
    laps = tyre.replay_laps()
    clean = ((laps["TrackStatus"] == "1") & (laps["LapNumber"] > 1) & laps["PitOutTime"].isna()
             & laps["PitInTime"].isna() & (laps["PaceVsField"] < 0.15))
    per = laps.groupby(["RaceId", "Driver"]).agg(Team=("Team", "first"), Circuit=("Circuit", "first"),
                                                 EventName=("EventName", "first"), Year=("Year", "first"),
                                                 Round=("Round", "first"), LapsDone=("LapNumber", "max"))
    per["Pace"] = laps[clean].groupby(["RaceId", "Driver"])["PaceVsField"].median()
    per["Circuit"] = per["Circuit"].astype(str)
    rows = per.reset_index().merge(ro.load_results()[["RaceId", "Driver", "Final", "Grid"]], on=["RaceId", "Driver"])
    rows["Retired"] = (rows["LapsDone"] < 0.9 * rows.groupby("RaceId")["LapsDone"].transform("max")).astype(int)
    return rows.sort_values(["RaceId", "Final"]).reset_index(drop=True)


# Recipe 2: what was known before each race, from the races before it only
def add_history(rows):
    rows = rows.sort_values(["RaceId", "Driver"]).reset_index(drop=True)
    by_driver = rows.groupby("Driver")
    before = lambda s, n: s.shift(1).rolling(n, min_periods=1).mean()
    rows["DriverForm5"] = by_driver["Final"].transform(lambda s: before(s, 5))
    rows["LastFinish"] = by_driver["Final"].shift(1)
    rows["Pace3"] = by_driver["Pace"].transform(lambda s: before(s, 3))          # below 0 = quicker than the field
    season = rows.groupby(["Year", "Driver"])
    rows["SeasonAvg"] = season["Final"].transform(lambda s: s.shift(1).expanding().mean())
    rows["SeasonWins"] = season["Final"].transform(lambda s: (s == 1).astype(float).shift(1).fillna(0).cumsum())
    rows["SeasonPodiums"] = season["Final"].transform(lambda s: (s <= 3).astype(float).shift(1).fillna(0).cumsum())
    # The team: its cars' average place, race pace and retirements, race by race
    team = rows.groupby(["Team", "RaceId"]).agg(Final=("Final", "mean"), Pace=("Pace", "mean"), Retired=("Retired", "mean"))
    by_team = team.groupby(level=0)
    team["TeamForm3"] = by_team["Final"].transform(lambda s: before(s, 3))
    team["TeamPace3"] = by_team["Pace"].transform(lambda s: before(s, 3))
    team["TeamDNF5"] = by_team["Retired"].transform(lambda s: before(s, 5))
    rows = rows.join(team[["TeamForm3", "TeamPace3", "TeamDNF5"]], on=["Team", "RaceId"])
    # This track before (earlier years)
    rows["TrackDriver"] = rows.groupby(["Driver", "Circuit"])["Final"].transform(lambda s: s.shift(1).expanding().mean())
    return rows


def stack(rows, feats, ks=range(1, MAX_K + 1)):
    ks = list(ks)
    X = rows[[f for f in feats if f not in K_FEATURES]].loc[rows.index.repeat(len(ks))].reset_index(drop=True)
    X["K"] = np.tile(ks, len(rows)).astype(float)
    for name, base in K_FEATURES.items():
        if name in feats:
            X[name] = (rows[base].values[:, None] - np.array(ks)[None, :]).ravel()
    return X[["K"] + feats]


def answers(rows, ks=range(1, MAX_K + 1)):
    return (rows["Final"].values[:, None] <= np.array(list(ks))[None, :]).astype(int).ravel()


def fit(rows, feats, tries):
    model = lgb.LGBMClassifier(**LGBM, **tries, monotone_constraints=[1] + [0] * len(feats))
    return model.fit(stack(rows, feats), answers(rows))


def chances(model, feats, race):
    """Every driver's chance of each top k in one race, fitted together (one winner, k cars in the top k)."""
    n = len(race)
    raw = model.predict_proba(stack(race, feats, range(1, n + 1)))[:, 1].reshape(n, n)
    return ro.fit_together(raw)


def summary(p):
    s = ro.summary(np.hstack([p, np.ones((len(p), max(0, ro.MAX_CARS - p.shape[1])))]))
    return s


# Recipe 3: two simple baselines - how often drivers with the same recent form (or grid slot) finished in the top k
def table_chances(train, race, key):
    def keys(rows):
        return rows[key].round().clip(1, 20).fillna(0).astype(int).astype(str)
    top = train["Final"].values[:, None] <= np.arange(1, MAX_K + 1)[None, :]
    groups = pd.DataFrame(top).groupby(keys(train).values)
    prior = np.arange(1, MAX_K + 1) / 20
    tab = (groups.sum() + 5 * prior) / (groups.size().values[:, None] + 5)
    raw = tab.reindex(keys(race).values).fillna(pd.Series(prior)).values
    n = len(race)
    raw = np.hstack([raw, np.ones((n, max(0, n - MAX_K)))])[:, :n]
    return ro.fit_together(raw)


def backtest(rows, races, tries, versions=VERSIONS):
    """Before each race: train on the races before it, predict it. One row per driver per race and way."""
    out = []
    for race_id in races:
        train, race = rows[rows["RaceId"] < race_id], rows[rows["RaceId"] == race_id].reset_index(drop=True)
        frame = race[["RaceId", "Year", "Round", "EventName", "Circuit", "Driver", "Team", "Grid", "Final"]].copy()
        for name, feats in versions.items():
            p = chances(fit(train, feats, tries), feats, race)
            for t, k in ro.SHOWN.items():
                frame[f"{name}|{t}"] = p[:, k - 1]
            frame[f"{name}|likely"] = summary(p)["likely"]
        for name, key in [("form table", "DriverForm5"), ("grid table", "Grid")]:
            p = table_chances(train, race, key)
            for t, k in ro.SHOWN.items():
                frame[f"{name}|{t}"] = p[:, k - 1]
        # The favourite of two simple rules: the last race's winner, and the season's points leader so far
        last = train[train["RaceId"] == train["RaceId"].max()]
        season = train[train["Year"] == race["Year"].iloc[0]]
        points = season.assign(P=season["Final"].map(POINTS).fillna(0)).groupby("Driver")["P"].sum()
        frame["last winner"] = frame["Driver"] == (last.loc[last["Final"] == 1, "Driver"].iloc[0] if (last["Final"] == 1).any() else None)
        frame["points leader"] = frame["Driver"] == (points.idxmax() if len(points) else None)
        out.append(frame)
    return pd.concat(out, ignore_index=True)


def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def scores(bt, way):
    out = {}
    for t, k in ro.SHOWN.items():
        y = (bt["Final"] <= k).astype(int).values
        out[f"{t.title()} log loss"] = logloss(y, bt[f"{way}|{t}"].values).mean()
    fav = bt.loc[bt.groupby("RaceId")[f"{way}|win"].idxmax()]
    out["Favourite won (%)"] = 100 * (fav["Final"] == 1).mean()
    out["Winner's chance (median)"] = bt.loc[bt["Final"] == 1, f"{way}|win"].median()
    return out


# Recipe 4: the next race - the drivers of the latest race, everything known now, and why (SHAP against an average
# driver in the training races, asked the same question)
def next_race(rows):
    """The next race from OpenF1 (this season's, or next season's first once this one is over), with its round as
    collect_openf1.py numbers it (FastF1's schedule). None without internet."""
    latest = rows[rows["RaceId"] == rows["RaceId"].max()]
    year, last_round = int(latest["Year"].iloc[0]), int(latest["Round"].iloc[0])
    try:
        for season in (year, year + 1):
            sessions = collect_openf1.get("sessions", year=season, session_name="Race")
            coming = sorted((s for s in sessions if not s.get("is_cancelled")
                             and pd.Timestamp(s["date_start"]) > pd.Timestamp.now(tz="UTC")), key=lambda s: s["date_start"])
            if coming:
                break
        else:
            return None
        s = coming[0]
        meeting = collect_openf1.get("meetings", meeting_key=s["meeting_key"])
        name = meeting[0]["meeting_name"] if meeting else f"{s['country_name']} Grand Prix"
        try:
            fastf1.Cache.enable_cache(CACHE_DIR)
            event = collect_openf1.rounds_by_date(season).get(pd.Timestamp(s["date_start"]).tz_convert("UTC").date())
        except Exception as err:                                                    # (FastF1 can't be reached)
            print("   no FastF1 schedule:", err)
            event = None
        rnd = int(event["RoundNumber"]) if event is not None else last_round + 1 if season == year else 1
        if season == year and rnd > last_round + 1:
            which = f"round {rnd - 1}" if rnd - 1 == last_round + 1 else f"rounds {last_round + 1}-{rnd - 1}"
            print(f"   {which} of {year} isn't saved yet: add it first (collect_openf1.py {year} {rnd - 1}, see the README)")
        return {"name": name, "date": s["date_start"][:10], "circuit": s["circuit_short_name"], "year": season, "round": rnd}
    except Exception as err:                                                        # no internet: no next race
        print("   no next race from OpenF1:", err)
        return None


def preview_next(rows, tries, info):
    feats = VERSIONS["before qualifying"]
    latest = rows[rows["RaceId"] == rows["RaceId"].max()]
    race_id = f"{info['year']}_{info['round']:02d}"
    coming = latest[["Driver", "Team"]].assign(RaceId=race_id, Year=info["year"], Round=info["round"], Circuit=info["circuit"],
                                               EventName=info["name"], Final=np.nan, Grid=np.nan, Pace=np.nan, Retired=np.nan)
    both = add_history(pd.concat([rows[driver_races_columns()], coming], ignore_index=True))
    train, race = both[both["RaceId"] < race_id], both[both["RaceId"] == race_id].reset_index(drop=True)
    model = fit(train, feats, tries)
    p = chances(model, feats, race)
    s = summary(p)
    out = []
    background = {k: stack(train.sample(400, random_state=42), feats, [k]).sample(100, random_state=42) for k in ro.SHOWN.values()}
    why = {t: shap.TreeExplainer(model.booster_, data=background[k], feature_perturbation="interventional",
                                 model_output="probability") for t, k in ro.SHOWN.items()}
    values = {t: why[t].shap_values(stack(race, feats, [k])) for t, k in ro.SHOWN.items()}
    for i, r in race.iterrows():
        chances_ = {t: round(float(p[i, k - 1]), 4) for t, k in ro.SHOWN.items()}
        target = next((t for t in ro.SHOWN if chances_[t] >= 0.1), "points")
        entry = {"driver": r["Driver"], "team": r["Team"], **chances_, "likely": int(s["likely"][i]), "for": target, "why": {}}
        for t, k in ro.SHOWN.items():
            v = pd.Series(values[t][i, 1:], index=feats)
            start = float(why[t].expected_value)
            order = v.abs().sort_values(ascending=False).index[:8]
            entry["why"][t] = {"chance": chances_[t], "start": round(start, 4), "model": round(start + float(v.sum()), 4),
                               "reasons": [{"text": describe(n, r, k), "points": round(float(v[n]), 4)} for n in order]}
        out.append(entry)
    out.sort(key=lambda e: (-e["win"], -e["podium"], -e["points"]))
    return {**info, "race_id": race_id, "version": "before qualifying", "trained_on": f"2022 - {rows['RaceId'].max()}",
            "made": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"), "drivers": out}


def driver_races_columns():
    return ["RaceId", "Year", "Round", "EventName", "Circuit", "Driver", "Team", "Final", "Grid", "Pace", "Retired"]


def describe(name, row, k):
    """A preview reason in plain English."""
    base = K_FEATURES.get(name, name)
    v = row[base]
    texts = {
        "DriverForm5": lambda v: f"averaged P{v:.1f} in the last 5 races",
        "LastFinish": lambda v: f"finished P{v:.0f} last time",
        "SeasonAvg": lambda v: f"averaged P{v:.1f} this season",
        "SeasonWins": lambda v: "no win this season yet" if v == 0 else f"{v:.0f} win{'s' if v > 1 else ''} this season",
        "SeasonPodiums": lambda v: "no podium this season yet" if v == 0 else f"{v:.0f} podium{'s' if v > 1 else ''} this season",
        "TeamForm3": lambda v: f"team averaged P{v:.1f} in the last 3 races",
        "Pace3": lambda v: f"race pace {abs(v):.1%} {'quicker' if v < 0 else 'slower'} than the field in the last 3 races",
        "TeamPace3": lambda v: f"team's race pace {abs(v):.1%} {'quicker' if v < 0 else 'slower'} than the field lately",
        "TeamDNF5": lambda v: "no retirements for the team lately" if v == 0 else f"team retired {v:.0%} of its cars in the last 5 races",
        "TrackDriver": lambda v: f"averaged P{v:.1f} here before",
        "Grid": lambda v: f"starts P{v:.0f}",
    }
    unknown = {"DriverForm5": "no recent races", "LastFinish": "no race last time", "SeasonAvg": "first race of the season",
               "TeamForm3": "no recent team races", "Pace3": "no recent race pace", "TeamPace3": "no recent team pace",
               "TeamDNF5": "no recent team races", "TrackDriver": "first time at this track", "Grid": "grid not known yet"}
    if pd.isna(v):
        return unknown.get(base, f"{base} not known")
    return texts.get(base, lambda v: f"{base} = {v}")(float(v))


# Recipe 5: the report and the charts
def report(bt, tune, chosen, tries):
    # The study's races only (2026 up to round 17), so the numbers stay the same when races are added; the app
    # shows the races after it too (the backtest file has them all)
    test = bt[(bt["Year"] >= 2025) & features.in_study(bt)]
    ways = ["before qualifying", "after qualifying", "form table", "grid table"]
    res = pd.DataFrame({w: scores(test, w) for w in ways}).T.rename_axis("Way")
    for rule in ["last winner", "points leader"]:
        fav = test[test[rule]]
        res.loc[f"{rule} (rule)", "Favourite won (%)"] = 100 * (fav["Final"] == 1).sum() / test["RaceId"].nunique()
    # 95% interval for the win log loss (whole races resampled)
    races = test["RaceId"].unique()
    by_race = {r: np.flatnonzero(test["RaceId"].values == r) for r in races}
    for w in ways:
        loss = logloss((test["Final"] == 1).astype(int).values, test[f"{w}|win"].values)
        draws = [loss[np.concatenate([by_race[r] for r in RNG.choice(races, len(races))])].mean() for _ in range(1000)]
        res.loc[w, "Win log loss 95% CI"] = f"{np.percentile(draws, 2.5):.3f}-{np.percentile(draws, 97.5):.3f}"
    # Chart: the real winner's chance before each race, before and after qualifying
    won = test[test["Final"] == 1].sort_values("RaceId")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(won))
    ax.bar(x - 0.2, won["before qualifying|win"], width=0.4, label="Before qualifying", color="tab:blue")
    ax.bar(x + 0.2, won["after qualifying|win"], width=0.4, label="After qualifying", color="tab:orange")
    ax.set_xticks(x, [f"{r[2:4]} R{r[5:]} {d}" for r, d in zip(won["RaceId"], won["Driver"])], rotation=90, fontsize=7)
    ax.set_ylabel("Win chance given to the real winner")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(f"Race preview: the winner's chance before each race of 2025 and 2026 (up to round {features.TEST_LAST_ROUND})")
    ax.legend()
    chart = save(fig, "preview_backtest.png")
    md = ["# Race preview: tested race by race (2025 and 2026)", "",
          f"Before each of the {test['RaceId'].nunique()} races (2025, and 2026 up to round {features.TEST_LAST_ROUND} as in "
          "the rest of the study) the model was trained again on every race before it "
          f"(from 2022), then asked about that race. Settings picked the same way on 2024: {tries}. "
          "*Form table* / *grid table*: how often drivers with the same recent average finish / grid slot finished in "
          "the top 1/3/10 before. *Rules*: the last race's winner wins; the season's points leader wins. "
          f"After qualifying it uses: {', '.join(VERSIONS['after qualifying'])} (picked on 2024).", "",
          table(res, 3, per={"Favourite won (%)": 1}), "", f"![Backtest]({chart})", "",
          "## Settings and after-qualifying features, tried race by race on 2024", "", table(tune, 4), "",
          "Written by `python race_preview.py`."]
    REPORT.write_text("\n".join(md), encoding="utf-8")
    return res


def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    start = time.time()
    rows = add_history(driver_races())
    races = sorted(rows["RaceId"].unique())
    print(f"{len(rows)} driver-races in {len(races)} races")
    # Settings: picked by testing race by race on 2024 (before qualifying, the harder one)
    tune = {}
    for k, tries in enumerate(TRIES):
        bt = backtest(rows, [r for r in races if r.startswith("2024")], tries, {"before qualifying": BEFORE})
        tune[str(tries)] = scores(bt, "before qualifying")
        print(f"   2024 with {tries}: win log loss {tune[str(tries)]['Win log loss']:.4f}")
    tune = pd.DataFrame(tune).T.rename_axis("Settings")
    chosen = tune["Win log loss"].idxmin()
    tries = TRIES[[str(t) for t in TRIES].index(chosen)]
    sets = {}
    for name, feats in AFTER_SETS.items():                                     # the after-qualifying features, on 2024
        bt = backtest(rows, [r for r in races if r.startswith("2024")], tries, {name: feats})
        sets[name] = scores(bt, name)
    sets = pd.DataFrame(sets).T.rename_axis("After qualifying")
    VERSIONS["after qualifying"] = AFTER_SETS[sets["Win log loss"].idxmin()]
    print(f"   after qualifying, picked on 2024: {sets['Win log loss'].idxmin()}")
    tune = pd.concat([tune.reset_index().rename(columns={"Settings": "Tried on 2024"}),
                      sets.reset_index().rename(columns={"After qualifying": "Tried on 2024"})], ignore_index=True).set_index("Tried on 2024")
    bt = backtest(rows, [r for r in races if r >= "2025"], tries)
    bt.to_parquet(BACKTEST_FILE, index=False)
    res = report(bt, tune, chosen, tries)
    print(res.round(3).to_string())

    info = next_race(rows)
    if info:
        nxt = preview_next(rows, tries, info)
        NEXT_FILE.write_text(json.dumps(nxt, indent=1), encoding="utf-8")
        print(f"\nNext race: {nxt['name']} ({nxt['date']}), before qualifying:")
        for d in nxt["drivers"][:6]:
            print(f"   {d['driver']}: win {d['win']:.0%}, podium {d['podium']:.0%}, points {d['points']:.0%}, likely P{d['likely']}"
                  f" - {d['why'][d['for']]['reasons'][0]['text']}")
    print(f"saved {BACKTEST_FILE}, {REPORT}, figures/preview_backtest.png" + (f", {NEXT_FILE}" if info else "")
          + f" ({time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
