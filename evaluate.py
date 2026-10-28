"""Evaluation for the dissertation: every number and chart for the results chapter, in one run.

RQ1  Can the model predict pit stops?  Where it works and where it fails, can its % be trusted, how early it warns.
RQ2  Are the SHAP reasons sound?       Do they follow F1 logic, match real stops, stay steady, really matter.
RQ3  Does it work live?                How long one lap takes: in replay, live on F1 25, and reading the game's packets.

Everything is measured on the 17 races of 2026, which were never used to train or tune the model.
    python evaluate.py        -> figures/eval_*.png and data/evaluation.md (tables ready for the write-up)
"""
import json
import tempfile
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

import features
from config import DATA_DIR, FIGURES_DIR, MODELS_DIR
from explain_pit_model import explain, load_model
from train_pit_model import real_stops, stops_warned

matplotlib.use("Agg")                         # save charts to files, don't open windows
import matplotlib.pyplot as plt               # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
RNG = np.random.default_rng(42)
BOOT = 1000                                   # bootstrap resamples (whole races are drawn, as races aren't independent laps)
REPORT = DATA_DIR / "evaluation.md"
MODELS = ["LightGBM", "Logistic regression", "Tyre age rule"]
KEYS = ["RaceId", "Driver", "LapNumber"]
NEUTRAL = ["SafetyCar", "VSC", "NeutralStarted", "LapsUnderNeutral", "Yellow"]
THEMES = {"Tyres": ["TyreLife", "LapsInStint", "TyreWear", "Compound", "Stint", "FreshTyre", "DryCompoundsUsed"],
          "Other cars' stops": ["FieldStopsLast2", "AheadStoppedRecently", "BehindStoppedRecently"],
          "Safety car / VSC": NEUTRAL, "Race progress": ["LapNumber", "LapsLeft", "RaceProgress"],
          "Gaps and position": ["Position", "GapAhead", "GapBehind", "GapToLeader"], "Track": ["Circuit"],
          "Weather": ["AirTemp", "TrackTemp", "Humidity", "Rainfall", "FieldOnWetTyres"], "Pace": ["PaceVsField"]}
THEME_OF = {f: theme for theme, names in THEMES.items() for f in names}
# What F1 logic says a higher value should do to the pit stop chance (only where there is one clear answer:
# e.g. both very early and very late in a race a stop is less likely, so race progress has no single direction)
EXPECTED = {"TyreLife": "more", "LapsInStint": "more", "TyreWear": "more", "SafetyCar": "more", "VSC": "more",
            "NeutralStarted": "more", "FieldStopsLast2": "more", "AheadStoppedRecently": "more",
            "BehindStoppedRecently": "more", "DryCompoundsUsed": "less", "Rainfall": "more", "GapBehind": "more"}


COUNTS = {"Laps", "Stops", "Alerts", "Times"}                     # columns shown as whole numbers


def table(df, digits=3, per=None):
    """A table in Markdown, for the write-up (per: digits for some columns, e.g. {"Share (%)": 1})."""
    df, per = df.reset_index(), per or {}

    def fmt(column, v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        if isinstance(v, (float, np.floating)):
            return f"{v:.0f}" if column in COUNTS else f"{v:.{per.get(column, digits)}f}"
        return str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    return "\n".join(lines + ["| " + " | ".join(fmt(c, v) for c, v in zip(df.columns, row)) + " |"
                              for row in df.itertuples(index=False)])


def save(fig, name):
    fig.savefig(FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f"figures/{name}"


# Recipe 1: the 2026 laps with the pit stop model's chance and SHAP values, and the two baselines' scores
def load():
    rows = pd.read_parquet(features.OUT_FILE)
    rows = rows[features.in_study(rows)]
    train, test = rows[rows["Year"] <= 2025], rows[rows["Year"] == 2026].reset_index(drop=True)
    booster, info = load_model()
    p, shap_values, base = explain(booster, info, test)
    saved = pd.read_parquet(DATA_DIR / "test_predictions.parquet")              # written by train_pit_model.py
    test = test.merge(saved[KEYS + ["Tyre age rule", "Logistic regression"]], on=KEYS, how="left", validate="one_to_one")
    test["LightGBM"] = p
    test["Label"] = "R" + test["Round"].astype(str).str.zfill(2) + " " + test["Circuit"].astype(str)
    return train, test, shap_values, base, booster, info


# ---------------------------------------------------------------------------------------------------------
# RQ1, recipe 2: PR-AUC with a 95% confidence interval (whole races resampled), and how often LightGBM wins
def overall(test):
    y, races = test["PitSoon"].values, test["RaceId"].values
    by_race = {r: np.flatnonzero(races == r) for r in np.unique(races)}
    names = list(by_race)
    draws = {m: [] for m in MODELS}
    for _ in range(BOOT):
        ix = np.concatenate([by_race[r] for r in RNG.choice(names, len(names))])
        for m in MODELS:
            draws[m].append(average_precision_score(y[ix], test[m].values[ix]))
    draws = {m: np.array(v) for m, v in draws.items()}
    out = pd.DataFrame({"PR-AUC": [average_precision_score(y, test[m]) for m in MODELS],
                        "95% CI from": [np.percentile(draws[m], 2.5) for m in MODELS],
                        "95% CI to": [np.percentile(draws[m], 97.5) for m in MODELS],
                        "LightGBM better in (% of resamples)": [None] + [100 * (draws["LightGBM"] > draws[m]).mean()
                                                                         for m in MODELS[1:]]},
                       index=pd.Index(MODELS, name="Model"))
    out.loc["Random guess"] = [y.mean(), None, None, None]
    return out


def scores(y, p, threshold):
    alert = p >= threshold
    ap = average_precision_score(y, p) if 0 < y.sum() < len(y) else np.nan
    return {"Laps": len(y), "Pit soon (%)": 100 * y.mean(), "PR-AUC": ap, "Lift over random": ap / y.mean(),
            "Precision": (alert & (y == 1)).sum() / max(alert.sum(), 1), "Recall": (alert & (y == 1)).sum() / max(y.sum(), 1)}


# RQ1, recipe 3: where the model works and where it fails. PR-AUC depends on how many laps are "pit soon"
# laps, so "lift" (PR-AUC / random guess) is what compares fairly between groups.
def breakdown(test, threshold):
    t = test
    neutral = (t["SafetyCar"] == 1) | (t["VSC"] == 1)
    wet = (t["FieldOnWetTyres"] > 0) | (t["Rainfall"] == 1)
    groups = {"All 2026 laps": t.index == t.index, "Green flag laps": ~neutral, "Safety car / VSC laps": neutral,
              "Dry laps": ~wet, "Wet laps (rain, or cars on wet tyres)": wet,
              "First stint": t["Stint"] == 1, "Second stint": t["Stint"] == 2, "Third stint or later": t["Stint"] >= 3,
              "First third of the race": t["RaceProgress"] <= 1 / 3,
              "Middle third": t["RaceProgress"].between(1 / 3, 2 / 3, inclusive="right"),
              "Last third": t["RaceProgress"] > 2 / 3,
              "On soft tyres": t["Compound"] == "SOFT", "On medium tyres": t["Compound"] == "MEDIUM",
              "On hard tyres": t["Compound"] == "HARD"}
    rows = {name: scores(t.loc[m, "PitSoon"].values, t.loc[m, "LightGBM"].values, threshold)
            for name, m in groups.items() if t.loc[m, "PitSoon"].sum() >= 10}
    by_group = pd.DataFrame(rows).T.rename_axis("Laps in the group")
    by_race = pd.DataFrame({label: scores(r["PitSoon"].values, r["LightGBM"].values, threshold)
                            for label, r in t.groupby("Label")}).T.rename_axis("Race")
    return by_group, by_race


def breakdown_charts(by_group, by_race):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    order = by_race.sort_values("PR-AUC")
    ax.barh(order.index, order["PR-AUC"], color="tab:blue", label="PR-AUC")
    ax.scatter(order["Pit soon (%)"] / 100, order.index, color="black", marker="|", s=200, label="Random guess")
    ax.set_xlabel("PR-AUC (higher is better; the black line is a random guess for that race)")
    ax.set_title("Pit stop in the next 3 laps: each 2026 race")
    ax.legend(loc="lower right")
    a = save(fig, "eval_by_race.png")
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    groups = by_group.drop("All 2026 laps")[::-1]
    ax.barh(groups.index, groups["Lift over random"], color="tab:orange")
    ax.axvline(by_group.loc["All 2026 laps", "Lift over random"], color="black", linestyle="--", label="All laps")
    ax.axvline(1, color="grey", linewidth=0.8)
    ax.set_xlabel("Lift: PR-AUC divided by a random guess (1 = no better than random)")
    ax.set_title("Where the model works best (2026)")
    ax.legend(loc="lower right")
    return a, save(fig, "eval_by_condition.png")


# RQ1, recipe 4: can the percentages be trusted? (when it says 30%, does a stop follow about 30% of the time?)
def calibration(test):
    y, out = test["PitSoon"].values, {}
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Perfect")
    for m, colour in [("LightGBM", "tab:blue"), ("Logistic regression", "tab:green")]:
        p = test[m].values
        edges = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
        b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
        said = np.array([p[b == k].mean() for k in range(len(edges) - 1)])
        happened = np.array([y[b == k].mean() for k in range(len(edges) - 1)])
        n = np.array([(b == k).sum() for k in range(len(edges) - 1)])
        ece = float((n * np.abs(said - happened)).sum() / n.sum())
        out[m] = {"Brier score": brier_score_loss(y, p), "Calibration error (ECE)": ece,
                  "Average chance given (%)": 100 * p.mean(), "Laps really before a stop (%)": 100 * y.mean(),
                  "Top 10% of laps: chance given (%)": 100 * said[-1], "Top 10% of laps: really pitted (%)": 100 * happened[-1]}
        ax.plot(said, happened, marker="o", color=colour, label=f"{m} (ECE {ece:.3f})")
    ax.set_xlabel("Chance the model gave (10 groups of laps)")
    ax.set_ylabel("Share of those laps really followed by a stop")
    ax.set_title("Can the percentages be trusted? (2026)")
    ax.legend()
    return pd.DataFrame(out).T.rename_axis("Model"), save(fig, "eval_calibration.png")


# RQ1, recipe 5: how long the warning is on before a stop, which stops are missed, and what the false alarms are.
# Stops count as in train_pit_model.py: those with at least one lap in the 3 laps before (a stop on lap 1 can't be warned).
def warnings_check(test, threshold):
    stops = real_stops()
    stops = stops[stops["RaceId"].isin(set(test["RaceId"]))]
    alert = (test.set_index(KEYS)["LightGBM"] >= threshold).to_dict()
    out = []
    for _, s in stops.iterrows():
        before = [alert.get((s["RaceId"], s["Driver"], s["LapNumber"] - k)) for k in range(1, 11)]
        if all(a is None for a in before[:features.HORIZON]):
            continue
        status = str(s["TrackStatus"])
        kind = ("Under the safety car / VSC" if any(c in status for c in "467") else
                "Wet tyres" if s["Compound"] in ("INTERMEDIATE", "WET") else
                "Early: tyres under 8 laps old" if s["TyreLife"] < 8 else "Normal stop")
        last = next((k for k, a in enumerate(before[:features.HORIZON], start=1) if a), None)   # latest warning
        on = 0
        if last:                                  # how many laps in a row the warning was on, up to that lap
            while last + on <= len(before) and before[last + on - 1]:
                on += 1
        out.append({"kind": kind, "warned": last is not None, "first": last + on - 1 if last else None})
    stops_df = pd.DataFrame(out)
    kinds = stops_df.groupby("kind").agg(Stops=("warned", "size"), **{"Warned 1-3 laps before (%)": ("warned", "mean")})
    kinds["Warned 1-3 laps before (%)"] *= 100
    # The headline comes from train_pit_model.py's own count, so the two scripts can never disagree
    warned, _, n_stops = stops_warned(test[KEYS], test["LightGBM"].values, threshold, stops)
    if n_stops != len(stops_df):
        print(f"  note: {len(stops_df)} stops in the breakdown, {n_stops} in train_pit_model.stops_warned")
    kinds.loc["All stops"] = [n_stops, 100 * warned]
    kinds = kinds.rename_axis("Kind of stop").astype({"Stops": int})

    # False alarms: alerts with no stop in the next 3 laps. How far away was the next stop?
    next_stop = {}
    for (race, driver), laps in stops.groupby(["RaceId", "Driver"])["LapNumber"]:
        next_stop[(race, driver)] = np.sort(laps.values)
    gap = []
    for race, driver, lap in test.loc[(test["LightGBM"] >= threshold) & (test["PitSoon"] == 0), KEYS].itertuples(index=False):
        later = next_stop.get((race, driver), np.array([]))
        later = later[later > lap]
        gap.append(later[0] - lap if len(later) else np.inf)
    gap = np.array(gap)
    false_alarms = pd.DataFrame({"Alerts": [len(gap), (gap <= 6).sum(), ((gap > 6) & np.isfinite(gap)).sum(), np.isinf(gap).sum()]},
                                index=pd.Index(["All false alarms", "Stop came 4-6 laps later (a little early)",
                                                "Stop came 7+ laps later", "No more stops in the race"], name="False alarm"))
    false_alarms["Share (%)"] = 100 * false_alarms["Alerts"] / max(len(gap), 1)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    leads = stops_df["first"].dropna().astype(int)
    a1.bar(*np.unique(leads, return_counts=True), color="tab:blue")
    a1.set_xlabel("Laps before the stop the warning was on without a break (10 = 10 or more)")
    a1.set_ylabel("Warned stops")
    a1.set_title(f"How early (median {leads.median():.0f} laps)")
    k = kinds.drop("All stops").sort_values("Warned 1-3 laps before (%)")
    bars = a2.barh([f"{n} ({int(c)})" for n, c in zip(k.index, k["Stops"])], k["Warned 1-3 laps before (%)"], color="tab:orange")
    a2.bar_label(bars, fmt="%.0f%%", padding=3)
    a2.set_xlim(0, 100)
    a2.axvline(kinds.loc["All stops", "Warned 1-3 laps before (%)"], color="black", linestyle="--", label="All stops")
    a2.set_xlabel("Warned 1-3 laps before (%)")
    a2.set_title("Which stops are found (number of stops)")
    a2.legend(loc="lower right")
    summary = {"stops": n_stops, "warned": 100 * warned, "median_lead": float(leads.median()),
               "lead_counts": dict(zip(*map(list, np.unique(leads, return_counts=True))))}
    return kinds, false_alarms, summary, save(fig, "eval_warnings.png")


# ---------------------------------------------------------------------------------------------------------
# RQ2, recipe 6: does each reason push the chance the way F1 logic says it should?
def logic_check(test, shap_values):
    out = {}
    for name in shap_values.abs().mean().sort_values(ascending=False).index[:15]:
        if name in features.CATEGORIES:
            by_cat = shap_values[name].groupby(test[name].astype(str)).mean().sort_values()
            out[name] = {"Higher value -> pit": f"most against: {by_cat.index[0]}, most for: {by_cat.index[-1]}",
                         "Spearman r": None, "F1 logic says": "-", "Agrees": "-"}
            continue
        r = test[name].corr(shap_values[name], method="spearman")
        seen = "weak (no clear effect)" if abs(r) < 0.1 else "more likely" if r > 0 else "less likely"
        expected = EXPECTED.get(name)
        out[name] = {"Higher value -> pit": seen, "Spearman r": r, "F1 logic says": expected or "no single direction",
                     "Agrees": "-" if expected is None else "weak" if abs(r) < 0.1 else "yes" if seen.startswith(expected) else "NO"}
    return pd.DataFrame(out).T.rename_axis("Feature (15 biggest)")


# RQ2, recipe 7: the reasons on the laps before real stops. Is the top reason what a strategist would say?
def reasons_before_stops(test, shap_values, threshold):
    found = (test["PitSoon"] == 1) & (test["LightGBM"] >= threshold)
    top = shap_values[found].idxmax(axis=1).map(THEME_OF)                    # the biggest reason pushing the chance up
    themes = (100 * top.value_counts(normalize=True)).rename("Top reason (%)").to_frame().rename_axis("Kind of reason")
    # Laps under the safety car / VSC before a stop: is the safety car among the top 3 reasons?
    sc = (test["PitSoon"] == 1) & ((test["SafetyCar"] == 1) | (test["VSC"] == 1))
    top3 = shap_values[sc].apply(lambda r: set(r.nlargest(3).index), axis=1)
    sc_share = 100 * top3.map(lambda s: bool(s & set(NEUTRAL))).mean() if sc.any() else np.nan
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    order = themes.sort_values("Top reason (%)")
    ax.bar_label(ax.barh(order.index, order["Top reason (%)"], color="tab:red"), fmt="%.1f%%", padding=3)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of correct warnings (%)")
    ax.set_title("The main reason the model gives before real stops (2026)")
    return themes, sc_share, int(sc.sum()), save(fig, "eval_reasons_before_stops.png")


# RQ2, recipe 8: do the explanations stay steady from one lap to the next for the same driver?
def stability(test, shap_values):
    order = test.sort_values(KEYS).index
    t, s = test.loc[order], shap_values.loc[order]
    next_lap = ((t["RaceId"].values[1:] == t["RaceId"].values[:-1]) & (t["Driver"].values[1:] == t["Driver"].values[:-1])
                & (np.diff(t["LapNumber"].values) == 1))
    ranks = s.abs().rank(axis=1).values
    z = (ranks - ranks.mean(axis=1, keepdims=True)) / ranks.std(axis=1, keepdims=True)
    corr = lambda a, b: (z[a] * z[b]).mean(axis=1)                          # Spearman r of the reasons' order
    top = np.abs(s.values).argmax(axis=1)
    top3 = np.argsort(-np.abs(s.values), axis=1)[:, :3]
    overlap = lambda a, b: np.array([len(set(top3[i]) & set(top3[j])) / 3 for i, j in zip(a, b)])
    a = np.flatnonzero(next_lap)
    b = a + 1
    shuffled = RNG.permutation(len(t))[: len(a)]
    other = RNG.permutation(len(t))[: len(a)]
    rows = {"Same driver, next lap": {"Same top reason (%)": 100 * (top[a] == top[b]).mean(),
                                      "Top 3 reasons shared (%)": 100 * overlap(a, b).mean(),
                                      "Order of all reasons (median Spearman r)": float(np.median(corr(a, b)))},
            "Two random laps (for comparison)": {"Same top reason (%)": 100 * (top[shuffled] == top[other]).mean(),
                                                 "Top 3 reasons shared (%)": 100 * overlap(shuffled, other).mean(),
                                                 "Order of all reasons (median Spearman r)": float(np.median(corr(shuffled, other)))}}
    return pd.DataFrame(rows).T.rename_axis("Pairs of laps")


# RQ2, recipe 9: do the reasons really matter? Take away the top k reasons (set them to a typical value)
# and see how much the chance drops, compared with taking away k reasons picked at random.
def faithfulness(test, shap_values, booster, info, train, threshold, ks=(1, 2, 3, 4, 5), tries=5):
    feats = info["features"]
    alerts = test["LightGBM"] >= threshold
    X, S = test.loc[alerts, feats].copy(), shap_values.loc[alerts, feats].values
    typical = {f: (train[f].mode()[0] if f in features.CATEGORIES else train[f].median()) for f in feats}
    p0 = test.loc[alerts, "LightGBM"].values

    def without(order, k):
        Xk = X.copy()
        for j, f in enumerate(feats):
            Xk.loc[(order[:, :k] == j).any(axis=1), f] = typical[f]
        return booster.predict(Xk)

    by_shap = np.argsort(-S, axis=1)                                         # the reasons pushing the chance up most
    rows = {}
    for k in ks:
        shap_drop = 100 * (p0 - without(by_shap, k)).mean()
        random_drop = np.mean([100 * (p0 - without(np.argsort(RNG.random(S.shape), axis=1), k)).mean() for _ in range(tries)])
        rows[k] = {"Drop: top reasons taken away (points)": shap_drop, "Drop: random reasons taken away (points)": random_drop,
                   "Still above the alert line (%)": 100 * (without(by_shap, k) >= threshold).mean()}
    out = pd.DataFrame(rows).T.rename_axis("Reasons taken away")
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.plot(out.index, out.iloc[:, 0], marker="o", color="tab:red", label="The top SHAP reasons")
    ax.plot(out.index, out.iloc[:, 1], marker="o", color="grey", label="Reasons picked at random")
    ax.set_xticks(list(out.index))
    ax.set_xlabel("Reasons taken away (set to a typical value)")
    ax.set_ylabel("Average drop in pit stop chance (% points)")
    ax.set_title(f"Do the reasons matter? ({alerts.sum()} alert laps, 2026)")
    ax.legend()
    return out, int(alerts.sum()), save(fig, "eval_faithfulness.png")


# ---------------------------------------------------------------------------------------------------------
# RQ3, recipe 10: how long one lap takes in replay mode (both models and SHAP for every driver on the lap)
def replay_speed(booster, info):
    laps = pd.read_parquet(DATA_DIR / "replay_laps.parquet")
    laps = laps[(laps["Year"] == 2026) & features.in_study(laps)]
    for col in features.CATEGORIES:
        laps[col] = laps[col].astype("category")
    tyre_model = lgb.Booster(model_file=str(MODELS_DIR / "tyre_model.txt"))
    tyre_feats = json.loads((MODELS_DIR / "tyre_model.json").read_text(encoding="utf-8"))["features"]
    ms, first = [], laps[laps["LapNumber"] == 1]
    explain(booster, info, first), tyre_model.predict(first[tyre_feats])     # once to warm up, as the app does at start
    for _, rows in laps.groupby(["RaceId", "LapNumber"]):
        start = time.perf_counter()
        explain(booster, info, rows)
        tyre_model.predict(rows[tyre_feats])
        ms.append(1000 * (time.perf_counter() - start))
    return np.array(ms)


# RQ3, recipe 11: live on F1 25 - the work after each lap as a recorded race grows, and reading the packets
def game_speed():
    import game_strategy
    import game_telemetry as game
    races = []
    for folder in game.GAME_DIR.iterdir() if game.GAME_DIR.exists() else []:
        if (folder / "session.json").exists() and folder.name != "0":
            s = game.Session.load(folder, light=True)
            if s.is_race() and not s.test:
                races.append((max(len(l) for l in s.laps.values()), folder, s))
    if not races:
        return None
    _, folder, s = max(races, key=lambda r: r[0])
    rows = game_strategy.lap_rows(s)
    status = {car: d["result_status"] for car, d in s.lap_now.items()}
    growth = []
    for n in range(1, int(max(r["LapNumber"] for r in rows)) + 1):
        start = time.perf_counter()                       # all the work the live page waits for after a lap:
        sub = [r for r in game_strategy.lap_rows(s) if r["LapNumber"] <= n]       # the laps so far as rows,
        w = game_strategy.model_all_laps(sub)                                      # features, both models, SHAP,
        game_strategy.assemble(w, status, s.drivers, s.player)                     # every car's reasons
        growth.append((n, len(sub), 1000 * (time.perf_counter() - start)))
    game.GAME_DIR = Path(tempfile.mkdtemp())                                # reading only: nothing is saved
    packets = list(game.read_raw(folder))
    recorder = game.Recorder(resume=False)
    start = time.perf_counter()
    for arrived, data in packets:
        recorder.handle(data, arrived, record=False)
    took = time.perf_counter() - start
    minutes = (packets[-1][0] - packets[0][0]) / 60
    return {"race": f"{s.info.get('track')} ({len(rows)} laps of {len(s.laps)} cars)", "growth": growth,
            "packets": len(packets), "decode_us": 1e6 * took / len(packets), "minutes": minutes,
            "game_rate": len(packets) / (60 * minutes)}


def speed_chart(replay_ms, game):
    fig, axes = plt.subplots(1, 2 if game else 1, figsize=(11 if game else 6, 4))
    axes = np.atleast_1d(axes)
    axes[0].hist(replay_ms, bins=40, range=(0, np.percentile(replay_ms, 99.5) * 1.2), color="tab:blue")
    axes[0].set_xlabel("Time for one lap (ms): pit chance + SHAP + tyre pick, every driver")
    axes[0].set_ylabel("Laps")
    axes[0].set_title(f"Replay of 2026 (median {np.median(replay_ms):.1f} ms)")
    if game:
        n, rows, ms = zip(*game["growth"])
        axes[1].plot(n, ms, marker="o", color="tab:red")
        axes[1].set_xlabel("Laps finished in the race")
        axes[1].set_ylabel("Time after each lap (ms)")
        axes[1].set_ylim(bottom=0)
        axes[1].set_title(f"Live on F1 25: {game['race']}")
    return save(fig, "eval_speed.png")


# ---------------------------------------------------------------------------------------------------------
def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    print("Loading the 2026 laps and the models...")
    train, test, shap_values, base, booster, info = load()
    threshold = info["threshold"]
    md = ["# Evaluation (2026 test races)", "",
          f"17 races, {len(test)} laps, never used to train or tune the models. Alert line: {threshold:.0%} "
          f"(picked on 2025). Written by `python evaluate.py`.", ""]

    print("RQ1: scores with confidence intervals...")
    ov = overall(test)
    by_group, by_race = breakdown(test, threshold)
    fig_race, fig_cond = breakdown_charts(by_group, by_race)
    cal, fig_cal = calibration(test)
    kinds, false_alarms, warn, fig_warn = warnings_check(test, threshold)
    md += ["## RQ1: can the model predict pit stops?", "",
           f"### Scores with 95% confidence intervals ({BOOT} resamples of whole races)", "",
           table(ov, per={"LightGBM better in (% of resamples)": 1}), "",
           "### Where it works and where it fails", "",
           "PR-AUC depends on how many laps are followed by a stop, so *lift* (PR-AUC / random guess) is the fair "
           "comparison between groups.", "", table(by_group), "", f"![By condition]({fig_cond})", "",
           "### Each race", "", table(by_race), "", f"![By race]({fig_race})", "",
           "### Can the percentages be trusted?", "", table(cal), "", f"![Calibration]({fig_cal})", "",
           "### Warnings before real stops", "",
           f"{warn['stops']} real stops in 2026 (with laps before them): {warn['warned']:.1f}% warned 1-3 laps before. "
           f"For those, the warning had been on without a break for a median of {warn['median_lead']:.0f} laps "
           f"(laps: stops = {', '.join(f'{int(k)}: {int(v)}' for k, v in warn['lead_counts'].items())}; 10 means 10 or more).",
           "", table(kinds, 1), "", table(false_alarms, 1), "",
           f"![Warnings]({fig_warn})", ""]

    print("RQ2: SHAP checks...")
    logic = logic_check(test, shap_values)
    themes, sc_share, sc_laps, fig_themes = reasons_before_stops(test, shap_values, threshold)
    steady = stability(test, shap_values)
    faithful, n_alerts, fig_faith = faithfulness(test, shap_values, booster, info, train, threshold)
    agree = (logic["Agrees"] == "yes").sum(), (logic["Agrees"] != "-").sum()       # weak ones count as not agreeing
    md += ["## RQ2: are the SHAP reasons sound? (checked without human participants)", "",
           f"### Do the reasons follow F1 logic? ({agree[0]} of {agree[1]} with one clear direction agree)", "",
           table(logic), "",
           "### The main reason before real stops", "",
           f"On the {sc_laps} laps under a safety car / VSC that came before a stop, the safety car was among the top 3 "
           f"reasons {sc_share:.0f}% of the time.", "", table(themes, 1), "", f"![Reasons]({fig_themes})", "",
           "### Do the explanations stay steady from lap to lap?", "", table(steady, 2), "",
           f"### Do the reasons really matter? ({n_alerts} alert laps)", "",
           "Each lap's top reasons are set to a typical value (median, or the most common tyre and track) and the "
           "model is asked again; the same with reasons picked at random.", "", table(faithful, 1), "",
           f"![Faithfulness]({fig_faith})", ""]

    print("RQ3: speed...")
    replay_ms = replay_speed(booster, info)
    game = game_speed()
    fig_speed = speed_chart(replay_ms, game)
    speed = {"Replay (2026)": {"What is timed": "pit chance + SHAP + tyre pick, every driver on one lap",
                               "Median (ms)": float(np.median(replay_ms)), "95% of laps under (ms)": float(np.percentile(replay_ms, 95)),
                               "Slowest (ms)": float(replay_ms.max()), "Times": len(replay_ms)}}
    if game:
        ms = np.array([g[2] for g in game["growth"]])
        speed["Live F1 25"] = {"What is timed": "after each lap: all laps so far as rows, features, both models, every car's SHAP reasons",
                               "Median (ms)": float(np.median(ms)), "95% of laps under (ms)": float(np.percentile(ms, 95)),
                               "Slowest (ms)": float(ms.max()), "Times": len(ms)}
        speed["Reading F1 25 packets"] = {"What is timed": "decoding one UDP packet from the game (in microseconds)",
                                          "Median (ms)": game["decode_us"] / 1000, "95% of laps under (ms)": None,
                                          "Slowest (ms)": None, "Times": game["packets"]}
    md += ["## RQ3: does it work live?", "", table(pd.DataFrame(speed).T.rename_axis("Where"), 2), ""]
    if game:
        md += [f"F1 25 sent {game['packets']} packets in {game['minutes']:.0f} minutes ({game['game_rate']:.0f} per second); "
               f"reading one takes about {game['decode_us']:.0f} microseconds, so the app could read about "
               f"{1e6 / game['decode_us']:.0f} per second. The page asks for news every second, so a prediction is on screen "
               "at most about a second after a lap ends.", ""]
    md += [f"![Speed]({fig_speed})", ""]
    REPORT.write_text("\n".join(md), encoding="utf-8")

    print(f"\nPR-AUC {ov.loc['LightGBM', 'PR-AUC']:.3f} (95% CI {ov.loc['LightGBM', '95% CI from']:.3f}-"
          f"{ov.loc['LightGBM', '95% CI to']:.3f}); calibration error {cal.loc['LightGBM', 'Calibration error (ECE)']:.3f}; "
          f"{warn['warned']:.0f}% of stops warned; SHAP logic {agree[0]}/{agree[1]}; replay {np.median(replay_ms):.1f} ms per lap")
    print(f"saved {REPORT} and figures/eval_*.png")


if __name__ == "__main__":
    main()
