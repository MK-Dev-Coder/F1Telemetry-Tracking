"""How fast do the models learn the 2026 rules? 2026 brought new cars, new tyres and new tracks, and the models
were trained on 2022-2025 only. Here each model is trained again before every 2026 race, with the 2026 races
before it added, and tested on that race - as a team would do through a season.

    Fixed       trained on 2022-2025 (the saved models, as in the other evaluations)
    Rolling     2022-2025 + the 2026 races so far, every lap counting the same
    Rolling x3  the same, with each 2026 lap counting three times (the new rules weigh more)

Same settings and number of trees as the saved models (nothing is tuned on 2026). Round 1 has no 2026 races before
it, so the comparison is on rounds 2-17.
    python evaluate_rolling.py   -> data/rolling_2026.md, data/rolling_2026.csv, figures/rolling_2026.png
"""
import json
import time
import warnings

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss

import features
import race_outcome as ro
import train_pit_model as pit
import train_race_model as race
import train_tyre_model as tyre
from config import DATA_DIR, FIGURES_DIR, MODELS_DIR
from evaluate import save, table
from explain_pit_model import load_model

matplotlib.use("Agg")                         # save charts to files, don't open windows
import matplotlib.pyplot as plt               # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
RNG = np.random.default_rng(42)
BOOT = 1000
REPORT = DATA_DIR / "rolling_2026.md"
WAYS = ["Fixed", "Rolling", "Rolling x3"]
COLOURS = {"Fixed": "tab:grey", "Rolling": "tab:blue", "Rolling x3": "tab:green"}
WEIGHT = 3                                     # how much a 2026 lap counts in "Rolling x3"


def weights(rows, way):
    return np.where(rows["Year"] == 2026, WEIGHT, 1.0) if way == "Rolling x3" else None


# Recipe 1: the pit stop model - the chance of a stop in the next 3 laps (PR-AUC: higher is better)
def pit_rounds(rounds):
    rows = pd.read_parquet(features.OUT_FILE)
    rows = rows[features.in_study(rows)]
    booster, info = load_model()
    feats, out = info["features"], []
    for r in rounds:
        test = rows[(rows["Year"] == 2026) & (rows["Round"] == r)]
        train = rows[(rows["Year"] <= 2025) | ((rows["Year"] == 2026) & (rows["Round"] < r))]
        p = {"Fixed": booster.predict(test[feats])}
        for way in WAYS[1:]:
            model = lgb.LGBMClassifier(**{**pit.LGBM_BASE, "n_estimators": info["trees"]}, **info["settings"])
            p[way] = model.fit(train[feats], train["PitSoon"], sample_weight=weights(train, way)).predict_proba(test[feats])[:, 1]
        out.append(pd.DataFrame({"Round": r, "y": test["PitSoon"].values, **p}))
        print(f"   pit stops, round {r}: " + ", ".join(f"{w} {average_precision_score(out[-1]['y'], out[-1][w]):.3f}" for w in WAYS))
    return pd.concat(out, ignore_index=True)


# Recipe 2: the tyre model - which compound at each pit stop (accuracy: higher is better)
def tyre_rounds(rounds):
    stops = tyre.stop_rows()
    stops = stops[features.in_study(stops)]
    info = json.loads((MODELS_DIR / "tyre_model.json").read_text(encoding="utf-8"))
    booster = lgb.Booster(model_file=str(MODELS_DIR / "tyre_model.txt"))
    out = []
    for r in rounds:
        test = stops[(stops["Year"] == 2026) & (stops["Round"] == r)]
        if test.empty:
            continue
        train = stops[(stops["Year"] <= 2025) | ((stops["Year"] == 2026) & (stops["Round"] < r))]
        p = {"Fixed": booster.predict(test[tyre.FEATURES])}
        for way in WAYS[1:]:
            model = lgb.LGBMClassifier(**{**tyre.LGBM_BASE, "n_estimators": info["trees"]}, **info["settings"])
            model.fit(train[tyre.FEATURES], train["Next"].map(tyre.CLASSES.index), sample_weight=weights(train, way))
            p[way] = model.predict_proba(test[tyre.FEATURES])
        frame = pd.DataFrame({"Round": r, "y": test["Next"].map(tyre.CLASSES.index).values})
        for way in WAYS:
            frame[way] = p[way].argmax(axis=1)
            frame[f"{way} loss"] = [-np.log(max(q[t], 1e-6)) for q, t in zip(p[way], frame["y"])]
        out.append(frame)
        print(f"   tyres, round {r}: " + ", ".join(f"{w} {(frame[w] == frame['y']).mean():.2f}" for w in WAYS))
    return pd.concat(out, ignore_index=True)


# Recipe 3: the race outcome model - win, podium and points chances (log loss: lower is better)
def race_rounds(rounds):
    booster, info = ro.load_model()
    rows = race.load_rows(ro.load_results())
    feats, out = info["features"], []
    for r in rounds:
        before = rows[(rows["Year"] <= 2025) | ((rows["Year"] == 2026) & (rows["Round"] < r))]
        here = (rows["Year"] == 2026) & (rows["Round"] == r)
        fixed_test = ro.add_pit_adjusted(rows[here].copy(), info["pit_loss"]).reset_index(drop=True)
        p = {"Fixed": ro.chances(booster, info, fixed_test)}
        table_ = ro.pit_loss_table(before)                        # the pit stop cost now knows the 2026 races too
        train = ro.add_pit_adjusted(before.copy(), table_).reset_index(drop=True)
        test = ro.add_pit_adjusted(rows[here].copy(), table_).reset_index(drop=True)
        for way in WAYS[1:]:
            model = lgb.LGBMClassifier(**{**race.LGBM_BASE, "n_estimators": info["trees"]}, **info["settings"],
                                       monotone_constraints=[1] + [0] * len(feats))
            w = weights(train, way)
            model.fit(ro.stack(train, feats), ro.answers(train), sample_weight=None if w is None else np.repeat(w, ro.MAX_K))
            p[way] = race.predict(model, feats, test)
        frame = test[["RaceId", "LapNumber", "Driver", "Final", "RaceProgress"]].assign(Round=r)
        for way in WAYS:
            for target, k in ro.SHOWN.items():
                frame[f"{way}|{target}"] = p[way][:, k - 1]
        out.append(frame)
        win = (frame["Final"] == 1).astype(int)
        print(f"   race outcome, round {r}: win log loss " + ", ".join(
            f"{w} {log_loss(win, np.clip(frame[f'{w}|win'], 1e-6, 1 - 1e-6), labels=[0, 1]):.3f}" for w in WAYS))
    return pd.concat(out, ignore_index=True)


# Recipe 4: one score per race and way, and the difference from the fixed model with a 95% interval
# (whole races resampled)
def per_race(scores):
    """scores: rounds x ways. The mean, and the rolling ways against the fixed one."""
    out = {}
    for way in WAYS:
        entry = {"Mean": scores[way].mean()}
        if way != "Fixed":
            diff = scores[way] - scores["Fixed"]
            draws = [diff.values[RNG.integers(0, len(diff), len(diff))].mean() for _ in range(BOOT)]
            entry.update({"Change": diff.mean(), "95% CI from": np.percentile(draws, 2.5), "95% CI to": np.percentile(draws, 97.5),
                          "Races better": int((diff * scores.attrs["better"] > 0).sum()), "Races": len(diff)})
        out[way] = entry
    return pd.DataFrame(out).T.rename_axis("Model")


def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    start = time.time()
    rounds = list(range(2, features.TEST_LAST_ROUND + 1))                                           # rounds 2-17
    print("Pit stop model:")
    pits = pit_rounds(rounds)
    print("Tyre model:")
    tyres = tyre_rounds(rounds)
    print("Race outcome model:")
    outcome = race_rounds(rounds)

    # One score per race
    pit_ap = pd.DataFrame({w: pits.groupby("Round").apply(lambda g, w=w: average_precision_score(g["y"], g[w]), include_groups=False) for w in WAYS})
    pit_ap.attrs["better"] = 1
    tyre_acc = pd.DataFrame({w: tyres.groupby("Round").apply(lambda g, w=w: (g[w] == g["y"]).mean(), include_groups=False) for w in WAYS})
    tyre_acc.attrs["better"] = 1
    race_loss = {}
    for target, k in ro.SHOWN.items():
        y = (outcome["Final"] <= k).astype(int)
        race_loss[target] = pd.DataFrame({w: (-(y * np.log(np.clip(outcome[f"{w}|{target}"], 1e-6, 1)) + (1 - y)
                                                 * np.log(np.clip(1 - outcome[f"{w}|{target}"], 1e-6, 1)))).groupby(outcome["Round"]).mean()
                                          for w in WAYS})
        race_loss[target].attrs["better"] = -1
    early = outcome[outcome["RaceProgress"] < 0.25]                                 # the start, where the grid did better
    y = (early["Final"] == 1).astype(int)
    race_early = pd.DataFrame({w: (-(y * np.log(np.clip(early[f"{w}|win"], 1e-6, 1)) + (1 - y)
                                     * np.log(np.clip(1 - early[f"{w}|win"], 1e-6, 1)))).groupby(early["Round"]).mean() for w in WAYS})
    race_early.attrs["better"] = -1
    soft = pd.DataFrame({w: [100 * (tyres[w] == 0).mean()] for w in WAYS}, index=["Picked soft (%)"]).T
    soft["Really soft (%)"] = 100 * (tyres["y"] == 0).mean()

    tables = {"Pit stop model: PR-AUC per race (higher is better)": per_race(pit_ap),
              "Tyre model: right compound per race (higher is better)": per_race(tyre_acc),
              "Race outcome: win log loss per race (lower is better)": per_race(race_loss["win"]),
              "Race outcome: podium log loss per race (lower is better)": per_race(race_loss["podium"]),
              "Race outcome: points log loss per race (lower is better)": per_race(race_loss["points"]),
              "Race outcome: win log loss in the first quarter of each race (lower is better)": per_race(race_early)}
    pd.concat({k: v for k, v in tables.items()}).to_csv(DATA_DIR / "rolling_2026.csv")

    # Chart: the running mean of each way's score, race by race through 2026
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    for ax, (title, scores) in zip(axes, [("Pit stop PR-AUC (higher is better)", pit_ap), ("Tyre pick: right compound (higher is better)", tyre_acc),
                                          ("Race outcome: win log loss (lower is better)", race_loss["win"])]):
        for w in WAYS:
            ax.plot(scores.index, scores[w].expanding().mean(), marker="o", color=COLOURS[w], label=w)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(scores.index[::2])
        ax.set_xlabel("2026 round (the running mean up to it)")
        ax.legend()
    fig.suptitle("Do the models get better as 2026 races are added?", fontsize=12)
    figure = save(fig, "rolling_2026.png")

    md = ["# How fast do the models learn the 2026 rules?", "",
          f"Each model is trained again before each 2026 race (rounds {rounds[0]}-{rounds[-1]}), with the 2026 races before "
          f"it added, and tested on that race. *Rolling x{WEIGHT}*: each 2026 lap counts {WEIGHT} times. Same settings and "
          "trees as the saved models. *Change* is the rolling model minus the fixed one, with a 95% interval from "
          f"{BOOT} resamples of the races; *Races better* counts the races where the rolling model scored better.", ""]
    for title, t in tables.items():
        md += [f"## {title}", "", table(t, 3, per={"Races better": 0, "Races": 0}), ""]
    md += ["## Soft tyres (the tyre model picked them too rarely in 2026)", "", table(soft, 1), "", f"![Rolling]({figure})", "",
           f"Took {(time.time() - start) / 60:.0f} minutes. Written by `python evaluate_rolling.py`."]
    REPORT.write_text("\n".join(md), encoding="utf-8")
    for title, t in tables.items():
        print(f"\n{title}\n{t.round(3).to_string()}")
    print(f"\n{soft.round(1).to_string()}\nsaved {REPORT}, data/rolling_2026.csv and {figure}")


if __name__ == "__main__":
    main()
