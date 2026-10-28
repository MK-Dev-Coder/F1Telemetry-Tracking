"""Evaluation of the race outcome model (each driver's chance to win, finish on the podium and score points,
lap by lap), for the dissertation. Everything is measured on the 17 races of 2026, never used to train or tune it.

RQ4a  How good are the chances?    Log loss and Brier score with 95% confidence intervals, against two baselines.
RQ4b  When does it know?           The scores by how far into the race; the winner's chance as the race goes on.
RQ4c  Can the % be trusted?        Calibration of the win, podium and points chances.
RQ4d  Are the SHAP reasons sound?  Do they follow racing logic, and which reasons matter most? Two case studies.
RQ4e  Is it fast enough live?      Time for one lap: every driver's chances and their reasons.
    python evaluate_race_model.py   -> figures/race_*.png and data/race_evaluation.md (run train_race_model.py first)
"""
import textwrap
import time
import warnings

import matplotlib
import numpy as np
import pandas as pd

import race_outcome as ro
from config import DATA_DIR, FIGURES_DIR
from evaluate import save, table
from explain_pit_model import as_points
from train_race_model import load_rows

matplotlib.use("Agg")                         # save charts to files, don't open windows
import matplotlib.pyplot as plt               # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
RNG = np.random.default_rng(42)
BOOT = 1000                                   # bootstrap resamples (whole races are drawn)
REPORT = DATA_DIR / "race_evaluation.md"
MODELS = ["LightGBM", "Position only", "Grid only"]
COLOURS = {"LightGBM": "tab:blue", "Position only": "tab:orange", "Grid only": "tab:grey"}
STAGES = [(0, 0.1, "Start (0-10%)"), (0.1, 0.25, "10-25%"), (0.25, 0.5, "25-50%"), (0.5, 0.75, "50-75%"),
          (0.75, 0.9, "75-90%"), (0.9, 1.01, "Last 10%")]
# What racing logic says a higher value should do to every chance (only where there is one clear answer:
# e.g. more laps to go is bad for the leader but good for a car behind, so it has no single direction)
EXPECTED = {"GapAhead": "less", "GapBehind": "more",
            "PlacesToK": "less", "MarginK": "more", "MarginKPerLap": "more", "GridToK": "less",
            "AdjustedPlacesToK": "less", "AdjustedMarginK": "more", "StopsVsAhead": "more", "StopsVsLeader": "more",
            "Pace": "less", "PaceVsAhead": "less", "PaceVsBehind": "less", "PaceVsLeader": "less",
            "TyreAgeVsAhead": "less", "TeamForm": "less", "DriverForm": "less"}


def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# Recipe 1: the 2026 laps with every model's chances (from train_race_model.py), and the model's SHAP values
def load():
    booster, info = ro.load_model()
    rows = load_rows(ro.load_results())                          # the results training used
    rows = ro.add_pit_adjusted(rows[rows["Year"] == 2026].copy(), info["pit_loss"]).reset_index(drop=True)
    saved = pd.read_parquet(DATA_DIR / "race_test_predictions.parquet")
    keys = ["RaceId", "Driver", "LapNumber"]
    rows = rows.merge(saved[keys + [c for c in saved if "|" in c]], on=keys, how="left", validate="one_to_one")
    rows["Label"] = "R" + rows["Round"].astype(str).str.zfill(2) + " " + rows["Circuit"].astype(str)
    rows["Stage"] = pd.cut(rows["RaceProgress"], [s[0] for s in STAGES] + [STAGES[-1][1]], right=False,
                           labels=[s[2] for s in STAGES])
    shap = {t: ro.explain(booster, info, rows, k) for t, k in ro.SHOWN.items()}
    return rows, shap, booster, info


# ---------------------------------------------------------------------------------------------------------
# RQ4a, recipe 2: log loss and Brier score with 95% confidence intervals (whole races resampled)
def overall(rows):
    races = rows["RaceId"].values
    by_race = {r: np.flatnonzero(races == r) for r in np.unique(races)}
    names, out = list(by_race), []
    for target, k in ro.SHOWN.items():
        y = (rows["Final"] <= k).astype(int).values
        loss = {m: logloss(y, rows[f"{m}|{target}"].values) for m in MODELS}
        draws = {m: [] for m in MODELS}
        for _ in range(BOOT):
            ix = np.concatenate([by_race[r] for r in RNG.choice(names, len(names))])
            for m in MODELS:
                draws[m].append(loss[m][ix].mean())
        draws = {m: np.array(v) for m, v in draws.items()}
        for m in MODELS:
            p = rows[f"{m}|{target}"].values
            out.append({"Chance": target.title(), "Model": m, "Log loss": loss[m].mean(),
                        "95% CI from": np.percentile(draws[m], 2.5), "95% CI to": np.percentile(draws[m], 97.5),
                        "Brier score": np.mean((p - y) ** 2),
                        "LightGBM better in (% of resamples)": None if m == "LightGBM" else 100 * (draws["LightGBM"] < draws[m]).mean()})
    return pd.DataFrame(out).set_index(["Chance", "Model"])


# ---------------------------------------------------------------------------------------------------------
# RQ4b, recipe 3: the scores at each stage of the race. "Favourite wins" = the driver with the highest win
# chance on that lap really won. The finish error compares the likely finish (the median) with the real one
# ("Current position" = the naive guess that nobody moves again).
def by_stage(rows):
    out = []
    for stage, part in rows.groupby("Stage", observed=True):
        laps = part.groupby(["RaceId", "LapNumber"])
        entry = {"Stage": stage, "Laps": laps.ngroups}
        for m in MODELS:
            entry[f"Win log loss: {m}"] = logloss((part["Final"] == 1).astype(int).values, part[f"{m}|win"].values).mean()
        for m in MODELS:
            fav = part.loc[part.groupby(["RaceId", "LapNumber"])[f"{m}|win"].idxmax()]
            entry[f"Favourite wins (%): {m}"] = 100 * (fav["Final"] == 1).mean()
        entry["Finish error: LightGBM"] = (part["LightGBM|likely"] - part["Final"]).abs().mean()
        entry["Finish error: Current position"] = (part["Position"] - part["Final"]).abs().mean()
        out.append(entry)
    return pd.DataFrame(out).set_index("Stage")


def stage_charts(stages, rows):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(stages))
    for m in MODELS:
        axes[0].plot(x, stages[f"Win log loss: {m}"], marker="o", color=COLOURS[m], label=m)
        axes[1].plot(x, stages[f"Favourite wins (%): {m}"], marker="o", color=COLOURS[m], label=m)
    axes[0].set_ylabel("Win log loss (lower is better)")
    axes[1].set_ylabel("Laps where the favourite really won (%)")
    for ax in axes:
        ax.set_xticks(x, stages.index, rotation=20)
        ax.set_xlabel("How far into the race")
        ax.legend()
    axes[0].set_title("How good the win chances are, by stage (2026)")
    axes[1].set_title("Does the favourite win? (2026)")
    first = save(fig, "race_by_stage.png")

    # The real winner's chance as the race goes on: the median over the 17 races, and the range
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    won = rows[rows["Final"] == 1].copy()
    won["Bucket"] = (won["RaceProgress"].clip(0, 0.999) * 20).astype(int)
    for m in MODELS:
        g = won.groupby("Bucket")[f"{m}|win"]
        mid = (g.median().index + 0.5) / 20
        ax.plot(mid, g.median(), color=COLOURS[m], label=f"{m} (median)")
        if m == "LightGBM":
            ax.fill_between(mid, g.quantile(0.25), g.quantile(0.75), color=COLOURS[m], alpha=0.15, label="LightGBM (middle half)")
    ax.set_xlabel("How far into the race")
    ax.set_ylabel("Win chance given to the real winner")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1)
    ax.set_title("How early does it see the winner? (17 races of 2026)")
    ax.legend()
    return first, save(fig, "race_winner_chance.png")


# ---------------------------------------------------------------------------------------------------------
# RQ4c, recipe 4: calibration - when it says 30%, does it happen about 30% of the time?
def calibration(rows):
    edges = [0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.98, 1.0001]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    out = []
    for ax, (target, k) in zip(axes, ro.SHOWN.items()):
        y = (rows["Final"] <= k).astype(int).values
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Perfect")
        for m in ["LightGBM", "Position only"]:
            p = rows[f"{m}|{target}"].values
            b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
            groups = [k_ for k_ in range(len(edges) - 1) if (b == k_).any()]
            said = np.array([p[b == g].mean() for g in groups])
            happened = np.array([y[b == g].mean() for g in groups])
            n = np.array([(b == g).sum() for g in groups])
            ece = float((n * np.abs(said - happened)).sum() / n.sum())
            out.append({"Chance": target.title(), "Model": m, "Calibration error (ECE)": ece,
                        "Average chance given (%)": 100 * p.mean(), "Really happened (%)": 100 * y.mean()})
            ax.plot(said, happened, marker="o", color=COLOURS[m], label=f"{m} (ECE {ece:.3f})")
        ax.set_title(f"{target.title()} chance (2026)")
        ax.set_xlabel("Chance the model gave (groups of laps)")
        ax.set_ylabel("Share of those that really happened")
        ax.legend()
    return pd.DataFrame(out).set_index(["Chance", "Model"]), save(fig, "race_calibration.png")


# ---------------------------------------------------------------------------------------------------------
# RQ4d, recipe 5: does each reason push the chances the way racing logic says? (Spearman correlation
# between a feature's value and its SHAP value; weak = |r| < 0.1, unused = the model never uses it)
def logic_check(rows, shap):
    out = []
    for name, want in EXPECTED.items():
        entry = {"Feature": name, "Racing logic: a higher value means": f"{want} likely"}
        agree = []
        for target, (values, _, words) in shap.items():
            if name not in values:
                continue
            ok = words[name].notna()
            r = words.loc[ok, name].corr(values.loc[ok, name], method="spearman")
            entry[f"r ({target})"] = r
            agree.append("unused" if pd.isna(r) else "weak" if abs(r) < 0.1 else "yes" if (r < 0) == (want == "less") else "no")
        if agree:
            entry["Agrees"] = ("yes" if all(a == "yes" for a in agree) else "no" if "no" in agree
                               else "unused" if all(a == "unused" for a in agree) else "weak")
            out.append(entry)
    return pd.DataFrame(out).set_index("Feature")


def importance(shap):
    """The average size of each reason, in % points, for the win, podium and points chances."""
    size = pd.DataFrame({target: values.abs().mean() for target, (values, _, _) in shap.items()})
    size = size.sort_values("win", ascending=False)
    top = size.head(14)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(top))
    for i, (target, colour) in enumerate(zip(ro.SHOWN, ["tab:red", "tab:purple", "tab:green"])):
        ax.barh(y + (i - 1) * 0.27, top[target], height=0.27, color=colour, label=target.title())
    ax.set_yticks(y, top.index)
    ax.set_xlabel("Average size of the reason (|SHAP|, % points)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("What the race outcome model leans on (2026)")
    ax.legend()
    return size.rename(columns=str.title), save(fig, "race_shap_importance.png")


# Recipe 6: two case studies - the lap where the model and the current order disagreed most about the winner
# and the model was right, and the biggest upset (the real winner's lowest chance in the second half)
def cases(rows, shap):
    leader = rows[rows["Position"] == 1].drop_duplicates(["RaceId", "LapNumber"]).set_index(["RaceId", "LapNumber"])["Driver"]
    fav = rows.loc[rows.groupby(["RaceId", "LapNumber"])["LightGBM|win"].idxmax()].set_index(["RaceId", "LapNumber"])
    known = leader.reindex(fav.index)
    disagree = fav[known.notna() & (fav["Driver"] != known) & (fav["Final"] == 1)]
    found = []
    if len(disagree):
        key = disagree["LightGBM|win"].idxmax()
        idx = rows.index[(rows["RaceId"] == key[0]) & (rows["LapNumber"] == key[1]) & (rows["Driver"] == disagree.loc[key, "Driver"])][0]
        found.append(("The model saw the winner before the order did", idx))
    late = rows[(rows["Final"] == 1) & (rows["RaceProgress"] >= 0.5)]
    found.append(("The biggest upset", late["LightGBM|win"].idxmin()))
    for old in FIGURES_DIR.glob("race_case*.png"):                     # the charts of an earlier run
        old.unlink()
    out = []
    for k, (kind, idx) in enumerate(found, start=1):
        values, start, words = shap["win"]
        row = words.loc[idx]
        points = values.loc[idx]
        model = start + points.sum()
        title = f"{row['Driver']} - {row['EventName']} {row['Year']} - lap {row['LapNumber']:.0f} (P{row['Position']:.0f})"
        lines = [f"{title}: {row['LightGBM|win']:.0%} chance to win (the leader on this lap: "
                 f"{leader.get((row['RaceId'], row['LapNumber']), '?')}). Actual: finished P{row['Final']:.0f}. ({kind})"]
        for name in points.abs().sort_values(ascending=False).index[:5]:
            lines.append(f"  {as_points(points[name]):>6}  {ro.describe(name, row, points[name], 1)}")
        out.append((kind, "\n".join(lines)))
        waterfall(row, points, start, model, row["LightGBM|win"],
                  f"race_case{k}_{row['Driver']}_{row['RaceId']}_lap{row['LapNumber']:.0f}.png", f"{title}\n{kind}")
    return out


def waterfall(row, points, start, model, shown, file, title):
    top = points.abs().sort_values(ascending=False).index[:8]
    bars = points[top][::-1]
    labels = [textwrap.fill(ro.describe(n, row, bars[n], 1), 45) for n in bars.index]
    rest, fitted = points.drop(top).sum(), shown - model
    values = [fitted, rest] + list(bars.values)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(["fitted so the field adds up to one winner", "all other features"] + labels, values,
            color=["#bbbbbb", "lightgrey"] + ["tab:red" if v > 0 else "tab:blue" for v in bars.values])
    for i, v in enumerate(values):
        ax.text(v, i, f" {as_points(v)} ", va="center", ha="left" if v >= 0 else "right", fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Change in win chance (from the average driver's {start:.1%} to {shown:.0%})")
    ax.set_title(title, fontsize=10)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    return save(fig, file)


# One race in full: every driver's win chance, lap by lap (the chart the app shows)
def race_chart(rows, race="2026_17"):
    part = rows[rows["RaceId"] == race]
    if part.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for driver, mine in part.groupby("Driver"):
        if mine["LightGBM|win"].max() >= 0.05:
            mine = mine.sort_values("LapNumber")
            line, = ax.plot(mine["LapNumber"], mine["LightGBM|win"], label=f"{driver} (finished P{mine['Final'].iloc[0]:.0f})")
    ax.set_xlabel("Lap")
    ax.set_ylabel("Win chance")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1)
    ax.set_title(f"Win chance, lap by lap - {part['EventName'].iloc[0]} {part['Year'].iloc[0]}")
    ax.legend()
    return save(fig, f"race_win_chance_{race}.png")


# ---------------------------------------------------------------------------------------------------------
# RQ4e, recipe 7: the time for one lap in the app - every driver's chances (fitted together) and the
# SHAP reasons for the win, podium and points chances
def speed(rows, booster, info):
    times, laps = [], list(rows.groupby(["RaceId", "LapNumber"]))
    for i in RNG.choice(len(laps), min(200, len(laps)), replace=False):          # 200 laps picked at random
        lap = laps[i][1]
        start = time.perf_counter()
        ro.summary(ro.chances(booster, info, lap))
        for k in ro.SHOWN.values():
            ro.explain(booster, info, lap, k)
        times.append(1000 * (time.perf_counter() - start))
    return np.array(times)


def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    print("Loading the 2026 laps, the chances and the SHAP values...")
    rows, shap, booster, info = load()
    md = ["# Evaluation of the race outcome model (2026 test races)", "",
          f"{rows['RaceId'].nunique()} races, {rows.groupby(['RaceId', 'LapNumber']).ngroups} laps, {len(rows)} driver-laps, "
          f"never used to train or tune the model (version: {info['version']}). Written by `python evaluate_race_model.py`.", "",
          "Baselines: *Position only* = how often drivers in the same position at the same stage of a race finished in the "
          "top 1/3/10 in 2022-2025; *Grid only* = the same from the starting position (it never changes during the race).", ""]

    print("RQ4a: scores with confidence intervals...")
    ov = overall(rows)
    md += ["## RQ4a: how good are the chances?", "", f"### Log loss and Brier score ({BOOT} resamples of whole races)", "",
           "Lower is better for both.", "", table(ov, per={"LightGBM better in (% of resamples)": 1}), ""]

    print("RQ4b: by stage of the race...")
    stages = by_stage(rows)
    fig_stage, fig_winner = stage_charts(stages, rows)
    md += ["## RQ4b: when does it know?", "", table(stages, per={c: 1 for c in stages if "%" in c}), "",
           f"![By stage]({fig_stage})", "", f"![The winner's chance]({fig_winner})", ""]
    chart = race_chart(rows)
    if chart:
        md += [f"![One race]({chart})", ""]

    print("RQ4c: calibration...")
    cal, fig_cal = calibration(rows)
    md += ["## RQ4c: can the percentages be trusted?", "", table(cal, per={"Average chance given (%)": 1, "Really happened (%)": 1}),
           "", f"![Calibration]({fig_cal})", ""]

    print("RQ4d: SHAP checks...")
    logic = logic_check(rows, shap)
    size, fig_size = importance(shap)
    agree = (logic["Agrees"] == "yes").sum(), len(logic)
    md += ["## RQ4d: are the SHAP reasons sound?", "",
           f"### Do the reasons follow racing logic? ({agree[0]} of {agree[1]} agree for all three chances)", "",
           "Spearman correlation between each feature's value and its SHAP value (weak = |r| < 0.1; unused = the model "
           "never uses it, so it has no direction).", "", table(logic, 2), "",
           "### What it leans on", "", table(size.head(14), 3), "", f"![Importance]({fig_size})", "", "### Case studies", ""]
    for kind, text in cases(rows, shap):
        md += [f"**{kind}**", "", "```", text, "```", ""]

    print("RQ4e: speed...")
    ms = speed(rows, booster, info)
    md += ["## RQ4e: is it fast enough live?", "",
           f"One lap (every driver's chances fitted together, and the SHAP reasons for the win, podium and points chances): "
           f"median {np.median(ms):.1f} ms, 95% of laps under {np.percentile(ms, 95):.1f} ms, slowest {ms.max():.1f} ms "
           f"({len(ms)} laps).", ""]
    REPORT.write_text("\n".join(md), encoding="utf-8")

    win = ov.xs("Win", level="Chance")
    print(f"\nWin log loss: LightGBM {win.loc['LightGBM', 'Log loss']:.3f} (95% CI {win.loc['LightGBM', '95% CI from']:.3f}-"
          f"{win.loc['LightGBM', '95% CI to']:.3f}), position only {win.loc['Position only', 'Log loss']:.3f}, grid only "
          f"{win.loc['Grid only', 'Log loss']:.3f}; SHAP logic {agree[0]}/{agree[1]}; {np.median(ms):.1f} ms per lap")
    print(f"saved {REPORT} and figures/race_*.png")


if __name__ == "__main__":
    main()
