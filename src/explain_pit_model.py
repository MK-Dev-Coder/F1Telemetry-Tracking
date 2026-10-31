import json
import textwrap

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import shap

import features
from config import DATA_DIR, FIGURES_DIR, MODELS_DIR
from train_pit_model import real_stops

matplotlib.use("Agg")                         # save charts to files, don't open windows
import matplotlib.pyplot as plt               # noqa: E402

TYRE_COLOURS = {"HARD": "grey", "MEDIUM": "gold", "SOFT": "red", "INTERMEDIATE": "green", "WET": "blue"}


# Recipe 1: load the saved model
def load_model():
    info = json.loads((MODELS_DIR / "pit_model.json").read_text(encoding="utf-8"))
    return lgb.Booster(model_file=str(MODELS_DIR / "pit_model.txt")), info


# Recipe 2: the chance of a stop in the next 3 laps, and the SHAP value of every feature.
# LightGBM works SHAP out exactly, in log-odds (the model's own scale): base + all SHAP values = prediction.
def explain(booster, info, rows):
    X = rows[info["features"]]
    shap_values = booster.predict(X, pred_contrib=True)
    return booster.predict(X), pd.DataFrame(shap_values[:, :-1], columns=info["features"], index=rows.index), shap_values[0, -1]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# For fans: turn the log-odds SHAP values into "% points". Every value is scaled by the same
# number, so the reasons keep their true sizes compared with each other, and the points add up
# exactly to (this prediction - the chance on a typical lap).
def to_points(base, values):
    change = values.sum()
    if abs(change) < 1e-9:
        scale = sigmoid(base) * (1 - sigmoid(base))             # no change: use the slope at the start
    else:
        scale = (sigmoid(base + change) - sigmoid(base)) / change
    return values * scale


# Recipe 3: each reason in plain English
def describe(name, row, info, effect=0.0):
    v = row[name]
    if pd.isna(v):
        return f"{name} unknown"
    compound = str(row["Compound"]).lower()
    if name == "Compound":
        return f"on {compound} tyres"
    if name == "Circuit":                                    # the track's own effect, up or down
        usual = info["usual_stint"]["track"].get(f"{row['Circuit']}|{row['Compound']}")
        more = "more" if effect > 0 else "less"
        return (f"at {row['Circuit']} teams pit {more} than at most tracks"
                + (f" ({compound} tyres usually last about {usual:.0f} laps here)" if usual else ""))
    v = float(v)
    laps = f"{v:.0f} lap" + ("" if round(v) == 1 else "s")             # 1 lap, 2 laps
    text = {
        "LapsInStint": f"{laps} on this set of tyres",
        "TyreLife": f"tyres are {laps} old",
        "RaceProgress": f"{v:.0%} of the race done",
        "LapsLeft": f"{laps} to go",
        "LapNumber": f"lap {v:.0f}",
        "DryCompoundsUsed": ("still has to use a second dry compound (F1 rule)" if v < 2
                             else "has already used two dry compounds"),
        "FieldStopsLast2": (f"{v:.0f} other cars pitted in the last 2 laps" if v else
                            "no other cars pitted in the last 2 laps"),
        "TyreWear": f"lap times {v:.1f} s slower than the best of this stint",
        "PaceVsField": f"{abs(v):.1%} {'slower' if v > 0 else 'faster'} than the field this lap",
        "Position": f"running P{v:.0f}",
        "GapAhead": f"{v:.1f} s behind the car ahead",
        "GapBehind": f"{v:.1f} s ahead of the car behind",
        "GapToLeader": f"{v:.0f} s behind the leader",
        "AheadStoppedRecently": "the car ahead just pitted" if v else "the car ahead hasn't just pitted",
        "BehindStoppedRecently": "the car behind just pitted" if v else "the car behind hasn't just pitted",
        "SafetyCar": "the safety car is out" if v else "no safety car",
        "VSC": "the virtual safety car is out" if v else "no virtual safety car",
        "NeutralStarted": "a safety car has just come out" if v else "no new safety car",
        "LapsUnderNeutral": f"{laps} behind the safety car",
        "Yellow": "yellow flags" if v else "no yellow flags",
        "Stint": {1: "still on the starting tyres", 2: "on the second set of tyres",
                  3: "on the third set of tyres"}.get(int(v), f"on tyre set {v:.0f}"),
        "FreshTyre": "these tyres were new" if v else "these tyres were already used",
        "FieldOnWetTyres": f"{v:.0%} of cars on wet-weather tyres",
        "AirTemp": f"air {v:.0f} C", "TrackTemp": f"track {v:.0f} C", "Humidity": f"humidity {v:.0f}%",
        "Rainfall": "rain" if v else "no rain",
    }
    return text.get(name, f"{name} = {v}")


def as_points(v):
    return f"{v:+.0%}" if abs(v) >= 0.01 else f"{v:+.1%}"         # small ones get a decimal: -0.3%


def explain_text(row, p, points, info, top=4):
    lines = [f"{row['Driver']} - {row['EventName']} {row['Year']} - lap {row['LapNumber']:.0f}: "
             f"{p:.0%} chance of a pit stop in the next {info['horizon_laps']} laps"]
    for name in points.abs().sort_values(ascending=False).index[:top]:
        lines.append(f"  {as_points(points[name]):>6}  {describe(name, row, info, points[name])}")
    return "\n".join(lines)


# Recipe 4: charts for the whole 2026 test season
def season_charts(rows, values, base, info):
    numbers = rows[info["features"]].copy()
    for col in features.CATEGORIES:
        if col in numbers:
            numbers[col] = numbers[col].cat.codes             # SHAP's charts need numbers
    explanation = shap.Explanation(values=values.values, base_values=np.full(len(values), base),
                                   data=numbers.values, feature_names=info["features"])
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    plt.title("What pushes the pit stop chance up or down (2026, log-odds)")
    plt.savefig(FIGURES_DIR / "shap_beeswarm_2026.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    shap.plots.bar(explanation, max_display=15, show=False)
    plt.title("Average size of each feature's effect (2026)")
    plt.savefig(FIGURES_DIR / "shap_importance_2026.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # Tyre age: how its effect grows, one colour per compound
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for compound, colour in TYRE_COLOURS.items():
        mine = (rows["Compound"] == compound).values
        if mine.sum() > 50:
            ax.scatter(rows["LapsInStint"][mine], values["LapsInStint"][mine], s=4, alpha=0.4,
                       color=colour, label=compound.title())
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Laps on this set of tyres")
    ax.set_ylabel("SHAP value (log-odds)\n<- less likely to pit | more likely to pit ->")
    ax.set_title("Older tyres push the pit stop chance up (2026)")
    ax.legend(markerscale=4)
    fig.savefig(FIGURES_DIR / "shap_tyre_age_2026.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# Do the explanations match F1 logic? Higher value -> more or less likely to pit?
def direction_check(rows, values, info):
    out = []
    for name in values.abs().mean().sort_values(ascending=False).index[:12]:
        if name in features.CATEGORIES:
            by_cat = values[name].groupby(rows[name].astype(str)).mean().sort_values()
            out.append(f"  {name:22s} most against a stop: {by_cat.index[0]}, most for a stop: {by_cat.index[-1]}")
        else:
            corr = rows[name].corr(values[name], method="spearman")
            word = "MORE" if corr > 0 else "LESS"
            out.append(f"  {name:22s} higher value -> {word} likely to pit (r = {corr:+.2f})")
    return "\n".join(out)


# Recipe 5: a waterfall chart for one prediction, in % points
def waterfall(row, p, points, base, info, file, title_extra):
    top = points.abs().sort_values(ascending=False).index[:8]
    bars = points[top][::-1]
    labels = [textwrap.fill(describe(n, row, info, bars[n]), 45) for n in bars.index]
    other = points.drop(top).sum()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(["all other features"] + labels, [other] + list(bars.values),
            color=["lightgrey"] + ["tab:red" if v > 0 else "tab:blue" for v in bars.values])
    for i, v in enumerate([other] + list(bars.values)):
        ax.text(v, i, f" {as_points(v)} ", va="center", ha="left" if v >= 0 else "right", fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Change in pit stop chance (starting from a typical lap: {sigmoid(base):.0%})")
    ax.set_title(f"{row['Driver']} - {row['EventName']} {row['Year']} - lap {row['LapNumber']:.0f}: "
                 f"{p:.0%} chance\n{title_extra}", fontsize=10)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.savefig(FIGURES_DIR / file, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Recipe 6: case studies from Singapore 2026 (the race we have from both sources, with 2 VSCs)
def pick_cases(rows, p, stops, race="2026_17"):
    here = rows["RaceId"] == race
    stops = stops[stops["RaceId"] == race]
    window = lambda s: here & (rows["Driver"] == s["Driver"]) & rows["LapNumber"].between(s["LapNumber"] - 3, s["LapNumber"] - 1)
    chance = pd.Series(p, index=rows.index)
    to_next = pd.Series(np.nan, index=rows.index)                       # laps until this driver's next stop
    for _, s in stops.iterrows():
        mine = here & (rows["Driver"] == s["Driver"]) & (rows["LapNumber"] < s["LapNumber"])
        to_next[mine] = np.fmin(to_next[mine], s["LapNumber"] - rows["LapNumber"][mine])
    cases, vsc_stop = [], None
    ver = rows.index[here & (rows["Driver"] == "VER") & (rows["LapNumber"] == 30)]
    if len(ver):
        cases.append(("Correct warning", ver[0]))
    for _, s in stops[stops["LapNumber"].between(40, 41)].iterrows():   # pitted just after the lap-39 VSC came out
        before = rows.index[window(s) & (rows["LapNumber"] == s["LapNumber"] - 1)]
        if len(before):
            cases.append(("Pit stop under the VSC", before[0]))
            vsc_stop = (s["Driver"], s["LapNumber"])
            break
    no_stop = here & (rows["PitSoon"] == 0)
    early = no_stop & to_next.between(4, 6)
    if early.any():
        cases.append(("Warning a little early", chance[early].idxmax()))
    cases.append(("False alarm", chance[no_stop & ~(to_next <= 6)].idxmax()))
    others = [s for _, s in stops.iterrows() if window(s).any() and (s["Driver"], s["LapNumber"]) != vsc_stop]
    lowest = min(others, key=lambda s: chance[window(s)].max())
    cases.append(("Missed stop", rows.index[window(lowest)][-1]))
    return cases


def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    booster, info = load_model()
    rows = pd.read_parquet(features.OUT_FILE)
    rows = rows[(rows["Year"] == 2026) & features.in_study(rows)]
    p, values, base = explain(booster, info, rows)
    print(f"Explained {len(rows)} laps of 2026. SHAP starting point (a typical lap): {sigmoid(base):.1%}")

    season_charts(rows, values, base, info)
    print("\nDo the explanations match F1 logic? (the 12 biggest features)")
    checks = direction_check(rows, values, info)
    print(checks)

    stops = real_stops()
    lines = ["Do the explanations match F1 logic?", checks, ""]
    for k, (kind, idx) in enumerate(pick_cases(rows, p, stops), start=1):
        row, i = rows.loc[idx], rows.index.get_loc(idx)
        later = stops[(stops["RaceId"] == row["RaceId"]) & (stops["Driver"] == row["Driver"])
                      & (stops["LapNumber"] > row["LapNumber"])]["LapNumber"]
        actual = f"actual: pitted on lap {later.min():.0f}" if len(later) else "actual: no more stops"
        points = to_points(base, values.loc[idx])
        text = explain_text(row, p[i], points, info) + f"\n  ({kind}, {actual})"
        print("\n" + text)
        lines += [f"Case {k}: {kind}", text, ""]
        waterfall(row, p[i], points, base, info, f"case{k}_{row['Driver']}_{row['RaceId']}_lap{row['LapNumber']:.0f}.png",
                  f"{kind} - {actual}")
    (DATA_DIR / "case_studies.txt").write_text("\n".join(lines), encoding="utf-8")

    out = rows[features.KEEP + ["LapNumber", "PitSoon"]].assign(Chance=p).join(values.add_prefix("shap_"))
    out.to_parquet(DATA_DIR / "shap_values_2026.parquet", index=False)
    print("\nsaved figures/shap_*.png, figures/case*.png, data/case_studies.txt, data/shap_values_2026.parquet")


if __name__ == "__main__":
    main()
