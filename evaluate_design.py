"""Two checks of the pit stop model's design, on the study's test races (2026, rounds 1-17), each model trained
and tuned exactly as train_pit_model.py does it (2022-2024 to train, 2025 to pick the settings, then 2022-2025 to
train again, then tested on 2026, which it never saw):

1. The horizon: why "a stop in the next 3 laps"? The same model for 1, 2, 3 and 5 laps. A longer window is
   easier to hit (more laps count as "a stop is coming"), so each PR-AUC is set against its own random guess (the
   share of such laps): the lift.
2. The feature groups: the model again without each group of features (tyres, gaps, the safety car...), to see
   how much each group adds: a second view of what SHAP says (python evaluate.py).

Intervals: 95%, from drawing whole races again 1,000 times (races aren't independent laps). The app, the saved
models and the study's numbers don't change: this only writes results/horizon.md, results/feature_groups.md (and
their .csv) and two charts. Run: python evaluate_design.py (about 2 minutes)."""
import json
import time

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import features
import train_pit_model as tpm
from config import FIGURES_DIR, MODELS_DIR, ROOT
from evaluate import THEMES

matplotlib.use("Agg")                         # save charts to files, don't open windows
import matplotlib.pyplot as plt               # noqa: E402

RESULTS_DIR = ROOT / "results"
HORIZONS = [1, 2, 3, 5]
BOOT = 1000
RNG = np.random.default_rng(42)


KEYS = ["RaceId", "Driver", "LapNumber"]


def targets(hs):
    """The target for each horizon h ("a real stop on one of the next h laps"), made by features.py's own code from
    every lap (the feature table leaves out the in-laps, so the stops themselves aren't in it)."""
    laps = pd.read_parquet(features.IN_FILE).sort_values(KEYS).reset_index(drop=True)
    out, saved = laps[KEYS].copy(), features.HORIZON
    try:
        for h in hs:
            features.HORIZON = h                                     # mark_stops reads it when it runs
            features.mark_stops(laps, laps.groupby(["RaceId", "Driver"]))
            out[h] = laps["PitSoon"].to_numpy()
    finally:
        features.HORIZON = saved
    return out


def target(frame, table, h):
    """One horizon's target for the rows of a feature table, in their order."""
    return frame[KEYS].merge(table[KEYS + [h]], on=KEYS, how="left")[h].to_numpy()


def fit_and_test(train, tune, both, test, feats):
    """Pick the settings on 2025 (trained on 2022-2024), train again on 2022-2025 and predict 2026."""
    score, tries, tuned = tpm.tune_lightgbm(train, tune, feats)
    trees = int(tuned.best_iteration_ * 1.1)                     # (as train_pit_model.py)
    final = lgb.LGBMClassifier(**{**tpm.LGBM_BASE, "n_estimators": trees}, **tries).fit(both[feats], both[tpm.TARGET])
    return final.predict_proba(test[feats])[:, 1], trees


def draws(test):
    """The rows of each resample: whole races drawn again, with replacement."""
    rows = list(test.reset_index(drop=True).groupby("RaceId").indices.values())
    picks = RNG.integers(0, len(rows), size=(BOOT, len(rows)))
    return [np.concatenate([rows[i] for i in pick]) for pick in picks]


def interval(values):
    return np.percentile(values, 2.5), np.percentile(values, 97.5)


def horizons(train, tune, both, test, feats, samples, table):
    out = []
    for h in HORIZONS:
        print(f"\nHorizon {h} lap{'s' if h > 1 else ''}:")
        frames = [f.assign(**{tpm.TARGET: target(f, table, h)}) for f in (train, tune, both, test)]
        p, trees = fit_and_test(*frames, feats)
        y = frames[3][tpm.TARGET].to_numpy()
        boot = [average_precision_score(y[i], p[i]) for i in samples]
        ap, share = average_precision_score(y, p), y.mean()
        lo, hi = interval(boot)
        out.append({"Horizon (laps)": h, "Laps with a stop coming": round(share, 4), "PR-AUC": round(ap, 3),
                    "PR-AUC low": round(lo, 3), "PR-AUC high": round(hi, 3), "Lift": round(ap / share, 2),
                    "ROC-AUC": round(roc_auc_score(y, p), 3), "Trees": trees, "p": p, "y": y})
        print(f"   PR-AUC {ap:.3f} ({lo:.3f}-{hi:.3f}), random guess {share:.3f}, lift {ap / share:.2f}")
    return out


def groups(train, tune, both, test, feats, samples, full_p):
    y = test[tpm.TARGET].to_numpy()
    full = average_precision_score(y, full_p)
    full_boot = np.array([average_precision_score(y[i], full_p[i]) for i in samples])
    out = []
    for name, names in THEMES.items():
        gone = [f for f in feats if f in names]
        if not gone:
            continue
        print(f"\nWithout {name} ({', '.join(gone)}):")
        p, _ = fit_and_test(train, tune, both, test, [f for f in feats if f not in gone])
        boot = np.array([average_precision_score(y[i], p[i]) for i in samples]) - full_boot
        ap = average_precision_score(y, p)
        lo, hi = interval(boot)
        out.append({"Group left out": name, "Features": len(gone), "PR-AUC without it": round(ap, 3),
                    "Change": round(ap - full, 3), "Change low": round(lo, 3), "Change high": round(hi, 3),
                    "Worse in (share of resamples)": round(float((boot < 0).mean()), 3)})
        print(f"   PR-AUC {ap:.3f}, change {ap - full:+.3f} ({lo:+.3f} to {hi:+.3f})")
    return full, sorted(out, key=lambda r: r["Change"])


def charts(hz, gr, full):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(hz))
    ax.bar(x - 0.2, [r["PR-AUC"] for r in hz], 0.4, label="LightGBM", color="#2f6db5",
           yerr=[[r["PR-AUC"] - r["PR-AUC low"] for r in hz], [r["PR-AUC high"] - r["PR-AUC"] for r in hz]], capsize=3)
    ax.bar(x + 0.2, [r["Laps with a stop coming"] for r in hz], 0.4, label="Random guess", color="#b9c2cf")
    for k, r in enumerate(hz):
        ax.text(k - 0.2, r["PR-AUC high"] + 0.01, f"x{r['Lift']:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x, [f"{r['Horizon (laps)']} lap{'s' if r['Horizon (laps)'] > 1 else ''}" for r in hz])
    ax.set_ylabel("PR-AUC on the 2026 test races")
    ax.set_title("A stop in the next n laps: the model against a random guess (x = lift)")
    ax.legend(frameon=False)
    fig.savefig(FIGURES_DIR / "eval_horizon.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    names = [r["Group left out"] for r in gr][::-1]
    vals = [r["Change"] for r in gr][::-1]
    err = [[r["Change"] - r["Change low"] for r in gr][::-1], [r["Change high"] - r["Change"] for r in gr][::-1]]
    ax.barh(names, vals, xerr=err, capsize=3, color=["#c0392b" if v < 0 else "#2e8b57" for v in vals])
    ax.axvline(0, color="#333", lw=1)
    ax.set_xlabel(f"Change in PR-AUC without the group (all features: {full:.3f})")
    ax.set_title("What each group of features adds to the pit stop model")
    fig.savefig(FIGURES_DIR / "eval_feature_groups.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    start = time.time()
    RESULTS_DIR.mkdir(exist_ok=True)
    info = json.loads((MODELS_DIR / "pit_model.json").read_text(encoding="utf-8"))
    feats = info["features"]                                         # the model the study uses ("v1")
    train, tune, test = tpm.load()
    both = pd.concat([train, tune])
    table = targets(HORIZONS)
    for f in (both, test):                                           # the 3-lap target must be the study's own
        same = (target(f, table, features.HORIZON) == f[tpm.TARGET].to_numpy()).mean()
        print(f"The rebuilt {features.HORIZON}-lap target matches the saved one on {same:.2%} of {len(f)} rows")
        if same < 1:
            raise SystemExit("The rebuilt target differs from the study's: stopping")
    samples = draws(test)
    hz = horizons(train, tune, both, test, feats, samples, table)
    full_p = next(r["p"] for r in hz if r["Horizon (laps)"] == features.HORIZON)
    full, gr = groups(train, tune, both, test, feats, samples, full_p)
    charts(hz, gr, full)

    table = pd.DataFrame([{k: v for k, v in r.items() if k not in ("p", "y")} for r in hz])
    table.to_csv(RESULTS_DIR / "horizon.csv", index=False)
    pd.DataFrame(gr).to_csv(RESULTS_DIR / "feature_groups.csv", index=False)
    lines = ["# Why a stop \"in the next 3 laps\"? The horizon", "",
             "The pit stop model trained and tuned as in `train_pit_model.py` (the study's features, 2022-2024 to "
             "train, 2025 to pick the settings, 2022-2025 to train again), for each horizon, and tested on the 17 "
             "test races of 2026. A longer window is easier to hit, so each PR-AUC is set against its random guess "
             "(the share of laps with a stop coming): the lift. 95% intervals from drawing whole races again.", "",
             "| Horizon | Laps with a stop coming | PR-AUC (95% interval) | Lift over a random guess | ROC-AUC |",
             "|---|---:|---|---:|---:|"]
    lines += [f"| {r['Horizon (laps)']} lap{'s' if r['Horizon (laps)'] > 1 else ''} | {r['Laps with a stop coming']:.1%} | "
              f"{r['PR-AUC']:.3f} ({r['PR-AUC low']:.3f}-{r['PR-AUC high']:.3f}) | {r['Lift']:.2f} | {r['ROC-AUC']:.3f} |" for r in hz]
    lines += ["", f"The 3-lap model is the study's (PR-AUC {full:.3f}). Chart: `figures/eval_horizon.png`.", ""]
    (RESULTS_DIR / "horizon.md").write_text("\n".join(lines), encoding="utf-8")
    lines = ["# What each group of features adds", "",
             f"The study's pit stop model (PR-AUC {full:.3f} on the 2026 test races) trained again without one group "
             "of features at a time, tuned and tested the same way. A drop means the group carries information the "
             "others don't; near zero means the other features cover it. 95% intervals from drawing whole races "
             "again (the same draws for every model).", "",
             "| Group left out | Features | PR-AUC without it | Change (95% interval) | Worse in |",
             "|---|---:|---:|---|---:|"]
    lines += [f"| {r['Group left out']} | {r['Features']} | {r['PR-AUC without it']:.3f} | {r['Change']:+.3f} "
              f"({r['Change low']:+.3f} to {r['Change high']:+.3f}) | {r['Worse in (share of resamples)']:.0%} |" for r in gr]
    lines += ["", "Chart: `figures/eval_feature_groups.png`. The groups are the ones `evaluate.py` uses for SHAP.", ""]
    (RESULTS_DIR / "feature_groups.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved results/horizon.md, results/feature_groups.md and two charts ({(time.time() - start) / 60:.0f} min)")


if __name__ == "__main__":
    main()
