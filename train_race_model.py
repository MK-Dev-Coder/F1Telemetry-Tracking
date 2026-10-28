"""Train the race outcome model: each driver's chance to win, finish on the podium and score points, lap by lap.

    python train_race_model.py   -> models/race_model.txt + .json, models/race_background.parquet (SHAP's average
                                    driver), data/race_results.parquet, data/race_model_results.csv,
                                    data/race_test_predictions.parquet (read by evaluate_race_model.py)
Train on 2022-2024, pick the settings and the features on 2025, retrain on 2022-2025, test on 2026.
"""
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

import features
import race_outcome as ro
import train_tyre_model as tyre
from config import DATA_DIR, MODELS_DIR

LGBM_BASE = dict(objective="binary", learning_rate=0.05, n_estimators=3000, subsample=0.8, subsample_freq=1,
                 colsample_bytree=0.8, reg_lambda=1.0, random_state=42, verbose=-1)
# Every lap is repeated once per k (20 times), so a leaf needs thousands of rows to hold more than a few races
LGBM_TRIES = [dict(num_leaves=7, min_child_samples=6000), dict(num_leaves=15, min_child_samples=2000),
              dict(num_leaves=15, min_child_samples=6000), dict(num_leaves=31, min_child_samples=6000)]
BASELINES = ["Grid only", "Position only"]


# Recipe 1: every lap with its features and the driver's final result (results: from ro.build_results or
# ro.load_results)
def load_rows(results):
    rows = ro.add_features(tyre.replay_laps(), results)
    rows = rows[rows["Final"].notna() & rows["Position"].notna() & features.in_study(rows)].reset_index(drop=True)
    return rows


# Recipe 2: the chances for every lap of a set of races, fitted together lap by lap (or as they come)
def predict(model, feats, rows, fit=True):
    raw = model.predict_proba(ro.stack(rows, feats, range(1, ro.MAX_CARS + 1)))[:, 1].reshape(len(rows), ro.MAX_CARS)
    return ro.by_lap(raw, rows, fit)


def scores(rows, p):
    """Log loss and Brier score for the win, podium and points chances, how often the favourite wins,
    and how far the likely finish (the median) is from the real one."""
    out = {}
    for name, k in ro.SHOWN.items():
        y, q = (rows["Final"] <= k).astype(int).values, np.clip(p[:, k - 1], 1e-6, 1 - 1e-6)
        out[f"{name.title()} log loss"] = log_loss(y, q, labels=[0, 1])
        out[f"{name.title()} Brier"] = brier_score_loss(y, q)
    out["Mean log loss"] = np.mean([out[f"{n.title()} log loss"] for n in ro.SHOWN])
    win = pd.Series(p[:, 0], index=rows.index)
    pick = rows.loc[win.groupby([rows["RaceId"], rows["LapNumber"]]).idxmax()]
    out["Favourite wins (%)"] = 100 * (pick["Final"] == 1).mean()
    out["Finish error (places)"] = np.abs(ro.summary(p)["likely"] - rows["Final"].values).mean()
    return out


# Recipe 3: two simple baselines - a table of how often drivers in the same spot finished in the top k
class Table:
    def __init__(self, key):
        self.key = key

    def keys(self, rows):
        if self.key == "Grid":
            return rows["Grid"].clip(upper=20).astype(int).astype(str)
        return rows["Position"].clip(upper=20).astype(int).astype(str) + "|" + (rows["RaceProgress"].clip(0, 0.999) * 10).astype(int).astype(str)

    def fit(self, rows):
        top = rows["Final"].values[:, None] <= np.arange(1, ro.MAX_CARS + 1)[None, :]
        prior = np.minimum(np.arange(1, ro.MAX_CARS + 1) / 20, 1)
        groups = pd.DataFrame(top).groupby(self.keys(rows).values)
        self.table = (groups.sum() + 5 * prior) / (groups.size().values[:, None] + 5)       # smoothed
        self.prior = prior
        return self

    def predict(self, rows):
        found = self.table.reindex(self.keys(rows).values)
        return found.fillna(pd.Series(self.prior)).values


def tune(train, tune_rows, feats):
    X, y = ro.stack(train, feats), ro.answers(train)
    Xt, yt = ro.stack(tune_rows, feats), ro.answers(tune_rows)
    best = None
    for tries in LGBM_TRIES:
        start = time.time()
        model = lgb.LGBMClassifier(**LGBM_BASE, **tries, monotone_constraints=[1] + [0] * len(feats))
        model.fit(X, y, eval_X=(Xt,), eval_y=(yt,), callbacks=[lgb.early_stopping(100, verbose=False)])
        score = scores(tune_rows, predict(model, feats, tune_rows))["Mean log loss"]
        print(f"   {tries}: {model.best_iteration_} trees, mean log loss on 2025 = {score:.4f} ({time.time() - start:.0f} s)")
        if best is None or score < best[0]:
            best = (score, tries, model)
    return best


def fit(rows, feats, tries, trees):
    model = lgb.LGBMClassifier(**{**LGBM_BASE, "n_estimators": trees}, **tries, monotone_constraints=[1] + [0] * len(feats))
    return model.fit(ro.stack(rows, feats), ro.answers(rows))


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    results, both = ro.build_results()
    same, winners = both["Final"] == both["Final_openf1"], both[both["Final"] == 1]
    print(f"Results: {results['RaceId'].nunique()} races ({(results.drop_duplicates('RaceId')['Source'] == 'official').sum()} "
          f"official, the rest OpenF1). Where both exist ({both['RaceId'].nunique()} races), OpenF1's order at the flag "
          f"matches the official result for {same.mean():.1%} of drivers and {(winners['Final_openf1'] == 1).mean():.0%} "
          "of winners (the rest: penalties given after the race).")
    rows = load_rows(results)
    # The pit stop cost of each track comes from the races used for training only (2022-24 to tune, 2022-25 at the end)
    tune_loss, final_loss = ro.pit_loss_table(rows[rows["Year"] <= 2024]), ro.pit_loss_table(rows[rows["Year"] <= 2025])
    before = ro.add_pit_adjusted(rows.copy(), tune_loss)
    train, tune_rows = (before[before["Year"] <= 2024].reset_index(drop=True), before[before["Year"] == 2025].reset_index(drop=True))
    test = ro.add_pit_adjusted(rows[rows["Year"] == 2026].copy(), final_loss).reset_index(drop=True)
    print(f"Laps: train {len(train)} ({train['RaceId'].nunique()} races), tune {len(tune_rows)}, test {len(test)} "
          f"({test['RaceId'].nunique()} races). A pit stop costs {final_loss['overall']:.0f} s on average.")

    # The settings are picked once, with every feature; then each feature group is added in turn,
    # with and without fitting the chances together
    every = [f for group in ro.GROUPS.values() for f in group]
    print("Tuning on 2025 (all features):")
    _, tries, tuned = tune(train, tune_rows, every)
    trees = tuned.best_iteration_
    versions, feats = {}, []
    for group, names in ro.GROUPS.items():
        feats = feats + names
        model = tuned if feats == every else fit(train, feats, tries, trees)
        name = f"+ {group}" if versions else group
        for together in (True, False):
            versions[(name, together)] = {"feats": list(feats), "model": model, "together": together,
                                          "tune": scores(tune_rows, predict(model, feats, tune_rows, fit=together))}
        print(f"   {name:20s} mean log loss on 2025 = {versions[(name, True)]['tune']['Mean log loss']:.4f} fitted together, "
              f"{versions[(name, False)]['tune']['Mean log loss']:.4f} as they come")
    # The chances are always fitted together: the screen needs one winner, three podium places and ten points places
    # (the raw scores are kept to show what it costs). Then the simplest version within 0.5% of the best on 2025
    # (fewer features: less to go wrong on new tracks and new rules).
    fitted = {n: v for (n, f), v in versions.items() if f}
    best = min(v["tune"]["Mean log loss"] for v in fitted.values())
    chosen = min((n for n, v in fitted.items() if v["tune"]["Mean log loss"] <= best * 1.005),
                 key=lambda n: (len(fitted[n]["feats"]), fitted[n]["tune"]["Mean log loss"]))
    feats, together = fitted[chosen]["feats"], True
    chosen = (chosen, True)
    print(f"Picked on 2025: {chosen[0]}, fitted together")

    # Retrain on 2022-2025 and test on 2026
    both = ro.add_pit_adjusted(rows[rows["Year"] <= 2025].copy(), final_loss).reset_index(drop=True)
    final = fit(both, feats, tries, int(trees * 1.1))
    tests = {"LightGBM": predict(final, feats, test, fit=together)}
    for name, key in zip(BASELINES, ["Grid", "Position"]):
        tests[name] = ro.by_lap(Table(key).fit(both).predict(test), test)
    results = pd.DataFrame({name: scores(test, p) for name, p in tests.items()}).T
    print("\nTest on 2026 (17 races the model never saw):")
    print(results.round(3).to_string())
    print("\nFeature groups on 2025 (each adds to the one before):")
    ablation = pd.DataFrame({f"{n} ({'fitted' if f else 'raw'})": v["tune"] for (n, f), v in versions.items()}).T
    print(ablation[["Mean log loss", "Win log loss", "Podium log loss", "Points log loss", "Favourite wins (%)",
                    "Finish error (places)"]].round(4).to_string())
    pd.concat([results.assign(Set="test 2026"), ablation.assign(Set="tune 2025")]).to_csv(DATA_DIR / "race_model_results.csv")

    importance = pd.Series(final.booster_.feature_importance("gain"), index=["K"] + feats).drop("K")
    print("Most useful features (gain):", (importance / importance.sum()).nlargest(8).round(3).to_dict())
    final.booster_.save_model(ro.MODEL_FILE)
    info = {"version": chosen[0], "features": feats, "settings": tries, "trees": int(trees * 1.1), "trained_on": "2022-2025",
            "max_k": ro.MAX_K, "fit_together": together, "pit_loss": final_loss}
    ro.INFO_FILE.write_text(json.dumps(info, indent=1), encoding="utf-8")
    ro.make_background(both, feats).to_parquet(ro.BACKGROUND_FILE, index=False)
    out = test[features.KEEP + ["LapNumber", "Position", "Grid", "Final", "RaceProgress", "LapsLeft"]].copy()
    for name, p in tests.items():
        for target, k in ro.SHOWN.items():
            out[f"{name}|{target}"] = p[:, k - 1]
        out[f"{name}|likely"], out[f"{name}|expected"] = ro.summary(p)["likely"], ro.summary(p)["expected"]
    out.to_parquet(DATA_DIR / "race_test_predictions.parquet", index=False)
    print("saved models/race_model.txt + .json + race_background.parquet, data/race_results.parquet, data/race_model_results.csv,",
          "data/race_test_predictions.parquet")


if __name__ == "__main__":
    main()
