import json

import lightgbm as lgb
import matplotlib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import features
from config import DATA_DIR, FIGURES_DIR, MODELS_DIR

matplotlib.use("Agg")                         # save charts to files, don't open windows
import matplotlib.pyplot as plt               # noqa: E402

TARGET = "PitSoon"
NEW = ["NeutralStarted", "LapsUnderNeutral", "TyreAgeVsUsual"]
ALL = features.FEATURES + ["TyreAgeVsUsual"]
VERSIONS = {                                  # the winner is picked on 2025, never on 2026
    "v1 (before)": [f for f in ALL if f not in NEW],
    "v2 (+ new features)": ALL,
    "v2 without Circuit": [f for f in ALL if f != "Circuit"],
}
LGBM_BASE = dict(objective="binary", metric="average_precision", learning_rate=0.03, n_estimators=3000,
                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                 random_state=42, verbose=-1)
LGBM_TRIES = [dict(num_leaves=15, min_child_samples=50), dict(num_leaves=31, min_child_samples=50),
              dict(num_leaves=63, min_child_samples=100), dict(num_leaves=31, min_child_samples=200)]


# Recipe 1: train on 2022-2024, tune on 2025, test on 2026 (never seen while building the model)
def load():
    rows = pd.read_parquet(features.OUT_FILE)
    rows = rows[features.in_study(rows)]
    return rows[rows["Year"] <= 2024], rows[rows["Year"] == 2025], rows[rows["Year"] == 2026]


def real_stops():
    laps = pd.read_parquet(features.IN_FILE).sort_values(["RaceId", "Driver", "LapNumber"]).reset_index(drop=True)
    features.mark_stops(laps, laps.groupby(["RaceId", "Driver"]))
    return laps[laps["IsStop"]]


# Recipe 2: the usual tyre age at a pit stop, for each track and compound (falls back to the
# compound, then to all stops, for a new track like Madrid 2026)
def usual_table(stops):
    return (stops.groupby(["Circuit", "Compound"])["TyreLife"].median(),
            stops.groupby("Compound")["TyreLife"].median(), stops["TyreLife"].median())


def usual_for(rows, table):
    track, compound, overall = table
    key = pd.MultiIndex.from_arrays([rows["Circuit"].astype(str), rows["Compound"].astype(str)])
    usual = pd.Series(track.reindex(key).values, index=rows.index)
    return usual.fillna(rows["Compound"].astype(str).map(compound)).fillna(overall)


# TyreAgeVsUsual = tyre age / usual age at a stop (1.0 = the age teams usually stop at).
# A race used for training gets its "usual" from the OTHER races only, so it can't peek at its own stops.
def add_usual(rows, stops, fit_races):
    fit_stops = stops[stops["RaceId"].isin(fit_races)]
    usual = usual_for(rows, usual_table(fit_stops))
    for race in set(rows["RaceId"]) & set(fit_races):
        mine = rows["RaceId"] == race
        usual[mine] = usual_for(rows[mine], usual_table(fit_stops[fit_stops["RaceId"] != race]))
    return rows.assign(TyreAgeVsUsual=rows["TyreLife"] / usual)


# Baseline 1, a simple rule: the older the tyres compared with the usual age, the higher the score
class TyreAgeRule:
    def fit(self, stops):
        self.table = usual_table(stops)
        return self

    def predict(self, rows):
        return (rows["TyreLife"] / usual_for(rows, self.table)).fillna(0).values


# Baseline 2: logistic regression (a straight-line model)
def logistic_model(feats):
    categories = [c for c in features.CATEGORIES if c in feats]
    numbers = [f for f in feats if f not in categories]
    prepare = ColumnTransformer([
        ("numbers", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numbers),
        ("categories", OneHotEncoder(handle_unknown="ignore"), categories)])
    return make_pipeline(prepare, LogisticRegression(max_iter=3000))


# Recipe 3: LightGBM - try a few settings, keep the one that scores best on 2025
def tune_lightgbm(train, tune, feats):
    best = None
    for tries in LGBM_TRIES:
        model = lgb.LGBMClassifier(**LGBM_BASE, **tries)
        model.fit(train[feats], train[TARGET], eval_X=(tune[feats],), eval_y=(tune[TARGET],),
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        score = average_precision_score(tune[TARGET], model.predict_proba(tune[feats])[:, 1])
        print(f"   {tries}: {model.best_iteration_} trees, PR-AUC on 2025 = {score:.3f}")
        if best is None or score > best[0]:
            best = (score, tries, model)
    return best


# The alert line: the probability above which we say "pit soon", picked on 2025 for the best F1
def best_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return float(thresholds[f1[:-1].argmax()])


# What fans care about: of the real pit stops, how many got a warning 1-3 laps before?
def stops_warned(rows, p, threshold, stops):
    alerts = rows[["RaceId", "Driver", "LapNumber"]].assign(alert=p >= threshold)
    pairs = stops[["RaceId", "Driver", "LapNumber"]].rename(columns={"LapNumber": "StopLap"})
    pairs = pairs[pairs["RaceId"].isin(rows["RaceId"])].merge(alerts, on=["RaceId", "Driver"])
    pairs = pairs[(pairs["StopLap"] - pairs["LapNumber"]).between(1, features.HORIZON)]
    pairs["lead"] = (pairs["StopLap"] - pairs["LapNumber"]).where(pairs["alert"])
    per_stop = pairs.groupby(["RaceId", "Driver", "StopLap"]).agg(warned=("alert", "any"), lead=("lead", "max"))
    return per_stop["warned"].mean(), per_stop["lead"].median(), len(per_stop)


def recall(y, p, threshold, mask=None):
    mask = slice(None) if mask is None else mask
    y, p = y[mask], p[mask]
    return ((p >= threshold) & (y == 1)).sum() / max((y == 1).sum(), 1)


def scores(y, p, threshold, rows, stops):
    alert = p >= threshold
    warned, lead, n_stops = stops_warned(rows, p, threshold, stops)
    neutral = ((rows["SafetyCar"] == 1) | (rows["VSC"] == 1)).values
    per_race = pd.Series(p, index=rows.index).groupby(rows["RaceId"]).apply(
        lambda s: average_precision_score(y[rows.index.get_indexer(s.index)], s))
    return {"PR-AUC": average_precision_score(y, p), "ROC-AUC": roc_auc_score(y, p),
            "Precision": (alert & (y == 1)).sum() / max(alert.sum(), 1), "Recall": recall(y, p, threshold),
            "Recall SC/VSC laps": recall(y, p, threshold, neutral),
            "Stops warned (%)": 100 * warned, "Warning lead (laps)": lead,
            "Worst race PR-AUC": per_race.min(), "Stops": n_stops}


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    train, tune, test = load()
    both = pd.concat([train, tune])
    stops = real_stops()
    train_races, both_races = set(train["RaceId"]), set(both["RaceId"])
    train, tune = add_usual(train, stops, train_races), add_usual(tune, stops, train_races)
    both, test = add_usual(both, stops, both_races), add_usual(test, stops, both_races)
    y = test[TARGET].values
    print(f"Rows: train {len(train)}, tune {len(tune)}, test {len(test)} - PitSoon on 2026 laps: {y.mean():.1%}")

    # Each version: pick settings and the alert line on 2025 (trained on 2022-24),
    # then retrain on 2022-25 and test on 2026
    versions = {}
    for name, feats in VERSIONS.items():
        print(f"{name} - tuning on 2025:")
        tune_score, tries, tuned = tune_lightgbm(train, tune, feats)
        trees = int(tuned.best_iteration_ * 1.1)                     # a little more data -> a few more trees
        final = lgb.LGBMClassifier(**{**LGBM_BASE, "n_estimators": trees}, **tries).fit(both[feats], both[TARGET])
        versions[name] = {"feats": feats, "tune": tune_score, "tries": tries, "trees": trees, "model": final,
                          "threshold": best_threshold(tune[TARGET], tuned.predict_proba(tune[feats])[:, 1]),
                          "p": final.predict_proba(test[feats])[:, 1]}
    chosen = max(versions, key=lambda n: versions[n]["tune"])
    best = versions[chosen]

    # Recipe 4: compare everything on 2026
    tests = {
        "Tyre age rule": (TyreAgeRule().fit(stops[stops["RaceId"].isin(both_races)]).predict(test),
                          best_threshold(tune[TARGET], TyreAgeRule().fit(stops[stops["RaceId"].isin(train_races)]).predict(tune))),
        "Logistic regression": (logistic_model(best["feats"]).fit(both[best["feats"]], both[TARGET]).predict_proba(test[best["feats"]])[:, 1],
                                best_threshold(tune[TARGET], logistic_model(best["feats"]).fit(train[best["feats"]], train[TARGET])
                                               .predict_proba(tune[best["feats"]])[:, 1])),
    }
    tests.update({f"LightGBM {name}": (v["p"], v["threshold"]) for name, v in versions.items()})
    results = pd.DataFrame({name: scores(y, p, t, test, stops) for name, (p, t) in tests.items()}).T
    results.insert(0, "PR-AUC on 2025", [None, None] + [v["tune"] for v in versions.values()])
    results.loc["Random guess", "PR-AUC"] = y.mean()
    print(f"\nTest on 2026 (17 races the model never saw). Picked on 2025: LightGBM {chosen}")
    print(results.round(3).to_string())
    results.to_csv(DATA_DIR / "model_results.csv")

    p, threshold = best["p"], best["threshold"]
    print(f"\nLightGBM {chosen}: {p[y == 1].mean():.0%} average chance on laps before a stop, "
          f"{p[y == 0].mean():.0%} on other laps. Brier score {brier_score_loss(y, p):.3f}")
    importance = pd.Series(best["model"].booster_.feature_importance("gain"), index=best["feats"])
    print("Most useful features (gain):", (importance / importance.sum()).nlargest(8).round(3).to_dict())

    # Recipe 5: save the chosen model, its alert line, the usual stint table, test predictions, a chart
    best["model"].booster_.save_model(MODELS_DIR / "pit_model.txt")
    track, compound, overall = usual_table(stops[stops["RaceId"].isin(both_races)])
    info = {"version": chosen, "features": best["feats"], "threshold": threshold, "settings": best["tries"],
            "trees": best["trees"], "trained_on": "2022-2025", "horizon_laps": features.HORIZON,
            "categories": {c: list(both[c].cat.categories) for c in features.CATEGORIES if c in best["feats"]},
            "usual_stint": {"track": {f"{c}|{k}": v for (c, k), v in track.items()},
                            "compound": compound.to_dict(), "overall": overall}}
    (MODELS_DIR / "pit_model.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    out = test[features.KEEP + ["LapNumber", "Circuit", TARGET]].copy()
    for name, (p_model, _) in tests.items():
        out[name] = p_model
    out.to_parquet(DATA_DIR / "test_predictions.parquet", index=False)

    for name, (p_model, _) in tests.items():
        precision, rec, _ = precision_recall_curve(y, p_model)
        plt.plot(rec, precision, label=f"{name} (PR-AUC {average_precision_score(y, p_model):.2f})")
    plt.axhline(y.mean(), color="grey", linestyle="--", label=f"Random guess ({y.mean():.2f})")
    plt.xlabel("Recall (share of pit stops found)")
    plt.ylabel("Precision (share of alerts that were right)")
    plt.title("Pit in the next 3 laps - test on 2026")
    plt.legend()
    plt.savefig(FIGURES_DIR / "pr_curve_2026.png", dpi=150, bbox_inches="tight")
    print("\nsaved models/pit_model.txt + .json, data/model_results.csv, data/test_predictions.parquet,",
          "figures/pr_curve_2026.png")


if __name__ == "__main__":
    main()
