import json

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import features
from config import DATA_DIR, FIGURES_DIR, MODELS_DIR

matplotlib.use("Agg")                         # save charts to files, don't open windows
import matplotlib.pyplot as plt               # noqa: E402

WET = "INTER/WET"                             # intermediates and full wets together (only 11 full-wet stops)
CLASSES = ["SOFT", "MEDIUM", "HARD", WET]
CATEGORIES = ["Compound", "Circuit"]
FEATURES = CATEGORIES + [
    "TyreLife", "Stint", "UsedSoft", "UsedMedium", "UsedHard", "DryCompoundsUsed",            # this driver's tyres
    "LapNumber", "LapsLeft", "RaceProgress",                                                  # race progress
    "FieldOnSoft", "FieldOnMedium", "FieldOnHard", "FieldOnWetTyres",                         # what the field runs
    "FitSoftRecent", "FitMediumRecent", "FitHardRecent", "FitWetRecent",                      # what others just fitted
    "SafetyCar", "VSC", "Rainfall", "TrackTemp", "AirTemp", "Humidity", "Position",           # conditions
]
LGBM_BASE = dict(objective="multiclass", num_class=len(CLASSES), metric="multi_logloss", learning_rate=0.03,
                 n_estimators=3000, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                 random_state=42, verbose=-1)
LGBM_TRIES = [dict(num_leaves=7, min_child_samples=20), dict(num_leaves=15, min_child_samples=20),
              dict(num_leaves=7, min_child_samples=50), dict(num_leaves=15, min_child_samples=50)]


def tyre_class(compound):
    return compound.replace({"INTERMEDIATE": WET, "WET": WET})


# Recipe 1: one row per real pit stop, with everything known as the car comes in.
# The target is the compound on the next lap.
def stop_rows():
    laps = add_tyre_features(features.build())
    stops = laps[laps["IsStop"] & laps["Next"].isin(CLASSES)].copy()
    for col in CATEGORIES:
        stops[col] = stops[col].astype("category")
    return stops.reset_index(drop=True)


# The tyre model's features for every lap (so the app can say "if he pits now: ...")
def add_tyre_features(laps):
    by_driver = laps.groupby(["RaceId", "Driver"])
    laps["Next"] = tyre_class(by_driver["Compound"].shift(-1))
    for compound, name in [("SOFT", "UsedSoft"), ("MEDIUM", "UsedMedium"), ("HARD", "UsedHard")]:
        laps[name] = (laps["Compound"] == compound).groupby([laps["RaceId"], laps["Driver"]]).cummax().astype(int)
        laps[f"FieldOn{compound.title()}"] = (laps["Compound"] == compound).groupby(
            [laps["RaceId"], laps["LapNumber"]]).transform("mean")

    # What the other cars fitted at their stops in the last 3 laps (teams often copy each other)
    stops = laps[laps["IsStop"] & laps["Next"].isin(CLASSES)]
    fits = stops.groupby(["RaceId", "LapNumber", "Next"]).size().unstack(fill_value=0).reindex(columns=CLASSES, fill_value=0)
    grid = laps[["RaceId", "LapNumber"]].drop_duplicates().set_index(["RaceId", "LapNumber"]).sort_index()
    fits = grid.join(fits).fillna(0)
    recent = fits.groupby(level="RaceId").transform(lambda s: s.rolling(3, min_periods=1).sum().shift(1)).fillna(0)
    recent.columns = ["FitSoftRecent", "FitMediumRecent", "FitHardRecent", "FitWetRecent"]
    return laps.join(recent, on=["RaceId", "LapNumber"])


# Every lap of every race with all the features of both models (built once, then read back from a file; built
# again when all_laps.parquet is newer). Used by the app and by the race outcome model.
REPLAY_FILE = DATA_DIR / "replay_laps.parquet"


def replay_laps():
    if REPLAY_FILE.exists() and REPLAY_FILE.stat().st_mtime > features.IN_FILE.stat().st_mtime:
        return pd.read_parquet(REPLAY_FILE)
    print("Building the replay file (once)...")
    laps = add_tyre_features(features.build())
    laps.to_parquet(REPLAY_FILE, index=False)
    return laps


# Baseline 1: always the compound fitted most often in the training races
class MostCommon:
    def fit(self, rows):
        self.most = rows["Next"].mode()[0]
        return self

    def predict(self, rows):
        return np.array([self.most] * len(rows))


# Baseline 2: a simple strategist's rule
def strategist_rule(rows):
    picks = []
    for _, r in rows.iterrows():
        unused = [c for c, used in [("HARD", r["UsedHard"]), ("MEDIUM", r["UsedMedium"]), ("SOFT", r["UsedSoft"])] if not used]
        if r["Rainfall"] or r["FieldOnWetTyres"] >= 0.5:
            picks.append(WET)                                        # it's wet: wet-weather tyres
        elif r["Compound"] in ("INTERMEDIATE", "WET"):
            picks.append("MEDIUM")                                   # drying track: back to slicks
        elif r["LapsLeft"] <= 20:
            picks.append("SOFT" if "SOFT" in unused or r["DryCompoundsUsed"] >= 2 else "MEDIUM")   # short sprint to the end
        elif r["DryCompoundsUsed"] < 2:
            picks.append(unused[0] if unused else "HARD")            # F1 rule: use a second compound
        else:
            picks.append("MEDIUM" if r["LapsLeft"] <= 35 else "HARD")
    return np.array(picks)


# Baseline 3: logistic regression
def logistic_model():
    numbers = [f for f in FEATURES if f not in CATEGORIES]
    prepare = ColumnTransformer([
        ("numbers", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numbers),
        ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORIES)])
    return make_pipeline(prepare, LogisticRegression(max_iter=5000))


# Recipe 2: LightGBM - try a few settings, keep the one with the best log loss on 2025
def tune_lightgbm(train, tune):
    best = None
    y_train, y_tune = train["Next"].map(CLASSES.index), tune["Next"].map(CLASSES.index)
    for tries in LGBM_TRIES:
        model = lgb.LGBMClassifier(**LGBM_BASE, **tries)
        model.fit(train[FEATURES], y_train, eval_X=(tune[FEATURES],), eval_y=(y_tune,),
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        loss = log_loss(y_tune, model.predict_proba(tune[FEATURES]), labels=range(len(CLASSES)))
        print(f"   {tries}: {model.best_iteration_} trees, log loss on 2025 = {loss:.3f}, "
              f"accuracy {accuracy_score(y_tune, model.predict(tune[FEATURES])):.3f}")
        if best is None or loss < best[0]:
            best = (loss, tries, model)
    return best


def scores(y, pick, proba=None):
    present = [c for c in CLASSES if c in set(y)]
    out = {"Accuracy": accuracy_score(y, pick), "Macro F1": f1_score(y, pick, labels=present, average="macro")}
    if proba is not None:
        top2 = np.argsort(-proba, axis=1)[:, :2]
        out["Top-2 accuracy"] = np.mean([CLASSES.index(t) in row for t, row in zip(y, top2)])
        out["Log loss"] = log_loss([CLASSES.index(t) for t in y], proba, labels=range(len(CLASSES)))
    return out


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    stops = stop_rows()
    stops = stops[stops["Next"].isin(CLASSES) & features.in_study(stops)]
    train, tune, test = stops[stops["Year"] <= 2024], stops[stops["Year"] == 2025], stops[stops["Year"] == 2026]
    both = pd.concat([train, tune])
    print(f"Pit stops: train {len(train)}, tune {len(tune)}, test {len(test)}")
    print("Fitted at 2026 stops:", test["Next"].value_counts().to_dict())

    print("LightGBM - tuning on 2025:")
    _, tries, tuned = tune_lightgbm(train, tune)
    trees = int(tuned.best_iteration_ * 1.1)
    final = lgb.LGBMClassifier(**{**LGBM_BASE, "n_estimators": trees}, **tries).fit(
        both[FEATURES], both["Next"].map(CLASSES.index))
    proba = final.predict_proba(test[FEATURES])
    lgbm_pick = np.array(CLASSES)[proba.argmax(axis=1)]

    logit = logistic_model().fit(both[FEATURES], both["Next"])
    logit_proba = pd.DataFrame(logit.predict_proba(test[FEATURES]), columns=logit.classes_).reindex(columns=CLASSES, fill_value=0).values

    y = test["Next"].values
    results = pd.DataFrame({
        "Always the most common": scores(y, MostCommon().fit(both).predict(test)),
        "Strategist rule": scores(y, strategist_rule(test)),
        "Logistic regression": scores(y, logit.predict(test[FEATURES]), logit_proba),
        "LightGBM": scores(y, lgbm_pick, proba),
    }).T
    print("\nTest on 2026 (577 pit stops the model never saw):")
    print(results.round(3).to_string())
    results.to_csv(DATA_DIR / "tyre_model_results.csv")

    per_class = pd.DataFrame({c: {"Stops": int((y == c).sum()), "Found (recall)": ((y == c) & (lgbm_pick == c)).sum() / max((y == c).sum(), 1),
                                  "Predicted": int((lgbm_pick == c).sum())} for c in CLASSES}).T
    print("\nLightGBM per compound on 2026:")
    print(per_class.round(2).to_string())
    importance = pd.Series(final.booster_.feature_importance("gain"), index=FEATURES)
    print("Most useful features (gain):", (importance / importance.sum()).nlargest(8).round(3).to_dict())

    # Confusion matrix: what was fitted (rows) vs what the model said (columns)
    present = [c for c in CLASSES if (y == c).any()]
    matrix = confusion_matrix(y, lgbm_pick, labels=CLASSES, normalize="true")[[CLASSES.index(c) for c in present]]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(present)), [f"{c} ({(y == c).sum()})" for c in present])
    for i in range(len(present)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{matrix[i, j]:.0%}", ha="center", va="center", color="white" if matrix[i, j] > 0.5 else "black")
    ax.set_xlabel("Model's pick")
    ax.set_ylabel("Compound actually fitted (number of stops)")
    ax.set_title("Which tyre at the pit stop? - test on 2026")
    fig.savefig(FIGURES_DIR / "tyre_confusion_2026.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    final.booster_.save_model(MODELS_DIR / "tyre_model.txt")
    info = {"features": FEATURES, "classes": CLASSES, "settings": tries, "trees": trees, "trained_on": "2022-2025",
            "categories": {c: list(both[c].cat.categories) for c in CATEGORIES}}
    (MODELS_DIR / "tyre_model.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    out = test[features.KEEP + ["LapNumber", "Circuit", "Compound", "Next"]].copy()
    out["LightGBM pick"] = lgbm_pick
    for k, c in enumerate(CLASSES):
        out[f"Chance {c}"] = proba[:, k]
    out.to_parquet(DATA_DIR / "tyre_test_predictions.parquet", index=False)
    print("\nsaved models/tyre_model.txt + .json, data/tyre_model_results.csv, data/tyre_test_predictions.parquet,",
          "figures/tyre_confusion_2026.png")


if __name__ == "__main__":
    main()
