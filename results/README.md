# Results

The tables behind the dissertation's numbers. The scripts write most of them to `data/`, which isn't in this
repository (it holds Formula 1's data), so copies are kept here. After running a script again, copy its file here
to update it. The models' numbers are on the study's test races (2026, rounds 1–17), which no model saw while it
was built; the race preview is tested race by race on 2025 and 2026, and the source check is on the two races we
have from both sources.

| File | What it is | Made by |
|---|---|---|
| `model_results.csv` | The pit stop model (versions v1 and v2) against a random guess, the tyre-age rule and logistic regression: PR-AUC, ROC-AUC, precision, recall, stops warned | `python train_pit_model.py` |
| `tyre_model_results.csv` | The tyre choice model against "always the most common", a strategist rule and logistic regression | `python train_tyre_model.py` |
| `evaluation.md` | The evaluation of the research questions: confidence intervals, calibration, stops by condition, false alarms, the SHAP checks and speed | `python evaluate.py` |
| `race_model_results.csv` | The race outcome model (win, podium, points) against "grid only" and "position only" | `python train_race_model.py` |
| `race_evaluation.md` | The race outcome model's evaluation: intervals, calibration, by stage of the race, SHAP | `python evaluate_race_model.py` |
| `rolling_2026.md`, `rolling_2026.csv` | Every model trained again before each 2026 race, with the 2026 races before it | `python evaluate_rolling.py` |
| `preview_evaluation.md` | The race preview, before each race of 2025 and 2026 | `python race_preview.py` |
| `battle_evaluation.md` | The undercut threat and the battle forecast | `python battle.py` |
| `source_comparison.csv` | FastF1 against OpenF1 on two races (how far the two sources agree) | `python compare_sources.py` |
| `horizon.md`, `horizon.csv` | Why "a stop in the next 3 laps": the model for 1, 2, 3 and 5 laps | `python evaluate_design.py` (writes here) |
| `feature_groups.md`, `feature_groups.csv` | What each group of features adds (the model without each group) | `python evaluate_design.py` (writes here) |
| `cloud.md` | The cloud data layer: a new PC's download from S3 (files, size, time, every file the same) | `python aws_sync.py check` (writes here, once AWS is set up) |

The charts are in `figures/`.
