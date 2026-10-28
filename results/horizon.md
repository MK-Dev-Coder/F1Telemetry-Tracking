# Why a stop "in the next 3 laps"? The horizon

The pit stop model trained and tuned as in `train_pit_model.py` (the study's features, 2022-2024 to train, 2025 to pick the settings, 2022-2025 to train again), for each horizon, and tested on the 17 test races of 2026. A longer window is easier to hit, so each PR-AUC is set against its random guess (the share of laps with a stop coming): the lift. 95% intervals from drawing whole races again.

| Horizon | Laps with a stop coming | PR-AUC (95% interval) | Lift over a random guess | ROC-AUC |
|---|---:|---|---:|---:|
| 1 lap | 2.9% | 0.215 (0.178-0.259) | 7.41 | 0.852 |
| 2 laps | 5.8% | 0.267 (0.227-0.315) | 4.62 | 0.823 |
| 3 laps | 8.6% | 0.320 (0.270-0.388) | 3.73 | 0.817 |
| 5 laps | 14.2% | 0.432 (0.358-0.525) | 3.03 | 0.803 |

The 3-lap model is the study's (PR-AUC 0.320). Chart: `figures/eval_horizon.png`.
