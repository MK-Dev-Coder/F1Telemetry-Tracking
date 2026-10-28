# Evaluation (2026 test races)

17 races, 18953 laps, never used to train or tune the models. Alert line: 29% (picked on 2025). Written by `python evaluate.py`.

## RQ1: can the model predict pit stops?

### Scores with 95% confidence intervals (1000 resamples of whole races)

| Model | PR-AUC | 95% CI from | 95% CI to | LightGBM better in (% of resamples) |
|---|---|---|---|---|
| LightGBM | 0.320 | 0.270 | 0.388 |  |
| Logistic regression | 0.262 | 0.193 | 0.341 | 95.3 |
| Tyre age rule | 0.131 | 0.108 | 0.172 | 100.0 |
| Random guess | 0.086 |  |  |  |

### Where it works and where it fails

PR-AUC depends on how many laps are followed by a stop, so *lift* (PR-AUC / random guess) is the fair comparison between groups.

| Laps in the group | Laps | Pit soon (%) | PR-AUC | Lift over random | Precision | Recall |
|---|---|---|---|---|---|---|
| All 2026 laps | 18953 | 8.590 | 0.320 | 3.729 | 0.339 | 0.437 |
| Green flag laps | 17271 | 8.662 | 0.326 | 3.766 | 0.338 | 0.447 |
| Safety car / VSC laps | 1682 | 7.848 | 0.275 | 3.508 | 0.356 | 0.318 |
| Dry laps | 18569 | 8.266 | 0.322 | 3.897 | 0.336 | 0.455 |
| Wet laps (rain, or cars on wet tyres) | 384 | 24.219 | 0.482 | 1.992 | 0.706 | 0.129 |
| First stint | 6027 | 13.705 | 0.398 | 2.903 | 0.340 | 0.597 |
| Second stint | 7919 | 6.731 | 0.257 | 3.812 | 0.328 | 0.340 |
| Third stint or later | 5005 | 5.375 | 0.233 | 4.329 | 0.378 | 0.138 |
| First third of the race | 6717 | 8.307 | 0.257 | 3.089 | 0.287 | 0.375 |
| Middle third | 6376 | 11.810 | 0.401 | 3.399 | 0.369 | 0.572 |
| Last third | 5860 | 5.410 | 0.220 | 4.061 | 0.353 | 0.224 |
| On soft tyres | 4107 | 8.157 | 0.340 | 4.162 | 0.266 | 0.624 |
| On medium tyres | 6508 | 11.140 | 0.426 | 3.822 | 0.414 | 0.499 |
| On hard tyres | 8145 | 6.114 | 0.225 | 3.679 | 0.314 | 0.271 |

![By condition](figures/eval_by_condition.png)

### Each race

| Race | Laps | Pit soon (%) | PR-AUC | Lift over random | Precision | Recall |
|---|---|---|---|---|---|---|
| R01 Melbourne | 967 | 7.135 | 0.213 | 2.991 | 0.233 | 0.304 |
| R02 Shanghai | 905 | 5.967 | 0.453 | 7.586 | 0.531 | 0.315 |
| R03 Suzuka | 1079 | 6.673 | 0.540 | 8.094 | 0.565 | 0.722 |
| R04 Miami | 1020 | 5.882 | 0.495 | 8.414 | 0.355 | 0.550 |
| R05 Montreal | 1178 | 7.216 | 0.253 | 3.500 | 0.191 | 0.576 |
| R06 Monte Carlo | 1366 | 7.540 | 0.234 | 3.097 | 0.279 | 0.165 |
| R07 Catalunya | 1193 | 11.819 | 0.586 | 4.958 | 0.621 | 0.454 |
| R08 Spielberg | 1301 | 9.454 | 0.587 | 6.213 | 0.464 | 0.675 |
| R09 Silverstone | 1063 | 13.641 | 0.263 | 1.927 | 0.314 | 0.400 |
| R10 Spa-Francorchamps | 847 | 7.792 | 0.459 | 5.895 | 0.418 | 0.773 |
| R11 Hungaroring | 1388 | 9.510 | 0.368 | 3.868 | 0.360 | 0.553 |
| R12 Zandvoort | 1304 | 9.893 | 0.383 | 3.872 | 0.471 | 0.256 |
| R13 Monza | 1025 | 2.634 | 0.119 | 4.515 | 0.104 | 0.259 |
| R14 Madring | 1084 | 6.919 | 0.233 | 3.361 | 0.254 | 0.387 |
| R15 Baku | 947 | 7.920 | 0.319 | 4.032 | 0.199 | 0.493 |
| R16 Kuala Lumpur | 1072 | 18.377 | 0.466 | 2.534 | 0.465 | 0.168 |
| R17 Singapore | 1214 | 6.178 | 0.434 | 7.025 | 0.309 | 0.720 |

![By race](figures/eval_by_race.png)

### Can the percentages be trusted?

| Model | Brier score | Calibration error (ECE) | Average chance given (%) | Laps really before a stop (%) | Top 10% of laps: chance given (%) | Top 10% of laps: really pitted (%) |
|---|---|---|---|---|---|---|
| LightGBM | 0.069 | 0.023 | 10.716 | 8.590 | 45.242 | 34.968 |
| Logistic regression | 0.073 | 0.028 | 10.920 | 8.590 | 40.878 | 31.698 |

![Calibration](figures/eval_calibration.png)

### Warnings before real stops

561 real stops in 2026 (with laps before them): 53.5% warned 1-3 laps before. For those, the warning had been on without a break for a median of 4 laps (laps: stops = 1: 66, 2: 32, 3: 43, 4: 32, 5: 26, 6: 29, 7: 14, 8: 15, 9: 9, 10: 34; 10 means 10 or more).

| Kind of stop | Stops | Warned 1-3 laps before (%) |
|---|---|---|
| Early: tyres under 8 laps old | 22 | 0.0 |
| Normal stop | 319 | 63.3 |
| Under the safety car / VSC | 212 | 46.2 |
| Wet tyres | 8 | 0.0 |
| All stops | 561 | 53.5 |

| False alarm | Alerts | Share (%) |
|---|---|---|
| All false alarms | 1388 | 100.0 |
| Stop came 4-6 laps later (a little early) | 464 | 33.4 |
| Stop came 7+ laps later | 713 | 51.4 |
| No more stops in the race | 211 | 15.2 |

![Warnings](figures/eval_warnings.png)

## RQ2: are the SHAP reasons sound? (checked without human participants)

### Do the reasons follow F1 logic? (5 of 6 with one clear direction agree)

| Feature (15 biggest) | Higher value -> pit | Spearman r | F1 logic says | Agrees |
|---|---|---|---|---|
| LapsInStint | more likely | 0.956 | more | yes |
| Compound | most against: HARD, most for: SOFT |  | - | - |
| DryCompoundsUsed | less likely | -0.809 | less | yes |
| TyreLife | more likely | 0.940 | more | yes |
| RaceProgress | less likely | -0.409 | no single direction | - |
| Circuit | most against: Singapore, most for: Spa-Francorchamps |  | - | - |
| LapsLeft | weak (no clear effect) | 0.064 | no single direction | - |
| PaceVsField | more likely | 0.805 | no single direction | - |
| TyreWear | more likely | 0.944 | more | yes |
| FieldStopsLast2 | more likely | 0.875 | more | yes |
| Stint | less likely | -0.920 | no single direction | - |
| GapBehind | weak (no clear effect) | 0.046 | more | weak |
| GapAhead | more likely | 0.846 | no single direction | - |
| GapToLeader | more likely | 0.158 | no single direction | - |
| AirTemp | more likely | 0.631 | no single direction | - |

### The main reason before real stops

On the 132 laps under a safety car / VSC that came before a stop, the safety car was among the top 3 reasons 27% of the time.

| Kind of reason | Top reason (%) |
|---|---|
| Tyres | 71.4 |
| Other cars' stops | 15.8 |
| Track | 12.0 |
| Safety car / VSC | 0.3 |
| Pace | 0.3 |
| Gaps and position | 0.3 |

![Reasons](figures/eval_reasons_before_stops.png)

### Do the explanations stay steady from lap to lap?

| Pairs of laps | Same top reason (%) | Top 3 reasons shared (%) | Order of all reasons (median Spearman r) |
|---|---|---|---|
| Same driver, next lap | 80.74 | 84.33 | 0.96 |
| Two random laps (for comparison) | 18.09 | 36.21 | 0.78 |

### Do the reasons really matter? (2099 alert laps)

Each lap's top reasons are set to a typical value (median, or the most common tyre and track) and the model is asked again; the same with reasons picked at random.

| Reasons taken away | Drop: top reasons taken away (points) | Drop: random reasons taken away (points) | Still above the alert line (%) |
|---|---|---|---|
| 1 | 23.4 | 3.3 | 20.3 |
| 2 | 31.8 | 6.4 | 4.8 |
| 3 | 36.7 | 9.6 | 0.9 |
| 4 | 39.0 | 12.3 | 0.1 |
| 5 | 40.2 | 14.9 | 0.0 |

![Faithfulness](figures/eval_faithfulness.png)

## RQ3: does it work live?

| Where | What is timed | Median (ms) | 95% of laps under (ms) | Slowest (ms) | Times |
|---|---|---|---|---|---|
| Replay (2026) | pit chance + SHAP + tyre pick, every driver on one lap | 6.32 | 7.22 | 67.42 | 1024 |
| Live F1 25 | after each lap: all laps so far as rows, features, both models, every car's SHAP reasons | 76.39 | 88.00 | 88.11 | 13 |
| Reading F1 25 packets | decoding one UDP packet from the game (in microseconds) | 0.02 |  |  | 224647 |

F1 25 sent 224647 packets in 39 minutes (95 per second); reading one takes about 22 microseconds, so the app could read about 45276 per second. The page asks for news every second, so a prediction is on screen at most about a second after a lap ends.

![Speed](figures/eval_speed.png)
