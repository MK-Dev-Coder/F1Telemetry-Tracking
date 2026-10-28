# Evaluation of the race outcome model (2026 test races)

17 races, 1022 laps, 19552 driver-laps, never used to train or tune the model (version: + Pace and tyres). Written by `python evaluate_race_model.py`.

Baselines: *Position only* = how often drivers in the same position at the same stage of a race finished in the top 1/3/10 in 2022-2025; *Grid only* = the same from the starting position (it never changes during the race).

## RQ4a: how good are the chances?

### Log loss and Brier score (1000 resamples of whole races)

Lower is better for both.

| Chance | Model | Log loss | 95% CI from | 95% CI to | Brier score | LightGBM better in (% of resamples) |
|---|---|---|---|---|---|---|
| Win | LightGBM | 0.066 | 0.048 | 0.086 | 0.020 |  |
| Win | Position only | 0.087 | 0.064 | 0.112 | 0.026 | 100.0 |
| Win | Grid only | 0.096 | 0.056 | 0.159 | 0.023 | 94.3 |
| Podium | LightGBM | 0.152 | 0.127 | 0.179 | 0.047 |  |
| Podium | Position only | 0.168 | 0.149 | 0.186 | 0.053 | 97.8 |
| Podium | Grid only | 0.207 | 0.162 | 0.256 | 0.064 | 98.3 |
| Points | LightGBM | 0.346 | 0.245 | 0.465 | 0.103 |  |
| Points | Position only | 0.358 | 0.278 | 0.450 | 0.108 | 80.8 |
| Points | Grid only | 0.481 | 0.407 | 0.559 | 0.154 | 100.0 |

## RQ4b: when does it know?

| Stage | Laps | Win log loss: LightGBM | Win log loss: Position only | Win log loss: Grid only | Favourite wins (%): LightGBM | Favourite wins (%): Position only | Favourite wins (%): Grid only | Finish error: LightGBM | Finish error: Current position |
|---|---|---|---|---|---|---|---|---|---|
| Start (0-10%) | 94 | 0.104 | 0.112 | 0.090 | 51.1 | 51.1 | 70.2 | 2.916 | 3.149 |
| 10-25% | 152 | 0.103 | 0.123 | 0.092 | 41.4 | 41.4 | 71.1 | 2.563 | 2.889 |
| 25-50% | 253 | 0.078 | 0.110 | 0.094 | 56.5 | 51.4 | 72.7 | 2.129 | 2.424 |
| 50-75% | 258 | 0.057 | 0.086 | 0.094 | 71.3 | 69.8 | 77.5 | 1.705 | 1.802 |
| 75-90% | 155 | 0.029 | 0.036 | 0.099 | 91.0 | 91.0 | 77.4 | 1.060 | 1.105 |
| Last 10% | 110 | 0.013 | 0.012 | 0.111 | 99.1 | 98.2 | 77.3 | 0.357 | 0.337 |

![By stage](figures/race_by_stage.png)

![The winner's chance](figures/race_winner_chance.png)

![One race](figures/race_win_chance_2026_17.png)

## RQ4c: can the percentages be trusted?

| Chance | Model | Calibration error (ECE) | Average chance given (%) | Really happened (%) |
|---|---|---|---|---|
| Win | LightGBM | 0.015 | 5.2 | 5.2 |
| Win | Position only | 0.014 | 5.2 | 5.2 |
| Podium | LightGBM | 0.013 | 15.7 | 15.7 |
| Podium | Position only | 0.016 | 15.7 | 15.7 |
| Points | LightGBM | 0.030 | 52.2 | 52.1 |
| Points | Position only | 0.031 | 52.1 | 52.1 |

![Calibration](figures/race_calibration.png)

## RQ4d: are the SHAP reasons sound?

### Do the reasons follow racing logic? (12 of 15 agree for all three chances)

Spearman correlation between each feature's value and its SHAP value (weak = |r| < 0.1; unused = the model never uses it, so it has no direction).

| Feature | Racing logic: a higher value means | r (win) | r (podium) | r (points) | Agrees |
|---|---|---|---|---|---|
| GapAhead | less likely | 0.01 | 0.03 | 0.10 | weak |
| GapBehind | more likely | 0.32 | 0.17 | 0.20 | yes |
| PlacesToK | less likely | -0.71 | -0.83 | -0.90 | yes |
| MarginK | more likely | 0.59 | 0.35 | 0.79 | yes |
| MarginKPerLap | more likely | 0.94 | 0.96 | 0.99 | yes |
| GridToK | less likely | -0.66 | -0.75 | -0.90 | yes |
| AdjustedPlacesToK | less likely | -0.89 | -0.90 | -0.83 | yes |
| AdjustedMarginK | more likely | -0.14 | 0.15 | 0.80 | no |
| StopsVsAhead | more likely | -0.09 | -0.11 | -0.05 | no |
| StopsVsLeader | more likely | 0.43 | 0.43 | 0.39 | yes |
| Pace | less likely | -0.90 | -0.89 | -0.86 | yes |
| PaceVsAhead | less likely | -0.76 | -0.79 | -0.84 | yes |
| PaceVsBehind | less likely | -0.88 | -0.89 | -0.92 | yes |
| PaceVsLeader | less likely | -0.70 | -0.68 | -0.55 | yes |
| TyreAgeVsAhead | less likely | -0.45 | -0.54 | -0.54 | yes |

### What it leans on

| index | Win | Podium | Points |
|---|---|---|---|
| PlacesToK | 0.020 | 0.062 | 0.095 |
| MarginKPerLap | 0.016 | 0.051 | 0.089 |
| AdjustedMarginK | 0.008 | 0.036 | 0.060 |
| MarginK | 0.007 | 0.028 | 0.064 |
| GridToK | 0.006 | 0.020 | 0.043 |
| AdjustedPlacesToK | 0.005 | 0.017 | 0.023 |
| TyreAgeVsAhead | 0.004 | 0.006 | 0.010 |
| GapAhead | 0.004 | 0.005 | 0.009 |
| Pace | 0.003 | 0.012 | 0.025 |
| TyreLife | 0.002 | 0.007 | 0.016 |
| PaceVsBehind | 0.002 | 0.006 | 0.014 |
| PaceVsAhead | 0.002 | 0.003 | 0.006 |
| PaceVsLeader | 0.001 | 0.004 | 0.004 |
| GapBehind | 0.000 | 0.003 | 0.007 |

![Importance](figures/race_shap_importance.png)

### Case studies

**The model saw the winner before the order did**

```
RUS - Austrian Grand Prix 2026 - lap 23 (P2): 59% chance to win (the leader on this lap: ANT). Actual: finished P1. (The model saw the winner before the order did)
    +13%  1 place off the lead
    +11%  3.5 s clear of P2 once the stops still to make are counted
     +8%  needs 0.2 s a lap to catch P1
     +8%  started P1
     +7%  tyres 19 laps fresher than P1's
```

**The biggest upset**

```
ANT - Italian Grand Prix 2026 - lap 29 (P6): 5% chance to win (the leader on this lap: RUS). Actual: finished P1. (The biggest upset)
     +1%  tyres 25 laps fresher than P5's
     +1%  tyres 1 lap old
   -0.8%  5 places off the lead
   -0.5%  14.3 s behind P1 once the stops still to make are counted
   +0.5%  1.2 s a lap quicker than the field (last 5 green laps)
```

## RQ4e: is it fast enough live?

One lap (every driver's chances fitted together, and the SHAP reasons for the win, podium and points chances): median 184.2 ms, 95% of laps under 203.9 ms, slowest 231.2 ms (200 laps).
