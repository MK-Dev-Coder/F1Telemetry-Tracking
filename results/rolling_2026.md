# How fast do the models learn the 2026 rules?

Each model is trained again before each 2026 race (rounds 2-17), with the 2026 races before it added, and tested on that race. *Rolling x3*: each 2026 lap counts 3 times. Same settings and trees as the saved models. *Change* is the rolling model minus the fixed one, with a 95% interval from 1000 resamples of the races; *Races better* counts the races where the rolling model scored better.

## Pit stop model: PR-AUC per race (higher is better)

| Model | Mean | Change | 95% CI from | 95% CI to | Races better | Races |
|---|---|---|---|---|---|---|
| Fixed | 0.387 |  |  |  |  |  |
| Rolling | 0.394 | 0.007 | -0.007 | 0.021 | 10 | 16 |
| Rolling x3 | 0.390 | 0.003 | -0.013 | 0.018 | 11 | 16 |

## Tyre model: right compound per race (higher is better)

| Model | Mean | Change | 95% CI from | 95% CI to | Races better | Races |
|---|---|---|---|---|---|---|
| Fixed | 0.725 |  |  |  |  |  |
| Rolling | 0.737 | 0.011 | -0.008 | 0.029 | 7 | 16 |
| Rolling x3 | 0.736 | 0.010 | -0.013 | 0.033 | 8 | 16 |

## Race outcome: win log loss per race (lower is better)

| Model | Mean | Change | 95% CI from | 95% CI to | Races better | Races |
|---|---|---|---|---|---|---|
| Fixed | 0.066 |  |  |  |  |  |
| Rolling | 0.065 | -0.001 | -0.002 | 0.000 | 8 | 16 |
| Rolling x3 | 0.064 | -0.001 | -0.005 | 0.003 | 9 | 16 |

## Race outcome: podium log loss per race (lower is better)

| Model | Mean | Change | 95% CI from | 95% CI to | Races better | Races |
|---|---|---|---|---|---|---|
| Fixed | 0.151 |  |  |  |  |  |
| Rolling | 0.152 | 0.001 | 0.000 | 0.002 | 6 | 16 |
| Rolling x3 | 0.153 | 0.002 | -0.001 | 0.004 | 7 | 16 |

## Race outcome: points log loss per race (lower is better)

| Model | Mean | Change | 95% CI from | 95% CI to | Races better | Races |
|---|---|---|---|---|---|---|
| Fixed | 0.351 |  |  |  |  |  |
| Rolling | 0.351 | -0.000 | -0.003 | 0.004 | 9 | 16 |
| Rolling x3 | 0.354 | 0.003 | -0.002 | 0.009 | 6 | 16 |

## Race outcome: win log loss in the first quarter of each race (lower is better)

| Model | Mean | Change | 95% CI from | 95% CI to | Races better | Races |
|---|---|---|---|---|---|---|
| Fixed | 0.102 |  |  |  |  |  |
| Rolling | 0.099 | -0.003 | -0.006 | -0.000 | 8 | 16 |
| Rolling x3 | 0.097 | -0.005 | -0.011 | 0.000 | 9 | 16 |

## Soft tyres (the tyre model picked them too rarely in 2026)

| index | Picked soft (%) | Really soft (%) |
|---|---|---|
| Fixed | 24.0 | 31.4 |
| Rolling | 26.5 | 31.4 |
| Rolling x3 | 25.8 | 31.4 |

![Rolling](figures/rolling_2026.png)

Took 4 minutes. Written by `python evaluate_rolling.py`.