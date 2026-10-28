# What each group of features adds

The study's pit stop model (PR-AUC 0.320 on the 2026 test races) trained again without one group of features at a time, tuned and tested the same way. A drop means the group carries information the others don't; near zero means the other features cover it. 95% intervals from drawing whole races again (the same draws for every model).

| Group left out | Features | PR-AUC without it | Change (95% interval) | Worse in |
|---|---:|---:|---|---:|
| Tyres | 7 | 0.224 | -0.097 (-0.144 to -0.066) | 100% |
| Other cars' stops | 3 | 0.288 | -0.032 (-0.056 to +0.001) | 97% |
| Gaps and position | 4 | 0.300 | -0.021 (-0.039 to +0.001) | 97% |
| Safety car / VSC | 3 | 0.319 | -0.001 (-0.021 to +0.014) | 59% |
| Weather | 5 | 0.324 | +0.004 (-0.014 to +0.023) | 32% |
| Pace | 1 | 0.324 | +0.004 (-0.018 to +0.020) | 37% |
| Track | 1 | 0.329 | +0.009 (-0.013 to +0.037) | 19% |
| Race progress | 3 | 0.357 | +0.037 (+0.000 to +0.068) | 2% |

Chart: `figures/eval_feature_groups.png`. The groups are the ones `evaluate.py` uses for SHAP.
