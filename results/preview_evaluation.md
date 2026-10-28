# Race preview: tested race by race (2025 and 2026)

Before each of the 41 races (2025, and 2026 up to round 17 as in the rest of the study) the model was trained again on every race before it (from 2022), then asked about that race. Settings picked the same way on 2024: {'num_leaves': 7, 'min_child_samples': 400, 'n_estimators': 150}. *Form table* / *grid table*: how often drivers with the same recent average finish / grid slot finished in the top 1/3/10 before. *Rules*: the last race's winner wins; the season's points leader wins. After qualifying it uses: GridToK, Pace3, TeamPace3, SeasonWins (picked on 2024).

| Way | Win log loss | Podium log loss | Points log loss | Favourite won (%) | Winner's chance (median) | Win log loss 95% CI |
|---|---|---|---|---|---|---|
| before qualifying | 0.131 | 0.248 | 0.524 | 31.7 | 0.189 | 0.119-0.144 |
| after qualifying | 0.096 | 0.189 | 0.477 | 63.4 | 0.324 | 0.087-0.105 |
| form table | 0.147 | 0.280 | 0.564 | 22.0 | 0.145 | 0.132-0.161 |
| grid table | 0.090 | 0.200 | 0.492 | 68.3 | 0.503 | 0.074-0.109 |
| last winner (rule) |  |  |  | 36.6 |  |  |
| points leader (rule) |  |  |  | 26.8 |  |  |

![Backtest](figures/preview_backtest.png)

## Settings and after-qualifying features, tried race by race on 2024

| Tried on 2024 | Win log loss | Podium log loss | Points log loss | Favourite won (%) | Winner's chance (median) |
|---|---|---|---|---|---|
| {'num_leaves': 7, 'min_child_samples': 400, 'n_estimators': 150} | 0.1477 | 0.2903 | 0.4442 | 25.0000 | 0.1379 |
| {'num_leaves': 7, 'min_child_samples': 400, 'n_estimators': 300} | 0.1495 | 0.2990 | 0.4493 | 29.1667 | 0.1283 |
| {'num_leaves': 15, 'min_child_samples': 800, 'n_estimators': 200} | 0.1488 | 0.2967 | 0.4425 | 16.6667 | 0.1343 |
| grid and everything | 0.1288 | 0.2379 | 0.3679 | 33.3333 | 0.2427 |
| grid, pace and wins | 0.1245 | 0.2364 | 0.3651 | 37.5000 | 0.2071 |
| grid, form and pace | 0.1273 | 0.2341 | 0.3737 | 37.5000 | 0.2565 |
| grid only | 0.1270 | 0.2370 | 0.4254 | 45.8333 | 0.2496 |

Written by `python race_preview.py`.