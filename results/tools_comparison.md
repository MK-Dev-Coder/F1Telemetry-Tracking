# The existing tools against this project

The June proposal's comparison (its criteria, section 4.2), checked again on 29 October 2026 at each tool's own page,
with a column for this project and one more criterion: whether accuracy is reported. Usability wasn't tested with
users (no human participants), so this compares features and published evidence.

| Criterion | FastF1 / OpenF1 (data sources) | AWS F1 Insights (broadcast graphics) | f1-race-replay (desktop app) | Academic ML (research papers) | This project (web and desktop app) |
|---|---|---|---|---|---|
| Data access and granularity | High: timing, telemetry, tyres, weather. FastF1's archive needs an F1 TV login since Oct 2026; OpenF1 has every session since 2023 | High: 300 sensors per car, 1.1 million data points a second, 70 years of race data | Medium: FastF1's data | Varies, often a subset of seasons | High: 121,761 laps of 110 races (2022–2026), and the F1 25 / F1 26 game's full car telemetry |
| Predictive capability | None: data only | Full: pit window, pit stop strategy, undercut threat, battle forecast, alternative strategy | Partial: a tyre wear model (Bayesian); no pit stop or race predictions | Partial to full: pit stops [3], results [8], [18], strategy [19], [22], tyre energy [11] | Full: a pit stop in the next 3 laps, the tyre, win / podium / points, a race preview, undercut threat, battle forecast |
| Explainability | None | Limited: some graphics name their inputs; no reasons for a single prediction | None stated | Varies: SHAP [18], explainable RL [19], tyre energy [11] | Full: SHAP reasons in plain English for every prediction, checked for faithfulness and stability |
| Openness and extensibility | Open: FastF1 under the MIT licence; OpenF1's code on GitHub | Closed: no code or models published | Open: MIT licence | Varies: code sometimes released | Open: public GitHub repository, MIT licence, 187 tests on every push |
| Cost | Free; FastF1's archive now needs F1 TV; OpenF1's live data €9.90 a month | Free to watch; not available as a tool | Free | Papers, some behind a paywall | Free, on one PC (the optional AWS copy: a few cents a month) |
| Real-time capability | FastF1: after the session. OpenF1: live, about 3 s behind (paid) | Real time during races | After the session (replays) | Offline | Replays at real speed (6 ms a lap); live on the game (75 ms a lap); live real races need OpenF1's paid data |
| Target user | Developers, analysts | Broadcast viewers | Fans, hobbyists | Researchers | Fans, analysts, players of the game |
| Accuracy reported | Not applicable | None on its public page | None stated | Yes, each on its own data (not comparable) | Yes, on 17 unseen 2026 races: PR-AUC 0.32 (logistic regression 0.26, random 0.09); tyre 71% (a strategist's rule 59%) |

## What changed since the June table

- **FastF1:** since about 15 October 2026, F1's timing archive refuses downloads without an F1 TV login (FastF1 issue
  #978, open), so this project uses OpenF1 for 2023–2026 and FastF1's saved cache for 2022.
- **OpenF1** is added as a data source: free history since 2023 (a session counts as history 30 minutes after it
  ends), live data for €9.90 a month for personal use.
- **f1-race-replay** now has a Bayesian tyre wear model, so "visualisation only" is no longer exact; it still makes no
  pit stop or race predictions. Its README gives the MIT licence.
- **AWS F1 Insights** is the same kind of tool: its page lists about 20 insights, with no accuracy figures and no
  code or models.

## Sources (checked 29 October 2026)

- AWS F1 Insights: <https://aws.amazon.com/sports/f1/> (the insights, "300 sensors on each F1 race car",
  "1.1 million data points per second", "AI-powered on-screen graphics"); the June reference [6] is
  <https://aws.amazon.com/sports/f1/made-for-this/>.
- FastF1 [20]: <https://github.com/theOehrly/Fast-F1> (MIT licence) and issue #978,
  <https://github.com/theOehrly/Fast-F1/issues/978> (the archive returns HTTP 401 since about 15 October 2026).
- f1-race-replay [21]: <https://github.com/IAmTomShaw/f1-race-replay> (its README: features, MIT licence;
  `src/bayesian_tyre_model.py`).
- OpenF1: <https://openf1.org> (the free tier, €9.90 a month for live data, the 30-minute rule, about 3 seconds behind
  live); code: <https://github.com/br-g/openf1>.
- Academic ML: the dissertation's references [3], [8]–[11], [18], [19], [22].
- This project: `results/evaluation.md` and the README (the accuracy, the speed and the SHAP checks).
