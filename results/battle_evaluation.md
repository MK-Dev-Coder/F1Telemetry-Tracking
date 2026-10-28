# Undercut threat and battle forecast: tested on the 2026 test races (rounds 1-17)

Undercut: learned from 229 real undercuts of 2022-2025 (the car behind pitted one lap before the car just ahead, both under green on dry tyres). Gains in seconds, by the car ahead's tyre wear:

| Car ahead's tyre wear | Undercuts | Usual gain (middle half) |
|---|---:|---|
| under 0.3 s | 47 | 0.2 s (-0.9 to 1.9) |
| 0.3-0.7 s | 76 | 0.6 s (-0.6 to 2.4) |
| 0.7-1.2 s | 65 | 0.7 s (-0.8 to 2.3) |
| over 1.2 s | 41 | 1.3 s (-0.3 to 2.4) |

On the 33 real undercuts of the 2026 test races (it worked 9 times, 27%):

| Way | Brier score (lower is better) | Right at 50% |
|---|---:|---:|
| The chance by tyre wear (the app) | 0.140 | 82% |
| The chance without tyre wear | 0.146 | 82% |
| How often undercuts worked in 2022-2025 (19%) | 0.206 | 73% |

Battle forecast: 6,882 laps where a car was 1-10 s behind the car ahead under green. 'Within 5 laps' is checked only while both cars stayed out under green.

| Way | Said within 1 s in 5 laps | It happened | Of those that happened, foreseen | Error in laps (median, mean) |
|---|---:|---:|---:|---|
| Pace over the last 5 green laps (the app) | 744 | 68% | 54% of 936 | 1, 1.3 |
| The gap's change over the last 3 laps | 772 | 66% | 55% of 936 | 1, 1.3 |
