# Forecasting GTA VI's launch-day Twitch audience

How many people will watch Grand Theft Auto VI on Twitch in each hour of
its release day? This entry predicts hours 2–24 of that day as five
quantiles per hour, given only hour 1's audience and a set of earlier
game launches to learn the day's shape from.

[![bounty](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ninochaan/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ninochaan/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

Entry for the [gta6-twitch-ccu-forecast-01 bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01).

## How it works

`main.py` is a single self-contained program (Python 3.13, standard
library only, deterministic, no network access):

1. **Pool.** Every comparable that is a genuine launch with a positive
   first-hour audience becomes an entry holding its release-day curve as
   log ratios to hour 1, plus its console flag and unlock kind.
2. **Centre.** Two centres are averaged in log space: a per-hour least
   squares fit over the console-only entries (slope shrunk ×0.8, clamped
   to [-0.5, 0]), and a similarity-weighted median of slope-adjusted log
   ratios over the whole pool, where similarity decays with log-size
   distance and with console/unlock mismatches.
3. **Calibration.** The ensemble is refitted on each pool entry, and the
   residuals of the eight entries most similar to the target become
   weighted empirical quantile offsets per hour (neighbouring hours
   pooled), so the intervals are as wide as similar launches actually
   were — no wider, no narrower.
4. **Fallbacks.** Scarce residuals or a malformed case degrade
   gracefully to a persistence forecast whose spread widens like
   0.15·sqrt(h − 1) in log space; the program never crashes.

Validation by leave-one-launch-out over the practice cases: mean pinball
loss ≈ 0.16 on the large-console protocol, against ≈ 0.32 for the
persistence baseline on the same targets.

## Running it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Each `<id>.in` case in the input directory produces a matching
`<id>.out` file with one row per forecast hour and five quantiles per
row.
