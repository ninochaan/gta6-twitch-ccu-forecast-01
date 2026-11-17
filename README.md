# GTA VI launch-day Twitch viewers forecast

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day (2026-11-19 UTC), hours 2–24, as five quantiles per
hour — from hour 1's print and the launch-day shapes of past console-only
game launches.

[![bounty](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ninochaan/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ninochaan/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

Entry for the [gta6-twitch-ccu-forecast-01 bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01).

## Model

`main.py` (Python 3.13, standard library only, deterministic, no network):

1. Builds a pool of past console-only launches (platforms mention no PC,
   real launches with a valid hour-1 print) and, per hour, their log-ratio
   `ln(viewers_h / viewers_1)` on release day.
2. Per hour, a plain OLS of log-ratio on centered `ln(p1)`, slope shrunk
   ×0.8 and clamped to [-0.5, 0]; spread from the IQR of the residuals /
   1.349, floored at 0.15 and capped at 1.5. Quantiles:
   `q_tau = p1 × exp(centre + z_tau × sigma)`.
3. Falls back to all comparables when a pool hour has fewer than 5 members,
   and to a widening persistence baseline when still fewer than 3. A bad
   case still writes a valid baseline forecast instead of crashing.

Leave-one-out over the console-only comparables in practice case p00:
mean pinball loss ≈ 0.17, vs ≈ 0.32 for the persistence baseline on the
same set (house baseline ≈ 0.387 on the full practice).

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
