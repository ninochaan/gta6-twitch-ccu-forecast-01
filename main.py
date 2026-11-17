"""Forecast GTA VI launch-day Twitch viewers, hours 2..24, from hour 1's print.

Model (written for gta6-twitch-ccu-forecast-01, Python 3.13 stdlib only):

The launch-day shape of a big console game on Twitch is fairly repeatable once
you scale it by how big hour 1 was, so this entry learns that shape from past
console-only launches and anchors it on the case's own hour-1 print.

For each case, with p1 = hour-1 viewers (at least 1):

1. Pool: from `comparables`, keep launches whose `platforms` mention no PC
   (case-insensitive word match on "pc"), that have a positive hour-1 print on
   their release day, and that are real launches (exclude the target-history
   entry, whose name contains "before release" or whose unlock starts with
   "Not a launch").  For each pool member and each hour h = 2..24, the
   log-ratio is ln(viewers_h / viewers_1) on its release day.
2. Per hour h, a plain OLS of log-ratio on centered ln(p1) across the pool
   members that have that hour: slope = cov/var, shrunk x0.8 and clamped to
   [-0.5, 0]; centre = mean_ratio + slope * (ln(p1) - mean_ln_p1).  The spread
   sigma is the IQR of the residuals / 1.349, floored at 0.15 and capped at
   1.5.  Quantiles: q_tau = p1 * exp(centre + z_tau * sigma), with
   z = [-1.645, -0.674, 0, 0.674, 1.645] for levels [0.05, 0.25, 0.5, 0.75,
   0.95], rounded to 2 decimals half-up, made non-decreasing by cumulative
   max, and clipped to (0, 1e9].
3. Fallbacks: if fewer than 5 pool members have a given hour, redo the same
   regression over ALL comparables with a valid hour-1 print; if still fewer
   than 3, use the persistence baseline for that hour,
   q_tau = p1 * exp(z_tau * 0.15 * sqrt(h - 1)).  A missing/invalid p1 means
   p1 = 1 and the baseline for every hour.  A bad case never crashes the run:
   it still writes a valid baseline forecast.

The program reads every *.in in $MAKEFUN_INPUT_DIR (default /input) and writes
<id>.out to $MAKEFUN_OUTPUT_DIR (default /output).  Deterministic, no
network, no randomness: two runs write identical bytes.
"""

import json
import math
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
BASE_SIGMA = 0.15
MIN_SIGMA = 0.15
MAX_SIGMA = 1.5
MIN_POOL_HOUR = 5
MIN_FALLBACK_HOUR = 3


def round2(x):
    """Round to 2 decimals, half up (like the starter's round2)."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value == int(value) else value


def parse_time(s):
    """Parse an ISO-8601 UTC timestamp into a datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def has_pc(platforms):
    """True when the platforms string mentions PC as a whole word."""
    return re.search(r"\bpc\b", (platforms or "").lower()) is not None


def is_target_history(comp):
    """True for the non-launch GTA VI history entry tucked into comparables."""
    name = (comp.get("name") or "").lower()
    unlock = comp.get("unlock") or ""
    return "before release" in name or unlock.startswith("Not a launch")


def release_day_ratios(comp):
    """(ln(hour-1 print), {hour: ln(viewers_h / viewers_1)}) for release day.

    Returns None when the comparable has no valid positive hour-1 print.
    Hours with no positive count are skipped (they carry no ratio).
    """
    try:
        day_start = parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    hourly = comp.get("hourly") or []
    by_from = {}
    for row in hourly:
        try:
            by_from[parse_time(row["from"])] = row.get("viewers")
        except (KeyError, TypeError, ValueError):
            continue
    first = by_from.get(day_start)
    if not first or first <= 0:
        return None
    ratios = {}
    for h in range(2, 25):
        viewers = by_from.get(day_start + timedelta(hours=h - 1))
        if viewers and viewers > 0:
            ratios[h] = math.log(viewers / first)
    return math.log(max(1, first)), ratios


def pool_members(comparables):
    """Console-only real launches with a valid hour-1 print, plus their ratios.

    Returns (console_pool, all_valid): each a list of (ln(p1), {hour: ratio}).
    """
    console = []
    all_valid = []
    for comp in comparables or []:
        parsed = release_day_ratios(comp)
        if parsed is None:
            continue
        all_valid.append(parsed)
        if not has_pc(comp.get("platforms")) and not is_target_history(comp):
            console.append(parsed)
    return console, all_valid


def quantile(values, q):
    """Linear-interpolation quantile of a sorted list (q in [0, 1])."""
    n = len(values)
    if n == 1:
        return values[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def regress_hour(members, hour, ln_p1):
    """OLS log-ratio forecast for one hour from a pool of members.

    Returns (centre, sigma) or None when too few members have this hour.
    """
    xs = []
    ys = []
    for ln_first, ratios in members:
        if hour in ratios:
            xs.append(ln_first)
            ys.append(ratios[hour])
    if len(xs) < 1:
        return None
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var = sum((x - mean_x) ** 2 for x in xs)
    if var > 0:
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        slope = cov / var
    else:
        slope = 0.0
    slope = max(-0.5, min(0.0, slope * 0.8))
    centre = mean_y + slope * (ln_p1 - mean_x)
    residuals = sorted(
        y - (mean_y + slope * (x - mean_x)) for x, y in zip(xs, ys)
    )
    sigma = (quantile(residuals, 0.75) - quantile(residuals, 0.25)) / 1.349
    sigma = max(MIN_SIGMA, min(MAX_SIGMA, sigma))
    return centre, sigma


def baseline_quantiles(p1, hour):
    """Persistence forecast: every hour stays at p1, spread widening in log."""
    spread = BASE_SIGMA * math.sqrt(hour - 1)
    return [p1 * math.exp(z * spread) for z in Z]


def clean_quantiles(values):
    """Round half-up, enforce non-decreasing, clip to (0, 1e9]."""
    out = []
    prev = 0.0
    for value in values:
        q = round2(value)
        if q < 0.01:
            q = 0.01
        q = max(q, prev)
        q = min(q, 1e9)
        out.append(q)
        prev = q
    return out


def hour_quantiles(p1, hour, console_pool, all_pool):
    """Five quantiles for one hour: regression, fallback pool, or baseline."""
    ln_p1 = math.log(max(1.0, p1))
    for members, minimum in ((console_pool, MIN_POOL_HOUR),
                             (all_pool, MIN_FALLBACK_HOUR)):
        have = sum(1 for _, ratios in members if hour in ratios)
        if have >= minimum:
            fitted = regress_hour(members, hour, ln_p1)
            if fitted is not None:
                centre, sigma = fitted
                return clean_quantiles(
                    [p1 * math.exp(centre + z * sigma) for z in Z]
                )
    return clean_quantiles(baseline_quantiles(p1, hour))


def hour1_print(case):
    """Hour-1 viewers from the case, at least 1; 1 when missing/invalid."""
    try:
        known = case.get("known") or []
        first = next((k for k in known if k.get("hour") == 1), None)
        if first is not None:
            viewers = first.get("viewers")
            if isinstance(viewers, (int, float)) and math.isfinite(viewers):
                return max(1.0, float(viewers))
    except Exception:
        pass
    return 1.0


def forecast_case(case):
    """Forecast hours 2..24 for one case dict; never raises on bad data."""
    p1 = hour1_print(case)
    try:
        # The scored levels are the standard five; any other level list still
        # gets these five quantiles in the case's level order (the z values
        # only exist for the standard five, so there is nothing else to emit).
        forecast_rows = case.get("forecast") or [
            {"hour": h} for h in range(2, 25)
        ]
        console_pool, all_pool = pool_members(case.get("comparables"))
        hours = []
        for row in forecast_rows:
            h = int(row["hour"])
            if not 2 <= h <= 24:
                continue
            hours.append(
                {"hour": h, "q": hour_quantiles(p1, h, console_pool, all_pool)}
            )
        hours.sort(key=lambda r: r["hour"])
        return {"hours": hours}
    except Exception:
        # Last resort: a valid persistence forecast anchored on p1.
        return {
            "hours": [
                {"hour": h, "q": clean_quantiles(baseline_quantiles(p1, h))}
                for h in range(2, 25)
            ]
        }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            case = {}
        forecast = forecast_case(case)
        out_path = OUTPUT_DIR / (path.stem + ".out")
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
