"""Launch-day Twitch audience forecast for GTA VI (make.fun entry v3).

Competition: gta6-twitch-ccu-forecast-01.  Python 3.13, stdlib only.

Premise: once a console launch's day is rescaled by its first-hour
audience, the remaining hours follow a shape that repeats from launch to
launch.  The forecaster therefore studies the historical launches inside
each case's `comparables`, expresses every one of them as log ratios to
its own first hour, and transfers that shape onto the case at hand.

Pipeline per case (p1 = the case's hour-1 audience, at least 1):

Pool.  Comparables that are genuine launches with a positive first-hour
print are parsed into entries holding their hourly log-ratio curve, the
log of their first-hour audience, a console-only flag, an unlock kind
(rolling / global / unknown, classified from the unlock note's wording),
and pre-release ramp measurements.  The one comparable that is the
target's own pre-release history is recognised and left out.

Centre.  The forecast centre is the plain average, in log space, of:
  (a) a per-hour least-squares fit of log ratio on centred log first-hour
      audience across the console-only entries, slope shrunk by 0.8 and
      clamped to [-0.5, 0]; and
  (b) a similarity-weighted median of slope-adjusted log ratios across
      all entries, the similarity of an entry being exp(-d/2) for
      d = ((log-size gap)/1.5)^2 + 2.0 per mismatched trait (console
      flag, unlock kind), and the adjusting slope being the weighted
      per-hour slope under the same shrink and clamp.
The target counts as a console launch; its unlock kind is classified
from the case notes by the same wording rules, defaulting to rolling.

Spread.  The ensemble centre is also fitted to every pool entry in turn,
and the residuals of those fits are gathered for the eight entries most
similar to the target.  The five level offsets at hour h are weighted
empirical quantiles of the residuals pooled from hours h-1, h and h+1
(weights: similarity, discounted by 1/(1 + hour distance)), recentred on
their median so the centre stays the median forecast.  Where residuals
are too scarce, or a case is malformed, the forecast falls back to
persistence: p1 * exp(z * 0.15 * sqrt(h - 1)) per level.

Output values are rounded half-up to two decimals, made non-decreasing
across levels, and clipped into (0, 1e9].  Input: every *.in file in
$MAKEFUN_INPUT_DIR (default /input).  Output: <id>.out in
$MAKEFUN_OUTPUT_DIR (default /output).  No randomness, no network:
repeated runs are byte-identical, and bad input still produces a valid
baseline forecast rather than an exception.
"""

import json, math, os, re
from datetime import datetime, timedelta
from pathlib import Path


def _env_path(var, fallback):
    return Path(os.environ.get(var, fallback))


INPUT_DIR = _env_path("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = _env_path("MAKEFUN_OUTPUT_DIR", "/output")

LEVELS = [0.05, 0.25, 0.5,
          0.75, 0.95]
Z = [-1.645, -0.674, 0.0,
     0.674, 1.645]
HOURS = list(range(2, 25))
BASE_SIGMA = 0.15
# Kernel and calibration settings, picked by leave-one-out validation.
KERNEL = {"ell": 1.5, "w_console": 2.0, "w_uclass": 2.0, "shrink": 0.8}
CAL_SCALE, CAL_HOUR_POOL, CAL_NEIGHBOURS = 1.0, 1, 8


def round2(x):
    """Round half-up to two decimals (the starter's rounding rule)."""
    hundredths = x * 100.0
    units = math.floor(hundredths)
    if hundredths - units >= 0.5:
        units += 1
    rounded = units / 100.0
    return int(rounded) if rounded == int(rounded) else rounded


def parse_time(s):
    """Parse an ISO-8601 UTC timestamp string into a datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def has_pc(platforms):
    """Whether a platforms string names PC as a whole word."""
    return re.search(r"\bpc\b", (platforms or "").lower()) is not None


def is_target_history(comp):
    """Whether a comparable is the target's own pre-release history entry."""
    name = (comp.get("name") or "").lower()
    unlock = comp.get("unlock") or ""
    return "before release" in name or unlock.startswith("Not a launch")


def classify_unlock(unlock):
    """Bucket a free-text unlock note: history, global, rolling, unknown."""
    text = (unlock or "").lower()
    rules = (
        ("history", ("not a launch",)),
        ("global", ("simultaneous", "single global", "single simultaneous",
                    "global unlock", "one global", "same time")),
        ("rolling", ("midnight", "rolling")),
    )
    for label, words in rules:
        if any(w in text for w in words):
            return label
    return "unknown"


def _hour_map(hourly):
    """Map each hourly row's start time to its viewer count.

    Rows whose timestamps do not parse are skipped.
    """
    by_start = {}
    for row in hourly or []:
        try:
            by_start[parse_time(row["from"])] = row.get("viewers")
        except (KeyError, TypeError, ValueError):
            continue
    return by_start


def release_day_ratios(comp):
    """(log of hour-1 audience, {hour: log ratio to hour 1}) for a launch.

    None when the comparable lacks a usable positive hour-1 print; hours
    without a positive count are absent from the ratio dict.
    """
    try:
        start = parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    by_start = _hour_map(comp.get("hourly"))
    opener = by_start.get(start)
    if not opener or opener <= 0:
        return None
    log_ratios = {}
    for h in range(2, 25):
        count = by_start.get(start + timedelta(hours=h - 1))
        if count and count > 0:
            log_ratios[h] = math.log(count / opener)
    return math.log(max(1, opener)), log_ratios


def pool_members(comparables):
    """Split comparables into (console launches, all valid launches).

    Entries are (log hour-1, ratio dict) pairs from release_day_ratios.
    The local leave-one-out harness names its targets through this split.
    """
    console_pool = []
    every_valid = []
    for comp in comparables or []:
        pair = release_day_ratios(comp)
        if pair is None:
            continue
        every_valid.append(pair)
        if not has_pc(comp.get("platforms")) and not is_target_history(comp):
            console_pool.append(pair)
    return console_pool, every_valid


def parse_launch(comp):
    """Parse one comparable into a pool entry dict, or None if unusable.

    An entry carries the release-day log-ratio curve, the log of the
    hour-1 audience, the console flag, the unlock kind, and pre-release
    ramp measurements (kept for the kernel's optional feature distances).
    """
    try:
        start = parse_time(comp["day_start"])
        by_start = _hour_map(comp.get("hourly"))
        opener = by_start.get(start)
        if not opener or opener <= 0:
            return None
        log_curve = {}
        for h in HOURS:
            v = by_start.get(start + timedelta(hours=h - 1))
            if v and v > 0:
                log_curve[h] = math.log(v / opener)
        channels1 = None
        for row in comp.get("hourly") or []:
            try:
                if parse_time(row["from"]) == start:
                    channels1 = row.get("channels")
            except (KeyError, TypeError, ValueError):
                continue
        # Strength of the three pre-release days relative to hour 1.
        pre_vals = [v for t, v in by_start.items()
                    if start - timedelta(hours=72) <= t < start and v and v > 0]
        pre_peak = max(pre_vals) / opener if pre_vals else 0.0
        # Ramp readings from the individual hours just before the start.
        pre_hours = {}
        for k in range(1, 25):
            v = by_start.get(start - timedelta(hours=k))
            if v and v > 0:
                pre_hours[k] = v
        last14 = [v for k, v in pre_hours.items() if k <= 14]
        peak14 = max(last14) if len(last14) >= 6 else None
        mean6 = (sum(v for k, v in pre_hours.items() if k <= 6)
                 / max(1, sum(1 for k in pre_hours if k <= 6))) if pre_hours else None
        unlock_note = comp.get("unlock") or ""
        return {
            "name": comp["name"],
            "ln_first": math.log(max(1, opener)),
            "first": opener,
            "curve": log_curve,
            "is_console": not has_pc(comp.get("platforms")),
            "unlock_kind": classify_unlock(unlock_note),
            "is_early": pre_peak > 0.35,
            "ln_pre_peak": math.log(peak14 / opener) if peak14 else None,
            "ln_pre_mean6": math.log(mean6 / opener) if mean6 else None,
            "ln_pre_last": math.log(pre_hours[1] / opener) if 1 in pre_hours else None,
            "ln_per_channel": (math.log(opener / channels1)
                               if channels1 and channels1 > 0 else None),
        }
    except Exception:
        return None



def interp_quantile(ordered, q):
    """Quantile of a sorted list by linear interpolation (q in [0, 1])."""
    n = len(ordered)
    if n == 1:
        return ordered[0]
    position = q * (n - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, n - 1)
    frac_part = position - lower
    return ordered[lower] * (1.0 - frac_part) + ordered[upper] * frac_part


def fit_ols(xs, ys):
    """Least-squares summary of y on x: (mean x, mean y, slope)."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    variance = sum((x - mean_x) ** 2 for x in xs)
    if variance > 0:
        gradient = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / variance
    else:
        gradient = 0.0
    return mean_x, mean_y, gradient


def make_baseline_offsets():
    """Persistence as log offsets: flat centre, spread widening with h."""
    return {h: [z * BASE_SIGMA * math.sqrt(h - 1) for z in Z] for h in HOURS}


BASELINE_LOG_OFFSETS = make_baseline_offsets()


def filter_pool(entries, params):
    """Select a sub-pool by the params' "pool" mode."""
    mode = params.get("pool", "all")
    if mode == "all":
        return list(entries)
    if mode == "console":
        return [e for e in entries if e["is_console"]]
    if mode == "rolling":
        return [e for e in entries if e["unlock_kind"] == "rolling"]
    if mode == "console_or_rolling":
        return [e for e in entries if e["is_console"] or e["unlock_kind"] == "rolling"]
    if mode == "not_global":
        return [e for e in entries if e["unlock_kind"] != "global"]
    raise ValueError(mode)


# == centres ==============================================================
def ols_centre(ln_first, entries, params):
    """Per-hour least-squares log-ratio centre over console-only entries."""
    console_entries = filter_pool(entries, {"pool": params.get("pool", "console")})
    centres = {}
    for h in HOURS:
        size_logs = [e["ln_first"] for e in console_entries if h in e["curve"]]
        ratio_vals = [e["curve"][h] for e in console_entries if h in e["curve"]]
        if len(size_logs) < 3:
            centres[h] = 0.0
            continue
        mean_x, mean_y, gradient = fit_ols(size_logs, ratio_vals)
        gradient = min(0.0, max(-0.5, gradient * 0.8))
        centres[h] = mean_y + gradient * (ln_first - mean_x)
    return centres


def kernel_sq_dist(launch, target, ln_first, params):
    """Squared similarity distance between a pool entry and the target."""
    dist_sq = ((launch["ln_first"] - ln_first) / params.get("ell", 1.5)) ** 2
    if params.get("w_console", 0.0) and bool(launch["is_console"]) != bool(target.get("is_console", True)):
        dist_sq += params["w_console"]
    if params.get("w_uclass", 0.0) and launch["unlock_kind"] != target.get("unlock_kind", "rolling"):
        dist_sq += params["w_uclass"]
    if params.get("w_early", 0.0) and bool(launch["is_early"]) != bool(target.get("is_early", False)):
        dist_sq += params["w_early"]
    for key, ell in (("ln_pre_peak", "ell_pre_peak"), ("ln_per_channel", "ell_per_channel")):
        if params.get(ell):
            tv = target.get(key)
            mv = launch.get(key)
            if tv is not None and mv is not None:
                dist_sq += ((mv - tv) / params[ell]) ** 2
    return dist_sq


def kernel_weights(entries, ln_first, params, target):
    return [math.exp(-0.5 * kernel_sq_dist(e, target, ln_first, params)) for e in entries]


def fill_feats(given):
    """Target feature dict with the defaults filled in."""
    merged = {"is_console": True, "unlock_kind": "rolling", "is_early": False,
              "ln_pre_peak": None, "ln_per_channel": None,
              "ln_pre_mean6": None, "ln_pre_last": None}
    if given:
        for k in merged:
            if k in given and given[k] is not None:
                merged[k] = given[k]
    return merged


def weighted_quantile(weighted_items, q):
    """Weighted quantile by linear interpolation on the weighted CDF."""
    weighted_items = sorted(weighted_items)
    weight_sum = sum(wt for _, wt in weighted_items)
    if weight_sum <= 0:
        return weighted_items[len(weighted_items) // 2][0]
    cum = 0.0
    lo_val, lo_cum = weighted_items[0][0], 0.0
    for val, wt in weighted_items:
        cum += wt
        c = cum / weight_sum
        if c >= q:
            if c == lo_cum:
                return val
            return lo_val + (val - lo_val) * (q - lo_cum) / (c - lo_cum)
        lo_val, lo_cum = val, c
    return weighted_items[-1][0]


def kernel_median_centre(ln_first, entries, params, feats=None):
    """Similarity-weighted median of slope-adjusted log ratios."""
    target = fill_feats(feats)
    weights = kernel_weights(entries, ln_first, params, target)
    centres = {}
    for h in HOURS:
        points = [(e["ln_first"], e["curve"][h], wt)
                  for e, wt in zip(entries, weights) if h in e["curve"] and wt > 1e-9]
        if len(points) < 3:
            centres[h] = None
            continue
        # Weighted slope at this hour, shrunk and clamped, then adjust.
        w_total = sum(wt for _, _, wt in points)
        w_mean_x = sum(wt * x for x, _, wt in points) / w_total
        w_mean_y = sum(wt * y for _, y, wt in points) / w_total
        w_var = sum(wt * (x - w_mean_x) ** 2 for x, _, wt in points)
        raw_slope = (sum(wt * (x - w_mean_x) * (y - w_mean_y) for x, y, wt in points) / w_var
                     if w_var > 0 else 0.0)
        slope = min(0.0, max(-0.5, raw_slope * params.get("shrink", 0.8)))
        adjusted = [(y - slope * (x - ln_first), wt) for x, y, wt in points]
        centres[h] = weighted_quantile(adjusted, 0.5)
    return centres


def ensemble_centre(ln_first, entries, params, feats=None):
    """Plain average, in log space, of the OLS and kernel-median centres."""
    from_ols = ols_centre(ln_first, entries, params)
    from_median = kernel_median_centre(ln_first, entries, params, feats)
    out = {}
    for h in HOURS:
        present = [c for c in (from_ols.get(h), from_median.get(h)) if c is not None]
        out[h] = sum(present) / len(present) if present else None
    return out


# == calibration ==========================================================
def calibrated_model(centre_fn, dist_params, scale=1.0, hour_pool=1, neighbours=0):
    """Wrap a centre function with empirical residual calibration.

    The centre is refitted for every pool entry (each entry predicted
    from the whole pool, itself included).  At hour h the offsets are
    the weighted empirical quantiles of those residuals pooled over
    h +- hour_pool, weights being similarity to the target discounted
    by 1/(1 + hour distance); the result is
    centre + (q_tau - q_50) * scale.  When neighbours > 0 the residual
    set is first restricted to the target's most similar entries,
    because distant pool members are systematically harder to predict
    and would over-widen the intervals.
    """
    def model(ln_first, entries, params, feats=None):
        p = dict(params)
        target = fill_feats(feats)
        centres = centre_fn(ln_first, entries, p, feats)
        refits = [centre_fn(e["ln_first"], entries, p, e) for e in entries]
        weights = [math.exp(-0.5 * kernel_sq_dist(e, target, ln_first, dist_params))
                   for e in entries]
        if neighbours and 0 < neighbours < len(weights):
            thresh = sorted(weights, reverse=True)[neighbours - 1]
            weights = [wt if wt >= thresh else 0.0 for wt in weights]
        resid_by_hour = {h: [] for h in HOURS}
        for e, fit, wt in zip(entries, refits, weights):
            if wt <= 1e-6:
                continue
            for h, y in e["curve"].items():
                if fit.get(h) is not None:
                    resid_by_hour[h].append((y - fit[h], wt))
        out = {}
        for h in HOURS:
            centre_h = centres.get(h)
            hour_pairs = []
            for hh in range(h - hour_pool, h + hour_pool + 1):
                for r, wt in resid_by_hour.get(hh, []):
                    hour_pairs.append((r, wt / (1 + abs(hh - h))))
            if centre_h is None or len(hour_pairs) < 5:
                out[h] = BASELINE_LOG_OFFSETS[h]
                continue
            quantile_offsets = [weighted_quantile(hour_pairs, tau) for tau in LEVELS]
            median_offset = quantile_offsets[2]
            out[h] = [centre_h + (qo - median_offset) * scale for qo in quantile_offsets]
        return out
    return model


MODEL = calibrated_model(ensemble_centre, KERNEL, scale=CAL_SCALE,
                         hour_pool=CAL_HOUR_POOL, neighbours=CAL_NEIGHBOURS)


# == case scaffolding =====================================================
def baseline_quantiles(p1, hour):
    """Persistence: the audience stays at p1, the spread grows in log."""
    width = BASE_SIGMA * math.sqrt(hour - 1)
    return [p1 * math.exp(z * width) for z in Z]


def clean_quantiles(values):
    """Round half-up, enforce non-decreasing order, clip into (0, 1e9]."""
    cleaned = []
    floor_so_far = 0.0
    for value in values:
        q = min(1e9, max(floor_so_far, max(round2(value), 0.01)))
        cleaned.append(q)
        floor_so_far = q
    return cleaned


def hour1_print(case):
    """The case's hour-1 audience, at least 1; 1 when missing or invalid."""
    try:
        for row in case.get("known") or []:
            if row.get("hour") == 1:
                viewers = row.get("viewers")
                if isinstance(viewers, (int, float)) and math.isfinite(viewers):
                    return max(1.0, float(viewers))
                return 1.0
    except Exception:
        pass
    return 1.0


def target_features(case):
    """Kernel features of the target itself: a console launch whose
    unlock kind is classified from the case notes by the same wording
    rules used for pool members (rolling when there are no notes)."""
    notes = case.get("notes") or ""
    kind = classify_unlock(notes) if notes else "rolling"
    if kind == "history":
        kind = "rolling"
    return {"is_console": True, "unlock_kind": kind, "is_early": False,
            "ln_pre_peak": None, "ln_per_channel": None}


def forecast_case(case):
    """Forecast hours 2..24 for one case dict; never raises on bad data."""
    p1 = hour1_print(case)
    try:
        # The scored levels are the standard five; another level list
        # still gets these five quantiles in the case's level order (the
        # z values only exist for the standard five).
        requested = case.get("forecast") or [{"hour": h} for h in range(2, 25)]
        pool_entries = []
        for comp in case.get("comparables") or []:
            if is_target_history(comp):
                continue
            parsed = parse_launch(comp)
            if parsed is not None:
                pool_entries.append(parsed)
        log_offsets = MODEL(math.log(max(1.0, p1)), pool_entries, KERNEL,
                            target_features(case))
        rows = []
        for row in requested:
            h = int(row["hour"])
            if not 2 <= h <= 24:
                continue
            rows.append({"hour": h,
                         "q": clean_quantiles([p1 * math.exp(o) for o in log_offsets[h]])})
        rows.sort(key=lambda r: r["hour"])
        return {"hours": rows}
    except Exception:
        # Last resort: a valid persistence forecast anchored on p1.
        fallback_rows = [
            {"hour": h, "q": clean_quantiles(baseline_quantiles(p1, h))}
            for h in range(2, 25)
        ]
        return {"hours": fallback_rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for in_path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case_data = json.loads(in_path.read_text(encoding="utf-8"))
        except Exception:
            case_data = {}
        result = forecast_case(case_data)
        text = json.dumps(result, separators=(",", ":")) + "\n"
        dest = OUTPUT_DIR / (in_path.stem + ".out")
        dest.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
