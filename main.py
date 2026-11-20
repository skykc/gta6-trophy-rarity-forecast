"""Refresh-aware earners/players trophy forecaster.

Forecasts every GTA VI trophy's Exophase earn rate at a case's target as five
quantiles (5/25/50/75/95%), from the case's history of page reads.

Idea, in one paragraph: an earn rate is a ratio, earners over players, and the
two counts move on different clocks.  Exophase refreshes its page in jumps, so
consecutive reads that print identical numbers are one snapshot, not many
observations.  Between distinct snapshots both counts grow roughly linearly
against log-time x = ln(1 + hours since launch), so each count is fitted
separately in log space and extrapolated to the target; the forecast rate is
the ratio of the two extrapolations, which makes dilution automatic: new
players start with zero trophies, so as players outgrow earners the rate of a
hard trophy falls.  When history holds only one snapshot there is nothing to
fit, so a launch-night prior is used instead: one refresh before the target
that multiplies players several-fold while earners creep up slowly.  Quantiles
are a logit-normal band around the median whose width grows with the horizon
and with how badly the log-time fits wobble.

Standard library only, deterministic, no network.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Launch moment the log-time clock is anchored to.
LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# --- Tunable model constants (grid-searched on the three practice answer
# scenarios A/B/C; see the tuning notes in the repo history) ---
# Log-time slope of ln(players) when history shows a single snapshot: the
# launch-night refresh took players 638 -> 3868, i.e. ln(6.06) over the
# practice log-time span.
PRIOR_PLAYER_SLOPE = 5.65
# Log-time slope of ln(earners) in the same single-snapshot prior.  Together
# with PRIOR_PLAYER_SLOPE it puts the practice median rate at roughly
# launch_rate x 0.21 (earners up ~29% while players jump ~6x).
PRIOR_EARNER_SLOPE = 0.80
# How much the prior earner slope tilts with a trophy's launch rate: easy
# (common) trophies pick up earners faster than rare ones.  Slope shift is
# GAMMA * ln(rate / median_rate).
EARNER_RATE_GAMMA = 0.15
# Logit spread: s = SPREAD_BASE + SPREAD_DAY * sqrt(horizon_days), plus a
# residual term from the log-time fit (capped).  Wide enough that the 90%
# band covers the plausible refresh-factor range, not just the median guess.
SPREAD_BASE = 0.25
SPREAD_DAY = 0.30
SPREAD_RESID_CAP = 0.50
# The pending refresh is a discrete event, not a smooth slope: its size
# reflects growth accumulated while the page sat stale, so it does not shrink
# just because the checkpoint sits 30 minutes closer to the target.  The
# single-snapshot prior therefore never extrapolates over less than
# PRIOR_DX_FLOOR of log-time -- the launch-night practice span (03:00 to
# 04:30 UTC), over which the prior player slope reproduces the observed
# first-refresh jump, 638 -> 3868 players.
PRIOR_DX_FLOOR = 0.32
# Long-run slopes used only as the shrinkage target when a log-time fit
# exists (2+ distinct snapshots).  These are NOT the launch-night prior
# above: a slope of 5.65 extrapolated over a 4.5-day horizon would predict
# absurd player counts, while the growth actually observed over launch night
# as a whole (638 -> 3868 players while log-time rose ~1.9) is a slope near
# 1.  Earners grow slower than players, hence 0.5.
FIT_PLAYER_SLOPE_PRIOR = 1.0
FIT_EARNER_SLOPE_PRIOR = 0.5
# How strongly a fitted slope is trusted over the prior, by number of
# distinct snapshots n: weight = n / (n + SHRINK_N).
SHRINK_N = 3.0
# A trophy sitting at 100%: its median decays slowly with the horizon in
# logit space (new players have not all earned it yet) and its band is wide
# below, so a late drop is covered without paying much when it stays at 100.
TOP_MEDIAN_DECAY = 1.6
TOP_SPREAD_BASE = 1.5
TOP_SPREAD_DAY = 1.0
# Hard clamps on fitted log-time slopes, to keep wild extrapolations in check.
PLAYER_SLOPE_MIN, PLAYER_SLOPE_MAX = 0.0, 8.0
EARNER_SLOPE_MIN, EARNER_SLOPE_MAX = -1.0, 8.0


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def log_time(moment):
    """ln(1 + hours since launch).  Negative just before launch is fine."""
    hours = (moment - LAUNCH).total_seconds() / 3600.0
    return math.log(max(1.0 + hours, 0.05))


def least_squares(xs, ys):
    """Slope, intercept and RMS residual of the least-squares line y ~ x."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0, my, 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    resid = math.sqrt(sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)) / n)
    return slope, intercept, resid


def distinct_snapshots(history):
    """Collapse consecutive reads that print identical numbers into one
    snapshot.  A snapshot's effective time is the midpoint of the interval
    over which it was displayed (first to last read showing it): the refresh
    that produced it happened before its first read, and it stayed current
    through its last read.  Using the midpoint keeps a long-lived snapshot
    from being dated hours before it stopped being current, which would
    stretch every extrapolation anchored on it (refresh inversion, cheap
    version)."""
    snaps = []
    for read in history:
        key = (read["players_tracked"], tuple(read["rates"]))
        if snaps and snaps[-1]["key"] == key:
            snaps[-1]["last_at"] = parse_time(read["at"])
            continue
        snaps.append({
            "key": key,
            "first_at": parse_time(read["at"]),
            "last_at": parse_time(read["at"]),
            "players": read["players_tracked"],
            "rates": list(read["rates"]),
        })
    for snap in snaps:
        snap["t"] = snap["first_at"] + (snap["last_at"] - snap["first_at"]) / 2
    return snaps


def forecast_case(case):
    trophies = case["trophies"]
    n = len(trophies)
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0
    sqrt_days = math.sqrt(horizon_days)
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    x_now = log_time(at)
    x_tgt = log_time(target)

    # Last non-null rate per trophy, for fallbacks.
    last_rate = [0.0] * n
    for read in history:
        for i, r in enumerate(read["rates"]):
            if r is not None:
                last_rate[i] = r
    if not history:
        # No evidence at all: flat 1% with a very wide band.
        return {"trophies": [
            {"id": t["id"], "q": quantiles(1.0, 2.0)} for t in trophies
        ]}

    snaps = distinct_snapshots(history)
    xs = [log_time(s["t"]) for s in snaps]
    last_snap = snaps[-1]
    x_last = xs[-1]
    p_last = float(last_snap["players"])
    # Anchor for the extrapolation: with a fittable history the line is
    # re-anchored on the last snapshot; with a single snapshot there is no
    # line, and the prior jump is measured from the checkpoint state (the
    # page at the checkpoint is the snapshot), not from when the snapshot
    # first appeared hours earlier -- and over at least PRIOR_DX_FLOOR of
    # log-time, so a later checkpoint does not shrink the pending refresh.
    if len(snaps) >= 2:
        x_anchor = x_last
    else:
        x_anchor = x_tgt - max(x_tgt - x_now, PRIOR_DX_FLOOR)

    # Median launch rate, used only to tilt the earner prior by easiness.
    positive = sorted(r for r in last_rate if r > 0)
    r_med = positive[len(positive) // 2] if positive else 1.0

    # --- players: fit ln(P) vs log-time when possible, else the prior. ---
    if len(snaps) >= 2:
        ys_p = [math.log(max(s["players"], 1)) for s in snaps]
        b_p_fit, a_p, _ = least_squares(xs, ys_p)
        w = len(snaps) / (len(snaps) + SHRINK_N)
        b_p = w * b_p_fit + (1.0 - w) * FIT_PLAYER_SLOPE_PRIOR
        b_p = min(PLAYER_SLOPE_MAX, max(PLAYER_SLOPE_MIN, b_p))
        a_p = math.log(p_last) - b_p * x_anchor
    else:
        b_p = PRIOR_PLAYER_SLOPE
        a_p = math.log(p_last) - b_p * x_anchor
    p_pred = math.exp(a_p + b_p * x_tgt)
    p_pred = max(p_pred, p_last)

    base_spread = SPREAD_BASE + SPREAD_DAY * sqrt_days

    rows = []
    for i, trophy in enumerate(trophies):
        r0 = last_rate[i]
        # --- earners for this trophy: fit ln(E) vs log-time, else prior. ---
        if len(snaps) >= 2:
            ex, ey = [], []
            for s, x in zip(snaps, xs):
                r = s["rates"][i]
                if r is None or s["players"] <= 0:
                    continue
                e = max(r * s["players"] / 100.0, 0.5)
                ex.append(x)
                ey.append(math.log(e))
            if len(ex) >= 2:
                b_e_fit, a_e, resid = least_squares(ex, ey)
                w = len(ex) / (len(ex) + SHRINK_N)
                prior_e = FIT_EARNER_SLOPE_PRIOR + EARNER_RATE_GAMMA * math.log(max(r0, 0.01) / r_med)
                b_e = w * b_e_fit + (1.0 - w) * prior_e
                b_e = min(EARNER_SLOPE_MAX, max(EARNER_SLOPE_MIN, b_e))
                e_last = max(r0 * p_last / 100.0, 0.5)
                a_e = math.log(e_last) - b_e * x_anchor
                resid_term = min(resid, SPREAD_RESID_CAP)
            else:
                b_e = PRIOR_EARNER_SLOPE + EARNER_RATE_GAMMA * math.log(max(r0, 0.01) / r_med)
                e_last = max(r0 * p_last / 100.0, 0.5)
                a_e = math.log(e_last) - b_e * x_anchor
                resid_term = 0.0
        else:
            b_e = PRIOR_EARNER_SLOPE + EARNER_RATE_GAMMA * math.log(max(r0, 0.01) / r_med)
            e_last = max(r0 * p_last / 100.0, 0.5)
            a_e = math.log(e_last) - b_e * x_anchor
            resid_term = 0.0
        e_pred = math.exp(a_e + b_e * x_tgt)
        e_pred = max(e_pred, max(r0 * p_last / 100.0, 0.0))
        e_pred = min(e_pred, p_pred)

        median = 100.0 * e_pred / p_pred if p_pred > 0 else r0
        median = clamp_pct(median)
        spread = base_spread + resid_term

        if r0 >= 100.0:
            # Everybody has it so far.  Median decays a little with horizon
            # (late joiners have not earned it yet); the lower band is wide.
            med_logit = logit(99.99) - TOP_MEDIAN_DECAY * sqrt_days
            spread = max(spread, TOP_SPREAD_BASE + TOP_SPREAD_DAY * sqrt_days)
            rows.append({"id": trophy["id"],
                         "q": quantiles_from_logit(med_logit, spread)})
        else:
            rows.append({"id": trophy["id"],
                         "q": quantiles(median, spread)})
    return {"trophies": rows}


def quantiles(median_pct, spread):
    return quantiles_from_logit(logit(median_pct), spread)


def quantiles_from_logit(med_logit, spread):
    out = []
    prev = None
    for z in Z:
        v = round(inv_logit(med_logit + z * spread), 4)
        v = min(99.99, max(0.01, v))
        if prev is not None and v < prev:
            v = prev
        out.append(v)
        prev = v
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
