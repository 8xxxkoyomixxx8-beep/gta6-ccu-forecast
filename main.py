"""Launch-day Twitch CCU forecaster (gta6-twitch-ccu-forecast-01), Python, standard library only.

The model, in one paragraph: every graded case of this contest (the four practice launches and
Grand Theft Auto VI itself) is a console game that unlocks at midnight local time, rolling around
the world, so by hour 1 of the UTC release day (00:00-01:00) the game is already live almost
everywhere except the Americas and its category is already drawing a large audience. The case file
carries dozens of past launches' full hourly curves (`comparables`). For each comparable that was
itself already live at hour 1 -- its hour-1 viewers at least LIVE_FRAC of its own release-day peak,
which separates the rolling launches from the ones that unlocked at a single later UTC instant and
sat near zero until then -- take the log ratio log(viewers_h / viewers_1) of every release-day hour
to its own hour 1. The forecast for hour h is hour 1's print times the exponential of the empirical
quantiles of those log ratios across the pool: the pinball loss is scored on the log scale, so the
empirical quantiles of the log ratios are the forecasts that minimise it when the target launch is
exchangeable with the pool. The pool's spread across launches is the forecast's spread: it is wide
where launches disagreed (the US unlock waves, the evening prime-time peak) and narrow where they
moved together.

Run by the grader as `python3 -P -s /bot/main.py` in a sandbox with one case file in /input; run
locally with MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR set. Deterministic: no randomness, no clock,
no network; the same case file always produces the same bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves this file's directory off sys.path; nothing else is imported, but keep the
# line so a sibling module would be found if one is ever added.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each output row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# A comparable counts as a rolling launch -- already live at hour 1, as this contest's targets
# are -- when its hour-1 viewers reach this fraction of its release-day peak. In the practice pool
# the launches are bimodal on this measure (at most ~0.011, or at least ~0.082), so any threshold
# inside that gap selects the same pool.
LIVE_FRAC = 0.03

# Below this many pool members the live-at-hour-1 filter is dropped rather than forecast from a
# handful of curves (a guard for a differently shaped final pool; the practice pool keeps 29).
MIN_POOL = 8


def parse_ts(text: str) -> datetime:
    """An ISO 8601 UTC timestamp from the case file, as an aware datetime."""
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def day_viewers(comp: dict) -> dict[int, float]:
    """A comparable's release-day hourly viewers keyed by hour 1..24, from its `hourly` points.

    Only points falling exactly on an hour boundary of the release day count; a point covers the
    hour starting at its `from`, so the point at day_start is hour 1.
    """
    start = parse_ts(comp["day_start"])
    day: dict[int, float] = {}
    for point in comp["hourly"]:
        offset = (parse_ts(point["from"]) - start).total_seconds()
        if 0 <= offset < 86400 and offset % 3600 == 0:
            day[int(offset // 3600) + 1] = point["viewers"]
    return day


def pool_log_ratios(comparables: list[dict], game_name: str, exclude_name: str | None = None) -> list[dict[int, float]]:
    """The log-ratio curves log(viewers_h / viewers_1) of the pool's rolling launches.

    A comparable joins the pool when it has an hour-1 print of at least 1 viewer and that print is
    at least LIVE_FRAC of its release-day peak. The case's own game (its pre-release category
    history rides along as a comparable in the final) and `exclude_name` never join.
    """
    def collect(require_live: bool) -> list[dict[int, float]]:
        pool: list[dict[int, float]] = []
        for comp in comparables:
            name = comp["name"]
            if name == exclude_name or game_name in name:
                continue
            day = day_viewers(comp)
            first = day.get(1)
            if first is None or first < 1:
                continue
            if require_live and first < LIVE_FRAC * max(day.values()):
                continue
            pool.append({h: math.log(day[h] / first) for h in range(2, 25) if day.get(h, 0) >= 1})
        return pool

    pool = collect(require_live=True)
    if len(pool) < MIN_POOL:
        pool = collect(require_live=False)
    return pool


def empirical_quantile(sorted_values: list[float], tau: float) -> float:
    """The tau quantile of sorted values, linearly interpolated between order statistics."""
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up; a whole number comes back as an int for compact JSON."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def forecast_hours(pool: list[dict[int, float]], p1: float, hours: list[int]) -> dict[int, list]:
    """The five quantiles of each requested hour: p1 scaled by the pool's log-ratio quantiles."""
    out: dict[int, list] = {}
    for h in hours:
        ratios = sorted(curve[h] for curve in pool if h in curve)
        if not ratios:
            ratios = [0.0]
        quantiles = [p1 * math.exp(empirical_quantile(ratios, tau)) for tau in LEVELS]
        # Sorting guards the non-decreasing rule against interpolation ties; clamping keeps every
        # value inside the contest's (0, 1e9] window.
        out[h] = [round2(min(1e9, max(0.01, q))) for q in sorted(quantiles)]
    return out


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    pool = pool_log_ratios(case["comparables"], case["game"]["name"])
    hours = [row["hour"] for row in case["forecast"]]
    by_hour = forecast_hours(pool, p1, hours)
    return {"hours": [{"hour": h, "q": by_hour[h]} for h in hours]}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
