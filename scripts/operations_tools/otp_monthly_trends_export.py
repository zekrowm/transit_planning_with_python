"""Process OTP CSV to compute metrics and export tables/plots.

This script ingests a CSV with columns resembling:
    Route, Direction, Month, Day of the Week, Sum # On Time, Sum # Early, Sum # Late

It produces:
  1) A cleaned, processed CSV with columns:
     route_raw, route_clean, direction, month_label, period, dow,
     on_time, early, late, total_trips, pct_on_time, pct_early, pct_late
  2) Line plots (PNG) of OTP over time for each Route/Direction, split into:
     - Weekdays (Mon–Fri): shows average Weekday OTP with a min–max band (per month)
     - Saturday
     - Sunday
  3) Two plain-text trend logs:
     - otp_trend_summary_all.txt: every route/direction with a single headline
       trend number (weekday OTP slope, in percentage points per month) plus
       Saturday and Sunday slopes for reference.
     - otp_trend_summary_concerning.txt: the most concerning subset (default
       bottom 10% by composite concern score combining current gap-to-standard
       and projected decline). Tunable via --concerning-pct.

Design choices:
  - "Route" is cleaned to the part before "-" and then all non-alphanumerics are stripped.
  - Percentages are computed as share of total processed trips on a 0–100 scale.
  - "Month" (e.g., "Apr") is mapped to a "YY-MM" period using a simple 12-month window rule:
      Given CURRENT_YY_MM (default: "25-10" => Oct 2025), months with a month
      number <= current month are assigned CURRENT_YEAR; otherwise CURRENT_YEAR-1.
    Example with CURRENT_YY_MM="25-10":
      Jan..Oct -> 2025; Nov..Dec -> 2024.
  - "Day of the Week" is normalized to title case with values among:
      Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==============================
# CONFIGURATION
# ==============================

DEFAULT_INPUT_CSV: str = r"file\path\to\your\CLEVER_Runtime_and_OTP_by_Month.csv"

# Network paths provided by requester (escape backslashes if editing here).
DEFAULT_OUT_TABLE_DIR: str = r"folder\path\to\your\output"
DEFAULT_OUT_PLOTS_DIR: str = r"folder\path\to\your\plots"

# Current period indicator in 'YY-MM' format (YY two-digit year, MM two-digit month).
# '25-10' means October 2025.
DEFAULT_CURRENT_YY_MM: str = "25-10"  # Update with your own

# Agency OTP standard (fraction). 0.85 = 85%.
DEFAULT_OTP_STANDARD: float = 0.85

# Routes to exclude entirely from processing (e.g., test/fake routes that
# appear in the source extract but should not be reported).
# Entries are matched against `route_clean` (uppercase alphanumerics with
# everything before the first '-' kept and the rest dropped). The same
# cleaning is applied to entries here, so any of "999", "999 - Test",
# or "Route 999" will blacklist the same route.
# Override at the CLI with --blacklist-routes "999,888".
DEFAULT_BLACKLISTED_ROUTES: Tuple[str, ...] = ()

# Output filenames
OUTPUT_TABLE_FILENAME: str = "otp_processed.csv"
OUTPUT_TREND_ALL_FILENAME: str = "otp_trend_summary_all.txt"
OUTPUT_TREND_CONCERNING_FILENAME: str = "otp_trend_summary_concerning.txt"

# Default fraction of routes flagged as "most concerning" (e.g., 0.10 = bottom 10%).
DEFAULT_CONCERNING_PCT: float = 0.10

# Smoothing window (months) applied as a trailing rolling mean before fitting
# the trend. Filters out single-month blips so we measure sustained change.
TREND_SMOOTHING_WINDOW: int = 3

# Minimum number of raw monthly periods required to compute a trend at all.
# Below this, slope on a smoothed 12-month-or-less series isn't meaningful.
MIN_PERIODS_FOR_TREND: int = 6

# Horizon (in years) used when projecting current annualized trend into the
# concern score. concern_score = max(0, standard - current) + max(0, -trend_pp_per_year) * HORIZON
CONCERN_HORIZON_YEARS: float = 1.0

# ==============================
# DATA STRUCTURES
# ==============================


@dataclass(frozen=True)
class Config:
    """Runtime configuration."""

    input_csv: Path
    out_table_dir: Path
    out_plots_dir: Path
    current_yy_mm: str
    otp_standard: float
    concerning_pct: float
    blacklisted_routes: frozenset


# ==============================
# HELPERS
# ==============================

MONTH_ABBR_TO_NUM: Dict[str, int] = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

VALID_DOWS: Tuple[str, ...] = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)


def parse_current_yy_mm(current_yy_mm: str) -> Tuple[int, int]:
    """Parse 'YY-MM' like '25-10' into (year, month).

    Args:
        current_yy_mm: Two-digit year and month separated by '-', e.g., '25-10'.

    Returns:
        A tuple of (year, month), e.g., (2025, 10).

    Raises:
        ValueError: If the string is malformed.
    """
    if not re.fullmatch(r"\d{2}-\d{2}", current_yy_mm):
        raise ValueError(f"Invalid CURRENT_YY_MM format: {current_yy_mm!r}. Expected 'YY-MM'.")
    yy_str, mm_str = current_yy_mm.split("-")
    year = 2000 + int(yy_str)
    month = int(mm_str)
    if not (1 <= month <= 12):
        raise ValueError(f"Invalid month in CURRENT_YY_MM: {month}.")
    return year, month


def clean_route(route: str) -> str:
    """Return cleaned route key: take part before '-', strip non-alphanumerics, uppercase."""
    if route is None:
        return ""
    part = route.split("-", 1)[0]
    cleaned = re.sub(r"[^A-Za-z0-9]", "", part)
    return cleaned.upper()


def month_to_period(month_label: str, ref_year: int, ref_month: int) -> str:
    """Convert a month label (e.g., 'Apr') to 'YY-MM' using a 12-month window rule."""
    if not isinstance(month_label, str):
        raise ValueError(f"Invalid month label: {month_label!r}")
    key = month_label.strip().lower()[:3]
    if key not in MONTH_ABBR_TO_NUM:
        raise ValueError(f"Unrecognized month label: {month_label!r}")
    m = MONTH_ABBR_TO_NUM[key]
    y = ref_year if m <= ref_month else ref_year - 1
    return f"{str(y)[-2:]}-{m:02d}"


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Coerce string-like numeric series with commas/decimals to float."""
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan})
        .astype(float)
    )


def _normalize_dow(val: object) -> str:
    """Normalize a day-of-week value to title-case names Monday..Sunday."""
    s = str(val).strip()
    if not s:
        return ""
    t = s.lower()
    # Common variants
    mapping = {
        "mon": "Monday",
        "monday": "Monday",
        "tue": "Tuesday",
        "tues": "Tuesday",
        "tuesday": "Tuesday",
        "wed": "Wednesday",
        "weds": "Wednesday",
        "wednesday": "Wednesday",
        "thu": "Thursday",
        "thur": "Thursday",
        "thurs": "Thursday",
        "thursday": "Thursday",
        "fri": "Friday",
        "friday": "Friday",
        "sat": "Saturday",
        "saturday": "Saturday",
        "sun": "Sunday",
        "sunday": "Sunday",
    }
    return mapping.get(t, s.title())


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize input column names to expected canonical names."""
    colmap: Dict[str, str] = {}
    for c in df.columns:
        key = re.sub(r"\s+", " ", c.strip().lower())
        key = key.replace("#", "").replace("  ", " ")
        if key in {"route"}:
            colmap[c] = "route"
        elif key in {"direction"}:
            colmap[c] = "direction"
        elif key in {"month"}:
            colmap[c] = "month_label"
        elif key in {"day of the week", "day-of-week", "day_of_week", "dow"}:
            colmap[c] = "dow"
        elif key in {
            "sum on time",
            "sum ontime",
            "sum on_time",
            "sum on-time",
            "sum ontime trips",
            "sum on time trips",
        }:
            colmap[c] = "on_time"
        elif key in {"sum early", "sum early trips"}:
            colmap[c] = "early"
        elif key in {"sum late", "sum late trips"}:
            colmap[c] = "late"
        elif key in {"sum  on time", "sum  # on time", "sum # on time"}:
            colmap[c] = "on_time"
        elif key in {"sum  # early", "sum # early"}:
            colmap[c] = "early"
        elif key in {"sum  # late", "sum # late"}:
            colmap[c] = "late"
    df = df.rename(columns=colmap)
    required = {"route", "direction", "month_label", "dow", "on_time", "early", "late"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns after normalization: {sorted(missing)}")
    # Preserve original column order while ensuring required columns exist.
    ordered_required = ["route", "direction", "month_label", "dow", "on_time", "early", "late"]
    extra_cols = [c for c in df.columns if c not in ordered_required]
    cols = ordered_required + extra_cols
    return df[cols]


def process(
    df: pd.DataFrame,
    current_yy_mm: str,
    blacklisted_routes: frozenset = frozenset(),
) -> pd.DataFrame:
    """Compute totals and percentages and produce a tidy DataFrame.

    Args:
        df: Standardized input DataFrame.
        current_yy_mm: Reference period in 'YY-MM' form.
        blacklisted_routes: Optional set of route keys (will be passed through
            clean_route) to drop entirely. Useful for excluding test/fake routes.
    """
    ref_year, ref_month = parse_current_yy_mm(current_yy_mm)
    df = df.copy()

    # Drop fully-empty trailer rows (e.g., blank lines at end of CSV exports).
    # A row is "empty" if route, direction, and month_label are all NA/blank.
    key_cols = ["route", "direction", "month_label"]
    is_blank = (
        df[key_cols]
        .apply(
            lambda s: s.astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "<NA>": np.nan})
        )
        .isna()
        .all(axis=1)
    )
    n_blank = int(is_blank.sum())
    if n_blank:
        logging.info("Dropping %d empty rows (likely trailer padding).", n_blank)
        df = df.loc[~is_blank].reset_index(drop=True)

    df["route_raw"] = df["route"].astype(str).str.strip()
    df["route_clean"] = df["route_raw"].map(clean_route)

    # Apply route blacklist. Entries are run through clean_route too so users
    # can supply them in whatever form is convenient.
    if blacklisted_routes:
        normalized_blacklist = frozenset(
            cleaned for cleaned in (clean_route(r) for r in blacklisted_routes) if cleaned
        )
        if normalized_blacklist:
            mask = df["route_clean"].isin(normalized_blacklist)
            n_drop = int(mask.sum())
            if n_drop:
                dropped_routes = sorted(df.loc[mask, "route_clean"].unique())
                logging.info(
                    "Dropping %d rows from %d blacklisted route(s): %s",
                    n_drop,
                    len(dropped_routes),
                    ", ".join(dropped_routes),
                )
                df = df.loc[~mask].reset_index(drop=True)
            # Warn about blacklist entries that didn't match anything in the data
            present = set(df["route_clean"].unique()) | set(dropped_routes if n_drop else [])
            unused = sorted(normalized_blacklist - present)
            if unused:
                logging.warning(
                    "Blacklist entries with no matching rows in input: %s",
                    ", ".join(unused),
                )

    df["direction"] = df["direction"].astype(str).str.strip()
    df["month_label"] = df["month_label"].astype(str).str.strip()
    df["dow"] = df["dow"].map(_normalize_dow)
    # Validate DOW values (allow blanks; they will drop out in splits that require them)
    bad = ~df["dow"].isin(VALID_DOWS)
    if bad.any():
        # Keep but mark; user can inspect if needed.
        n_bad = int(bad.sum())
        logging.warning(
            "Found %d rows with unrecognized DOW values; "
            "they will be ignored in DOW-specific plots.",
            n_bad,
        )

    for col in ("on_time", "early", "late"):
        df[col] = coerce_numeric(df[col])

    df["period"] = df["month_label"].map(lambda m: month_to_period(m, ref_year, ref_month))
    df["total_trips"] = df[["on_time", "early", "late"]].sum(axis=1, skipna=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        df["pct_on_time"] = (df["on_time"] / df["total_trips"]) * 100.0
        df["pct_early"] = (df["early"] / df["total_trips"]) * 100.0
        df["pct_late"] = (df["late"] / df["total_trips"]) * 100.0

    order_cols = [
        "route_raw",
        "route_clean",
        "direction",
        "month_label",
        "period",
        "dow",
        "on_time",
        "early",
        "late",
        "total_trips",
        "pct_on_time",
        "pct_early",
        "pct_late",
    ]
    df = (
        df[order_cols].sort_values(by=["route_clean", "direction", "period"]).reset_index(drop=True)
    )
    return df


def ensure_dir(path: Path) -> None:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def export_table(df: pd.DataFrame, out_dir: Path, filename: str) -> Path:
    """Export processed table to CSV."""
    ensure_dir(out_dir)
    out_path = out_dir / filename
    df.to_csv(out_path, index=False)
    return out_path


# ==============================
# TREND SUMMARY (single-number per route/direction)
# ==============================


def _period_to_month_index(period: str) -> int:
    """Convert 'YY-MM' to a monotonically increasing month index (months since 2000-01)."""
    yy, mm = period.split("-")
    return (2000 + int(yy)) * 12 + int(mm)


def _trip_weighted_otp_by_period(df_subset: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a (route, direction, day-set) subset into one row per period.

    Returns a DataFrame with columns ['period', 'pct_on_time'] where pct_on_time
    is computed as sum(on_time) / sum(total_trips) * 100 across the included rows
    for that period (i.e., trip-weighted, so heavier service days dominate).
    """
    if df_subset.empty:
        return pd.DataFrame(columns=["period", "pct_on_time"])
    g = df_subset.groupby("period", as_index=False)[["on_time", "total_trips"]].sum(min_count=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        g["pct_on_time"] = np.where(
            g["total_trips"] > 0, (g["on_time"] / g["total_trips"]) * 100.0, np.nan
        )
    return g[["period", "pct_on_time"]].dropna(subset=["pct_on_time"])


def _slope_pp_per_month(per_period: pd.DataFrame) -> float:
    """Return OLS slope of pct_on_time vs month index, in percentage points per month.

    Returns NaN if fewer than 2 distinct periods are available.
    """
    if len(per_period) < 2:
        return float("nan")
    x = np.array([_period_to_month_index(p) for p in per_period["period"]], dtype=float)
    y = per_period["pct_on_time"].to_numpy(dtype=float)
    if np.unique(x).size < 2:
        return float("nan")
    # np.polyfit returns [slope, intercept]
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def _smoothed_series(per_period: pd.DataFrame, window: int) -> pd.DataFrame:
    """Apply a trailing rolling mean to pct_on_time.

    Per-period rows must already be sorted ascending by period. Drops leading
    rows that don't have a full window (so we never mix partial-window means
    with full-window means in the trend fit).
    """
    if per_period.empty or len(per_period) < window:
        return per_period.iloc[0:0]
    smoothed = per_period.copy()
    smoothed["pct_on_time"] = (
        smoothed["pct_on_time"].rolling(window=window, min_periods=window).mean()
    )
    return smoothed.dropna(subset=["pct_on_time"]).reset_index(drop=True)


def _summarize_subset(df_subset: pd.DataFrame) -> Dict[str, float]:
    """Compute trend, current, mean, and n_periods for a subset of rows.

    The trend metric is the slope of a trailing-rolling-mean smoothed series,
    annualized to percentage points per year. Smoothing filters single-month
    blips so the metric reflects sustained change rather than noise. Requires
    at least MIN_PERIODS_FOR_TREND raw periods; otherwise trend is NaN.

    The "current" value is the most recent smoothed (trailing rolling mean)
    OTP, not the most recent single month, for the same noise-rejection reason.
    """
    per_period = _trip_weighted_otp_by_period(df_subset)
    if per_period.empty:
        return {
            "n_periods": 0,
            "trend_pp_per_year": float("nan"),
            "current": float("nan"),
            "mean": float("nan"),
        }
    per_period = (
        per_period.assign(_idx=per_period["period"].map(_period_to_month_index))
        .sort_values("_idx")
        .reset_index(drop=True)
    )

    n = len(per_period)
    raw_mean = float(per_period["pct_on_time"].mean())

    # Trend on the smoothed series (only if enough raw months exist).
    if n >= MIN_PERIODS_FOR_TREND:
        smoothed = _smoothed_series(per_period, TREND_SMOOTHING_WINDOW)
        if len(smoothed) >= 2:
            slope_pp_per_month = _slope_pp_per_month(smoothed)
            trend_pp_per_year = slope_pp_per_month * 12.0
            current = float(smoothed["pct_on_time"].iloc[-1])
        else:
            trend_pp_per_year = float("nan")
            current = float(per_period["pct_on_time"].iloc[-1])
    else:
        trend_pp_per_year = float("nan")
        # Fall back to the raw most-recent value when we can't smooth.
        current = float(per_period["pct_on_time"].iloc[-1])

    return {
        "n_periods": int(n),
        "trend_pp_per_year": trend_pp_per_year,
        "current": current,
        "mean": raw_mean,
    }


def compute_trend_summary(df: pd.DataFrame, otp_standard: float) -> pd.DataFrame:
    """Build a per-(route, direction) trend summary.

    Headline metric is the WEEKDAY OTP trend in percentage points per year, fit
    on a trailing-3-month-rolling-mean smoothed series so single-month blips
    don't drive it. Saturday and Sunday trends are reported in the same units
    for reference.

    Concern score (higher = more concerning) combines:
      * how far the smoothed weekday OTP currently sits below the standard, plus
      * the projected further drop over CONCERN_HORIZON_YEARS at the current trend.

    Note: with only 12 months of input, trend cannot be cleanly separated from
    seasonality. The smoothing reduces single-month noise but does not remove
    seasonal effects. True year-over-year analysis requires >= 15 months of
    input data and a different period-mapping scheme.

    Args:
        df: Processed DataFrame from `process()`.
        otp_standard: OTP threshold as a fraction (e.g., 0.85).

    Returns:
        DataFrame with one row per (route_clean, direction).
    """
    std_pct = otp_standard * 100.0
    weekday_set = set(VALID_DOWS[:5])  # Mon..Fri

    rows: List[Dict[str, object]] = []
    for (route_clean, direction), g in df.groupby(["route_clean", "direction"], dropna=False):
        wd = _summarize_subset(g[g["dow"].isin(weekday_set)])
        sat = _summarize_subset(g[g["dow"] == "Saturday"])
        sun = _summarize_subset(g[g["dow"] == "Sunday"])

        # Concern score uses smoothed weekday values; NaN-safe.
        wd_trend = wd["trend_pp_per_year"]
        wd_current = wd["current"]
        gap_below = max(0.0, std_pct - wd_current) if not np.isnan(wd_current) else 0.0
        decline_per_year = max(0.0, -wd_trend) if not np.isnan(wd_trend) else 0.0
        concern_score = gap_below + decline_per_year * CONCERN_HORIZON_YEARS

        rows.append(
            {
                "route_clean": route_clean,
                "direction": direction,
                "n_periods_wd": wd["n_periods"],
                "trend_wd": wd["trend_pp_per_year"],
                "current_wd": wd["current"],
                "mean_wd": wd["mean"],
                "trend_sat": sat["trend_pp_per_year"],
                "current_sat": sat["current"],
                "trend_sun": sun["trend_pp_per_year"],
                "current_sun": sun["current"],
                "below_standard": (
                    bool(wd_current < std_pct) if not np.isnan(wd_current) else False
                ),
                "declining": (bool(wd_trend < 0) if not np.isnan(wd_trend) else False),
                "concern_score": concern_score,
            }
        )

    return pd.DataFrame(rows)


def _fmt_signed(val: float, width: int = 6, decimals: int = 2) -> str:
    """Format a signed float; show 'n/a' for NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "n/a".rjust(width)
    return f"{val:+{width}.{decimals}f}"


def _fmt_unsigned(val: float, width: int = 5, decimals: int = 1) -> str:
    """Format an unsigned float; show 'n/a' for NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "n/a".rjust(width)
    return f"{val:{width}.{decimals}f}"


def _flags_for_row(row: pd.Series, std_pct: float) -> str:
    """Compose a short flag string for the headline table."""
    flags: List[str] = []
    if row["declining"]:
        flags.append("↓ declining")
    if row["below_standard"]:
        flags.append("below std")
    return ", ".join(flags)


def format_trend_log(
    summary: pd.DataFrame,
    *,
    title: str,
    current_yy_mm: str,
    otp_standard: float,
    period_min: str,
    period_max: str,
    sort_by: str = "route",
) -> str:
    """Render the trend summary as a fixed-width plain-text log."""
    std_pct = otp_standard * 100.0

    if sort_by == "concern":
        summary = summary.sort_values(
            ["concern_score", "route_clean", "direction"], ascending=[False, True, True]
        )
    else:
        summary = summary.sort_values(["route_clean", "direction"])

    sep = "=" * 130
    sub = "-" * 130

    lines: List[str] = []
    lines.append(sep)
    lines.append(title)
    lines.append(sep)
    lines.append(f"Reference period   : {current_yy_mm}")
    lines.append(f"OTP standard       : {std_pct:.1f}%")
    lines.append(f"Periods analyzed   : {period_min} through {period_max}")
    lines.append(f"Routes/directions  : {len(summary)}")
    lines.append("")
    lines.append("Headline metric: WEEKDAY OTP trend, in percentage points per YEAR.")
    lines.append(
        f"  Computed on a trailing {TREND_SMOOTHING_WINDOW}-month rolling mean of "
        "trip-weighted OTP, then OLS slope * 12."
    )
    lines.append(
        "  Smoothing filters single-month blips so the metric reflects sustained change "
        "rather than noise."
    )
    lines.append(
        f"  Requires >= {MIN_PERIODS_FOR_TREND} periods of data; otherwise reported as n/a."
    )
    lines.append(
        "  CURR = most recent trailing-window OTP (%); MEAN = simple mean across all months (%)."
    )
    lines.append(
        f"  Concern score = max(0, {std_pct:.0f} - CURR_WD) + max(0, -TREND_WD) * "
        f"{CONCERN_HORIZON_YEARS:g}  (higher = more concerning)."
    )
    lines.append("")
    lines.append("Caveat: with ~12 months of input, trend cannot be fully separated from")
    lines.append("seasonality. Smoothing reduces single-month noise but a residual seasonal")
    lines.append("signal may remain. Year-over-year analysis (preferred) needs >= 15 months.")
    lines.append("")

    # Header
    header = (
        f"{'ROUTE':<8}{'DIRECTION':<22}"
        f"{'N':>4}  "
        f"{'TREND_WD':>9} {'CURR_WD':>8} {'MEAN_WD':>8}  "
        f"{'TREND_SAT':>10} {'CURR_SAT':>9}  "
        f"{'TREND_SUN':>10} {'CURR_SUN':>9}  "
        f"{'CONCERN':>8}  "
        f"FLAGS"
    )
    sub_header = (
        f"{'':<8}{'':<22}"
        f"{'':>4}  "
        f"{'(pp/yr)':>9} {'(%)':>8} {'(%)':>8}  "
        f"{'(pp/yr)':>10} {'(%)':>9}  "
        f"{'(pp/yr)':>10} {'(%)':>9}  "
        f"{'':>8}  "
    )
    lines.append(header)
    lines.append(sub_header)
    lines.append(sub)

    for _, row in summary.iterrows():
        route = str(row["route_clean"])[:8]
        direction = str(row["direction"])[:22]
        line = (
            f"{route:<8}{direction:<22}"
            f"{int(row['n_periods_wd']):>4}  "
            f"{_fmt_signed(row['trend_wd'], 9, 2)} "
            f"{_fmt_unsigned(row['current_wd'], 8, 1)} "
            f"{_fmt_unsigned(row['mean_wd'], 8, 1)}  "
            f"{_fmt_signed(row['trend_sat'], 10, 2)} "
            f"{_fmt_unsigned(row['current_sat'], 9, 1)}  "
            f"{_fmt_signed(row['trend_sun'], 10, 2)} "
            f"{_fmt_unsigned(row['current_sun'], 9, 1)}  "
            f"{_fmt_unsigned(row['concern_score'], 8, 2)}  "
            f"{_flags_for_row(row, std_pct)}"
        )
        lines.append(line)

    lines.append("")
    lines.append(sep)
    return "\n".join(lines) + "\n"


def export_trend_logs(
    proc: pd.DataFrame,
    out_dir: Path,
    *,
    current_yy_mm: str,
    otp_standard: float,
    concerning_pct: float,
) -> Tuple[Path, Path]:
    """Compute the trend summary and write the all-routes and concerning logs.

    Returns:
        (path_all, path_concerning)
    """
    ensure_dir(out_dir)

    summary = compute_trend_summary(proc, otp_standard=otp_standard)

    # Period span (for the header)
    periods_sorted = _sorted_periods(proc["period"])
    period_min = periods_sorted[0] if periods_sorted else "n/a"
    period_max = periods_sorted[-1] if periods_sorted else "n/a"

    # All-routes log
    all_text = format_trend_log(
        summary,
        title="OTP TREND SUMMARY — All Routes & Directions",
        current_yy_mm=current_yy_mm,
        otp_standard=otp_standard,
        period_min=period_min,
        period_max=period_max,
        sort_by="route",
    )
    path_all = out_dir / OUTPUT_TREND_ALL_FILENAME
    path_all.write_text(all_text, encoding="utf-8")

    # Concerning subset: top N by concern score, with at least 1 row.
    n = len(summary)
    if n == 0:
        k = 0
    else:
        k = max(1, int(np.ceil(n * concerning_pct)))
    concerning = summary.sort_values(
        ["concern_score", "route_clean", "direction"], ascending=[False, True, True]
    ).head(k)

    concerning_text = format_trend_log(
        concerning,
        title=(f"OTP TREND SUMMARY — Most Concerning {concerning_pct * 100:.0f}% ({k} of {n})"),
        current_yy_mm=current_yy_mm,
        otp_standard=otp_standard,
        period_min=period_min,
        period_max=period_max,
        sort_by="concern",
    )
    path_concerning = out_dir / OUTPUT_TREND_CONCERNING_FILENAME
    path_concerning.write_text(concerning_text, encoding="utf-8")

    return path_all, path_concerning


def _period_key(p: str) -> Tuple[int, int]:
    """Sort key for 'YY-MM' period strings."""
    yy, mm = p.split("-")
    return (2000 + int(yy), int(mm))


def _sorted_periods(series: pd.Series) -> List[str]:
    """Return sorted unique periods."""
    periods = sorted(series.dropna().unique(), key=_period_key)
    return list(periods)


def plot_series_for_groups(df: pd.DataFrame, out_dir: Path, otp_standard: float) -> None:
    """Create OTP plots split by Weekdays, Saturday, and Sunday.

    For each (route_clean, direction):
      - Weekdays (Mon–Fri):
          * Blue line = average OTP across Mon–Fri per period.
          * Light-blue band = min..max OTP among Mon–Fri per period.
          * Orange and green lines = average Early% and Late% across Mon–Fri per period.
      - Saturday:
          * Lines for OTP, Early%, Late% (period means).
      - Sunday:
          * Lines for OTP, Early%, Late% (period means).

    Args:
        df: Processed DataFrame with percent columns on a 0–100 scale and 'dow'.
        out_dir: Directory where plot PNGs are written.
        otp_standard: OTP threshold as a fraction (0.85 -> draw at 85%).
    """
    ensure_dir(out_dir)
    std_y = otp_standard * 100.0

    groups = df.groupby(["route_clean", "direction"], dropna=False)

    for (route_clean, direction), g in groups:
        g = g.copy()

        # Period ordering for x-axis
        periods = _sorted_periods(g["period"])
        if not periods:
            continue
        x = np.arange(len(periods))

        # ---------- Weekdays (Mon–Fri) ----------
        g_wd = g[g["dow"].isin(VALID_DOWS[:5])].copy()  # Mon..Fri
        if not g_wd.empty:
            # Per-period, per-weekday means for all three metrics
            wd_means = g_wd.groupby(["period", "dow"], as_index=False)[
                ["pct_on_time", "pct_early", "pct_late"]
            ].mean()

            # Average across weekdays per period (one value per period per metric)
            wd_avg = wd_means.groupby("period", as_index=False)[
                ["pct_on_time", "pct_early", "pct_late"]
            ].mean()

            # Range across weekdays per period for OTP only
            wd_min = (
                wd_means.groupby("period", as_index=False)["pct_on_time"]
                .min()
                .rename(columns={"pct_on_time": "min_otp"})
            )
            wd_max = (
                wd_means.groupby("period", as_index=False)["pct_on_time"]
                .max()
                .rename(columns={"pct_on_time": "max_otp"})
            )

            # Align to full period index
            avg_on = wd_avg.set_index("period")["pct_on_time"].reindex(periods)
            avg_erl = wd_avg.set_index("period")["pct_early"].reindex(periods)
            avg_lat = wd_avg.set_index("period")["pct_late"].reindex(periods)
            min_map = wd_min.set_index("period")["min_otp"].reindex(periods)
            max_map = wd_max.set_index("period")["max_otp"].reindex(periods)

            plt.figure()

            # Blue OTP average line — guard against empty return on all-NaN data
            # (matplotlib 3.9+ returns [] instead of [Line2D] for fully-NaN series).
            _lines = plt.plot(x, avg_on.values, marker="o", label="Weekday On-time % (avg Mon–Fri)")
            if _lines:
                line_on = _lines[0]

                # Light-blue band (OTP min–max range across Mon–Fri)
                base_color = line_on.get_color()
                plt.fill_between(
                    x,
                    min_map.to_numpy().astype(float),
                    max_map.to_numpy().astype(float),
                    alpha=0.2,
                    label="Weekday OTP range (min–max Mon–Fri)",
                    facecolor=base_color,
                    edgecolor="none",
                )

                # Early and Late average lines (defaults to Matplotlib colors)
                plt.plot(x, avg_erl.values, marker="o", label="Weekday Early % (avg Mon–Fri)")
                plt.plot(x, avg_lat.values, marker="o", label="Weekday Late % (avg Mon–Fri)")

                # Reference line
                plt.axhline(
                    y=std_y,
                    linestyle="--",
                    color="red",
                    linewidth=1,
                    label=f"OTP Standard ({otp_standard * 100:.0f}%)",
                )

                plt.xticks(ticks=x, labels=periods, rotation=45, ha="right")
                plt.ylim(0, 100)
                plt.xlabel("Period (YY-MM)")
                plt.ylabel("Percent of trips")
                title = f"{route_clean} — {direction} — Weekdays (Mon–Fri)"
                plt.title(title)
                plt.legend()
                plt.tight_layout()
                dir_slug = direction.replace("/", "-").replace(" ", "")
                fname = f"{route_clean}_{dir_slug}_Weekdays_otp_trend.png"
                out_path = out_dir / fname
                plt.savefig(out_path, dpi=150)

            plt.close()

        # ---------- Saturday ----------
        g_sat = g[g["dow"] == "Saturday"].copy()
        if not g_sat.empty:
            sat_means = g_sat.groupby("period", as_index=False)[
                ["pct_on_time", "pct_early", "pct_late"]
            ].mean()
            sat_on = sat_means.set_index("period")["pct_on_time"].reindex(periods)
            sat_erl = sat_means.set_index("period")["pct_early"].reindex(periods)
            sat_lat = sat_means.set_index("period")["pct_late"].reindex(periods)

            plt.figure()
            plt.plot(x, sat_on.values, marker="o", label="Saturday On-time %")
            plt.plot(x, sat_erl.values, marker="o", label="Saturday Early %")
            plt.plot(x, sat_lat.values, marker="o", label="Saturday Late %")
            plt.axhline(
                y=std_y,
                linestyle="--",
                color="red",
                linewidth=1,
                label=f"OTP Standard ({otp_standard * 100:.0f}%)",
            )
            plt.xticks(ticks=x, labels=periods, rotation=45, ha="right")
            plt.ylim(0, 100)
            plt.xlabel("Period (YY-MM)")
            plt.ylabel("Percent of trips")
            title = f"{route_clean} — {direction} — Saturday"
            plt.title(title)
            plt.legend()
            plt.tight_layout()
            fname = f"{route_clean}_{direction.replace('/', '-')}_Saturday_otp_trend.png".replace(
                " ", ""
            )
            out_path = out_dir / fname
            plt.savefig(out_path, dpi=150)
            plt.close()

        # ---------- Sunday ----------
        g_sun = g[g["dow"] == "Sunday"].copy()
        if not g_sun.empty:
            sun_means = g_sun.groupby("period", as_index=False)[
                ["pct_on_time", "pct_early", "pct_late"]
            ].mean()
            sun_on = sun_means.set_index("period")["pct_on_time"].reindex(periods)
            sun_erl = sun_means.set_index("period")["pct_early"].reindex(periods)
            sun_lat = sun_means.set_index("period")["pct_late"].reindex(periods)

            plt.figure()
            plt.plot(x, sun_on.values, marker="o", label="Sunday On-time %")
            plt.plot(x, sun_erl.values, marker="o", label="Sunday Early %")
            plt.plot(x, sun_lat.values, marker="o", label="Sunday Late %")
            plt.axhline(
                y=std_y,
                linestyle="--",
                color="red",
                linewidth=1,
                label=f"OTP Standard ({otp_standard * 100:.0f}%)",
            )
            plt.xticks(ticks=x, labels=periods, rotation=45, ha="right")
            plt.ylim(0, 100)
            plt.xlabel("Period (YY-MM)")
            plt.ylabel("Percent of trips")
            title = f"{route_clean} — {direction} — Sunday"
            plt.title(title)
            plt.legend()
            plt.tight_layout()
            fname = f"{route_clean}_{direction.replace('/', '-')}_Sunday_otp_trend.png".replace(
                " ", ""
            )
            out_path = out_dir / fname
            plt.savefig(out_path, dpi=150)
            plt.close()


def read_csv_safely(path: Path) -> pd.DataFrame:
    """Read CSV with basic error handling and helpful diagnostics."""
    try:
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Input CSV not found: {path}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV {path}: {e}") from e
    return df


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    p = argparse.ArgumentParser(description="Process OTP CSV, export table and plots.")
    p.add_argument("--input", type=str, default=DEFAULT_INPUT_CSV, help="Path to input CSV.")
    p.add_argument(
        "--out-table",
        type=str,
        default=DEFAULT_OUT_TABLE_DIR,
        help="Directory for processed CSV output.",
    )
    p.add_argument(
        "--out-plots",
        type=str,
        default=DEFAULT_OUT_PLOTS_DIR,
        help="Directory for plot PNG outputs.",
    )
    p.add_argument(
        "--current", type=str, default=DEFAULT_CURRENT_YY_MM, help="Current period in 'YY-MM'."
    )
    p.add_argument(
        "--otp-standard",
        type=float,
        default=DEFAULT_OTP_STANDARD,
        help="Agency OTP standard as a fraction (e.g., 0.85 for 85%).",
    )
    p.add_argument(
        "--concerning-pct",
        type=float,
        default=DEFAULT_CONCERNING_PCT,
        help=(
            "Fraction of route/direction groups to flag as 'most concerning' "
            "in the concerning .txt log (e.g., 0.10 for the top 10%%). "
            "Always rounds up to at least 1 row."
        ),
    )
    p.add_argument(
        "--blacklist-routes",
        type=str,
        default=None,
        help=(
            "Comma-separated list of routes to exclude (e.g., '999,888,TEST'). "
            "Matched against the cleaned route key, so any form is accepted. "
            "If omitted, uses DEFAULT_BLACKLISTED_ROUTES from the config section."
        ),
    )
    return p


# --- Default-filepath safety helpers ----------------------------------------
_PLACEHOLDER_MARKERS: tuple[str, ...] = (
    "path\\to\\",
    "path/to/",
    "your\\",
    "/your/",
    "\\your\\",
    "your/",
    "edit me",
    "edit here",
    "yyyy_mm",
    "your_gtfs_folder_path",
    "your_output_folder_path",
)


def _is_placeholder_path(p: object) -> bool:
    """Return True if *p* still points at a default placeholder location."""
    if p is None:
        return False
    s = str(p).lower()
    if not s:
        return False
    return any(marker in s for marker in _PLACEHOLDER_MARKERS)


# ==============================
# MAIN
# ==============================


def main(argv: List[str] | None = None) -> None:
    """Entrypoint.

    Args:
        argv: Optional explicit argv list (e.g., [] for notebooks). If None, uses sys.argv.
    """

    placeholders = {
        "DEFAULT_INPUT_CSV": DEFAULT_INPUT_CSV,
        "DEFAULT_OUT_TABLE_DIR": DEFAULT_OUT_TABLE_DIR,
        "DEFAULT_OUT_PLOTS_DIR": DEFAULT_OUT_PLOTS_DIR,
    }
    unset = [name for name, p in placeholders.items() if _is_placeholder_path(p)]
    if unset:
        logging.warning(
            "Default placeholder filepaths detected for: %s. "
            "Update the CONFIGURATION section of this script with real paths "
            "before running. Exiting without processing.",
            ", ".join(unset),
        )
        return
    parser = build_arg_parser()
    # Accept unknown args to be notebook/IPython friendly (swallows "-f <kernel.json>").
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        logging.warning("Ignoring unknown CLI args (likely from IPython): %s", unknown)
    # Parse the blacklist: CLI overrides the constant if provided.
    if args.blacklist_routes is not None:
        blacklist = frozenset(
            entry.strip() for entry in args.blacklist_routes.split(",") if entry.strip()
        )
    else:
        blacklist = frozenset(DEFAULT_BLACKLISTED_ROUTES)

    cfg = Config(
        input_csv=Path(args.input).expanduser(),
        out_table_dir=Path(args.out_table).expanduser(),
        out_plots_dir=Path(args.out_plots).expanduser(),
        current_yy_mm=args.current,
        otp_standard=args.otp_standard,
        concerning_pct=args.concerning_pct,
        blacklisted_routes=blacklist,
    )
    logging.info("Reading: %s", cfg.input_csv)
    raw = read_csv_safely(cfg.input_csv)
    logging.info("Rows read: %d", len(raw))
    logging.info("Normalizing columns...")
    norm = standardize_columns(raw)
    logging.info("Processing with CURRENT_YY_MM='%s'...", cfg.current_yy_mm)
    proc = process(norm, cfg.current_yy_mm, blacklisted_routes=cfg.blacklisted_routes)
    if proc.empty:
        logging.error(
            "No rows remain after processing. "
            "Check the input file and the route blacklist (currently: %s).",
            sorted(cfg.blacklisted_routes) if cfg.blacklisted_routes else "(empty)",
        )
        return
    logging.info("Exporting table to: %s", cfg.out_table_dir)
    out_table_path = export_table(proc, cfg.out_table_dir, OUTPUT_TABLE_FILENAME)
    logging.info("Wrote table: %s", out_table_path)
    logging.info("Exporting trend summary logs to: %s", cfg.out_table_dir)
    path_all, path_concerning = export_trend_logs(
        proc,
        cfg.out_table_dir,
        current_yy_mm=cfg.current_yy_mm,
        otp_standard=cfg.otp_standard,
        concerning_pct=cfg.concerning_pct,
    )
    logging.info("Wrote trend log (all):        %s", path_all)
    logging.info("Wrote trend log (concerning): %s", path_concerning)
    logging.info("Generating plots in: %s", cfg.out_plots_dir)
    plot_series_for_groups(proc, cfg.out_plots_dir, cfg.otp_standard)
    logging.info("Plot export complete.")
    n_groups = proc.groupby(["route_clean", "direction"], dropna=False).ngroups
    logging.info(
        "Processed %d rows across %d route/direction groups.",
        len(proc),
        n_groups,
    )
    logging.info("otp_monthly_trends_export.py completed successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
