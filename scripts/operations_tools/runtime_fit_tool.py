"""Trip-level runtime diagnostics and schedule performance review.

This module analyzes observed bus trips to evaluate how actual runtime and timing
deviations compare to scheduled values. It generates both row-level flags and
summary statistics, emphasizing on-time performance (OTP) and 85th-percentile runtime.

Designed to support schedule tuning, the script suggests time-of-day bands using
Fisher–Jenks segmentation and provides visual diagnostics for start time, runtime,
and deviation patterns.

Outputs per route include:
- CSV: Trips with deviations and OTP compliance flags
- XLSX: Summarized runtime and OTP statistics
- XLSX: Suggested time bands for runtime adjustment
- PNG: Diagnostic plots (e.g., runtime boxplots, schedule vs. 85th percentile)

Assumes route-wise CSVs of trip observations with key timestamp columns.
"""

from __future__ import annotations

import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable, Final, Iterable, List, Sequence, TypeAlias
# May need to comment out TypeAlias import for old ArcPro Python versions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import difflib

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_ROOT_DIR: Final[Path] = Path(r"Path\To\Your\Data_Folder_with_observed_trips")
OUTPUT_ROOT_DIR: Final[Path] = Path(r"Path\To\Your\Output_Folder")

ROUTES_TO_INCLUDE: Final[set[str]] = {"101", "202"}  # optional whitelist

DATE_START: Final[pd.Timestamp] = pd.Timestamp("2024-06-30")
DATE_END: Final[pd.Timestamp] = pd.Timestamp("2025-07-24")

LOW_SAMPLE_FRAC: Final[float] = 0.20  # 20 % of the median n_events

MAX_TIME_BANDS: Final[int | None] = 6  # None ⇒ no hard cap
ENFORCE_MIN_BAND_SIZE: Final[bool] = True  # toggle merging on/off
MIN_BAND_SIZE: Final[int] = 2  # ignored when above is False

WRITE_EXCLUSIONS: Final[bool] = True  # write events_excluded_*.csv
SPLIT_BY_DIRECTION: Final[bool] = True  # keep current folder depth; False → flat

# Paired stems so all writers reference one source of truth
_RETAINED_STEM: Final[str] = "events_retained"
_EXCLUDED_STEM: Final[str] = "events_excluded"

EXCLUDE_DATES: Final[list[str]] = [
    "2025-01-01",
    "2025-01-20",
    "2025-02-17",
    "2025-05-26",
    "2025-06-19",
    "2025-07-04",
    "2025-09-01",
    "2025-10-13",
    "2025-11-11",
    "2025-11-27",
    "2025-11-28",
    "2025-12-25",
]

SERVICE_DAY_FILTER: Final[str | None] = "WEEKDAY"  # "SATURDAY" | "SUNDAY" | None

OTP_EARLY_MIN: Final[int] = -1
OTP_LATE_MIN: Final[int] = 6
OTP_TARGET_PCT: Final[float] = 85.0

TIME_COL_NAME: Final[str] = "trip_start_time"
_DOW_CHOICES: Final[set[str | None]] = {None, "", "WEEKDAY", "SATURDAY", "SUNDAY"}

# Outlier‑trimming settings (per‑trip)
TRIM_OUTLIERS: Final[bool] = True
TRIM_FRAC: Final[float] = 0.01  # drop shortest & longest 1 %

# ─── derived paths (initial stubs – reassigned per route in main()) ─── #
OUTPUT_DIR: Path = OUTPUT_ROOT_DIR
PLOTS_DIR: Path = OUTPUT_DIR / "plots"

# Valid directions to enforce and normalize to (keep UPPERCASE strings)
ALLOWED_DIRECTIONS: List[str] = [
    "NORTHBOUND",
    "SOUTHBOUND",
    "EASTBOUND",
    "WESTBOUND",
    "LOOP",
]

# ---------------------------------------------------------------------
# TYPE ALIASES
# ---------------------------------------------------------------------

PlotFunc: TypeAlias = Callable[[pd.DataFrame], None]

# =============================================================================
# FUNCTIONS
# =============================================================================

def _detect_sep(path: Path) -> str:
    """Return delimiter based on extension (csv → comma, others → tab)."""
    return "," if path.suffix.lower() == ".csv" else "\t"


def _ensure_plot_dirs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _trim_pct(series: pd.Series, frac: float = TRIM_FRAC) -> pd.Series:
    """Trim the lowest and highest *frac* proportion of values."""
    if series.empty or frac <= 0:
        return series
    lo = series.quantile(frac)
    hi = series.quantile(1 - frac)
    return series[(series >= lo) & (series <= hi)]


def _safe_plot(plot_func: PlotFunc, df: pd.DataFrame) -> None:
    """Safely call a plotting function if the DataFrame is not empty.

    Args:
        plot_func: A function that takes a DataFrame and produces a plot.
        df: Data to plot. If empty, the plot function is not called.
    """
    if not df.empty:
        plot_func(df)
    else:
        print("   ⚠  No rows after filters; skipping plots.")


def normalize_direction_value(value: str, allowed: List[str]) -> str:
    """Coerce a free-text direction to one of the allowed values.

    Uses difflib to guess the closest allowed label. Always returns one of
    ``allowed``. Emits a warning if the original does not exactly match.

    Examples:
        "NOTHTBOUND" -> "NORTHBOUND"
        "west bound" -> "WESTBOUND"
    """
    val = (value or "").strip().upper()
    if val in allowed:
        return val
    guess = difflib.get_close_matches(val, allowed, n=1, cutoff=0.6)
    if guess:
        fixed = guess[0]
        warnings.warn(f"Direction normalized: {value!r} → {fixed!r}", RuntimeWarning, stacklevel=2)
        return fixed
    warnings.warn(f"Unrecognized direction {value!r}; defaulting to 'LOOP'", RuntimeWarning, stacklevel=2)
    return "LOOP"


def normalize_directions_column(df: pd.DataFrame, allowed: List[str]) -> pd.DataFrame:
    """Normalize the 'Direction' column in place to the allowed list."""
    df = df.copy()
    df["Direction"] = df["Direction"].astype(str).map(lambda s: normalize_direction_value(s, allowed))
    return df


# -----------------------------------------------------------------------------
# ROUTE DISCOVERY HELPERS
# -----------------------------------------------------------------------------

_route_token_re = re.compile(r"([0-9]{1,4})")


def _clean_route_id(raw: str | float | int) -> str:
    """Canonical 1‑to‑4 digit route ID from a Route cell."""
    txt = str(raw)
    m = _route_token_re.search(txt)
    if not m:
        raise ValueError(f"Cannot parse route from value {txt!r}")
    return m.group(1).lstrip("0") or "0"


def _discover_route_csvs(root: Path, wanted: set[str] | None = None) -> dict[str, list[Path]]:
    """Crawl *root* recursively and build {route → [files]}.

    * A file is linked to **every** route ID that appears in its Route column,
      so mixed files get processed by each relevant route run.
    * Route column header may be “route”, “Route_ID”, “routeName”, etc.
    * Honors *wanted* whitelist (1‑to‑4 digit IDs with leading zeros stripped).
    """
    buckets: dict[str, list[Path]] = defaultdict(list)
    route_hdr_re = re.compile(r"\s*route\w*\s*", flags=re.I)

    for p in root.rglob("*.csv"):
        try:
            # read just the header to find the Route‑like column
            header = pd.read_csv(p, sep=_detect_sep(p), nrows=0).columns
            route_col = next(col for col in header if route_hdr_re.fullmatch(col))
        except StopIteration:
            print(f"!! {p.name}: no Route column; skipped")
            continue
        except Exception as e:
            print(f"!! {p.name}: {e}; skipped")
            continue

        try:
            routes = (
                pd.read_csv(
                    p,
                    sep=_detect_sep(p),
                    usecols=[route_col],
                    dtype=str,
                    low_memory=False,
                )[route_col]
                .dropna()
                .unique()
            )
            ids = {_clean_route_id(r) for r in routes}
        except Exception as e:
            print(f"!! {p.name}: {e}; skipped")
            continue

        if wanted:
            ids &= wanted
        if not ids:
            continue  # file has no wanted routes

        for rid in ids:
            buckets[rid].append(p)

    return buckets


def load_trip_files(files: Iterable[Path]) -> pd.DataFrame:
    """Load and concatenate observed trip CSV files.

    Args:
        files: List of file paths to read.

    Returns:
        Combined DataFrame with parsed time columns and unified schema.
    """
    frames = [pd.read_csv(p, sep=_detect_sep(p), dtype=str, low_memory=False) for p in files]
    df = pd.concat(frames, ignore_index=True)
    for col in (
        "Scheduled Start Time",
        "Scheduled Finish Time",
        "Actual Start Time",
        "Actual Finish Time",
    ):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def extract_trip_start_time(df: pd.DataFrame, trip_col: str = "Trip") -> pd.DataFrame:
    """Extract trip start time (HH:MM) from a Trip column.

    Args:
        df: Trip data with a string-based Trip column.
        trip_col: Name of the Trip column to extract time from.

    Returns:
        Copy of input DataFrame with a new 'trip_start_time' column.
    """
    df = df.copy()
    df[TIME_COL_NAME] = df[trip_col].str.extract(r"^\s*([0-2]?\d:[0-5]\d)")[0]
    return df


def filter_date_range(df: pd.DataFrame) -> pd.DataFrame:
    """Filter trips to only those within the configured date range.

    Args:
        df: Trip records with 'Scheduled Start Time' as datetime.

    Returns:
        Filtered DataFrame with rows inside DATE_START and DATE_END.
    """
    return df.loc[df["Scheduled Start Time"].between(DATE_START, DATE_END)].copy()


def filter_routes(df: pd.DataFrame, wanted: set[str]) -> pd.DataFrame:
    """Filter trips to only those matching desired route IDs.

    Args:
        df: Trip data with a 'Route' column.
        wanted: Canonical route IDs (1–4 digits, leading zeros removed).

    Returns:
        Subset of DataFrame containing only wanted route IDs.
    """
    wanted_canon = {str(x).lstrip("0") for x in wanted}
    route_num = df["Route"].astype(str).str.extract(r"^\s*([0-9]{1,4})")[0].str.lstrip("0")
    return df.loc[route_num.isin(wanted_canon)].copy()


def filter_holidays(df: pd.DataFrame, dates: Iterable[str]) -> pd.DataFrame:
    """Remove trips that occur on excluded holiday dates.

    Args:
        df: Trip data with 'Scheduled Start Time' as datetime.
        dates: Iterable of holiday dates as YYYY-MM-DD strings.

    Returns:
        DataFrame excluding rows on the given dates.
    """
    bad = {pd.to_datetime(d).date() for d in dates}
    return df.loc[~df["Scheduled Start Time"].dt.date.isin(bad)].copy()


def filter_service_day(df: pd.DataFrame, which: str | None) -> pd.DataFrame:
    """Filter trips by service day: WEEKDAY, SATURDAY, or SUNDAY.

    Args:
        df: Trip data with 'Scheduled Start Time' as datetime.
        which: Day type to keep. Must be 'WEEKDAY', 'SATURDAY', 'SUNDAY', or None.

    Returns:
        DataFrame filtered to the selected day type.
    """
    if which not in _DOW_CHOICES:
        raise ValueError("SERVICE_DAY_FILTER must be WEEKDAY, SATURDAY, SUNDAY or None")
    if not which:
        return df
    dow = df["Scheduled Start Time"].dt.dayofweek
    keep = (dow <= 4) if which == "WEEKDAY" else (dow == 5 if which == "SATURDAY" else dow == 6)
    return df.loc[keep].copy()


def add_deviation_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add schedule vs. actual time deviation columns to the DataFrame.

    Args:
        df: Trip data with scheduled and actual time columns.

    Returns:
        DataFrame with new columns for start/finish/runtime deviations.
    """
    df = df.copy()
    df["start_dev_min"] = (
        df["Actual Start Time"] - df["Scheduled Start Time"]
    ).dt.total_seconds() / 60
    df["finish_dev_min"] = (
        df["Actual Finish Time"] - df["Scheduled Finish Time"]
    ).dt.total_seconds() / 60
    df["scheduled_runtime_min"] = (
        df["Scheduled Finish Time"] - df["Scheduled Start Time"]
    ).dt.total_seconds() / 60
    df["actual_runtime_min"] = (
        df["Actual Finish Time"] - df["Actual Start Time"]
    ).dt.total_seconds() / 60
    df["runtime_dev_min"] = df["actual_runtime_min"] - df["scheduled_runtime_min"]
    return df


def add_otp_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add both legacy *on_time* and new *within_window* boolean columns.

    The columns are identical for one release cycle so downstream code
    can migrate from *on_time* to *within_window* at leisure.
    """
    flag = df["start_dev_min"].between(
        OTP_EARLY_MIN,
        OTP_LATE_MIN,
        inclusive="both",
    )
    return df.assign(on_time=flag, within_window=flag)


def _box_by_trip(
    df: pd.DataFrame,
    col: str,
    title: str,
    file_name: str,
    shade_range: tuple[float, float] | None = None,
) -> None:
    data = df[[TIME_COL_NAME, col]].dropna()
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=data, x=TIME_COL_NAME, y=col, whis=(0, 100), showfliers=False)
    if shade_range:
        plt.axhspan(*shade_range, color="g", alpha=0.15)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title(title)
    plt.ylabel("Minutes")
    plt.xlabel("Scheduled trip start")
    plt.xticks(rotation=90)
    plt.tight_layout()
    _ensure_plot_dirs()
    plt.savefig(PLOTS_DIR / file_name, dpi=150)
    plt.close()


def plot_start_dev_shaded(df: pd.DataFrame) -> None:
    """Create a boxplot of start-time deviations with OTP shading.

    Args:
        df: Trip data with 'start_dev_min' and 'trip_start_time' columns.
    """
    _box_by_trip(
        df,
        "start_dev_min",
        "Start‑time deviation – shaded OTP window",
        "box_start_dev_shaded.png",
        shade_range=(OTP_EARLY_MIN, OTP_LATE_MIN),
    )


def plot_start_dev_plain(df: pd.DataFrame) -> None:
    """Box‑plot start‑time deviation without OTP shading.

    Args:
        df: Fully filtered trip‑level data containing
            ``'start_dev_min'`` and ``'trip_start_time'`` columns.
    """
    _box_by_trip(
        df,
        "start_dev_min",
        "Start‑time deviation",
        "box_start_dev.png",
    )


def plot_finish_dev_shaded(df: pd.DataFrame) -> None:
    """Box‑plot finish‑time deviation with the OTP window shaded.

    Args:
        df: Trip‑level DataFrame with ``'finish_dev_min'`` present.
    """
    _box_by_trip(
        df,
        "finish_dev_min",
        "Finish‑time deviation – shaded OTP window",
        "box_finish_dev_shaded.png",
        shade_range=(OTP_EARLY_MIN, OTP_LATE_MIN),  # same thresholds for illustration
    )


def plot_finish_dev_plain(df: pd.DataFrame) -> None:
    """Box‑plot finish‑time deviation without shading.

    Args:
        df: Trip‑level DataFrame with ``'finish_dev_min'`` present.
    """
    _box_by_trip(
        df,
        "finish_dev_min",
        "Finish‑time deviation",
        "box_finish_dev.png",
    )


def plot_runtime_dev(df: pd.DataFrame) -> None:
    """Box‑plot runtime deviation (actual – scheduled) by trip.

    Args:
        df: Trip‑level DataFrame with ``'runtime_dev_min'`` present.
    """
    _box_by_trip(
        df,
        "runtime_dev_min",
        "Runtime deviation by trip",
        "box_runtime_dev.png",
    )


def _day_tag() -> str:
    return (SERVICE_DAY_FILTER or "ALL").upper()


def write_row_level(df: pd.DataFrame) -> None:
    """CSV of events retained for analysis (after all filters)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lead = ["Route", "Direction", "TripID", TIME_COL_NAME]
    ordered = df[lead + [c for c in df.columns if c not in lead]].sort_values(
        by=["Route", "Direction", TIME_COL_NAME, "TripID"],
        ignore_index=True,
    )
    ordered.to_csv(
        OUTPUT_DIR / f"{_RETAINED_STEM}_{_day_tag()}.csv",
        index=False,
    )


def _sched_mode_with_warning(s: pd.Series, trip_id: str | int) -> float:
    """Return the (possibly multi‑modal) mode of *s*, warning if ambiguous."""
    modes = s.mode(dropna=True)
    if modes.empty:
        return float("nan")
    if len(modes) > 1:
        warnings.warn(
            f"TripID {trip_id} has multiple scheduled runtimes "
            f"{modes.tolist()}; using their median.",
            RuntimeWarning,
            stacklevel=2,
        )
    return modes.median() if len(modes) > 1 else modes.iloc[0]


def write_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise each start-time token and save an XLSX sheet."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trip_grp = df.groupby(TIME_COL_NAME, sort=False)
    summary = trip_grp.agg(
        n_events=("Actual Start Time", lambda s: s.notna().sum()),
        otp_pct=("on_time", lambda s: s.mean() * 100.0),
    )

    summary["scheduled_runtime_mode"] = trip_grp["scheduled_runtime_min"].apply(
        lambda s: _sched_mode_with_warning(s, trip_id=s.name)
    )

    def _runtime_stats(series: pd.Series) -> pd.Series:
        data = _trim_pct(series) if TRIM_OUTLIERS else series
        return pd.Series(
            {
                "runtime_mean_min": data.mean(),
                "runtime_median_min": data.median(),
                "runtime_p85_min": data.quantile(0.85),
            }
        )

    summary = summary.join(trip_grp["actual_runtime_min"].apply(_runtime_stats).unstack())

    # New, clearer columns kept in parallel for one cycle
    summary["pct_within_window"] = summary["otp_pct"]
    summary["under_target"] = summary["otp_pct"] < OTP_TARGET_PCT
    summary["below_target_pct"] = summary["under_target"]

    # Chronological order improves readability
    summary.reset_index(drop=False, inplace=True)
    summary.sort_values(by=TIME_COL_NAME, inplace=True, ignore_index=True)

    summary.to_excel(
        OUTPUT_DIR / f"trip_summary_{_day_tag()}.xlsx",
        index=False,
        engine="openpyxl",
    )
    return summary


def plot_runtime_p85_vs_sched(df: pd.DataFrame) -> None:
    """Bar‐plot scheduled vs. 85th‑percentile runtime for each start time.

    The function computes, for every distinct ``trip_start_time`` token:

    * **Scheduled runtime** – the modal (most common) scheduled runtime
      across trips starting at that time (with ambiguity warnings via
      ``_sched_mode_with_warning``).

    * **85th‑percentile actual runtime** – calculated on actual runtimes
      after optional outlier trimming (controlled by ``TRIM_OUTLIERS`` and
      ``TRIM_FRAC``).

    A grouped bar chart is saved to
    ``PLOTS_DIR / "bar_runtime_p85_vs_sched.png"``.  It visually highlights
    start‑times where the observed P85 runtime exceeds the scheduled value.

    Parameters
    ----------
    df : pandas.DataFrame
        Fully filtered trip‑level data for a single route.

    Notes:
    -----
    * Relies on the global constants ``TIME_COL_NAME``, ``TRIM_OUTLIERS``,
      and ``TRIM_FRAC`` defined earlier in the module.
    * Uses the helper ``_sched_mode_with_warning`` for scheduled runtimes
      and ``_trim_pct`` for optional outlier removal.
    """
    if df.empty:
        print("   ⚠  No rows after filters; skipping plots.")
        return

    # ── 1.  Aggregate per start‑time token ────────────────────────────
    grp = df.groupby(TIME_COL_NAME, sort=False)

    sched_runtime = grp["scheduled_runtime_min"].apply(
        lambda s: _sched_mode_with_warning(s, trip_id=s.name)
    )

    def _p85(series: pd.Series) -> float:
        data = _trim_pct(series, TRIM_FRAC) if TRIM_OUTLIERS else series
        return data.quantile(0.85)

    p85_runtime = grp["actual_runtime_min"].apply(_p85)

    summary = pd.DataFrame(
        {
            "trip_start_time": sched_runtime.index,
            "scheduled_min": sched_runtime.values,
            "p85_min": p85_runtime.values,
        }
    ).dropna(subset=["scheduled_min", "p85_min"])

    if summary.empty:
        print("   ⚠  No valid data for runtime P85 vs. scheduled plot.")
        return

    # ── 2.  Reshape to long format for seaborn ────────────────────────
    tidy = summary.melt(
        id_vars="trip_start_time",
        value_vars=["scheduled_min", "p85_min"],
        var_name="type",
        value_name="runtime_min",
    )

    # ── 3.  Plot ──────────────────────────────────────────────────────
    n_tokens = summary.shape[0]
    fig_w = max(12, n_tokens * 0.45)  # widen for many start‑times
    plt.figure(figsize=(fig_w, 6))
    sns.barplot(
        data=tidy,
        x="trip_start_time",
        y="runtime_min",
        hue="type",
        dodge=True,
    )

    plt.title("Scheduled vs. 85th‑percentile runtime by trip start time")
    plt.xlabel("Scheduled trip start (HH:MM)")
    plt.ylabel("Runtime (minutes)")
    plt.xticks(rotation=90)
    plt.legend(title="", labels=["Scheduled", "85th‑percentile"], loc="upper right")
    plt.tight_layout()

    _ensure_plot_dirs()
    plt.savefig(PLOTS_DIR / "bar_runtime_p85_vs_sched.png", dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# DIRECTION HELPER
# -----------------------------------------------------------------------------

_DIR_SLUG_RE: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+", flags=re.I)


def _dir_slug(value: str | int | float | None) -> str:
    """Return a filesystem‑safe slug for *value* (e.g., `0_Westbound`)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unknown"

    txt = str(value).strip()
    # Common English labels → deterministic slugs so that 0/1 and text agree
    canonical = {
        "0": "0_Westbound",
        "1": "1_Eastbound",
        "2": "2_Northbound",
        "3": "3_Southbound",
        "WESTBOUND": "0_Westbound",
        "EASTBOUND": "1_Eastbound",
        "NORTHBOUND": "2_Northbound",
        "SOUTHBOUND": "3_Southbound",
    }.get(txt.upper())

    if canonical:
        return canonical

    # Fallback – strip junk and collapse whitespace to “safe” slug
    clean = _DIR_SLUG_RE.sub("_", txt).strip("_")
    return clean or "unknown"


def export_trimmed_outliers(
    df: pd.DataFrame,
    *,
    group_key: str = TIME_COL_NAME,
    runtime_col: str = "actual_runtime_min",
    frac: float = TRIM_FRAC,
) -> None:
    """Write a CSV of rows dropped by the ±*frac* quantile filter."""
    if df.empty or frac <= 0 or not WRITE_EXCLUSIONS:
        return

    keep_frames: list[pd.DataFrame] = []

    for _, sub in df.groupby(group_key, sort=False):
        runtimes = sub[runtime_col]
        if runtimes.empty:
            continue
        lo, hi = runtimes.quantile([frac, 1 - frac])
        mask = (runtimes < lo) | (runtimes > hi)
        if mask.any():
            keep_frames.append(sub.loc[mask])

    if not keep_frames:
        return

    outliers = pd.concat(keep_frames, ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fname = OUTPUT_DIR / f"{_EXCLUDED_STEM}_{_day_tag()}.csv"
    outliers.to_csv(fname, index=False)
    print(f"   ⤷ {len(outliers):,} excluded rows ➜ {fname.name}")


def log_low_sample_start_times(
    df: pd.DataFrame,
    thresh_frac: float = LOW_SAMPLE_FRAC,
    *,
    exclude_dates: Iterable[str] | None = None,
) -> None:
    """Save a CSV listing start‑times that have *unusually few* observations.

    A start‑time (e.g. “06:15”) is flagged when its row count is **below
    thresh_frac × median(row counts)** across all start‑times in *df*.

    The CSV (one per route) lives alongside the other artefacts and
    contains:

    * `trip_start_time – HH:MM string extracted earlier
    * `n_obs            – observations for that token after filtering
    * `dates_run        – comma‑separated YYYY‑MM‑DD values

    Parameters
    ----------
    df : pandas.DataFrame
        The fully filtered route‑level dataframe.
    thresh_frac : float, default `LOW_SAMPLE_FRAC
        Fraction of the median observation count below which a
        start‑time is considered under‑sampled.
    exclude_dates : Iterable[str] | None, optional
        Your current `EXCLUDE_DATES list (YYYY‑MM‑DD strings).
        Supplying it lets the function tell you which flagged dates are
        *not yet* excluded.
    """
    # ------------------------------------------------------------------ #
    # 1.  Observation counts per start‑time and threshold calculation.   #
    # ------------------------------------------------------------------ #
    counts = df.groupby(TIME_COL_NAME)["Actual Start Time"].count()
    if counts.empty:
        return  # no rows after earlier filters

    cutoff = counts.median() * thresh_frac
    sparse_tokens = counts[counts < cutoff]
    if sparse_tokens.empty:
        return  # nothing to flag

    # ------------------------------------------------------------------ #
    # 2.  Assemble diagnostic rows, including the dates run.             #
    # ------------------------------------------------------------------ #
    warned_df = (
        df[df[TIME_COL_NAME].isin(sparse_tokens.index)]
        .loc[:, [TIME_COL_NAME, "Scheduled Start Time"]]
        .assign(n_obs=lambda x: x.groupby(TIME_COL_NAME)["Scheduled Start Time"].transform("size"))
    )

    out = (
        warned_df.groupby(TIME_COL_NAME, sort=False)
        .agg(
            n_obs=("n_obs", "first"),
            dates_run=(
                "Scheduled Start Time",
                lambda s: ", ".join(sorted({d.strftime("%Y-%m-%d") for d in s.dt.date})),
            ),
        )
        .reset_index()
    )

    # ------------------------------------------------------------------ #
    # 3.  Write the CSV and emit an actionable console message.          #
    # ------------------------------------------------------------------ #
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fname = OUTPUT_DIR / f"low_sample_start_times_{_day_tag()}.csv"
    out.to_csv(fname, index=False)

    print(
        f"   ⚠  {len(out)} low‑sample start‑times logged "
        f"(<{thresh_frac:.0%} of median obs) ➜ {fname.name}"
    )

    # If an EXCLUDE_DATES list is provided, point out any new dates.
    if exclude_dates is not None:
        flagged_dates: set[str] = {
            d for dates_str in out["dates_run"] for d in dates_str.split(", ")
        }
        missing = flagged_dates - {str(d) for d in exclude_dates}
        if missing:
            print(f"      ↪ Consider adding these to EXCLUDE_DATES: {sorted(missing)}")


def _cum_sums(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    csum = np.insert(np.cumsum(arr), 0, 0.0)
    csum_sq = np.insert(np.cumsum(arr**2), 0, 0.0)
    return csum, csum_sq


def _ssq(csum: np.ndarray, csum_sq: np.ndarray, i: int, j: int) -> float:
    n = j - i
    if n == 0:
        return 0.0
    s, s2 = csum[j] - csum[i], csum_sq[j] - csum_sq[i]
    return s2 - (s * s) / n


def _fisher_jenks(values: Sequence[float], k: int) -> List[int]:
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if not 2 <= k <= n:
        raise ValueError("k must be in 2‥n")

    csum, csum_sq = _cum_sums(arr)
    dp = np.full((k, n + 1), np.inf)
    idx = np.zeros((k, n + 1), dtype=int)

    dp[0, 1:] = [_ssq(csum, csum_sq, 0, m) for m in range(1, n + 1)]

    for g in range(1, k):
        for m in range(g + 1, n + 1):
            best, best_s = np.inf, g
            for s in range(g, m):
                cost = dp[g - 1, s] + _ssq(csum, csum_sq, s, m)
                if cost < best:
                    best, best_s = cost, s
            dp[g, m], idx[g, m] = best, best_s

    bkpts, m = [], n
    for g in range(k - 1, 0, -1):
        s = idx[g, m]
        bkpts.append(s)
        m = s
    return sorted(bkpts)  # len == k‑1


def _hhmm_to_minutes(s: pd.Series) -> pd.Series:
    """Convert an `HH:MM string to minutes after midnight.

    Accepts hours `00 through 29 so that after‑midnight tokens
    such as `"24:05" parse without error.  Any value that cannot be
    interpreted returns `NaN (float) so it can be removed later with
    `dropna().

    Parameters
    ----------
    s : pandas.Series
        Series of strings in *HH:MM* format.

    Returns:
    -------
    pandas.Series
        Numeric minutes after midnight (float); invalid inputs → NaN.
    """
    parts = s.str.split(":", n=1, expand=True)
    h = pd.to_numeric(parts[0], errors="coerce")
    m = pd.to_numeric(parts[1], errors="coerce")
    return h * 60 + m


# -----------------------------------------------------------------------------
#  PUBLIC API
# -----------------------------------------------------------------------------
def suggest_time_bands(
    summary: pd.DataFrame,
    *,
    max_bands: int | None = MAX_TIME_BANDS,
    enforce_min_size: bool = ENFORCE_MIN_BAND_SIZE,
    min_band_size: int = MIN_BAND_SIZE,
) -> pd.DataFrame:
    """Create Fisher–Jenks time‑of‑day bands from the 85th‑percentile runtime.

    Handles start‑time tokens up to 29:59 and works with older pandas
    versions that lack `reset_index(names=...).
    """
    need = {"trip_start_time", "runtime_p85_min"}
    miss = need - set(summary.columns)
    if miss:
        raise KeyError(f"summary missing columns {miss}")

    # ── 1. prepare ordered DF with numeric surrogate _t ──────────────
    df = (
        summary[["trip_start_time", "runtime_p85_min"]]
        .assign(_t=_hhmm_to_minutes(summary["trip_start_time"]))
        .dropna(subset=["_t", "runtime_p85_min"])
        .sort_values("_t", kind="mergesort")
        .reset_index(drop=True)
    )

    # ── 2. Fisher–Jenks segmentation ─────────────────────────────────
    n = len(df)
    k0 = max(int(np.ceil(np.sqrt(n))), 2)
    k = k0 if max_bands is None or max_bands <= 0 else min(k0, max_bands)

    breaks = _fisher_jenks(df["runtime_p85_min"].to_numpy(), k=k)

    labels = np.zeros(n, dtype=int)
    for i, b in enumerate(breaks, start=1):
        labels[b:] += 1
    df["_band"] = labels

    # ── 3. optional merge of undersized bands ─────────────────────────
    if enforce_min_size and min_band_size > 1:
        changed = True
        while changed:
            sizes = df["_band"].value_counts().sort_index()
            small = sizes[sizes < min_band_size].index
            if small.empty:
                changed = False
                continue
            for bid in small:
                idx = sizes.index.get_loc(bid)
                opts = []
                if idx > 0:
                    left = sizes.index[idx - 1]
                    diff = abs(
                        df.loc[df["_band"] == bid, "runtime_p85_min"].mean()
                        - df.loc[df["_band"] == left, "runtime_p85_min"].mean()
                    )
                    opts.append((left, diff))
                if idx < len(sizes) - 1:
                    right = sizes.index[idx + 1]
                    diff = abs(
                        df.loc[df["_band"] == bid, "runtime_p85_min"].mean()
                        - df.loc[df["_band"] == right, "runtime_p85_min"].mean()
                    )
                    opts.append((right, diff))
                merge_into = min(opts, key=lambda t: t[1])[0]
                df.loc[df["_band"] == bid, "_band"] = merge_into

            # re‑number after merges
            remap = {old: new for new, old in enumerate(sorted(df["_band"].unique()))}
            df["_band"] = df["_band"].map(remap)

    # ── 4. assemble output table (pandas ≤1.5 compatible) ─────────────
    bands = (
        df.groupby("_band", sort=True, observed=True)
        .agg(
            start_time=("trip_start_time", "first"),
            end_time=("trip_start_time", "last"),
            n_tokens=("trip_start_time", "size"),
            p85_mean_min=("runtime_p85_min", "mean"),
        )
        .reset_index()  # '_band' -> column
        .rename(columns={"_band": "band_id"})
        .assign(band_id=lambda x: x["band_id"] + 1)
    )

    return bands


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:  # pragma: no cover
    """Run the end-to-end analysis for every route.

    * If ``SPLIT_BY_DIRECTION`` is **True** (default), rows are subdivided
      by the ``Direction`` column and written to
      ``<OUTPUT_ROOT>/<route>/<direction-slug>/``.
    * If **False**, all rows for the route are written directly to
      ``<OUTPUT_ROOT>/<route>/``.
    """
    # ------------------------------------------------------------------ #
    # 0.  Locate all CSVs and assign them to routes                      #
    # ------------------------------------------------------------------ #
    whitelist = {r.lstrip("0") for r in ROUTES_TO_INCLUDE} if ROUTES_TO_INCLUDE else None
    print(f"→ Crawling {INPUT_ROOT_DIR} for CSVs …")
    route_files = _discover_route_csvs(INPUT_ROOT_DIR, whitelist)

    if not route_files:
        raise FileNotFoundError("No CSV files found under the supplied folder.")

    # ------------------------------------------------------------------ #
    # 1.  Process each route                                             #
    # ------------------------------------------------------------------ #
    for route, paths in sorted(route_files.items(), key=lambda t: int(t[0])):
        print(f"— Processing route {route} ({len(paths)} files) …")

        # ── 1 a.  Load + global filters (shared before direction split) ──
        base_df = (
            load_trip_files(paths)
            .pipe(extract_trip_start_time)
            .pipe(filter_date_range)
            .pipe(filter_routes, {route})
            .pipe(filter_holidays, EXCLUDE_DATES)
            .pipe(filter_service_day, SERVICE_DAY_FILTER)
            .pipe(add_deviation_cols)
            .pipe(add_otp_flag)  # adds both on_time & within_window
            .pipe(normalize_directions_column, ALLOWED_DIRECTIONS)  # normalize Direction values
        )

        if base_df.empty:
            print("   ⚠  No rows left after filtering; skipping route.")
            continue

        # ── 1 b.  Quick sanity printout — median obs per start-time ──────
        print(
            "   observation median:",
            base_df.groupby(TIME_COL_NAME)["Actual Start Time"].count().median(),
        )
        log_low_sample_start_times(
            base_df,
            thresh_frac=LOW_SAMPLE_FRAC,
            exclude_dates=EXCLUDE_DATES,
        )

        # ------------------------------------------------------------------
        # 2.  Split by Direction (optional)                                #
        # ------------------------------------------------------------------
        if SPLIT_BY_DIRECTION:
            if "Direction" not in base_df.columns:
                print("   ⚠  'Direction' column missing; treating all rows as one direction.")
                base_df["Direction"] = "unknown"
            dir_groups = base_df.groupby("Direction", sort=False)
        else:
            # Flatten: treat everything as a single pseudo-direction “all”
            dir_groups = [("all", base_df)]

        for dir_val, dir_df in dir_groups:
            if dir_df.empty:
                continue

            # ---- 2 a.  Sanitise direction for folder names ---------------
            dir_slug = re.sub(r"[^0-9A-Za-z_-]+", "_", str(dir_val)).strip("_") or "all"

            # ---- 2 b.  Point writers/plotters at the correct folder ------
            global OUTPUT_DIR, PLOTS_DIR
            OUTPUT_DIR = (
                OUTPUT_ROOT_DIR / route / dir_slug
                if SPLIT_BY_DIRECTION
                else OUTPUT_ROOT_DIR / route
            )
            PLOTS_DIR = OUTPUT_DIR / "plots"

            print(
                f"   ↳ Direction {dir_val!r}: {len(dir_df):,} rows "
                f"➜ {OUTPUT_DIR.relative_to(OUTPUT_ROOT_DIR)}"
            )

            # ---- 2 c.  Persist outliers (optional) -----------------------
            if TRIM_OUTLIERS:
                export_trimmed_outliers(dir_df)

            # ---- 2 d.  Core exports --------------------------------------
            write_row_level(dir_df)
            summary = write_summary_table(dir_df)
            bands = suggest_time_bands(summary)
            bands.to_excel(
                OUTPUT_DIR / f"time_bands_{_day_tag()}.xlsx",
                index=False,
                engine="openpyxl",
            )
            print(f"      → Suggested {len(bands)} time bands saved.")

            # ---- 2 e.  Plots ---------------------------------------------
            plot_funcs = [
                plot_start_dev_shaded,
                plot_start_dev_plain,
                plot_finish_dev_shaded,
                plot_finish_dev_plain,
                plot_runtime_dev,
                plot_runtime_p85_vs_sched,
            ]
            for func in plot_funcs:
                _safe_plot(func, dir_df)

            print(f"      ✓ Direction {dir_val!r} done.")

        print(f"✓ Finished route {route}")

    print("✓✓ All routes processed.")


if __name__ == "__main__":
    main()
