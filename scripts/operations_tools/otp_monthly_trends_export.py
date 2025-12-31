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

# Output filenames
OUTPUT_TABLE_FILENAME: str = "otp_processed.csv"

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


def process(df: pd.DataFrame, current_yy_mm: str) -> pd.DataFrame:
    """Compute totals and percentages and produce a tidy DataFrame."""
    ref_year, ref_month = parse_current_yy_mm(current_yy_mm)
    df = df.copy()
    df["route_raw"] = df["route"].astype(str).str.strip()
    df["route_clean"] = df["route_raw"].map(clean_route)
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

            # Blue OTP average line
            (line_on,) = plt.plot(
                x, avg_on.values, marker="o", label="Weekday On-time % (avg Mon–Fri)"
            )

            # Light-blue band (OTP min–max range across Mon–Fri)
            base_color = line_on.get_color()
            plt.fill_between(
                x,
                min_map.values.astype(float),
                max_map.values.astype(float),
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
            fname = f"{route_clean}_{direction.replace('/', '-')}_Weekdays_otp_trend.png".replace(
                " ", ""
            )
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
    return p


# ==============================
# MAIN
# ==============================


def main(argv: List[str] | None = None) -> None:
    """Entrypoint.

    Args:
        argv: Optional explicit argv list (e.g., [] for notebooks). If None, uses sys.argv.
    """
    parser = build_arg_parser()
    # Accept unknown args to be notebook/IPython friendly (swallows "-f <kernel.json>").
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        logging.warning("Ignoring unknown CLI args (likely from IPython): %s", unknown)
    cfg = Config(
        input_csv=Path(args.input).expanduser(),
        out_table_dir=Path(args.out_table).expanduser(),
        out_plots_dir=Path(args.out_plots).expanduser(),
        current_yy_mm=args.current,
        otp_standard=args.otp_standard,
    )
    logging.info("Reading: %s", cfg.input_csv)
    raw = read_csv_safely(cfg.input_csv)
    logging.info("Rows read: %d", len(raw))
    logging.info("Normalizing columns...")
    norm = standardize_columns(raw)
    logging.info("Processing with CURRENT_YY_MM='%s'...", cfg.current_yy_mm)
    proc = process(norm, cfg.current_yy_mm)
    logging.info("Exporting table to: %s", cfg.out_table_dir)
    out_table_path = export_table(proc, cfg.out_table_dir, OUTPUT_TABLE_FILENAME)
    logging.info("Wrote table: %s", out_table_path)
    logging.info("Generating plots in: %s", cfg.out_plots_dir)
    plot_series_for_groups(proc, cfg.out_plots_dir, cfg.otp_standard)
    logging.info("Plot export complete.")
    n_groups = proc.groupby(["route_clean", "direction"], dropna=False).ngroups
    logging.info(
        "Processed %d rows across %d route/direction groups.",
        len(proc),
        n_groups,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
