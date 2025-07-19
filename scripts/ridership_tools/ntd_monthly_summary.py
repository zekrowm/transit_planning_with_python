"""Analyze and summarize NTD monthly ridership data.

This script loads NTD Excel files, cleans and validates the data,
classifies routes, calculates key metrics, and exports summaries
and plots by month, route, and service type.

Features:
    - Reads and processes all configured Excel files.
    - Derives indicators like passengers per hour, trip, and mile.
    - Aggregates results by route and service type.
    - Supports user-defined time windows (e.g., fiscal years).
    - Exports Excel and CSV files, plus optional plots.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_ROOT: Final[Path] = Path(r"Path\To\Your\NTD_Folder")  # input files
OUTPUT_DIR: Final[Path] = Path(r"Path\To\Your\Output\Folder")  # results

REQUIRED_NUMERIC_COLS: Final[list[str]] = [
    "MTH_BOARD",
    "MTH_REV_HOURS",
    "MTH_PASS_MILES",
    "ASCH_TRIPS",
    "DAYS",
    "REV_MILES",
]

# -----------------------------------------------------------------------------
#  Workbook catalogue
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PeriodSpec:
    """Workbook name and destination sheet."""

    filename: str
    sheet: str


PERIODS: Final[dict[str, PeriodSpec]] = {
    "Jul-2024": PeriodSpec("JULY 2024 NTD RIDERSHIP BY ROUTE.xlsx", "Temporary_Query_N"),
    "Aug-2024": PeriodSpec("AUGUST 2024  NTD RIDERSHIP REPORT BY ROUTE.xlsx", "Temporary_Query_N"),
    "Sep-2024": PeriodSpec("SEPTEMBER 2024 NTD RIDERSHIP BY ROUTE.xlsx", "Sep.2024 Finals"),
    "Oct-2024": PeriodSpec("NTD RIDERSHIP BY ROUTE _ OCTOBER _2024.xlsx", "Temporary_Query_N"),
    "Nov-2024": PeriodSpec("NTD RIDERSHIP BY ROUTE-NOVEMBER 2024.xlsx", "Temporary_Query_N"),
    "Dec-2024": PeriodSpec("NTD RIDERSHIP BY MONTH_DECEMBER 2024.xlsx", "Dec. 2024"),
    "Jan-2025": PeriodSpec("NTD RIDERSHIP BY MONTH-JANUARY 2025.xlsx", "Jan. 2025"),
    "Feb-2025": PeriodSpec("NTD RIDERSHIP BY MONTH-FEBRUARY 2025.xlsx", "Feb. 2025"),
    "Mar-2025": PeriodSpec("MARCH 2025 NTD MONTHLY RIDERSHIP.xlsx", "March 2025"),
    "Apr-2025": PeriodSpec("APRIL 2025 NTD MONTHLY RIDERSHIP.xlsx", "April 2025"),
    "May-2025": PeriodSpec("May 2025 NTD MONTHLY RIDERSHIP.xlsx", "May 2025"),
    # "Jun-2025": PeriodSpec("JUNE 2025 NTD RIDERSHIP BY MONTH.xlsx",          "Jun. 2025"),
}
ORDERED_PERIODS: Final[list[str]] = list(PERIODS)
SERVICE_PERIODS: Final[list[str]] = ["Weekday", "Saturday", "Sunday"]

# -----------------------------------------------------------------------------
#  Optional user-defined time windows
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class TimeWindow:
    """Named span of dates over which to aggregate results."""

    label: str  # e.g. "FY25" or "Summer2024"
    start: datetime  # inclusive
    end: datetime  # inclusive


# Analysts can edit or append to this list without touching code elsewhere.
TIME_WINDOWS: Final[list[TimeWindow]] = [
    TimeWindow("FY21", datetime(2020, 7, 1), datetime(2021, 6, 30)),
    TimeWindow("FY22", datetime(2021, 7, 1), datetime(2022, 6, 30)),
    TimeWindow("FY23", datetime(2022, 7, 1), datetime(2023, 6, 30)),
    TimeWindow("FY24", datetime(2023, 7, 1), datetime(2024, 6, 30)),
    TimeWindow("FY25", datetime(2024, 7, 1), datetime(2025, 6, 30)),
    # TimeWindow("Summer2024", datetime(2024, 6, 1), datetime(2024, 8, 31)),
]

# -----------------------------------------------------------------------------
#  Classification dictionaries
# -----------------------------------------------------------------------------

SERVICE_TYPE_DICT: Final[dict[str, list[str]]] = {
    "local": [
        "101",
        "202",
    ],
    "express": [
        "303",
        "404",
    ],
    "circulator": [
        "505",
        "606",
    ],
    "feeder": [
        "707",
        "808",
    ],
}

CORRIDOR_DICT: Final[dict[str, list[str]]] = {
    "route_one_corridor": ["101", "202"],
    "i_2_corridor": ["303", "404"],
    "route_three_corridor": ["505", "606"],
    "i_4_corridor": ["707", "808"],
}

# -----------------------------------------------------------------------------
#  Plot behaviour
# -----------------------------------------------------------------------------

PLOT_CONFIG: Final[dict[str, bool]] = {
    "plot_total_ridership": False,
    "plot_weekday_avg": True,
    "plot_saturday_avg": False,
    "plot_sunday_avg": False,
    "plot_revenue_hours": False,
    "plot_trips": False,
    "plot_revenue_miles": False,
    "plot_pph": False,
    "plot_ppt": False,
    "plot_ppm": False,
}

PLOT_STYLE: Final[dict[str, Any]] = {
    "figsize": (9, 5),
    "marker": "o",
    "linestyle": "-",
    "rotation": 45,
    "grid": True,
}

# =============================================================================
# FUNCTIONS
# =============================================================================

def _is_excluded(period: str, route: str) -> bool:
    """Return True if (*period*, *route*) is in EXCLUDE_DATA."""
    spec = EXCLUDE_DATA.get(period)
    if spec is None:                       # period not excluded
        return False
    if spec == "*":                        # whole system excluded
        return True
    return route in spec                   # route‑specific exclusion


def slice_for_window(df: pd.DataFrame, window: TimeWindow) -> pd.DataFrame:
    """Return rows in *df* whose month-start date falls inside *window*."""
    if "period_dt" not in df.columns:
        # Parse once – assumes "Jul-2024", "Aug-2024", …
        df = df.assign(period_dt=pd.to_datetime(df["period"], format="%b-%Y", errors="coerce"))
    mask = (df["period_dt"] >= window.start) & (df["period_dt"] <= window.end)
    return df.loc[mask].copy()


def safe_float(value: Any) -> float | None:
    """Return float if *value* looks numeric; else ``None``."""
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def safe_div(
    numerator: float | int,
    denominator: float | int,
    precision: int = 1,
) -> float | None:
    """Divide with protective zero handling."""
    try:
        return round(numerator / denominator, precision)  # type: ignore[arg-type]
    except (ZeroDivisionError, TypeError):
        return None


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Upper-case, trim, and replace spaces with underscores in columns."""
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.upper().str.replace(" ", "_", regex=False)
    return df


def read_excel_data() -> dict[str, pd.DataFrame]:
    """Load, clean, and filter every monthly workbook."""
    converters = {col: safe_float for col in REQUIRED_NUMERIC_COLS}
    data: dict[str, pd.DataFrame] = {}

    for period in ORDERED_PERIODS:
        spec = PERIODS[period]
        path: Path = DATA_ROOT / spec.filename
        print(f"→ Reading {period:<8} ({spec.filename}) … ", end="", flush=True)

        # --- load ---------------------------------------------------------- #
        df = pd.read_excel(path, sheet_name=spec.sheet, converters=converters)
        df = normalise_columns(df)

        # --- guard-rail ---------------------------------------------------- #
        missing = [c for c in REQUIRED_NUMERIC_COLS if c not in df.columns]
        if missing:
            raise KeyError(
                f"{period}: workbook '{spec.filename}' is missing "
                f"required column(s): {', '.join(missing)}"
            )

        # --- cleansing ----------------------------------------------------- #
        df.dropna(subset=["ROUTE_NAME", "MTH_BOARD"], inplace=True)
        df = df[df["MTH_BOARD"] != 0]
        df = df[df["SERVICE_PERIOD"].isin(SERVICE_PERIODS)].copy()
        df["ROUTE_NAME"] = (
            df["ROUTE_NAME"]
            .astype(str)
            .str.strip()
            .str.upper()
            .str.replace(" ", "", regex=False)
            .apply(lambda x: re.sub(r"\.0$", "", x))
        )

        data[period] = df
        print(f"{len(df):,} rows")

    return data


def classify_route(route_name: str) -> str:
    """Return the first matching service-type label for *route_name*."""
    for stype, lst in SERVICE_TYPE_DICT.items():
        if route_name in lst:
            return stype
    return "unknown"


def classify_corridor(route_name: str) -> list[str]:
    """Return all corridor labels containing *route_name*."""
    matches = [c for c, lst in CORRIDOR_DICT.items() if route_name in lst]
    return matches or ["other"]


def calculate_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Append trip- and mileage-based indicators to *df*."""
    df = df.copy()
    df["TOTAL_TRIPS"] = df["ASCH_TRIPS"] * df["DAYS"]
    df["BOARDS_PER_HOUR"] = safe_div_vec(df["MTH_BOARD"], df["MTH_REV_HOURS"])
    df["MTH_REV_MILES"] = df["REV_MILES"] * df["DAYS"]
    df["PASSENGERS_PER_TRIP"] = safe_div_vec(df["MTH_BOARD"], df["TOTAL_TRIPS"])
    df["PASSENGERS_PER_MILE"] = safe_div_vec(df["MTH_BOARD"], df["MTH_REV_MILES"], 3)

    df["TOTAL_TRIPS"] = df["TOTAL_TRIPS"].round(1)
    df["BOARDS_PER_HOUR"] = df["BOARDS_PER_HOUR"].round(1)
    df["PASSENGERS_PER_TRIP"] = df["PASSENGERS_PER_TRIP"].round(1)
    df["PASSENGERS_PER_MILE"] = df["PASSENGERS_PER_MILE"].round(3)
    return df


def safe_div_vec(a: pd.Series, b: pd.Series, precision: int = 1) -> pd.Series:
    """Vectorised safe_div for Series."""
    with pd.option_context("mode.use_inf_as_na", True):
        result = a.astype(float) / b.astype(float)
    return result.round(precision)


def aggregate_by_service_type(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise performance at the service-type level."""
    grouped = (
        df.groupby("service_type")
        .agg(
            {
                "MTH_BOARD": "sum",
                "MTH_REV_HOURS": "sum",
                "MTH_PASS_MILES": "sum",
                "MTH_REV_MILES": "sum",
                "TOTAL_TRIPS": "sum",
            }
        )
        .reset_index()
    )

    grouped["BOARDS_PER_HOUR"] = safe_div_vec(grouped["MTH_BOARD"], grouped["MTH_REV_HOURS"])
    grouped["PASSENGERS_PER_TRIP"] = safe_div_vec(grouped["MTH_BOARD"], grouped["TOTAL_TRIPS"])
    grouped["PASSENGERS_PER_MILE"] = safe_div_vec(grouped["MTH_BOARD"], grouped["MTH_REV_MILES"], 3)

    totals = grouped[
        ["MTH_BOARD", "MTH_REV_HOURS", "MTH_PASS_MILES", "MTH_REV_MILES", "TOTAL_TRIPS"]
    ].sum()

    total_row = {
        "service_type": "TOTAL",
        "MTH_BOARD": totals["MTH_BOARD"],
        "MTH_REV_HOURS": totals["MTH_REV_HOURS"],
        "MTH_PASS_MILES": totals["MTH_PASS_MILES"],
        "MTH_REV_MILES": totals["MTH_REV_MILES"],
        "TOTAL_TRIPS": totals["TOTAL_TRIPS"],
        "BOARDS_PER_HOUR": safe_div(totals["MTH_BOARD"], totals["MTH_REV_HOURS"]),
        "PASSENGERS_PER_TRIP": safe_div(totals["MTH_BOARD"], totals["TOTAL_TRIPS"]),
        "PASSENGERS_PER_MILE": safe_div(totals["MTH_BOARD"], totals["MTH_REV_MILES"], 3),
    }

    grouped = pd.concat([grouped, pd.DataFrame([total_row])], ignore_index=True)
    return grouped


def route_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create a per-route performance table."""
    agg_cols = {
        "MTH_BOARD": "sum",
        "DAYS": "sum",
        "MTH_REV_HOURS": "sum",
        "MTH_PASS_MILES": "sum",
        "MTH_REV_MILES": "sum",
        "TOTAL_TRIPS": "sum",
    }
    totals = df.groupby(["service_type", "ROUTE_NAME"], as_index=False).agg(agg_cols)

    totals["BOARDS_PER_HOUR"] = safe_div_vec(totals["MTH_BOARD"], totals["MTH_REV_HOURS"])
    totals["PASSENGERS_PER_TRIP"] = safe_div_vec(totals["MTH_BOARD"], totals["TOTAL_TRIPS"])
    totals["PASSENGERS_PER_MILE"] = safe_div_vec(totals["MTH_BOARD"], totals["MTH_REV_MILES"], 3)
    totals["DAILY_AVG"] = safe_div_vec(totals["MTH_BOARD"], totals["DAYS"])

    totals.sort_values(["ROUTE_NAME", "service_type"], inplace=True, ignore_index=True)
    return totals


def detect_negative_trends_12m(
    df_time: pd.DataFrame,
    window: int = ROLLING_WINDOW,
    pct_threshold: float = DECLINE_THRESH_PCT,
    min_coverage: float = MIN_COVERAGE,
    confirm_prev: bool = REQUIRE_TWO_MONTHS,
) -> pd.DataFrame:
    """Flag routes with two‑month declines vs. 12‑month baseline.

    Returns one row per route flagged.
    """
    flags: list[dict[str, Any]] = []

    metrics = ["weekday_avg", "pph", "ppt", "ppm"]  # adjust if desired
    periods = ORDERED_PERIODS  # already chronologically sorted

    for route, grp in df_time.groupby("route"):
        grp = grp.set_index("period").reindex(periods)  # align to master timeline

        for metric in metrics:
            vals = pd.to_numeric(grp[metric], errors="coerce")

            # Build exclusion mask
            excl_mask = [ _is_excluded(p, route) for p in periods ]
            vals = vals.mask(excl_mask)              # convert excluded to NaN

            if vals.notna().sum() < window + 1:
                continue  # not enough data overall

            latest = vals.iloc[-1]
            prev   = vals.iloc[-2]

            # Rolling baseline excludes latest month
            baseline_window = vals.iloc[-(window + 1):-1]  # previous 12
            valid_fraction = baseline_window.notna().mean()

            if latest is None or pd.isna(latest) or valid_fraction < min_coverage:
                continue  # insufficient baseline

            baseline_mean = baseline_window.mean(skipna=True)
            pct_change = 100 * (latest - baseline_mean) / baseline_mean

            # Optionally require previous month below baseline too
            prev_ok = (
                not confirm_prev
                or (prev is not None and not pd.isna(prev) and prev < baseline_mean)
            )

            if pct_change <= -pct_threshold and prev_ok:
                flags.append(
                    {
                        "route": route,
                        "metric": metric,
                        "latest_value": latest,
                        "baseline_mean": baseline_mean,
                        "pct_change": pct_change,
                        "window_months": int(baseline_window.notna().sum()),
                    }
                )

    return pd.DataFrame(flags)


def write_trend_log(df_flags: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> Path:
    """Write a plain‑text summary of flagged routes. Return the file path."""
    out_path = output_dir / "NegativeTrendFlags.txt"

    if df_flags.empty:
        print("No negative trends detected.")
        # create/overwrite an empty file to avoid downstream errors
        out_path.write_text("# No negative trends detected.\n", encoding="utf-8")
        return out_path

    lines: list[str] = []
    header = (
        "# Routes flagged for negative trends\n"
        f"# Generated {datetime.now():%Y-%m-%d %H:%M}\n\n"
    )
    lines.append(header)

    for route, grp in df_flags.groupby("route"):
        lines.append(f"Route {route}:")
        for _, row in grp.iterrows():
            lines.append(
                f"  • {row['metric']} ↓ {abs(row['pct_change']):.1f}% "
                f"(latest = {row['latest_value']:.1f}, "
                f"baseline = {row['baseline_mean']:.1f})"
            )
        lines.append("")  # blank line between routes

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Trend log written → {out_path}")
    return out_path


def build_monthly_timeseries(all_data: pd.DataFrame) -> pd.DataFrame:
    """Convert the year-to-date table into a plotting-ready time series.

    The result has one row per (period, route) plus a *SYSTEMWIDE* row,
    with these columns:

        period | route | total_ridership | weekday_avg | saturday_avg |
        sunday_avg | revenue_hours | trips | revenue_miles |
        pph | ppt | ppm
    """
    # 1. Aggregate to (period, route, service_period)
    group_cols = ["period", "ROUTE_NAME", "SERVICE_PERIOD"]
    agg = all_data.groupby(group_cols, as_index=False).agg(
        {
            "MTH_BOARD": "sum",
            "DAYS": "sum",
            "MTH_REV_HOURS": "sum",
            "TOTAL_TRIPS": "sum",
            "MTH_REV_MILES": "sum",
        }
    )

    def _sum_pair(dfsub: pd.DataFrame, daytype: str) -> tuple[float, float]:
        row = dfsub.loc[dfsub["SERVICE_PERIOD"] == daytype]
        return (row["MTH_BOARD"].iat[0], row["DAYS"].iat[0]) if not row.empty else (0, 0)

    rows: list[dict[str, Any]] = []
    for (period, route), grp in agg.groupby(["period", "ROUTE_NAME"]):
        tr = grp["MTH_BOARD"].sum()
        hrs = grp["MTH_REV_HOURS"].sum()
        trips = grp["TOTAL_TRIPS"].sum()
        miles = grp["MTH_REV_MILES"].sum()

        wd_b, wd_d = _sum_pair(grp, "Weekday")
        sa_b, sa_d = _sum_pair(grp, "Saturday")
        su_b, su_d = _sum_pair(grp, "Sunday")

        rows.append(
            {
                "period": period,
                "route": route,
                "total_ridership": tr,
                "weekday_avg": safe_div(wd_b, wd_d),
                "saturday_avg": safe_div(sa_b, sa_d),
                "sunday_avg": safe_div(su_b, su_d),
                "revenue_hours": hrs,
                "trips": trips,
                "revenue_miles": miles,
                "pph": safe_div(tr, hrs),
                "ppt": safe_div(tr, trips),
                "ppm": safe_div(tr, miles, 3),
            }
        )

    df_time = pd.DataFrame(rows)

    # 2. Add systemwide rows
    sys_rows: list[dict[str, Any]] = []
    for period in ORDERED_PERIODS:
        dfp = df_time[df_time["period"] == period]
        tr = dfp["total_ridership"].sum()
        hrs = dfp["revenue_hours"].sum()
        trips = dfp["trips"].sum()
        miles = dfp["revenue_miles"].sum()

        agg_wd = agg[(agg["period"] == period) & (agg["SERVICE_PERIOD"] == "Weekday")]
        agg_sa = agg[(agg["period"] == period) & (agg["SERVICE_PERIOD"] == "Saturday")]
        agg_su = agg[(agg["period"] == period) & (agg["SERVICE_PERIOD"] == "Sunday")]

        sys_rows.append(
            {
                "period": period,
                "route": "SYSTEMWIDE",
                "total_ridership": tr,
                "weekday_avg": safe_div(agg_wd["MTH_BOARD"].sum(), agg_wd["DAYS"].sum()),
                "saturday_avg": safe_div(agg_sa["MTH_BOARD"].sum(), agg_sa["DAYS"].sum()),
                "sunday_avg": safe_div(agg_su["MTH_BOARD"].sum(), agg_su["DAYS"].sum()),
                "revenue_hours": hrs,
                "trips": trips,
                "revenue_miles": miles,
                "pph": safe_div(tr, hrs),
                "ppt": safe_div(tr, trips),
                "ppm": safe_div(tr, miles, 3),
            }
        )

    return pd.concat([df_time, pd.DataFrame(sys_rows)], ignore_index=True)


def plot_metric_over_time(df_time: pd.DataFrame, metric: str) -> None:
    """Create a line plot of *metric* by month for every route."""
    plot_dir = OUTPUT_DIR / "plots" / metric
    plot_dir.mkdir(parents=True, exist_ok=True)

    df_m = df_time[["period", "route", metric]].copy()
    df_m[metric] = pd.to_numeric(df_m[metric], errors="coerce")

    for route in sorted(df_m["route"].unique()):
        df_r = df_m[df_m["route"] == route]
        y_vals = [
            df_r.loc[df_r["period"] == p, metric].squeeze() if p in df_r["period"].values else None
            for p in ORDERED_PERIODS
        ]
        if all(v is None or pd.isna(v) for v in y_vals):
            continue  # nothing to plot

        plt.figure(figsize=PLOT_STYLE["figsize"])
        plt.plot(
            ORDERED_PERIODS,
            y_vals,
            marker=PLOT_STYLE["marker"],
            linestyle=PLOT_STYLE["linestyle"],
        )
        plt.title(f"{metric.replace('_', ' ').title()} – Route {route}")
        plt.xlabel("Month")
        plt.ylabel(metric.replace("_", " ").title())
        plt.xticks(rotation=PLOT_STYLE["rotation"])
        plt.grid(PLOT_STYLE["grid"])

        numeric = [v for v in y_vals if v is not None and not pd.isna(v)]
        if numeric:
            plt.ylim(0, max(numeric) * 1.1)

        plt.tight_layout()
        plt.savefig(plot_dir / f"{metric}_route_{route}.png", dpi=150)
        plt.close()


def generate_all_plots(df_time: pd.DataFrame) -> None:
    """Iterate over :pydata:`PLOT_CONFIG` and generate enabled plots."""
    metric_map = {
        "plot_total_ridership": "total_ridership",
        "plot_weekday_avg": "weekday_avg",
        "plot_saturday_avg": "saturday_avg",
        "plot_sunday_avg": "sunday_avg",
        "plot_revenue_hours": "revenue_hours",
        "plot_trips": "trips",
        "plot_revenue_miles": "revenue_miles",
        "plot_pph": "pph",
        "plot_ppt": "ppt",
        "plot_ppm": "ppm",
    }
    for flag, col in metric_map.items():
        if PLOT_CONFIG.get(flag, False):
            print(f"Plotting {col} …")
            plot_metric_over_time(df_time, col)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Run the end-to-end NTD performance workflow.

    Export policy
    -------------
    * “Complete” tables (all periods combined, or any full fiscal-year slice)
      → **CSV only** – lightweight and ready for BI/plotting ingestion.
    * All other deliverables (route-level, service-type, monthly workbooks, etc.)
      → **XLSX only** – analyst-friendly, no redundant CSV versions.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # === STEP 1: READ EXCEL FILES ============================================
    print("=== STEP 1: READ EXCEL FILES ===")
    data_dict = read_excel_data()

    # === STEP 2: CLASSIFY & DERIVE ===========================================
    print("\n=== STEP 2: CLASSIFY & DERIVE ===")
    for period, df in data_dict.items():
        df["service_type"] = df["ROUTE_NAME"].apply(classify_route)
        df["corridors"] = df["ROUTE_NAME"].apply(classify_corridor)
        df = calculate_derived_columns(df)
        df["period"] = period
        data_dict[period] = df

    all_data = pd.concat(data_dict.values(), ignore_index=True)

    unknown = (
        all_data.loc[all_data["service_type"] == "unknown", "ROUTE_NAME"]
        .drop_duplicates()
        .sort_values()
    )
    if not unknown.empty:
        print("Unclassified routes:", ", ".join(unknown))

    # === STEP 3: EXPORT COMPLETE DATASETS ====================================
    print("\n=== STEP 3: EXPORT COMPLETE DATASETS ===")

    # 3.1  Master CSV (all periods) – single file for pipelines
    all_data.to_csv(
        OUTPUT_DIR / "DetailedAllPeriods_for_plotting.csv",
        index=False,
    )
    print("Combined CSV exported.")

    # 3.2  Convenience workbook – one sheet per month (no giant sheet)
    with pd.ExcelWriter(OUTPUT_DIR / "MonthlySheets.xlsx") as xw:
        for period in ORDERED_PERIODS:
            data_dict[period].to_excel(xw, sheet_name=period, index=False)
    print("Monthly workbook exported.")

    # === STEP 4: ROUTE-LEVEL SUMMARIES (FULL FY-25) ==========================
    print("\n=== STEP 4: ROUTE-LEVEL SUMMARIES ===")
    subsets = {
        "Combined": all_data,
        "Weekday": all_data[all_data["SERVICE_PERIOD"] == "Weekday"],
        "Saturday": all_data[all_data["SERVICE_PERIOD"] == "Saturday"],
        "Sunday": all_data[all_data["SERVICE_PERIOD"] == "Sunday"],
    }
    for label, subset in subsets.items():
        out = route_level_summary(subset)
        with pd.ExcelWriter(OUTPUT_DIR / f"RouteLevelSummary_{label}.xlsx") as xw:
            out.to_excel(xw, sheet_name=f"{label}_Route_Level", index=False)
        print(f"{label} summary exported.")

    # === STEP 5: TIME-SERIES PLOTS ===========================================
    print("\n=== STEP 5: TIME-SERIES PLOTS ===")
    ts = build_monthly_timeseries(all_data)
    generate_all_plots(ts)

    # === STEP 6: USER-DEFINED TIME WINDOWS ===================================
    print("\n=== STEP 6: TIME-WINDOW OUTPUTS ===")
    for tw in TIME_WINDOWS:
        w_dir = OUTPUT_DIR / tw.label
        w_dir.mkdir(parents=True, exist_ok=True)

        subset = slice_for_window(all_data, tw)
        if subset.empty:
            print(f"⚠︎ {tw.label}: no rows inside {tw.start:%Y-%m-%d} → {tw.end:%Y-%m-%d}")
            continue

        # 6.1  Raw slice (complete FY etc.) → CSV only
        subset.to_csv(
            w_dir / f"detailed_{tw.label}_for_plotting.csv",
            index=False,
        )

        # 6.2  Route-level summaries (Combined + day-type splits) → XLSX only
        subsets_tw = {
            "Combined": subset,
            "Weekday": subset[subset["SERVICE_PERIOD"] == "Weekday"],
            "Saturday": subset[subset["SERVICE_PERIOD"] == "Saturday"],
            "Sunday": subset[subset["SERVICE_PERIOD"] == "Sunday"],
        }
        for lbl, df_sub in subsets_tw.items():
            rl = route_level_summary(df_sub)
            rl.to_excel(
                w_dir / f"RouteLevelSummary_{lbl}.xlsx",
                index=False,
            )

        # 6.3  Service-type aggregation for the window → XLSX only
        st = aggregate_by_service_type(subset)
        st.to_excel(
            w_dir / f"AggByServiceType_{tw.label}.xlsx",
            index=False,
        )

        print(f"{tw.label}: {len(subset):,} rows → {w_dir.relative_to(OUTPUT_DIR)}")

    print("\nAll processing complete.")


if __name__ == "__main__":
    main()
