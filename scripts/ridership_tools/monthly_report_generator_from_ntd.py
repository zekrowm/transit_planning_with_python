"""Generates a Monthly summary performance report with plots and Excel exports based on historical NTD ridership reports.

Transforms a fiscal year of monthly Excel workbooks into a National
Transit Database (NTD) performance package:

* Cleans and merges the monthly data.
* Classifies each route by service type and corridor.
* Derives passengers-per-hour, -trip, and -mile metrics.
* Writes multi-sheet Excel summaries (detailed, service-type, route-level).
* Optionally produces time-series PNG plots for selected metrics.

Typical use cases
-----------------
* Monthly and year-to-date performance monitoring.
* Rapid generation of NTD-style summaries for board reports.
* Historical trend analysis and data visualization.

Attributes:
----------
CONFIG : dict[str, Any]
    File paths, ordered periods, classification dictionaries, and constants
    used throughout the pipeline.
PLOT_CONFIG : dict[str, bool]
    Feature flags controlling which plots are created.
PLOT_STYLE : dict[str, Any]
    Matplotlib keyword arguments that standardize the look of every plot.
"""

from __future__ import annotations

import os
import re
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "periods": {
        "Jul-2024": {
            "file_path": r"\\Your\File\Path\JULY 2024 NTD RIDERSHIP BY ROUTE.XLSX",
            "sheet_name": "Temporary_Query_N",
        },
        "Aug-2024": {
            "file_path": r"\\Your\File\Path\AUGUST 2024  NTD RIDERSHIP REPORT BY ROUTE.XLSX",
            "sheet_name": "Temporary_Query_N",
        },
        "Sep-2024": {
            "file_path": r"\\Your\File\Path\SEPTEMBER 2024 NTD RIDERSHIP BY ROUTE.XLSX",
            "sheet_name": "Sep.2024 Finals",
        },
        "Oct-2024": {
            "file_path": r"\\Your\File\Path\NTD RIDERSHIP BY ROUTE _ OCTOBER _2024.XLSX",
            "sheet_name": "Temporary_Query_N",
        },
        "Nov-2024": {
            "file_path": r"\\Your\File\Path\NTD RIDERSHIP BY ROUTE-NOVEMBER 2024.xlsx",
            "sheet_name": "Temporary_Query_N",
        },
        "Dec-2024": {
            "file_path": r"\\Your\File\Path\NTD RIDERSHIP BY MONTH_DECEMBER 2024.XLSX",
            "sheet_name": "Dec. 2024",
        },
        "Jan-2025": {
            "file_path": r"\\Your\File\Path\NTD_files_FY25\NTD RIDERSHIP BY MONTH-JANUARY 2025.xlsx",
            "sheet_name": "Jan. 2025",
        },
        "Feb-2025": {
            "file_path": r"\\Your\File\Path\NTD RIDERSHIP BY MONTH-FEBRUARY 2025.xlsx",
            "sheet_name": "Feb. 2025",
        },
        "Mar-2025": {
            "file_path": r"\\Your\File\Path\MARCH 2025 NTD RIDERSHIP BY MONTH.xlsx",
            "sheet_name": "Mar. 2025",
        },
        "Apr-2025": {
            "file_path": r"\\Your\File\Path\APRIL 2025 NTD RIDERSHIP BY MONTH.xlsx",
            "sheet_name": "Apr. 2025",
        },
        "May-2025": {
            "file_path": r"\\Your\File\Path\MAY 2025 NTD RIDERSHIP BY MONTH.xlsx",
            "sheet_name": "May. 2025",
        },
        "Jun-2025": {
            "file_path": r"\\Your\File\Path\JUNE 2025 NTD RIDERSHIP BY MONTH.xlsx",
            "sheet_name": "Jun. 2025",
        },
    },
    "ordered_periods": [
        "Jul-2024",
        "Aug-2024",
        "Sep-2024",
        "Oct-2024",
        "Nov-2024",
        "Dec-2024",
        "Jan-2025",
        "Feb-2025",
        "Mar-2025",
        "Apr-2025",
        "May-2025",
        "Jun-2025",
    ],
    "SERVICE_TYPE_DICT": {
        "local": ["101", "201", "301"],
        "express": ["102", "202", "302"],
    },
    "CORRIDOR_DICT": {
        "corridor_one": ["101", "102"],
        "corridor_two": ["201", "202"],
        "corridor_three": ["301", "302"],
    },
    "converters": {
        "MTH_BOARD": lambda x: float(str(x).replace(",", "")) if x else None,
        "MTH_REV_HOURS": lambda x: float(str(x).replace(",", "")) if x else None,
        "MTH_PASS_MILES": lambda x: float(str(x).replace(",", "")) if x else None,
        "ASCH_TRIPS": lambda x: float(str(x).replace(",", "")) if x else None,
        "ACTUAL_TRIPS": lambda x: float(str(x).replace(",", "")) if x else None,
        "DAYS": lambda x: float(str(x).replace(",", "")) if x else None,
        "REV_MILES": lambda x: float(str(x).replace(",", "")) if x else None,
    },
    "SERVICE_PERIODS": ["Weekday", "Saturday", "Sunday"],
    "output_dir": r"\\Path\to\Your\Output_Folder",
}

# -----------------------------------------------------------------------------
# PLOT CONFIGURATION BOOLEANS
# -----------------------------------------------------------------------------
# Set any of these to False if you do NOT want that particular plot generated.
PLOT_CONFIG = {
    "plot_total_ridership": True,
    "plot_weekday_avg": True,
    "plot_saturday_avg": False,
    "plot_sunday_avg": False,
    "plot_revenue_hours": False,
    "plot_trips": False,
    "plot_revenue_miles": False,
    "plot_pph": True,  # passengers per hour
    "plot_ppt": True,  # passengers per trip
    "plot_ppm": True,  # passengers per mile
}

# Matplotlib settings for plot style
PLOT_STYLE = {
    "figsize": (9, 5),
    "marker": "o",
    "linestyle": "-",
    "rotation": 45,
    "grid": True,
}

# =============================================================================
# FUNCTIONS
# =============================================================================


def safe_float(value: Any) -> float | None:
    """Return a float if *value* looks numeric, otherwise ``None``.

    The function is resilient to:

    * Empty strings or whitespace.
    * Excel “NaN” placeholders.
    * Thousands separators (commas).
    * Zero-like inputs (``0``, ``"0"``, ``"0.00"``).

    Args:
        value: Arbitrary cell content from an Excel sheet.

    Returns:
        A ``float`` or ``None`` when conversion is impossible.
    """
    if pd.isna(value):
        return None
    s = str(value).strip()
    if s == "":
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def safe_div(numerator: float | int, denominator: float | int, precision: int = 1) -> float | None:
    """Divide *numerator* by *denominator* with graceful zero handling.

    Args:
        numerator: Dividend.
        denominator: Divisor. If falsy or zero, the function returns ``None``.
        precision: Number of decimal places in the rounded result.

    Returns:
        The rounded quotient, or ``None`` when division is undefined.
    """
    try:
        return round(numerator / denominator, precision)  # type: ignore[arg-type]
    except (ZeroDivisionError, TypeError):
        return None


def read_excel_data(config: dict) -> dict[str, pd.DataFrame]:
    """Load, clean, and filter the monthly Excel worksheets."""
    numeric_cols = [
        "MTH_BOARD",
        "MTH_REV_HOURS",
        "MTH_PASS_MILES",
        "ACTUAL_TRIPS",
        "ASCH_TRIPS",
        "DAYS",
        "REV_MILES",
    ]
    converters = {col: safe_float for col in numeric_cols}

    sp_filter = config["SERVICE_PERIODS"]
    data_dict: dict[str, pd.DataFrame] = {}

    for period in config["ordered_periods"]:
        info = config["periods"][period]
        df = pd.read_excel(
            info["file_path"],
            sheet_name=info["sheet_name"],
            converters=converters,
        )

        # --- identical logic below this line ---
        df.dropna(subset=["ROUTE_NAME", "MTH_BOARD"], inplace=True)
        df = df[df["MTH_BOARD"] != 0]
        df = df[df["SERVICE_PERIOD"].isin(sp_filter)].copy()
        df["ROUTE_NAME"] = (
            df["ROUTE_NAME"]
            .astype(str)
            .str.strip()
            .str.upper()
            .str.replace(" ", "", regex=False)
            .apply(lambda x: re.sub(r"\.0$", "", x))
        )
        data_dict[period] = df

    return data_dict


def classify_route(route_name: str, cfg: dict) -> str:
    """Map a route name to its first matching service type.

    Args:
        route_name: Canonical route identifier (e.g. ``"101"``).
        cfg: The :pydata:`CONFIG` dictionary containing ``SERVICE_TYPE_DICT``.

    Returns:
        The service-type key (e.g. ``"local"``) or:

        * ``"unknown"`` – route found in no list.
        * ``"SYSTEMWIDE"`` – classification dictionary is empty.
    """
    st_dict = cfg["SERVICE_TYPE_DICT"]
    if not st_dict:
        return "SYSTEMWIDE"
    for service_type, route_list in st_dict.items():
        if route_name in route_list:
            return service_type
    return "unknown"


def classify_corridor(route_name: str, cfg: dict) -> list:
    """Return all corridors that include *route_name*.

    Args:
        route_name: Canonical route identifier.
        cfg: The :pydata:`CONFIG` dictionary with ``CORRIDOR_DICT``.

    Returns:
        A non-empty list of corridor labels.  ``["other"]`` when no corridor
        matches.
    """
    corridor_dict = cfg["CORRIDOR_DICT"]
    corridors = []
    for c_name, r_list in corridor_dict.items():
        if route_name in r_list:
            corridors.append(c_name)
    return corridors if corridors else ["other"]


def calculate_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Append trip- and mileage-based indicators to *df*.

    Added columns
    -------------
    TOTAL_TRIPS
        ``ASCH_TRIPS * DAYS``.
    BOARDS_PER_HOUR
        ``MTH_BOARD / MTH_REV_HOURS``.
    PASSENGERS_PER_TRIP
        ``MTH_BOARD / TOTAL_TRIPS``.
    MTH_REV_MILES
        ``REV_MILES * DAYS``.
    PASSENGERS_PER_MILE
        ``MTH_BOARD / MTH_REV_MILES``.

    Args:
        df: Monthly ridership table returned by :func:`read_excel_data`.

    Returns:
        A copy of *df* with the five derived metrics, rounded appropriately.
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # 1. total scheduled trips in the month
    # ------------------------------------------------------------------
    df["TOTAL_TRIPS"] = df["ASCH_TRIPS"] * df["DAYS"]

    # ------------------------------------------------------------------
    # 2. passengers per revenue hour
    # ------------------------------------------------------------------
    df["BOARDS_PER_HOUR"] = df.apply(
        lambda r: r["MTH_BOARD"] / r["MTH_REV_HOURS"] if r["MTH_REV_HOURS"] else None,
        axis=1,
    )

    # ------------------------------------------------------------------
    # 3. passengers per scheduled trip
    # ------------------------------------------------------------------
    df["PASSENGERS_PER_TRIP"] = df.apply(
        lambda r: r["MTH_BOARD"] / r["TOTAL_TRIPS"] if r["TOTAL_TRIPS"] else None,
        axis=1,
    )

    # ------------------------------------------------------------------
    # 4. revenue miles & passengers per mile
    # ------------------------------------------------------------------
    df["MTH_REV_MILES"] = df["REV_MILES"] * df["DAYS"]
    df["PASSENGERS_PER_MILE"] = df.apply(
        lambda r: r["MTH_BOARD"] / r["MTH_REV_MILES"] if r["MTH_REV_MILES"] else None,
        axis=1,
    )

    # ------------------------------------------------------------------
    # 5. tidy rounding
    # ------------------------------------------------------------------
    df["BOARDS_PER_HOUR"] = df["BOARDS_PER_HOUR"].round(1)
    df["PASSENGERS_PER_TRIP"] = df["PASSENGERS_PER_TRIP"].round(1)
    df["PASSENGERS_PER_MILE"] = df["PASSENGERS_PER_MILE"].round(3)
    df["TOTAL_TRIPS"] = df["TOTAL_TRIPS"].round(1)

    return df


def aggregate_by_service_type(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize performance at the service-type level.

    Sums raw totals and recomputes ratio metrics, then appends a systemwide
    **TOTAL** row.

    Args:
        df: Detailed ridership data for a single period or year-to-date.

    Returns:
        A tidy DataFrame with one row per service type plus the final **TOTAL**
        row, ready for Excel export.
    """
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

    # Derived columns
    grouped["BOARDS_PER_HOUR"] = (grouped["MTH_BOARD"] / grouped["MTH_REV_HOURS"]).round(1)
    grouped["PASSENGERS_PER_TRIP"] = (grouped["MTH_BOARD"] / grouped["TOTAL_TRIPS"]).round(1)
    grouped["PASSENGERS_PER_MILE"] = (grouped["MTH_BOARD"] / grouped["MTH_REV_MILES"]).round(3)

    # Build TOTAL row
    sums = grouped[
        ["MTH_BOARD", "MTH_REV_HOURS", "MTH_PASS_MILES", "MTH_REV_MILES", "TOTAL_TRIPS"]
    ].sum()

    total_row = {
        "service_type": "TOTAL",
        "MTH_BOARD": sums["MTH_BOARD"],
        "MTH_REV_HOURS": sums["MTH_REV_HOURS"],
        "MTH_PASS_MILES": sums["MTH_PASS_MILES"],
        "MTH_REV_MILES": sums["MTH_REV_MILES"],
        "TOTAL_TRIPS": sums["TOTAL_TRIPS"],
        "BOARDS_PER_HOUR": (
            round(sums["MTH_BOARD"] / sums["MTH_REV_HOURS"], 1) if sums["MTH_REV_HOURS"] else None
        ),
        "PASSENGERS_PER_TRIP": (
            round(sums["MTH_BOARD"] / sums["TOTAL_TRIPS"], 1) if sums["TOTAL_TRIPS"] else None
        ),
        "PASSENGERS_PER_MILE": (
            round(sums["MTH_BOARD"] / sums["MTH_REV_MILES"], 3) if sums["MTH_REV_MILES"] else None
        ),
    }

    # <<<< REPLACED .append() WITH pd.concat() >>>>
    grouped = pd.concat([grouped, pd.DataFrame([total_row])], ignore_index=True)

    return grouped


def route_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Produce a route-by-route performance table.

    The function aggregates across all months in *df* while preserving the
    supplied service-period subset (Weekday-only, Saturday-only, etc.).

    Args:
        df: Input data containing the derived columns created by
            :func:`calculate_derived_columns`.

    Returns:
        A DataFrame sorted by ``ROUTE_NAME`` and ``service_type`` with
        per-route totals plus ``DAILY_AVG`` (boardings per calendar day).
    """
    # --------------------------------------------------
    # 1. Aggregate totals for each (service_type, ROUTE_NAME)
    # --------------------------------------------------
    agg_cols = {
        "MTH_BOARD": "sum",
        "DAYS": "sum",
        "MTH_REV_HOURS": "sum",
        "MTH_PASS_MILES": "sum",
        "MTH_REV_MILES": "sum",
        "TOTAL_TRIPS": "sum",
    }
    route_totals = df.groupby(["service_type", "ROUTE_NAME"], as_index=False).agg(agg_cols)

    # --------------------------------------------------
    # 2. Derived columns
    # --------------------------------------------------
    route_totals["BOARDS_PER_HOUR"] = (
        route_totals["MTH_BOARD"] / route_totals["MTH_REV_HOURS"]
    ).round(1)

    route_totals["PASSENGERS_PER_TRIP"] = (
        route_totals["MTH_BOARD"] / route_totals["TOTAL_TRIPS"]
    ).round(1)

    route_totals["PASSENGERS_PER_MILE"] = (
        route_totals["MTH_BOARD"] / route_totals["MTH_REV_MILES"]
    ).round(3)

    route_totals["DAILY_AVG"] = route_totals.apply(
        lambda r: round(r["MTH_BOARD"] / r["DAYS"], 1) if r["DAYS"] else None, axis=1
    )

    # --------------------------------------------------
    # 3. Sort and return
    # --------------------------------------------------
    route_totals.sort_values(["ROUTE_NAME", "service_type"], inplace=True, ignore_index=True)

    return route_totals


def build_monthly_timeseries(all_data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Convert the combined year-to-date table into a plotting-ready time series.

    Each record corresponds to (*period*, *route*) and includes systemwide
    rows.  The routine aggregates raw totals, calculates daily averages for
    each day type, and re-computes PPH/PPT/PPM ratios.

    Args:
        all_data: Concatenated output of every month after classification and
            derived-metric processing.
        config: Global :pydata:`CONFIG` used for the ordered period sequence.

    Returns:
        A wide DataFrame with these columns:

        ``period`` | ``route`` | ``total_ridership`` | ``weekday_avg`` |
        ``saturday_avg`` | ``sunday_avg`` | ``revenue_hours`` | ``trips`` |
        ``revenue_miles`` | ``pph`` | ``ppt`` | ``ppm``.
    """
    # For convenience, let’s keep the original columns:
    # MTH_BOARD, DAYS, MTH_REV_HOURS, TOTAL_TRIPS, MTH_REV_MILES
    # along with SERVICE_PERIOD so we can separate weekday/sat/sun.

    # Group by (period, route_name, service_period) to sum the relevant columns
    group_cols = ["period", "ROUTE_NAME", "SERVICE_PERIOD"]
    agg_df = all_data.groupby(group_cols, as_index=False).agg(
        {
            "MTH_BOARD": "sum",
            "DAYS": "sum",
            "MTH_REV_HOURS": "sum",
            "TOTAL_TRIPS": "sum",
            "MTH_REV_MILES": "sum",
        }
    )

    # We want each row to correspond to (period, route).
    # We'll pivot the daily boardings for weekday/sat/sun so we can form averages.

    def get_daytype_sum(dfsub: pd.DataFrame, daytype: str) -> tuple[float, float]:
        row = dfsub.loc[dfsub["SERVICE_PERIOD"] == daytype]
        if row.empty:
            return (0, 0)  # (boardings, days)
        return (row["MTH_BOARD"].values[0], row["DAYS"].values[0])

    rows = []
    for (period, route), df_grp in agg_df.groupby(["period", "ROUTE_NAME"]):
        # Sum across all day types for total ridership, hours, trips, miles
        total_ridership = df_grp["MTH_BOARD"].sum()
        revenue_hours = df_grp["MTH_REV_HOURS"].sum()
        total_trips = df_grp["TOTAL_TRIPS"].sum()
        revenue_miles = df_grp["MTH_REV_MILES"].sum()

        # For weekday avg, sat avg, sun avg, we look specifically at each day type
        wd_board, wd_days = get_daytype_sum(df_grp, "Weekday")
        sat_board, sat_days = get_daytype_sum(df_grp, "Saturday")
        sun_board, sun_days = get_daytype_sum(df_grp, "Sunday")

        weekday_avg = safe_div(wd_board, wd_days)
        saturday_avg = safe_div(sat_board, sat_days)
        sunday_avg = safe_div(sun_board, sun_days)

        pph = round(total_ridership / revenue_hours, 1) if revenue_hours else None
        ppt = round(total_ridership / total_trips, 1) if total_trips else None
        ppm = round(total_ridership / revenue_miles, 3) if revenue_miles else None

        rows.append(
            {
                "period": period,
                "route": route,
                "total_ridership": total_ridership,
                "weekday_avg": weekday_avg,
                "saturday_avg": saturday_avg,
                "sunday_avg": sunday_avg,
                "revenue_hours": revenue_hours,
                "trips": total_trips,
                "revenue_miles": revenue_miles,
                "pph": pph,
                "ppt": ppt,
                "ppm": ppm,
            }
        )

    df_time = pd.DataFrame(rows)

    # Also build a systemwide row by summing across all routes
    # for each period.
    syswide_rows = []
    for period in config["ordered_periods"]:
        df_period = df_time[df_time["period"] == period]
        # Sum columns
        total_ridership = df_period["total_ridership"].sum()
        revenue_hours = df_period["revenue_hours"].sum()
        total_trips = df_period["trips"].sum()
        revenue_miles = df_period["revenue_miles"].sum()

        # Weighted daily averages for weekday/sat/sun (or we can do sum of board/days again)
        # If you prefer a simpler approach, we can sum the board/days across all routes.
        # But for now, let's just do a simple sum->divide approach, same logic as above:
        df_p = agg_df[(agg_df["period"] == period) & (agg_df["SERVICE_PERIOD"] == "Weekday")]
        sys_wd_board = df_p["MTH_BOARD"].sum()
        sys_wd_days = df_p["DAYS"].sum()

        df_s = agg_df[(agg_df["period"] == period) & (agg_df["SERVICE_PERIOD"] == "Saturday")]
        sys_sat_board = df_s["MTH_BOARD"].sum()
        sys_sat_days = df_s["DAYS"].sum()

        df_su = agg_df[(agg_df["period"] == period) & (agg_df["SERVICE_PERIOD"] == "Sunday")]
        sys_sun_board = df_su["MTH_BOARD"].sum()
        sys_sun_days = df_su["DAYS"].sum()

        weekday_avg = safe_div(sys_wd_board, sys_wd_days, 1)
        saturday_avg = safe_div(sys_sat_board, sys_sat_days, 1)
        sunday_avg = safe_div(sys_sun_board, sys_sun_days, 1)

        pph = round(total_ridership / revenue_hours, 1) if revenue_hours else None
        ppt = round(total_ridership / total_trips, 1) if total_trips else None
        ppm = round(total_ridership / revenue_miles, 3) if revenue_miles else None

        syswide_rows.append(
            {
                "period": period,
                "route": "SYSTEMWIDE",
                "total_ridership": total_ridership,
                "weekday_avg": weekday_avg,
                "saturday_avg": saturday_avg,
                "sunday_avg": sunday_avg,
                "revenue_hours": revenue_hours,
                "trips": total_trips,
                "revenue_miles": revenue_miles,
                "pph": pph,
                "ppt": ppt,
                "ppm": ppm,
            }
        )

    df_sys = pd.DataFrame(syswide_rows)
    df_time = pd.concat([df_time, df_sys], ignore_index=True)

    return df_time


def plot_metric_over_time(df_time: pd.DataFrame, metric: str, config: dict) -> None:
    """Create a line plot of *metric* by month for every route.

    Files are saved as PNG under ``<output_dir>/plots/<metric>/``.

    Args:
        df_time: Time-series DataFrame returned by
            :func:`build_monthly_timeseries`.
        metric: Column name in *df_time* to plot.
        config: Global :pydata:`CONFIG` providing ``output_dir`` and
            ``ordered_periods``.
    """
    output_dir = config["output_dir"]
    plot_dir = os.path.join(output_dir, "plots", metric)
    os.makedirs(plot_dir, exist_ok=True)

    # Sort the df_time by period in the correct order
    # We rely on config['ordered_periods'] for the x-axis sequence.
    ordered_periods = config["ordered_periods"]

    # Keep just relevant columns
    # df_time has columns 'period', 'route', and metric
    # Filter out rows that don't have values for this metric
    df_metric = df_time[["period", "route", metric]].copy()

    # Convert metric to float, fill NaN with 0 or skip if you prefer
    # (Alternatively, we can skip plotting if all are None.)
    df_metric[metric] = pd.to_numeric(df_metric[metric], errors="coerce")

    for route in sorted(df_metric["route"].unique()):
        df_r = df_metric[df_metric["route"] == route].copy()

        # Build y-values in the correct period order
        y_vals = []
        x_labels = []
        for p in ordered_periods:
            row = df_r[df_r["period"] == p]
            if not row.empty:
                val = row[metric].values[0]
            else:
                val = None
            y_vals.append(val)
            x_labels.append(p)

        # If all Nones, skip
        if all(v is None or pd.isna(v) for v in y_vals):
            continue

        plt.figure(figsize=PLOT_STYLE["figsize"])
        plt.plot(
            x_labels,
            y_vals,
            marker=PLOT_STYLE["marker"],
            linestyle=PLOT_STYLE["linestyle"],
        )
        plt.title(f"{metric.replace('_', ' ').title()} Over Time - Route {route}")
        plt.xlabel("Month")
        plt.ylabel(metric.replace("_", " ").title())
        plt.xticks(rotation=PLOT_STYLE["rotation"])
        plt.grid(PLOT_STYLE["grid"])

        # Try to set a nice y-limit
        # If we have numeric data, let’s set bottom=0, top at 110% of max
        numeric_vals = [v for v in y_vals if v is not None and not pd.isna(v)]
        if numeric_vals:
            y_max = max(numeric_vals) * 1.1
            plt.ylim(0, y_max if y_max > 0 else 1)

        fname = f"{metric}_route_{route}.png"
        outpath = os.path.join(plot_dir, fname)
        plt.tight_layout()
        plt.savefig(outpath, dpi=150)
        plt.close()


def generate_all_plots(df_time: pd.DataFrame, config: dict, plot_config: dict) -> None:
    """Iterate over PLOT_CONFIG flags and call plot_metric_over_time when enabled.

    Args:
        df_time: Time-series DataFrame from build_monthly_timeseries.
        config: Global configuration dictionary.
        plot_config: Boolean flags mapping plot names to metrics.
    """
    # Map the boolean keys to the actual column in df_time
    # You can rename them however you'd like.
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

    for config_key, col_name in metric_map.items():
        if plot_config.get(config_key, False):
            print(f"Generating plots for {col_name} ...")
            plot_metric_over_time(df_time, col_name, config)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Execute the NTD performance workflow.

    Steps
    -----
    1. Read and clean each monthly workbook.
    2. Classify routes/corridors; compute derived metrics.
    3. Export detailed and service-type Excel workbooks.
    4. Export four route-level workbooks (combined + per day type).
    5. Generate optional time-series plots.

    Raises:
        FileNotFoundError: If any configured Excel file is missing.
        PermissionError: If output workbooks cannot be written.
    """
    # ------------------------------------------------------------------
    # 1. READ ALL PERIODS
    # ------------------------------------------------------------------
    data_dict = read_excel_data(CONFIG)

    # ------------------------------------------------------------------
    # 2. CLASSIFY + DERIVED COLUMNS
    # ------------------------------------------------------------------
    for period, df in data_dict.items():
        df["service_type"] = df["ROUTE_NAME"].apply(lambda r: classify_route(r, CONFIG))
        df["corridors"] = df["ROUTE_NAME"].apply(lambda r: classify_corridor(r, CONFIG))
        df = calculate_derived_columns(df)
        df["period"] = period
        data_dict[period] = df  # store back

    # Concatenate for YTD operations
    all_data = pd.concat(data_dict.values(), ignore_index=True)

    # Warn about routes that were not classified
    unknown_routes = all_data.loc[all_data["service_type"] == "unknown", "ROUTE_NAME"].unique()
    if unknown_routes.size:
        print("\nRoutes not classified by SERVICE_TYPE_DICT:")
        print(", ".join(sorted(unknown_routes)))

    # ------------------------------------------------------------------
    # 3. EXPORT DETAILED & SERVICE-TYPE FILES (unchanged)
    # ------------------------------------------------------------------
    output_dir = CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # 3-A  DetailedAllPeriods + monthly sheets
    file1_path = os.path.join(output_dir, "DetailedAllPeriods_andMonthlySheets.xlsx")
    with pd.ExcelWriter(file1_path) as writer:
        all_data.to_excel(writer, sheet_name="DetailedAllPeriods", index=False)
        for period in CONFIG["ordered_periods"]:
            data_dict[period].to_excel(writer, sheet_name=period, index=False)
    print("Detailed data exported.")

    # 3-B  Aggregated by service type (YTD + monthly)
    file2_path = os.path.join(output_dir, "AggByServiceType.xlsx")
    with pd.ExcelWriter(file2_path) as writer:
        aggregate_by_service_type(all_data).to_excel(writer, sheet_name="YTD", index=False)
        for period in CONFIG["ordered_periods"]:
            aggregate_by_service_type(data_dict[period]).to_excel(
                writer, sheet_name=period, index=False
            )
    print("Service-type summaries exported.")

    # ------------------------------------------------------------------
    # 4. EXPORT FOUR ROUTE-LEVEL SUMMARY FILES
    # ------------------------------------------------------------------
    summary_sets = {
        "Combined": all_data,
        "Weekday": all_data[all_data["SERVICE_PERIOD"] == "Weekday"],
        "Saturday": all_data[all_data["SERVICE_PERIOD"] == "Saturday"],
        "Sunday": all_data[all_data["SERVICE_PERIOD"] == "Sunday"],
    }

    for label, subset in summary_sets.items():
        summary_df = route_level_summary(subset)
        file_path = os.path.join(output_dir, f"RouteLevelSummary_{label}.xlsx")
        with pd.ExcelWriter(file_path) as writer:
            sheet_name = f"{label}_Route_Level"
            summary_df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"{label} route-level summary exported → {file_path}")

    # ------------------------------------------------------------------
    # 5. OPTIONAL PLOTS
    # ------------------------------------------------------------------
    df_time = build_monthly_timeseries(all_data, CONFIG)
    generate_all_plots(df_time, CONFIG, PLOT_CONFIG)

    print("All processing complete.")


if __name__ == "__main__":
    main()
