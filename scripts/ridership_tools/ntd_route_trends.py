"""Subset and summarize NTD monthly ridership for selected routes (config-driven).

Reads NTD monthly Excel workbooks listed in PERIODS, filters to configured routes,
and exports per-route monthly summaries and trend plots:
    - Weekday/Saturday/Sunday monthly totals
    - Weekday/Saturday/Sunday per-day averages by month

It logs warnings when:
    - A month in the requested range is missing from PERIODS
    - A workbook file is missing
    - A route-month-service period is missing
    - Ridership is 0 while DAYS > 0 (possible localized data outage)

Optionally prompts users for manual fixes (enter corrected MTH_BOARD and/or DAYS).

Outputs per route (folder: OUTPUT_ROOT/route_<ROUTE>/):
    - monthly_long.csv (month x service_period rows)
    - monthly_wide.csv (one row per month; totals + averages columns)
    - outage_flags.csv
    - plots/monthly_totals.png
    - plots/daily_averages.png
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_ROOT: Final[Path] = Path(r"Path\To\Your\Input_Folder")
OUTPUT_ROOT: Final[Path] = Path(r"Path\To\Your\Output_Folder")

# Requested overall date range (inclusive month starts).
START_MONTH: Final[str] = "Jan-2024"
END_MONTH: Final[str] = "Dec-2025"

# Route subset.
ROUTES: Final[list[str]] = ["101", "202", "303"]

# Service periods expected.
SERVICE_PERIODS: Final[list[str]] = ["Weekday", "Saturday", "Sunday"]

# Prompt users to manually override missing/zero values.
PROMPT_FOR_FIXES: Final[bool] = False

# Logging level (INFO recommended; DEBUG if troubleshooting).
LOG_LEVEL: Final[int] = logging.INFO

# Required columns for this simplified workflow.
REQUIRED_COLS: Final[list[str]] = ["ROUTE_NAME", "SERVICE_PERIOD", "MTH_BOARD", "DAYS"]


@dataclass(frozen=True)
class PeriodSpec:
    """Workbook name and destination sheet."""

    filename: str
    sheet: str


# In-script manifest. Standardized naming pattern.
PERIODS: Final[dict[str, PeriodSpec]] = {
    # 2024
    "Jan-2024": PeriodSpec("JAN 2024 NTD RIDERSHIP BY ROUTE.xlsx", "Temporary_Query_N"),
    "Feb-2024": PeriodSpec("FEB 2024 NTD RIDERSHIP BY ROUTE.xlsx", "Temporary_Query_N"),
    "Mar-2024": PeriodSpec("MAR 2024 NTD RIDERSHIP BY ROUTE.xlsx", "Temporary_Query_N"),
    "Apr-2024": PeriodSpec("APR 2024 NTD RIDERSHIP BY ROUTE.xlsx", "Temporary_Query_N"),
    "May-2024": PeriodSpec("MAY 2024 NTD RIDERSHIP BY ROUTE.xlsx", "Temporary_Query_N"),
    "Jun-2024": PeriodSpec("JUN 2024 NTD RIDERSHIP BY ROUTE.xlsx", "Temporary_Query_N"),
    "Jul-2024": PeriodSpec("JUL 2024 NTD RIDERSHIP BY ROUTE.xlsx", "Temporary_Query_N"),
    "Aug-2024": PeriodSpec("AUG 2024 NTD RIDERSHIP BY ROUTE.xlsx", "Temporary_Query_N"),
    "Sep-2024": PeriodSpec("SEP 2024 NTD RIDERSHIP BY ROUTE.xlsx", "Temporary_Query_N"),
    "Oct-2024": PeriodSpec("OCT 2024 NTD RIDERSHIP BY ROUTE.xlsx", "Temporary_Query_N"),
    "Nov-2024": PeriodSpec("NOV 2024 NTD RIDERSHIP BY ROUTE.xlsx", "Temporary_Query_N"),
    "Dec-2024": PeriodSpec("DEC 2024 NTD RIDERSHIP BY ROUTE.xlsx", "Temporary_Query_N"),
    # 2025
    "Jan-2025": PeriodSpec("JAN 2025 NTD RIDERSHIP BY ROUTE.xlsx", "Temporary_Query_N"),
    "Feb-2025": PeriodSpec("FEB 2025 NTD RIDERSHIP BY ROUTE.xlsx", "Temporary_Query_N"),
    "Mar-2025": PeriodSpec("MAR 2025 NTD RIDERSHIP BY ROUTE.xlsx", "Temporary_Query_N"),
    "Apr-2025": PeriodSpec("APR 2025 NTD RIDERSHIP BY ROUTE.xlsx", "Temporary_Query_N"),
    "May-2025": PeriodSpec("MAY 2025 NTD RIDERSHIP BY ROUTE.xlsx", "Temporary_Query_N"),
    "Jun-2025": PeriodSpec("JUN 2025 NTD RIDERSHIP BY ROUTE.xlsx", "Temporary_Query_N"),
    "Jul-2025": PeriodSpec("JUL 2025 NTD RIDERSHIP BY ROUTE.xlsx", "Temporary_Query_N"),
    "Aug-2025": PeriodSpec("AUG 2025 NTD RIDERSHIP BY ROUTE.xlsx", "Temporary_Query_N"),
    "Sep-2025": PeriodSpec("SEP 2025 NTD RIDERSHIP BY ROUTE.xlsx", "Temporary_Query_N"),
    "Oct-2025": PeriodSpec("OCT 2025 NTD RIDERSHIP BY ROUTE.xlsx", "Temporary_Query_N"),
    "Nov-2025": PeriodSpec("NOV 2025 NTD RIDERSHIP BY ROUTE.xlsx", "Temporary_Query_N"),
    "Dec-2025": PeriodSpec("DEC 2025 NTD RIDERSHIP BY ROUTE.xlsx", "Temporary_Query_N"),
}

PLOT_STYLE: Final[dict[str, Any]] = {
    "figsize": (10, 5),
    "marker": "o",
    "linestyle": "-",
    "grid": True,
    "rotation": 45,
    "dpi": 150,
}

# =============================================================================
# HELPERS
# =============================================================================


def parse_month(value: str) -> datetime:
    """Parse 'Mon-YYYY' into a month-start datetime."""
    dt = datetime.strptime(value.strip(), "%b-%Y")
    return datetime(dt.year, dt.month, 1)


def format_month(dt: datetime) -> str:
    """Format a month-start datetime as 'Mon-YYYY'."""
    return dt.strftime("%b-%Y")


def month_range(start: datetime, end: datetime) -> list[datetime]:
    """Return inclusive list of month-start datetimes from start to end."""
    if start > end:
        return []
    months = pd.date_range(start=start, end=end, freq="MS").to_pydatetime().tolist()
    return [datetime(m.year, m.month, 1) for m in months]


def safe_float(value: Any) -> float | None:
    """Return float if value looks numeric; else None."""
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Upper-case, trim, and replace spaces with underscores in columns."""
    out = df.copy()
    out.columns = out.columns.astype(str).str.strip().str.upper().str.replace(" ", "_", regex=False)
    return out


def normalise_route(value: Any) -> str:
    """Normalize route values to a compact token (e.g., 610.0 -> '610')."""
    s = str(value).strip().upper().replace(" ", "")
    s = re.sub(r"\.0$", "", s)
    return s


def normalise_service_period(value: Any) -> str:
    """Normalize service period values to {Weekday, Saturday, Sunday} where possible."""
    s = str(value).strip().lower()
    mapping = {
        "weekday": "Weekday",
        "week day": "Weekday",
        "wkday": "Weekday",
        "wkdy": "Weekday",
        "sat": "Saturday",
        "saturday": "Saturday",
        "sun": "Sunday",
        "sunday": "Sunday",
    }
    return mapping.get(s, str(value).strip())


def ordered_periods_in_range(start: datetime, end: datetime) -> list[str]:
    """Return PERIODS keys sorted chronologically and filtered to [start, end]."""
    keys = list(PERIODS.keys())
    keys.sort(key=parse_month)
    out: list[str] = []
    for k in keys:
        dt = parse_month(k)
        if start <= dt <= end:
            out.append(k)
    return out


# =============================================================================
# IO + TRANSFORM
# =============================================================================


def read_month_workbook(period: str, spec: PeriodSpec) -> pd.DataFrame:
    """Read one workbook and return normalized rows (unfiltered)."""
    path = DATA_ROOT / spec.filename
    if not path.exists():
        logging.warning("Workbook missing on disk: %s (period=%s)", path, period)
        return pd.DataFrame()

    converters: Any = {"MTH_BOARD": safe_float, "DAYS": safe_float}
    try:
        df = pd.read_excel(path, sheet_name=spec.sheet, converters=converters)
    except Exception:
        logging.exception(
            "Failed to read workbook: %s (period=%s sheet=%s)", path, period, spec.sheet
        )
        return pd.DataFrame()

    df = normalise_columns(df)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        logging.warning(
            "Workbook missing required columns (period=%s file=%s): %s",
            period,
            path.name,
            ", ".join(missing),
        )
        return pd.DataFrame()

    out = df.copy()
    out["ROUTE_NAME"] = out["ROUTE_NAME"].apply(normalise_route)
    out["SERVICE_PERIOD"] = out["SERVICE_PERIOD"].apply(normalise_service_period)

    out["period"] = period
    out["period_dt"] = parse_month(period)
    return out


def load_raw_subset(periods: list[str]) -> pd.DataFrame:
    """Load all months in `periods` and filter to configured routes + service periods."""
    frames: list[pd.DataFrame] = []
    route_set = set(ROUTES)
    sp_set = set(SERVICE_PERIODS)

    for period in periods:
        spec = PERIODS.get(period)
        if spec is None:
            continue
        df = read_month_workbook(period, spec)
        if df.empty:
            continue

        df = df[df["ROUTE_NAME"].isin(route_set)].copy()
        df = df[df["SERVICE_PERIOD"].isin(sp_set)].copy()

        frames.append(df)
        logging.info("Loaded %s: %d rows kept", period, len(df))

    if not frames:
        return pd.DataFrame(columns=[*REQUIRED_COLS, "period", "period_dt"])

    return pd.concat(frames, ignore_index=True)


def aggregate_monthly_long(
    raw: pd.DataFrame,
    expected_months: list[datetime],
) -> tuple[pd.DataFrame, set[tuple[str, datetime, str]]]:
    """Aggregate raw rows to monthly-long and reindex to a full grid.

    Returns:
        monthly_long: one row per (route, period_dt, service_period)
        observed_keys: set of (route, period_dt, service_period) observed in raw aggregates
    """
    if raw.empty:
        idx = pd.MultiIndex.from_product(
            [ROUTES, expected_months, SERVICE_PERIODS],
            names=["route", "period_dt", "service_period"],
        )
        monthly_long = idx.to_frame(index=False)
        monthly_long["period"] = monthly_long["period_dt"].apply(format_month)
        monthly_long["mth_board"] = pd.NA
        monthly_long["days"] = pd.NA
        monthly_long["daily_avg"] = pd.NA
        return monthly_long, set()

    agg = (
        raw.groupby(["ROUTE_NAME", "period_dt", "SERVICE_PERIOD"], as_index=False)
        .agg({"MTH_BOARD": "sum", "DAYS": "sum"})
        .rename(
            columns={
                "ROUTE_NAME": "route",
                "SERVICE_PERIOD": "service_period",
                "MTH_BOARD": "mth_board",
                "DAYS": "days",
            }
        )
    )

    observed_keys = set(
        (str(r), dt, str(sp))
        for r, dt, sp in zip(agg["route"], agg["period_dt"], agg["service_period"], strict=True)
    )

    idx = pd.MultiIndex.from_product(
        [ROUTES, expected_months, SERVICE_PERIODS],
        names=["route", "period_dt", "service_period"],
    )
    monthly_long = (
        agg.set_index(["route", "period_dt", "service_period"]).reindex(idx).reset_index()
    )
    monthly_long["period"] = monthly_long["period_dt"].apply(format_month)

    monthly_long["daily_avg"] = pd.NA
    has_days = pd.to_numeric(monthly_long["days"], errors="coerce") > 0
    monthly_long.loc[has_days, "daily_avg"] = pd.to_numeric(
        monthly_long.loc[has_days, "mth_board"], errors="coerce"
    ) / pd.to_numeric(monthly_long.loc[has_days, "days"], errors="coerce")

    return monthly_long, observed_keys


def flag_outages(
    monthly_long: pd.DataFrame,
    expected_months: list[datetime],
    observed_keys: set[tuple[str, datetime, str]],
) -> pd.DataFrame:
    """Flag missing/zero/suspicious values, and log warnings."""
    flags: list[dict[str, Any]] = []

    expected_month_set = set(expected_months)
    available_manifest_months = {parse_month(k) for k in PERIODS.keys()}

    # Missing months in manifest
    for m in sorted(expected_month_set):
        if m not in available_manifest_months:
            logging.warning("Missing month in PERIODS manifest: %s", format_month(m))

    # Row-level flags
    for _, r in monthly_long.iterrows():
        route = str(r["route"])
        period_dt = r["period_dt"]
        period = str(r["period"])
        sp = str(r["service_period"])

        key = (route, period_dt, sp)
        was_observed = key in observed_keys

        b_num = pd.to_numeric(r["mth_board"], errors="coerce")
        d_num = pd.to_numeric(r["days"], errors="coerce")

        if not was_observed:
            flags.append(
                {
                    "route": route,
                    "period": period,
                    "service_period": sp,
                    "flag": "missing_service_period",
                    "mth_board": pd.NA,
                    "days": pd.NA,
                }
            )
            continue

        if pd.isna(b_num) and pd.isna(d_num):
            flags.append(
                {
                    "route": route,
                    "period": period,
                    "service_period": sp,
                    "flag": "missing_entry",
                    "mth_board": pd.NA,
                    "days": pd.NA,
                }
            )
            continue

        if not pd.isna(d_num) and d_num == 0:
            flags.append(
                {
                    "route": route,
                    "period": period,
                    "service_period": sp,
                    "flag": "zero_days",
                    "mth_board": b_num,
                    "days": d_num,
                }
            )

        if not pd.isna(b_num) and not pd.isna(d_num) and b_num == 0 and d_num > 0:
            flags.append(
                {
                    "route": route,
                    "period": period,
                    "service_period": sp,
                    "flag": "zero_ridership_nonzero_days",
                    "mth_board": b_num,
                    "days": d_num,
                }
            )

    out = pd.DataFrame(flags).drop_duplicates(ignore_index=True)

    for _, f in out.iterrows():
        logging.warning(
            "Flag: route=%s period=%s service_period=%s flag=%s boards=%s days=%s",
            f["route"],
            f["period"],
            f["service_period"],
            f["flag"],
            f["mth_board"],
            f["days"],
        )

    return out


def apply_manual_fixes(monthly_long: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Prompt user to manually override missing/zero values (if enabled)."""
    if not PROMPT_FOR_FIXES or flags.empty:
        return monthly_long

    updated = monthly_long.copy()
    stop = False

    def prompt_float(msg: str) -> float | None | str:
        raw = input(msg).strip()
        if raw.lower() == "q":
            return "q"
        if raw == "":
            return None
        try:
            return float(raw.replace(",", ""))
        except ValueError:
            logging.warning("Invalid numeric input %r; keeping existing value.", raw)
            return None

    fixable = flags[
        flags["flag"].isin(
            {"missing_service_period", "missing_entry", "zero_ridership_nonzero_days"}
        )
    ].copy()
    fixable.sort_values(["route", "period", "service_period", "flag"], inplace=True)

    for _, f in fixable.iterrows():
        if stop:
            break

        route = str(f["route"])
        period = str(f["period"])
        sp = str(f["service_period"])
        flag = str(f["flag"])

        mask = (
            (updated["route"] == route)
            & (updated["period"] == period)
            & (updated["service_period"] == sp)
        )
        if mask.sum() != 1:
            logging.warning(
                "Fix skipped; could not uniquely locate row: %s %s %s", route, period, sp
            )
            continue

        cur_days = updated.loc[mask, "days"].iloc[0]

        logging.warning("Interactive fix candidate: %s %s %s (%s)", route, period, sp, flag)

        if flag in {"missing_service_period", "missing_entry"}:
            b_in = prompt_float(
                f"[{route} | {period} | {sp}] Missing. Enter MTH_BOARD (Enter=skip, q=quit): "
            )
            if b_in == "q":
                stop = True
                break
            d_in = prompt_float(
                f"[{route} | {period} | {sp}] Missing. Enter DAYS (Enter=skip, q=quit): "
            )
            if d_in == "q":
                stop = True
                break

            if b_in is not None:
                updated.loc[mask, "mth_board"] = b_in
            if d_in is not None:
                updated.loc[mask, "days"] = d_in

        elif flag == "zero_ridership_nonzero_days":
            b_in = prompt_float(
                f"[{route} | {period} | {sp}] boards=0 days={cur_days}. Enter corrected MTH_BOARD "
                "(Enter=keep 0, q=quit): "
            )
            if b_in == "q":
                stop = True
                break
            if b_in is not None:
                updated.loc[mask, "mth_board"] = b_in

    # Recompute daily_avg after edits
    updated["daily_avg"] = pd.NA
    has_days = pd.to_numeric(updated["days"], errors="coerce") > 0
    updated.loc[has_days, "daily_avg"] = pd.to_numeric(
        updated.loc[has_days, "mth_board"], errors="coerce"
    ) / pd.to_numeric(updated.loc[has_days, "days"], errors="coerce")

    return updated


def to_wide(monthly_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot monthly-long into a single row per month with totals and averages."""
    base = monthly_long[
        ["route", "period_dt", "period", "service_period", "mth_board", "daily_avg"]
    ].copy()

    totals = base.pivot_table(
        index=["route", "period_dt", "period"],
        columns="service_period",
        values="mth_board",
        aggfunc="first",
    )
    avgs = base.pivot_table(
        index=["route", "period_dt", "period"],
        columns="service_period",
        values="daily_avg",
        aggfunc="first",
    )

    totals.columns = [f"{c.lower()}_total" for c in totals.columns]
    avgs.columns = [f"{c.lower()}_avg" for c in avgs.columns]

    out = pd.concat([totals, avgs], axis=1).reset_index()
    out.sort_values(["route", "period_dt"], inplace=True, ignore_index=True)
    return out


# =============================================================================
# PLOTTING + EXPORT
# =============================================================================


def plot_route_totals(route_dir: Path, route: str, wide: pd.DataFrame) -> None:
    """Plot monthly totals for a single route."""
    df = wide[wide["route"] == route].copy()
    if df.empty:
        return

    out_path = route_dir / "plots" / "monthly_totals.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=PLOT_STYLE["figsize"])
    x = df["period_dt"]

    for col in ["weekday_total", "saturday_total", "sunday_total"]:
        if col in df.columns:
            plt.plot(
                x,
                df[col],
                marker=PLOT_STYLE["marker"],
                linestyle=PLOT_STYLE["linestyle"],
                label=col,
            )

    plt.title(f"Monthly Ridership Totals – Route {route}")
    plt.xlabel("Month")
    plt.ylabel("Boardings (Monthly Total)")
    plt.grid(PLOT_STYLE["grid"])
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=PLOT_STYLE["rotation"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_STYLE["dpi"])
    plt.close()


def plot_route_avgs(route_dir: Path, route: str, wide: pd.DataFrame) -> None:
    """Plot per-day averages for a single route."""
    df = wide[wide["route"] == route].copy()
    if df.empty:
        return

    out_path = route_dir / "plots" / "daily_averages.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=PLOT_STYLE["figsize"])
    x = df["period_dt"]

    for col in ["weekday_avg", "saturday_avg", "sunday_avg"]:
        if col in df.columns:
            plt.plot(
                x,
                df[col],
                marker=PLOT_STYLE["marker"],
                linestyle=PLOT_STYLE["linestyle"],
                label=col,
            )

    plt.title(f"Per-Day Ridership Averages – Route {route}")
    plt.xlabel("Month")
    plt.ylabel("Boardings (Per Day Average)")
    plt.grid(PLOT_STYLE["grid"])
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=PLOT_STYLE["rotation"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_STYLE["dpi"])
    plt.close()


def export_route(
    route: str, monthly_long: pd.DataFrame, wide: pd.DataFrame, flags: pd.DataFrame
) -> None:
    """Export CSVs and plots into a per-route folder."""
    route_dir = OUTPUT_ROOT / f"route_{route}"
    route_dir.mkdir(parents=True, exist_ok=True)

    monthly_long[monthly_long["route"] == route].to_csv(route_dir / "monthly_long.csv", index=False)
    wide[wide["route"] == route].to_csv(route_dir / "monthly_wide.csv", index=False)
    flags[flags["route"] == route].to_csv(route_dir / "outage_flags.csv", index=False)

    plot_route_totals(route_dir, route, wide)
    plot_route_avgs(route_dir, route, wide)

    logging.info("Exported %s", route_dir)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Run the end-to-end subset workflow."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(message)s")

    start_dt = parse_month(START_MONTH)
    end_dt = parse_month(END_MONTH)
    expected_months = month_range(start_dt, end_dt)

    periods = ordered_periods_in_range(start_dt, end_dt)
    if not periods:
        logging.warning("No PERIODS entries overlap requested range %s..%s", START_MONTH, END_MONTH)

    raw = load_raw_subset(periods)

    monthly_long, observed_keys = aggregate_monthly_long(raw, expected_months)
    flags = flag_outages(monthly_long, expected_months, observed_keys)

    monthly_long = apply_manual_fixes(monthly_long, flags)

    # Recompute flags after manual edits (so exports reflect final used values).
    flags = flag_outages(monthly_long, expected_months, observed_keys)

    wide = to_wide(monthly_long)

    for route in ROUTES:
        export_route(route, monthly_long, wide, flags)

    combined_dir = OUTPUT_ROOT / "_combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    monthly_long.to_csv(combined_dir / "all_routes_monthly_long.csv", index=False)
    wide.to_csv(combined_dir / "all_routes_monthly_wide.csv", index=False)
    flags.to_csv(combined_dir / "all_routes_outage_flags.csv", index=False)

    logging.info("Done. Outputs written to: %s", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
