"""Transforms a *compiled* long-format NTD dataset into wide-format and creates plots.

Input:  One CSV/XLSX with columns
        - ROUTE_NAME   (string)
        - PERIOD       (e.g. "September-23" or "2023-09")
        - DAY_TYPE     ("Weekday", "Saturday", "Sunday"; case-insensitive)
        - MTH_BOARD    (monthly boardings for that day-type)
        - DAYS         (number of days in the month of that day-type)

Output: Wide CSV identical to the one produced by the old script, plus
        optional plots.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, List, Sequence, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================
# --- Part 0 : Input / output --------------------------------------------------
COMPILED_INPUT_FILE: str = r"\\File\Path\To\Your\Input_Folder\Compiled_NTD_Data.csv"
OUTPUT_DIR: str = r"\\File\Path\To\Your\Output_Folder"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Columns as they exist in the compiled file -----------------------------
COL_ROUTE: str = "ROUTE_NAME"
COL_PERIOD: str = "MTH_YR"
COL_DAYTYPE: str = "SERVICE_PERIOD"
COL_RIDERSHIP: str = "MTH_BOARD"
COL_DAYS: str = "DAYS"

# --- Business rules ----------------------------------------------------------
ROUTES_TO_EXCLUDE: List[str] = ["101", "202", "303"]

# --- Plotting ---------------------------------------------------------------
ENABLE_PLOTTING = True
ROUTES_OF_INTEREST: List[str] = []  # keep empty = all
PLOTS_OUTPUT_FOLDER = os.path.join(OUTPUT_DIR, "Plots")
os.makedirs(PLOTS_OUTPUT_FOLDER, exist_ok=True)
FIG_SIZE = (10, 6)
MARKER_STYLE = "o"
LINE_STYLE = "-"
LINE_WIDTH = 2.0
DAYTYPES_TO_PLOT = [
    "WeekdayTotal",
    "SaturdayTotal",
    "SundayTotal",
    "MonthlyTotal",
    "WeekdayAverage",
    "SaturdayAverage",
    "SundayAverage",
]

# Mapping of canonical day types
DAYTYPE_NORMALISER = {
    "WEEKDAY": "WEEKDAYS",
    "SATURDAY": "SATURDAYS",
    "SUNDAY": "SUNDAYS",
}


# =============================================================================
# FUNCTIONS
# =============================================================================
def load_compiled_dataset(path: str) -> pd.DataFrame:
    """Read input file, validate/clean columns, and return a long-format DataFrame ready for pivoting.

    Expected external constants:
        COL_ROUTE, COL_PERIOD, COL_DAYTYPE, COL_RIDERSHIP, COL_DAYS
        ROUTES_TO_EXCLUDE, DAYTYPE_NORMALISER
    """
    # --------------------------------------------------------------------- I/O
    if not os.path.exists(path):
        sys.exit(f"ERROR: compiled input file not found → {path}")

    read_func = pd.read_excel if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv
    # Read as *object* so nothing is silently coerced to float
    df = read_func(path, dtype="object")

    # ------------------------------------------------------------------ schema
    required = [COL_ROUTE, COL_PERIOD, COL_DAYTYPE, COL_RIDERSHIP, COL_DAYS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: compiled file missing required columns: {missing}")

    # ------------------------------------------------------------- PERIOD CLEAN

    def _normalise_period(v: Any) -> str | float:
        """Ensure period values are clean strings (handles NaN, 202309.0, etc.).

        Args:  # <--- Corrected indentation
            v: The raw period value coming from the spreadsheet.  Can be a string,
               an Excel float/int, or a pandas NA/NumPy NaN.

        Returns: # <--- Corrected indentation
            A canonicalised representation:
            * `str` when the value is valid (e.g. ``"202309"`` or ``"Sep-23"``).
            * `float("nan")` when the value is missing/blank.
        """
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, float)):  # Excel numeric artefacts
            return str(int(v))
        return str(v).strip()

    df[COL_PERIOD] = df[COL_PERIOD].apply(_normalise_period)

    # ------------------------------------------------------------- ROUTE CLEAN
    df[COL_ROUTE] = (
        df[COL_ROUTE]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(" ", "", regex=False)
        .str.replace(r"\.0$", "", regex=True)
    )

    # Exclude unwanted routes
    df = df[~df[COL_ROUTE].isin(ROUTES_TO_EXCLUDE)].copy()

    # ------------------------------------------------------- DAY-TYPE CLEANING
    df[COL_DAYTYPE] = df[COL_DAYTYPE].astype(str).str.strip()
    # Drop rows with blank SERVICE_PERIOD values (totals/notes)
    df = df[df[COL_DAYTYPE].notna() & (df[COL_DAYTYPE] != "")]

    # Map to canonical names (WEEKDAYS / SATURDAYS / SUNDAYS)
    df[COL_DAYTYPE] = (
        df[COL_DAYTYPE]
        .str.upper()
        .map(DAYTYPE_NORMALISER)  # returns NaN if unmapped
        .fillna(df[COL_DAYTYPE])
    )

    unknown = df[~df[COL_DAYTYPE].isin(DAYTYPE_NORMALISER.values())][COL_DAYTYPE].unique()
    if unknown.size:
        print(
            f"WARNING: unknown DAY_TYPE values encountered {unknown}; rows kept but may be ignored later."
        )

    # ------------------------------------------------------------- NUMERIC CAST
    for col in (COL_RIDERSHIP, COL_DAYS):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # ------------------------------------------------------------- MONTH ABBR.
    def to_month_abbr(s: str) -> str:
        """Parse many period formats → 'Sep-23'. Falls back to original string."""
        fmts = ("%B-%y", "%Y-%m", "%b-%y", "%Y%m", "%m/%Y")
        for fmt in fmts:
            try:
                return pd.to_datetime(s, format=fmt).strftime("%b-%y")
            except ValueError:
                continue
        # last-chance flexible parse
        dt = pd.to_datetime(s, errors="coerce")
        return dt.strftime("%b-%y") if pd.notna(dt) else str(s).strip()

    df["MONTH_ABBR"] = df[COL_PERIOD].apply(to_month_abbr)

    # --------------------------------------------------------- DUPLICATE CHECK
    dup_mask = df.duplicated([COL_ROUTE, "MONTH_ABBR", COL_DAYTYPE], keep=False)
    if dup_mask.any():
        dups = df.loc[dup_mask, [COL_ROUTE, COL_PERIOD, COL_DAYTYPE]]
        print("WARNING: duplicate (route, month, day-type) rows detected:\n", dups.head())
        # Aggregate duplicates (summing ridership and days)
        df = df.groupby([COL_ROUTE, "MONTH_ABBR", COL_DAYTYPE], as_index=False).agg(
            {COL_RIDERSHIP: "sum", COL_DAYS: "sum"}
        )

    return df


def pivot_to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """Convert the long-format DataFrame into a wide, route-by-month table.

    Output columns follow the pattern
        '<Mon-YY>_<Weekday|Saturday|Sunday><Total|Days|Average>'
    plus an aggregate
        '<Mon-YY>_MonthlyTotal'.

    Args:
        df_long: Long-format DataFrame returned by ``load_compiled_dataset``.

    Returns:
        Wide-format DataFrame with one row per route and one column per
        (month × metric) combination.
    """
    # ──────────────────────────────── averages ───────────────────────────────
    df_long = df_long.copy()
    df_long["AVERAGE"] = np.where(
        df_long[COL_DAYS] > 0,
        df_long[COL_RIDERSHIP] / df_long[COL_DAYS],
        0,
    )

    # ──────────────────────────────── pivots ────────────────────────────────
    pieces: list[pd.DataFrame] = []
    for metric, val_col in [
        ("Total", COL_RIDERSHIP),
        ("Days", COL_DAYS),
        ("Average", "AVERAGE"),
    ]:
        tmp = df_long.pivot(
            index=COL_ROUTE,
            columns=["MONTH_ABBR", COL_DAYTYPE],
            values=val_col,
        )

        # Flatten MultiIndex column labels in a mypy-friendly way
        cols: Sequence[Tuple[str, str]] = cast("Sequence[Tuple[str, str]]", tmp.columns.to_list())
        tmp.columns = [
            f"{month}_{day_type.title().rstrip('s')}{metric}" for month, day_type in cols
        ]
        pieces.append(tmp)

    # Concatenate metric tables horizontally
    wide = pd.concat(pieces, axis=1)

    # Bring ROUTE_NAME out of the index
    wide.reset_index(inplace=True)

    # ──────────────────────── monthly grand totals ──────────────────────────
    month_tags = {c.split("_")[0] for c in wide.columns if c.endswith("WeekdayTotal")}
    for month in month_tags:
        total_cols = [
            f"{month}_{x}Total"
            for x in ("Weekday", "Saturday", "Sunday")
            if f"{month}_{x}Total" in wide.columns
        ]
        if total_cols:
            wide[f"{month}_MonthlyTotal"] = wide[total_cols].sum(axis=1)

    # ────────────────────────────── route sorting ───────────────────────────
    try:
        wide["ROUTE_SORT"] = wide[COL_ROUTE].astype(int)
    except ValueError:
        wide["ROUTE_SORT"] = wide[COL_ROUTE]

    wide.sort_values("ROUTE_SORT", inplace=True)
    wide.drop(columns="ROUTE_SORT", inplace=True)

    return wide


def save_wide_csv(df_wide: pd.DataFrame) -> str:
    """Saves the wide-format DataFrame to a CSV file.

    The output file name is "Consolidated_Ridership_Data.csv" and it will be
    saved in the directory specified by `OUTPUT_DIR`.

    Args:
        df_wide (pd.DataFrame): The DataFrame in wide format to be saved.

    Returns:
        str: The full path to the saved CSV file.
    """
    out_csv = os.path.join(OUTPUT_DIR, "Consolidated_Ridership_Data.csv")
    df_wide.to_csv(out_csv, index=False)
    print(f"Wide file saved → {out_csv}")
    return out_csv


# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------
def plot_ridership(df_wide: pd.DataFrame) -> None:
    """Generates and saves ridership plots for each route and specified metric.

    Plots are created for each route present in `df_wide` (or a subset if
    `ROUTES_OF_INTEREST` is specified). Each plot displays a time series of
    ridership for a specific day type and metric (e.g., "WeekdayTotal",
    "MonthlyAverage"). Plots are saved as PNG files in the directory
    specified by `PLOTS_OUTPUT_FOLDER`.

    Expected external constants:
        COL_ROUTE, ROUTES_OF_INTEREST, DAYTYPES_TO_PLOT, PLOTS_OUTPUT_FOLDER,
        FIG_SIZE, MARKER_STYLE, LINE_STYLE, LINE_WIDTH

    Args:
        df_wide (pd.DataFrame): The wide-format DataFrame containing ridership
                                data.
    """
    pattern = re.compile(
        r"^([A-Za-z]{3}-\d{2})_(Weekday|Saturday|Sunday|Monthly)(Total|Days|Average)$"
    )

    matched_meta = []
    for col in df_wide.columns:
        if col == COL_ROUTE:
            continue
        m = pattern.match(col)
        if m:
            matched_meta.append((col, *m.groups()))  # (col, month, daytype, suffix)

    if not matched_meta:
        print("No *_Total/Days/Average columns found. Nothing to plot.")
        return

    routes = (
        df_wide if not ROUTES_OF_INTEREST else df_wide[df_wide[COL_ROUTE].isin(ROUTES_OF_INTEREST)]
    )

    for _, row in routes.iterrows():
        route = row[COL_ROUTE]
        # bucket by variable name ("WeekdayTotal", "MonthlyTotal", …)
        series_lookup: Dict[str, List[Tuple[str, Any]]] = {}
        for col, mon, dt, suf in matched_meta:
            tag = f"{dt}{suf}"  # e.g., WeekdayTotal
            if tag not in DAYTYPES_TO_PLOT:
                continue
            series_lookup.setdefault(tag, []).append((mon, row[col]))

        # plot one figure per tag
        for tag, pts in series_lookup.items():
            pts.sort(key=lambda x: pd.to_datetime(x[0], format="%b-%y", errors="coerce"))
            months, vals = zip(*pts) if pts else ([], [])
            if not months:
                continue

            plt.figure(figsize=FIG_SIZE)
            plt.plot(
                months,
                vals,
                marker=MARKER_STYLE,
                linestyle=LINE_STYLE,
                linewidth=LINE_WIDTH,
            )
            plt.title(f"{tag} — Route {route}")
            plt.xlabel("Month")
            plt.ylabel("Ridership")
            plt.xticks(rotation=45, ha="right")
            plt.grid(True)
            plt.tight_layout()

            fname = re.sub(r"[^\w]+", "_", f"Route_{route}_{tag}.png")
            path_out = os.path.join(PLOTS_OUTPUT_FOLDER, fname)
            try:
                plt.savefig(path_out)
                print(f"Plot saved: {path_out}")
            finally:
                plt.close()


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    """Main function to execute the data transformation and plotting workflow.

    Loads the compiled NTD dataset, transforms it into a wide format,
    saves the wide-format data to a CSV, and optionally generates ridership plots.
    """
    df_long = load_compiled_dataset(COMPILED_INPUT_FILE)
    if df_long.empty:
        sys.exit("ERROR: compiled dataset is empty after cleaning.")

    df_wide = pivot_to_wide(df_long)
    save_wide_csv(df_wide)

    if ENABLE_PLOTTING:
        plot_ridership(df_wide)
    else:
        print("Plotting disabled.")

    print("\nDone.")


if __name__ == "__main__":
    main()
