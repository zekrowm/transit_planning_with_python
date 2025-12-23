#!/usr/bin/env python3
"""Processes ridecheck runtime data and generates trip-level runtime summaries.

This script reads an Excel export from Ridecheck or other transit software and
generates wide-format CSVs for each route and direction. Each output file
includes one row per trip and one column per segment, with runtimes pivoted
from the long-form table.

It supports three types of runtime output:
    - Actual runtime
    - Scheduled runtime
    - Difference (Actual - Scheduled)

Optionally, it appends total trip runtime columns and writes a separate
log of trips that deviate significantly from the schedule—based on either
absolute time difference or percentage overage.

Typical use case:
    - Pivot runtimes for quality control
    - Identify outlier trips
    - Prepare data for visualization or reporting

Output:
    - CSVs with pivoted runtimes (one per route/direction/metric)
    - A plain-text anomaly log (optional)
"""

from __future__ import annotations

import logging
import pathlib
import re
import sys
from typing import Final, Iterable

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_PATH: Final = r"\\Path\To\Your\RUNTIME_SUMMARY_BY_ROUTE_AND_DIRECTION_(CHART).XLSX"
OUTPUT_DIR: Final = pathlib.Path(r"\\Path\To\Your\Output\Folder")
OUTPUT_DIR.mkdir(exist_ok=True)

METRICS: Final[dict[str, str]] = {
    "RUNNING_TIME_ACT": "act",
    "RUNNING_TIME_SCH": "sch",
    "RUNNING_TIME_DIFF": "diff",
}

# Trip‑total feature (must stay ON if you want the log)
ADD_TOTAL_COLUMNS: Final[bool] = True
TOTAL_NAMES: Final[dict[str, str]] = {
    "RUNNING_TIME_ACT": "TOTAL_ACT",
    "RUNNING_TIME_SCH": "TOTAL_SCH",
    "RUNNING_TIME_DIFF": "TOTAL_DIFF",
}

WRITE_LOG: Final[bool] = True  # master switch
THRESH_MINUTES: Final[float] = 10.0  # minutes
THRESH_PCT: Final[float] = 0.10  # 10 %
LOG_FILENAME: Final[str] = "runtime_anomalies.txt"

INDEX_COLS: Final[list[str]] = ["ROUTE_NUMBER", "DIRECTION_NAME", "TRIP_KEY"]

# =============================================================================
# FUNCTIONS
# =============================================================================


def read_and_clean(filepath: str | pathlib.Path) -> pd.DataFrame:
    """Load a Ridecheck export and apply minimal cleaning.

    * Normalises ``TRIP_START_TIME`` to 24‑hour ``HH:MM`` text.
    * Builds a compact ``SEGMENT_LABEL`` as
      ``<TIMEPOINT_NAME_1>-<TIMEPOINT_NAME_2>``.

    Args:
        filepath: Path to the Excel file to ingest (absolute or relative).

    Returns:
        Cleaned long‑format :class:`pandas.DataFrame` ready for pivoting.
    """
    df = pd.read_excel(filepath, engine="openpyxl")

    # Clean TRIP_START_TIME → HH:MM 24‑hr
    if pd.api.types.is_datetime64_any_dtype(df["TRIP_START_TIME"]):
        df["TRIP_START_TIME"] = df["TRIP_START_TIME"].dt.strftime("%H:%M").fillna("")
    else:
        df["TRIP_START_TIME"] = (
            df["TRIP_START_TIME"]
            .astype(str)
            .str.extract(r"(\d{1,2}:\d{2})", expand=False)
            .fillna("")
        )

    # Create human‑readable segment label
    df["SEGMENT_LABEL"] = (
        df["TIMEPOINT_NAME_1"].str.strip() + "-" + df["TIMEPOINT_NAME_2"].str.strip()
    )
    return df


def _append_totals(
    wide: pd.DataFrame,
    metric: str,
    seg_columns: Iterable[str],
) -> pd.DataFrame:
    if not ADD_TOTAL_COLUMNS:
        return wide

    total_col = TOTAL_NAMES.get(metric, f"TOTAL_{metric}")
    wide[total_col] = wide.loc[:, seg_columns].sum(axis=1, skipna=True)
    return wide


def pivot_one_metric(
    df: pd.DataFrame,
    metric: str,
    segment_order: dict[str, int],
) -> pd.DataFrame:
    """Pivot a single runtime metric to wide trip‑segment format.

    Args:
        df: Long‑format data for **one** route and direction.
        metric: Runtime column to pivot (e.g. ``"RUNNING_TIME_ACT"``).
        segment_order: Mapping ``SEGMENT_LABEL → SORT_ORDER_1`` that
            preserves the correct left‑to‑right column order.

    Returns:
        Wide‑format :class:`pandas.DataFrame` with one row per trip and
        one column per segment, plus optional total column.
    """
    wide = (
        df.pivot_table(
            index=INDEX_COLS + ["TRIP_START_TIME"],
            columns="SEGMENT_LABEL",
            values=metric,
            aggfunc="first",
        )
        .rename_axis(columns=None)
        .reset_index()
    )

    seg_cols = [c for c in wide.columns if c not in INDEX_COLS + ["TRIP_START_TIME"]]
    seg_cols_sorted = sorted(seg_cols, key=lambda s: segment_order.get(s, 9_999_999))
    wide = wide[INDEX_COLS + ["TRIP_START_TIME"] + seg_cols_sorted]

    wide = _append_totals(wide, metric, seg_cols_sorted)
    return wide


def slugify(text: str) -> str:
    """Convert free‑form text to a filesystem‑safe slug.

    Collapses whitespace to underscores and removes any character that is
    **not** ``A–Z``, ``a–z``, ``0–9``, ``.``, ``_`` or ``-``.

    Args:
        text: Arbitrary string (route, direction, etc.).

    Returns:
        Sanitised slug suitable for use in file names.
    """
    return re.sub(r"[^A-Za-z0-9._-]", "", re.sub(r"\s+", "_", text.strip()))


def _row_exceeds_threshold(total_sch: float, total_diff: float) -> bool:
    """True when |diff| > minute threshold OR > pct threshold of schedule."""
    if pd.isna(total_sch) or total_sch == 0:
        return False
    abs_diff = abs(total_diff)
    return abs_diff > THRESH_MINUTES or abs_diff / total_sch > THRESH_PCT


def _format_log_line(row: pd.Series) -> str:
    return (
        f"{row.ROUTE_NUMBER},{row.DIRECTION_NAME},{row.TRIP_KEY},"
        f"{row.TRIP_START_TIME},{row.TOTAL_SCH:.2f},{row.TOTAL_ACT:.2f},"
        f"{row.TOTAL_DIFF:+.2f}"
    )


def export_tables_and_log(
    df: pd.DataFrame,
    metrics: dict[str, str],
    output_dir: pathlib.Path,
) -> None:
    """Pivot tables, write per‑route CSVs, and emit an anomaly log.

    Args:
        df: Cleaned long‑format DataFrame for all routes.
        metrics: Mapping of runtime‑column → filename‑suffix.
        output_dir: Folder where all output files and the log are written.
    """
    if WRITE_LOG and not ADD_TOTAL_COLUMNS:
        logging.warning("⚠  WRITE_LOG is True but ADD_TOTAL_COLUMNS is False – logging disabled.")
        log_entries: list[str] = []
    else:
        log_entries: list[str] = []

    # ------------------------------------------------------------------ #
    # 1.  Iterate over each (Route, Direction) group
    # ------------------------------------------------------------------ #
    for (route, direction), group in df.groupby(["ROUTE_NUMBER", "DIRECTION_NAME"]):
        # Build a look‑up for SORT_ORDER_1 → ensures lateral column order
        order_map = (
            group[["SEGMENT_LABEL", "SORT_ORDER_1"]]
            .drop_duplicates()
            .set_index("SEGMENT_LABEL")["SORT_ORDER_1"]
            .to_dict()
        )

        # 1‑A.  Pivot each metric to wide format and write the CSV
        pivots: dict[str, pd.DataFrame] = {}
        for metric, suffix in metrics.items():
            wide = pivot_one_metric(group, metric, order_map)
            pivots[metric] = wide

            fname = f"{slugify(str(route))}_{slugify(str(direction))}_{suffix}.csv"
            wide.to_csv(output_dir / fname, index=False)
            logging.info("✓ Wrote %s", fname)

        # ------------------------------------------------------------------ #
        # 1‑B.  Build the anomaly log (needs trip‑total columns)
        # ------------------------------------------------------------------ #
        if WRITE_LOG and ADD_TOTAL_COLUMNS:
            try:
                sch = pivots["RUNNING_TIME_SCH"]
                act = pivots["RUNNING_TIME_ACT"]
                diff = pivots["RUNNING_TIME_DIFF"]
            except KeyError:
                # One of the pivots is missing; skip logging for this group.
                continue

            # Join TOTAL_SCH, TOTAL_ACT, TOTAL_DIFF into a single table
            totals = (
                sch[INDEX_COLS + ["TRIP_START_TIME", TOTAL_NAMES["RUNNING_TIME_SCH"]]]
                .rename(columns={TOTAL_NAMES["RUNNING_TIME_SCH"]: "TOTAL_SCH"})
                .merge(
                    act[INDEX_COLS + ["TRIP_START_TIME", TOTAL_NAMES["RUNNING_TIME_ACT"]]].rename(
                        columns={TOTAL_NAMES["RUNNING_TIME_ACT"]: "TOTAL_ACT"}
                    ),
                    on=INDEX_COLS + ["TRIP_START_TIME"],
                    how="inner",
                )
                .merge(
                    diff[INDEX_COLS + ["TRIP_START_TIME", TOTAL_NAMES["RUNNING_TIME_DIFF"]]].rename(
                        columns={TOTAL_NAMES["RUNNING_TIME_DIFF"]: "TOTAL_DIFF"}
                    ),
                    on=INDEX_COLS + ["TRIP_START_TIME"],
                    how="inner",
                )
            )

            # Flag rows exceeding either threshold
            mask = totals.apply(lambda r: _row_exceeds_threshold(r.TOTAL_SCH, r.TOTAL_DIFF), axis=1)
            flagged = totals.loc[mask]

            # NEW — robust list‑building (avoids the .tolist() AttributeError)
            log_entries.extend(_format_log_line(row) for _, row in flagged.iterrows())

    # ------------------------------------------------------------------ #
    # 2.  Write the consolidated log file
    # ------------------------------------------------------------------ #
    if WRITE_LOG and ADD_TOTAL_COLUMNS and log_entries:
        log_path = output_dir / LOG_FILENAME
        header = (
            "ROUTE_NUMBER,DIRECTION_NAME,TRIP_KEY,TRIP_START_TIME,TOTAL_SCH,TOTAL_ACT,TOTAL_DIFF\n"
        )
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write(header)
            fh.write("\n".join(log_entries))
        logging.info("✓ Wrote anomaly log: %s", log_path)
    elif WRITE_LOG and ADD_TOTAL_COLUMNS:
        logging.info("✓ No trip exceeded thresholds – nothing written to log.")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Entry‑point when the module is executed as a script."""
    try:
        df = read_and_clean(INPUT_PATH)
    except FileNotFoundError as err:
        sys.exit(f"Input file not found: {err.filename!s}")

    export_tables_and_log(df, METRICS, OUTPUT_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
