"""Processes stop-level ridership data from an Excel file.

Reads an input Excel file (RIDERSHIP_BY_ROUTE_AND_STOP_(ALL_TIME_PERIODS).XLSX),
optionally filters by route or stop ID, aggregates boardings and alightings by
stop and time period, and saves the results to a new Excel file. Aggregated data
can optionally be rounded or categorized into bins.

The script is designed for analysts and data scientists who need a quick and
repeatable tool for ad-hoc stop ridership data requests, and it is suitable
for use in environments like ArcGIS Pro or Jupyter Notebooks.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE_PATH: Path = Path(r"\\Path\To\Your\RIDERSHIP_BY_ROUTE_AND_STOP_(ALL_TIME_PERIODS).XLSX")
OUTPUT_FILE_SUFFIX: str = "_processed"
OUTPUT_FILE_EXTENSION: str = ".xlsx"
# If OUTPUT_DIR is None ⇒ use same directory as INPUT_FILE_PATH
OUTPUT_DIR: Path | None = Path(r"\\Path\\To\\Output\\Folder")  # e.g. r"C:\Data\Outputs"

# ROUTES = keep-only list   |  ROUTES_EXCLUDE = toss-out list
ROUTES: List[str] = []  # keep these (empty → keep all)
ROUTES_EXCLUDE: List[str] = []  # drop these (empty → drop none)

# If True → aggregate across all selected routes at the same stop (adds a "ROUTES" column).
# If False → keep routes separate in aggregations (adds/retains a "ROUTE_NAME" column).
AGGREGATE_ROUTES_TOGETHER: bool = True

# Optional STOP_IDS filter list
STOP_IDS: List[int] = []  # keep these (empty → keep all)

# Optional TIME_PERIOD aggregation list
TIME_PERIODS: List[str] = [
    #    "AM EARLY",
    "AM PEAK",
    #    "MIDDAY",
    "PM PEAK",
    #    "PM LATE",
    #    "PM NITE",
]  # empty ⇒ skip time-period breakdown

# If True → round ridership columns in "Original" sheet and (if bins are off)
# round aggregated totals.  If False → leave raw numeric precision.
APPLY_ROUNDING: bool = True

# If True → convert aggregated totals to textual bins; numeric rounding is
# suppressed.  If False → leave numeric (subject to APPLY_ROUNDING).
AGGREGATE_BIN_RANGES: bool = False

REQUIRED_COLUMNS: Sequence[str] = (
    "TIME_PERIOD",
    "ROUTE_NAME",
    "STOP",
    "STOP_ID",
    "BOARD_ALL",
    "ALIGHT_ALL",
)
COLUMNS_TO_RETAIN: Sequence[str] = (
    "ROUTE_NAME",
    "STOP",
    "STOP_ID",
    "BOARD_ALL",
    "ALIGHT_ALL",
)

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# =============================================================================
# FUNCTIONS
# =============================================================================


def bin_ridership_value(value: float) -> str:
    """Convert a numeric ridership value into a categorical range.

    Args:
        value: The boarding/alighting count for a stop.

    Returns:
        A string bucket—``"0-4.9"``, ``"5-24.9"``, or ``"25 or more"``.
    """
    if value < 5:
        return "0-4.9"
    if value < 25:
        return "5-24.9"
    return "25 or more"


def aggregate_by_stop(
    data_subset: pd.DataFrame,
    *,
    aggregate_routes_together: bool,
) -> pd.DataFrame:
    """Aggregate boardings/alightings by stop, with optional cross-route rollup.

    When ``aggregate_routes_together`` is True, rows for different routes serving the
    same stop are summed together and a ``ROUTES`` column lists the unique routes.
    When False, routes are *not* combined; the output is keyed by ``ROUTE_NAME``,
    ``STOP``, and ``STOP_ID``. In that case, ``ROUTE_NAME`` is a regular column.

    Args:
        data_subset: DataFrame already filtered to rows of interest.
        aggregate_routes_together: If True, aggregate across routes at a stop.

    Returns:
        Aggregated DataFrame with ``BOARD_ALL_TOTAL`` and ``ALIGHT_ALL_TOTAL`` plus:
          * If together:  one row per (STOP, STOP_ID) and a ``ROUTES`` column.
          * If separate:  one row per (ROUTE_NAME, STOP, STOP_ID).
    """
    cols_needed: List[str] = [
        "STOP",
        "STOP_ID",
        "BOARD_ALL",
        "ALIGHT_ALL",
        "ROUTE_NAME",
    ]
    subset: pd.DataFrame = data_subset[cols_needed].copy()
    subset["ROUTE_NAME"] = subset["ROUTE_NAME"].astype(str).str.strip()

    if aggregate_routes_together:
        grouping_cols: List[str] = ["STOP", "STOP_ID"]
        aggregated: pd.DataFrame = (
            subset.groupby(grouping_cols, as_index=False)
            .agg(
                {
                    "BOARD_ALL": "sum",
                    "ALIGHT_ALL": "sum",
                    "ROUTE_NAME": lambda x: ", ".join(sorted(pd.Series(x).dropna().unique())),
                }
            )
            .rename(
                columns={
                    "BOARD_ALL": "BOARD_ALL_TOTAL",
                    "ALIGHT_ALL": "ALIGHT_ALL_TOTAL",
                    "ROUTE_NAME": "ROUTES",
                }
            )
        )
    else:
        # Keep routes separate in the aggregates.
        grouping_cols = ["ROUTE_NAME", "STOP", "STOP_ID"]
        aggregated = (
            subset.groupby(grouping_cols, as_index=False)
            .agg({"BOARD_ALL": "sum", "ALIGHT_ALL": "sum"})
            .rename(columns={"BOARD_ALL": "BOARD_ALL_TOTAL", "ALIGHT_ALL": "ALIGHT_ALL_TOTAL"})
        )

    return aggregated


def log_missing_stop_ids(
    requested_ids: Sequence[int], present_ids: pd.Series | Sequence[int]
) -> None:
    """Warn when any requested *STOP_ID* is absent from *present_ids*.

    Args:
        requested_ids: The ``STOP_IDS`` list provided in CONFIGURATION.
        present_ids:  A 1-D iterable (e.g. ``DataFrame['STOP_ID']``) of IDs that
            survived the filter chain.

    Raises:
        TypeError: If *present_ids* cannot be coerced to a set of ``int``.
    """
    if not requested_ids:  # nothing requested ⇒ nothing to warn about
        return

    try:
        missing: set[int] = set(requested_ids) - {int(x) for x in present_ids}
    except (ValueError, TypeError) as exc:
        raise TypeError(
            "Unable to evaluate present_ids when checking for missing STOP_IDs."
        ) from exc

    if missing:
        logging.warning(
            "The following requested STOP_IDs were not found in the processed "
            "dataset and therefore will not appear in the output: %s",
            sorted(missing),
        )
    else:
        logging.info("All requested STOP_IDs are present in the processed data.")


def read_excel_file(input_file: Path) -> pd.DataFrame:
    """Load an Excel workbook into a DataFrame.

    Args:
        input_file: Absolute or relative path to the workbook.

    Returns:
        The first sheet of *input_file* as a ``pandas.DataFrame``.

    Raises:
        SystemExit: If the file does not exist or cannot be parsed.
    """
    try:
        return pd.read_excel(input_file)
    except FileNotFoundError:
        logger.error("The file '%s' does not exist.", input_file)
        sys.exit(1)
    except ValueError as exc:  # pandas re-raises most Excel errors as ValueError
        logger.error("Error reading the Excel file: %s", exc)
        sys.exit(1)


def verify_required_columns(data_frame: pd.DataFrame, required_columns: Sequence[str]) -> None:
    """Ensure *data_frame* contains all columns listed in *required_columns*.

    Args:
        data_frame: The DataFrame to validate.
        required_columns: Names that must be present.

    Raises:
        SystemExit: If any required column is missing.
    """
    missing_columns: List[str] = [col for col in required_columns if col not in data_frame.columns]
    if missing_columns:
        logger.error("Missing columns: %s", missing_columns)
        sys.exit(1)


def filter_data(
    data_frame: pd.DataFrame,
    routes: Sequence[str] | None = None,
    stop_ids: Sequence[int] | None = None,
    routes_exclude: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Apply optional filters in a deterministic order.

    Filters are applied in-place (on a copy) with this precedence:

    1. Inclusive   ``routes``        → keep-only rows where ROUTE_NAME ∈ routes
    2. Exclusive   ``routes_exclude``→ drop-only  rows where ROUTE_NAME ∈ routes_exclude
    3. Inclusive   ``stop_ids``      → keep-only rows where STOP_ID   ∈ stop_ids

    Empty/None arguments are ignored.

    Args:
        data_frame: Original DataFrame.
        routes: Route names to *keep* (inclusive).
        stop_ids: Stop IDs to *keep* (inclusive).
        routes_exclude: Route names to *drop* (exclusive).

    Returns:
        A filtered DataFrame.
    """
    df: pd.DataFrame = data_frame.copy()

    if routes:
        df = df[df["ROUTE_NAME"].isin(routes)]

    if routes_exclude:
        df = df[~df["ROUTE_NAME"].isin(routes_exclude)]

    if stop_ids:
        df = df[df["STOP_ID"].isin(stop_ids)]

    return df


def write_to_excel(
    output_file: Path,
    filtered_data: pd.DataDataFrame,
    aggregated_peaks: Dict[str, pd.DataFrame],
    all_time_aggregated: pd.DataFrame,
) -> None:
    """Write the processed data sets to *output_file* in a sensible order.

    The workbook receives:
        1. ``Original``           – raw (but optionally rounded) rows
        2. ``All Time Periods``   – aggregation across *all* rows
        3. One sheet per entry in ``aggregated_peaks`` (already upper-cased)

    Args:
        output_file: Path to the workbook to create/overwrite.
        filtered_data: DataFrame for the "Original" sheet.
        aggregated_peaks: Mapping ``TIME_PERIOD → aggregated DataFrame``.
        all_time_aggregated: Aggregation across *all* rows.

    Raises:
        SystemExit: On any I/O error (permission, disk full, etc.).
    """
    try:
        with pd.ExcelWriter(output_file) as writer:
            filtered_data.to_excel(writer, sheet_name="Original", index=False)
            all_time_aggregated.to_excel(writer, sheet_name="All Time Periods", index=False)
            for period, df_agg in aggregated_peaks.items():
                df_agg.to_excel(writer, sheet_name=period, index=False)

        adjust_excel_formatting(output_file)
        logger.info("The processed file has been saved as '%s'.", output_file)
    except (OSError, PermissionError) as exc:
        logger.error("Error writing the processed Excel file: %s", exc)
        sys.exit(1)


def adjust_excel_formatting(output_file: Path) -> None:
    """Bold the header row and auto-size columns for *output_file*.

    Args:
        output_file: Path to an existing XLSX workbook.

    Raises:
        SystemExit: If the workbook cannot be opened or saved.
    """
    try:
        workbook = load_workbook(output_file)
        for sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]

            # Bold the header row
            for cell in sheet[1]:
                cell.font = Font(bold=True)

            # Auto-size column widths
            for col_idx, column_cells in enumerate(sheet.columns, start=1):
                max_length: int = 0
                col_letter: str = get_column_letter(col_idx)
                for cell in column_cells:
                    cell_val = str(cell.value) if cell.value is not None else ""
                    max_length = max(max_length, len(cell_val))
                sheet.column_dimensions[col_letter].width = max_length + 2

        workbook.save(output_file)
    except (OSError, PermissionError) as exc:
        logger.error("Error adjusting Excel formatting: %s", exc)
        sys.exit(1)


def process_aggregations(
    filtered_data: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], pd.DataFrame]:
    """Orchestrate rounding/bucketing and aggregation for output sheets.

    This function:
      1) Standardizes ``ROUTE_NAME`` as ``str``.
      2) Builds per-period slices if ``TIME_PERIODS`` is set.
      3) Aggregates for "All Time Periods" and each time-period slice.
         The cross-route behavior is controlled by
         ``AGGREGATE_ROUTES_TOGETHER``:
           * True  → one row per (STOP, STOP_ID) with ``ROUTES`` listing routes.
           * False → one row per (ROUTE_NAME, STOP, STOP_ID).
      4) Optionally rounds the "Original" sheet numeric columns.
      5) Applies binning or rounding to the aggregated totals.

    Args:
        filtered_data: The DataFrame *after* optional route/stop filtering.
            Must contain at least the columns listed in ``REQUIRED_COLUMNS``.

    Returns:
        A 3-tuple:
            * Final version of ``filtered_data`` (possibly rounded for "Original").
            * Mapping ``TIME_PERIOD → aggregated DataFrame`` (sheet-ready).
            * Aggregation across *all* rows and periods.
    """
    # ──────────────────────────────────────────────────────────────────
    # 0. Standardize ROUTE_NAME to string before any aggregation
    # ──────────────────────────────────────────────────────────────────
    filtered_data["ROUTE_NAME"] = filtered_data["ROUTE_NAME"].astype(str).str.strip()

    # ──────────────────────────────────────────────────────────────────
    # 1. Build per-period subsets (if any)
    # ──────────────────────────────────────────────────────────────────
    peak_data_dict: Dict[str, pd.DataFrame] = {}
    if TIME_PERIODS:
        for period in TIME_PERIODS:
            subset = filtered_data[filtered_data["TIME_PERIOD"] == period.upper()]
            # Keep only the columns needed for the "Original" per-period sheet seeds
            peak_data_dict[period] = subset[list(COLUMNS_TO_RETAIN)]

    # ──────────────────────────────────────────────────────────────────
    # 2. Aggregate (all-time + each slice); route roll-up is configurable
    # ──────────────────────────────────────────────────────────────────
    all_time_aggregated: pd.DataFrame = aggregate_by_stop(
        filtered_data, aggregate_routes_together=AGGREGATE_ROUTES_TOGETHER
    )

    aggregated_peaks: Dict[str, pd.DataFrame] = {}
    if TIME_PERIODS:
        for period, data_subset in peak_data_dict.items():
            aggregated_peaks[period] = aggregate_by_stop(
                data_subset, aggregate_routes_together=AGGREGATE_ROUTES_TOGETHER
            )

    # ──────────────────────────────────────────────────────────────────
    # 3. Optional rounding for "Original" sheet
    # ──────────────────────────────────────────────────────────────────
    if APPLY_ROUNDING:
        filtered_data[["BOARD_ALL", "ALIGHT_ALL"]] = filtered_data[
            ["BOARD_ALL", "ALIGHT_ALL"]
        ].round(1)

    # ──────────────────────────────────────────────────────────────────
    # 4. Format aggregated columns (bins OR decimal rounding)
    # ──────────────────────────────────────────────────────────────────
    all_aggregations: List[pd.DataFrame] = [all_time_aggregated] + list(aggregated_peaks.values())
    for df_agg in all_aggregations:
        if AGGREGATE_BIN_RANGES:
            for col in ("BOARD_ALL_TOTAL", "ALIGHT_ALL_TOTAL"):
                df_agg[col] = df_agg[col].apply(bin_ridership_value)
        elif APPLY_ROUNDING:
            df_agg[["BOARD_ALL_TOTAL", "ALIGHT_ALL_TOTAL"]] = df_agg[
                ["BOARD_ALL_TOTAL", "ALIGHT_ALL_TOTAL"]
            ].round(1)

    return filtered_data, aggregated_peaks, all_time_aggregated


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:  # noqa: D401 – imperative mood is OK for main entry point
    """Run the full read → filter → aggregate → write pipeline."""
    input_file: Path = INPUT_FILE_PATH

    # Build output file path
    base_name: str = input_file.stem
    output_fname: str = f"{base_name}{OUTPUT_FILE_SUFFIX}{OUTPUT_FILE_EXTENSION}"
    if OUTPUT_DIR:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_file: Path = OUTPUT_DIR / output_fname
    else:
        output_file = input_file.parent / output_fname

    # Read and validate
    ridership_df: pd.DataFrame = read_excel_file(input_file)
    verify_required_columns(ridership_df, REQUIRED_COLUMNS)

    # Apply optional filters
    filtered_data: pd.DataFrame = filter_data(
        ridership_df,
        routes=ROUTES,
        stop_ids=STOP_IDS,
        routes_exclude=ROUTES_EXCLUDE,
    )

    # Log missing optional STOP_IDS
    log_missing_stop_ids(STOP_IDS, filtered_data["STOP_ID"])

    # Standardise TIME_PERIOD values
    filtered_data["TIME_PERIOD"] = filtered_data["TIME_PERIOD"].astype(str).str.strip().str.upper()

    # Aggregate + format
    final_filtered, aggregated_peaks, all_time_aggregated = process_aggregations(filtered_data)

    # Write to disk
    write_to_excel(
        output_file,
        final_filtered,
        aggregated_peaks,
        all_time_aggregated,
    )


if __name__ == "__main__":
    main()
