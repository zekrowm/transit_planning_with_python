"""
Identify bus stops that merit amenity upgrades based on ridership thresholds.

Reads stop-level ridership and amenity data from Excel, optionally aggregates
duplicate stop IDs, and flags stops that meet usage thresholds but lack
shelters, benches, trash cans, or pads.

The script loads an Excel workbook containing stop-level ridership and amenity
information, normalises and (optionally) aggregates duplicate *STOP_ID*s,
computes boolean “FLAG_*” columns indicating where usage warrants an upgrade
but the amenity is missing, and writes a multi-sheet Excel workbook:

* **Raw Data** – unmodified import
* **All Flags** – every stop plus boolean flag columns
* **Shelter / Bench / TrashCan / Pad** – one sheet per amenity listing only the
  stops that require that specific upgrade

Typical use-cases include batch reviews of bus-stop needs based on ArcGIS
outputs or planning reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

RIDERSHIP_XLSX: Path = Path(r"Your\File\Path\To\STOP_USAGE_(BY_STOP_ID).xlsx")
RIDERSHIP_SHEET: int | str = 0
OUTPUT_FOLDER: Path = Path(r"Your\Folder\Path\To\Output")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

RIDERSHIP_FIELD: str = "XBOARDINGS"
STOP_ID_FIELD: str = "STOP_ID"

AMENITIES: Dict[str, Dict[str, Any]] = {
    "Shelter": {"field": "SHELTER", "thresh": 25},
    "Bench": {"field": "BENCH", "thresh": 10},
    "TrashCan": {"field": "TRASHCAN", "thresh": 10},
    "Pad": {"field": "PAD", "thresh": 1},
}

# True ▸ always aggregate
# False ▸ never aggregate
# "auto" ▸ aggregate only if duplicate STOP_IDs are present
AGGREGATE_BY_STOP: bool | str = "auto"

# =============================================================================
# FUNCTIONS
# =============================================================================


def _standardise_yn(series: pd.Series) -> pd.Series:
    """Normalise a *Y/N* column.

    Args:
        series (pd.Series): A pandas Series that may contain 'Y', 'N', blanks,
            or NaN values in any mixture of case and whitespace.

    Returns:
        pd.Series: The same Series with blanks/NaNs replaced by ``'N'``,
        whitespace trimmed, and all values uppercase.
    """
    return series.fillna("N").astype(str).str.strip().str.upper()


def _load_ridership_data(path: Path, sheet: int | str) -> pd.DataFrame:
    """Load the raw Excel sheet as naïve strings.

    Args:
        path (Path): Path to the XLSX file.
        sheet (int | str): Sheet index or sheet name.

    Returns:
        pd.DataFrame: DataFrame with every column coerced to ``str`` for
        predictable downstream cleanup.
    """
    return pd.read_excel(path, sheet_name=sheet, dtype=str)


def _prepare_amenity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure amenity columns exist and are Y/N-standardised.

    Missing amenity columns are invented and initialised to ``'N'``.

    Args:
        df (pd.DataFrame): Raw ridership DataFrame.

    Returns:
        pd.DataFrame: DataFrame with all amenity columns present and cleaned.
    """
    for cfg in AMENITIES.values():
        col = cfg["field"]
        if col not in df.columns:
            df[col] = "N"
        df[col] = _standardise_yn(df[col])
    return df


def _convert_ridership(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the ridership column to integer.

    Args:
        df (pd.DataFrame): DataFrame containing a column named
            :pydata:`RIDERSHIP_FIELD`.

    Returns:
        pd.DataFrame: The same DataFrame with :pydata:`RIDERSHIP_FIELD`
        cast to ``int``, non-numeric values coerced to zero.
    """
    df[RIDERSHIP_FIELD] = (
        pd.to_numeric(df[RIDERSHIP_FIELD], errors="coerce").fillna(0).astype(int)
    )
    return df


def _needs_aggregation(df: pd.DataFrame) -> bool:
    """Determine whether STOP_ID-level aggregation is required.

    Args:
        df (pd.DataFrame): The current DataFrame.

    Returns:
        bool: ``True`` if aggregation should be performed, ``False`` otherwise.
    """
    table = {
        True: True,
        False: False,
        "auto": df[STOP_ID_FIELD].duplicated().any(),
    }
    return table[AGGREGATE_BY_STOP]


def _aggregate_by_stop(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per STOP_ID.

    Ridership is summed; amenity columns are logically OR-ed (any 'Y' ⇒ 'Y').

    Args:
        df (pd.DataFrame): DataFrame containing duplicate *STOP_ID* rows.

    Returns:
        pd.DataFrame: Aggregated DataFrame with unique *STOP_ID*s.
    """
    agg_map: Dict[str, Any] = {RIDERSHIP_FIELD: "sum"}  # type: ignore[arg-type]
    for cfg in AMENITIES.values():
        col = cfg["field"]
        agg_map[col] = lambda s: "Y" if (s.str.upper() == "Y").any() else "N"  # type: ignore[var-annotated]
    return df.groupby(STOP_ID_FIELD, as_index=False).agg(agg_map)


def _compute_flags(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Add ``FLAG_*`` columns and a summary column.

    Args:
        df (pd.DataFrame): DataFrame after aggregation (or not).

    Returns:
        Tuple[pd.DataFrame, List[str]]: *(modified_df, list_of_flag_columns)*.
    """
    for name, cfg in AMENITIES.items():
        amen_col = cfg["field"]
        thresh = cfg["thresh"]
        flag_col = f"FLAG_{name.upper()}"
        df[flag_col] = (df[RIDERSHIP_FIELD] >= thresh) & (df[amen_col] != "Y")

    flag_cols = [c for c in df.columns if c.startswith("FLAG_")]
    df["NEEDS_IMPROVEMENT"] = df[flag_cols].any(axis=1)
    return df, flag_cols


def _write_workbook(
    raw_df: pd.DataFrame, processed_df: pd.DataFrame, out_path: Path
) -> None:
    """Write the results workbook with multiple sheets.

    Args:
        raw_df (pd.DataFrame): Unmodified import DataFrame.
        processed_df (pd.DataFrame): DataFrame with flag columns added.
        out_path (Path): Destination XLSX path.
    """
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        raw_df.to_excel(writer, sheet_name="Raw Data", index=False)
        processed_df.to_excel(writer, sheet_name="All Flags", index=False)
        for name in AMENITIES:
            flag_col = f"FLAG_{name.upper()}"
            processed_df[processed_df[flag_col]].to_excel(
                writer, sheet_name=name, index=False
            )


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Run the full ETL pipeline to flag bus stops for amenity upgrades.

    Workflow:

    1. Load ridership workbook.
    2. Verify required ridership column exists.
    3. Normalise amenity columns.
    4. Convert ridership to integer.
    5. Optionally aggregate duplicate *STOP_ID*s.
    6. Compute ``FLAG_*`` columns and ``NEEDS_IMPROVEMENT``.
    7. Write a multi-sheet Excel workbook summarising results.
    8. Print a concise console summary.

    Raises:
        ValueError: If :pydata:`RIDERSHIP_FIELD` is missing from the input file.
    """
    df_raw = _load_ridership_data(RIDERSHIP_XLSX, RIDERSHIP_SHEET)

    if RIDERSHIP_FIELD not in df_raw.columns:
        raise ValueError(f"Column '{RIDERSHIP_FIELD}' not found in workbook.")

    df_raw = _prepare_amenity_columns(df_raw)
    df_raw = _convert_ridership(df_raw)

    need_agg = _needs_aggregation(df_raw)
    df_processed = _aggregate_by_stop(df_raw) if need_agg else df_raw.copy()

    df_processed, _ = _compute_flags(df_processed)

    out_xlsx = OUTPUT_FOLDER / "stops_needing_improvement.xlsx"
    _write_workbook(df_raw, df_processed, out_xlsx)

    # Console summary
    print(f"\n✓ Workbook created: {out_xlsx}")
    if need_agg:
        dup_ct = df_raw.shape[0] - df_processed.shape[0]
        print(f"  (Aggregated {dup_ct} duplicate STOP_ID rows.)")


if __name__ == "__main__":  # pragma: no cover
    main()
