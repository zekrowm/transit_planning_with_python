"""Flag stops needing amenity upgrades using ridership thresholds and amenity data.

This script reads stop-level ridership and amenity information from one or
(optionally) two Excel workbooks. It normalizes and (optionally) aggregates
duplicate STOP_IDs, then computes boolean "FLAG_*" columns to highlight where
usage warrants an upgrade but the amenity is missing.

It outputs a multi-sheet Excel workbook and a plain-text log file:

    stops_needing_improvement.xlsx:
        Raw Data – Unmodified import.
        All Flags – Every stop plus boolean flag columns for each amenity and a
        NEEDS_IMPROVEMENT summary flag.
        Shelter / Bench / TrashCan / Pad – One sheet per amenity, listing only
        the stops that require that specific upgrade.
    stops_needing_improvement.txt: A concise summary of flagged stops by
    category, followed by a detailed, human-readable list of all stops
    identified for improvement.

Typical use-cases include batch reviews of bus-stop needs based on ArcGIS
outputs, planning reports, or independently maintained amenity inventories.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# Ridership source workbook
RIDERSHIP_XLSX: Path = Path(r"Your\File\Path\To\STOP_USAGE_(BY_STOP_ID).xlsx")
RIDERSHIP_SHEET: int | str = 0
# Output folder (Excel + TXT will be written here)
OUTPUT_FOLDER: Path = Path(r"Your\Folder\Path\To\Output")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Fields in ridership workbook
RIDERSHIP_FIELD: str = "XBOARDINGS"
STOP_ID_FIELD: str = "STOP_ID"

# Amenity thresholds and fields (must match column names after standardisation)
AMENITIES: Dict[str, Dict[str, Any]] = {
    "Shelter": {"field": "SHELTER", "thresh": 25},
    "Bench": {"field": "BENCH", "thresh": 10},
    "TrashCan": {"field": "TRASHCAN", "thresh": 10},
    "Pad": {"field": "PAD", "thresh": 1},
}

# Aggregation behaviour: True | False | "auto"
AGGREGATE_BY_STOP: bool | str = "auto"

# -----------------------------------------------------------------------------
# OPTIONAL SECOND WORKBOOK – AMENITY DETAILS
# -----------------------------------------------------------------------------

AMENITIES_XLSX: Path = Path(r"Your\File\Path\To\bus_stop_amenities.xlsx")
AMENITIES_SHEET: int | str = 0
AMENITY_JOIN_FIELD: str = "stop_code"
TXT_LOG_PATH: Path = OUTPUT_FOLDER / "stops_needing_improvement.txt"

# -----------------------------------------------------------------------------
# AMENITY FIELD MAPPINGS
# -----------------------------------------------------------------------------

_AMENITY_ALIASES: Dict[str, str] = {
    "bus_shelte": "SHELTER",
    "bus_shelter": "SHELTER",
    "pad": "PAD",
    "bench": "BENCH",
    "trash_can": "TRASHCAN",
    "trashcan": "TRASHCAN",
}

# =============================================================================
# FUNCTIONS
# =============================================================================


def _standardise_yn(series: pd.Series) -> pd.Series:
    """Normalise a Y/N column to uppercase 'Y' or 'N' with no whitespace."""
    return series.fillna("N").astype(str).str.strip().str.upper()


def _load_ridership_data(path: Path, sheet: int | str) -> pd.DataFrame:
    """Load ridership workbook, coercing all columns to strings."""
    return pd.read_excel(path, sheet_name=sheet, dtype=str)


def _prepare_amenity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure every expected amenity column exists and is Y/N-standardised."""
    for cfg in AMENITIES.values():
        col = cfg["field"]
        if col not in df.columns:
            df[col] = "N"
        df[col] = _standardise_yn(df[col])
    return df


def _convert_ridership(df: pd.DataFrame) -> pd.DataFrame:
    """Cast the ridership column to int, coercing non-numerics to zero."""
    df[RIDERSHIP_FIELD] = pd.to_numeric(df[RIDERSHIP_FIELD], errors="coerce").fillna(0).astype(int)
    return df


def _needs_aggregation(df: pd.DataFrame) -> bool:
    """Decide if STOP_ID-level aggregation should be performed."""
    decision_map = {
        True: True,
        False: False,
        "auto": df[STOP_ID_FIELD].duplicated().any(),
    }
    return decision_map[AGGREGATE_BY_STOP]


def _aggregate_by_stop(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate duplicate STOP_ID rows: sum ridership, OR amenities."""
    agg_map: Dict[str, Any] = {RIDERSHIP_FIELD: "sum"}
    for cfg in AMENITIES.values():
        col = cfg["field"]
        agg_map[col] = lambda s: "Y" if (s.str.upper() == "Y").any() else "N"
    return df.groupby(STOP_ID_FIELD, as_index=False).agg(agg_map)


def _compute_flags(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Add FLAG_* columns and a summary NEEDS_IMPROVEMENT column."""
    for name, cfg in AMENITIES.items():
        col = cfg["field"]
        thresh = cfg["thresh"]
        flag_col = f"FLAG_{name.upper()}"
        df[flag_col] = (df[RIDERSHIP_FIELD] >= thresh) & (df[col] != "Y")
    flag_cols = [c for c in df.columns if c.startswith("FLAG_")]
    df["NEEDS_IMPROVEMENT"] = df[flag_cols].any(axis=1)
    return df, flag_cols


def _write_workbook(raw_df: pd.DataFrame, processed_df: pd.DataFrame, out_path: Path) -> None:
    """Write multi-sheet Excel: Raw Data, All Flags, plus one per amenity."""
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        raw_df.to_excel(writer, sheet_name="Raw Data", index=False)
        processed_df.to_excel(writer, sheet_name="All Flags", index=False)
        for name in AMENITIES:
            flag_col = f"FLAG_{name.upper()}"
            processed_df[processed_df[flag_col]].to_excel(writer, sheet_name=name, index=False)


def _load_amenity_data(path: Path, sheet: int | str) -> pd.DataFrame:
    """Read and sanitise the separate amenities workbook."""
    df = pd.read_excel(path, sheet_name=sheet, dtype=str)
    # Normalise column names and apply known aliases
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={k: v for k, v in _AMENITY_ALIASES.items() if k in df.columns})
    # Standardise any amenity columns present
    for cfg in AMENITIES.values():
        col = cfg["field"]
        if col in df.columns:
            df[col] = _standardise_yn(df[col])
    return df


def _merge_ridership_and_amenities(rider_df: pd.DataFrame, amen_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join amenity info onto ridership on STOP_ID_FIELD ↔ AMENITY_JOIN_FIELD."""
    rider_df[STOP_ID_FIELD] = rider_df[STOP_ID_FIELD].astype(str).str.strip()
    amen_df[AMENITY_JOIN_FIELD] = amen_df[AMENITY_JOIN_FIELD].astype(str).str.strip()

    keep = [AMENITY_JOIN_FIELD] + [cfg["field"] for cfg in AMENITIES.values()]
    amen_subset = amen_df[keep]

    merged = rider_df.merge(
        amen_subset,
        how="left",
        left_on=STOP_ID_FIELD,
        right_on=AMENITY_JOIN_FIELD,
        suffixes=("", "_amen"),
    )

    # Coalesce: prefer explicit amenity file, fall back to ridership data
    for cfg in AMENITIES.values():
        col = cfg["field"]
        alt = f"{col}_amen"
        if alt in merged.columns:
            merged[col] = merged[col].fillna(merged[alt])
            merged.drop(columns=[alt], inplace=True)
    merged.drop(columns=[AMENITY_JOIN_FIELD], errors="ignore", inplace=True)
    return merged


def _write_txt_log(processed_df: pd.DataFrame, flag_cols: List[str], out_path: Path) -> None:
    """Write a plain-text summary of flagged stops and counts."""
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"Run date: {pd.Timestamp.now():%Y-%m-%d %H:%M}\n\n")
        f.write("Stops needing improvement by category\n")
        f.write("──────────────────────────────────────\n")
        for name in AMENITIES:
            col = f"FLAG_{name.upper()}"
            f.write(f"{name:10s}: {processed_df[col].sum():>6}\n")
        f.write(f"\nTotal flagged stops: {processed_df['NEEDS_IMPROVEMENT'].sum()}\n\n")

        f.write("Detailed list (one row per stop)\n")
        f.write("─────────────────────────────────\n")
        cols = (
            [STOP_ID_FIELD, RIDERSHIP_FIELD]
            + [cfg["field"] for cfg in AMENITIES.values()]
            + flag_cols
        )
        f.write(processed_df[processed_df["NEEDS_IMPROVEMENT"]][cols].to_string(index=False))


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Run the ETL pipeline and produce both Excel and text outputs."""
    # 1–2. LOAD SOURCE FILES
    df_raw = _load_ridership_data(RIDERSHIP_XLSX, RIDERSHIP_SHEET)
    df_amen = _load_amenity_data(AMENITIES_XLSX, AMENITIES_SHEET)

    # 3. MERGE + VALIDATE
    df_raw = _merge_ridership_and_amenities(df_raw, df_amen)
    if RIDERSHIP_FIELD not in df_raw.columns:
        raise ValueError(f"Column '{RIDERSHIP_FIELD}' not found in workbook.")

    # 4–5. CLEAN & AGGREGATE
    df_raw = _prepare_amenity_columns(df_raw)
    df_raw = _convert_ridership(df_raw)

    need_agg = _needs_aggregation(df_raw)
    df_processed = _aggregate_by_stop(df_raw) if need_agg else df_raw.copy()

    # 6. COMPUTE FLAGS
    df_processed, flag_cols = _compute_flags(df_processed)

    # 7. WRITE OUTPUTS
    out_xlsx = OUTPUT_FOLDER / "stops_needing_improvement.xlsx"
    _write_workbook(df_raw, df_processed, out_xlsx)
    _write_txt_log(df_processed, flag_cols, TXT_LOG_PATH)

    # 8. CONSOLE SUMMARY
    print(f"\n✓ Workbook created: {out_xlsx}")
    print(f"✓ Text log created: {TXT_LOG_PATH}")
    if need_agg:
        dup_ct = df_raw.shape[0] - df_processed.shape[0]
        print(f"  (Aggregated {dup_ct} duplicate STOP_ID rows.)")


if __name__ == "__main__":
    main()
