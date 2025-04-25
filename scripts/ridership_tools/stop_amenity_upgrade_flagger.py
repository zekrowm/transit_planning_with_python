"""
Flag bus stops that should receive amenity upgrades
 • Reads the ridership workbook you already have.
 • (Optionally) aggregates duplicate STOP_IDs.
 • Flags each stop that meets a ridership threshold AND is missing the amenity.
 • Writes a multi-sheet .xlsx using pandas + openpyxl
"""

# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

import os
import pandas as pd

RIDERSHIP_XLSX  = r"Your\File\Path\To\STOP_USAGE_(BY_STOP_ID).xlsx"
RIDERSHIP_SHEET = 0                           # sheet index or name
OUTPUT_FOLDER   = r"Your\Folder\Path\To\Output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

RIDERSHIP_FIELD = "XBOARDINGS"                # column to test vs. thresholds
ID_FIELDS       = ["STOP_ID", "ROUTE_NAME", "STOP_NAME"]

AMENITIES = {
    "Shelter":  {"field": "SHELTER",  "thresh": 25},
    "Bench":    {"field": "BENCH",    "thresh": 10},
    "TrashCan": {"field": "TRASHCAN", "thresh": 10},
    "Pad":      {"field": "PAD",      "thresh": 1},
}

# "auto"  → aggregate only when STOP_ID duplicates exist
# True    → always aggregate      False → never aggregate
AGGREGATE_BY_STOP = "auto"
STOP_ID_FIELD     = "STOP_ID"

# ==================================================================================================
# FUNCTIONS
# ==================================================================================================

def _standardise_yn(series: pd.Series) -> pd.Series:
    """Return the column capitalised with blanks / NaNs → 'N'."""
    return series.fillna("N").astype(str).str.strip().str.upper()


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per STOP_ID, summing ridership & OR-ing amenities."""
    agg_map = {RIDERSHIP_FIELD: "sum"}
    for cfg in AMENITIES.values():
        col = cfg["field"]
        agg_map[col] = lambda s: "Y" if (s.str.upper() == "Y").any() else "N"
    return df.groupby(STOP_ID_FIELD, as_index=False).agg(agg_map)


# ==================================================================================================
# MAIN
# ==================================================================================================

def main():
    # 1 ▸ Read Excel --------------------------------------------------------
    df_raw = pd.read_excel(RIDERSHIP_XLSX, sheet_name=RIDERSHIP_SHEET, dtype=str)

    # 2 ▸ Ensure ridership column exists -----------------------------------
    if RIDERSHIP_FIELD not in df_raw.columns:
        raise ValueError(f"Column '{RIDERSHIP_FIELD}' not found in workbook.")

    # 3 ▸ Add any missing amenity columns (all 'N') -------------------------
    for cfg in AMENITIES.values():
        col = cfg["field"]
        if col not in df_raw.columns:
            df_raw[col] = "N"
        df_raw[col] = _standardise_yn(df_raw[col])

    # 4 ▸ Convert ridership to numeric -------------------------------------
    df_raw[RIDERSHIP_FIELD] = (
        pd.to_numeric(df_raw[RIDERSHIP_FIELD], errors="coerce").fillna(0)
    )

    # 5 ▸ Decide whether to aggregate --------------------------------------
    need_agg = {
        True:  True,
        False: False,
        "auto": df_raw[STOP_ID_FIELD].duplicated().any()
    }[AGGREGATE_BY_STOP]

    df = _aggregate(df_raw) if need_agg else df_raw.copy()

    # 6 ▸ Build FLAG_ columns ----------------------------------------------
    for name, cfg in AMENITIES.items():
        amen_col = cfg["field"]
        thresh   = cfg["thresh"]
        flag_col = f"FLAG_{name.upper()}"
        df[flag_col] = (
            (df[RIDERSHIP_FIELD] >= thresh) &
            (df[amen_col] != "Y")          # always true if column was invented
        )

    flag_cols = [c for c in df.columns if c.startswith("FLAG_")]
    df["NEEDS_IMPROVEMENT"] = df[flag_cols].any(axis=1)

    # 7 ▸ Write the workbook with openpyxl (default engine) -----------------
    out_xlsx = os.path.join(OUTPUT_FOLDER, "stops_needing_improvement.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:  # ← change here
        df_raw.to_excel(writer, sheet_name="Raw Data",  index=False)
        df.to_excel(writer,     sheet_name="All Flags", index=False)
        for name in AMENITIES:
            flag_col = f"FLAG_{name.upper()}"
            df[df[flag_col]].to_excel(writer, sheet_name=name, index=False)

    # 8 ▸ Summary -----------------------------------------------------------
    print(f"\n✓ Workbook created: {out_xlsx}")
    if need_agg:
        dup_ct = df_raw.shape[0] - df.shape[0]
        print(f"  (Aggregated {dup_ct} duplicate STOP_ID rows.)")


if __name__ == "__main__":
    main()
