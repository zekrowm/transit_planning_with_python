"""Merge and (optionally) FIPS-filter Census block shapefiles using ArcPy.

The script:

1.  Loads a list of TIGER Census Block, Block Group, Tract (and other) shapefiles.
2.  Merges them to a single temporary feature class.
3.  If a FIPS list is provided:
      - Adds a new 'FIPS' text field (STATEFP20 + COUNTYFP20).
      - Selects only the requested municipalities.
4.  Writes the final geometry (plus all original attributes) to the
   user-specified output feature class or shapefile.

Run inside an ArcGIS Pro (or ArcMap 10.8) Python environment where `arcpy`
is available.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import List, Sequence

import arcpy

# =============================================================================
# CONFIGURATION
# =============================================================================

# Full paths to TABBLOCK20 shapefiles (block level)
BLOCK_SHP_FILES: List[str] = [
    r"C:\full\path\to\tl_2023_11_tabblock20.shp",
    r"C:\full\path\to\tl_2023_24_tabblock20.shp",
    r"C:\full\path\to\tl_2023_51_tabblock20.shp",
]

# Optional: Leave empty ([]) to skip filtering
# FIPS codes include 2-digit state code, 3-digit county/city/other local code
FIPS_TO_FILTER: List[str] = [
    "11001",
    "24031",
    "24033",
    "51683",
    "51685",
    "51059",
    "51013",
    "51510",
    "51600",
    "51610",
    "51107",
    "51153",
]

# Shapefile (*.shp) or feature class inside a file geodatabase
OUTPUT_PATH: str = r"Path\To\Your\Output_Folder\va_md_dc_blocks_fips_merge.shp"

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

arcpy.env.overwriteOutput = True
arcpy.SetLogHistory(False)  # Prevent clutter in the project history

# =============================================================================
# FUNCTIONS
# =============================================================================


def merge_shapefiles(shp_paths: Sequence[str], out_fc: str) -> str:
    """Merge multiple shapefiles into a single feature class.

    Args:
        shp_paths: Iterable of input shapefile paths.
        out_fc:     Output feature class (can be in_memory FC).

    Returns:
        Path to the merged feature class.
    """
    logging.info("Merging %d shapefile(s)…", len(shp_paths))
    arcpy.management.Merge(list(shp_paths), out_fc)
    logging.info("Merge complete → %s", out_fc)
    return out_fc


def ensure_fips_field(
    fc: str,
    fips_field: str = "FIPS",
    state_candidates: tuple[str, ...] = ("STATEFP20", "STATEFP"),
    county_candidates: tuple[str, ...] = ("COUNTYFP20", "COUNTYFP"),
) -> None:
    """Make sure <fips_field> exists and fill it with 5-digit county FIPS codes.

    *Works even when some blocks have nulls – those FIPS stay NULL.*
    """
    fld_lookup = {f.name.upper(): f.name for f in arcpy.ListFields(fc)}
    state = next(
        (fld_lookup[c.upper()] for c in state_candidates if c.upper() in fld_lookup),
        None,
    )
    county = next(
        (fld_lookup[c.upper()] for c in county_candidates if c.upper() in fld_lookup),
        None,
    )

    if not state or not county:
        raise RuntimeError("STATE / COUNTY fields not found. Check your shapefiles.")

    if fips_field not in fld_lookup:
        logging.info("Adding %s field…", fips_field)
        arcpy.management.AddField(fc, fips_field, "TEXT", field_length=5)

    logging.info("Populating %s…", fips_field)
    with arcpy.da.UpdateCursor(fc, [state, county, fips_field]) as cur:
        for row in cur:
            st, co = row[0], row[1]
            row[2] = f"{str(st).zfill(2)}{str(co).zfill(3)}" if st and co else None
            cur.updateRow(row)


def filter_by_fips(
    in_fc: str,
    out_fc: str,
    fips_values: Sequence[str],
    fips_field: str = "FIPS",
) -> str:
    """Copy only the requested counties (or everything if list is empty)."""
    if not fips_values:
        logging.info("FIPS filter empty – exporting full dataset.")
        arcpy.management.CopyFeatures(in_fc, out_fc)
        return out_fc

    sql = f"{fips_field} IN ({', '.join(map(repr, fips_values))})"
    logging.info("Applying filter: %s", sql)

    lyr = "fips_select_lyr"
    arcpy.management.MakeFeatureLayer(in_fc, lyr, sql)
    if int(arcpy.management.GetCount(lyr)[0]) == 0:
        logging.warning("No features matched the FIPS list – output will be empty.")

    arcpy.management.CopyFeatures(lyr, out_fc)
    arcpy.management.Delete(lyr)
    return out_fc


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Script entry point."""
    try:
        # ---------------------------------------------------------------------
        # 1. Merge shapefiles
        # ---------------------------------------------------------------------
        merged_fc = merge_shapefiles(
            BLOCK_SHP_FILES,
            out_fc=r"in_memory\merged_blocks",
        )

        # ---------------------------------------------------------------------
        # 2. Add and populate FIPS (needed for filter OR later data joins)
        # ---------------------------------------------------------------------
        ensure_fips_field(merged_fc)

        # ---------------------------------------------------------------------
        # 3. Filter (if requested) and export
        # ---------------------------------------------------------------------
        output_folder = os.path.dirname(OUTPUT_PATH)
        if output_folder and not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        final_fc = filter_by_fips(
            merged_fc,
            out_fc=OUTPUT_PATH,
            fips_values=FIPS_TO_FILTER,
        )

        logging.info("Finished – output written to: %s", final_fc)

    except arcpy.ExecuteError:
        logging.error(arcpy.GetMessages(2))
        sys.exit(1)
    except Exception:
        logging.exception("Unexpected error:")
        sys.exit(1)


if __name__ == "__main__":
    main()
