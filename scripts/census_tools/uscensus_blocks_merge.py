"""Merge and (optionally) FIPS-filter TIGER/Line shapefiles with ArcPy.

Workflow:
    1. Recursively find TIGER/Line shapefiles under INPUT_DIR that match INPUT_GLOB.
    2. Merge them into a single temporary feature class.
    3. Ensure a 5-character county FIPS field exists (default: 'FIPS').
       This is built from either:
           (STATEFP20 or STATEFP) + (COUNTYFP20 or COUNTYFP).
    4. If FIPS_TO_FILTER is non-empty, select and export only those counties.
    5. Write the result to OUTPUT_PATH (FGDB feature class or shapefile).

Notes/limitations compared to the GeoPandas version:
    - Zipped shapefiles (tl_XXXX_YY_tabblock20.zip) are NOT opened directly.
      You must unzip them or point INPUT_DIR at a folder of *.shp files.
    - All inputs should be in the same coordinate system.
    - Field name length rules will follow the target workspace (FGDB is fine).
"""

from __future__ import annotations

import fnmatch
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import arcpy

# =============================================================================
# CONFIGURATION
# =============================================================================

# Root folder that contains one or more TIGER/Line shapefiles.
INPUT_DIR: str = r"C:\path\to\your\tiger_shapefiles"

# Unix-style glob that must match the **.shp** filenames you want.
# e.g. "tl_2023_11_tabblock20.shp" or simply "*.shp"
INPUT_GLOB: str = "tl_*_*_*.shp"

# Optional FIPS filter — leave empty ([]) to export everything
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

# Output feature class or shapefile.
# Examples:
#   r"C:\output\admin.gdb\va_md_dc_blocks_fips_merge"
#   r"C:\output\va_md_dc_blocks_fips_merge.shp"
OUTPUT_PATH: str = r"C:\output\va_md_dc_blocks_fips_merge.shp"

# Name of the FIPS field we will guarantee exists
FIPS_FIELD_NAME: str = "FIPS"

# Candidate source fields in TIGER
STATE_CANDIDATES: tuple[str, ...] = ("STATEFP20", "STATEFP")
COUNTY_CANDIDATES: tuple[str, ...] = ("COUNTYFP20", "COUNTYFP")

# Whether to use in_memory or scratch GDB for temporary steps
USE_IN_MEMORY: bool = True

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

# ArcPy can also echo messages to the GP window
def _gp(msg: str, level: str = "info") -> None:
    if level == "warning":
        arcpy.AddWarning(msg)
    elif level == "error":
        arcpy.AddError(msg)
    else:
        arcpy.AddMessage(msg)

# =============================================================================
# FUNCTIONS
# =============================================================================

def discover_tiger_datasets(root_dir: str | Path, pattern: str) -> list[str]:
    """Return absolute paths to TIGER shapefiles matching ``pattern``.

    This is a straight recursive file-system crawl and does NOT open
    zipped shapefiles. If you have *.zip TIGER, unzip them first.

    Args:
        root_dir: Directory to crawl recursively.
        pattern:  Filename glob to match (applied to basename only).

    Returns:
        Sorted list of absolute shapefile paths.
    """
    root_path = Path(root_dir).expanduser().resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(f"{root_path} is not a valid directory")

    matched: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root_path):
        for name in filenames:
            if name.lower().endswith(".shp") and fnmatch.fnmatch(name, pattern):
                matched.append(str(Path(dirpath, name).resolve()))

    if not matched:
        raise FileNotFoundError(f"No shapefiles matching '{pattern}' were found under {root_path}")

    matched.sort()
    LOGGER.info("Discovered %d shapefile(s) to merge", len(matched))
    _gp(f"Discovered {len(matched)} shapefile(s) to merge.")
    return matched


def merge_shapefiles(
    shp_paths: Sequence[str],
    *,
    workspace: str | None = None,
    out_name: str = "tiger_merge_tmp",
) -> str:
    """Merge multiple shapefiles into a temporary feature class.

    Args:
        shp_paths: Input shapefile paths.
        workspace: Where to create the temp FC; if None, uses in_memory/scratch.
        out_name:  Name of the output FC (no path).

    Returns:
        Full path to the merged feature class.
    """
    if not shp_paths:
        raise ValueError("No input shapefiles provided to merge")

    if workspace is None:
        if USE_IN_MEMORY:
            workspace = "in_memory"
        else:
            workspace = arcpy.env.scratchGDB

    arcpy.env.overwriteOutput = True
    out_fc = os.path.join(workspace, out_name)

    LOGGER.info("Merging %d input shapefile(s) → %s", len(shp_paths), out_fc)
    _gp(f"Merging {len(shp_paths)} input shapefile(s) → {out_fc}")
    arcpy.management.Merge(shp_paths, out_fc)
    return out_fc


def _find_first_existing_field(feature_class: str, candidates: Iterable[str]) -> str | None:
    """Return the first field name from candidates that exists in feature_class."""
    existing_fields = {f.name for f in arcpy.ListFields(feature_class)}
    for cand in candidates:
        if cand in existing_fields:
            return cand
    return None


def ensure_fips_field(
    feature_class: str,
    *,
    fips_field: str = FIPS_FIELD_NAME,
    state_candidates: Sequence[str] = STATE_CANDIDATES,
    county_candidates: Sequence[str] = COUNTY_CANDIDATES,
) -> None:
    """Ensure a 5-digit FIPS field exists by concatenating state + county.

    Args:
        feature_class: Feature class to modify.
        fips_field:    Name of the FIPS field to create/populate.
        state_candidates: Possible TIGER fields for state code.
        county_candidates: Possible TIGER fields for county code.

    Raises:
        KeyError: If required source fields are missing.
    """
    fields = {f.name: f for f in arcpy.ListFields(feature_class)}
    if fips_field in fields:
        LOGGER.info("Field %s already present — skipping creation", fips_field)
        _gp(f"Field {fips_field} already present — skipping creation.")
        return

    state_field = _find_first_existing_field(feature_class, state_candidates)
    county_field = _find_first_existing_field(feature_class, county_candidates)
    if state_field is None or county_field is None:
        raise KeyError(
            "Required columns not found. Expected one of "
            f"{state_candidates} and one of {county_candidates}."
        )

    # Add the field
    LOGGER.info("Adding FIPS field %s", fips_field)
    _gp(f"Adding FIPS field {fips_field}")
    arcpy.management.AddField(
        in_table=feature_class,
        field_name=fips_field,
        field_type="TEXT",
        field_length=5,
    )

    # Populate
    fields_to_update = (state_field, county_field, fips_field)
    with arcpy.da.UpdateCursor(feature_class, fields_to_update) as cursor:
        for row in cursor:
            state_val = str(row[0]).zfill(2) if row[0] is not None else ""
            county_val = str(row[1]).zfill(3) if row[1] is not None else ""
            row[2] = f"{state_val}{county_val}"
            cursor.updateRow(row)

    LOGGER.info("Populated new column %s", fips_field)
    _gp(f"Populated new column {fips_field}")


def filter_by_fips(
    feature_class: str,
    fips_values: Sequence[str],
    *,
    fips_field: str = FIPS_FIELD_NAME,
    workspace: str | None = None,
    out_name: str = "tiger_merge_fips_tmp",
) -> str:
    """Optionally filter a feature class by FIPS list.

    If fips_values is empty, returns the original feature_class.

    Args:
        feature_class: Input FC that already has a FIPS field.
        fips_values:   List of FIPS codes (strings) to select.
        fips_field:    Name of FIPS field.
        workspace:     Where to write filtered output.
        out_name:      Name of filtered output FC.

    Returns:
        Path to filtered feature class (may be same as input if no filter).
    """
    if not fips_values:
        LOGGER.info("FIPS filter empty — exporting full dataset")
        _gp("FIPS filter empty — exporting full dataset")
        return feature_class

    if workspace is None:
        if USE_IN_MEMORY:
            workspace = "in_memory"
        else:
            workspace = arcpy.env.scratchGDB

    out_fc = os.path.join(workspace, out_name)

    # Build SQL like: "FIPS" IN ('11001','24031',...)
    # Respect workspace SQL rules
    sql_values = ",".join(f"'{val}'" for val in fips_values)
    where_clause = f"{arcpy.AddFieldDelimiters(feature_class, fips_field)} IN ({sql_values})"

    LOGGER.info("Selecting by FIPS → %s", where_clause)
    _gp("Selecting by FIPS...")

    arcpy.management.MakeFeatureLayer(feature_class, "fips_lyr", where_clause)
    count = int(arcpy.management.GetCount("fips_lyr").getOutput(0))
    if count == 0:
        LOGGER.warning("No features matched the FIPS list — output will be empty")
        _gp("Warning: No features matched the FIPS list — output will be empty.", "warning")

    arcpy.management.CopyFeatures("fips_lyr", out_fc)

    LOGGER.info("Selected %d feature(s)", count)
    _gp(f"Selected {count} feature(s)")
    return out_fc


def write_output(in_fc: str, out_path: str) -> None:
    """Copy final FC to the requested output path.

    Args:
        in_fc:    Feature class to export.
        out_path: Destination path (FGDB FC or shapefile).
    """
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    LOGGER.info("Writing output → %s", out_path)
    _gp(f"Writing output → {out_path}")
    arcpy.management.CopyFeatures(in_fc, out_path)
    _gp("Output written successfully.")
    LOGGER.info("Output written successfully")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Top-level workflow controller."""
    try:
        shp_paths = discover_tiger_datasets(INPUT_DIR, INPUT_GLOB)

        merged_fc = merge_shapefiles(shp_paths)

        ensure_fips_field(
            merged_fc,
            fips_field=FIPS_FIELD_NAME,
            state_candidates=STATE_CANDIDATES,
            county_candidates=COUNTY_CANDIDATES,
        )

        final_fc = filter_by_fips(
            merged_fc,
            FIPS_TO_FILTER,
            fips_field=FIPS_FIELD_NAME,
        )

        write_output(final_fc, OUTPUT_PATH)

        LOGGER.info("Finished successfully")
        _gp("Finished successfully.")

    except Exception as exc:  # noqa: BLE001
        # ArcPy environment should see this
        LOGGER.exception("Processing failed: %s", exc)
        _gp(f"Processing failed: {exc}", "error")
        # For script tools, it's fine to raise
        sys.exit(1)


if __name__ == "__main__":
    main()
