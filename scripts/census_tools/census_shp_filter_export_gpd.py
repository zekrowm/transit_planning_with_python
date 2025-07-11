"""Merge and (optionally) FIPS-filter TIGER/Line shapefiles with GeoPandas.

The script:

1.  Loads a list of TIGER Census Block, Block Group, Tract (and other) shapefiles.
2.  Merges them to a single temporary feature class.
3.  If a FIPS list is provided:
      - Adds a new 'FIPS' text field (STATEFP20 + COUNTYFP20).
      - Selects only the requested municipalities.
4.  Writes the final geometry (plus all original attributes) to the
   user-specified output feature class or shapefile.

The relies solely on the open-source stack (GeoPandas, Fiona, PyProj). There
are several limitations users should be aware of:

- File-geodatabase output is not supported natively.  Use a Shapefile,
  GeoPackage, or another driver supported by your GDAL build.
- All input layers must share the same CRS.  The script checks and aborts
  if they do not match.
- Shapefile column names are capped at 10 characters.  If you need full-length
  attribute names, write to a GeoPackage (`.gpkg`) instead.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import List, Sequence

import geopandas as gpd
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# Full paths to TABBLOCK20 (or similar) shapefiles
BLOCK_SHP_FILES: List[str] = [
    r"C:\full\path\to\tl_2023_11_tabblock20.shp",
    r"C:\full\path\to\tl_2023_24_tabblock20.shp",
    r"C:\full\path\to\tl_2023_51_tabblock20.shp",
]

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

# Output file (Shapefile *.shp, GeoPackage *.gpkg, etc.)
OUTPUT_PATH: str = r"C:\output\va_md_dc_blocks_fips_merge.shp"

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

# =============================================================================
# FUNCTIONS
# =============================================================================


def read_shapefile(path: str) -> gpd.GeoDataFrame:
    """Read a single shapefile as a GeoDataFrame.

    Args:
        path: Absolute path to a shapefile.

    Returns:
        GeoDataFrame with all original attributes preserved.
    """
    LOGGER.info("Reading %s", path)
    gdf = gpd.read_file(path)

    # Normalise critical ID columns to string (they are *sometimes* integers)
    for col in ("STATEFP20", "STATEFP", "COUNTYFP20", "COUNTYFP"):
        if col in gdf.columns:
            gdf[col] = gdf[col].astype(str).str.zfill(2 if "STATE" in col else 3)

    return gdf


def merge_shapefiles(shp_paths: Sequence[str]) -> gpd.GeoDataFrame:
    """Load and concatenate multiple shapefiles.

    The CRS of each input layer must match, otherwise an exception is raised.
    """
    gdfs: list[gpd.GeoDataFrame] = [read_shapefile(p) for p in shp_paths]

    # ------------------------------------------------------------------ CRS check
    crs_set = {str(gdf.crs) for gdf in gdfs}
    if len(crs_set) != 1:
        raise RuntimeError(
            "CRS mismatch between input layers: %s.  Re-project first." % ", ".join(crs_set)
        )

    merged = gpd.GeoDataFrame(
        pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs, geometry="geometry"
    )
    LOGGER.info("Merged %d input files → %d features", len(shp_paths), len(merged))
    return merged


def ensure_fips_column(
    gdf: gpd.GeoDataFrame,
    *,
    fips_col: str = "FIPS",
    state_candidates: tuple[str, ...] = ("STATEFP20", "STATEFP"),
    county_candidates: tuple[str, ...] = ("COUNTYFP20", "COUNTYFP"),
) -> gpd.GeoDataFrame:
    """Add a 5-digit county FIPS column if it does not already exist.

    Args:
        gdf:      GeoDataFrame to modify.
        fips_col: Name of the output column.
        state_candidates: Ordered list of possible state code fields.
        county_candidates: Ordered list of possible county code fields.

    Returns:
        The modified GeoDataFrame (same object, returned for convenience).
    """
    if fips_col in gdf.columns:
        LOGGER.info("Field %s already present — skipping creation", fips_col)
        return gdf

    state_field = next((c for c in state_candidates if c in gdf.columns), None)
    county_field = next((c for c in county_candidates if c in gdf.columns), None)
    if state_field is None or county_field is None:
        raise KeyError(
            "Required columns not found.  Expected one of %s and one of %s."
            % (state_candidates, county_candidates)
        )

    gdf[fips_col] = gdf[state_field].astype(str).str.zfill(2) + gdf[county_field].astype(
        str
    ).str.zfill(3)
    LOGGER.info("Populated new column %s", fips_col)
    return gdf


def filter_by_fips(
    gdf: gpd.GeoDataFrame,
    fips_values: Sequence[str],
    *,
    fips_col: str = "FIPS",
) -> gpd.GeoDataFrame:
    """Return a view containing only requested FIPS codes (or everything)."""
    if not fips_values:
        LOGGER.info("FIPS filter empty — exporting full dataset")
        return gdf

    mask = gdf[fips_col].isin(fips_values)
    selected = gdf.loc[mask].copy()
    if selected.empty:
        LOGGER.warning("No features matched the FIPS list — output will be empty")
    else:
        LOGGER.info("Selected %d of %d features", len(selected), len(gdf))
    return selected


def write_output(gdf: gpd.GeoDataFrame, out_path: str) -> None:
    """Write the GeoDataFrame to disk.

    The driver is inferred from the file extension; `.shp` and `.gpkg` are safe.
    """
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    LOGGER.info("Writing %d features → %s", len(gdf), out_path)
    gdf.to_file(out_path)
    LOGGER.info("Output written successfully")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Top-level workflow controller."""
    try:
        # 1. Merge input layers
        merged = merge_shapefiles(BLOCK_SHP_FILES)

        # 2. Make sure we have a FIPS field
        ensure_fips_column(merged)

        # 3. Optional subset
        final_gdf = filter_by_fips(merged, FIPS_TO_FILTER)

        # 4. Export
        write_output(final_gdf, OUTPUT_PATH)

        LOGGER.info("Finished successfully")

    except Exception:  # noqa: BLE001
        # Never swallow exceptions silently in a batch script
        LOGGER.exception("Processing failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
