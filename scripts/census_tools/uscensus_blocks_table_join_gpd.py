"""GeoPandas-only join of Census block geometry to an attribute table.

This module assumes you already have:

    1. a merged / FIPS-filtered TIGER `TABBLOCK20` geometry; and
    2. a fully joined block + tract attribute CSV.

No further data wrangling is performed here—just a clean spatial join.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final, Literal

import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame
from pandas import DataFrame

# =============================================================================
# CONFIGURATION
# =============================================================================

SHAPEFILE_PATH: Final[str] = r"PATH\TO\SHP\va_md_dc_blocks_fips_merge.shp"
TABLE_CSV_PATH: Final[str] = r"PATH\TO\CSV\joined_blocks.csv"
OUTPUT_PATH: Final[str] = r"PATH\TO\OUTPUT\va_md_dc_blocks_plus_data.shp"
MAX_FIELD_LEN: Final[int] = 10  # Shapefile DBF column-name limit

LEFT_KEY: Final[str] = "GEOID20"  # geometry field carrying the 15-digit ID
RIGHT_KEY: Final[str] = "GEO_ID"  # CSV field carrying the 15-digit ID
FORCE_FLOAT: Final[bool] = True  # cast nullable Int64 → float64

DERIVATION_SRC: Final[str] = "GEO_ID_blk"  # column that still has the 24-char GEOID

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s :: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

# =============================================================================
# FUNCTIONS
# =============================================================================


def load_blocks(shp_path: str, key: str = LEFT_KEY) -> GeoDataFrame:
    """Read block geometry from *shp_path*.

    Args:
        shp_path: Path to the shapefile or GeoPackage layer.
        key:      Column expected to hold the block identifier.

    Returns:
    -------
        GeoDataFrame with *key* coerced to ``str``.

    Raises:
    ------
    KeyError
        If *key* is missing from the file.
    """
    LOGGER.info("Reading geometry: %s", shp_path)
    gdf: GeoDataFrame = gpd.read_file(shp_path)
    if key not in gdf.columns:
        raise KeyError(f"'{key}' not found in {shp_path}")
    gdf[key] = gdf[key].astype(str)
    return gdf


def load_attributes(csv_path: str, key: str = RIGHT_KEY) -> DataFrame:
    df = pd.read_csv(csv_path, dtype=str)

    if key in df.columns:
        # Always normalize to 15-digit FIPS
        df[key] = df[key].str[-15:]
    elif DERIVATION_SRC in df.columns:
        df[key] = df[DERIVATION_SRC].str[-15:]
    else:
        raise KeyError(
            f"Neither '{key}' nor '{DERIVATION_SRC}' found in {csv_path}."
        )

    return df


def join_blocks_to_attributes(
    blocks: GeoDataFrame,
    attrs: DataFrame,
    left_key: str = LEFT_KEY,
    right_key: str = RIGHT_KEY,
    how: Literal["left", "right", "outer", "inner", "cross"] = "left",
) -> GeoDataFrame:
    """Merge *attrs* onto *blocks* on the specified keys.

    Args:
        blocks:    Geometry-bearing GeoDataFrame.
        attrs:     Attribute DataFrame.
        left_key:  Join field in *blocks*.
        right_key: Join field in *attrs*.
        how:       Pandas merge strategy.

    Returns:
    -------
        GeoDataFrame with geometry and attribute columns.

    Raises:
    ------
    ValueError
        If duplicates in either key violate a 1 : 1 expectation.
    """
    LOGGER.info("Merging geometry (%d) with table (%d)…", len(blocks), len(attrs))
    merged: GeoDataFrame = blocks.merge(
        attrs,
        left_on=left_key,
        right_on=right_key,
        how=how,
        validate="1:1",
    )
    LOGGER.info("Merged result → %d rows, %d columns", *merged.shape)

    if FORCE_FLOAT:
        _cast_int64_to_float(merged)

    return merged


def save_output(gdf: GeoDataFrame, out_path: str) -> None:
    """Write *gdf* to disk, creating parent dirs if needed.

    If *out_path* ends in “.shp” the driver is forced to
    “ESRI Shapefile”, column names are truncated to <= 10 characters,
    and the implicit pandas index is suppressed.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.lower().endswith(".shp"):
        _truncate_field_names(gdf)
        driver: str | None = "ESRI Shapefile"
    else:
        driver = None  # let Fiona infer (GeoPackage, Parquet, etc.)

    LOGGER.info("Writing output → %s", path.resolve())
    gdf.to_file(out_path, driver=driver, index=False)
    LOGGER.info("Finished")


def _truncate_field_names(gdf: GeoDataFrame, max_len: int = MAX_FIELD_LEN) -> None:
    """Ensure every attribute name fits the Shapefile 10-char DBF limit.

    Renames are applied **in place**.  When a truncated name collides
    with one already used, a numeric suffix is appended to make it unique.
    """
    renames: dict[str, str] = {}
    seen: set[str] = set()

    for col in list(gdf.columns):
        if col == gdf.geometry.name:
            continue  # geometry column is not stored in the DBF
        new = col[:max_len]
        counter = 1
        while new in seen:
            new = f"{col[: max_len - len(str(counter))]}{counter}"
            counter += 1
        if new != col:
            renames[col] = new
            seen.add(new)
        else:
            seen.add(col)

    if renames:
        LOGGER.warning(
            "Truncated %d column name(s) to %d chars: %s",
            len(renames),
            max_len,
            renames,
        )
        gdf.rename(columns=renames, inplace=True)


# -----------------------------------------------------------------------------
# PRIVATE HELPERS
# -----------------------------------------------------------------------------


def _cast_int64_to_float(gdf: GeoDataFrame) -> None:
    """Convert nullable Int64 columns to float64 in-place for shapefile safety.

    Shapefile drivers cannot store pandas' nullable integer extension type.
    """
    int_cols: list[str] = [
        str(col) for col, dtype in gdf.dtypes.items() if pd.api.types.is_integer_dtype(dtype)
    ]
    if int_cols:
        LOGGER.debug("Casting %d Int64 column(s) → float64: %s", len(int_cols), int_cols)
        gdf[int_cols] = gdf[int_cols].astype("float64")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Script entry point."""
    blocks_gdf = load_blocks(SHAPEFILE_PATH)
    attrs_df = load_attributes(TABLE_CSV_PATH)

    joined = join_blocks_to_attributes(blocks_gdf, attrs_df)
    save_output(joined, OUTPUT_PATH)


if __name__ == "__main__":
    main()
