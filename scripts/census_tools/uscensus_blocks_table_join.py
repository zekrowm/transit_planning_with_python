"""ArcPy-only join of Census block geometry to an attribute table (CSV → table).

This module assumes you already have:

    1. a merged / FIPS-filtered TIGER TABBLOCK20 feature class or shapefile; and
    2. a fully joined block + tract attribute CSV (i.e. one row per block).

No further data wrangling is performed here—just a clean, key-based join
implemented using ArcPy tooling.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Final

import arcpy

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input geometry holding the block polygons
SHAPEFILE_PATH: Final[str] = r"PATH\TO\SHP\va_md_dc_blocks_fips_merge.shp"

# CSV with one record per block and a 15-digit block FIPS somewhere in it
TABLE_CSV_PATH: Final[str] = r"PATH\TO\CSV\joined_blocks.csv"

# Output feature class or shapefile
OUTPUT_PATH: Final[str] = r"PATH\TO\OUTPUT\va_md_dc_blocks_plus_data.shp"

# Field in the geometry that should hold the 15-digit block ID
LEFT_KEY: Final[str] = "GEOID20"

# Preferred field in the CSV that should hold the 15-digit block ID
RIGHT_KEY: Final[str] = "GEO_ID"

# Fallback source in the CSV that may still hold a 24-char GEOID, from which
# we take the rightmost 15 chars.
DERIVATION_SRC: Final[str] = "GEO_ID_blk"

# Whether to try to coerce integer-ish fields to double after the join.
# ArcPy won't have the same "nullable Int64" issue as pandas, but CSV →
# table inference can produce short integers; some teams prefer doubles
# for export safety.
FORCE_FLOAT: Final[bool] = False

# ArcGIS environment overrides
arcpy.env.overwriteOutput = True

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
# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------


def _list_field_names(dataset: str) -> list[str]:
    """Return a simple list of field names on *dataset*."""
    return [f.name for f in arcpy.ListFields(dataset)]


def _field_exists(dataset: str, field_name: str) -> bool:
    """Check whether *field_name* exists on *dataset*."""
    return field_name in _list_field_names(dataset)


def _add_text_key_field(
    dataset: str,
    source_field: str,
    target_field: str,
    length: int = 15,
) -> str:
    """Add a text field and calculate it to the rightmost *length* chars.

    This is the ArcPy equivalent of:

        df[key] = df[key].str[-15:]

    Args:
        dataset: Feature class or table path.
        source_field: Existing field from which to derive the 15-char ID.
        target_field: Name of the new text field to create.
        length: Desired number of rightmost characters (default 15).

    Returns:
        Name of the new field.
    """
    if not _field_exists(dataset, source_field):
        raise KeyError(f"'{source_field}' not found in {dataset}")

    if not _field_exists(dataset, target_field):
        LOGGER.info("Adding text key field '%s' to %s", target_field, dataset)
        arcpy.management.AddField(
            in_table=dataset,
            field_name=target_field,
            field_type="TEXT",
            field_length=length,
        )

    # Rightmost N chars expression; works in Pro with PYTHON3
    expr = f"!{source_field}![(-{length}):]"  # e.g. !GEO_ID![-15:]
    LOGGER.info("Calculating %s from %s (rightmost %d chars)", target_field, source_field, length)
    arcpy.management.CalculateField(
        in_table=dataset,
        field=target_field,
        expression=expr,
        expression_type="PYTHON3",
    )
    return target_field


def _ensure_feature_class_key(fc: str, key: str) -> str:
    """Ensure the feature class *fc* has a 15-char text join key.

    If *key* exists and is already text, we still normalize it to rightmost 15
    chars in a new field called ``<key>_15``. This avoids in-place type edits
    and keeps the original data intact.

    Args:
        fc: Input feature class or shapefile.
        key: Field expected to hold the block identifier.

    Returns:
        The name of the **normalized** text key field to use for joining.
    """
    if not _field_exists(fc, key):
        raise KeyError(f"'{key}' not found in {fc}")

    # We do not assume the original key is text or 15 chars, so we derive a new one.
    norm_field = f"{key}_15"
    return _add_text_key_field(fc, key, norm_field, length=15)


def _load_csv_to_memory(csv_path: str) -> str:
    """Load a CSV into in_memory and return the table name."""
    out_name = "attrs_csv"
    LOGGER.info("Loading CSV → in_memory table: %s → %s", csv_path, out_name)
    arcpy.management.TableToTable(
        in_rows=csv_path,
        out_path="in_memory",
        out_name=out_name,
    )
    return f"in_memory\\{out_name}"


def _ensure_table_key(table: str, preferred: str, fallback: str) -> str:
    """Ensure the attribute table has a normalized 15-char text key.

    This mimics the pandas logic in the GeoPandas version: first try
    *preferred*; if not present, try *fallback*; otherwise raise.

    Args:
        table: Table path (likely in_memory).
        preferred: Column expected to hold the 15-digit block FIPS.
        fallback: Column that may hold a 24-character GEOID.

    Returns:
        Name of text field to use for the join.
    """
    table_fields = _list_field_names(table)

    if preferred in table_fields:
        return _add_text_key_field(table, preferred, f"{preferred}_15", length=15)
    if fallback in table_fields:
        return _add_text_key_field(table, fallback, f"{fallback}_15", length=15)

    raise KeyError(
        f"Neither '{preferred}' nor '{fallback}' found in attribute table {table}.",
    )


def _maybe_force_float(fc: str) -> None:
    """Optionally coerce short/long integer fields in *fc* to double.

    ArcGIS is a little more tolerant than the shapefile+Fiona+pandas stack,
    so this is OFF by default. If you turn it on, we alter fields in place
    where supported (geodatabases). Shapefiles do not support AlterField
    for type changes.
    """
    if not FORCE_FLOAT:
        return

    desc = arcpy.Describe(fc)
    is_gdb = desc.dataType in ("FeatureClass", "Table") and desc.path.lower().endswith(".gdb")
    if not is_gdb:
        LOGGER.warning(
            "FORCE_FLOAT is True, but '%s' is not in a file gdb. Skipping numeric coercion.",
            fc,
        )
        return

    for field in arcpy.ListFields(fc):
        if field.type in ("SmallInteger", "Integer"):
            new_type = "DOUBLE"
            LOGGER.info(
                "Altering field %s (%s → %s) on %s",
                field.name,
                field.type,
                new_type,
                fc,
            )
            arcpy.management.AlterField(
                in_table=fc,
                field=field.name,
                new_field_type=new_type,
            )


def _ensure_temp_gdb(base_path: str) -> str:
    """Ensure a scratch FGDB next to *base_path*; return its path."""
    base_dir = os.path.dirname(base_path)
    gdb_path = os.path.join(base_dir, "scratch_join.gdb")
    if not arcpy.Exists(gdb_path):
        LOGGER.info("Creating scratch GDB at %s", gdb_path)
        arcpy.management.CreateFileGDB(base_dir, "scratch_join.gdb")
    return gdb_path


# -----------------------------------------------------------------------------
# CORE LOGIC
# -----------------------------------------------------------------------------


def join_blocks_to_attributes(
    blocks_fc: str,
    attrs_csv: str,
    out_path: str,
    left_key: str = LEFT_KEY,
    right_key: str = RIGHT_KEY,
    derivation_src: str = DERIVATION_SRC,
) -> None:
    """Join block geometry to attribute rows and write the result.

    Args:
        blocks_fc: Path to feature class / shapefile with block polygons.
        attrs_csv: Path to the CSV containing block-level attributes.
        out_path: Where to write the joined output.
        left_key: Field in the feature class that has the 15-digit GEOID.
        right_key: Field in the CSV expected to have the 15-digit GEOID.
        derivation_src: Fallback CSV field from which to derive the 15-digit GEOID.

    Raises:
        KeyError: If required join fields are missing.
    """
    LOGGER.info("Start join: %s ← %s", blocks_fc, attrs_csv)

    # Work on an in-memory copy of the blocks so we can derive a text key
    blocks_mem = "in_memory\\blocks_fc"
    LOGGER.info("Copying features to memory: %s → %s", blocks_fc, blocks_mem)
    arcpy.management.CopyFeatures(blocks_fc, blocks_mem)

    left_join_field = _ensure_feature_class_key(blocks_mem, left_key)

    attrs_mem = _load_csv_to_memory(attrs_csv)
    right_join_field = _ensure_table_key(attrs_mem, right_key, derivation_src)

    # Perform the join in memory
    joined_view = "joined_view"
    LOGGER.info(
        "Joining in memory: %s.%s → %s.%s",
        blocks_mem,
        left_join_field,
        attrs_mem,
        right_join_field,
    )
    arcpy.management.MakeFeatureLayer(blocks_mem, joined_view)
    arcpy.management.AddJoin(
        in_layer_or_view=joined_view,
        in_field=left_join_field,
        join_table=attrs_mem,
        join_field=right_join_field,
        join_type="KEEP_ALL",  # left join semantics
    )

    # Export the joined layer to the desired output
    out_path = str(Path(out_path))
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if out_path.lower().endswith(".shp"):
        # Shapefile: we first copy to a temp GDB to keep long names,
        # then export to shp and warn about truncation.
        scratch_gdb = _ensure_temp_gdb(out_path)
        tmp_fc = os.path.join(scratch_gdb, "joined_blocks")
        LOGGER.info("Copying joined layer → %s (temp GDB)", tmp_fc)
        arcpy.management.CopyFeatures(joined_view, tmp_fc)

        LOGGER.info("Exporting temp GDB feature class → shapefile %s", out_path)
        arcpy.management.FeatureClassToFeatureClass(
            in_features=tmp_fc,
            out_path=str(out_dir),
            out_name=Path(out_path).name,
        )
        LOGGER.warning(
            "Output is a shapefile → ArcGIS will truncate field names to 10 chars where needed.",
        )
    else:
        # File GDB or other supported output
        LOGGER.info("Copying joined layer → %s", out_path)
        arcpy.management.CopyFeatures(joined_view, out_path)

    # Optionally coerce integers to doubles (only reliable in GDB)
    _maybe_force_float(out_path)

    LOGGER.info("Finished join → %s", out_path)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Script entry point."""
    join_blocks_to_attributes(
        blocks_fc=SHAPEFILE_PATH,
        attrs_csv=TABLE_CSV_PATH,
        out_path=OUTPUT_PATH,
        left_key=LEFT_KEY,
        right_key=RIGHT_KEY,
        derivation_src=DERIVATION_SRC,
    )


if __name__ == "__main__":
    main()
