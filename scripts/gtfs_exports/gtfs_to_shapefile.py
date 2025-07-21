"""Converts GTFS `stops.txt` and `shapes.txt` files into ESRI Shapefiles.

Exports GTFS stops as point features and routes as LineStrings using
standard WGS 84 coordinates. Designed for notebook workflows.
Supports configurable default input/output paths and selective export.

Inputs:
    - GTFS directory with `stops.txt` (required) and `shapes.txt` (optional)
    - Optional export type: "stops", "lines", or "both"

Outputs:
    - `gtfs_stops.shp`: Shapefile of transit stop points
    - `gtfs_lines.shp`: Shapefile of transit route line geometries
"""

from pathlib import Path
from typing import Literal, Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point

# ===========================================================================
# CONFIGURATION
# ===========================================================================

GTFS_CRS = "EPSG:4326"  # Standard CRS for GTFS (WGS 84)
# Type alias for export choices for clarity
ExportKind = Literal["stops", "lines", "both"]

# REQUIRED: Default path to the directory containing GTFS .txt files
DEFAULT_GTFS_DIR: Optional[Path] = Path(r"/path/to/your/default_gtfs_folder")  # <-- EDIT ME

# REQUIRED: Default path to the directory where Shapefiles will be saved
DEFAULT_OUTPUT_DIR: Optional[Path] = Path(r"/path/to/your/default_output_folder")  # <-- EDIT ME
# Set to None if you always want to provide paths as arguments
# DEFAULT_GTFS_DIR = None
# DEFAULT_OUTPUT_DIR = None

# ===========================================================================
# FUNCTIONS
# ===========================================================================

def read_stops(gtfs_dir: Path) -> gpd.GeoDataFrame:
    """Reads GTFS 'stops.txt' file into a Point GeoDataFrame.

    Args:
        gtfs_dir: Path to the directory containing the GTFS files.

    Returns:
        A GeoDataFrame containing stop locations as Points.

    Raises:
        FileNotFoundError: If 'stops.txt' is not found in gtfs_dir.
        ValueError: If required columns are missing or lat/lon are invalid.
    """
    file_path = gtfs_dir / "stops.txt"
    # print(f"Reading stops from: {file_path}") # Optional: uncomment for verbose output
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path}")

    try:
        df = pd.read_csv(file_path, dtype={"stop_id": str})
    except Exception as e:
        raise ValueError(f"Could not read stops.txt: {e}") from e

    required = {"stop_id", "stop_name", "stop_lat", "stop_lon"}
    if not required.issubset(df.columns):
        missing = sorted(list(required.difference(df.columns)))
        raise ValueError(f"Missing required columns in stops.txt: {', '.join(missing)}")

    # Validate and clean coordinate columns
    for col in ["stop_lat", "stop_lon"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Warning: Non-numeric values found in '{col}'. Attempting conversion.")
            original_count = len(df)
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.dropna(subset=[col], inplace=True)
            if len(df) < original_count:
                print(
                    f"Warning: Dropped {original_count - len(df)} stops "
                    f"due to invalid values in '{col}'."
                )

    if df.empty:
        print("Warning: No valid stop data found after cleaning.")
        return gpd.GeoDataFrame(columns=list(required) + ["geometry"], geometry=[], crs=GTFS_CRS)

    try:
        geometry = [Point(xy) for xy in zip(df["stop_lon"], df["stop_lat"])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=GTFS_CRS)
    except Exception as e:
        raise ValueError(f"Stop geometry creation failed: {e}") from e

    # Keep essential columns
    essential_cols = ["stop_id", "stop_name", "stop_lat", "stop_lon", "geometry"]
    cols_to_keep = [col for col in essential_cols if col in gdf.columns]
    gdf = gdf[cols_to_keep]

    # print(f"Successfully processed {len(gdf)} stops.") # Optional: uncomment for verbose output
    return gdf


def read_shapes(gtfs_dir: Path) -> gpd.GeoDataFrame:
    """Reads GTFS 'shapes.txt' file into a LineString GeoDataFrame.

    If 'shapes.txt' is missing, an empty GeoDataFrame is returned.

    Args:
        gtfs_dir: Path to the directory containing the GTFS files.

    Returns:
        A GeoDataFrame containing shape geometries as LineStrings. Returns an
        empty GeoDataFrame if 'shapes.txt' is missing or invalid.

    Raises:
        ValueError: If 'shapes.txt' exists but is missing required columns
                    or contains invalid coordinate/sequence data.
    """
    file_path = gtfs_dir / "shapes.txt"
    # print(f"Reading shapes from: {file_path}") # Optional: uncomment for verbose output
    if not file_path.exists():
        print("Info: Optional file 'shapes.txt' not found. Skipping shapes.")
        return gpd.GeoDataFrame(columns=["shape_id", "geometry"], geometry=[], crs=GTFS_CRS)

    try:
        df = pd.read_csv(file_path, dtype={"shape_id": str})
    except Exception as e:
        raise ValueError(f"Could not read shapes.txt: {e}") from e

    required = {"shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"}
    if not required.issubset(df.columns):
        missing = sorted(list(required.difference(df.columns)))
        raise ValueError(f"Missing required columns in shapes.txt: {', '.join(missing)}")

    # Validate and clean coordinate and sequence columns
    coord_cols = ["shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"]
    for col in coord_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Warning: Non-numeric values found in '{col}'. Attempting conversion.")
            original_count = len(df)
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.dropna(subset=[col], inplace=True)
            if len(df) < original_count:
                print(
                    f"Warning: Dropped {original_count - len(df)} shape points "
                    f"due to invalid values in '{col}'."
                )

    if df.empty:
        print("Warning: No valid shape point data found after cleaning.")
        return gpd.GeoDataFrame(columns=["shape_id", "geometry"], geometry=[], crs=GTFS_CRS)

    # Ensure sequence is integer and sort points correctly
    df["shape_pt_sequence"] = df["shape_pt_sequence"].astype(int)
    df.sort_values(by=["shape_id", "shape_pt_sequence"], inplace=True)

    # Create LineString geometries
    records: list[dict] = []
    try:
        for shape_id, group in df.groupby("shape_id", sort=False):
            coordinates = list(zip(group["shape_pt_lon"], group["shape_pt_lat"]))
            if len(coordinates) < 2:
                print(f"Warning: Shape ID {shape_id} skipped: has fewer than 2 valid points.")
                continue
            line = LineString(coordinates)
            records.append({"shape_id": shape_id, "geometry": line})
    except Exception as e:
        raise ValueError(f"Shape geometry creation failed: {e}") from e

    if not records:
        print("Warning: No valid line geometries constructed from shapes.txt.")
        return gpd.GeoDataFrame(columns=["shape_id", "geometry"], geometry=[], crs=GTFS_CRS)

    gdf = gpd.GeoDataFrame(records, crs=GTFS_CRS)
    # print(f"Successfully constructed {len(gdf)} line geometries.") # Optional verbose output
    return gdf


def export_gdf(gdf: gpd.GeoDataFrame, out_path: Path) -> None:
    """Exports a GeoDataFrame to an ESRI Shapefile.

    Creates the output directory if needed. Skips export if GDF is empty.

    Args:
        gdf: The GeoDataFrame to export.
        out_path: Full path for the output Shapefile (e.g., /path/to/output.shp).

    Raises:
        IOError: If the file cannot be written.
    """
    if gdf.empty:
        print(f"Info: Skipping export for {out_path.name}: No data.")
        return

    try:
        # Ensure output directory exists
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Export to Shapefile
        gdf.to_file(out_path, driver="ESRI Shapefile", index=False)
        print(f"Successfully exported {len(gdf)} features to: {out_path}")
    except Exception as e:
        # Raise as an IOError for clearer upstream handling
        raise IOError(f"Could not write shapefile {out_path}: {e}") from e


# --- Main Orchestration Function (Core Logic) ---


def gtfs_to_shapefiles(
    gtfs_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    kind: ExportKind = "both",
) -> None:
    """Converts GTFS stops and/or shapes files to ESRI Shapefiles.

    Reads data from GTFS directory and writes Shapefiles to output directory.
    Uses default paths from the User Configuration section if arguments
    are not provided.

    Args:
        gtfs_dir: Path to the GTFS directory. If None, uses
                  DEFAULT_GTFS_DIR from module configuration.
        output_dir: Path to the output directory. If None, uses
                    DEFAULT_OUTPUT_DIR from module configuration.
        kind: Specifies elements to export ("stops", "lines", "both").
              Defaults to "both".

    Raises:
        ValueError: If required path arguments are None and defaults are also None.
        NotADirectoryError: If resolved gtfs_dir does not exist or is not a directory.
        FileNotFoundError: If 'stops.txt' is required but not found.
        ValueError: If GTFS files have missing columns or invalid data.
        IOError: If shapefiles cannot be written to the output directory.
    """
    # Resolve paths using defaults if arguments are None
    resolved_gtfs_dir = gtfs_dir if gtfs_dir is not None else DEFAULT_GTFS_DIR
    resolved_output_dir = output_dir if output_dir is not None else DEFAULT_OUTPUT_DIR

    # Validate that paths are set either via args or defaults
    if resolved_gtfs_dir is None:
        raise ValueError("GTFS input directory is not specified and no default is set.")
    if resolved_output_dir is None:
        raise ValueError("Output directory is not specified and no default is set.")

    print("-" * 50)
    print("Starting GTFS to Shapefile conversion...")
    print(f"Input GTFS Directory: {resolved_gtfs_dir}")
    print(f"Output Directory: {resolved_output_dir}")
    print(f"Export Type: {kind}")
    print("-" * 50)

    if not resolved_gtfs_dir.is_dir():
        raise NotADirectoryError(
            f"Input GTFS directory not found or is not a directory: {resolved_gtfs_dir}"
        )

    # Ensure output directory exists before processing files
    try:
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise IOError(f"Could not create output directory {resolved_output_dir}: {e}") from e

    # --- Process Stops ---
    if kind in ("stops", "both"):
        print("\nProcessing Stops...")
        try:
            stops_gdf = read_stops(resolved_gtfs_dir)
            export_gdf(stops_gdf, resolved_output_dir / "gtfs_stops.shp")
        except (FileNotFoundError, ValueError, IOError, NotADirectoryError) as e:
            print(f"ERROR processing stops: {e}")
            # Decide if you want to stop or continue if stops fail
            # raise # Uncomment to stop execution on error
        except Exception as e:
            print(f"An unexpected error occurred during stops processing: {e}")
            # raise # Uncomment to stop execution on error

    # --- Process Shapes (Lines) ---
    if kind in ("lines", "both"):
        print("\nProcessing Shapes (Lines)...")
        try:
            lines_gdf = read_shapes(resolved_gtfs_dir)
            export_gdf(lines_gdf, resolved_output_dir / "gtfs_lines.shp")
        except (ValueError, IOError) as e:
            print(f"ERROR processing shapes: {e}")
            # raise # Uncomment to stop execution on error
        except Exception as e:
            print(f"An unexpected error occurred during shapes processing: {e}")
            # raise # Uncomment to stop execution on error

    print("-" * 50)
    print("Conversion finished.")
    # Provide context requested
    print(f"Current time: {pd.Timestamp.now(tz='US/Eastern').strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print("-" * 50)


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    """Runs example scenarios for the GTFS to Shapefile conversion.

    This function is executed only when the script is run directly.
    It demonstrates calling `gtfs_to_shapefiles` using both configured
    default paths and explicitly provided paths.
    """
    # Scenario 1: Use default paths configured at the top of the file
    # Make sure DEFAULT_GTFS_DIR and DEFAULT_OUTPUT_DIR are set correctly above!
    print("\nRunning example using default paths from configuration...")
    try:
        # Check if defaults are actually set before running
        if DEFAULT_GTFS_DIR and DEFAULT_OUTPUT_DIR:
            # Create dummy directories/files for the example if they don't exist
            # In real use, you'd point the defaults to existing data.
            if not DEFAULT_GTFS_DIR.exists():
                DEFAULT_GTFS_DIR.mkdir(parents=True)
                print(f"Created dummy GTFS dir: {DEFAULT_GTFS_DIR}")
                # Add dummy files if dir was just created
                with open(DEFAULT_GTFS_DIR / "stops.txt", "w") as f:
                    f.write("stop_id,stop_name,stop_lat,stop_lon\nS1,Stop 1,38.8,-77.0")
                with open(DEFAULT_GTFS_DIR / "shapes.txt", "w") as f:
                    f.write(
                        "shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence\nSHP1,38.8,-77.0,1\nSHP1,38.9,-77.1,2"
                    )

            if not DEFAULT_OUTPUT_DIR.exists():
                DEFAULT_OUTPUT_DIR.mkdir(parents=True)
                print(f"Created dummy Output dir: {DEFAULT_OUTPUT_DIR}")

            # Call the core function without path arguments
            gtfs_to_shapefiles(kind="both")
        else:
            print("Skipping default path example: Default paths not configured.")

    except Exception as e:
        print(f"ERROR during default path example: {e}")

    # Scenario 2: Override default paths by providing arguments
    print("\nRunning example overriding default paths...")
    try:
        # Define specific paths for this run
        specific_gtfs_path = Path("./example_gtfs_data")
        specific_output_path = Path("./example_output_data")

        # Create dummy data for this specific run
        specific_gtfs_path.mkdir(parents=True, exist_ok=True)
        specific_output_path.mkdir(parents=True, exist_ok=True)
        with open(specific_gtfs_path / "stops.txt", "w") as f:
            f.write("stop_id,stop_name,stop_lat,stop_lon\nS10,Stop 10,38.85,-77.05")
        # No shapes.txt for this example to test that case

        # Call the core function with specific path arguments
        gtfs_to_shapefiles(
            gtfs_dir=specific_gtfs_path,
            output_dir=specific_output_path,
            kind="stops",  # Only export stops for this example
        )
    except Exception as e:
        print(f"ERROR during specific path example: {e}")

    # Optional cleanup of example directories (uncomment if desired)
    # import shutil
    # if Path("./example_gtfs_data").exists(): shutil.rmtree("./example_gtfs_data")
    # if Path("./example_output_data").exists(): shutil.rmtree("./example_output_data")
    # print("\nCleaned up example directories.")


if __name__ == "__main__":
    # This block executes only when the script is run directly.
    # It calls the main() function which contains the example usage scenarios.
    main()
