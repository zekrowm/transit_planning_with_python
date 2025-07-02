"""Converts GTFS shapes.txt data into a WGS-84 polyline shapefile.

This module reads the 'shapes.txt' file from a specified GTFS input folder,
processes the geographic points, and generates a polyline feature for each
unique shape_id. It then exports these polylines into a new shapefile.
Optionally, it can filter shapes based on route_short_name values found in
'routes.txt' and 'trips.txt'.

Requires ArcGIS Pro-installed ArcPy (no pandas / geopandas needed).
"""

import csv
from pathlib import Path

import arcpy

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FOLDER = Path(r"Path\To\YourGTFS_Folder")  # folder containing shapes.txt
OUTPUT_FOLDER = Path(r"Path\To\Your\Output_Folder")  # where the shapefile will go
OUTPUT_NAME = "gtfs_shapes.shp"  # .shp is required
SR_EPSG = 4326  # WGS‑84

# Optional route filter in list
ROUTE_FILTER: list[str] = []

# =============================================================================
# FUNCTIONS
# =============================================================================


def read_shapes_txt(txt_path: Path) -> dict[str, list[tuple[int, float, float]]]:
    """Parses shapes.txt and returns a dictionary of shapes.

    The dictionary is keyed by `shape_id`, with each value being a list of
    (sequence, longitude, latitude) tuples, sorted by sequence.

    Args:
        txt_path: The full path to the `shapes.txt` file.

    Returns:
        A dictionary where keys are `shape_id` (str) and values are lists of
        tuples, each representing a point: `(shape_pt_sequence, shape_pt_lon, shape_pt_lat)`.

    Raises:
        ValueError: If `shapes.txt` is missing required columns.
        FileNotFoundError: If `txt_path` does not exist.
    """
    shapes: dict[str, list[tuple[int, float, float]]] = {}
    with txt_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        # basic column check
        required = {"shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"shapes.txt missing required columns: {', '.join(missing)}")

        for row in reader:
            sid = row["shape_id"]
            lat = float(row["shape_pt_lat"])
            lon = float(row["shape_pt_lon"])
            seq = int(row["shape_pt_sequence"])
            shapes.setdefault(sid, []).append((seq, lon, lat))

    # sort each list by sequence
    for sid in shapes:
        shapes[sid].sort(key=lambda t: t[0])

    return shapes


def filter_shapes_by_route(
    shapes: dict[str, list[tuple[int, float, float]]], route_filter: list[str], gtfs_folder: Path
) -> dict[str, list[tuple[int, float, float]]]:
    """Filters shapes based on associated `route_short_name` values.

    Returns only those entries in `shapes` whose `shape_id` is used by a trip
    on a route with `route_short_name` in the provided `route_filter` list.
    If `route_filter` is empty, no filtering is applied, and the original
    `shapes` dictionary is returned.

    Args:
        shapes: A dictionary of shapes, as returned by `read_shapes_txt`.
        route_filter: A list of `route_short_name` strings to filter by.
            If empty, no filtering occurs.
        gtfs_folder: The path to the GTFS directory containing `routes.txt`
            and `trips.txt`.

    Returns:
        A new dictionary containing only the shapes that match the filtering criteria.
        If `route_filter` is empty, the original `shapes` dictionary is returned.

    Raises:
        FileNotFoundError: If `routes.txt` or `trips.txt` are not found when filtering.
    """
    if not route_filter:
        return shapes

    # 1. Load route_id → route_short_name
    routes_file = gtfs_folder / "routes.txt"
    route_id_to_short: dict[str, str] = {}
    with routes_file.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            route_id_to_short[row["route_id"]] = row["route_short_name"]

    # 2. Build shape_id → list(route_short_name)
    trips_file = gtfs_folder / "trips.txt"
    shape_to_shorts: dict[str, list[str]] = {}
    with trips_file.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sid = row.get("shape_id")
            short = route_id_to_short.get(row.get("route_id", ""))
            if sid and short:
                shape_to_shorts.setdefault(sid, []).append(short)

    # 3. Filter: keep shapes where any associated short is in our list
    filtered: dict[str, list[tuple[int, float, float]]] = {}
    for sid, pts in shapes.items():
        shorts = shape_to_shorts.get(sid, [])
        if any(short in route_filter for short in shorts):
            filtered[sid] = pts

    print(f"Filtered shapes: {len(filtered)} of {len(shapes)} retained for routes {route_filter}")
    return filtered


def create_output_fc(out_folder: Path, name: str, spatial_ref: arcpy.SpatialReference) -> str:
    """Creates an empty polyline feature class and returns its full path.

    If the output feature class already exists, it will be overwritten.
    A 'shape_id' text field is added to the feature class.

    Args:
        out_folder: The `Path` object for the output directory.
        name: The name of the output shapefile (e.g., "gtfs_shapes.shp").
        spatial_ref: An `arcpy.SpatialReference` object for the feature class.

    Returns:
        The full string path to the newly created (or overwritten) feature class.
    """
    out_folder.mkdir(parents=True, exist_ok=True)
    out_fc = str(out_folder / name)  # ArcPy expects a string path
    if arcpy.Exists(out_fc):  # overwrite if it already exists
        arcpy.management.Delete(out_fc)

    arcpy.management.CreateFeatureclass(
        out_path=str(out_folder),
        out_name=name,
        geometry_type="POLYLINE",
        spatial_reference=spatial_ref,
    )
    # Add a TEXT field for shape_id (10 is fine for most GTFS feeds; increase if needed)
    arcpy.management.AddField(out_fc, "shape_id", "TEXT", field_length=50)
    return out_fc


def insert_shapes(
    out_fc: str,
    shapes: dict[str, list[tuple[int, float, float]]],
    spatial_ref: arcpy.SpatialReference,
) -> None:
    """Inserts GTFS shapes as polyline features into the output feature class.

    Each entry in the `shapes` dictionary is converted into an `arcpy.Polyline`
    object and inserted as a new row into the specified feature class, along
    with its `shape_id`.

    Args:
        out_fc: The full string path to the output feature class (e.g., a shapefile).
        shapes: A dictionary of shapes, as returned by `read_shapes_txt`
            (potentially after filtering).
        spatial_ref: An `arcpy.SpatialReference` object to define the
            coordinate system of the inserted geometries.
    """
    with arcpy.da.InsertCursor(out_fc, ["shape_id", "SHAPE@"]) as cursor:
        for sid, pts in shapes.items():
            arr = arcpy.Array([arcpy.Point(lon, lat) for _, lon, lat in pts])
            poly = arcpy.Polyline(arr, spatial_ref)
            cursor.insertRow((sid, poly))


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Main function to execute the GTFS shapes to shapefile conversion.

    This function orchestrates the entire process:
    1. Reads shapes data from `shapes.txt`.
    2. Applies an optional route filter to the shapes.
    3. Creates a new, empty polyline feature class.
    4. Inserts the processed shapes into the feature class.
    5. Prints a completion message.

    Raises:
        FileNotFoundError: If `shapes.txt` is not found at the specified input path.
        ValueError: If `shapes.txt` is missing required columns.
        Exception: Catches and prints any general ArcPy or other errors during execution.
    """
    shapes_txt = INPUT_FOLDER / "shapes.txt"
    if not shapes_txt.exists():
        raise FileNotFoundError(f"{shapes_txt} not found.")

    print(f"Reading {shapes_txt} …")
    shapes = read_shapes_txt(shapes_txt)

    # *** apply your route filter here ***
    shapes = filter_shapes_by_route(shapes, ROUTE_FILTER, INPUT_FOLDER)

    sr = arcpy.SpatialReference(SR_EPSG)
    print("Creating output feature class …")
    out_fc = create_output_fc(OUTPUT_FOLDER, OUTPUT_NAME, sr)

    print("Writing geometries …")
    insert_shapes(out_fc, shapes, sr)

    print(f"✅ Done!  Shapefile saved at: {out_fc}")


if __name__ == "__main__":
    main()
