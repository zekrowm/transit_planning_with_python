"""Analyze transit service coverage of strategic sites using GTFS and GIS data.

This script evaluates how well individual transit routes serve strategically important
locations—such as public housing developments, high schools, hospitals, parks, and other
community facilities—based on spatial proximity.

Intended Use
------------
This tool is designed to support transit planning by quantifying access to key destinations.
Results can inform decisions about route coverage, service prioritization, and equity evaluation.

Assumptions
-----------
- GTFS and GIS layers are projected in a CRS using feet or meters.
- Buffer distance is assumed to be in feet (auto-converted to meters if needed).
- Each shapefile includes a column with a readable feature name (e.g., school name).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable, List, Mapping

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import LineString
from shapely.ops import unary_union

# =============================================================================
# CONFIGURATION
# =============================================================================

# Top‑level directories
GTFS_DIR = Path(r"data/gtfs")  # folder containing GTFS .txt files
SHP_INPUT_DIR = Path(r"data/shapefiles")  # folder with .shp layers to test
OUTPUT_DIR = Path(r"output")  # where CSVs and PNGs are written

# List of `(filename, id_column)` describing each layer to test
# (filenames are relative to SHP_INPUT_DIR)
LAYER_SPECS: list[tuple[str, str]] = [
    ("Hospitals_and_Urgent_Care_Facilities.shp", "DESCRIPTIO"),
    ("School_Facilities.shp", "SCHOOL_NAM"),
    ("Libraries.shp", "DESCRIPTIO"),
]

# Optional filter: only analyze these route_id values.
# Leave empty (`[]`) to process every route in routes.txt
ROUTE_FILTER: list[str] = []

# Analysis options
USE_SHAPE_BUFFER = True  # True → buffer route geometry; False → buffer stops
BUFFER_DIST_FT = 1320.0  # ¼ mile in feet
PLOT_FIG_DPI = 250  # resolution for PNG exports

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("gtfs_buffer_analysis")

# =============================================================================
# FUNCTIONS
# =============================================================================


def _load_gtfs_tables(gtfs_dir: Path) -> Mapping[str, pd.DataFrame]:
    """Load GTFS text files into pandas DataFrames.

    Args:
        gtfs_dir: Directory containing GTFS .txt files.

    Returns:
        Mapping keyed by table name (without .txt) to DataFrame.
    """
    tables = {}
    for fn in ["routes", "trips", "stop_times", "stops", "shapes"]:
        path = gtfs_dir / f"{fn}.txt"
        if not path.exists():
            raise FileNotFoundError(path)
        tables[fn] = pd.read_csv(path)
        log.debug("Loaded %s (%d rows)", fn, len(tables[fn]))
    return tables


def _prepare_route_buffers(
    tables: Mapping[str, pd.DataFrame],
    use_shape_buffer: bool,
    buffer_dist_ft: float,
) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame with one buffered geometry per route_id.

    Depending on *use_shape_buffer*, the buffer is built around the union of
    (a) the route's shape(s) or (b) all its stops.

    The returned GDF is in the CRS of the original GTFS shapes; if that CRS
    uses meters, the function converts *buffer_dist_ft* accordingly.

    Raises:
        ValueError: If shapes.txt lacks an EPSG code in the header.
    """
    # Load shapes as GeoSeries
    shapes_df = tables["shapes"]
    if {"shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"}.difference(
        shapes_df.columns
    ):
        raise ValueError("shapes.txt missing required columns")

    # Convert shape points to LineStrings
    shapes_df.sort_values(["shape_id", "shape_pt_sequence"], inplace=True)
    lines = (
        shapes_df.groupby("shape_id")
        .apply(
            lambda grp: LineString(
                grp[["shape_pt_lon", "shape_pt_lat"]].to_numpy(dtype=float)
            )
        )
        .to_frame(name="geometry")
    )
    shapes_gdf = gpd.GeoDataFrame(lines, geometry="geometry", crs="EPSG:4326")

    # Trips to route mapping
    trips = tables["trips"][["route_id", "trip_id", "shape_id"]]
    route_shapes = (
        trips.drop_duplicates(subset=["route_id", "shape_id"])
        .groupby("route_id")["shape_id"]
        .apply(list)
    )

    # Stops GeoDataFrame
    stops = tables["stops"][["stop_id", "stop_lat", "stop_lon"]].copy()
    stops = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops.stop_lon, stops.stop_lat),
        crs="EPSG:4326",
    )

    # Choose projected CRS (meters) to allow buffering
    projected_crs = "EPSG:3857"
    shapes_gdf = shapes_gdf.to_crs(projected_crs)
    stops = stops.to_crs(projected_crs)
    buff_dist_m = buffer_dist_ft * 0.3048  # convert to meters

    buffers: List[dict[str, object]] = []
    for route_id, shp_ids in route_shapes.items():
        if ROUTE_FILTER and route_id not in ROUTE_FILTER:
            continue

        if use_shape_buffer:
            geoms = shapes_gdf.loc[shp_ids, "geometry"]
        else:
            trip_stops = (
                tables["stop_times"]
                .merge(
                    trips[trips.route_id == route_id][["trip_id"]],
                    on="trip_id",
                    how="inner",
                )["stop_id"]
                .unique()
            )
            geoms = stops[stops.stop_id.isin(trip_stops)].geometry

        if geoms.empty:
            log.warning("No geometry for route %s – skipped", route_id)
            continue

        buf = unary_union(list(geoms)).buffer(buff_dist_m)
        buffers.append({"route_id": route_id, "geometry": buf})

    buffer_gdf = gpd.GeoDataFrame(buffers, geometry="geometry", crs=projected_crs)
    return buffer_gdf


def _load_layers(
    layer_specs: Iterable[tuple[str, str]],
    shp_dir: Path,
) -> dict[str, gpd.GeoDataFrame]:
    """Recursively load each designated shapefile (case‑insensitive search).

    The search now walks *all* subfolders under *shp_dir* using ``Path.rglob``.
    If multiple copies of the same filename are discovered, the first match in
    lexicographic order is used and a warning is logged.

    Args:
        layer_specs: Tuples of (filename, id_column).
        shp_dir: Root directory to search.

    Returns
    -------
    dict[str, gpd.GeoDataFrame]
        Mapping of the *original* filename to its loaded GeoDataFrame.
    """
    layers: dict[str, gpd.GeoDataFrame] = {}

    for filename, id_col in layer_specs:
        # Case‑insensitive recursive search for the .shp
        matches = sorted(
            p for p in shp_dir.rglob("*.shp") if p.name.lower() == filename.lower()
        )

        if not matches:
            log.warning("Layer %s NOT FOUND anywhere under %s", filename, shp_dir)
            continue
        if len(matches) > 1:
            log.warning("Multiple copies of %s found; using %s", filename, matches[0])

        path = matches[0]

        try:
            gdf = gpd.read_file(path)
        except Exception as exc:  # pragma: no cover
            log.warning("Failed to read %s – %s", path, exc)
            continue

        if id_col not in gdf.columns:
            log.warning("Column %s missing in %s – skipped", id_col, path)
            continue

        layers[filename] = gdf[[id_col, "geometry"]].to_crs("EPSG:3857")
        log.info("Loaded %s (%d features)", path.relative_to(shp_dir), len(gdf))

    return layers


def _count_features(
    route_buffers: gpd.GeoDataFrame,
    layers: Mapping[str, gpd.GeoDataFrame],
    layer_specs: Iterable[tuple[str, str]],
    output_dir: Path,
) -> pd.DataFrame:
    """For each route, count intersecting features and write per‑route CSV/PNG.

    Returns:
        A summary DataFrame indexed by route_id with feature counts.
    """
    summary_records: list[dict[str, object]] = []

    for _, route_row in route_buffers.iterrows():
        route_id = route_row.route_id
        buf_geom = route_row.geometry

        per_route_counts: dict[str, object] = {"route_id": route_id}
        feature_name_lists: dict[str, List[str]] = {}

        for filename, id_col in layer_specs:
            if filename not in layers:
                continue
            layer_gdf = layers[filename]
            hits = layer_gdf[layer_gdf.intersects(buf_geom)]
            per_route_counts[filename] = len(hits)
            feature_name_lists[filename] = hits[id_col].astype(str).tolist()

        # Save per‑route CSV
        csv_rows = [
            {
                "layer": fname,
                "count": per_route_counts.get(fname, 0),
                "names": ", ".join(feature_name_lists.get(fname, [])),
            }
            for fname, _ in layer_specs
            if fname in layers
        ]
        pd.DataFrame(csv_rows).to_csv(
            output_dir / f"{route_id}_feature_summary.csv", index=False
        )

        # Plot quick map
        fig, ax = plt.subplots(figsize=(6, 6), dpi=PLOT_FIG_DPI)
        gpd.GeoSeries([buf_geom]).plot(ax=ax, facecolor="none", edgecolor="black")
        for fname in feature_name_lists:
            layers[fname][layers[fname].intersects(buf_geom)].plot(
                ax=ax, label=fname.split(".")[0]
            )
        ax.set_title(f"Route {route_id} buffer & intersecting features")
        ax.axis("off")
        ax.legend()
        fig_path = output_dir / f"{route_id}_buffer_plot.png"
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)

        summary_records.append(per_route_counts)
        log.info("Processed route %s – PNG & CSV written", route_id)

    summary_df = (
        pd.DataFrame(summary_records).set_index("route_id").fillna(0).astype(int)
    )
    return summary_df


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Run the GTFS feature‑coverage analysis."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading GTFS from %s", GTFS_DIR)
    tables = _load_gtfs_tables(GTFS_DIR)

    log.info("Building route buffers (use_shape_buffer=%s)", USE_SHAPE_BUFFER)
    route_buffers = _prepare_route_buffers(tables, USE_SHAPE_BUFFER, BUFFER_DIST_FT)

    if route_buffers.empty:
        log.error("No buffers produced – nothing to do")
        return

    log.info("Loading designated shapefiles")
    layers = _load_layers(LAYER_SPECS, SHP_INPUT_DIR)

    if not layers:
        log.error("No valid layers loaded – nothing to analyze")
        return

    log.info("Counting features per route")
    summary_df = _count_features(route_buffers, layers, LAYER_SPECS, OUTPUT_DIR)

    # Save summary CSV
    summary_path = OUTPUT_DIR / "all_routes_feature_summary.csv"
    summary_df.to_csv(summary_path)
    log.info("Summary written to %s", summary_path)
    log.info("Done.")


if __name__ == "__main__":
    main()
