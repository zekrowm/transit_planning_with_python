"""Validate GTFS route and stop data against a transit-system shapefile.

The script compares GTFS route information to a system shapefile,
identifies mismatches, flags stops outside a distance buffer, and
optionally produces per-route visualisations.

Outputs
-------
gtfs_shp_comparison.csv
    Similarity scores and flags for each route.
problem_stops.shp
    Stops either unmatched to a route or outside the buffer.
problem_stops.xlsx
    Tabular version (with distances) of the shapefile above.
gtfs_shp_comparison_with_flags.csv
    Route table indicating whether a route has flagged stops.
problem_stops_<route>.jpeg
    One JPEG per route that has out-of-buffer stops.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from rapidfuzz import fuzz
from shapely.geometry import Point, base

# =============================================================================
# CONFIGURATION (STATIC)
# =============================================================================

GTFS_DIR: Path = Path(r"\\your_project_folder\system_gtfs")
SHAPEFILE_PATH: Path = Path(
    r"\\your_project_folder\your_transit_system\your_transit_system.shp"
)
OUTPUT_DIR: Path = Path(r"\\your_project_folder\output")

ROUTE_NUMBER_COLUMN = "ROUTE_NUMB"
ROUTE_NAME_COLUMN = "ROUTE_NAME"

DISTANCE_ALLOWANCE_FT = 100  # feet

INPUT_CRS = "EPSG:4326"  # WGS 84
PROJECTED_CRS = "EPSG:26918"  # NAD83 / UTM 18 N  (adjust if needed)
OUTPUT_CRS = "EPSG:4326"  # WGS 84

LOG_LEVEL = logging.INFO

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

# =============================================================================
# CORE FUNCTIONS
# =============================================================================


def load_gtfs_data(gtfs_dir: Path) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """Load the four mandatory GTFS text files.

    Args:
        gtfs_dir: Directory containing *routes.txt*, *stops.txt*,
            *trips.txt* and *stop_times.txt*.

    Returns:
        A 4-tuple ``(routes, stops, trips, stop_times)`` of pandas
        ``DataFrame`` objects, in that exact order.

    Raises:
        FileNotFoundError: If any of the required GTFS files are absent.
    """
    required = ["routes.txt", "stops.txt", "trips.txt", "stop_times.txt"]
    paths = {name: gtfs_dir / name for name in required}

    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Required GTFS files not found: {', '.join(missing)}")

    LOGGER.info("Loading GTFS files …")
    return (
        pd.read_csv(paths["routes.txt"]),
        pd.read_csv(paths["stops.txt"]),
        pd.read_csv(paths["trips.txt"]),
        pd.read_csv(paths["stop_times.txt"]),
    )


def load_shapefile(shp_path: Path, crs: str) -> gpd.GeoDataFrame:
    """Read the route shapefile and coerce it to the desired CRS.

    Args:
        shp_path: Absolute or relative path to the shapefile on disk.
        crs: EPSG code (e.g. ``"EPSG:4326"``) that the output GeoDataFrame
            should use.

    Returns:
        The route shapefile as a ``geopandas.GeoDataFrame`` in *crs*.

    Raises:
        FileNotFoundError: If *shp_path* does not exist.
    """
    if not shp_path.is_file():
        raise FileNotFoundError(f"Shapefile not found: {shp_path}")

    LOGGER.info("Loading route shapefile …")
    shp_df = gpd.read_file(shp_path)

    if shp_df.crs is None:
        LOGGER.warning("Shapefile CRS undefined; setting to %s", crs)
        shp_df.set_crs(crs, inplace=True)
    else:
        shp_df = shp_df.to_crs(crs)

    return shp_df


def preprocess_data(
    routes: DataFrame, shp: gpd.GeoDataFrame, route_number_col: str
) -> Tuple[DataFrame, gpd.GeoDataFrame]:
    """Clean and align route identifiers prior to merging.

    * Whitespace is stripped.
    * Identifiers are coerced to *str*.
    * Embedded spaces are removed from shapefile route numbers.

    Args:
        routes: ``routes.txt`` DataFrame.
        shp:   Route shapefile GeoDataFrame.
        route_number_col: Name of the column in *shp* that stores the
            route short name or number.

    Returns:
        A tuple ``(routes_clean, shp_clean)`` whose copies are ready for
        merging.
    """
    routes = routes.copy()
    shp = shp.copy()

    routes["route_short_name_str"] = routes["route_short_name"].astype(str).str.strip()
    shp[f"{route_number_col}_str"] = (
        shp[route_number_col].astype(str).str.replace(" ", "").str.strip()
    )

    return routes, shp


def merge_and_score(
    routes: DataFrame,
    shp: gpd.GeoDataFrame,
    route_number_col: str,
    route_name_col: str,
) -> DataFrame:
    """Merge GTFS and shapefile routes and compute fuzzy-match scores.

    Args:
        routes: Cleaned ``routes.txt`` table (see :func:`preprocess_data`).
        shp: Cleaned shapefile GeoDataFrame (see :func:`preprocess_data`).
        route_number_col: Column in *shp* that stores the route short name.
        route_name_col:   Column in *shp* that stores the route long name.

    Returns:
        A merged ``DataFrame`` with added columns:

        * ``short_name_score`` / ``long_name_score`` – RapidFuzz ratios.
        * ``short_name_exact_match`` / ``long_name_exact_match`` – booleans.
    """
    LOGGER.info("Merging GTFS and shapefile routes …")
    merged = pd.merge(
        routes,
        shp,
        left_on="route_short_name_str",
        right_on=f"{route_number_col}_str",
        how="outer",
        suffixes=("_gtfs", "_shp"),
    )

    LOGGER.info("Computing similarity scores …")
    merged["short_name_score"] = merged.apply(
        lambda row: fuzz.ratio(
            str(row["route_short_name_str"]), str(row[f"{route_number_col}_str"])
        ),
        axis=1,
    )
    merged["long_name_score"] = merged.apply(
        lambda row: fuzz.ratio(
            str(row.get("route_long_name", "")), str(row.get(route_name_col, ""))
        ),
        axis=1,
    )

    merged["short_name_exact_match"] = merged["short_name_score"] == 100
    merged["long_name_exact_match"] = merged["long_name_score"] == 100
    return merged


def export_comparison(merged: DataFrame, out_dir: Path) -> DataFrame:
    """Write the comparison table to disk and log match statistics.

    Args:
        merged: Output of :func:`merge_and_score`.
        out_dir: Directory where *gtfs_shp_comparison.csv* should be saved.

    Returns:
        The same ``DataFrame`` that was passed in (handy for chaining).
    """
    LOGGER.info("Exporting route comparison CSV …")
    out_path = out_dir / "gtfs_shp_comparison.csv"
    merged.to_csv(out_path, index=False)

    pct_short = merged["short_name_exact_match"].mean() * 100
    pct_long = merged["long_name_exact_match"].mean() * 100
    LOGGER.info("Exact short-name matches: %.2f%%", pct_short)
    LOGGER.info("Exact long-name  matches: %.2f%%", pct_long)
    LOGGER.info("Route comparison exported to %s", out_path)
    return merged


def convert_stops_to_gdf(stops: DataFrame, crs: str) -> gpd.GeoDataFrame:
    """Convert a ``stops.txt`` table to a GeoDataFrame.

    Args:
        stops: GTFS *stops.txt* DataFrame.
        crs:   Desired geographic CRS (e.g. ``"EPSG:4326"``).

    Returns:
        A point-geometry ``GeoDataFrame`` using latitude/longitude columns.
    """
    stops = stops.copy()
    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
    stops = stops.dropna(subset=["stop_lat", "stop_lon"])

    geometry = [Point(xy) for xy in zip(stops["stop_lon"], stops["stop_lat"])]
    return gpd.GeoDataFrame(stops, geometry=geometry, crs=crs)


def _ensure_projected(gdf: gpd.GeoDataFrame, projected_crs: str) -> gpd.GeoDataFrame:
    """Return *gdf* in *projected_crs*.

    Helper for distance calculations that require a projected CRS.

    Args:
        gdf: Input GeoDataFrame.
        projected_crs: Target projected CRS (EPSG code).

    Returns:
        The GeoDataFrame re-projected to *projected_crs*.
    """
    return gdf.to_crs(projected_crs)


def prepare_geometries(
    stops: gpd.GeoDataFrame,
    shp: gpd.GeoDataFrame,
    projected_crs: str,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Project inputs and ensure consistent geometry column names.

    Args:
        stops: Point GeoDataFrame of GTFS stops.
        shp:   Line or polyline GeoDataFrame of transit routes.
        projected_crs: CRS in which distance buffering will occur.

    Returns:
        A tuple ``(stops_projected, shp_projected)``.
    """
    stops = _ensure_projected(stops, projected_crs)
    shp = _ensure_projected(shp, projected_crs)

    if "route_geometry" not in shp.columns:
        shp = shp.rename(columns={"geometry": "route_geometry"})
    return stops, shp


def _distance_row(row: pd.Series) -> float | None:
    """Compute the planar distance between a stop and its route geometry.

    Args:
        row: A pandas ``Series`` that must contain ``"geometry"`` (point)
            and ``"route_geometry"`` (line).

    Returns:
        The distance in **metres**, or ``None`` if either geometry is null.
    """
    geom: base.BaseGeometry | None = row["geometry"]
    route_geom: base.BaseGeometry | None = row["route_geometry"]
    return geom.distance(route_geom) if geom and route_geom else None


def identify_problem_stops(
    routes: DataFrame,
    stops: DataFrame,
    trips: DataFrame,
    stop_times: DataFrame,
    matched_routes: DataFrame,
    shp: gpd.GeoDataFrame,
    input_crs: str,
    projected_crs: str,
    output_crs: str,
    distance_ft: int,
    route_number_col: str,
) -> Tuple[gpd.GeoDataFrame, List[str], DataFrame]:
    """Locate stops that do not meet spatial QA criteria.

    Args:
        routes: ``routes.txt`` table.
        stops:  ``stops.txt`` table.
        trips:  ``trips.txt`` table.
        stop_times: ``stop_times.txt`` table.
        matched_routes: Subset of *routes* that exactly match the shapefile.
        shp: Route shapefile GeoDataFrame.
        input_crs: CRS of the raw GTFS coordinates.
        projected_crs: CRS used for distance measurement.
        output_crs: CRS for final outputs.
        distance_ft: Buffer allowance in feet.
        route_number_col: Column in *shp* that stores the route number.

    Returns:
        Tuple ``(problem_stops_gdf, offending_route_ids, geometry_lookup)`` where

        * **problem_stops_gdf** – GeoDataFrame of all flagged stops,
        * **offending_route_ids** – list of GTFS ``route_id`` values that failed,
        * **geometry_lookup** – helper table mapping each ``route_id`` to its
          shapefile geometry.
    """
    LOGGER.info("Identifying problem stops …")

    # --- matched routes / trips / stops
    matched_trips = trips[trips["route_id"].isin(matched_routes["route_id"])]
    matched_stop_times = stop_times[
        stop_times["trip_id"].isin(matched_trips["trip_id"])
    ]
    matched_stops = stops[stops["stop_id"].isin(matched_stop_times["stop_id"])]

    matched_stops_gdf = convert_stops_to_gdf(matched_stops, input_crs)
    matched_stops_gdf, shp = prepare_geometries(matched_stops_gdf, shp, projected_crs)

    geometry_lookup = matched_routes.merge(
        shp[[f"{route_number_col}_str", "route_geometry"]],
        on=f"{route_number_col}_str",
        how="left",
    )[["route_id", "route_short_name", "route_geometry"]].drop_duplicates()

    stop_route_pairs = (
        matched_stop_times[["stop_id", "trip_id"]]
        .merge(matched_trips[["trip_id", "route_id"]], on="trip_id")
        .drop_duplicates()
        .merge(
            matched_stops_gdf[["stop_id", "geometry", "stop_name"]],
            on="stop_id",
        )
        .merge(geometry_lookup, on="route_id")
    )

    # --- distance tests
    stop_route_pairs["distance_m"] = stop_route_pairs.apply(_distance_row, axis=1)
    stop_route_pairs["distance_ft"] = stop_route_pairs["distance_m"] * 3.28084
    stop_route_pairs["within_allowance"] = (
        stop_route_pairs["distance_ft"] <= distance_ft
    )

    flagged = stop_route_pairs.loc[~stop_route_pairs["within_allowance"]].copy()
    flagged["reason"] = f"Not within {distance_ft} ft"

    routes_with_flagged = flagged["route_id"].dropna().unique().tolist()

    # --- unmatched stops (no matching route at all)
    unmatched_route_ids = set(routes["route_id"]).difference(matched_routes["route_id"])
    unmatched_trips = trips[trips["route_id"].isin(unmatched_route_ids)]
    unmatched_stop_times = stop_times[
        stop_times["trip_id"].isin(unmatched_trips["trip_id"])
    ]
    unmatched_stops = stops[stops["stop_id"].isin(unmatched_stop_times["stop_id"])]

    unmatched_gdf = convert_stops_to_gdf(unmatched_stops, input_crs).to_crs(
        projected_crs
    )
    unmatched_gdf["reason"] = "No matching route"
    unmatched_gdf["distance_ft"] = pd.NA
    unmatched_gdf["route_id"] = pd.NA

    # --- route lists per stop (for context)
    route_per_stop = (
        stop_times.merge(trips[["trip_id", "route_id"]], on="trip_id")
        .merge(routes[["route_id", "route_short_name"]], on="route_id")
        .groupby("stop_id")["route_short_name"]
        .unique()
        .rename("routes_serving_stop")
        .apply(lambda names: ", ".join(map(str, names)))
        .reset_index()
    )

    flagged = flagged.merge(route_per_stop, on="stop_id", how="left")
    unmatched_gdf = unmatched_gdf.merge(route_per_stop, on="stop_id", how="left")

    # --- combine & convert to final GeoDataFrame
    problems = pd.concat(
        [
            flagged[
                [
                    "stop_id",
                    "stop_name",
                    "geometry",
                    "reason",
                    "distance_ft",
                    "routes_serving_stop",
                    "route_id",
                ]
            ],
            unmatched_gdf[
                [
                    "stop_id",
                    "stop_name",
                    "geometry",
                    "reason",
                    "distance_ft",
                    "routes_serving_stop",
                    "route_id",
                ]
            ],
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["stop_id", "route_id"])

    problems_gdf = gpd.GeoDataFrame(problems, geometry="geometry", crs=projected_crs)
    problems_gdf = problems_gdf.to_crs(output_crs)

    return problems_gdf, routes_with_flagged, geometry_lookup


# =============================================================================
# MAIN LOGIC
# =============================================================================


def main() -> None:
    """Run the end-to-end GTFS vs. shapefile QA workflow.

    Orchestrates data loading, merging, spatial checks, exports and optional
    per-route visualisations.  Results are written to *OUTPUT_DIR* as CSV,
    Excel, shapefile and JPEG files.  See the module-level docstring for the
    exact output list.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- data ingestion --------------------------------------------------
    routes, stops, trips, stop_times = load_gtfs_data(GTFS_DIR)
    shp = load_shapefile(SHAPEFILE_PATH, INPUT_CRS)

    # --- merging & comparison -------------------------------------------
    routes, shp = preprocess_data(routes, shp, ROUTE_NUMBER_COLUMN)
    comparison = merge_and_score(routes, shp, ROUTE_NUMBER_COLUMN, ROUTE_NAME_COLUMN)
    comparison = export_comparison(comparison, OUTPUT_DIR)

    matched_routes = comparison[comparison["short_name_exact_match"]]

    # --- stop QA ---------------------------------------------------------
    problems_gdf, flagged_routes, geometry_lookup = identify_problem_stops(
        routes,
        stops,
        trips,
        stop_times,
        matched_routes,
        shp,
        INPUT_CRS,
        PROJECTED_CRS,
        OUTPUT_CRS,
        DISTANCE_ALLOWANCE_FT,
        ROUTE_NUMBER_COLUMN,
    )

    # --- outputs ---------------------------------------------------------
    shp_path = OUTPUT_DIR / "problem_stops.shp"
    problems_gdf.to_file(shp_path)
    LOGGER.info("Problem stops written to %s", shp_path)

    # add buffer-flag column to comparison table
    if "route_id" in comparison.columns:
        comparison["has_stops_outside_buffer"] = comparison["route_id"].isin(
            flagged_routes
        )
        flags_path = OUTPUT_DIR / "gtfs_shp_comparison_with_flags.csv"
        comparison.to_csv(flags_path, index=False)
        LOGGER.info("Comparison (with flags) written to %s", flags_path)

    # Excel export
    problems_xlsx = problems_gdf.merge(
        geometry_lookup[["route_id", "route_short_name"]], on="route_id", how="left"
    ).loc[:, ["stop_id", "stop_name", "route_short_name", "distance_ft", "reason"]]
    xlsx_path = OUTPUT_DIR / "problem_stops.xlsx"
    problems_xlsx.to_excel(xlsx_path, index=False)
    LOGGER.info("Problem stops (Excel) written to %s", xlsx_path)

    # --- per-route visualisations ---------------------------------------
    if flagged_routes:
        LOGGER.info("Creating per-route warning plots …")
        for rid in flagged_routes:
            row = geometry_lookup.loc[geometry_lookup["route_id"] == rid]
            if row.empty or row["route_geometry"].isna().all():
                continue

            short_name = row["route_short_name"].iat[0]
            route_geom = row["route_geometry"].iat[0]

            route_gdf = gpd.GeoDataFrame(
                {"route_id": [rid], "route_short_name": [short_name]},
                geometry=[route_geom],
                crs=PROJECTED_CRS,
            ).to_crs(OUTPUT_CRS)

            stops_gdf = problems_gdf.loc[problems_gdf["route_id"] == rid]

            fig, ax = plt.subplots(figsize=(8, 8))
            route_gdf.plot(ax=ax, edgecolor="black", facecolor="none")
            stops_gdf.plot(ax=ax, markersize=6)
            ax.set_title(f"Route {short_name} — Problem Stops")

            img_path = OUTPUT_DIR / f"problem_stops_{short_name}_warning.jpeg"
            plt.savefig(img_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            LOGGER.warning(
                "Route '%s' has stops outside the %d-ft buffer (plot saved at %s)",
                short_name,
                DISTANCE_ALLOWANCE_FT,
                img_path,
            )
    else:
        LOGGER.info("All stops fall within the %d-ft buffer.", DISTANCE_ALLOWANCE_FT)


if __name__ == "__main__":
    main()
