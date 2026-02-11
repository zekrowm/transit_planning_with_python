"""GTFS skipped-stop flagger.

This module detects likely skipped-stop errors between overlapping routes in a
single GTFS feed using stop sequences only (no geometry).

At a high level, the script:

  - Loads stops.txt, trips.txt, and stop_times.txt from a GTFS directory.
  - For each (route_id, direction_id), selects the trip with the most distinct
    stops as the representative pattern.
  - Builds ordered logical stop sequences using a configurable stop key
    (stop_id or stop_code).
  - Treats a configurable set of base routes as references and compares each
    base (route_id, direction_id) against all other route/direction pairs.
  - Identifies segments bounded by shared stops where one route has a small
    number of interior stops that the other route does not serve, flagging
    these as potential skipped-stop issues.

Results are written to a CSV in OUTPUT_DIR for manual review. Configuration
options (GTFS paths, base routes, and tuning thresholds) are defined in the
Configuration section below.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, cast

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from shapely.geometry import LineString, Point
from shapely.ops import substring

# =============================================================================
# Configuration
# =============================================================================

# Directory containing a single GTFS feed:
#   - stops.txt
#   - trips.txt
#   - stop_times.txt
#   - (optionally shapes.txt, routes.txt if you extend later)
GTFS_DIR = Path("/path/to/gtfs_directory")

# Output directory and filenames.
# The main CSV will contain one row per detected skipped-stop segment.
OUTPUT_DIR = Path("./output/gtfs_skipped_stop_flagger")
OUTPUT_FILENAME = "skipped_stop_segments.csv"

# Optional: a second CSV for the barebones 2-1-2 diagnostic, if you keep it.
BAREBONES_2_1_2_FILENAME = "barebones_2_1_2_patterns.csv"

# Subfolder for PNG plots.
PLOT_DIR = OUTPUT_DIR / "segment_plots"

# CRS settings.
GTFS_CRS = "EPSG:4326"
PROJECTED_CRS = "EPSG:26918"  # NAD83 / UTM 18N (Mid-Atlantic US).

# Use stop_code vs stop_id as the logical key for stops.
USE_STOP_CODE = True
STOP_KEY_FIELD = "stop_code" if USE_STOP_CODE else "stop_id"

# Route whitelist:
#   - Names (short_name or long_name) that should be ignored entirely
#     (e.g., express patterns that intentionally skip stops).
#   - Route IDs that should be ignored explicitly.
ROUTE_NAME_WHITELIST: Set[str] = set()
ROUTE_NAME_WHITELIST_FIELD = "route_short_name"  # or "route_long_name"
ROUTE_ID_WHITELIST: Set[str] = set()

# Target routes: leave empty to analyze all routes, or specify a subset.
# Example: {"101"} or {"101", "202"}.
TARGET_ROUTE_IDS: Set[str] = set()

# Minimum number of shared logical stops between two routes to consider the pair.
MIN_SHARED_STOPS_FOR_PAIR = 2

# Minimum index distance (in stops) between boundary stops to treat as a segment.
# For example, 2 means there must be at least one interior stop on at least one route.
MIN_SEGMENT_SPAN_STOPS = 2

# Maximum Hausdorff distance between representative shapes (meters) for two routes
# to be considered "same corridor". If None, no shape-based filtering is applied.
MAX_SHAPE_HAUSDORFF_M: Optional[float] = 80.0

# Maximum perpendicular distance (meters) from a stop to a route's shape to consider
# the stop as lying on that route's path.
MAX_STOP_TO_SHAPE_M = 30.0

# Padding (meters) around the segment extent along each shape for the "within path"
# test. This guards against small shape–stop misalignments.
SEGMENT_MEASURE_PADDING_M = 50.0

# Buffer distance (meters) used for overview plots when no flagged stops exist.
ROUTE_OVERVIEW_BUFFER_M = 30.0

# =============================================================================
# Plotting constants
# =============================================================================

BUFFER_ZORDER = 0
LINE_ZORDER = 1
STOP_ZORDER = 3
TEXT_ZORDER = 4
GRID_ZORDER = 0.5  # optional, to keep grid under lines but above buffer

# =============================================================================
# Types
# =============================================================================

RouteKey = Tuple[str, str]  # (route_id, direction_id)


@dataclass
class GTFSContext:
    """Container for prepared GTFS-derived objects used in QA and plotting."""

    stops_df: pd.DataFrame
    trips_df: pd.DataFrame
    stop_times_df: pd.DataFrame
    shapes_df: pd.DataFrame
    routes_df: pd.DataFrame

    stops_gdf_geo: gpd.GeoDataFrame
    stops_gdf_proj: gpd.GeoDataFrame
    stop_key_lookup: Dict[str, str]
    stop_names: Dict[str, str]

    route_sequences: Dict[RouteKey, List[str]]
    route_shapes_geo: Dict[RouteKey, LineString]
    route_shapes_proj: Dict[RouteKey, LineString]

    all_route_keys: List[RouteKey]
    base_route_keys: List[RouteKey]
    route_id_whitelist: Set[str]


# =============================================================================
# GTFS helpers
# =============================================================================


def choose_representative_trip_ids_max_stops(
    trips_df: pd.DataFrame,
    stop_times_df: pd.DataFrame,
) -> Dict[RouteKey, str]:
    """Choose a representative trip_id per (route_id, direction_id) with most stops.

    For each (route_id, direction_id), the representative trip is the one that
    serves the largest number of distinct stop_ids.

    Args:
        trips_df: DataFrame from trips.txt.
        stop_times_df: DataFrame from stop_times.txt.

    Returns:
        Mapping from (route_id, direction_id) -> representative trip_id.
    """
    trips = trips_df[["route_id", "direction_id", "trip_id"]].copy()
    trips["route_id"] = trips["route_id"].astype(str)
    trips["direction_id"] = trips["direction_id"].astype(str)
    trips["trip_id"] = trips["trip_id"].astype(str)

    st = stop_times_df[["trip_id", "stop_id"]].copy()
    st["trip_id"] = st["trip_id"].astype(str)
    st["stop_id"] = st["stop_id"].astype(str)

    merged = st.merge(trips, on="trip_id", how="inner")

    # Count distinct stops per trip within each (route, direction).
    counts = (
        merged.groupby(["route_id", "direction_id", "trip_id"])["stop_id"]
        .nunique()
        .reset_index(name="n_stops")
    )

    # Sort so that within each (route, direction) the trip with the most stops
    # appears first.
    counts = counts.sort_values(
        ["route_id", "direction_id", "n_stops"],
        ascending=[True, True, False],
    )

    # Take the top trip per (route, direction).
    reps = counts.drop_duplicates(subset=["route_id", "direction_id"])

    result: Dict[RouteKey, str] = {}
    for _, row in reps.iterrows():
        route_id = str(row["route_id"])
        direction_id = str(row["direction_id"])
        trip_id = str(row["trip_id"])
        result[(route_id, direction_id)] = trip_id

    return result


def normalize_direction_id(series: pd.Series) -> pd.Series:
    """Normalize GTFS direction_id to string labels.

    Args:
        series: Series containing direction_id values (typically 0/1 or NaN).

    Returns:
        Series of strings representing direction_id values.
    """
    return series.astype("Int64").astype(str)


def load_gtfs_tables(gtfs_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load required GTFS tables from the specified directory.

    Args:
        gtfs_dir: Directory containing GTFS CSV files.

    Returns:
        Dictionary mapping table name to DataFrame.

    Raises:
        FileNotFoundError: If required GTFS files are missing.
        ValueError: If required columns are missing.
    """
    required_files = {
        "stops": "stops.txt",
        "trips": "trips.txt",
        "stop_times": "stop_times.txt",
        "shapes": "shapes.txt",
        "routes": "routes.txt",
    }

    tables: Dict[str, pd.DataFrame] = {}
    for key, filename in required_files.items():
        path = gtfs_dir / filename
        if not path.exists():
            msg = f"Required GTFS file not found: {path}"
            raise FileNotFoundError(msg)
        tables[key] = pd.read_csv(path)

    trips = tables["trips"].copy()
    if "direction_id" not in trips.columns:
        msg = "trips.txt must contain a 'direction_id' column."
        raise ValueError(msg)

    trips["direction_id"] = normalize_direction_id(trips["direction_id"])
    trips["route_id"] = trips["route_id"].astype(str)
    trips["trip_id"] = trips["trip_id"].astype(str)
    tables["trips"] = trips

    stop_times = tables["stop_times"].copy()
    stop_times["trip_id"] = stop_times["trip_id"].astype(str)
    stop_times["stop_id"] = stop_times["stop_id"].astype(str)
    tables["stop_times"] = stop_times

    stops = tables["stops"].copy()
    stops["stop_id"] = stops["stop_id"].astype(str)
    tables["stops"] = stops

    return tables


def build_shapes_gdf(shapes_df: pd.DataFrame, crs: str) -> gpd.GeoDataFrame:
    """Build a LineString GeoDataFrame from GTFS shapes.txt.

    Args:
        shapes_df: DataFrame containing shapes.txt.
        crs: Coordinate reference system for the shapes (typically EPSG:4326).

    Returns:
        GeoDataFrame with one LineString per shape_id.

    Raises:
        ValueError: If required columns are missing.
    """
    required_cols = {
        "shape_id",
        "shape_pt_lat",
        "shape_pt_lon",
        "shape_pt_sequence",
    }
    missing = required_cols - set(shapes_df.columns)
    if missing:
        msg = f"shapes.txt is missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    shapes_df = shapes_df.copy()
    shapes_df = shapes_df.sort_values(["shape_id", "shape_pt_sequence"])

    grouped = shapes_df.groupby("shape_id", sort=False)

    records: List[Dict[str, object]] = []
    for shape_id, group in grouped:
        points: List[Point] = [
            Point(lon, lat)
            for lon, lat in zip(
                group["shape_pt_lon"],
                group["shape_pt_lat"],
                strict=True,
            )
        ]
        if len(points) < 2:
            continue
        records.append({"shape_id": str(shape_id), "geometry": LineString(points)})

    shapes_gdf = cast("Any", gpd.GeoDataFrame)(data=records, crs=crs)
    shapes_gdf.set_index("shape_id", inplace=True)
    return shapes_gdf


def build_stops_gdf(
    stops_df: pd.DataFrame,
    crs: str,
    stop_key_field: str,
) -> gpd.GeoDataFrame:
    """Build a Point GeoDataFrame from GTFS stops.txt.

    The GeoDataFrame index is set to the chosen stop key field
    (stop_id or stop_code), as configured via stop_key_field.

    Args:
        stops_df: DataFrame containing stops.txt.
        crs: Coordinate reference system for stop locations (typically EPSG:4326).
        stop_key_field: Column name to use as the logical stop key, e.g.
            "stop_id" or "stop_code".

    Returns:
        GeoDataFrame with one Point per stop key.

    Raises:
        ValueError: If required columns are missing.
    """
    required_cols = {"stop_id", "stop_lat", "stop_lon"}
    missing = required_cols - set(stops_df.columns)
    if missing:
        msg = f"stops.txt is missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    if stop_key_field not in stops_df.columns:
        msg = (
            "stops.txt does not contain the configured stop key field "
            f"'{stop_key_field}'. Available columns: {sorted(stops_df.columns)}"
        )
        raise ValueError(msg)

    stops_df = stops_df.copy()
    stops_df["stop_id"] = stops_df["stop_id"].astype(str)
    stops_df[stop_key_field] = stops_df[stop_key_field].astype(str)

    geometry = gpd.points_from_xy(stops_df["stop_lon"], stops_df["stop_lat"])
    stops_gdf = cast("Any", gpd.GeoDataFrame)(data=stops_df, geometry=geometry, crs=crs)
    stops_gdf.set_index(stop_key_field, inplace=True)
    return stops_gdf


def select_representative_shapes(trips_df: pd.DataFrame) -> pd.DataFrame:
    """Select a representative shape_id for each (route_id, direction_id).

    The representative shape_id is chosen as the one with the highest trip count
    for that route/direction.

    Args:
        trips_df: DataFrame from trips.txt, including route_id, shape_id,
            and normalized direction_id.

    Returns:
        DataFrame with columns: route_id, direction_id, shape_id.

    Raises:
        ValueError: If required columns are missing.
    """
    required_cols = {"route_id", "shape_id", "direction_id"}
    missing = required_cols - set(trips_df.columns)
    if missing:
        msg = f"trips.txt is missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    df = trips_df.copy()
    df["route_id"] = df["route_id"].astype(str)
    df["shape_id"] = df["shape_id"].astype(str)
    df["direction_id"] = df["direction_id"].astype(str)

    counts = (
        df.groupby(["route_id", "direction_id", "shape_id"]).size().reset_index(name="trip_count")
    )
    counts = counts.sort_values("trip_count", ascending=False)
    reps = counts.drop_duplicates(subset=["route_id", "direction_id"])
    reps = reps[["route_id", "direction_id", "shape_id"]].reset_index(drop=True)
    return reps


def build_route_shapes_from_reps(
    reps: pd.DataFrame,
    shapes_gdf: gpd.GeoDataFrame,
) -> Dict[RouteKey, LineString]:
    """Build LineString geometry per (route_id, direction_id) using reps.

    Args:
        reps: DataFrame with route_id, direction_id, shape_id.
        shapes_gdf: GeoDataFrame of shapes (index = shape_id) in a given CRS.

    Returns:
        Mapping from (route_id, direction_id) to LineString geometry.
    """
    result: Dict[RouteKey, LineString] = {}

    for _, row in reps.iterrows():
        route_id = str(row["route_id"])
        direction_id = str(row["direction_id"])
        shape_id = str(row["shape_id"])

        if shape_id not in shapes_gdf.index:
            continue

        geom = shapes_gdf.loc[shape_id, "geometry"]
        if not isinstance(geom, LineString):
            continue

        result[(route_id, direction_id)] = geom

    return result


def build_route_id_whitelist(
    routes_df: pd.DataFrame,
    route_name_whitelist: Set[str],
    route_name_field: str,
    route_id_whitelist: Set[str],
) -> Set[str]:
    """Build a set of route_ids to ignore based on names and explicit IDs.

    Args:
        routes_df: DataFrame from routes.txt.
        route_name_whitelist: Set of route names (e.g., short names) to ignore.
        route_name_field: Column in routes.txt used for matching route names.
        route_id_whitelist: Explicit set of route_ids to ignore.

    Returns:
        Set of route_ids to ignore.
    """
    result: Set[str] = {str(r) for r in route_id_whitelist}

    if route_name_whitelist:
        if route_name_field not in routes_df.columns:
            logging.warning(
                "Warning: routes.txt does not contain the route name field "
                "'%s'. Route name whitelist will be ignored.",
                route_name_field,
            )
            return result

        df = routes_df.copy()
        df["route_id"] = df["route_id"].astype(str)
        df[route_name_field] = df[route_name_field].astype(str)

        mask = df[route_name_field].isin(route_name_whitelist)
        matched_ids = df.loc[mask, "route_id"].astype(str).tolist()
        result.update(matched_ids)

    return result


def build_stop_key_lookup(
    stops_df: pd.DataFrame,
    stop_key_field: str,
) -> Dict[str, str]:
    """Build a mapping from GTFS stop_id to logical stop key.

    Args:
        stops_df: DataFrame from stops.txt.
        stop_key_field: Logical stop key field, e.g., "stop_id" or "stop_code".

    Returns:
        Dictionary mapping stop_id -> logical stop key as string.
    """
    if stop_key_field not in stops_df.columns:
        msg = f"stops.txt is missing the stop key field '{stop_key_field}'."
        raise ValueError(msg)

    df = stops_df[["stop_id", stop_key_field]].drop_duplicates("stop_id").copy()
    df["stop_id"] = df["stop_id"].astype(str)
    df[stop_key_field] = df[stop_key_field].astype(str)

    return dict(zip(df["stop_id"], df[stop_key_field], strict=True))


def build_stop_names_lookup(
    stops_df: pd.DataFrame,
    stop_key_field: str,
) -> Dict[str, str]:
    """Build a mapping from logical stop key to stop_name.

    Args:
        stops_df: DataFrame from stops.txt.
        stop_key_field: Logical stop key field, e.g., "stop_id" or "stop_code".

    Returns:
        Dictionary mapping logical stop key -> stop_name.
    """
    if "stop_name" not in stops_df.columns:
        msg = "stops.txt is missing the 'stop_name' column."
        raise ValueError(msg)

    df = stops_df[[stop_key_field, "stop_name"]].copy()
    df = df[pd.notna(df[stop_key_field])]

    df[stop_key_field] = df[stop_key_field].astype(str)
    df["stop_name"] = df["stop_name"].astype(str)

    return dict(zip(df[stop_key_field], df["stop_name"], strict=True))


def choose_representative_trip_ids(
    trips_df: pd.DataFrame,
    reps: pd.DataFrame,
) -> Dict[RouteKey, str]:
    """Choose one representative trip_id per (route_id, direction_id).

    The representative trip is chosen among trips using the representative
    shape_id for that route/direction. If none exist, fall back to any trip
    for that route/direction.

    Args:
        trips_df: DataFrame from trips.txt.
        reps: DataFrame with columns route_id, direction_id, shape_id.

    Returns:
        Mapping from (route_id, direction_id) -> representative trip_id.
    """
    df = trips_df.copy()
    df["route_id"] = df["route_id"].astype(str)
    df["direction_id"] = df["direction_id"].astype(str)
    df["shape_id"] = df["shape_id"].astype(str)
    df["trip_id"] = df["trip_id"].astype(str)

    result: Dict[RouteKey, str] = {}

    for _, row in reps.iterrows():
        route_id = str(row["route_id"])
        direction_id = str(row["direction_id"])
        shape_id = str(row["shape_id"])

        mask = (
            (df["route_id"] == route_id)
            & (df["direction_id"] == direction_id)
            & (df["shape_id"] == shape_id)
        )
        subset = df.loc[mask, "trip_id"]

        if subset.empty:
            mask2 = (df["route_id"] == route_id) & (df["direction_id"] == direction_id)
            subset = df.loc[mask2, "trip_id"]

        if subset.empty:
            continue

        rep_trip_id = str(subset.iloc[0])
        result[(route_id, direction_id)] = rep_trip_id

    return result


def build_route_sequences(
    stop_times_df: pd.DataFrame,
    stop_key_lookup: Mapping[str, str],
    rep_trip_ids: Mapping[RouteKey, str],
) -> Dict[RouteKey, List[str]]:
    """Build ordered logical stop sequences for each representative trip.

    Args:
        stop_times_df: DataFrame from stop_times.txt.
        stop_key_lookup: Mapping from stop_id to logical stop key.
        rep_trip_ids: Mapping from (route_id, direction_id) to trip_id.

    Returns:
        Mapping from (route_id, direction_id) to list of logical stop keys.
    """
    df = stop_times_df[["trip_id", "stop_id", "stop_sequence"]].copy()
    df["trip_id"] = df["trip_id"].astype(str)
    df["stop_id"] = df["stop_id"].astype(str)

    sequences: Dict[RouteKey, List[str]] = {}

    valid_trip_ids: Set[str] = {str(tid) for tid in rep_trip_ids.values()}
    df = df[df["trip_id"].isin(valid_trip_ids)]

    for key, trip_id in rep_trip_ids.items():
        trip_id_str = str(trip_id)
        sub = df[df["trip_id"] == trip_id_str].copy()
        if sub.empty:
            continue

        sub = sub.sort_values("stop_sequence")

        seq: List[str] = []
        last_key: Optional[str] = None
        for _, row in sub.iterrows():
            stop_id = str(row["stop_id"])
            stop_key = stop_key_lookup.get(stop_id)
            if stop_key is None:
                continue
            if stop_key == last_key:
                continue
            seq.append(stop_key)
            last_key = stop_key

        if len(seq) >= 2:
            sequences[key] = seq

    return sequences


def hausdorff_distance_safe(
    geom_a: Optional[LineString],
    geom_b: Optional[LineString],
) -> Optional[float]:
    """Compute Hausdorff distance between two geometries, if both exist.

    Args:
        geom_a: First LineString geometry or None.
        geom_b: Second LineString geometry or None.

    Returns:
        Hausdorff distance in geometry units (meters if projected),
        or None if either geometry is missing.
    """
    if geom_a is None or geom_b is None:
        return None
    return float(geom_a.hausdorff_distance(geom_b))


def shares_only_terminal_stops(
    base_seq: Sequence[str],
    other_seq: Sequence[str],
) -> bool:
    """Return True if all shared stops are terminals in *both* sequences.

    This treats a route pair as "overlapping only at terminals" if:
      - There is at least one shared stop, and
      - For every shared stop, it is either the first or last stop in BOTH
        sequences (no shared interior stop in either route).

    Args:
        base_seq: Ordered logical stops for the base route/direction.
        other_seq: Ordered logical stops for the other route/direction.

    Returns:
        True if they only share terminal stops, False otherwise.
    """
    shared = set(base_seq).intersection(other_seq)
    if not shared:
        return False

    base_pos: Dict[str, int] = {s: i for i, s in enumerate(base_seq)}
    other_pos: Dict[str, int] = {s: i for i, s in enumerate(other_seq)}

    base_last = len(base_seq) - 1
    other_last = len(other_seq) - 1

    for stop in shared:
        i = base_pos.get(stop)
        j = other_pos.get(stop)
        if i is None or j is None:
            # Shouldn't happen, but be defensive.
            continue

        is_base_terminal = i in (0, base_last)
        is_other_terminal = j in (0, other_last)

        # If this shared stop is interior in either sequence, they share
        # a real segment and we should NOT exclude this pair.
        if not (is_base_terminal and is_other_terminal):
            return False

    return True


def segment_hausdorff_distance(
    base_key: RouteKey,
    other_key: RouteKey,
    start_key: str,
    end_key: str,
    shapes_proj: Mapping[RouteKey, LineString],
    stops_gdf_proj: gpd.GeoDataFrame,
    padding_m: float,
) -> Optional[float]:
    """Compute Hausdorff distance between the route shapes over a segment.

    The segment is defined by the stop pair (start_key, end_key). For each
    route, we:

      - Project the start/end stop points onto the route's projected shape
        (distance along the line).
      - Extract the substring between those measures, expanded by `padding_m`
        on both sides, clipped to [0, length].
      - Compute Hausdorff distance between the two substrings.

    Args:
        base_key: (route_id, direction_id) for the base route.
        other_key: (route_id, direction_id) for the other route.
        start_key: Logical stop key for the segment start.
        end_key: Logical stop key for the segment end.
        shapes_proj: Mapping from route key to projected LineString.
        stops_gdf_proj: Stops GeoDataFrame in PROJECTED_CRS, indexed by stop key.
        padding_m: Padding (meters) to extend the segment on each side along
            the shape.

    Returns:
        Hausdorff distance in meters between the segment substrings, or None if
        shapes or stop geometries are missing or degenerate.
    """
    base_shape = shapes_proj.get(base_key)
    other_shape = shapes_proj.get(other_key)
    if base_shape is None or other_shape is None:
        return None

    if start_key not in stops_gdf_proj.index or end_key not in stops_gdf_proj.index:
        return None

    start_pt = stops_gdf_proj.loc[start_key, "geometry"]
    end_pt = stops_gdf_proj.loc[end_key, "geometry"]

    # Measures along base shape.
    m0_base = float(base_shape.project(start_pt))
    m1_base = float(base_shape.project(end_pt))
    if m0_base == m1_base:
        return None
    if m0_base > m1_base:
        m0_base, m1_base = m1_base, m0_base
    m0_base = max(0.0, m0_base - padding_m)
    m1_base = min(base_shape.length, m1_base + padding_m)

    # Measures along other shape.
    m0_other = float(other_shape.project(start_pt))
    m1_other = float(other_shape.project(end_pt))
    if m0_other == m1_other:
        return None
    if m0_other > m1_other:
        m0_other, m1_other = m1_other, m0_other
    m0_other = max(0.0, m0_other - padding_m)
    m1_other = min(other_shape.length, m1_other + padding_m)

    # Extract substrings.
    try:
        seg_base = substring(base_shape, m0_base, m1_base)
        seg_other = substring(other_shape, m0_other, m1_other)
    except Exception:  # defensive: geometries can sometimes be weird
        return None

    if seg_base.is_empty or seg_other.is_empty:
        return None

    return float(seg_base.hausdorff_distance(seg_other))


# =============================================================================
# Segment alignment and comparison
# =============================================================================


def find_aligned_common_stops(
    base_seq: Sequence[str],
    other_seq: Sequence[str],
) -> List[Tuple[int, int]]:
    """Find common stops with consistent direction between two sequences.

    The algorithm:
      - For each stop in base_seq, find its first occurrence in other_seq.
      - Keep only those matches.
      - Enforce strictly increasing indices in other_seq to preserve direction.

    Args:
        base_seq: Ordered list of stop keys for the base route.
        other_seq: Ordered list of stop keys for the other route.

    Returns:
        List of (base_index, other_index) for aligned common stops in order.
    """
    other_pos: Dict[str, int] = {}
    for idx, s in enumerate(other_seq):
        if s not in other_pos:
            other_pos[s] = idx

    raw_pairs: List[Tuple[int, int]] = []
    for i, stop in enumerate(base_seq):
        j = other_pos.get(stop)
        if j is not None:
            raw_pairs.append((i, j))

    aligned: List[Tuple[int, int]] = []
    last_other_idx = -1
    for i, j in raw_pairs:
        if j > last_other_idx:
            aligned.append((i, j))
            last_other_idx = j

    return aligned


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    """Return unique items in the order of first appearance.

    Args:
        items: Iterable of strings.

    Returns:
        List of unique strings in original order.
    """
    seen: Set[str] = set()
    result: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def compare_segments_for_route_pair(
    base_key: RouteKey,
    other_key: RouteKey,
    sequences: Mapping[RouteKey, Sequence[str]],
    stop_names: Mapping[str, str],
    shapes_proj: Mapping[RouteKey, LineString],
    stops_gdf_proj: gpd.GeoDataFrame,
    max_shape_hausdorff_m: Optional[float],
    max_stop_to_shape_m: float,  # unused currently, kept for signature compat
    segment_measure_padding_m: float,
) -> List[Dict[str, object]]:
    """Compare shared segments between two routes using sequences + shapes.

    Design:
      - Use logical stop sequences to identify candidate segments:
          * consecutive shared stops (in the same order on both routes)
            define a segment boundary.
      - For each segment, optionally apply a geometry gating step:
          * Build route-substrings between start/end stops on each shape.
          * If the segment Hausdorff distance exceeds max_shape_hausdorff_m,
            treat this as a divergent corridor and skip the segment.
      - For segments that pass gating, compare interior subsequences:
          * base_interior = stops between boundaries on the base.
          * other_interior = stops between boundaries on the other.
          * Any stop present only on one side's interior is flagged.

    This generalizes the 2-1-2 pattern to arbitrary interior lengths.

    Args:
        base_key: (route_id, direction_id) for the base route.
        other_key: (route_id, direction_id) for the other route.
        sequences: Mapping from route key to ordered stop key list.
        stop_names: Mapping from stop key to stop_name.
        shapes_proj: Mapping from route key to projected LineString.
        stops_gdf_proj: Stops GeoDataFrame (PROJECTED_CRS), indexed by stop key.
        max_shape_hausdorff_m: Maximum allowed Hausdorff distance (meters)
            between the two route substrings for a segment to be considered
            “same corridor”. If None, geometry gating is disabled.
        max_stop_to_shape_m: Unused (kept for API compatibility).
        segment_measure_padding_m: Padding (meters) to extend the segment on
            both sides along each shape when computing the substring.

    Returns:
        List of dictionaries describing segment-level stop mismatches.
    """
    base_seq = sequences.get(base_key)
    other_seq = sequences.get(other_key)
    if base_seq is None or other_seq is None:
        return []

    if not base_seq or not other_seq:
        return []

    # Shared stops must exist at all.
    shared_stops = set(base_seq).intersection(other_seq)
    if len(shared_stops) < MIN_SHARED_STOPS_FOR_PAIR:
        return []

    # Map each stop in the other sequence to its first index.
    other_pos: Dict[str, int] = {}
    for idx, s in enumerate(other_seq):
        if s not in other_pos:
            other_pos[s] = idx

    # Shared indices along the base route in base order.
    base_shared_idx: List[int] = [i for i, s in enumerate(base_seq) if s in other_pos]
    if len(base_shared_idx) < 2:
        return []

    results: List[Dict[str, object]] = []
    base_route_id, base_dir = base_key
    other_route_id, other_dir = other_key

    for i0, i1 in zip(base_shared_idx[:-1], base_shared_idx[1:], strict=False):
        # Enforce a minimum span in index space to avoid trivial segments.
        if i1 - i0 < MIN_SEGMENT_SPAN_STOPS:
            continue

        start_key = base_seq[i0]
        end_key = base_seq[i1]

        # The boundaries must appear in the same order in the other sequence.
        j0 = other_pos.get(start_key)
        j1 = other_pos.get(end_key)
        if j0 is None or j1 is None or j0 >= j1:
            continue

        # Optional segment-level geometry gating.
        if max_shape_hausdorff_m is not None:
            seg_hd = segment_hausdorff_distance(
                base_key=base_key,
                other_key=other_key,
                start_key=start_key,
                end_key=end_key,
                shapes_proj=shapes_proj,
                stops_gdf_proj=stops_gdf_proj,
                padding_m=segment_measure_padding_m,
            )
            # If we got a valid distance and it's too large, treat this as a
            # divergent corridor and skip the segment.
            if seg_hd is not None and seg_hd > max_shape_hausdorff_m:
                continue

        # Interior subsequences between the boundary stops.
        base_interior = list(base_seq[i0 + 1 : i1])
        other_interior = list(other_seq[j0 + 1 : j1])

        # Deduplicate while preserving order.
        def _unique_preserve_order(seq: Sequence[str]) -> List[str]:
            seen_local: Set[str] = set()
            out: List[str] = []
            for x in seq:
                if x in seen_local:
                    continue
                seen_local.add(x)
                out.append(x)
            return out

        base_interior_u = _unique_preserve_order(base_interior)
        other_interior_u = _unique_preserve_order(other_interior)

        # If both interiors are empty, nothing to compare.
        if not base_interior_u and not other_interior_u:
            continue

        set_base_int = set(base_interior_u)
        set_other_int = set(other_interior_u)

        stops_only_on_base = [s for s in base_interior_u if s not in set_other_int]
        stops_only_on_other = [s for s in other_interior_u if s not in set_base_int]

        # If there are no unique stops, this segment is consistent.
        if not stops_only_on_base and not stops_only_on_other:
            continue

        results.append(
            {
                "base_route_id": base_route_id,
                "base_direction_id": base_dir,
                "other_route_id": other_route_id,
                "other_direction_id": other_dir,
                # Geometry-related field can be filled in later if you want to
                # record segment_hd; for now, keep schema stable.
                "shape_hausdorff_distance_m": None,
                "segment_start_stop_key": start_key,
                "segment_start_stop_name": stop_names.get(start_key, ""),
                "segment_end_stop_key": end_key,
                "segment_end_stop_name": stop_names.get(end_key, ""),
                "base_segment_stop_keys": ";".join(base_interior_u),
                "other_segment_stop_keys": ";".join(other_interior_u),
                "stops_only_on_base": ";".join(stops_only_on_base),
                "stops_only_on_base_names": ";".join(
                    stop_names.get(s, "") for s in stops_only_on_base
                ),
                "stops_only_on_other": ";".join(stops_only_on_other),
                "stops_only_on_other_names": ";".join(
                    stop_names.get(s, "") for s in stops_only_on_other
                ),
            }
        )

    return results


# =============================================================================
# Plotting helpers
# =============================================================================


def _find_segment_indices(
    seq: Sequence[str],
    start_key: str,
    end_key: str,
) -> Tuple[int, int]:
    """Find index span [i0, i1] for a segment between two shared stops.

    Uses the first occurrence of start_key, and the first occurrence of end_key
    after that.

    Args:
        seq: Ordered stop key sequence.
        start_key: Start stop key.
        end_key: End stop key.

    Returns:
        Tuple of (start_index, end_index).

    Raises:
        KeyError: If start or end stop are not found in the expected order.
    """
    try:
        i0 = seq.index(start_key)
    except ValueError as exc:
        msg = f"Start stop key {start_key} not in sequence."
        raise KeyError(msg) from exc

    try:
        i1 = seq.index(end_key, i0 + 1)
    except ValueError as exc:
        msg = f"End stop key {end_key} not found in sequence after start ({start_key})."
        raise KeyError(msg) from exc

    return i0, i1


def _parse_semicolon_list(value: str) -> List[str]:
    """Parse a semicolon-separated string into a list of non-empty tokens.

    Args:
        value: Semicolon-separated string.

    Returns:
        List of non-empty tokens. Empty/None input yields an empty list.
    """
    if not isinstance(value, str) or not value:
        return []
    return [token for token in value.split(";") if token]


def plot_mismatch_segment(
    row: pd.Series,
    route_sequences: Mapping[RouteKey, Sequence[str]],
    route_shapes_geo: Mapping[RouteKey, LineString],
    stops_gdf_geo: gpd.GeoDataFrame,
    stop_names: Mapping[str, str],
) -> Optional[Path]:
    """Create a PNG plot for one mismatched segment.

    Args:
        row: Row from the mismatch DataFrame.
        route_sequences: Mapping from (route_id, direction_id) to stop sequences.
        route_shapes_geo: Mapping from (route_id, direction_id) to LineString
            geometries in GTFS_CRS for plotting.
        stops_gdf_geo: Stops GeoDataFrame (GTFS_CRS), indexed by logical stop key.
        stop_names: Mapping from logical stop key to stop_name.

    Returns:
        Path to the saved PNG file, or None if plotting was not possible.
    """
    base_route_id = str(row["base_route_id"])
    base_dir = str(row["base_direction_id"])
    other_route_id = str(row["other_route_id"])
    other_dir = str(row["other_direction_id"])
    start_key = str(row["segment_start_stop_key"])
    end_key = str(row["segment_end_stop_key"])

    base_key: RouteKey = (base_route_id, base_dir)
    other_key: RouteKey = (other_route_id, other_dir)

    base_seq = route_sequences.get(base_key)
    other_seq = route_sequences.get(other_key)
    base_geom = route_shapes_geo.get(base_key)
    other_geom = route_shapes_geo.get(other_key)

    if base_seq is None or other_seq is None:
        logging.info("Skipping plot: missing sequences for %s or %s.", base_key, other_key)
        return None
    if base_geom is None or other_geom is None:
        logging.info("Skipping plot: missing shapes for %s or %s.", base_key, other_key)
        return None

    try:
        i0, i1 = _find_segment_indices(base_seq, start_key, end_key)
        j0, j1 = _find_segment_indices(other_seq, start_key, end_key)
    except KeyError as exc:
        logging.info("Skipping plot for %s vs %s: %s", base_key, other_key, exc)
        return None

    base_segment_keys = base_seq[i0 : i1 + 1]
    other_segment_keys = other_seq[j0 : j1 + 1]

    unique_base = set(_parse_semicolon_list(row.get("stops_only_on_base", "")))
    unique_other = set(_parse_semicolon_list(row.get("stops_only_on_other", "")))

    base_interior = set(base_segment_keys[1:-1])
    other_interior = set(other_segment_keys[1:-1])

    shared_interior = base_interior.intersection(other_interior) - unique_base - unique_other

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot route shapes (under stops).
    x_b, y_b = base_geom.xy
    x_o, y_o = other_geom.xy
    ax.plot(
        x_b,
        y_b,
        label=f"{base_route_id} dir {base_dir}",
        linewidth=2,
        zorder=LINE_ZORDER,
    )
    ax.plot(
        x_o,
        y_o,
        label=f"{other_route_id} dir {other_dir}",
        linestyle="--",
        linewidth=2,
        zorder=LINE_ZORDER,
    )

    def _scatter_keys(
        keys: Iterable[str],
        marker: str,
        size: float,
        label: str,
    ) -> None:
        xs: List[float] = []
        ys: List[float] = []
        for key in keys:
            if key not in stops_gdf_geo.index:
                continue
            pt = stops_gdf_geo.loc[key, "geometry"]
            xs.append(pt.x)
            ys.append(pt.y)
        if xs:
            ax.scatter(
                xs,
                ys,
                marker=marker,
                s=size,
                label=label,
                zorder=STOP_ZORDER,
            )

    # Start/end stops.
    _scatter_keys([start_key, end_key], marker="o", size=40, label="segment boundary")

    # Shared interior stops.
    _scatter_keys(shared_interior, marker=".", size=25, label="shared interior")

    # Unique to base/other.
    _scatter_keys(unique_base, marker="^", size=45, label="only on base route")
    _scatter_keys(unique_other, marker="v", size=45, label="only on other route")

    # Annotate unique stops (on top of markers).
    for key in sorted(unique_base.union(unique_other)):
        if key not in stops_gdf_geo.index:
            continue
        pt = stops_gdf_geo.loc[key, "geometry"]
        label = stop_names.get(key, key)
        ax.annotate(
            label,
            (pt.x, pt.y),
            textcoords="offset points",
            xytext=(2, 2),
            fontsize=7,
            zorder=TEXT_ZORDER,
        )

    title = (
        f"{base_route_id} dir {base_dir} vs {other_route_id} dir {other_dir}\n"
        f"{start_key} ({stop_names.get(start_key, '')}) -> "
        f"{end_key} ({stop_names.get(end_key, '')})"
    )
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linewidth=0.3, zorder=GRID_ZORDER)

    safe_start = start_key.replace(" ", "_")
    safe_end = end_key.replace(" ", "_")
    filename = (
        f"seg_{base_route_id}_{base_dir}_vs_{other_route_id}_{other_dir}_"
        f"{safe_start}_{safe_end}.png"
    )
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOT_DIR / filename

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logging.info("Saved plot: %s", out_path)
    return out_path


def plot_route_pair_overview(
    base_key: RouteKey,
    other_key: RouteKey,
    ctx: GTFSContext,
    buffer_m: float,
) -> Optional[Path]:
    """Plot an overview of a route pair and an additional side-by-side view.

    The combined overview plot shows:
      - Base and comparison route shapes with buffers.
      - Shared stops between the two sequences.
      - Stops only on the base or only on the other sequence, each split into:
          * "corridor" (within buffer of both shapes)
          * "off-corridor" (outside buffer of at least one shape).

    The side-by-side plot shows:
      - Left: base route + its relevant stops.
      - Right: other route + its relevant stops.

    Args:
        base_key: (route_id, direction_id) for the base route.
        other_key: (route_id, direction_id) for the comparison route.
        ctx: GTFSContext with sequences, shapes, and stops.
        buffer_m: Buffer distance in meters for each route shape; also used as
            the distance threshold for classifying corridor vs off-corridor
            non-shared stops.

    Returns:
        Path to the saved combined PNG file, or None if plotting was not
        possible. The side-by-side PNG is also written but its path is not
        returned.
    """
    base_seq = ctx.route_sequences.get(base_key)
    other_seq = ctx.route_sequences.get(other_key)
    if base_seq is None or other_seq is None:
        return None

    base_geom_geo = ctx.route_shapes_geo.get(base_key)
    other_geom_geo = ctx.route_shapes_geo.get(other_key)
    base_geom_proj = ctx.route_shapes_proj.get(base_key)
    other_geom_proj = ctx.route_shapes_proj.get(other_key)

    if base_geom_geo is None or other_geom_geo is None:
        return None
    if base_geom_proj is None or other_geom_proj is None:
        return None

    base_route_id, base_dir = base_key
    other_route_id, other_dir = other_key

    shared = set(base_seq).intersection(other_seq)
    base_only = set(base_seq) - shared
    other_only = set(other_seq) - shared

    # --------------------------------------------------------------------- #
    # Helpers shared by both plots
    # --------------------------------------------------------------------- #
    def _plot_route_with_buffer(
        ax: Axes,
        geom_geo: LineString,
        geom_proj: LineString,
        label: str,
        linestyle: str,
    ) -> None:
        """Plot a route LineString and its buffer on the given axis."""
        if buffer_m > 0.0:
            buf_proj = geom_proj.buffer(buffer_m)
            buf_geo = gpd.GeoSeries([buf_proj], crs=PROJECTED_CRS).to_crs(GTFS_CRS).iloc[0]
            if buf_geo.geom_type == "Polygon":
                x_buf, y_buf = buf_geo.exterior.xy
                ax.fill(
                    x_buf,
                    y_buf,
                    alpha=0.15,
                    zorder=BUFFER_ZORDER,
                )
            elif buf_geo.geom_type == "MultiPolygon":
                for poly in buf_geo.geoms:
                    x_buf, y_buf = poly.exterior.xy
                    ax.fill(
                        x_buf,
                        y_buf,
                        alpha=0.15,
                        zorder=BUFFER_ZORDER,
                    )

        x, y = geom_geo.xy
        ax.plot(
            x,
            y,
            label=label,
            linestyle=linestyle,
            linewidth=2,
            zorder=LINE_ZORDER,
        )

    def _scatter_keys(
        ax: Axes,
        keys: Iterable[str],
        marker: str,
        size: float,
        label: str,
    ) -> None:
        """Scatter a set of stops (using geographic CRS for plotting)."""
        xs: List[float] = []
        ys: List[float] = []
        for key in keys:
            if key not in ctx.stops_gdf_geo.index:
                continue
            pt = ctx.stops_gdf_geo.loc[key, "geometry"]
            xs.append(pt.x)
            ys.append(pt.y)
        if xs:
            ax.scatter(
                xs,
                ys,
                marker=marker,
                s=size,
                label=label,
                zorder=STOP_ZORDER,
            )

    def _split_near_far(
        keys: Iterable[str],
        max_dist_m: float,
    ) -> Tuple[Set[str], Set[str]]:
        """Split stops into near/far based on distance to BOTH projected shapes."""
        near: Set[str] = set()
        far: Set[str] = set()

        for key in keys:
            if key not in ctx.stops_gdf_proj.index:
                continue
            pt_proj = ctx.stops_gdf_proj.loc[key, "geometry"]
            d_base = pt_proj.distance(base_geom_proj)
            d_other = pt_proj.distance(other_geom_proj)

            if d_base <= max_dist_m and d_other <= max_dist_m:
                near.add(key)
            else:
                far.add(key)

        return near, far

    base_only_near, base_only_far = _split_near_far(base_only, buffer_m)
    other_only_near, other_only_far = _split_near_far(other_only, buffer_m)

    # --------------------------------------------------------------------- #
    # 1) Combined overview plot
    # --------------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(6, 6))

    _plot_route_with_buffer(
        ax=ax,
        geom_geo=base_geom_geo,
        geom_proj=base_geom_proj,
        label=f"{base_route_id} dir {base_dir}",
        linestyle="-",
    )
    _plot_route_with_buffer(
        ax=ax,
        geom_geo=other_geom_geo,
        geom_proj=other_geom_proj,
        label=f"{other_route_id} dir {other_dir}",
        linestyle="--",
    )

    _scatter_keys(ax, shared, marker=".", size=20, label="shared stops")

    _scatter_keys(
        ax,
        base_only_near,
        marker="^",
        size=55,
        label="only on base (corridor)",
    )
    _scatter_keys(
        ax,
        other_only_near,
        marker="v",
        size=55,
        label="only on other (corridor)",
    )

    _scatter_keys(
        ax,
        base_only_far,
        marker="^",
        size=30,
        label="only on base (off-corridor)",
    )
    _scatter_keys(
        ax,
        other_only_far,
        marker="v",
        size=30,
        label="only on other (off-corridor)",
    )

    title = f"Overview: {base_route_id} dir {base_dir} vs {other_route_id} dir {other_dir}"
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linewidth=0.3, zorder=GRID_ZORDER)
    ax.legend(loc="best", fontsize=8)

    safe_base = f"{base_route_id}_{base_dir}".replace(" ", "_")
    safe_other = f"{other_route_id}_{other_dir}".replace(" ", "_")
    filename_combined = f"overview_{safe_base}_vs_{safe_other}.png"

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path_combined = PLOT_DIR / filename_combined
    fig.savefig(out_path_combined, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logging.info("Saved overview plot: %s", out_path_combined)

    # --------------------------------------------------------------------- #
    # 2) Side-by-side plot
    # --------------------------------------------------------------------- #
    fig2, (ax_left, ax_right) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 5),
    )

    # Compute common bounds so both panels use the same extent.
    all_x: List[float] = []
    all_y: List[float] = []
    for geom in (base_geom_geo, other_geom_geo):
        x_vals, y_vals = geom.xy
        all_x.extend(x_vals)
        all_y.extend(y_vals)
    if all_x and all_y:
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        for ax_sub in (ax_left, ax_right):
            ax_sub.set_xlim(x_min, x_max)
            ax_sub.set_ylim(y_min, y_max)

    # Left panel: base route.
    _plot_route_with_buffer(
        ax=ax_left,
        geom_geo=base_geom_geo,
        geom_proj=base_geom_proj,
        label=f"{base_route_id} dir {base_dir}",
        linestyle="-",
    )
    _scatter_keys(ax_left, shared, marker=".", size=20, label="shared stops")
    _scatter_keys(
        ax_left,
        base_only_near,
        marker="^",
        size=55,
        label="only on base (corridor)",
    )
    _scatter_keys(
        ax_left,
        base_only_far,
        marker="^",
        size=30,
        label="only on base (off-corridor)",
    )
    ax_left.set_title(f"{base_route_id} dir {base_dir}")
    ax_left.set_aspect("equal", adjustable="datalim")
    ax_left.grid(True, linewidth=0.3, zorder=GRID_ZORDER)
    ax_left.legend(loc="best", fontsize=7)

    # Right panel: other route.
    _plot_route_with_buffer(
        ax=ax_right,
        geom_geo=other_geom_geo,
        geom_proj=other_geom_proj,
        label=f"{other_route_id} dir {other_dir}",
        linestyle="-",
    )
    _scatter_keys(ax_right, shared, marker=".", size=20, label="shared stops")
    _scatter_keys(
        ax_right,
        other_only_near,
        marker="v",
        size=55,
        label="only on other (corridor)",
    )
    _scatter_keys(
        ax_right,
        other_only_far,
        marker="v",
        size=30,
        label="only on other (off-corridor)",
    )
    ax_right.set_title(f"{other_route_id} dir {other_dir}")
    ax_right.set_aspect("equal", adjustable="datalim")
    ax_right.grid(True, linewidth=0.3, zorder=GRID_ZORDER)
    ax_right.legend(loc="best", fontsize=7)

    fig2.suptitle(
        f"Side-by-side: {base_route_id} dir {base_dir} vs {other_route_id} dir {other_dir}"
    )

    filename_side = f"overview_{safe_base}_vs_{safe_other}_side_by_side.png"
    out_path_side = PLOT_DIR / filename_side
    fig2.savefig(out_path_side, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    logging.info("Saved side-by-side overview plot: %s", out_path_side)

    return out_path_combined


# =============================================================================
# Orchestration
# =============================================================================


def prepare_gtfs_context() -> GTFSContext:
    """Load GTFS and build all derived structures needed for QA and plotting.

    Returns:
        GTFSContext containing DataFrames, GeoDataFrames, sequences, shapes,
        and route key sets.
    """
    logging.info("Loading GTFS tables from: %s", GTFS_DIR)
    tables = load_gtfs_tables(GTFS_DIR)

    stops_df = tables["stops"]
    trips_df = tables["trips"]
    stop_times_df = tables["stop_times"]
    shapes_df = tables["shapes"]
    routes_df = tables["routes"]

    route_id_whitelist = build_route_id_whitelist(
        routes_df=routes_df,
        route_name_whitelist={str(x) for x in ROUTE_NAME_WHITELIST},
        route_name_field=ROUTE_NAME_WHITELIST_FIELD,
        route_id_whitelist={str(x) for x in ROUTE_ID_WHITELIST},
    )

    logging.info("Building shapes GeoDataFrame...")
    shapes_gdf_geo = build_shapes_gdf(shapes_df, GTFS_CRS)
    shapes_gdf_proj = shapes_gdf_geo.to_crs(PROJECTED_CRS)

    logging.info("Selecting representative shapes per (route, direction)...")
    reps = select_representative_shapes(trips_df)

    logging.info(
        "Choosing representative trip_ids per (route, direction) based on "
        "trips with the most stops..."
    )
    rep_trip_ids = choose_representative_trip_ids_max_stops(
        trips_df=trips_df,
        stop_times_df=stop_times_df,
    )

    logging.info("Building stop key and name lookups...")
    stop_key_lookup = build_stop_key_lookup(stops_df, STOP_KEY_FIELD)
    stop_names = build_stop_names_lookup(stops_df, STOP_KEY_FIELD)

    logging.info("Building route sequences from representative trips...")
    route_sequences = build_route_sequences(
        stop_times_df=stop_times_df,
        stop_key_lookup=stop_key_lookup,
        rep_trip_ids=rep_trip_ids,
    )

    if not route_sequences:
        logging.info("No route sequences could be built. Nothing to compare.")
        return GTFSContext(
            stops_df=stops_df,
            trips_df=trips_df,
            stop_times_df=stop_times_df,
            shapes_df=shapes_df,
            routes_df=routes_df,
            stops_gdf_geo=cast("Any", gpd.GeoDataFrame)(data=[], geometry=[]),
            stops_gdf_proj=cast("Any", gpd.GeoDataFrame)(data=[], geometry=[]),
            stop_key_lookup=stop_key_lookup,
            stop_names=stop_names,
            route_sequences={},
            route_shapes_geo={},
            route_shapes_proj={},
            all_route_keys=[],
            base_route_keys=[],
            route_id_whitelist=route_id_whitelist,
        )

    logging.info("Building route shapes for plotting and distance calculations...")
    route_shapes_geo = build_route_shapes_from_reps(reps, shapes_gdf_geo)
    route_shapes_proj = build_route_shapes_from_reps(reps, shapes_gdf_proj)

    logging.info("Building stops GeoDataFrames...")
    stops_gdf_geo = build_stops_gdf(stops_df, GTFS_CRS, STOP_KEY_FIELD)
    stops_gdf_proj = stops_gdf_geo.to_crs(PROJECTED_CRS)

    all_route_keys = list(route_sequences.keys())

    if TARGET_ROUTE_IDS:
        base_keys = [key for key in all_route_keys if key[0] in TARGET_ROUTE_IDS]
    else:
        base_keys = all_route_keys

    base_keys = [key for key in base_keys if key[0] not in route_id_whitelist]

    logging.info(
        "Prepared %d route/direction pairs; %d selected as base.",
        len(all_route_keys),
        len(base_keys),
    )

    return GTFSContext(
        stops_df=stops_df,
        trips_df=trips_df,
        stop_times_df=stop_times_df,
        shapes_df=shapes_df,
        routes_df=routes_df,
        stops_gdf_geo=stops_gdf_geo,
        stops_gdf_proj=stops_gdf_proj,
        stop_key_lookup=stop_key_lookup,
        stop_names=stop_names,
        route_sequences=route_sequences,
        route_shapes_geo=route_shapes_geo,
        route_shapes_proj=route_shapes_proj,
        all_route_keys=all_route_keys,
        base_route_keys=base_keys,
        route_id_whitelist=route_id_whitelist,
    )


def run_segment_comparison(ctx: GTFSContext) -> pd.DataFrame:
    """Run segment-level stop comparison between routes.

    For each base (route_id, direction_id), compare its logical stop sequence
    against all other route/direction sequences and identify segments bounded
    by pairs of shared stops where the interior subsequences differ.

    Geometry is used at the *segment* level via a Hausdorff distance gate
    (configured by MAX_SHAPE_HAUSDORFF_M). For each candidate segment
    between two shared stops, the corresponding shape substrings are
    compared; segments whose substrings are too far apart are treated as
    different corridors and skipped.

    Args:
        ctx: Prepared GTFSContext.

    Returns:
        DataFrame of segment-level mismatches between routes.
    """
    if not ctx.route_sequences:
        return pd.DataFrame()

    logging.info(
        "Comparing segments for %d base route/direction pairs out of %d total.",
        len(ctx.base_route_keys),
        len(ctx.all_route_keys),
    )

    results: List[Dict[str, object]] = []

    for base_key in ctx.base_route_keys:
        base_route_id, base_dir = base_key
        base_seq = ctx.route_sequences.get(base_key)
        if base_seq is None:
            continue

        logging.info("  Base route %s (direction %s)", base_route_id, base_dir)

        for other_key in ctx.all_route_keys:
            if other_key == base_key:
                continue

            other_route_id, _ = other_key
            if other_route_id in ctx.route_id_whitelist:
                continue

            other_seq = ctx.route_sequences.get(other_key)
            if other_seq is None:
                continue

            # Optional: skip route pairs that only share terminal stops.
            if shares_only_terminal_stops(base_seq, other_seq):
                continue

            segment_results = compare_segments_for_route_pair(
                base_key=base_key,
                other_key=other_key,
                sequences=ctx.route_sequences,
                stop_names=ctx.stop_names,
                shapes_proj=ctx.route_shapes_proj,
                stops_gdf_proj=ctx.stops_gdf_proj,
                max_shape_hausdorff_m=MAX_SHAPE_HAUSDORFF_M,
                max_stop_to_shape_m=MAX_STOP_TO_SHAPE_M,
                segment_measure_padding_m=SEGMENT_MEASURE_PADDING_M,
            )

            if segment_results:
                results.extend(segment_results)

    if not results:
        logging.info("No segment-level stop mismatches were identified.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values(
        [
            "base_route_id",
            "base_direction_id",
            "other_route_id",
            "other_direction_id",
            "segment_start_stop_key",
            "segment_end_stop_key",
        ]
    ).reset_index(drop=True)

    logging.info("Identified %d mismatched segments for review.", len(df))
    return df


def run_plotting(ctx: GTFSContext, mismatches_df: pd.DataFrame) -> None:
    """Generate PNG plots for mismatched segments or overview plots if none.

    If mismatches_df is non-empty, one plot is generated per segment showing:
      - Base and other shapes.
      - Segment boundary stops.
      - Shared interior stops.
      - Unique stops on base and other within that segment.

    If mismatches_df is empty, overview plots are generated only for route pairs that:
      - Share at least one logical stop, and
      - Do not overlap solely at terminal stops in both sequences.
    """
    # Case 1: we have real segment mismatches -> plot them.
    if not mismatches_df.empty:
        logging.info("Generating plots in: %s", PLOT_DIR)
        for _, row in mismatches_df.iterrows():
            plot_mismatch_segment(
                row=row,
                route_sequences=ctx.route_sequences,
                route_shapes_geo=ctx.route_shapes_geo,
                stops_gdf_geo=ctx.stops_gdf_geo,
                stop_names=ctx.stop_names,
            )
        return

    # Case 2: no mismatches -> optional overview plots, but only where
    # there is a meaningful shared interior stop.
    logging.info(
        "Mismatch DataFrame is empty; generating filtered overview plots only "
        "for route pairs that share interior stops."
    )

    for base_key in ctx.base_route_keys:
        base_seq = ctx.route_sequences.get(base_key)
        if base_seq is None:
            continue

        for other_key in ctx.all_route_keys:
            if other_key == base_key:
                continue
            if other_key[0] in ctx.route_id_whitelist:
                continue

            other_seq = ctx.route_sequences.get(other_key)
            if other_seq is None:
                continue

            # Must share at least one logical stop.
            shared = set(base_seq).intersection(other_seq)
            if not shared:
                continue

            # Skip if they only overlap at terminal stops in both sequences.
            if shares_only_terminal_stops(base_seq, other_seq):
                continue

            # At this point, the pair shares at least one non-terminal stop:
            # generate an overview plot.
            plot_route_pair_overview(
                base_key=base_key,
                other_key=other_key,
                ctx=ctx,
                buffer_m=ROUTE_OVERVIEW_BUFFER_M,
            )


def main() -> None:
    """Entry point for running the segment stop comparison QA check."""
    logging.basicConfig(level=logging.INFO)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        ctx = prepare_gtfs_context()
    except (FileNotFoundError, ValueError) as exc:
        logging.error("Error during GTFS preparation: %s", exc)
        sys.exit(1)

    if not ctx.route_sequences:
        # Nothing to compare; bail out.
        sys.exit(0)

    try:
        mismatches_df = run_segment_comparison(ctx)
    except (FileNotFoundError, ValueError) as exc:
        logging.error("Error during segment comparison: %s", exc)
        sys.exit(1)

    # Always write the CSV, even if it's empty.
    output_path = OUTPUT_DIR / OUTPUT_FILENAME
    mismatches_df.to_csv(output_path, index=False)
    logging.info("Results exported to: %s", output_path)

    # Always run plotting; it already knows how to handle the empty case.
    run_plotting(ctx, mismatches_df)


if __name__ == "__main__":
    main()
