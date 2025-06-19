"""Performs internal QA checks on a GTFS feed to identify common structural issues.

Checks include orphan stops, unused routes/trips, isolated trips, shapes far from stops,
disconnected stop sequences, unrealistic timings, and malformed stop distances. Each rule
outputs a report if any issues are found.

Inputs:
    - GTFS folder with stops.txt, stop_times.txt, routes.txt, trips.txt, shapes.txt
    - Configurable constants (paths, filters, thresholds) defined in the script

Outputs:
    - CSV reports in the output folder for each failing rule (if applicable)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Set

import networkx as nx
import pandas as pd
from shapely import geometry as sgeom
from shapely.ops import nearest_points

# =============================================================================
# CONFIGURATION
# =============================================================================

GTFS_PATH = Path(r"\\Path\To\Your\GTFS_Folder")  # Edit Path
OUTPUT_PATH: Path | None = Path(r"\\Path\To\Your\Output_Folder")  # Edit Path

# Route filtering (by route_short_name)
FILTER_IN_ROUTES: List[str] = []  # analyse only these
FILTER_OUT_ROUTES: List[str] = ["9999A", "9999B", "9999C"]  # Edit as needed

# Thresholds (internal checks)
SHAPE_DISTANCE_THRESHOLD_FT = 328  # ~100 m ≈ 328 ft
UNREALISTIC_SPEED_MPH = 60.0  # flag anything > 60 mph
LONG_GAP_MIN = 60  # minutes

# -----------------------------------------------------------------------------

DEG_TO_MILES = 69.172  # rough miles per degree
DEG_TO_FEET = DEG_TO_MILES * 5280

# =============================================================================
# FUNCTIONS
# =============================================================================


def read_txts(folder: Path, *names: str) -> Dict[str, pd.DataFrame]:
    """Read one or more GTFS text files from the given folder.

    Args:
        folder: Path to the GTFS folder.
        *names: Filenames (without .txt extension) to read.

    Returns:
        Dictionary mapping each file name to a DataFrame. If a file is missing, an empty
        DataFrame is returned in its place with a warning.
    """
    dfs: Dict[str, pd.DataFrame] = {}
    for n in names:
        fp = folder / f"{n}.txt"
        if not fp.exists():
            logging.warning("Missing %s.txt – skipping related tests.", n)
            dfs[n] = pd.DataFrame()
            continue
        dfs[n] = pd.read_csv(fp, dtype=str, low_memory=False)
        logging.info("Loaded %s (%d rows).", fp.name, len(dfs[n]))
    return dfs


def safe_write(
    df: pd.DataFrame, out_dir: Path, fname: str, *, write_empty: bool = False
) -> None:
    """Write a DataFrame to CSV in the specified output directory.

    Args:
        df: DataFrame to write.
        out_dir: Output directory to create if it does not exist.
        fname: Output file name, e.g. "orphan_stops.csv".
        write_empty: If True, creates an empty CSV even when df is empty.
    """
    tag = fname[:-4] if fname.lower().endswith(".csv") else fname
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / fname

    if df.empty:
        if write_empty:  # keep old behaviour when explicitly asked
            out_path.touch(exist_ok=True)
            logging.info("✓ %s: no issues found (empty file created).", tag)
        else:  # new default: skip creating the file
            logging.info("✓ %s: no issues found (file skipped).", tag)
        return

    df.to_csv(out_path, index=False)
    logging.info("⚠ %s: %d records written to %s", tag, len(df), out_path)


def haversine_miles(lat1, lon1, lat2, lon2) -> float:
    """Calculate great-circle distance between two lat/lon points in miles.

    Args:
        lat1: Latitude of point 1.
        lon1: Longitude of point 1.
        lat2: Latitude of point 2.
        lon2: Longitude of point 2.

    Returns:
        Distance in miles as a float.
    """
    from math import atan2, cos, radians, sin, sqrt

    R_MI = 3958.8
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    )
    return 2 * R_MI * atan2(sqrt(a), sqrt(1 - a))


def parse_time(t: str) -> int | None:
    """Parse a GTFS time string (hh:mm:ss) into seconds after midnight.

    Args:
        t: Time string in HH:MM:SS format.

    Returns:
        Time in seconds as an integer, or None if parsing fails.
    """
    try:
        hh, mm, ss = map(int, t.split(":"))
        return hh * 3600 + mm * 60 + ss
    except Exception:  # noqa: BLE001
        return None


def build_route_id_set(routes: pd.DataFrame) -> Set[str]:
    """Return the set of route_id values that pass the include/exclude rules."""
    if "route_short_name" not in routes.columns:
        logging.warning(
            "route_short_name column missing – route filters skipped entirely."
        )
        return set(routes["route_id"])

    df = routes.copy()

    # include-only filter
    if FILTER_IN_ROUTES:
        df_in = df[df["route_short_name"].isin(FILTER_IN_ROUTES)]
        missing_in = set(FILTER_IN_ROUTES) - set(df_in["route_short_name"])
        if missing_in:
            logging.warning("FILTER_IN_ROUTES not found: %s", ", ".join(missing_in))
        df = df_in

    # exclude filter (always applied)
    if FILTER_OUT_ROUTES:
        df = df[~df["route_short_name"].isin(FILTER_OUT_ROUTES)]
        missing_out = set(FILTER_OUT_ROUTES) - set(routes["route_short_name"])
        if missing_out:
            logging.info(
                "FILTER_OUT_ROUTES not present in feed: %s", ", ".join(missing_out)
            )

    selected = set(df["route_id"])
    if not selected:
        logging.error("After filtering, zero routes remain. Validation aborted.")
        sys.exit(1)
    logging.info("Route filter leaves %d routes for validation.", len(selected))
    return selected


def orphan_stops(stops: pd.DataFrame, stop_times: pd.DataFrame) -> pd.DataFrame:
    """Identify stops that are not used in any trip.

    Args:
        stops: DataFrame of stops.txt.
        stop_times: DataFrame of stop_times.txt.

    Returns:
        DataFrame of orphan stops with columns ['stop_id', 'stop_name'].
    """
    used = set(stop_times["stop_id"].unique())
    return stops.loc[~stops["stop_id"].isin(used), ["stop_id", "stop_name"]]


def unused_routes(routes: pd.DataFrame, trips: pd.DataFrame) -> pd.DataFrame:
    """Identify routes with no associated trips.

    Args:
        routes: DataFrame of routes.txt.
        trips: DataFrame of trips.txt.

    Returns:
        DataFrame of unused routes.
    """
    used = set(trips["route_id"].unique())
    return routes.loc[~routes["route_id"].isin(used)].copy()


def unused_trips(trips: pd.DataFrame, stop_times: pd.DataFrame) -> pd.DataFrame:
    """Identify trips that do not appear in stop_times.txt.

    Args:
        trips: DataFrame of trips.txt.
        stop_times: DataFrame of stop_times.txt.

    Returns:
        DataFrame of unused trips.
    """
    used = set(stop_times["trip_id"].unique())
    return trips.loc[~trips["trip_id"].isin(used)].copy()


def isolated_trips(stop_times: pd.DataFrame) -> pd.DataFrame:
    """Identify trips that use only stops not used by any other trip.

    Args:
        stop_times: DataFrame of stop_times.txt.

    Returns:
        DataFrame with one column, 'trip_id', listing isolated trips.
    """
    counts = stop_times.groupby("stop_id")["trip_id"].nunique()
    unique_stops = counts[counts == 1].index
    grp = stop_times.groupby("trip_id")["stop_id"].apply(
        lambda s: set(s).issubset(unique_stops)
    )
    return pd.DataFrame({"trip_id": grp[grp].index})


def shapes_far_from_stops(
    trips: pd.DataFrame,
    shapes: pd.DataFrame,
    stop_times: pd.DataFrame,
    stops: pd.DataFrame,
) -> pd.DataFrame:
    """Identify trips whose stops are too far from the shape path.

    Args:
        trips: DataFrame of trips.txt.
        shapes: DataFrame of shapes.txt.
        stop_times: DataFrame of stop_times.txt.
        stops: DataFrame of stops.txt.

    Returns:
        DataFrame of trips with route_id, trip_id, and shape_id that violate the threshold.
    """
    if sgeom is None or shapes.empty:
        logging.warning(
            "Shapely not installed or shapes.txt missing – skipping rule 4."
        )
        return pd.DataFrame()

    # Ensure correct dtypes
    shapes = shapes.astype({"shape_pt_lat": float, "shape_pt_lon": float})
    shapes["shape_id"] = shapes["shape_id"].astype(str)
    trips["shape_id"] = trips["shape_id"].astype(str)

    # Precompute LineString geometry by shape_id
    geom_by_id: dict[str, sgeom.LineString] = {
        str(shape_id): sgeom.LineString(df[["shape_pt_lon", "shape_pt_lat"]].values)
        for shape_id, df in shapes.sort_values(
            ["shape_id", "shape_pt_sequence"]
        ).groupby("shape_id")
    }

    def far(trip: pd.Series) -> bool:
        shape_id = trip.shape_id
        line = geom_by_id.get(shape_id)
        if not line:
            return False
        tid = trip.trip_id
        merged = stop_times.loc[stop_times.trip_id == tid].merge(
            stops[["stop_id", "stop_lat", "stop_lon"]], on="stop_id"
        )
        for _, row in merged.iterrows():
            p = sgeom.Point(float(row.stop_lon), float(row.stop_lat))
            _, nearest = nearest_points(p, line)
            if p.distance(nearest) * DEG_TO_FEET > SHAPE_DISTANCE_THRESHOLD_FT:
                return True
        return False

    mask = trips["shape_id"].notna() & trips.apply(far, axis=1)
    return trips.loc[mask, ["route_id", "trip_id", "shape_id"]]


def hanging_segments(stop_times: pd.DataFrame, stops: pd.DataFrame) -> pd.DataFrame:
    """Detect disconnected stop subgraphs in the network.

    Args:
        stop_times: DataFrame of stop_times.txt.
        stops: DataFrame of stops.txt.

    Returns:
        DataFrame listing each disconnected component with size and sample stop_id.
    """
    if nx is None:
        logging.warning("NetworkX not installed – skipping rule 6.")
        return pd.DataFrame()

    G: nx.Graph = nx.Graph()
    G.add_nodes_from(stops.stop_id)

    st_sorted = stop_times.sort_values(["trip_id", "stop_sequence"])
    for _, grp in st_sorted.groupby("trip_id"):
        ids = grp.stop_id.tolist()
        G.add_edges_from(zip(ids[:-1], ids[1:]))

    comps = list(nx.connected_components(G))
    if len(comps) <= 1:
        return pd.DataFrame()

    comps.sort(key=len, reverse=True)
    hangers = comps[1:]
    return pd.DataFrame(
        {
            "component_no": range(1, len(hangers) + 1),
            "n_stops": [len(c) for c in hangers],
            "sample_stop_id": [next(iter(c)) for c in hangers],
        }
    )

def unrealistic_timings(
    stop_times: pd.DataFrame,
    stops: pd.DataFrame,
) -> pd.DataFrame:
    """Detect trip segments with unrealistically high speeds or long time gaps.

    Args:
        stop_times: DataFrame of stop_times.txt.
        stops: DataFrame of stops.txt.

    Returns:
        DataFrame of problematic trip segments with computed speed and time gap.
    """
    st = stop_times.merge(
        stops[["stop_id", "stop_lat", "stop_lon"]], on="stop_id", how="left"
    )

    offenders: List[Dict] = []
    for tid, grp in st.groupby("trip_id"):
        grp = grp.sort_values("stop_sequence")
        prev = None
        for _, row in grp.iterrows():
            tsec = parse_time(row.departure_time or row.arrival_time)
            if prev and tsec and prev["t"]:
                dt = tsec - prev["t"]
                if dt <= 0:
                    continue
                dist_mi = haversine_miles(
                    prev["lat"], prev["lon"], float(row.stop_lat), float(row.stop_lon)
                )
                speed = (dist_mi / dt) * 3600
                if speed > UNREALISTIC_SPEED_MPH or dt / 60 > LONG_GAP_MIN:
                    offenders.append(
                        {
                            "trip_id": tid,
                            "prev_stop": prev["sid"],
                            "next_stop": row.stop_id,
                            "gap_min": round(dt / 60, 1),
                            "speed_mph": round(speed, 1),
                        }
                    )
            prev = {
                "sid": row.stop_id,
                "t": tsec,
                "lat": float(row.stop_lat),
                "lon": float(row.stop_lon),
            }
    return pd.DataFrame(offenders)


def bad_stop_sequences(stop_times: pd.DataFrame) -> pd.DataFrame:
    """Detect trips with decreasing shape_dist_traveled values.

    Args:
        stop_times: DataFrame of stop_times.txt.

    Returns:
        DataFrame of trip_ids with malformed stop sequences.
    """
    if "shape_dist_traveled" not in stop_times.columns:
        logging.warning("shape_dist_traveled missing – skipping rule 8.")
        return pd.DataFrame()

    mask = (
        stop_times.sort_values(["trip_id", "stop_sequence"])
        .groupby("trip_id")["shape_dist_traveled"]
        .apply(lambda d: (d.astype(float).diff() < 0).any())
    )
    return pd.DataFrame({"trip_id": mask[mask].index})


# =============================================================================
# MAIN
# =============================================================================


def main(gtfs_path: Path, out_path: Path) -> None:
    """Run all internal validation checks on the specified GTFS dataset.

    Args:
        gtfs_path: Path to the GTFS folder.
        out_path: Path to the output folder where reports will be saved.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if not gtfs_path.is_dir():
        logging.error("%s is not a directory. Check GTFS_PATH in CONFIG.", gtfs_path)
        sys.exit(1)

    # Load all the required GTFS tables
    dfs = read_txts(
        gtfs_path,
        "stops",
        "stop_times",
        "routes",
        "trips",
        "shapes",
    )
    stops, stop_times, routes, trips, shapes = (
        dfs["stops"],
        dfs["stop_times"],
        dfs["routes"],
        dfs["trips"],
        dfs["shapes"],
    )

    # Verify required columns
    required = {
        "stops": ["stop_id"],
        "stop_times": ["trip_id", "stop_id"],
        "routes": ["route_id"],
        "trips": ["trip_id", "route_id"],
    }
    for fname, cols in required.items():
        missing = set(cols) - set(dfs[fname].columns)
        if missing:
            logging.error(
                "%s.txt missing required columns: %s – aborting validation.",
                fname,
                ", ".join(missing),
            )
            return

    # Apply route filters early
    selected_route_ids = build_route_id_set(routes)
    routes = routes[routes.route_id.isin(selected_route_ids)]
    trips = trips[trips.route_id.isin(selected_route_ids)]
    stop_times = stop_times[stop_times.trip_id.isin(trips.trip_id)]

    if not shapes.empty and "shape_id" in trips.columns:
        shapes = shapes[shapes.shape_id.isin(trips.shape_id.dropna())]

    # Run each rule and collect its DataFrame
    reports = [
        ("orphan_stops.csv", orphan_stops(stops, stop_times)),
        ("unused_routes.csv", unused_routes(routes, trips)),
        ("unused_trips.csv", unused_trips(trips, stop_times)),
        ("isolated_trips.csv", isolated_trips(stop_times)),
        (
            "shapes_far_from_stops.csv",
            shapes_far_from_stops(trips, shapes, stop_times, stops),
        ),
        ("hanging_segments.csv", hanging_segments(stop_times, stops)),
        ("unrealistic_timings.csv", unrealistic_timings(stop_times, stops)),
        ("bad_stop_sequences.csv", bad_stop_sequences(stop_times)),
    ]

    # Write them all out with the updated safe_write signature
    for fname, df in reports:
        safe_write(df, out_path, fname)

    logging.info("🎉 Validation finished. Reports in %s", out_path.resolve())


if __name__ == "__main__":
    # default to GTFS folder if OUTPUT_PATH is None or empty string
    out_folder = OUTPUT_PATH or GTFS_PATH
    main(GTFS_PATH, out_folder)
