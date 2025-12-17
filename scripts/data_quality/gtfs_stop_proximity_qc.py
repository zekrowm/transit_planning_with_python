"""GTFS stop proximity checker.

Reads GTFS stops.txt and flags stops closer than a configured distance threshold
(default 50 feet).

Optionally excludes ("passes") stops whose stop_name contains any configured safe
words (e.g., "bay", "metro"). When enabled, any close pair where either stop is
"safe" is ignored entirely (i.e., not emitted to outputs).

Optionally excludes close stop pairs that appear to be "across the street" stops:
pairs served by the same route, but in different directions (e.g., one stop only
served by dir 0 trips for route X, the other only served by dir 1 trips for route X).

Outputs:
- CSV of close stop pairs (with distance + safe flags)
- CSV of per-stop summary (count of close neighbors)
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

GTFS_DIR = Path(r"C:\path\to\gtfs")  # folder containing GTFS .txt files
OUT_DIR = Path(r"C:\path\to\output")  # output folder for CSVs

THRESHOLD_FEET = 50.0

# Safe-word handling: stops with these words/phrases in stop_name are exempted from pair output.
# Add/remove terms here. Matching is case-insensitive.
SAFE_WORDS: list[str] = [
    "bay",
    # "metro",
    # "station",
]
SAFE_WORD_MATCH_WHOLE_WORD = True  # True -> \bterm\b, False -> substring match
PASS_SAFE_STOPS = True  # if True, skip any close pair where either stop is "safe"

# Across-the-street handling (directional pairs on same route):
# If True, exclude pairs that share a route but are served exclusively by opposite directions.
EXCLUDE_OPPOSITE_DIRECTION_SAME_ROUTE_PAIRS = True

LOG_LEVEL = "INFO"  # DEBUG | INFO | WARNING

LOGGER = logging.getLogger(__name__)

# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

EARTH_RADIUS_M = 6_371_000.0
FEET_PER_M = 3.280839895


def _meters_to_feet(meters: float) -> float:
    """Convert meters to feet."""
    return meters * FEET_PER_M


def _approx_xy_meters(
    lat_deg: pd.Series,
    lon_deg: pd.Series,
) -> tuple[pd.Series, pd.Series, float, float]:
    """Project lat/lon to local x/y in meters using equirectangular approximation."""
    lat0_rad = math.radians(float(lat_deg.mean()))
    lon0_rad = math.radians(float(lon_deg.mean()))

    lat_rad = lat_deg.astype(float).map(math.radians)
    lon_rad = lon_deg.astype(float).map(math.radians)

    x_m = (lon_rad - lon0_rad) * math.cos(lat0_rad) * EARTH_RADIUS_M
    y_m = (lat_rad - lat0_rad) * EARTH_RADIUS_M
    return x_m, y_m, lat0_rad, lon0_rad


def _euclid_feet(dx_m: float, dy_m: float) -> float:
    """Euclidean distance (feet) from delta meters."""
    return _meters_to_feet(math.hypot(dx_m, dy_m))


# =============================================================================
# GRID INDEXING (FAST NEIGHBOR SEARCH)
# =============================================================================


@dataclass(frozen=True)
class GridParams:
    """Parameters defining a grid used for near-neighbor lookup."""

    cell_size_m: float


def _grid_cell(x_m: float, y_m: float, cell_size_m: float) -> tuple[int, int]:
    """Return integer grid cell coordinates for a point."""
    return (int(math.floor(x_m / cell_size_m)), int(math.floor(y_m / cell_size_m)))


def _neighbor_cells(cell: tuple[int, int]) -> Iterable[tuple[int, int]]:
    """Yield the 3x3 neighborhood (cell and its 8 adjacent cells)."""
    cx, cy = cell
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            yield (cx + dx, cy + dy)


# =============================================================================
# SAFE WORD HELPERS
# =============================================================================


def compile_safe_words_regex(words: list[str], whole_word: bool) -> re.Pattern[str]:
    """Compile a case-insensitive regex that matches any provided words/phrases."""
    cleaned = [w.strip() for w in words if w and w.strip()]
    if not cleaned:
        return re.compile(r"a\A", flags=re.IGNORECASE)  # match nothing

    parts: list[str] = []
    for w in cleaned:
        esc = re.escape(w)
        parts.append(rf"\b{esc}\b" if whole_word else esc)

    return re.compile("|".join(parts), flags=re.IGNORECASE)


def add_safe_flag(df: pd.DataFrame, safe_words: list[str], whole_word: bool) -> pd.DataFrame:
    """Add is_safe_stop flag based on stop_name matching safe words/phrases."""
    rx = compile_safe_words_regex(safe_words, whole_word=whole_word)
    out = df.copy()
    out["is_safe_stop"] = out["stop_name"].astype(str).map(lambda s: bool(rx.search(s)))
    return out


# =============================================================================
# GTFS LOADING
# =============================================================================


def load_stops(stops_txt: Path) -> pd.DataFrame:
    """Load GTFS stops.txt with minimal validation."""
    df = pd.read_csv(stops_txt, dtype={"stop_id": "string", "stop_name": "string"})
    required = {"stop_id", "stop_name", "stop_lat", "stop_lon"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"stops.txt missing required columns: {missing}")

    df = df[list(required)].copy()
    df["stop_lat"] = pd.to_numeric(df["stop_lat"], errors="coerce")
    df["stop_lon"] = pd.to_numeric(df["stop_lon"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["stop_lat", "stop_lon", "stop_id"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        LOGGER.warning("Dropped %s stops with missing/invalid lat/lon/stop_id.", dropped)

    df["stop_name"] = df["stop_name"].fillna("")
    return df


def build_stop_route_direction_index(gtfs_dir: Path) -> dict[str, dict[str, set[int]]]:
    """Build stop -> route -> set(direction_id) index.

    Uses stop_times.txt joined to trips.txt. This is used to suppress "opposite
    direction on same route" pairs.

    Returns:
        Dict[stop_id][route_id] = {direction_id, ...}

    Behavior:
        - If direction_id is missing from trips.txt, returns an empty dict (feature disabled).
        - Any non-numeric direction_id rows are dropped.
    """
    stop_times_path = gtfs_dir / "stop_times.txt"
    trips_path = gtfs_dir / "trips.txt"

    if not stop_times_path.exists() or not trips_path.exists():
        LOGGER.warning("Missing stop_times.txt or trips.txt; direction-based filtering disabled.")
        return {}

    stop_times = pd.read_csv(
        stop_times_path,
        usecols=["trip_id", "stop_id"],
        dtype={"trip_id": "string", "stop_id": "string"},
    )

    trips = pd.read_csv(trips_path, dtype={"trip_id": "string", "route_id": "string"})

    if "direction_id" not in trips.columns:
        LOGGER.warning("trips.txt has no direction_id; direction-based filtering disabled.")
        return {}

    trips = trips[["trip_id", "route_id", "direction_id"]].copy()
    trips["direction_id"] = pd.to_numeric(trips["direction_id"], errors="coerce").astype("Int64")
    trips = trips.dropna(subset=["direction_id", "route_id", "trip_id"]).copy()
    trips["direction_id"] = trips["direction_id"].astype(int)

    st = stop_times.merge(trips, on="trip_id", how="inner")
    if st.empty:
        return {}

    # Unique combos only
    st = st[["stop_id", "route_id", "direction_id"]].drop_duplicates()

    index: dict[str, dict[str, set[int]]] = {}
    for stop_id, route_id, direction_id in st.itertuples(index=False, name=None):
        index.setdefault(str(stop_id), {}).setdefault(str(route_id), set()).add(int(direction_id))

    return index


def is_opposite_direction_pair_same_route(
    stop_a: str,
    stop_b: str,
    index: dict[str, dict[str, set[int]]],
) -> bool:
    """Return True if stops look like opposite-direction-only on at least one shared route.

    Heuristic:
      - Find routes served by both stops.
      - If there exists a route where A is served only by {0} and B only by {1},
        or vice-versa, treat as across-the-street directional pair.

    If either stop lacks route/dir info, returns False.
    """
    a = index.get(stop_a)
    b = index.get(stop_b)
    if not a or not b:
        return False

    shared_routes = set(a).intersection(b)
    if not shared_routes:
        return False

    for r in shared_routes:
        a_dirs = a.get(r, set())
        b_dirs = b.get(r, set())
        if a_dirs == {0} and b_dirs == {1}:
            return True
        if a_dirs == {1} and b_dirs == {0}:
            return True

    return False


# =============================================================================
# CORE LOGIC
# =============================================================================


def find_close_stop_pairs(
    stops: pd.DataFrame,
    threshold_feet: float,
    pass_safe_stops: bool,
    exclude_opposite_direction_same_route_pairs: bool,
    stop_route_dir_index: dict[str, dict[str, set[int]]],
) -> pd.DataFrame:
    """Find stop pairs within the distance threshold, with optional filters."""
    if "is_safe_stop" not in stops.columns:
        raise ValueError("Expected stops to include 'is_safe_stop'. Call add_safe_flag() first.")

    threshold_m = threshold_feet / FEET_PER_M
    x_m, y_m, _, _ = _approx_xy_meters(stops["stop_lat"], stops["stop_lon"])
    work = stops.copy()
    work["x_m"] = x_m
    work["y_m"] = y_m

    params = GridParams(cell_size_m=threshold_m)

    grid: dict[tuple[int, int], list[int]] = {}
    for idx, (xv, yv) in enumerate(zip(work["x_m"], work["y_m"], strict=True)):
        cell = _grid_cell(float(xv), float(yv), params.cell_size_m)
        grid.setdefault(cell, []).append(idx)

    rows: list[dict[str, object]] = []
    n = len(work)

    for i in range(n):
        xi = float(work.at[i, "x_m"])
        yi = float(work.at[i, "y_m"])
        cell_i = _grid_cell(xi, yi, params.cell_size_m)

        for nc in _neighbor_cells(cell_i):
            for j in grid.get(nc, []):
                if j <= i:
                    continue

                i_safe = bool(work.at[i, "is_safe_stop"])
                j_safe = bool(work.at[j, "is_safe_stop"])
                if pass_safe_stops and (i_safe or j_safe):
                    continue

                stop_id_a = str(work.at[i, "stop_id"])
                stop_id_b = str(work.at[j, "stop_id"])

                if exclude_opposite_direction_same_route_pairs and stop_route_dir_index:
                    if is_opposite_direction_pair_same_route(
                        stop_id_a, stop_id_b, stop_route_dir_index
                    ):
                        continue

                dx = float(work.at[j, "x_m"]) - xi
                dy = float(work.at[j, "y_m"]) - yi

                if abs(dx) > threshold_m or abs(dy) > threshold_m:
                    continue

                dist_ft = _euclid_feet(dx, dy)
                if dist_ft > threshold_feet:
                    continue

                rows.append(
                    {
                        "stop_id_a": stop_id_a,
                        "stop_name_a": work.at[i, "stop_name"],
                        "is_safe_a": i_safe,
                        "stop_id_b": stop_id_b,
                        "stop_name_b": work.at[j, "stop_name"],
                        "is_safe_b": j_safe,
                        "distance_feet": round(dist_ft, 2),
                        "threshold_feet": threshold_feet,
                    }
                )

    pairs = pd.DataFrame.from_records(rows)
    if pairs.empty:
        return pairs

    return pairs.sort_values(["distance_feet"], ascending=[True]).reset_index(drop=True)


def summarize_by_stop(pairs: pd.DataFrame) -> pd.DataFrame:
    """Create per-stop counts of close neighbors (based on emitted pairs only)."""
    if pairs.empty:
        return pd.DataFrame(columns=["stop_id", "close_neighbor_pairs"])

    a = pairs[["stop_id_a"]].rename(columns={"stop_id_a": "stop_id"})
    b = pairs[["stop_id_b"]].rename(columns={"stop_id_b": "stop_id"})
    both = pd.concat([a, b], ignore_index=True)

    out = (
        both.groupby("stop_id", as_index=False)
        .size()
        .rename(columns={"size": "close_neighbor_pairs"})
        .sort_values(["close_neighbor_pairs"], ascending=[False])
        .reset_index(drop=True)
    )
    return out


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Run the stop proximity QC."""
    logging.basicConfig(level=getattr(logging, LOG_LEVEL), format="%(levelname)s: %(message)s")

    stops_txt = GTFS_DIR / "stops.txt"
    if not stops_txt.exists():
        raise FileNotFoundError(f"Could not find stops.txt at: {stops_txt}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stops = load_stops(stops_txt)
    stops = add_safe_flag(stops, safe_words=SAFE_WORDS, whole_word=SAFE_WORD_MATCH_WHOLE_WORD)

    stop_route_dir_index: dict[str, dict[str, set[int]]] = {}
    if EXCLUDE_OPPOSITE_DIRECTION_SAME_ROUTE_PAIRS:
        stop_route_dir_index = build_stop_route_direction_index(GTFS_DIR)

    pairs = find_close_stop_pairs(
        stops=stops,
        threshold_feet=float(THRESHOLD_FEET),
        pass_safe_stops=bool(PASS_SAFE_STOPS),
        exclude_opposite_direction_same_route_pairs=bool(
            EXCLUDE_OPPOSITE_DIRECTION_SAME_ROUTE_PAIRS
        ),
        stop_route_dir_index=stop_route_dir_index,
    )
    summary = summarize_by_stop(pairs)

    pairs_path = OUT_DIR / "close_stop_pairs.csv"
    summary_path = OUT_DIR / "close_stop_summary_by_stop.csv"

    pairs.to_csv(pairs_path, index=False)
    summary.to_csv(summary_path, index=False)

    LOGGER.info("Stops loaded: %s", len(stops))
    LOGGER.info("Close pairs found (after filtering): %s", 0 if pairs.empty else len(pairs))
    LOGGER.info("Wrote: %s", pairs_path)
    LOGGER.info("Wrote: %s", summary_path)


if __name__ == "__main__":
    main()
