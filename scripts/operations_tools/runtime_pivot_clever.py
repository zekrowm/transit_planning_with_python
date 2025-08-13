"""Summarize CLEVER segment-level running times for multiple service periods.

The script ingests one CLEVER export per service period (Weekday, Saturday,
Sunday, Other), filters the data set(s) by route, converts selected time
metrics from seconds to minutes, validates linearity of segment chains, and
emits one pivot-table CSV per *(route, direction, time metric)*.  Explicit
“_NoData.csv” placeholders are written whenever a direction—or an entire
route—produces no pivots so that downstream workflows can detect absence
deterministically.

Typical Usage
Run the script from ArcGIS Pro’s Python window or a Jupyter notebook.
"""

from __future__ import annotations

import os
from typing import List, Mapping, Optional, Sequence, Union, Iterable, Tuple, Dict, Set

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

WKDY_FILE_PATH = r"\\Your\Path\CLEVER_Runtime_by_Segment_by_Trip_Weekday.csv"
SAT_FILE_PATH = r""
SUN_FILE_PATH = r""
OTHER_FILE_PATH = r""

PARENT_OUTPUT_DIR = r"C:\Path\To\Outputs"  # or "" to save in current folder

ROUTES_TO_EXCLUDE = []
ROUTES_TO_INCLUDE = []

TIME_COLUMNS = {
    "Average Actual Running Time": "AvgActual(min)",
    "Average Deviation": "AvgDeviation(min)",
    "Average StartTPSScheduleDeviation": "StartTPSDev(min)",
    "Average StartTPScheduleDeviation": "StartTPSchedDev(min)",
}

# =============================================================================
# FUNCTIONS
# =============================================================================


def load_data(file_path: str) -> pd.DataFrame:
    """Return a DataFrame from a CSV or Excel file.

    The reader is selected by file-name extension.  Duplicate columns in the
    source (a frequent CLEVER quirk) are silently dropped.

    Args:
        file_path: Absolute or relative path to ``.csv``, ``.xls`` or
            ``.xlsx`` file.

    Returns:
        A :class:`~pandas.DataFrame` containing the file contents.

    Raises:
        ValueError: If *file_path* has an unsupported extension.
        FileNotFoundError: If *file_path* does not exist.
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file extension '{ext}' for file: {file_path}")

    df = df.loc[:, ~df.columns.duplicated()]
    return df


def filter_routes(
    df: pd.DataFrame,
    routes_to_exclude: Optional[Sequence[str]] = None,
    routes_to_include: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Filter a CLEVER export by *Branch* (i.e. route) column.

    Args:
        df: The raw CLEVER DataFrame.
        routes_to_exclude: Branch values to drop, or an empty list/None.
        routes_to_include: Branch values to retain, or an empty list/None.
            When supplied, *routes_to_exclude* is ignored for overlapping
            entries (i.e. explicit inclusion wins).

    Returns:
        The filtered DataFrame (copy).
    """
    routes_to_exclude = routes_to_exclude or []
    routes_to_include = routes_to_include or []

    if routes_to_exclude:
        df = df[~df["Branch"].isin(routes_to_exclude)]
    if routes_to_include:
        df = df[df["Branch"].isin(routes_to_include)]
    return df


def convert_time_columns(
    df: pd.DataFrame,
    time_columns: Optional[Union[Mapping[str, str], Sequence[str]]] = None,
) -> None:
    """Convert selected time columns from seconds to minutes *in-place*.

    Args:
        df: DataFrame whose columns are to be converted.
        time_columns: Either the same mapping used elsewhere in the script
            (dict of *original column* → *suffix*) or a list/tuple of raw
            column names.  Columns that are absent are ignored.

    Notes:
        The function mutates *df* and returns :pydata:`None`.
    """
    if isinstance(time_columns, dict):
        cols = list(time_columns.keys())
    else:
        cols = time_columns or []

    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col] / 60.0


def parse_trip_time(trip_str: str) -> Optional[int]:
    """Parse the *HH:MM* portion of a CLEVER *Trip* string."""
    if not isinstance(trip_str, str):
        return None
    token = trip_str.split(maxsplit=1)[0]  # e.g., '04:56' or '04:56:12'
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            t = pd.to_datetime(token, format=fmt).time()
            return t.hour * 60 + t.minute
        except ValueError:
            continue
    try:
        t = pd.to_datetime(token).time()
        return t.hour * 60 + t.minute
    except Exception:
        return None


def check_route_validity(
    segments: Sequence[str],
    variation_values: Sequence[str],
) -> bool:
    """Perform sanity checks on a collection of *START – END* segments.

    The following issues are flagged (warnings only):

    * Multiple *Variation* values in the same route/direction.
    * Malformed segment strings lacking ``" - "`` delimiter.
    * Branching or loops in the implied graph.
    * Missing or non-unique starting stop.
    * Dangling segments that cannot be chained into a single path.

    Args:
        segments: Unique *SegmentName* values for a route/direction.
        variation_values: Unique *Variation* values for the same slice.

    Returns:
        ``True`` when the segment chain appears linear; ``False`` otherwise.
    """
    is_valid = True
    if len(set(variation_values)) > 1:
        print(f"WARNING: Multiple Variation values: {set(variation_values)}")
        is_valid = False

    edges = []
    for seg in segments:
        if " - " not in seg:
            print(f"WARNING: Malformed segment: {seg}")
            is_valid = False
            continue
        start, end = seg.split(" - ")
        edges.append((start.strip(), end.strip()))

    if not edges:
        return False

    start_counts = {}
    for s, _ in edges:
        start_counts[s] = start_counts.get(s, 0) + 1
    for node, count in start_counts.items():
        if count > 1:
            print(f"WARNING: Branching at '{node}' ({count} times).")
            is_valid = False

    all_starts = [s for (s, _) in edges]
    all_ends = [e for (_, e) in edges]
    possible_starts = set(all_starts) - set(all_ends)
    if len(possible_starts) != 1:
        print(f"WARNING: No unique start. Found: {possible_starts}")
        is_valid = False

    if possible_starts:
        chain_start = list(possible_starts)[0]
    else:
        chain_start = edges[0][0]

    adjacency = {}
    for s, e in edges:
        if s in adjacency and adjacency[s] != e:
            continue
        adjacency[s] = e

    chain_stops = [chain_start]
    used_edges_count = 0
    while chain_stops[-1] in adjacency:
        next_stop = adjacency[chain_stops[-1]]
        chain_stops.append(next_stop)
        used_edges_count += 1
        if used_edges_count > len(edges):
            print("WARNING: Loop detected.")
            is_valid = False
            break

    if used_edges_count < len(edges):
        print(f"WARNING: Not all segments used ({used_edges_count} of {len(edges)}).")
        is_valid = False

    return is_valid


def _norm(s: str) -> str:
    """Normalize a stop name: trim and collapse internal whitespace."""
    return " ".join((s or "").strip().split())


def sort_route_segments(segments: Iterable[str]) -> List[str]:
    """Return original segment labels in travel order when a unique chain exists.

    Uses normalized nodes internally to infer order, but returns the exact original
    SegmentName labels for compatibility with pivot-table columns. If the segments
    do not form a single simple path (e.g., branching, loops, ambiguity), the
    original order is returned unchanged.
    """
    original: List[str] = list(segments)
    if not original:
        return original

    # Build mapping from normalized edge -> original label
    norm_edges: List[Tuple[str, str]] = []
    label_for_edge: Dict[Tuple[str, str], str] = {}
    for seg in original:
        if " - " not in seg:
            continue
        a_raw, b_raw = seg.split(" - ", 1)
        a, b = _norm(a_raw), _norm(b_raw)
        e = (a, b)
        # Keep first-seen label for this normalized edge
        if e not in label_for_edge:
            label_for_edge[e] = seg
            norm_edges.append(e)

    if not norm_edges:
        return original

    # Build successor map and degree counts; reject branching immediately.
    succ: Dict[str, str] = {}
    indeg: Dict[str, int] = {}
    outdeg: Dict[str, int] = {}
    nodes: Set[str] = set()

    for a, b in norm_edges:
        nodes.update([a, b])
        if a in succ and succ[a] != b:
            # Branching (multiple distinct successors) -> ambiguous
            return original
        succ[a] = b
        outdeg[a] = outdeg.get(a, 0) + 1
        indeg[b] = indeg.get(b, 0) + 1
        indeg.setdefault(a, 0)
        outdeg.setdefault(b, 0)

    # Candidate starts: prefer outdeg - indeg == 1, else nodes with indeg == 0, else all nodes.
    starts: List[str] = (
        [n for n in nodes if outdeg.get(n, 0) - indeg.get(n, 0) == 1]
        or [n for n in nodes if indeg.get(n, 0) == 0]
        or list(nodes)
    )

    def walk(start: str) -> List[str]:
        """Return ordered original labels if a full simple path exists from start."""
        used: Set[Tuple[str, str]] = set()
        order: List[str] = []
        cur: str = start
        steps: int = 0
        while cur in succ:
            nxt: str = succ[cur]
            e: Tuple[str, str] = (cur, nxt)
            if e in used:
                # Loop detected
                return []
            used.add(e)
            order.append(label_for_edge[e])
            cur = nxt
            steps += 1
            if steps > len(norm_edges):
                # Safety break for unexpected cycles
                return []
        # Accept only if every edge was used exactly once
        return order if len(used) == len(norm_edges) else []

    # Try each candidate start; accept a unique full-coverage solution.
    solutions: List[List[str]] = []
    for s in starts:
        sol = walk(s)
        if sol:
            solutions.append(sol)

    if not solutions:
        return original
    if any(solutions[0] != s for s in solutions[1:]):
        # Multiple distinct valid chains -> ambiguous
        return original
    return solutions[0]


def create_and_save_pivots(
    df: pd.DataFrame,
    output_subdir: str,
    dataset_label: str,
    time_columns_map: dict,
) -> None:
    """Create pivot-table CSVs for each *(route, direction, time metric)*."""
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)

    for route in df["Branch"].unique():
        route_df = df[df["Branch"] == route].copy()
        route_wrote_any_csv = False

        for direction in route_df["Direction"].unique():
            direction_df = route_df[route_df["Direction"] == direction].copy()

            # Sort chronologically via Trip
            if "Trip" in direction_df.columns:
                direction_df["time_sort"] = direction_df["Trip"].apply(parse_trip_time)
                direction_df.sort_values("time_sort", inplace=True, na_position="last")
                direction_df.drop(columns="time_sort", inplace=True)

            sorted_idx = (
                direction_df[["Branch", "Direction", "TripNo", "Variation", "Trip"]]
                .drop_duplicates()
                .set_index(["Branch", "Direction", "TripNo", "Variation", "Trip"])
                .index
            )

            # Validate & order segments
            segments = direction_df["SegmentName"].dropna().unique().tolist()
            variations = direction_df["Variation"].dropna().unique().tolist()
            check_route_validity(segments, variations)  # still logs warnings
            segments = sort_route_segments(segments)  # always try sorting

            direction_wrote_something = False

            # Pivot each time column
            for time_col, suffix in time_columns_map.items():
                if time_col not in direction_df.columns:
                    continue

                pivot_tbl = direction_df.pivot_table(
                    index=["Branch", "Direction", "TripNo", "Variation", "Trip"],
                    columns="SegmentName",
                    values=time_col,
                    aggfunc="mean",
                )
                if pivot_tbl.empty:
                    continue

                pivot_tbl = pivot_tbl.reindex(
                    columns=[s for s in segments if s in pivot_tbl.columns]
                )
                pivot_tbl = pivot_tbl.reindex(index=sorted_idx)
                pivot_tbl = pivot_tbl.round(1)

                csv_name = f"{dataset_label}_Route{route}_Dir{direction}_{suffix}.csv"
                csv_path = os.path.join(output_subdir, csv_name)
                pivot_tbl.to_csv(csv_path, float_format="%.1f")
                print(f"Created {csv_path}")

                direction_wrote_something = True
                route_wrote_any_csv = True

            if not direction_wrote_something:
                no_data_name = f"{dataset_label}_Route{route}_Dir{direction}_NoData.csv"
                no_data_path = os.path.join(output_subdir, no_data_name)
                pd.DataFrame(
                    {"Info": [f"No valid data for route {route}, direction {direction}."]}
                ).to_csv(no_data_path, index=False)
                print(f"Created {no_data_path}")

        if not route_wrote_any_csv:
            no_data_name = f"{dataset_label}_Route{route}_NoData.csv"
            no_data_path = os.path.join(output_subdir, no_data_name)
            pd.DataFrame({"Info": [f"No valid data for route {route}."]}).to_csv(
                no_data_path, index=False
            )
            print(f"Created {no_data_path}")


def process_file(file_path: str, dataset_label: str) -> None:
    """Run the full workflow for one CLEVER export.

    Args:
        file_path: Path to the Weekday/Saturday/Sunday/Other CLEVER file.
        dataset_label: Token used in output names (must match subdirectory).

    Raises:
        ValueError: Propagated from :func:`load_data` for unsupported file
            extensions.
    """
    print(f"\nProcessing file: {file_path} (label='{dataset_label}')")
    df = load_data(file_path)
    df = filter_routes(df, ROUTES_TO_EXCLUDE, ROUTES_TO_INCLUDE)
    convert_time_columns(df, TIME_COLUMNS)

    if PARENT_OUTPUT_DIR:
        output_subdir = os.path.join(PARENT_OUTPUT_DIR, dataset_label)
    else:
        output_subdir = dataset_label

    create_and_save_pivots(df, output_subdir, dataset_label, TIME_COLUMNS)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Dispatch processing for each configured service period.

    File paths left blank in *CONFIGURATION* are skipped with a console notice.
    """
    if WKDY_FILE_PATH:
        process_file(WKDY_FILE_PATH, "wkdy")
    else:
        print("Skipping 'wkdy' (blank path).")

    if SAT_FILE_PATH:
        process_file(SAT_FILE_PATH, "sat")
    else:
        print("Skipping 'sat' (blank path).")

    if SUN_FILE_PATH:
        process_file(SUN_FILE_PATH, "sun")
    else:
        print("Skipping 'sun' (blank path).")

    if OTHER_FILE_PATH:
        process_file(OTHER_FILE_PATH, "other")
    else:
        print("Skipping 'other' (blank path).")


if __name__ == "__main__":
    main()
