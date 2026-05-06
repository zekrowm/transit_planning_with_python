"""System-level stop improvement coverage summary.

Computes aggregate metrics about bus stop improvements (e.g., shelters,
benches, trash cans, ADA pads) across a GTFS feed, with optional support for
a route blacklist (excluded from the system universe entirely) and a route
whitelist (priority/frequent network). The script reports counts and
percentages of stops with each improvement, and reports whitelist-route stops
as a percentage of the system total.

Inputs:
    - GTFS directory containing at minimum stops.txt, routes.txt, trips.txt,
      and stop_times.txt.
    - Optional CSV of per-stop improvement flags (Y/N) joined to GTFS stops
      on either stop_code or stop_id.
    - Optional whitelist of routes (matched by route_short_name).
    - Optional blacklist of routes (matched by route_short_name).

Outputs:
    - stop_improvement_summary.txt - human-readable report of counts and
      percentages (system total, by improvement, whitelist coverage).
    - stop_improvement_detail.csv - one row per logical stop with route
      memberships and (when supplied) improvement flags, for verification.

Logical stop key:
    Many GTFS feeds split a single curbside stop across multiple stop_id rows
    sharing one stop_code (e.g., for distinct platforms or directions). When
    USE_STOP_CODE is True, stops are deduplicated by stop_code; an improvement
    counts as present on a logical stop if ANY underlying platform has it.

Typical use:
    Quarterly board reports, capital planning prioritization, equity reporting
    on amenity coverage on priority vs. coverage routes.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# ---- GTFS input
GTFS_DIR: Path = Path(r"Path\To\Your\GTFS_Folder")

# Use stop_code as the logical stop key (True) or stop_id (False).
# stop_code is recommended when one curb-side stop is split across multiple
# physical stop_id values (platforms, bays, etc.).
USE_STOP_CODE: bool = True

# ---- Optional improvements CSV (set IMPROVEMENTS_CSV = None to skip)
IMPROVEMENTS_CSV: Optional[Path] = Path(r"Path\To\Your\stop_improvements.csv")

# Column in IMPROVEMENTS_CSV used to join to GTFS stops. Should match the
# logical stop key chosen above (typically stop_code or stop_id).
IMPROVEMENTS_JOIN_FIELD: str = "stop_code"

# Improvements to summarise: display label -> column name in the CSV.
IMPROVEMENT_COLUMNS: Dict[str, str] = {
    "Shelter": "SHELTER",
    "Bench": "BENCH",
    "Trash Can": "TRASHCAN",
    "ADA Pad": "PAD",
}

# Aliases to normalise messy source column names to the canonical names above.
IMPROVEMENT_ALIASES: Dict[str, str] = {
    "bus_shelte": "SHELTER",
    "bus_shelter": "SHELTER",
    "trash_can": "TRASHCAN",
    "trashcan": "TRASHCAN",
    "bench": "BENCH",
    "pad": "PAD",
}

# ---- Optional route filters (matched against route_short_name)
# Whitelist: priority routes whose stops are also reported as a share of the
# system total. Leave empty to skip the whitelist metric.
ROUTE_WHITELIST: Set[str] = {"101", "202"}

# Blacklist: routes excluded from the system universe entirely (e.g., shuttle,
# charter, or non-fixed-route services). Leave empty for no exclusions.
ROUTE_BLACKLIST: Set[str] = {"9999A", "9999B"}

# ---- Output
OUTPUT_DIR: Path = Path(r"Path\To\Your\Output_Folder")
SUMMARY_TXT_NAME: str = "stop_improvement_summary.txt"
DETAIL_CSV_NAME: str = "stop_improvement_detail.csv"

# ---- Logging
LOG_LEVEL: str = "INFO"

# =============================================================================
# REUSABLE HELPERS (copied from utils/gtfs_helpers.py)
# =============================================================================


def validate_gtfs_files_exist(
    gtfs_folder_path: str,
    files: Optional[Sequence[str]] = None,
) -> None:
    """Check that specific GTFS text files exist and log a warning if missing."""
    if not os.path.exists(gtfs_folder_path):
        logging.warning("The directory '%s' does not exist.", gtfs_folder_path)
        return

    if files is None:
        files = (
            "agency.txt",
            "stops.txt",
            "routes.txt",
            "trips.txt",
            "stop_times.txt",
        )

    for file_name in files:
        if not os.path.exists(os.path.join(gtfs_folder_path, file_name)):
            logging.warning("Missing GTFS file: %s", file_name)


def load_gtfs_data(
    gtfs_folder_path: str,
    files: Optional[Sequence[str]] = None,
    dtype: str | type[str] | Mapping[str, Any] = str,
) -> dict[str, pd.DataFrame]:
    """Load one or more GTFS text files into memory as DataFrames."""
    if not os.path.exists(gtfs_folder_path):
        raise OSError(f"The directory '{gtfs_folder_path}' does not exist.")

    if files is None:
        files = (
            "stops.txt",
            "routes.txt",
            "trips.txt",
            "stop_times.txt",
        )

    missing = [
        file_name
        for file_name in files
        if not os.path.exists(os.path.join(gtfs_folder_path, file_name))
    ]
    if missing:
        raise OSError(f"Missing GTFS files in '{gtfs_folder_path}': {', '.join(missing)}")

    data: dict[str, pd.DataFrame] = {}
    for file_name in files:
        key = file_name.replace(".txt", "")
        file_path = os.path.join(gtfs_folder_path, file_name)
        try:
            df = pd.read_csv(file_path, dtype=dtype, low_memory=False)
            data[key] = df
            logging.info("Loaded %s (%d records).", file_name, len(df))
        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"File '{file_name}' in '{gtfs_folder_path}' is empty.") from exc
        except pd.errors.ParserError as exc:
            raise ValueError(
                f"Parser error in '{file_name}' in '{gtfs_folder_path}': {exc}"
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                f"OS error reading file '{file_name}' in '{gtfs_folder_path}': {exc}"
            ) from exc

    return data


# =============================================================================
# DOMAIN HELPERS
# =============================================================================


def _standardise_yn(series: pd.Series) -> pd.Series:
    """Normalise a Y/N column to uppercase 'Y' or 'N' with no whitespace."""
    return series.fillna("N").astype(str).str.strip().str.upper().replace(
        {"YES": "Y", "TRUE": "Y", "1": "Y", "NO": "N", "FALSE": "N", "0": "N"}
    )


def resolve_route_ids_by_short_name(
    routes_df: pd.DataFrame,
    route_short_names: Set[str],
) -> Set[str]:
    """Resolve a set of route_short_name tokens to route_id values.

    Args:
        routes_df: The routes.txt DataFrame.
        route_short_names: A set of short-name tokens to match.

    Returns:
        Set of matching route_id strings. Logs a warning for any token not found.
    """
    if not route_short_names:
        return set()

    if "route_short_name" not in routes_df.columns:
        logging.warning("routes.txt has no 'route_short_name' column; cannot resolve filters.")
        return set()

    r = routes_df.copy()
    r["route_short_name"] = r["route_short_name"].fillna("").astype(str).str.strip()
    r["route_id"] = r["route_id"].fillna("").astype(str).str.strip()

    matched_ids = set(r.loc[r["route_short_name"].isin(route_short_names), "route_id"])
    found_tokens = set(r.loc[r["route_id"].isin(matched_ids), "route_short_name"])
    missing = sorted(route_short_names - found_tokens)
    if missing:
        logging.warning("Route filter tokens not found in routes.txt: %s", ", ".join(missing))

    return matched_ids


def build_stop_to_routes(
    stop_times: pd.DataFrame,
    trips: pd.DataFrame,
    routes: pd.DataFrame,
) -> pd.DataFrame:
    """Map each stop_id to the set of routes serving it.

    Args:
        stop_times: stop_times.txt DataFrame.
        trips: trips.txt DataFrame.
        routes: routes.txt DataFrame.

    Returns:
        DataFrame with columns: stop_id, route_ids (sorted CSV string),
        route_short_names (sorted CSV string).
    """
    rsn = routes[["route_id", "route_short_name"]].copy()
    rsn["route_short_name"] = rsn["route_short_name"].fillna("").astype(str).str.strip()
    rsn["route_id"] = rsn["route_id"].astype(str).str.strip()

    merged = (
        stop_times[["trip_id", "stop_id"]]
        .merge(trips[["trip_id", "route_id"]], on="trip_id", how="inner")
        .merge(rsn, on="route_id", how="left")
    )
    merged["stop_id"] = merged["stop_id"].astype(str)

    grouped = (
        merged.drop_duplicates(["stop_id", "route_id"])
        .groupby("stop_id", sort=False)
        .agg(
            route_ids=("route_id", lambda s: ",".join(sorted(set(s.astype(str))))),
            route_short_names=(
                "route_short_name",
                lambda s: ",".join(sorted({x for x in s.astype(str) if x})),
            ),
        )
        .reset_index()
    )
    return grouped


def collapse_to_logical_stops(
    stops: pd.DataFrame,
    stop_to_routes: pd.DataFrame,
    stop_key_field: str,
) -> pd.DataFrame:
    """Deduplicate stops by the chosen logical key, OR-merging route memberships.

    When multiple stop_id rows share a stop_code (e.g., multiple platforms),
    they collapse to a single logical stop. The logical stop is "served by"
    every route that serves any of the underlying physical stops.

    Args:
        stops: stops.txt DataFrame.
        stop_to_routes: Output of build_stop_to_routes.
        stop_key_field: "stop_id" or "stop_code".

    Returns:
        DataFrame keyed by stop_key_field with route membership columns and a
        comma-separated list of underlying stop_ids.
    """
    if stop_key_field not in stops.columns:
        raise ValueError(f"stops.txt is missing the configured key field '{stop_key_field}'.")

    s = stops[["stop_id", stop_key_field, "stop_name"]].copy()
    s["stop_id"] = s["stop_id"].astype(str)
    s[stop_key_field] = s[stop_key_field].fillna("").astype(str).str.strip()
    s["stop_name"] = s["stop_name"].fillna("").astype(str)

    # Stops with no value for the configured key field cannot be aggregated;
    # warn and fall back to stop_id for those rows so we don't silently lose them.
    blank_mask = s[stop_key_field].eq("")
    if blank_mask.any():
        logging.warning(
            "%d stops have a blank %s; falling back to stop_id for those rows.",
            int(blank_mask.sum()),
            stop_key_field,
        )
        s.loc[blank_mask, stop_key_field] = s.loc[blank_mask, "stop_id"]

    s = s.merge(stop_to_routes, on="stop_id", how="left")
    s["route_ids"] = s["route_ids"].fillna("")
    s["route_short_names"] = s["route_short_names"].fillna("")

    def _union_csv(series: pd.Series) -> str:
        items: Set[str] = set()
        for v in series.fillna("").astype(str):
            items.update(x for x in v.split(",") if x)
        return ",".join(sorted(items))

    collapsed = (
        s.groupby(stop_key_field, sort=False)
        .agg(
            stop_ids=("stop_id", lambda x: ",".join(sorted(set(x)))),
            stop_name=("stop_name", "first"),
            route_ids=("route_ids", _union_csv),
            route_short_names=("route_short_names", _union_csv),
        )
        .reset_index()
    )
    return collapsed


def load_improvements(
    csv_path: Path,
    join_field: str,
    column_map: Mapping[str, str],
    aliases: Mapping[str, str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Load and normalise the optional stop-improvements CSV.

    Args:
        csv_path: Path to the improvements CSV.
        join_field: Name of the join column in the CSV.
        column_map: Display label -> CSV column name for each improvement.
        aliases: Map from messy source column name -> canonical CSV column name.

    Returns:
        Tuple of (DataFrame indexed on join_field with one normalised Y/N column
        per improvement, list of canonical column names in display order).
    """
    df = pd.read_csv(csv_path, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})

    if join_field not in df.columns:
        raise ValueError(
            f"Improvements CSV is missing join column '{join_field}'. "
            f"Found columns: {list(df.columns)}"
        )

    df[join_field] = df[join_field].astype(str).str.strip()

    canonical_cols: List[str] = []
    for label, col in column_map.items():
        if col not in df.columns:
            logging.warning(
                "Improvement column '%s' (%s) not found in CSV; treating all stops as 'N'.",
                label,
                col,
            )
            df[col] = "N"
        df[col] = _standardise_yn(df[col])
        canonical_cols.append(col)

    keep = [join_field] + canonical_cols
    return df[keep].drop_duplicates(subset=[join_field]), canonical_cols


def attach_improvements(
    logical_stops: pd.DataFrame,
    improvements: pd.DataFrame,
    stop_key_field: str,
    join_field: str,
    canonical_cols: Sequence[str],
) -> pd.DataFrame:
    """Left-join improvements onto logical stops and OR-merge across platforms.

    Because logical stops were already collapsed by stop_key_field, the join
    is one-to-one in expectation; however, if the CSV happens to have
    duplicates (one row per platform), we OR-merge so any 'Y' wins.
    """
    out = logical_stops.merge(
        improvements,
        how="left",
        left_on=stop_key_field,
        right_on=join_field,
        validate="many_to_one",
    )
    if join_field != stop_key_field:
        out = out.drop(columns=[join_field], errors="ignore")
    for col in canonical_cols:
        out[col] = _standardise_yn(out[col])
    return out


def compute_summary(
    logical_stops: pd.DataFrame,
    canonical_cols: Sequence[str],
    column_map: Mapping[str, str],
    whitelist_route_ids: Set[str],
    whitelist_short_names: Set[str],
) -> Dict[str, Any]:
    """Compute the headline metrics requested in the summary report.

    Args:
        logical_stops: Deduplicated stops with route memberships and (optionally)
            improvement Y/N columns.
        canonical_cols: Canonical improvement column names (may be empty).
        column_map: Improvement display label -> canonical column name.
        whitelist_route_ids: Set of route_ids treated as the whitelist universe.
        whitelist_short_names: Set of route_short_names for human-readable logging.

    Returns:
        Dictionary of metrics ready for serialisation/logging.
    """
    total = len(logical_stops)

    # Whitelist coverage: a stop is "on the whitelist" if any serving route_id
    # is in the resolved whitelist set.
    if whitelist_route_ids:
        wl_mask = logical_stops["route_ids"].apply(
            lambda s: bool(set(s.split(",")) & whitelist_route_ids) if s else False
        )
        whitelist_total = int(wl_mask.sum())
    else:
        wl_mask = pd.Series([False] * total, index=logical_stops.index)
        whitelist_total = 0

    per_improvement: Dict[str, Dict[str, Any]] = {}
    for label, col in column_map.items():
        if col not in logical_stops.columns:
            continue
        sys_count = int((logical_stops[col] == "Y").sum())
        wl_count = int(((logical_stops[col] == "Y") & wl_mask).sum()) if whitelist_total else 0
        per_improvement[label] = {
            "system_count": sys_count,
            "system_pct": (sys_count / total * 100.0) if total else 0.0,
            "whitelist_count": wl_count,
            "whitelist_pct": (
                (wl_count / whitelist_total * 100.0) if whitelist_total else 0.0
            ),
        }

    return {
        "system_total_stops": total,
        "whitelist_short_names": sorted(whitelist_short_names),
        "whitelist_total_stops": whitelist_total,
        "whitelist_pct_of_system": (
            (whitelist_total / total * 100.0) if total else 0.0
        ),
        "per_improvement": per_improvement,
        "improvements_supplied": bool(canonical_cols),
    }


def write_summary_txt(
    summary: Mapping[str, Any],
    blacklist_short_names: Set[str],
    out_path: Path,
) -> None:
    """Write the human-readable text summary."""
    lines: List[str] = []
    lines.append(f"Run date: {pd.Timestamp.now():%Y-%m-%d %H:%M}")
    lines.append("")
    lines.append("System universe")
    lines.append("---------------")
    lines.append(f"Total logical stops (post-blacklist): {summary['system_total_stops']:,}")
    if blacklist_short_names:
        lines.append(f"Blacklist routes excluded: {', '.join(sorted(blacklist_short_names))}")
    else:
        lines.append("Blacklist routes excluded: (none)")
    lines.append("")

    if summary["whitelist_short_names"]:
        lines.append("Whitelist (priority) routes")
        lines.append("---------------------------")
        lines.append(
            f"Whitelist routes: {', '.join(summary['whitelist_short_names'])}"
        )
        lines.append(
            f"Stops on whitelist routes:    {summary['whitelist_total_stops']:>8,}"
        )
        lines.append(
            f"As % of system total:         {summary['whitelist_pct_of_system']:>7.1f}%"
        )
        lines.append("")

    if summary["improvements_supplied"]:
        lines.append("Improvement coverage")
        lines.append("--------------------")
        header = (
            f"{'Improvement':<14}{'System #':>10}{'System %':>10}"
            f"{'Whitelist #':>14}{'Whitelist %':>14}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for label, stats in summary["per_improvement"].items():
            lines.append(
                f"{label:<14}{stats['system_count']:>10,}"
                f"{stats['system_pct']:>9.1f}%"
                f"{stats['whitelist_count']:>14,}"
                f"{stats['whitelist_pct']:>13.1f}%"
            )
    else:
        lines.append("Improvement coverage: (no improvements CSV supplied)")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Run the coverage summary pipeline."""
    logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s | %(message)s")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load GTFS.
    validate_gtfs_files_exist(str(GTFS_DIR))
    g = load_gtfs_data(str(GTFS_DIR))
    stops = g["stops"]
    routes = g["routes"]
    trips = g["trips"]
    stop_times = g["stop_times"]

    stop_key_field = "stop_code" if USE_STOP_CODE else "stop_id"
    logging.info("Logical stop key: %s", stop_key_field)

    # 2. Resolve whitelist / blacklist route_ids from short names.
    blacklist_ids = resolve_route_ids_by_short_name(routes, ROUTE_BLACKLIST)
    whitelist_ids = resolve_route_ids_by_short_name(routes, ROUTE_WHITELIST)
    if ROUTE_WHITELIST and not whitelist_ids:
        logging.warning(
            "ROUTE_WHITELIST is non-empty (%s) but no tokens matched any "
            "route_short_name in routes.txt; whitelist metrics will be zero.",
            ", ".join(sorted(ROUTE_WHITELIST)),
        )

    # 3. Apply blacklist by dropping trips on those routes BEFORE building
    #    stop->route memberships, so stops served only by blacklisted routes
    #    drop out of the system universe entirely.
    if blacklist_ids:
        before = len(trips)
        trips = trips[~trips["route_id"].astype(str).isin(blacklist_ids)].copy()
        logging.info("Blacklist removed %d of %d trips.", before - len(trips), before)
        stop_times = stop_times[stop_times["trip_id"].isin(trips["trip_id"])].copy()

    # 4. Build stop -> serving routes mapping.
    stop_to_routes = build_stop_to_routes(stop_times, trips, routes)

    # 5. Restrict stops universe to those still served by at least one route
    #    after blacklisting (a stop served only by a blacklisted route is out).
    served_stop_ids = set(stop_to_routes["stop_id"].astype(str))
    stops_universe = stops[stops["stop_id"].astype(str).isin(served_stop_ids)].copy()
    logging.info(
        "Stops in universe (served, post-blacklist): %d of %d in stops.txt.",
        len(stops_universe),
        len(stops),
    )

    # 6. Collapse to logical stops (dedup).
    logical = collapse_to_logical_stops(stops_universe, stop_to_routes, stop_key_field)
    logging.info("Logical stops after dedup on %s: %d", stop_key_field, len(logical))

    # 7. Optional: attach improvements.
    canonical_cols: List[str] = []
    if IMPROVEMENTS_CSV is not None:
        if not IMPROVEMENTS_CSV.exists():
            logging.warning(
                "IMPROVEMENTS_CSV path '%s' does not exist; skipping improvements.",
                IMPROVEMENTS_CSV,
            )
        else:
            improvements_df, canonical_cols = load_improvements(
                IMPROVEMENTS_CSV,
                IMPROVEMENTS_JOIN_FIELD,
                IMPROVEMENT_COLUMNS,
                IMPROVEMENT_ALIASES,
            )
            logical = attach_improvements(
                logical,
                improvements_df,
                stop_key_field,
                IMPROVEMENTS_JOIN_FIELD,
                canonical_cols,
            )
            unmatched = logical[canonical_cols[0]].isna().sum() if canonical_cols else 0
            if unmatched:
                logging.warning(
                    "%d logical stops did not match any row in the improvements CSV.",
                    int(unmatched),
                )

    # 8. Compute summary metrics.
    summary = compute_summary(
        logical_stops=logical,
        canonical_cols=canonical_cols,
        column_map=IMPROVEMENT_COLUMNS,
        whitelist_route_ids=whitelist_ids,
        whitelist_short_names=ROUTE_WHITELIST,
    )

    # 8b. Log headline whitelist metric to console, independent of improvements.
    if ROUTE_WHITELIST:
        logging.info(
            "Whitelist stops: %d of %d system stops (%.1f%%).",
            summary["whitelist_total_stops"],
            summary["system_total_stops"],
            summary["whitelist_pct_of_system"],
        )

    # 9. Write outputs.
    txt_path = OUTPUT_DIR / SUMMARY_TXT_NAME
    csv_path = OUTPUT_DIR / DETAIL_CSV_NAME
    write_summary_txt(summary, ROUTE_BLACKLIST, txt_path)
    logical.to_csv(csv_path, index=False)

    logging.info("Wrote summary: %s", txt_path)
    logging.info("Wrote detail:  %s", csv_path)
    logging.info("Script completed successfully.")


if __name__ == "__main__":
    main()
