"""GTFS Stop Impact Analyzer.

Parses GTFS data to evaluate the service impact of removing specific target routes
(defined in configuration). For every stop served by a target route, the script
determines if the stop is:

1. **Eliminated**: Served *only* by target routes (no alternative service exists).
2. **Impacted (Route Loss)**: Loses a target route but retains service from others.

The analysis accounts for specific service IDs (calendars) and output includes
Day-of-Week codes to identify if stops are eliminated only on specific days.

Output:
    Writes `stop_route_calendar_impacts.csv` containing stop-level summaries and
    per-service-id classifications.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

GTFS_DIR = Path(r"Path\To\GIS\Folder")
OUTPUT_DIR = Path(r"Path\To\Output_Folder")

# Match against route_short_name OR route_id.
TARGET_ROUTE_TOKENS = {"101"}

# Keep only platform stops (location_type 0 or blank) by default.
FILTER_TO_PLATFORM_STOPS = True

# Optional service_id filter:
# - None to include all service_ids
# - e.g. {"2","3","4"} to restrict analysis to those calendars only
SERVICE_ID_FILTER: set[str] | None = {"2", "3", "4"} # Replace with your values

OUTPUT_FILENAME = "stop_route_calendar_impacts.csv"

LOG_LEVEL = logging.INFO


# =============================================================================
# HELPERS
# =============================================================================


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing GTFS file: {path}")


def _read_gtfs_csv(gtfs_dir: Path, filename: str, usecols: list[str]) -> pd.DataFrame:
    path = gtfs_dir / filename
    _require_file(path)
    return pd.read_csv(path, dtype=str, usecols=usecols, low_memory=False)


def _as_sorted_csv(values: Iterable[str]) -> str:
    uniq = sorted({str(v) for v in values if v is not None and str(v).strip() != ""})
    return ",".join(uniq)


def _dow_code_from_calendar_row(row: pd.Series) -> str:
    """Return a code like 'M/T/W/R/F' or 'S' or 'U' based on calendar.txt flags."""
    mapping = [
        ("monday", "M"),
        ("tuesday", "T"),
        ("wednesday", "W"),
        ("thursday", "R"),
        ("friday", "F"),
        ("saturday", "S"),
        ("sunday", "U"),
    ]
    parts: list[str] = []
    for col, code in mapping:
        if str(row.get(col, "0")) == "1":
            parts.append(code)
    return "/".join(parts)


def _svc_ids_to_dow_list(service_ids_csv: str, svc_to_dow: dict[str, str]) -> str:
    """Convert '12,34' -> 'M/T/W/R/F,S' (best-effort; falls back to service_id if unknown)."""
    if not service_ids_csv:
        return ""
    out: list[str] = []
    for sid in [s.strip() for s in service_ids_csv.split(",") if s.strip()]:
        out.append(svc_to_dow.get(sid, sid))
    return ",".join(sorted(set(out)))


def _apply_service_id_filter_to_trips(
    trips: pd.DataFrame, service_filter: set[str] | None
) -> pd.DataFrame:
    """Optionally filter trips to a subset of service_ids."""
    if service_filter is None:
        return trips
    keep = {str(x) for x in service_filter}
    svc = trips["service_id"].astype(str)
    out = trips[svc.isin(keep)].copy()
    logging.info(
        "Service_id filter enabled: keeping %d/%d trips (service_ids=%s)",
        len(out),
        len(trips),
        ",".join(sorted(keep)),
    )
    return out


def _clean_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """Make the export more Excel-friendly: remove NaNs in object/text columns."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].fillna("")
    return out


# =============================================================================
# LOAD
# =============================================================================


def load_gtfs_tables(gtfs_dir: Path) -> dict[str, Optional[pd.DataFrame]]:
    """Load required GTFS files (stops, routes, trips, stop_times, calendar) from directory.

    Args:
        gtfs_dir: Path to the directory containing GTFS text files.

    Returns:
        A dictionary containing loaded DataFrames for each file key.
    """
    stops = _read_gtfs_csv(
        gtfs_dir,
        "stops.txt",
        usecols=[
            "stop_id",
            "stop_name",
            "stop_code",
            "stop_lat",
            "stop_lon",
            "location_type",
        ],
    ).drop_duplicates(subset=["stop_id"])

    routes = _read_gtfs_csv(
        gtfs_dir,
        "routes.txt",
        usecols=["route_id", "route_short_name", "route_long_name"],
    ).drop_duplicates(subset=["route_id"])

    trips = _read_gtfs_csv(
        gtfs_dir,
        "trips.txt",
        usecols=["trip_id", "route_id", "service_id"],
    ).drop_duplicates(subset=["trip_id"])

    stop_times = _read_gtfs_csv(
        gtfs_dir,
        "stop_times.txt",
        usecols=["trip_id", "stop_id"],
    )

    calendar = None
    cal_path = gtfs_dir / "calendar.txt"
    if cal_path.exists():
        calendar = _read_gtfs_csv(
            gtfs_dir,
            "calendar.txt",
            usecols=[
                "service_id",
                "monday",
                "tuesday",
                "wednesday",
                "thursday",
                "friday",
                "saturday",
                "sunday",
            ],
        ).drop_duplicates(subset=["service_id"])

    return {
        "stops": stops,
        "routes": routes,
        "trips": trips,
        "stop_times": stop_times,
        "calendar": calendar,
    }


# =============================================================================
# CORE
# =============================================================================


def identify_target_route_ids(routes: pd.DataFrame, tokens: set[str]) -> set[str]:
    """Resolve target route IDs from a set of route tokens (short names or IDs).

    Args:
        routes: The routes DataFrame.
        tokens: A set of strings to match against route_short_name or route_id.

    Returns:
        A set of resolved route_id strings found in the GTFS data.
    """
    r = routes.copy()
    r["route_short_name"] = r["route_short_name"].fillna("").astype(str).str.strip()
    r["route_id"] = r["route_id"].fillna("").astype(str).str.strip()

    mask = r["route_short_name"].isin(tokens) | r["route_id"].isin(tokens)
    target_ids = set(r.loc[mask, "route_id"].astype(str).tolist())

    found_tokens = set(
        r.loc[r["route_short_name"].isin(tokens), "route_short_name"].tolist()
    ) | set(r.loc[r["route_id"].isin(tokens), "route_id"].tolist())
    missing = sorted(tokens - found_tokens)
    if missing:
        logging.warning(
            "Some target tokens were not found in routes.txt (route_short_name or route_id): %s",
            ", ".join(missing),
        )

    logging.info(
        "Target route_ids resolved: %s",
        ", ".join(sorted(target_ids)) if target_ids else "(none)",
    )
    return target_ids


def build_stop_service_routes(
    stop_times: pd.DataFrame,
    trips: pd.DataFrame,
    routes: pd.DataFrame,
) -> pd.DataFrame:
    """Map every (stop_id, service_id) pair to the list of routes serving it.

    Args:
        stop_times: The stop_times DataFrame.
        trips: The trips DataFrame (optionally filtered).
        routes: The routes DataFrame.

    Returns:
        A DataFrame with unique (stop_id, service_id) rows and a list of route IDs/labels.
    """
    merged = stop_times.merge(trips, on="trip_id", how="inner", validate="many_to_one")
    merged = merged.merge(routes, on="route_id", how="left", validate="many_to_one")

    rsn = merged["route_short_name"].fillna("").astype(str).str.strip()
    merged["route_label"] = np.where(
        rsn.str.len() > 0, rsn, merged["route_id"].astype(str)
    )

    dedup = merged[
        ["stop_id", "service_id", "route_id", "route_label"]
    ].drop_duplicates()

    route_id_arr = (
        dedup.groupby(["stop_id", "service_id"])["route_id"]
        .unique()
        .reset_index(name="route_id_arr")
    )
    route_label_arr = (
        dedup.groupby(["stop_id", "service_id"])["route_label"]
        .unique()
        .reset_index(name="route_label_arr")
    )

    return route_id_arr.merge(
        route_label_arr,
        on=["stop_id", "service_id"],
        how="inner",
        validate="one_to_one",
    )


def attach_calendar_info(
    df: pd.DataFrame,
    calendar: Optional[pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Attach only day-of-week code; do not attach start/end dates or exceptions."""
    out = df.copy()
    svc_to_dow: dict[str, str] = {}

    if calendar is not None and not calendar.empty:
        cal = calendar.copy()
        cal["dow_code"] = cal.apply(_dow_code_from_calendar_row, axis=1)
        svc_to_dow = dict(
            zip(cal["service_id"].astype(str), cal["dow_code"].astype(str), strict=True)
        )
        out = out.merge(
            cal[["service_id", "dow_code"]],
            on="service_id",
            how="left",
            validate="many_to_one",
        )
    else:
        out["dow_code"] = ""
        logging.warning(
            "calendar.txt missing/empty; dow_code will be blank (days conversion will fall back)."
        )

    return out, svc_to_dow


def classify_impacts(
    stop_service_routes: pd.DataFrame,
    stops: pd.DataFrame,
    target_route_ids: set[str],
) -> pd.DataFrame:
    """Classify each stop-service pair based on whether it is served by target routes.

    classifications:
    - not_target: Not served by any target route.
    - target_only: Served ONLY by target routes (eliminated).
    - target_plus_other: Served by target routes AND other routes (route loss).

    Args:
        stop_service_routes: DataFrame mapping stops/services to routes.
        stops: The stops DataFrame (for attaching location info).
        target_route_ids: The set of route IDs considered 'target' (to be removed).

    Returns:
        DataFrame with 'classification' column and stop details attached.
    """
    out = stop_service_routes.copy()

    out["route_id_set"] = out["route_id_arr"].apply(lambda a: set(map(str, a.tolist())))
    out["target_route_ids_present"] = out["route_id_set"].apply(
        lambda s: sorted(s & target_route_ids)
    )
    out["other_route_ids_present"] = out["route_id_set"].apply(
        lambda s: sorted(s - target_route_ids)
    )
    out["served_by_any_target"] = out["target_route_ids_present"].apply(
        lambda lst: len(lst) > 0
    )

    def _classify_row(row: pd.Series) -> str:
        if not bool(row["served_by_any_target"]):
            return "not_target"
        if len(row["other_route_ids_present"]) == 0:
            return "target_only"
        return "target_plus_other"

    out["classification"] = out.apply(_classify_row, axis=1)

    # Stop attributes (keep lon/lat; drop empty parent/location fields entirely)
    s = stops.copy()
    s["location_type"] = s["location_type"].fillna("").astype(str)

    if FILTER_TO_PLATFORM_STOPS:
        s = s[s["location_type"].isin(["", "0"])].copy()

    out = out.merge(
        s[["stop_id", "stop_name", "stop_code", "stop_lon", "stop_lat"]],
        on="stop_id",
        how="left",
        validate="many_to_one",
    )

    # Single route list column for users (e.g., 101,202,622)
    out["routes_serving_stop"] = out["route_label_arr"].apply(_as_sorted_csv)

    out = out.drop(columns=["route_id_set"])
    return out


def add_stop_level_summary_columns(
    flagged: pd.DataFrame, svc_to_dow: dict[str, str]
) -> pd.DataFrame:
    """Compute stop-level summary columns and merge back onto each row (single-file workflow)."""

    def _svc_list(sub: pd.DataFrame, cls: str) -> str:
        svc = sorted(
            sub.loc[sub["classification"] == cls, "service_id"]
            .astype(str)
            .unique()
            .tolist()
        )
        return ",".join(svc)

    rows: list[dict[str, object]] = []
    for stop_id, sub in flagged.groupby("stop_id", sort=False):
        svc_only = _svc_list(sub, "target_only")
        svc_plus = _svc_list(sub, "target_plus_other")

        has_plus_other = bool((sub["classification"] == "target_plus_other").any())
        impact_category = "route_loss_only" if has_plus_other else "eliminated"

        rows.append(
            {
                "stop_id": stop_id,
                "impact_category": impact_category,
                # clearer names:
                "has_eliminated_days": bool(
                    (sub["classification"] == "target_only").any()
                ),
                "has_route_loss_days": has_plus_other,
                "service_ids_eliminated": svc_only,
                "service_ids_route_loss": svc_plus,
                "service_days_eliminated": _svc_ids_to_dow_list(svc_only, svc_to_dow),
                "service_days_route_loss": _svc_ids_to_dow_list(svc_plus, svc_to_dow),
            }
        )

    summary = pd.DataFrame(rows)
    return flagged.merge(summary, on="stop_id", how="left", validate="many_to_one")


def log_unique_stop_impacts(flagged: pd.DataFrame, stops_universe: pd.DataFrame) -> None:
    """Log unique-stop percentages (no double counting across service_ids)."""
    universe_stop_ids = set(stops_universe["stop_id"].astype(str).unique().tolist())
    total_universe = len(universe_stop_ids)

    flagged_stop_ids = set(flagged["stop_id"].astype(str).unique().tolist())
    total_affected = len(flagged_stop_ids)

    stop_has_plus_other = (
        flagged.groupby("stop_id")["classification"]
        .apply(lambda s: (s == "target_plus_other").any())
        .to_dict()
    )
    impacted_by_route_loss = {sid for sid, has in stop_has_plus_other.items() if has}
    eliminated_altogether = {sid for sid, has in stop_has_plus_other.items() if not has}

    def _pct(n: int, d: int) -> float:
        return 0.0 if d == 0 else (100.0 * n / d)

    logging.info(
        "Unique platform stops in universe: %d; unique stops affected by targets: %d "
        "(%.2f%% of universe)",
        total_universe,
        total_affected,
        _pct(total_affected, total_universe),
    )
    logging.info(
        "Unique stops impacted (route loss, still served by other routes): %d "
        "(%.2f%% of universe; %.2f%% of affected)",
        len(impacted_by_route_loss),
        _pct(len(impacted_by_route_loss), total_universe),
        _pct(len(impacted_by_route_loss), total_affected),
    )
    logging.info(
        "Unique stops eliminated (only target routes served them): %d "
        "(%.2f%% of universe; %.2f%% of affected)",
        len(eliminated_altogether),
        _pct(len(eliminated_altogether), total_universe),
        _pct(len(eliminated_altogether), total_affected),
    )


def write_single_output(df: pd.DataFrame, output_dir: Path, filename: str) -> None:
    """Write the resulting DataFrame to a CSV file.

    Args:
        df: The DataFrame to write.
        output_dir: The directory to write to (will be created if needed).
        filename: The name of the output CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    df.to_csv(out_path, index=False)
    logging.info("Wrote: %s (%d rows)", out_path, len(df))


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Main execution function."""
    logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s | %(message)s")

    logging.info("Loading GTFS tables from: %s", GTFS_DIR)
    t = load_gtfs_tables(GTFS_DIR)

    stops = t["stops"]
    routes = t["routes"]
    trips = t["trips"]
    stop_times = t["stop_times"]
    calendar = t["calendar"]

    assert isinstance(stops, pd.DataFrame)
    assert isinstance(routes, pd.DataFrame)
    assert isinstance(trips, pd.DataFrame)
    assert isinstance(stop_times, pd.DataFrame)

    # Universe for percent calcs (platform stops by default)
    stops_universe = stops.copy()
    stops_universe["location_type"] = stops_universe["location_type"].fillna("").astype(str)
    if FILTER_TO_PLATFORM_STOPS:
        stops_universe = stops_universe[
            stops_universe["location_type"].isin(["", "0"])
        ].copy()

    logging.info("Resolving target routes …")
    target_route_ids = identify_target_route_ids(
        routes=routes, tokens=set(TARGET_ROUTE_TOKENS)
    )
    if not target_route_ids:
        raise ValueError(
            "No target routes found. Check TARGET_ROUTE_TOKENS vs routes.txt fields."
        )

    trips_f = _apply_service_id_filter_to_trips(trips, SERVICE_ID_FILTER)
    if trips_f.empty:
        raise ValueError(
            "After applying SERVICE_ID_FILTER, no trips remain. "
            "Check that the service_ids exist in trips.txt."
        )

    logging.info("Building stop/service -> routes mapping …")
    stop_service_routes = build_stop_service_routes(
        stop_times=stop_times, trips=trips_f, routes=routes
    )

    logging.info("Classifying stop impacts …")
    classified = classify_impacts(
        stop_service_routes=stop_service_routes,
        stops=stops,
        target_route_ids=target_route_ids,
    )

    logging.info("Attaching calendar info …")
    classified, svc_to_dow = attach_calendar_info(
        classified, calendar=calendar if isinstance(calendar, pd.DataFrame) else None
    )

    # Keep only rows served by at least one target route
    flagged = classified[classified["served_by_any_target"]].copy()

    log_unique_stop_impacts(flagged, stops_universe)

    # Add stop-level summary columns onto each row (single file)
    flagged = add_stop_level_summary_columns(flagged, svc_to_dow)

    # Drop internal/debug columns and route-id splits (keep ONE route list column)
    flagged = flagged.drop(
        columns=[
            "route_id_arr",
            "route_label_arr",
            "served_by_any_target",
            "target_route_ids_present",
            "other_route_ids_present",
            "route_ids_serving_stop",
        ],
        errors="ignore",
    )

    # Clean NaNs in text fields
    flagged = _clean_for_export(flagged)

    # Impact category ordering + sort
    impact_order = pd.CategoricalDtype(
        categories=["eliminated", "route_loss_only"], ordered=True
    )
    flagged["impact_category"] = flagged["impact_category"].astype(impact_order)

    if "stop_name" in flagged.columns:
        flagged["stop_name"] = flagged["stop_name"].astype(str)

    flagged = flagged.sort_values(
        by=["impact_category", "stop_name", "stop_id", "service_id"],
        ascending=[True, True, True, True],
        kind="mergesort",
    )

    # Column ordering
    preferred_cols = [
        "impact_category",
        "stop_id",
        "stop_name",
        "stop_code",
        "stop_lon",
        "stop_lat",
        "service_id",
        "dow_code",
        "classification",
        "routes_serving_stop",
        # stop-level summary fields (repeated per row)
        "has_eliminated_days",
        "has_route_loss_days",
        "service_ids_eliminated",
        "service_ids_route_loss",
        "service_days_eliminated",
        "service_days_route_loss",
    ]
    cols = [c for c in preferred_cols if c in flagged.columns] + [
        c for c in flagged.columns if c not in preferred_cols
    ]
    flagged = flagged[cols]

    logging.info(
        "Flagged rows (stop_id + service_id with target service): %d", len(flagged)
    )
    write_single_output(flagged, OUTPUT_DIR, OUTPUT_FILENAME)

    logging.info("Done ✔")


if __name__ == "__main__":
    main()
