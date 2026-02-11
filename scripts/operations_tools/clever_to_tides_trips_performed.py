"""Convert CLEVER export into TIDES-compliant trips_performed records.

This module reads a CLEVER “Event Runtime Analysis” report (CSV) and produces a
`trips_performed.csv` file that conforms to the TIDES `trips_performed` schema.
It normalizes common real-world export issues (inconsistent whitespace, AM/PM
timestamps, mixed null tokens) and applies schema-aligned data quality rules so
the output can be ingested by downstream validation and analytics pipelines.

Key behaviors:
- Parses scheduled and actual timestamps robustly (tolerates AM/PM and extra
  whitespace). Missing start/end times are permitted; rows are not dropped solely
  for partial timing data.
- Derives `service_date` from Scheduled Start Time when available, otherwise
  falls back to Actual Start Time. Rows that cannot be dated are excluded.
- Requires `vehicle_id` (as per schema). Rows with missing or unusable vehicle
  identifiers are excluded and counted in logs.
- Extracts `route_id` from the human-readable Route field by taking the token to
  the left of "-" and trimming whitespace (e.g., "301 - Telegraph Rd" -> "301").
- Preserves the CLEVER TripID as `trip_id_scheduled` when it matches the GTFS
  `trip_id`. `trip_id_performed` is chosen to remain unique within `service_date`
  (using the scheduled trip id when safe, otherwise a stable derived identifier).
- Optionally filters to a single CLEVER Trip Type (e.g., "Revenue"). When
  enabled, the module logs how many rows were removed by each other trip type.
- Maps CLEVER Trip Type values into the schema-constrained TIDES `trip_type`
  enumeration (e.g., "Revenue" -> "In service", "Pull-In" -> "Pullin").

The conversion is designed to be deterministic: given the same input file and
configuration, the output identifiers and records are stable across runs.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_CSV: Path = Path(r"Path\To\Event Runtime Analysis.csv")
OUTPUT_CSV: Path = Path(r"Path\To\trips_performed.csv")

# If set (e.g., "Revenue"), keeps only CLEVER Trip Type == this value.
# If None/blank, keeps everything and logs nothing about filtering.
KEEP_CLEVER_TRIP_TYPE: str | None = "Revenue"

# Column names that may vary between exports (edit as needed).
OPERATOR_COL: str = "Operator"

# Set these to match your export headers.
TRIP_START_STOP_COL: str | None = None  # e.g., "First Stop"
TRIP_END_STOP_COL: str | None = "Last Stop"  # e.g., "Last Stop"

# Direction mapping. Must be 0/1 if present; unmapped values -> NA.
DIRECTION_TEXT_TO_ID: dict[str, int] = {
    "NORTHBOUND": 0,
    "WESTBOUND": 0,
    "SOUTHBOUND": 1,
    "EASTBOUND": 1,
}

# CLEVER Trip Type -> TIDES schema trip_type enum
# Allowed enum values include: "In service", "Deadhead", "Pullout", "Pullin", ...
# (see schema for full list)
CLEVER_TO_TIDES_TRIP_TYPE: dict[str, str] = {
    "REVENUE": "In service",
    "DEADHEAD": "Deadhead",
    "LAYOVER": "Layover",
    "PULL-OUT": "Pullout",
    "PULL OUT": "Pullout",
    "PULL OUT ": "Pullout",
    "PULLOUT": "Pullout",
    "PULL-IN": "Pullin",
    "PULL IN": "Pullin",
    "PULLIN": "Pullin",
    # If you have agency-specific labels, map them here.
}

# Output columns per schema (do NOT include "date"—schema does not define it).
TIDES_COLS: list[str] = [
    "service_date",
    "trip_id_performed",
    "vehicle_id",
    "trip_id_scheduled",
    "route_id",
    "route_type",
    "shape_id",
    "pattern_id",
    "direction_id",
    "operator_id",
    "block_id",
    "trip_start_stop_id",
    "trip_end_stop_id",
    "schedule_trip_start",
    "schedule_trip_end",
    "actual_trip_start",
    "actual_trip_end",
    "trip_type",
    "schedule_relationship",
    "ntd_mode",
    "route_type_agency",
]

# Required CLEVER columns for this converter (excluding optional operator/stops).
REQ_COLS: list[str] = [
    "Vehicle",
    "Route",
    "Direction",
    "Block",
    "TripID",
    "Trip Type",
    "Scheduled Start Time",
    "Scheduled Finish Time",
    "Actual Start Time",
    "Actual Finish Time",
]

# =============================================================================
# HELPERS
# =============================================================================


def normalize_dt_series(series: pd.Series) -> pd.Series:
    """Normalize datetime-like strings prior to pandas parsing."""
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "N/A": pd.NA})
    )


def parse_route_id_from_route_text(route_series: pd.Series) -> pd.Series:
    """Parse route_id from CLEVER 'Route' text left of '-' (trimmed)."""
    route_text = route_series.astype("string").fillna("").str.strip()
    left = route_text.str.split("-", n=1, expand=True)[0].str.strip()

    # Prefer a 1–4 digit token when present; otherwise keep the left token.
    digits = left.str.extract(r"^\s*([0-9]{1,4})\s*$")[0]
    return digits.fillna(left).replace({"": pd.NA})


def normalize_vehicle_id(vehicle_series: pd.Series) -> pd.Series:
    """Normalize vehicle_id values (handle floats like '7785.0', trim whitespace)."""
    raw = vehicle_series.astype("string").str.strip()
    return (
        raw.str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "<NA>": pd.NA})
    )


def stable_id(*parts: str) -> str:
    """Short deterministic ID from multiple fields (sha1 truncated)."""
    raw = "||".join(parts).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def direction_id_from_text(series: pd.Series) -> pd.Series:
    """Map Direction text to direction_id (nullable Int64)."""
    s = series.astype("string").str.strip().str.upper()
    mapped = s.map(DIRECTION_TEXT_TO_ID)
    return mapped.astype("Int64")


def dt_to_iso(series: pd.Series) -> pd.Series:
    """Datetime -> ISO string (YYYY-MM-DDTHH:MM:SS); NaT -> NaN."""
    return series.dt.strftime("%Y-%m-%dT%H:%M:%S")


def summarize_trip_type_drops(
    df: pd.DataFrame,
    trip_type_col: str,
    *,
    keep_trip_type: str | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Optionally filter to a single trip type and return drop counts by type."""
    if not keep_trip_type:
        return df, {}

    keep_norm = str(keep_trip_type).strip()
    if not keep_norm:
        return df, {}

    s = df[trip_type_col].astype("string").str.strip()
    mask_keep = s.eq(keep_norm)

    dropped = s.loc[~mask_keep].fillna("<NA>")
    dropped_counts = dropped.value_counts(dropna=False).to_dict()
    return df.loc[mask_keep].copy(), {str(k): int(v) for k, v in dropped_counts.items()}


def map_clever_trip_type_to_tides(series: pd.Series) -> pd.Series:
    """Map CLEVER Trip Type labels to schema-conformant TIDES trip_type values."""
    s = series.astype("string").str.strip()
    upper = s.str.upper()

    mapped = upper.map(CLEVER_TO_TIDES_TRIP_TYPE)

    # If not mapped:
    # - If original is missing -> NA
    # - Else default to "Other not in service" (schema enum)
    out = mapped.where(mapped.notna(), other="Other not in service")
    out = out.where(s.notna(), other=pd.NA)
    return out


def choose_trip_id_performed(
    service_date: pd.Series,
    trip_id_scheduled: pd.Series,
    vehicle_id: pd.Series,
    best_start_dt: pd.Series,
) -> tuple[pd.Series, int]:
    """Use GTFS trip_id as trip_id_performed when unique within service_date; else hash.

    Returns:
        (trip_id_performed_series, n_dupe_rows)
    """
    base = pd.DataFrame(
        {
            "service_date": service_date.astype("string"),
            "trip_id_scheduled": trip_id_scheduled.astype("string"),
        }
    )

    dup = base.duplicated(keep=False) & trip_id_scheduled.notna()

    best_start_str = best_start_dt.astype("datetime64[ns]").astype("string").fillna("")
    hashed = pd.Series(
        [
            f"perf_{stable_id(str(sd), str(tid), str(veh), str(bs))}"
            for sd, tid, veh, bs in zip(
                service_date.fillna(""),
                trip_id_scheduled.fillna(""),
                vehicle_id.fillna(""),
                best_start_str,
                strict=True,
            )
        ],
        index=service_date.index,
        dtype="string",
    )

    perf = trip_id_scheduled.where(~dup, other=hashed)
    return perf, int(dup.sum())


# =============================================================================
# CORE CONVERSION
# =============================================================================


def clever_to_tides(df: pd.DataFrame) -> pd.DataFrame:
    """Convert CLEVER Event Runtime Analysis export to TIDES trips_performed.csv."""
    df = df.dropna(how="all").copy()

    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required CLEVER columns: {missing}")

    # Optional CLEVER Trip Type filter (keep only Revenue, etc.)
    df, dropped_counts = summarize_trip_type_drops(
        df,
        "Trip Type",
        keep_trip_type=KEEP_CLEVER_TRIP_TYPE,
    )
    if dropped_counts:
        total_dropped = sum(dropped_counts.values())
        logging.warning(
            "Trip Type filter enabled (keep=%r): dropped %d rows of other types: %s",
            KEEP_CLEVER_TRIP_TYPE,
            total_dropped,
            dropped_counts,
        )

    if df.empty:
        logging.warning("No rows remain after Trip Type filtering.")
        return pd.DataFrame(columns=TIDES_COLS)

    # Parse datetimes robustly
    sched_start_dt = pd.to_datetime(
        normalize_dt_series(df["Scheduled Start Time"]),
        errors="coerce",
    )
    sched_end_dt = pd.to_datetime(
        normalize_dt_series(df["Scheduled Finish Time"]),
        errors="coerce",
    )
    actual_start_dt = pd.to_datetime(
        normalize_dt_series(df["Actual Start Time"]),
        errors="coerce",
    )
    actual_end_dt = pd.to_datetime(
        normalize_dt_series(df["Actual Finish Time"]),
        errors="coerce",
    )

    # Date the trip: prefer scheduled start; fallback to actual start
    best_start_dt = sched_start_dt.where(sched_start_dt.notna(), actual_start_dt)

    # Drop rows that cannot be dated at all
    mask_undated = best_start_dt.isna()
    if mask_undated.any():
        n_drop = int(mask_undated.sum())
        logging.warning(
            "Dropping %d rows: neither Scheduled Start Time nor Actual Start Time is parseable.",
            n_drop,
        )
        df = df.loc[~mask_undated].copy()
        sched_start_dt = sched_start_dt.loc[~mask_undated]
        sched_end_dt = sched_end_dt.loc[~mask_undated]
        actual_start_dt = actual_start_dt.loc[~mask_undated]
        actual_end_dt = actual_end_dt.loc[~mask_undated]
        best_start_dt = best_start_dt.loc[~mask_undated]

    # vehicle_id is required by schema
    vehicle_id = normalize_vehicle_id(df["Vehicle"])
    mask_no_vehicle = vehicle_id.isna()
    if mask_no_vehicle.any():
        n_drop = int(mask_no_vehicle.sum())
        logging.warning("Dropping %d rows: missing Vehicle / vehicle_id.", n_drop)
        df = df.loc[~mask_no_vehicle].copy()
        vehicle_id = vehicle_id.loc[~mask_no_vehicle]
        sched_start_dt = sched_start_dt.loc[~mask_no_vehicle]
        sched_end_dt = sched_end_dt.loc[~mask_no_vehicle]
        actual_start_dt = actual_start_dt.loc[~mask_no_vehicle]
        actual_end_dt = actual_end_dt.loc[~mask_no_vehicle]
        best_start_dt = best_start_dt.loc[~mask_no_vehicle]

    if df.empty:
        logging.warning("No rows remain after dropping missing vehicle_id.")
        return pd.DataFrame(columns=TIDES_COLS)

    out = pd.DataFrame(index=df.index)

    # Required schema fields
    out["service_date"] = best_start_dt.dt.date.astype("string")
    out["vehicle_id"] = vehicle_id

    # GTFS trip_id from CLEVER TripID (per your statement)
    out["trip_id_scheduled"] = df["TripID"].astype("string").str.strip().replace({"": pd.NA})

    # Per schema: trip_id_performed must be unique within service_date
    out["trip_id_performed"], n_dupe_rows = choose_trip_id_performed(
        out["service_date"],
        out["trip_id_scheduled"],
        out["vehicle_id"],
        best_start_dt,
    )
    if n_dupe_rows:
        logging.warning(
            "trip_id_scheduled duplicates within service_date for %d rows; "
            "used hashed trip_id_performed for those rows.",
            n_dupe_rows,
        )

    # Optional schema fields we can populate
    out["route_id"] = parse_route_id_from_route_text(df["Route"])
    out["direction_id"] = direction_id_from_text(df["Direction"])
    out["block_id"] = df["Block"].astype("string").str.strip().replace({"": pd.NA})

    if OPERATOR_COL in df.columns:
        out["operator_id"] = df[OPERATOR_COL].astype("string").str.strip().replace({"": pd.NA})
    else:
        out["operator_id"] = pd.NA

    # Stops (text IDs are fine)
    if TRIP_START_STOP_COL and TRIP_START_STOP_COL in df.columns:
        out["trip_start_stop_id"] = (
            df[TRIP_START_STOP_COL].astype("string").str.strip().replace({"": pd.NA})
        )
    else:
        out["trip_start_stop_id"] = pd.NA

    if TRIP_END_STOP_COL and TRIP_END_STOP_COL in df.columns:
        out["trip_end_stop_id"] = (
            df[TRIP_END_STOP_COL].astype("string").str.strip().replace({"": pd.NA})
        )
    else:
        out["trip_end_stop_id"] = pd.NA

    # Times (allowed to be missing)
    out["schedule_trip_start"] = dt_to_iso(sched_start_dt)
    out["schedule_trip_end"] = dt_to_iso(sched_end_dt)
    out["actual_trip_start"] = dt_to_iso(actual_start_dt)
    out["actual_trip_end"] = dt_to_iso(actual_end_dt)

    # trip_type must match schema enum; map from CLEVER Trip Type
    out["trip_type"] = map_clever_trip_type_to_tides(df["Trip Type"])

    # Schedule relationship enum
    out["schedule_relationship"] = "Scheduled"

    # Fields not available from this export (leave blank)
    out["route_type"] = pd.NA
    out["shape_id"] = pd.NA
    out["pattern_id"] = pd.NA
    out["ntd_mode"] = pd.NA
    out["route_type_agency"] = pd.NA

    # Final order (schema-defined columns; extras left out)
    out = out.reindex(columns=TIDES_COLS)

    # Log missing actual times (expected; do not drop)
    n_missing_actual_start = int(out["actual_trip_start"].isna().sum())
    n_missing_actual_end = int(out["actual_trip_end"].isna().sum())
    if n_missing_actual_start or n_missing_actual_end:
        logging.warning(
            "Kept rows with missing actual times: %d missing actual start, %d missing actual end.",
            n_missing_actual_start,
            n_missing_actual_end,
        )

    # Required field checks per schema (service_date, trip_id_performed, vehicle_id)
    for req in ("service_date", "trip_id_performed", "vehicle_id"):
        if out[req].isna().any():
            bad = out.loc[out[req].isna()].head(10)
            raise ValueError(f"Nulls found in required column '{req}'. Sample:\n{bad}")

    return out


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Run the CLEVER → TIDES trips_performed conversion pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    logging.info("Read %d rows, %d columns", len(df), df.shape[1])

    out = clever_to_tides(df)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    logging.info("Wrote %d rows -> %s", len(out), OUTPUT_CSV)


if __name__ == "__main__":
    main()
