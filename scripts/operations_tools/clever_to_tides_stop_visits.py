"""Convert CLEVER export into TIDES-compliant stop_visits records.

This module reads a CLEVER “Stop Visit Events” report (CSV) and produces a
`stop_visits.csv` file aligned with the TIDES `stop_visits` schema. It
normalizes common real-world export issues (inconsistent whitespace, AM/PM
timestamps, mixed null tokens) and applies pragmatic, schema-aware rules so
the output can be ingested by downstream validation and analytics pipelines.

The CLEVER Stop Visit Events report is timepoint-based rather than true stop-
ordered. This converter therefore preserves CLEVER `Timepoint Order` values
as both `trip_stop_sequence` and `scheduled_stop_sequence`. While this may
violate strict stop-order expectations in the TIDES schema, it reflects the
best available ordering in the source data. Potential sequencing issues are
logged for visibility.

Key behaviors:
- Parses scheduled and actual stop timestamps robustly (tolerates AM/PM,
  inconsistent whitespace, and partial availability). Rows are not dropped
  solely due to incomplete timing data.
- Uses Scheduled Passing Time for scheduled arrival and departure when
  present. Uses Arrival Time and Departure Time for actual stop times, with
  fallback to Actual Time when needed.
- Computes `dwell` in seconds when actual arrival and departure are present
  and non-negative; invalid dwell values are left blank and logged.
- Derives `service_date` from the CLEVER Date field and extracts
  `trip_id_performed` from the CLEVER Trip token (or a stable derived
  identifier when configured to match a hashed `trips_performed` strategy).
- Extracts `stop_id` from CLEVER Timepoint ID, marks all records as
  `timepoint = True`, and synthesizes `pattern_id` from available route,
  direction, and variation fields.
- Emits schema-required fields using schema-valid types and enum values, and
  leaves unsupported stop-level attributes (APC, door events, ramp activity,
  revenue) blank with explicit warnings.

The conversion is deterministic: given the same input file and configuration,
the output identifiers and records are stable across runs.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_CSV: Path = Path(r"Path\To\Stop Visit Events.csv")
OUTPUT_CSV: Path = Path(r"Path\To\stop_visits.csv")

# Trip ID strategy:
# - "token": trip_id_performed = second token in CLEVER "Trip" (default)
# - "hashed": trip_id_performed = stable hash of (service_date, trip_token, vehicle_id)
#
# If you already generate trip_id_performed in trips_performed.csv as a hash, use "hashed" here too.
TRIP_ID_MODE: str = "token"  # "token" | "hashed"

# Column names that may vary between CLEVER exports (edit as needed).
DATE_COL: str = "Date"
TRIP_COL: str = "Trip"
ROUTE_COL: str = "Route"
DIRECTION_COL: str = "Direction"
VEHICLE_COL: str = "Vehicle"
VARIATION_COL: str = "Variation"
TIMEPOINT_ORDER_COL: str = "Timepoint Order"
TIMEPOINT_ID_COL: str = "Timepoint ID"

# Updated time columns (present in your new sample).
ACTUAL_TIME_COL: str = "Actual Time"
ARRIVAL_TIME_COL: str = "Arrival Time"
DEPARTURE_TIME_COL: str = "Departure Time"
SCHEDULED_PASSING_TIME_COL: str = "Scheduled Passing Time"

# TIDES stop_visits output columns in the exact order of the template.
TIDES_COLS: list[str] = [
    "service_date",
    "trip_id_performed",
    "trip_stop_sequence",
    "scheduled_stop_sequence",
    "pattern_id",
    "vehicle_id",
    "dwell",
    "stop_id",
    "timepoint",
    "schedule_arrival_time",
    "schedule_departure_time",
    "actual_arrival_time",
    "actual_departure_time",
    "distance",
    "boarding_1",
    "alighting_1",
    "boarding_2",
    "alighting_2",
    "departure_load",
    "door_open",
    "door_close",
    "door_status",
    "ramp_deployed_time",
    "ramp_failure",
    "kneel_deployed_time",
    "lift_deployed_time",
    "bike_rack_deployed",
    "bike_load",
    "revenue",
    "number_of_transactions",
    "schedule_relationship",
]

# Minimal required CLEVER columns to produce a useful stop_visits file.
REQ_COLS: list[str] = [
    DATE_COL,
    TRIP_COL,
    TIMEPOINT_ORDER_COL,
    TIMEPOINT_ID_COL,
]

# Optional CLEVER columns (used when present).
OPT_COLS: list[str] = [
    ROUTE_COL,
    DIRECTION_COL,
    VEHICLE_COL,
    VARIATION_COL,
    ACTUAL_TIME_COL,
    ARRIVAL_TIME_COL,
    DEPARTURE_TIME_COL,
    SCHEDULED_PASSING_TIME_COL,
]

# =============================================================================
# HELPERS
# =============================================================================


def normalize_text(series: pd.Series) -> pd.Series:
    """Normalize text values: strip, collapse whitespace, coerce common blanks to NA."""
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "N/A": pd.NA})
    )


def parse_service_date(date_series: pd.Series) -> pd.Series:
    """Parse CLEVER Date (m/d/YYYY) into ISO service_date (YYYY-MM-DD)."""
    s = normalize_text(date_series)
    dt = pd.to_datetime(s, errors="coerce", format="%m/%d/%Y")
    bad = int(dt.isna().sum())
    if bad:
        logging.warning("Failed to parse %d Date values; emitting blanks for those rows.", bad)
    return dt.dt.strftime("%Y-%m-%d").astype("string")


def split_trip_token(trip_series: pd.Series) -> pd.Series:
    """Extract the second token from CLEVER Trip like "04:02 1550064" -> "1550064".

    If parsing fails, emits NA and logs a warning.
    """
    s = normalize_text(trip_series).fillna("")
    parts = s.str.split(r"\s+", n=1, expand=True)

    trip_token = (
        parts[1].astype("string").fillna("").str.strip()
        if parts.shape[1] > 1
        else pd.Series("", index=s.index, dtype="string")
    )

    if (trip_token == "").any():
        logging.warning(
            "Some Trip values did not include a second token (trip token). "
            "trip_id_performed may be blank for those rows."
        )

    return trip_token.replace({"": pd.NA}).astype("string")


def stable_id(*parts: str) -> str:
    """Short deterministic ID from multiple fields (sha1 truncated)."""
    raw = "||".join(parts).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def parse_route_short(route_text: pd.Series) -> pd.Series:
    """Route like '101 - Fort Hunt - Mount Vernon' -> '101' (best-effort)."""
    s = normalize_text(route_text).fillna("")
    left = s.str.split("-", n=1, expand=True)[0].astype("string").str.strip()
    digits = left.str.extract(r"^\s*([0-9A-Za-z]+)\s*$")[0]
    return digits.fillna(left).replace({"": pd.NA}).astype("string")


def normalize_vehicle_id(vehicle_series: pd.Series) -> pd.Series:
    """Normalize vehicle id values (handle floats like '7906.0', trim whitespace)."""
    s = normalize_text(vehicle_series)
    return (
        s.str.replace(r"\.0$", "", regex=True)
        .replace({"<NA>": pd.NA})
        .astype("string")
    )


def normalize_dt_series(series: pd.Series) -> pd.Series:
    """Normalize datetime-like strings prior to pandas parsing."""
    return normalize_text(series)


def parse_dt(series: pd.Series, *, col_name: str) -> pd.Series:
    """Parse datetime series robustly; logs count of unparseable values."""
    dt = pd.to_datetime(
        normalize_dt_series(series),
        errors="coerce",
        infer_datetime_format=True,
    )
    bad = int(dt.isna().sum())
    if bad:
        logging.warning("Failed to parse %d values in %r.", bad, col_name)
    return dt


def dt_to_iso(series: pd.Series) -> pd.Series:
    """Datetime -> ISO string (YYYY-MM-DDTHH:MM:SS); NaT -> NA."""
    return series.dt.strftime("%Y-%m-%dT%H:%M:%S").astype("string")


def warn_missing_columns(df: pd.DataFrame) -> None:
    """Warn if optional columns are missing, or raise error if required columns are missing."""
    missing_req = [c for c in REQ_COLS if c not in df.columns]
    if missing_req:
        raise ValueError(f"Missing required CLEVER columns: {missing_req}")

    missing_opt = [c for c in OPT_COLS if c not in df.columns]
    if missing_opt:
        logging.warning("Missing optional CLEVER columns (will continue): %s", missing_opt)


def normalize_timepoint_order(series: pd.Series) -> pd.Series:
    """Parse Timepoint Order as integer-ish sequence.

    Policy:
    - If values parse as numeric:
        - if any are 0, we shift those by +1 and warn (schema expects min 1).
        - if any are < 0, we drop them to NA and warn.
    - If unparseable, NA with warning elsewhere.
    """
    seq = pd.to_numeric(series, errors="coerce")

    neg = seq.notna() & (seq < 0)
    if neg.any():
        logging.warning(
            "Found %d rows with negative %r; setting those sequences to blank.",
            int(neg.sum()),
            TIMEPOINT_ORDER_COL,
        )
        seq = seq.mask(neg)

    zeros = seq.notna() & (seq == 0)
    if zeros.any():
        logging.warning(
            "Found %d rows with %r == 0; shifting those to 1 (adding +1).",
            int(zeros.sum()),
            TIMEPOINT_ORDER_COL,
        )
        seq = seq.mask(zeros, 1)

    # Keep as nullable Int64
    return seq.round(0).astype("Int64")


def warn_nonconsecutive_sequences(
    df_out: pd.DataFrame,
    *,
    service_date_col: str = "service_date",
    trip_id_col: str = "trip_id_performed",
    seq_col: str = "trip_stop_sequence",
) -> None:
    """Warn if sequences within (service_date, trip_id_performed) are not consecutive starting at 1.

    We do NOT fix them (user asked to accept timepoint order), but we do make it visible.
    """
    if df_out.empty:
        return

    key_cols = [service_date_col, trip_id_col]
    missing_keys = [c for c in key_cols + [seq_col] if c not in df_out.columns]
    if missing_keys:
        return

    tmp = df_out[key_cols + [seq_col]].copy()
    tmp = tmp.dropna(subset=key_cols, how="any")

    def _is_consecutive(s: pd.Series) -> bool:
        vals = pd.to_numeric(s, errors="coerce").dropna().astype(int).tolist()
        if not vals:
            return True
        vals_sorted = sorted(set(vals))
        return vals_sorted[0] == 1 and vals_sorted == list(range(1, vals_sorted[-1] + 1))

    bad_groups = 0
    for _, g in tmp.groupby(key_cols, dropna=False):
        if not _is_consecutive(g[seq_col]):
            bad_groups += 1

    if bad_groups:
        logging.warning(
            "Timepoint-based sequencing: %d trip(s) have non-consecutive %r values. "
            "This may violate strict TIDES schema expectations for stop order.",
            bad_groups,
            seq_col,
        )


# =============================================================================
# CORE CONVERSION
# =============================================================================


def clever_to_tides(df: pd.DataFrame) -> pd.DataFrame:
    """Convert CLEVER Stop Visit Events export to TIDES stop_visits.csv."""
    df = df.dropna(how="all").copy()
    warn_missing_columns(df)

    out = pd.DataFrame(index=df.index)

    # service_date
    out["service_date"] = parse_service_date(df[DATE_COL])

    # Trip token + vehicle_id (needed for optional hashed ID mode)
    trip_token = split_trip_token(df[TRIP_COL])

    if VEHICLE_COL in df.columns:
        vehicle_id = normalize_vehicle_id(df[VEHICLE_COL])
    else:
        vehicle_id = pd.Series(pd.NA, index=df.index, dtype="string")

    # trip_id_performed
    trip_id_mode = str(TRIP_ID_MODE).strip().lower()
    if trip_id_mode not in {"token", "hashed"}:
        logging.warning("TRIP_ID_MODE=%r is invalid; defaulting to 'token'.", TRIP_ID_MODE)
        trip_id_mode = "token"

    if trip_id_mode == "token":
        out["trip_id_performed"] = trip_token
    else:
        # Hash inputs to align with a hashed trips_performed strategy
        # Note: if your trips_performed hash uses different fields, change this to match exactly.
        service_date_str = out["service_date"].fillna("").astype("string")
        trip_tok_str = trip_token.fillna("").astype("string")
        veh_str = vehicle_id.fillna("").astype("string")

        out["trip_id_performed"] = [
            f"perf_{stable_id(str(sd), str(tt), str(vv))}"
            for sd, tt, vv in zip(service_date_str, trip_tok_str, veh_str, strict=True)
        ]
        out["trip_id_performed"] = (
            pd.Series(out["trip_id_performed"], index=df.index, dtype="string")
            .replace({"perf_" + stable_id("", "", ""): pd.NA})
        )

    # sequences + stop_id
    seq = normalize_timepoint_order(df[TIMEPOINT_ORDER_COL])
    bad_seq = int(seq.isna().sum())
    if bad_seq:
        logging.warning(
            "Failed to parse %d %r values as integers; emitting blanks for those rows.",
            bad_seq,
            TIMEPOINT_ORDER_COL,
        )

    out["trip_stop_sequence"] = seq
    out["scheduled_stop_sequence"] = seq

    out["stop_id"] = normalize_text(df[TIMEPOINT_ID_COL])

    # vehicle_id
    out["vehicle_id"] = vehicle_id
    if out["vehicle_id"].isna().any():
        logging.warning(
            "vehicle_id is missing on %d rows.", int(out["vehicle_id"].isna().sum())
        )

    # pattern_id (synthetic)
    route_short = (
        parse_route_short(df[ROUTE_COL])
        if ROUTE_COL in df.columns
        else pd.Series(pd.NA, index=df.index)
    )
    direction = (
        normalize_text(df[DIRECTION_COL])
        if DIRECTION_COL in df.columns
        else pd.Series(pd.NA, index=df.index)
    )

    if VARIATION_COL in df.columns:
        var = pd.to_numeric(df[VARIATION_COL], errors="coerce").astype("Int64").astype("string")
        var = var.replace({"<NA>": pd.NA})
    else:
        var = pd.Series(pd.NA, index=df.index, dtype="string")

    pattern = (
        route_short.fillna("").astype("string").str.strip()
        + "|"
        + direction.fillna("").astype("string").str.strip()
        + "|"
        + var.fillna("").astype("string").str.strip()
    ).str.strip("|")
    out["pattern_id"] = pattern.replace({"": pd.NA}).astype("string")

    # timepoint flag (boolean per schema)
    out["timepoint"] = True

    # --- Times: scheduled + actual
    # Scheduled Passing Time -> schedule_arrival_time + schedule_departure_time
    if SCHEDULED_PASSING_TIME_COL in df.columns:
        sched_dt = parse_dt(df[SCHEDULED_PASSING_TIME_COL], col_name=SCHEDULED_PASSING_TIME_COL)
        sched_iso = dt_to_iso(sched_dt).where(sched_dt.notna(), pd.NA)
        out["schedule_arrival_time"] = sched_iso
        out["schedule_departure_time"] = sched_iso
    else:
        out["schedule_arrival_time"] = pd.NA
        out["schedule_departure_time"] = pd.NA

    # Arrival/Departure -> actual arrival/departure; fallback to Actual Time
    arrival_dt = (
        parse_dt(df[ARRIVAL_TIME_COL], col_name=ARRIVAL_TIME_COL)
        if ARRIVAL_TIME_COL in df.columns
        else pd.Series(pd.NaT, index=df.index)
    )
    depart_dt = (
        parse_dt(df[DEPARTURE_TIME_COL], col_name=DEPARTURE_TIME_COL)
        if DEPARTURE_TIME_COL in df.columns
        else pd.Series(pd.NaT, index=df.index)
    )
    actual_dt = (
        parse_dt(df[ACTUAL_TIME_COL], col_name=ACTUAL_TIME_COL)
        if ACTUAL_TIME_COL in df.columns
        else pd.Series(pd.NaT, index=df.index)
    )

    arrival_best = arrival_dt.where(arrival_dt.notna(), actual_dt)
    depart_best = depart_dt.where(depart_dt.notna(), actual_dt)

    out["actual_arrival_time"] = dt_to_iso(arrival_best).where(arrival_best.notna(), pd.NA)
    out["actual_departure_time"] = dt_to_iso(depart_best).where(depart_best.notna(), pd.NA)

    # dwell seconds (only when both present and non-negative)
    dwell_sec = (depart_best - arrival_best).dt.total_seconds()
    dwell_ok = dwell_sec.notna() & (dwell_sec >= 0)
    if (~dwell_ok & dwell_sec.notna()).any():
        logging.warning(
            "Found %d rows with negative dwell (departure before arrival); "
            "leaving dwell blank for those rows.",
            int((~dwell_ok & dwell_sec.notna()).sum()),
        )
    out["dwell"] = dwell_sec.where(dwell_ok, pd.NA).round(0).astype("Int64")

    # Fields CLEVER Stop Visit Events doesn't provide: leave blank
    blanks = [
        "distance",
        "boarding_1",
        "alighting_1",
        "boarding_2",
        "alighting_2",
        "departure_load",
        "door_open",
        "door_close",
        "door_status",
        "ramp_deployed_time",
        "ramp_failure",
        "kneel_deployed_time",
        "lift_deployed_time",
        "bike_rack_deployed",
        "bike_load",
        "revenue",
        "number_of_transactions",
    ]
    for c in blanks:
        out[c] = pd.NA

    # schedule_relationship: schema enum value
    out["schedule_relationship"] = "Scheduled"

    # Warn about strict-schema expectations we are knowingly violating / risking
    warn_nonconsecutive_sequences(out)

    # Minimal sanity warnings
    if out["service_date"].isna().any():
        logging.warning(
            "service_date is missing on %d rows.", int(out["service_date"].isna().sum())
        )
    if out["stop_id"].isna().any():
        logging.warning("stop_id is missing on %d rows.", int(out["stop_id"].isna().sum()))
    if out["trip_id_performed"].isna().any():
        logging.warning(
            "trip_id_performed is missing on %d rows.",
            int(out["trip_id_performed"].isna().sum())
        )
    if out["trip_stop_sequence"].isna().any():
        logging.warning(
            "trip_stop_sequence is missing on %d rows (unparseable %r).",
            int(out["trip_stop_sequence"].isna().sum()),
            TIMEPOINT_ORDER_COL,
        )

    # Final order + stable dtypes for CSV output
    out = out.reindex(columns=TIDES_COLS)

    # Convert nullable Int64 columns to plain strings/blank for cleaner CSV, if desired.
    # Keep as numeric if you prefer strict typing in downstream ingestion.
    # Here we keep numeric for seq + dwell, which is generally what schemas expect.
    return out


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Execute the conversion from CLEVER Stop Visit Events to TIDES stop_visits."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    logging.info("Read %d rows, %d columns", len(df), df.shape[1])

    out = clever_to_tides(df)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    logging.info("Wrote %d rows -> %s", len(out), OUTPUT_CSV)


if __name__ == "__main__":
    main()
