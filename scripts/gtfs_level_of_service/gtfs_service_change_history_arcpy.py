"""
Analyzes and compares multiple GTFS datasets to identify changes in transit service.

Generates Excel reports summarizing differences in stops, level of service, and routes
between sequential GTFS snapshots.

Usage:
    - Configure GTFS dataset paths in `MULTIPLE_GTFS_CONFIGS`.
    - Run as a standalone script or from a notebook environment.

Outputs:
    - Excel workbooks highlighting route and stop-level service changes.
"""

import os
import math
import pandas as pd
import numpy as np

# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

MULTIPLE_GTFS_CONFIGS = [
    {"name": "Jan_2025",  "path": r"<GTFS_DATA_PATH_JAN_2025>"},
    {"name": "Jun_2025",  "path": r"<GTFS_DATA_PATH_JUN_2025>"},
    # add more chronologically…
]

OUTPUT_DIRECTORY   = r"<OUTPUT_DIRECTORY>"
OUTPUT_EXCEL_NAME_STOPS   = "stop_change_report.xlsx"        # original
OUTPUT_EXCEL_NAME_METRIC  = "route_metrics_by_signup.xlsx"   # NEW
OUTPUT_EXCEL_NAME_DELTA   = "service_level_changes.xlsx"     # NEW

# Tolerance (°) beyond which a coordinate change counts as “moved”
COORD_TOLERANCE_DEG = 0.00001     # ≈ 1 m at mid-latitudes

# Optional routes to filter out from ALL outputs/comparisons
ROUTE_FILTER_OUT = ["9999A", "9999B", "9999C"]   # empty list ⇒ no filtering
FILTER_SET       = set(ROUTE_FILTER_OUT)

# Time-of-day blocks used for the median-headway calculation
TIME_BLOCKS = {
    "AM"    : ("04:00", "09:00"),
    "MIDDAY": ("09:00", "15:00"),
    "PM"    : ("15:00", "21:00"),
    "NIGHT" : ("21:00", "28:00"),   # 28:00 = 04:00 next day
}

# ==================================================================================================
# FUNCTIONS
# ==================================================================================================

def _keep_changed(df):
    """
    Return only those routes where at least one metric actually changed
    (span_minutes, trips_count, or median_headway_min).
    """
    mask = (
        (df["span_delta"].fillna(0)  != 0) |
        (df["trips_delta"].fillna(0) != 0) |
        (df["hdwy_delta"].fillna(0)  != 0)
    )
    return df.loc[mask]

def _haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres (optional – not required for exact diff)."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlamb = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2)**2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlamb / 2)**2)
    return 2 * R * math.asin(math.sqrt(a))


def _parse_gtfs_time(t):
    """
    GTFS ‘HH:MM:SS’ strings can run past 24:00.  Return a pandas Timedelta.
    Invalid / missing strings ⇒ NaT.
    """
    try:
        hh, mm, ss = map(int, t.split(":"))
        secs = hh*3600 + mm*60 + ss
        return pd.Timedelta(seconds=secs)
    except Exception:
        return pd.NaT


def _format_td(td):
    """HH:MM string from Timedelta (None for NaT)."""
    if pd.isna(td):
        return None
    total_min = int(td.total_seconds() // 60)
    hh, mm = divmod(total_min, 60)
    return f"{hh:02d}:{mm:02d}"


def _assign_block(td):
    """Return the block label for a time-of-day Timedelta."""
    for blk, (start_s, end_s) in TIME_BLOCKS.items():
        s_h, s_m = map(int, start_s.split(":"))
        e_h, e_m = map(int, end_s.split(":"))
        start_td = pd.Timedelta(hours=s_h, minutes=s_m)
        end_td   = pd.Timedelta(hours=e_h, minutes=e_m)
        if start_td <= td < end_td:
            return blk
    return None
  
# --------------------------------------------------------------------------------------------------
# DATA LOADING
# --------------------------------------------------------------------------------------------------

FILES_NEEDED_STOP = ["stops.txt", "routes.txt", "trips.txt", "stop_times.txt"]
FILES_NEEDED_METR = FILES_NEEDED_STOP + ["calendar.txt"]          # for service filter


def _check_files(base, files):
    for f in files:
        if not os.path.exists(os.path.join(base, f)):
            raise FileNotFoundError(f"Required GTFS file missing: {f} in {base}")


def load_gtfs_basic(path):
    """(unchanged) – minimal stop-level data for the stop diff workbook."""
    _check_files(path, FILES_NEEDED_STOP)

    stops = pd.read_csv(
        os.path.join(path, "stops.txt"),
        dtype={
            "stop_id": str, "stop_code": str, "stop_name": str,
            "stop_lat": float, "stop_lon": float,
        },
        usecols=["stop_id", "stop_code", "stop_name", "stop_lat", "stop_lon"],
    )

    routes = pd.read_csv(
        os.path.join(path, "routes.txt"),
        usecols=["route_id", "route_short_name"], dtype=str,
    )
    trips = pd.read_csv(
        os.path.join(path, "trips.txt"),
        usecols=["trip_id", "route_id"], dtype=str,
    )
    stop_times = pd.read_csv(
        os.path.join(path, "stop_times.txt"),
        usecols=["trip_id", "stop_id"], dtype=str,
    )

    merged = (stop_times.merge(trips,  on="trip_id", how="left")
                         .merge(routes, on="route_id", how="left")
                         .dropna(subset=["route_short_name"]))

    if FILTER_SET:
        merged = merged[~merged["route_short_name"].isin(FILTER_SET)]

    stop_to_routes = (merged.groupby("stop_id")["route_short_name"]
                             .apply(set).to_dict())

    return stops, stop_to_routes


def load_route_metrics(path):
    """
    Load GTFS files from the given path and compute per‐route weekday metrics:
      - first_trip_time (HH:MM)
      - last_trip_time  (HH:MM)
      - span_minutes    (int)
      - trips_count     (int)
      - median_headway_min (float or None)

    Assumes calendar.txt uses “1”/“0” for days, and filters for Monday–Friday service.
    """
    # Verify required files exist
    _check_files(path, FILES_NEEDED_METR)

    # 1. Read GTFS tables
    routes = pd.read_csv(
        os.path.join(path, "routes.txt"),
        usecols=["route_id", "route_short_name"],
        dtype=str,
    )
    trips = pd.read_csv(
        os.path.join(path, "trips.txt"),
        usecols=["trip_id", "route_id", "service_id"],
        dtype=str,
    )
    stop_times = pd.read_csv(
        os.path.join(path, "stop_times.txt"),
        usecols=["trip_id", "stop_sequence", "departure_time"],
        dtype=str,
    )

    # 2. Filter to weekday-only service (all five weekdays == "1")
    cal_path = os.path.join(path, "calendar.txt")
    if os.path.exists(cal_path):
        cal = pd.read_csv(cal_path, dtype=str)
        mask = (
            (cal["monday"]    == "1") &
            (cal["tuesday"]   == "1") &
            (cal["wednesday"] == "1") &
            (cal["thursday"]  == "1") &
            (cal["friday"]    == "1")
        )
        weekday_sids = set(cal.loc[mask, "service_id"])
        trips = trips[trips["service_id"].isin(weekday_sids)]

    # 3. Merge trips→routes and apply any route filter
    trip_routes = trips.merge(routes, on="route_id", how="left")
    if FILTER_SET:
        trip_routes = trip_routes[
            ~trip_routes["route_short_name"].isin(FILTER_SET)
        ]

    # 4. Determine the “first” stop of each trip, regardless of its numeric label
    #    a) ensure stop_sequence is numeric
    stop_times["stop_sequence"] = pd.to_numeric(
        stop_times["stop_sequence"], errors="coerce"
    )
    #    b) pick the row with the minimum sequence per trip
    idx_first = stop_times.groupby("trip_id")["stop_sequence"].idxmin()
    st = stop_times.loc[idx_first].merge(
        trip_routes, on="trip_id", how="inner"
    )
    if st.empty:
        raise ValueError("No usable stop_times after first-stop extraction.")

    # 5. Parse departure_time → Timedelta, drop invalid
    st["td"] = st["departure_time"].apply(_parse_gtfs_time)
    st = st.dropna(subset=["td"])

    # 6. Assign each departure to a time block for headway calc
    st["block"] = st["td"].apply(_assign_block)

    # 7. Build metrics per route
    metrics = []
    for rt, grp in st.groupby("route_short_name"):
        times = grp["td"].sort_values()
        first_td = times.iloc[0]
        last_td  = times.iloc[-1]

        span_min = int((last_td - first_td).total_seconds() // 60)
        trips_ct = len(times)

        # compute median headway across all blocks
        headway_series = []
        for _, blk_df in grp.groupby("block"):
            if len(blk_df) < 2:
                continue
            diffs = blk_df["td"].sort_values().diff().dropna()
            headway_series.append(diffs.dt.total_seconds() / 60)
        med_hw = None
        if headway_series:
            med_hw = float(pd.concat(headway_series).median())

        metrics.append({
            "route_short_name":     rt,
            "first_trip_time":      _format_td(first_td),
            "last_trip_time":       _format_td(last_td),
            "span_minutes":         span_min,
            "trips_count":          trips_ct,
            "median_headway_min":   med_hw,
        })

    # 8. Return a clean DataFrame
    return (
        pd.DataFrame(metrics)
          .sort_values("route_short_name")
          .reset_index(drop=True)
    )

# ──────────────────────────────────────────────────────────────────────────────
# STOP-LEVEL COMPARISON (unchanged except output file name constant)
# ──────────────────────────────────────────────────────────────────────────────
# … compare_signups(), etc. unchanged – omitted here for brevity …

def build_service_level_changes(prev_df, curr_df, prev_label, curr_label):
    """Return sheets for service-level deltas + route add/delete lists."""
    prev_df = prev_df.set_index("route_short_name")
    curr_df = curr_df.set_index("route_short_name")

    added   = sorted(set(curr_df.index) - set(prev_df.index))
    deleted = sorted(set(prev_df.index) - set(curr_df.index))

    # Core metric deltas
    common = sorted(set(prev_df.index) & set(curr_df.index))
    deltas = []
    for rt in common:
        row_prev = prev_df.loc[rt]
        row_curr = curr_df.loc[rt]

        def _delta(col):
            a, b = row_prev[col], row_curr[col]
            if pd.isna(a) or pd.isna(b):
                return np.nan
            return b - a

        deltas.append({
            "route_short_name": rt,
            "span_old_min":  row_prev["span_minutes"],
            "span_new_min":  row_curr["span_minutes"],
            "span_delta":    _delta("span_minutes"),
            "trips_old":     row_prev["trips_count"],
            "trips_new":     row_curr["trips_count"],
            "trips_delta":   _delta("trips_count"),
            "hdwy_old_min":  row_prev["median_headway_min"],
            "hdwy_new_min":  row_curr["median_headway_min"],
            "hdwy_delta":    _delta("median_headway_min"),
        })

    df_delta   = pd.DataFrame(deltas).sort_values("route_short_name")
    df_added   = pd.DataFrame({"route_short_name": added})
    df_deleted = pd.DataFrame({"route_short_name": deleted})

    key = f"{prev_label}→{curr_label}"
    sheets = {
        f"ServiceChange_{key}":    df_delta,
        f"Routes_Added_{key}":     df_added,
        f"Routes_Deleted_{key}":   df_deleted,
    }
    return sheets

# ==================================================================================================
# MAIN
# ==================================================================================================

def main():
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    # ─── LOAD GTFS FOR EACH SIGN-UP ─────────────────────────────────────────────
    signups_stops      = {}
    signups_routes_map = {}
    signups_metrics    = {}

    for cfg in MULTIPLE_GTFS_CONFIGS:
        name, path = cfg["name"], cfg["path"]
        print(f"Loading GTFS for {name} …")
        stops, stop_routes = load_gtfs_basic(path)
        signups_stops[name]       = stops
        signups_routes_map[name]  = stop_routes

        print(f"Building route metrics for {name} …")
        signups_metrics[name]     = load_route_metrics(path)
    print("All GTFS loads complete.\n")

    # ─── STOP-LEVEL CHANGE WORKBOOK ────────────────────────────────────────────
    all_sheets_stop = {}
    for i in range(1, len(MULTIPLE_GTFS_CONFIGS)):
        prev = MULTIPLE_GTFS_CONFIGS[i-1]["name"]
        curr = MULTIPLE_GTFS_CONFIGS[i]["name"]
        sheets = compare_signups(
            prev, curr,
            signups_stops[prev], signups_stops[curr],
            signups_routes_map[prev], signups_routes_map[curr],
        )
        all_sheets_stop.update(sheets)

    with pd.ExcelWriter(
        os.path.join(OUTPUT_DIRECTORY, OUTPUT_EXCEL_NAME_STOPS),
        engine="openpyxl"
    ) as xls:
        for sheet_name, df in all_sheets_stop.items():
            df.to_excel(xls, sheet_name=sheet_name[:31], index=False)
    print("✓ stop-level change workbook written.")

    # ─── ROUTE METRICS + DELTA-ONLY SHEETS ─────────────────────────────────────
    metrics_path = os.path.join(OUTPUT_DIRECTORY, OUTPUT_EXCEL_NAME_METRIC)
    with pd.ExcelWriter(metrics_path, engine="openpyxl") as xls:
        # 1) write one sheet per sign-up
        for label, df in signups_metrics.items():
            df.to_excel(xls, sheet_name=label[:31], index=False)

        # 2) for each pair of sign-ups, append only‐changed routes
        for i in range(1, len(MULTIPLE_GTFS_CONFIGS)):
            prev = MULTIPLE_GTFS_CONFIGS[i-1]["name"]
            curr = MULTIPLE_GTFS_CONFIGS[i]["name"]

            # build the full delta DataFrame
            delta_dict = build_service_level_changes(
                signups_metrics[prev], signups_metrics[curr], prev, curr
            )
            full = delta_dict[f"ServiceChange_{prev}→{curr}"]

            # keep only the rows where something actually changed
            changed = _keep_changed(full)
            if not changed.empty:
                sheet_nm = f"Changes_{prev}_to_{curr}"[:31]
                changed.to_excel(xls, sheet_name=sheet_nm, index=False)
    print("✓ route metrics workbook (with delta-only sheets) written.")

    # ─── SERVICE-LEVEL CHANGE WORKBOOK ─────────────────────────────────────────
    delta_sheets = {}
    for i in range(1, len(MULTIPLE_GTFS_CONFIGS)):
        prev = MULTIPLE_GTFS_CONFIGS[i-1]["name"]
        curr = MULTIPLE_GTFS_CONFIGS[i]["name"]
        sheets = build_service_level_changes(
            signups_metrics[prev], signups_metrics[curr], prev, curr
        )
        delta_sheets.update(sheets)

    with pd.ExcelWriter(
        os.path.join(OUTPUT_DIRECTORY, OUTPUT_EXCEL_NAME_DELTA),
        engine="openpyxl"
    ) as xls:
        for sheet_name, df in delta_sheets.items():
            df.to_excel(xls, sheet_name=sheet_name[:31], index=False)
    print("✓ service-level change workbook written.")

if __name__ == "__main__":
    main()
