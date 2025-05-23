"""
Script Name:
    gtfs_stop_pattern_exporter.py

Purpose:
    Extracts unique stop patterns from GTFS data, optionally filters them,
    calculates earliest departure times, and exports the patterns to
    Excel workbooks organized by route, direction, and service ID.
    Output subfolders can be structured based on service days if
    calendar data is available.

Inputs:
    1. GTFS files (stops.txt, trips.txt, stop_times.txt, routes.txt,
      and optionally calendar.txt) located in INPUT_DIR.
    2. Configuration constants (e.g., INPUT_DIR, OUTPUT_DIR,
       FILTER_IN_ROUTE_SHORT_NAMES, CONVERT_TO_MILES,
       EXPORT_TIMEPOINTS_ONLY).

Outputs:
    1. Excel workbooks (.xlsx) containing stop patterns, saved in
       subfolders within OUTPUT_DIR. Subfolder names are derived from
       service IDs and optionally service days (e.g.,
       'calendar_123_mon_tue'). Workbook names include route short
       name, service ID, and SIGNUP_NAME.

Dependencies:
    logging, os, collections (defaultdict), numpy, pandas, openpyxl
"""

import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils import get_column_letter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_DIR = r"\\Path\\To\\Your\\GTFS"
OUTPUT_DIR = r"\\Path\\To\\Output"
TRIPS_FILE = "trips.txt"
STOP_TIMES_FILE = "stop_times.txt"
STOPS_FILE = "stops.txt"
ROUTES_FILE = "routes.txt"
CALENDAR_FILE = "calendar.txt"  # for subfolder naming if available

FILTER_IN_ROUTE_SHORT_NAMES = []
FILTER_OUT_ROUTE_SHORT_NAMES = []
FILTER_IN_CALENDAR_IDS = []

SIGNUP_NAME = "January2025Signup"

INPUT_DISTANCE_UNIT = "meters"
CONVERT_TO_MILES = True
EXPORT_TIMEPOINTS_ONLY = True
VALIDATE_TIMEPOINT_DISTANCE = True

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------


def is_number(value):
    """
    Check if a value can be converted to a float.
    """
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def convert_dist_to_miles(distance, input_unit):
    """
    Convert 'distance' to miles if configured; otherwise return as-is.
    """
    if not CONVERT_TO_MILES or pd.isna(distance):
        return distance
    if input_unit.lower() == "feet":
        conv = 5280.0
    elif input_unit.lower() == "meters":
        conv = 1609.34
    else:
        logging.warning("Unknown distance unit '%s'. No conversion done.", input_unit)
        conv = 1.0
    return distance / conv


def parse_time_to_minutes(time_str):
    """
    Parse HH:MM:SS (GTFS style) to float minutes. Return None if invalid.
    Allows hours >= 24 for service passing midnight.
    """
    if not isinstance(time_str, str):
        return None
    parts = time_str.strip().split(":")
    if len(parts) != 3:
        return None
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        return hours * 60 + minutes + seconds / 60.0
    except (TypeError, ValueError):
        return None


def minutes_to_hhmm(minutes_val):
    """
    Convert float minutes to 'HH:MM' (24-hour). Return empty string if invalid.
    """
    if minutes_val is None or pd.isna(minutes_val):
        return ""
    total_minutes = int(minutes_val)
    hours = total_minutes // 60
    mins = total_minutes % 60
    return f"{hours:02d}:{mins:02d}"


def format_service_id_folder_name(service_id, calendar_df):
    """
    Build a subfolder name like 'calendar_3_mon_tue_fri' or 'calendar_10_none' if no days.
    If calendar_df is None or doesn't contain the service_id, fallback to 'calendar_<service_id>'.
    """
    if calendar_df is None or calendar_df.empty:
        return f"calendar_{service_id}"

    row = calendar_df[calendar_df["service_id"] == str(service_id)]
    if row.empty:
        return f"calendar_{service_id}"

    row = row.iloc[0]  # Take the first matching row
    day_map = [
        ("monday", "mon"),
        ("tuesday", "tue"),
        ("wednesday", "wed"),
        ("thursday", "thu"),
        ("friday", "fri"),
        ("saturday", "sat"),
        ("sunday", "sun"),
    ]
    served_days = []
    for col, short_name in day_map:
        if str(row.get(col, "0")) == "1":
            served_days.append(short_name)

    if served_days:
        day_str = "_".join(served_days)
    else:
        day_str = "none"

    return f"calendar_{service_id}_{day_str}"


# -----------------------------------------------------------------------------
# LOADING GTFS
# -----------------------------------------------------------------------------


def load_gtfs_files(input_dir):
    """
    Load the core GTFS files and return as a dict.
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input dir does not exist: {input_dir}")

    stops = pd.read_csv(os.path.join(input_dir, STOPS_FILE))
    trips = pd.read_csv(os.path.join(input_dir, TRIPS_FILE))
    stop_times = pd.read_csv(os.path.join(input_dir, STOP_TIMES_FILE))
    routes = pd.read_csv(os.path.join(input_dir, ROUTES_FILE))

    logging.info("Loaded GTFS files successfully.")
    return {"stops": stops, "trips": trips, "stop_times": stop_times, "routes": routes}


def filter_trips(trips_df, routes_df, cal_ids):
    """
    Filter trips by desired route short names and calendar service IDs.

    Includes trips whose route_short_name is in FILTER_IN_ROUTE_SHORT_NAMES,
    excludes those in FILTER_OUT_ROUTE_SHORT_NAMES, and filters by any specified
    service IDs in cal_ids.
    """
    merged = pd.merge(
        trips_df, routes_df[["route_id", "route_short_name"]], on="route_id", how="left"
    )
    # Filter in
    if FILTER_IN_ROUTE_SHORT_NAMES:
        merged = merged[merged["route_short_name"].isin(FILTER_IN_ROUTE_SHORT_NAMES)]
    # Filter out
    if FILTER_OUT_ROUTE_SHORT_NAMES:
        merged = merged[~merged["route_short_name"].isin(FILTER_OUT_ROUTE_SHORT_NAMES)]
    # Filter calendar
    if cal_ids and "service_id" in merged.columns:
        merged = merged[merged["service_id"].isin(cal_ids)]
    elif cal_ids:
        logging.warning("No service_id in data for filtering.")
    return merged


# -----------------------------------------------------------------------------
# BUILD PATTERNS
# -----------------------------------------------------------------------------


def generate_unique_patterns(trips_df, stop_times_df, stops_df):
    """
    Creates unique patterns keyed by (route_id, direction_id, service_id, pattern_stops).
    Each pattern is a tuple of stops: [ (stop_id, distance_from_previous), ... ]
    """
    tmp = pd.merge(
        stop_times_df,
        trips_df[["trip_id", "route_id", "direction_id", "service_id"]],
        on="trip_id",
        how="inner",
    )
    if "shape_dist_traveled" not in tmp.columns:
        tmp["shape_dist_traveled"] = np.nan

    tmp = pd.merge(tmp, stops_df[["stop_id", "stop_name"]], on="stop_id", how="left")
    tmp.sort_values(["trip_id", "stop_sequence"], inplace=True)

    # If we only want timepoints
    if EXPORT_TIMEPOINTS_ONLY and "timepoint" in tmp.columns:
        tmp = tmp[tmp["timepoint"] == 1]

    # For distance validation
    trip_distances = {}
    if VALIDATE_TIMEPOINT_DISTANCE and EXPORT_TIMEPOINTS_ONLY:
        for trip_id_val, group_sub in tmp.groupby("trip_id"):
            group_sub = group_sub.dropna(subset=["shape_dist_traveled"])
            if group_sub.empty:
                trip_distances[trip_id_val] = None
            else:
                dist_val = (
                    group_sub.iloc[-1]["shape_dist_traveled"]
                    - group_sub.iloc[0]["shape_dist_traveled"]
                )
                trip_distances[trip_id_val] = convert_dist_to_miles(
                    dist_val, INPUT_DISTANCE_UNIT
                )

    # Build patterns
    patterns_list = []
    for trip_id_val, group_sub in tmp.groupby("trip_id"):
        group_sub = group_sub.sort_values("stop_sequence")
        stops_for_trip = []
        prev_dist_val = None

        for _, row in group_sub.iterrows():
            stop_id_val = row["stop_id"]
            shape_val = row["shape_dist_traveled"]
            if prev_dist_val is None:
                dist_str = "-"
            else:
                if pd.notnull(shape_val) and pd.notnull(prev_dist_val):
                    diff = convert_dist_to_miles(
                        shape_val - prev_dist_val, INPUT_DISTANCE_UNIT
                    )
                    dist_str = f"{diff:.2f}" if diff else ""
                else:
                    dist_str = ""
            stops_for_trip.append((stop_id_val, dist_str))
            if pd.notnull(shape_val):
                prev_dist_val = shape_val
            else:
                prev_dist_val = None

        # Validate timepoint distance
        if VALIDATE_TIMEPOINT_DISTANCE and EXPORT_TIMEPOINTS_ONLY:
            sum_seg = 0.0
            for _, dist_segment in stops_for_trip:
                if dist_segment not in ("-", "", None) and is_number(dist_segment):
                    sum_seg += float(dist_segment)
            full_trip = trip_distances.get(trip_id_val, None)
            if (full_trip is not None) and abs(sum_seg - full_trip) > 0.02:
                logging.warning(
                    "Trip %s sum of segments=%.2f vs. full=%.2f mismatch >0.02",
                    trip_id_val,
                    sum_seg,
                    full_trip,
                )

        first_row = group_sub.iloc[0]
        route_id_val = first_row["route_id"]
        direction_id_val = first_row["direction_id"]
        service_id_val = first_row["service_id"]

        patterns_list.append(
            {
                "trip_id": trip_id_val,
                "route_id": route_id_val,
                "direction_id": direction_id_val,
                "service_id": service_id_val,
                "pattern_stops": tuple(stops_for_trip),
            }
        )

    # Accumulate unique patterns
    patterns_dict = {}
    for record in patterns_list:
        key = (
            record["route_id"],
            record["direction_id"],
            record["service_id"],
            record["pattern_stops"],
        )
        if key not in patterns_dict:
            patterns_dict[key] = {
                "route_id": record["route_id"],
                "direction_id": record["direction_id"],
                "service_id": record["service_id"],
                "pattern_stops": record["pattern_stops"],
                "trip_count": 0,
                "trip_ids": [],
            }
        patterns_dict[key]["trip_count"] += 1
        patterns_dict[key]["trip_ids"].append(record["trip_id"])

    logging.info("Found %d unique patterns.", len(patterns_dict))
    return patterns_dict


def assign_pattern_ids(patterns_dict):
    """
    For each route/service/direction group, assign a pattern_id in ascending order.
    """
    group_map = defaultdict(list)
    for pattern_val in patterns_dict.values():
        route_id_val = pattern_val["route_id"]
        dir_id_val = pattern_val["direction_id"]
        srv_id_val = pattern_val["service_id"]
        group_map[(route_id_val, srv_id_val, dir_id_val)].append(pattern_val)

    out = []
    for (route_id_val, service_id_val, direction_id_val), recs in group_map.items():
        # sort stable by pattern_stops
        recs = sorted(recs, key=lambda x: x["pattern_stops"])
        for idx, pattern_rec in enumerate(recs, 1):
            pattern_rec["pattern_id"] = idx
            out.append(
                {
                    "route_id": pattern_rec["route_id"],
                    "direction_id": pattern_rec["direction_id"],
                    "service_id": pattern_rec["service_id"],
                    "pattern_stops": pattern_rec["pattern_stops"],
                    "trip_count": pattern_rec["trip_count"],
                    "trip_ids": pattern_rec["trip_ids"],
                    "pattern_id": idx,
                }
            )
    logging.info("Assigned pattern IDs to pattern records.")
    return out


# -----------------------------------------------------------------------------
# EARLIEST START TIME
# -----------------------------------------------------------------------------


def compute_earliest_start_times(pattern_records, stop_times_df):
    """
    For each pattern, find the earliest arrival/departure time for the first stop
    of each trip in that pattern.
    """
    if "arrival_time" not in stop_times_df.columns:
        for rec in pattern_records:
            rec["earliest_time_minutes"] = None
            rec["earliest_time_str"] = ""
        return

    stop_times_by_trip = stop_times_df.groupby("trip_id")
    for rec in pattern_records:
        trip_ids = rec["trip_ids"]
        earliest_val = None
        for t_id in trip_ids:
            if t_id not in stop_times_by_trip.groups:
                continue
            group_2 = stop_times_by_trip.get_group(t_id).sort_values("stop_sequence")
            if group_2.empty:
                continue
            arr = group_2.iloc[0].get("arrival_time", "")
            dep = group_2.iloc[0].get("departure_time", "")
            arr_minutes = parse_time_to_minutes(str(arr)) if arr else None
            dep_minutes = parse_time_to_minutes(str(dep)) if dep else None

            candidates = []
            if arr_minutes is not None:
                candidates.append(arr_minutes)
            if dep_minutes is not None:
                candidates.append(dep_minutes)
            if not candidates:
                continue
            this_min = min(candidates)
            if earliest_val is None or this_min < earliest_val:
                earliest_val = this_min
        rec["earliest_time_minutes"] = earliest_val
        rec["earliest_time_str"] = minutes_to_hhmm(earliest_val) if earliest_val else ""


# -----------------------------------------------------------------------------
# MASTER TRIP
# -----------------------------------------------------------------------------


def find_master_trip_stops(
    route_id_val, direction_id_val, relevant_trips, stop_times_df, stops_df
):
    """
    Among 'relevant_trips' for route+direction, find the trip with the most stops/timepoints.
    Return a list of (stop_id, stop_name).
    """
    if relevant_trips.empty:
        return []
    st_sub = stop_times_df[stop_times_df["trip_id"].isin(relevant_trips["trip_id"])]
    if "timepoint" in st_sub.columns and EXPORT_TIMEPOINTS_ONLY:
        st_sub = st_sub[st_sub["timepoint"] == 1]

    sizes = st_sub.groupby("trip_id").size()
    if sizes.empty:
        return []
    best_trip_id = sizes.idxmax()
    best_group = st_sub[st_sub["trip_id"] == best_trip_id].sort_values("stop_sequence")
    best_group = pd.merge(
        best_group, stops_df[["stop_id", "stop_name"]], on="stop_id", how="left"
    )

    out_list = []
    for _, row in best_group.iterrows():
        stop_id_val = row["stop_id"]
        stop_name_val = row.get("stop_name", "Unknown")
        out_list.append((stop_id_val, stop_name_val))
    return out_list


def forward_match_pattern_to_master(pattern_stops, master_stops):
    """
    Forward-only match: place pattern distances in the matching columns (by stop_id).
    The first matched pattern stop => '-'.
    """
    result = [""] * len(master_stops)
    i = 0
    j = 0
    while i < len(master_stops) and j < len(pattern_stops):
        master_sid = master_stops[i][0]
        pat_sid = pattern_stops[j][0]
        dist_str = pattern_stops[j][1]
        if master_sid == pat_sid:
            result[i] = dist_str
            i += 1
            j += 1
        else:
            i += 1
    return result


# -----------------------------------------------------------------------------
# EXCEL EXPORT
# -----------------------------------------------------------------------------


def create_workbook():
    """
    Create and return a new openpyxl Workbook instance with the default sheet removed.
    """
    workbook = Workbook()
    default_sheet = workbook.active
    workbook.remove(default_sheet)
    return workbook


def fill_worksheet_for_direction(
    workbook,
    sheet_title,
    route_short_name,
    direction_id_val,
    service_id_val,
    pattern_records_dir,
    master_stops,
):
    """
    Create one Excel sheet for route/direction. Each pattern is one row, columns:
    [
        Route, Direction, Calendar (service_id), Pattern ID, Trip Count,
        Earliest Start Time, <one column per master_stop>
    ].
    """
    try:
        worksheet = workbook.create_sheet(title=sheet_title)
    except ValueError:
        # If there's a naming conflict or length issue, slightly rename
        worksheet = workbook.create_sheet(title=f"{sheet_title[:25]}_X")

    if not master_stops:
        worksheet.append(["No master stops found for direction."])
        return

    header = [
        "Route",
        "Direction",
        "Calendar (service_id)",
        "Pattern ID",
        "Trip Count",
        "Earliest Start Time",
    ]
    for _, stop_name_val in master_stops:
        header.append(stop_name_val)
    worksheet.append(header)

    # Sort by earliest_time_minutes
    pattern_records_dir = sorted(
        pattern_records_dir,
        key=lambda rec: (
            rec.get("earliest_time_minutes") is None,
            rec.get("earliest_time_minutes", 9999999),
        ),
    )

    for rec in pattern_records_dir:
        pat_id = rec["pattern_id"]
        trip_count = rec["trip_count"]
        earliest_str = rec.get("earliest_time_str", "")
        pattern_stops = rec["pattern_stops"]
        row_distances = forward_match_pattern_to_master(pattern_stops, master_stops)

        row_data = [
            route_short_name,
            direction_id_val,
            service_id_val,
            pat_id,
            trip_count,
            earliest_str,
        ]
        row_data.extend(row_distances)
        worksheet.append(row_data)

    # Set column widths
    for col_index, _ in enumerate(header, 1):
        col_letter = get_column_letter(col_index)
        worksheet.column_dimensions[col_letter].width = 30


def export_patterns_to_excel(
    pattern_records, routes_df, stop_times_df, stops_df, calendar_df=None
):
    """
    For each (route_id, service_id), group patterns by direction, create a subfolder
    named for that service_id's days (if calendar.txt loaded), and save an Excel workbook
    with one sheet per direction.
    """
    group_map = defaultdict(list)
    for pat_rec in pattern_records:
        rid_val = pat_rec["route_id"]
        sid_val = pat_rec["service_id"]
        group_map[(rid_val, sid_val)].append(pat_rec)

    for (rid_val, sid_val), group_list in group_map.items():
        route_info = routes_df[routes_df["route_id"] == rid_val]
        if not route_info.empty:
            short_name = route_info.iloc[0].get("route_short_name", f"Route_{rid_val}")
        else:
            short_name = f"Route_{rid_val}"

        folder_name = format_service_id_folder_name(sid_val, calendar_df)
        folder_path = os.path.join(OUTPUT_DIR, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        workbook = create_workbook()

        # Group by direction_id
        dir_map = defaultdict(list)
        for rec_dir in group_list:
            d_id_val = rec_dir["direction_id"]
            dir_map[d_id_val].append(rec_dir)

        for direction_val, recs_dir in dir_map.items():
            # gather all trip_ids
            all_trip_ids = set()
            for pattern_rec in recs_dir:
                all_trip_ids.update(pattern_rec["trip_ids"])

            st_sub = stop_times_df[stop_times_df["trip_id"].isin(all_trip_ids)]
            if "timepoint" in st_sub.columns and EXPORT_TIMEPOINTS_ONLY:
                st_sub = st_sub[st_sub["timepoint"] == 1]

            # pick trip with max # stops => master_stops
            sizes = st_sub.groupby("trip_id").size()
            if sizes.empty:
                master_stops = []
            else:
                best_tid = sizes.idxmax()
                merged_group = st_sub[st_sub["trip_id"] == best_tid]
                merged_group = merged_group.sort_values("stop_sequence")
                merged_group = pd.merge(
                    merged_group,
                    stops_df[["stop_id", "stop_name"]],
                    on="stop_id",
                    how="left",
                )
                master_stops = []
                for _, rowz in merged_group.iterrows():
                    master_stops.append(
                        (rowz["stop_id"], rowz.get("stop_name", "Unknown"))
                    )

            sheet_title = f"Dir{direction_val}"
            fill_worksheet_for_direction(
                workbook,
                sheet_title,
                short_name,
                direction_val,
                sid_val,
                recs_dir,
                master_stops,
            )

        filename = f"{short_name}_{sid_val}_{SIGNUP_NAME}.xlsx"
        full_filepath = os.path.join(folder_path, filename)
        try:
            workbook.save(full_filepath)
            logging.info("Saved workbook: %s", full_filepath)
        except Exception as exc:
            logging.error("Could not save workbook '%s': %s", filename, exc)


# =============================================================================
# MAIN
# =============================================================================


def main():
    """
    Main entry point for generating and exporting unique GTFS stop patterns.

    Steps:
    1) Create the output directory if it doesn’t exist.
    2) Optionally load calendar.txt for subfolder naming by service day.
    3) Load the core GTFS tables (stops, trips, stop_times, routes).
    4) Filter trips if necessary based on route short names and calendar IDs.
    5) Build unique stop patterns for each trip group.
    6) Compute earliest start times for each pattern.
    7) Export the patterns to Excel, creating subfolders grouped by service_id.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    calendar_df = None
    cal_path = os.path.join(INPUT_DIR, CALENDAR_FILE)
    if os.path.exists(cal_path):
        try:
            calendar_df = pd.read_csv(cal_path, dtype=str)
            logging.info("Loaded calendar.txt successfully.")
        except Exception as exc:
            logging.warning("Could not load calendar.txt: %s", exc)
            calendar_df = None
    else:
        logging.info(
            "No calendar.txt found; subfolders will be 'calendar_<service_id>' only."
        )

    try:
        gtfs_data = load_gtfs_files(INPUT_DIR)
    except Exception as exc:
        logging.error("Failed to load GTFS: %s", exc)
        return

    stops_df = gtfs_data["stops"]
    trips_df = gtfs_data["trips"]
    stop_times_df = gtfs_data["stop_times"]
    routes_df = gtfs_data["routes"]

    filtered_trips = filter_trips(trips_df, routes_df, cal_ids=FILTER_IN_CALENDAR_IDS)
    if filtered_trips.empty:
        logging.error("No trips after filtering. Exiting.")
        return

    patterns_dict = generate_unique_patterns(filtered_trips, stop_times_df, stops_df)
    if not patterns_dict:
        logging.warning("No patterns found. Exiting.")
        return

    pattern_records = assign_pattern_ids(patterns_dict)
    if not pattern_records:
        logging.warning("No pattern records. Exiting.")
        return

    compute_earliest_start_times(pattern_records, stop_times_df)
    export_patterns_to_excel(
        pattern_records, routes_df, stop_times_df, stops_df, calendar_df
    )

    logging.info("Processing complete.")


if __name__ == "__main__":
    main()
