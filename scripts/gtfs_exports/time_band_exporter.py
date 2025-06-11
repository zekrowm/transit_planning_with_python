"""
Export consolidated time-band tables from schedule-based data.
Groups trips by exact pattern and segment runtimes, then outputs
Excel workbooks by route and service ID.

For every group of trips that share **exactly the same** time-point
pattern *and* identical segment run-times, produce one row like:

* FrTime / ToTime  – earliest & latest first-stop departures.
* Segment columns  – runtime (min) from previous time-point; first cell “–”.
* One workbook per (route_id, service_id); one sheet per direction.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import pandas as pd
from openpyxl import Workbook
from openpyxl.utils import get_column_letter

# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

GTFS_FOLDER = Path(r"G:\projects\dot\zkrohmal\_data\gtfs\connector_gtfs_2025_06_06")
OUTPUT_FOLDER = Path(
    r"\\S40SHAREPGC01\DOTWorking\zkrohm\data_requests\stop_pattern_exporter_test\output"
)

# Optional filters – leave empty to take everything
FILTER_IN_ROUTE_SHORT_NAMES: List[str] = []
FILTER_OUT_ROUTE_SHORT_NAMES: List[str] = []
FILTER_IN_SERVICE_IDS: List[str] = []
FILTER_OUT_SERVICE_IDS: List[str] = []

# --------------------------------------------------------------------------------------------------
# LOGGING
# --------------------------------------------------------------------------------------------------

LOG_LEVEL = logging.INFO
EXPORT_TIMEPOINTS_ONLY = True         # keep only stops where timepoint == 1
MISSING_TIME = "–"

_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$")

# ==================================================================================================
# FUNCTIONS
# ==================================================================================================

def hhmmss_to_min(t: Optional[str]) -> Optional[int]:
    """HH:MM[:SS] → minutes (can be ≥24 h)."""
    if not isinstance(t, str):
        return None
    m = _TIME_RE.match(t.strip())
    if not m:
        return None
    h, mm, ss = m.groups()
    return int(h) * 60 + int(mm) + round(int(ss or 0) / 60)


def min_to_hhmm(mn: Optional[int]) -> str:
    """Minutes → 'H:MM'; preserves ≥24 h."""
    if mn is None:
        return MISSING_TIME
    h, m = divmod(mn, 60)
    return f"{h}:{m:02d}"


def safe_sheet(name: str) -> str:
    """Excel-safe sheet title ≤31 chars."""
    return re.sub(r"[:\\/*?\[\]]", "_", name)[:31] or "Sheet"


# ─────────────────────────  LOAD GTFS  ────────────────────────────── #
REQ = ["trips.txt", "stop_times.txt", "routes.txt", "stops.txt"]


def load_gtfs(folder: Path) -> Dict[str, pd.DataFrame]:
    missing = [f for f in REQ if not (folder / f).exists()]
    if missing:
        raise FileNotFoundError("Missing GTFS file(s): " + ", ".join(missing))

    data = {f[:-4]: pd.read_csv(folder / f, dtype=str, low_memory=False) for f in REQ}

    st = data["stop_times"]
    st["stop_sequence"] = pd.to_numeric(st["stop_sequence"], errors="raise")
    if "timepoint" in st.columns:
        st["timepoint"] = pd.to_numeric(st["timepoint"], errors="coerce")
    return data


# ──────────────────  PER-TRIP PATTERN & RUNTIMES  ─────────────────── #
PatternTuple = Tuple[str, ...]
RuntimeSegTuple = Tuple[Union[int, str], ...]  # first element is "–"


def segment_runtimes(grp: pd.DataFrame) -> RuntimeSegTuple:
    """Return tuple of segment runtimes; first element '–'."""
    times: List[Optional[int]] = []
    for _, row in grp.iterrows():
        dep = hhmmss_to_min(row.get("departure_time"))
        arr = hhmmss_to_min(row.get("arrival_time"))
        times.append(dep if dep is not None else arr)

    segs: List[Union[int, str]] = [MISSING_TIME]
    for a, b in zip(times, times[1:]):
        segs.append(b - a if a is not None and b is not None else "")
    return tuple(segs)


def build_index(
    gtfs: Dict[str, pd.DataFrame]
) -> Tuple[
    pd.DataFrame,
    Dict[int, PatternTuple],
    Dict[int, RuntimeSegTuple],
    Dict[int, List[str]],
]:
    """
    Build trip-level index plus lookup dictionaries.

    Returns
    -------
    trips_df      rows = one trip
    stop_dict     pattern_hash → stop_id tuple
    seg_dict      seg_hash → segment runtime tuple
    header_names  pattern_hash → list of 'Stop name (code)'
    """
    trips = gtfs["trips"].copy()
    routes = gtfs["routes"][["route_id", "route_short_name"]]
    trips = trips.merge(routes, on="route_id", how="left")

    # Apply filters
    if FILTER_IN_ROUTE_SHORT_NAMES:
        trips = trips[trips["route_short_name"].isin(FILTER_IN_ROUTE_SHORT_NAMES)]
    if FILTER_OUT_ROUTE_SHORT_NAMES:
        trips = trips[~trips["route_short_name"].isin(FILTER_OUT_ROUTE_SHORT_NAMES)]
    if FILTER_IN_SERVICE_IDS:
        trips = trips[trips["service_id"].isin(FILTER_IN_SERVICE_IDS)]
    if FILTER_OUT_SERVICE_IDS:
        trips = trips[~trips["service_id"].isin(FILTER_OUT_SERVICE_IDS)]

    if trips.empty:
        return pd.DataFrame(), {}, {}, {}

    st = gtfs["stop_times"].merge(
        trips[["trip_id", "route_id", "service_id", "direction_id"]],
        on="trip_id",
        how="inner",
    )

    if EXPORT_TIMEPOINTS_ONLY and "timepoint" in st.columns:
        st = st[st["timepoint"] == 1]

    st = st.sort_values(["trip_id", "stop_sequence"])

    stop_dict: Dict[int, PatternTuple] = {}
    seg_dict: Dict[int, RuntimeSegTuple] = {}
    header_names: Dict[int, List[str]] = {}
    rows = []

    stops_lookup = (
        gtfs["stops"][["stop_id", "stop_name", "stop_code"]]
        .set_index("stop_id")
        .to_dict("index")
    )

    for trip_id, grp in st.groupby("trip_id"):
        pattern: PatternTuple = tuple(grp["stop_id"])
        pat_hash = hash(pattern)
        stop_dict.setdefault(pat_hash, pattern)

        segs = segment_runtimes(grp)
        seg_hash = hash(segs)
        seg_dict.setdefault(seg_hash, segs)

        if pat_hash not in header_names:
            header_names[pat_hash] = [
                f"{rec['stop_name']} ({rec['stop_code']})"
                if (rec := stops_lookup.get(sid)) is not None
                else sid
                for sid in pattern
            ]

        start = hhmmss_to_min(
            grp.iloc[0].get("departure_time") or grp.iloc[0].get("arrival_time")
        )
        if start is None:
            continue

        rows.append(
            {
                "route_id": grp.iloc[0]["route_id"],
                "service_id": grp.iloc[0]["service_id"],
                "direction_id": grp.iloc[0]["direction_id"],
                "pattern_hash": pat_hash,
                "seg_hash": seg_hash,
                "start": start,
            }
        )

    return pd.DataFrame(rows), stop_dict, seg_dict, header_names


# ───────────────────  GROUP into TIME-BAND ROWS  ──────────────────── #
def make_bands(idx: pd.DataFrame) -> pd.DataFrame:
    gb = idx.groupby(
        ["route_id", "service_id", "direction_id", "pattern_hash", "seg_hash"]
    )
    out = (
        gb["start"]
        .agg(["min", "max", "count"])
        .rename(columns={"min": "FrTime", "max": "ToTime", "count": "Total"})
        .reset_index()
        .sort_values("FrTime")
    )
    return out


# ──────────────────────────  EXCEL EXPORT  ────────────────────────── #
def export_excel(
    bands: pd.DataFrame,
    stop_dict: Dict[int, PatternTuple],
    seg_dict: Dict[int, RuntimeSegTuple],
    header_names: Dict[int, List[str]],
    routes: pd.DataFrame,
) -> None:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    route_short = routes.set_index("route_id")["route_short_name"].to_dict()

    for (rid, sid), grp_rs in bands.groupby(["route_id", "service_id"]):
        wb = Workbook()
        wb.remove(wb.active)

        for did, grp_dir in grp_rs.groupby("direction_id"):
            ws = wb.create_sheet(safe_sheet(f"Dir_{did or 'X'}"))

            # header uses first pattern in this direction
            first_pat = grp_dir.iloc[0]["pattern_hash"]
            hdr = ["Pattern", "FrTime", "ToTime", "Total", *header_names[first_pat]]
            ws.append(hdr)

            for _, r in grp_dir.iterrows():
                segs = seg_dict[r["seg_hash"]]
                ws.append(
                    [
                        r["pattern_hash"],
                        min_to_hhmm(int(r["FrTime"])),
                        min_to_hhmm(int(r["ToTime"])),
                        r["Total"],
                        *segs,
                    ]
                )

            for col in range(1, len(hdr) + 1):
                ws.column_dimensions[get_column_letter(col)].width = 18

        fname = OUTPUT_FOLDER / (
            f"route_{route_short.get(rid, rid)}_cal{sid}_timeband_table.xlsx"
        )
        wb.save(fname)
        logging.info("Wrote %s", fname)

# ==================================================================================================
# MAIN
# ==================================================================================================

def main() -> None:
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.info("GTFS folder   : %s", GTFS_FOLDER)
    logging.info("Output folder : %s", OUTPUT_FOLDER)

    gtfs = load_gtfs(GTFS_FOLDER)
    idx, stop_dict, seg_dict, header_names = build_index(gtfs)

    if idx.empty:
        logging.warning("Nothing to export – check filters or timepoints.")
        return

    bands = make_bands(idx)
    export_excel(bands, stop_dict, seg_dict, header_names, gtfs["routes"])

    logging.info("Finished – %d band rows.", len(bands))


if __name__ == "__main__":
    main()
