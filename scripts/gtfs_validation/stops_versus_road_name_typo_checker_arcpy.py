"""Identifies potential typos in GTFS stop names by comparing them to nearby roadway names.

Creates a spatial buffer around each GTFS stop, joins to intersecting roadway segments,
and uses fuzzy string matching to flag stop names that are similar—but not identical—
to adjacent street names. Intended for QA of stop name consistency in GIS-based transit data.

Outputs:
    - CSV of potential stop name typos and similarity scores.
    - File geodatabase with intermediate feature classes for inspection.

Typical use:
    Run in ArcGIS Pro's Python environment or any ArcPy-enabled session
    after configuring input paths and parameters at the top of the script.
"""

import difflib
import logging
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import arcpy  # type: ignore
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

GTFS_FOLDER = r"path\to\your\GTFS"
ROADWAYS_PATH = r"path\to\your\roadways.shp"
OUTPUT_DIR = r"path\to\output"  # any writable folder
OUTPUT_CSV = "potential_typos.csv"

# Spatial references
STOPS_CRS = 4326  # GTFS lat/lon – WGS-84
TARGET_CRS = 2248  # example: VA North (US ft). change if needed

# Processing parameters
BUFFER_DISTANCE = 50
BUFFER_DISTANCE_UNIT = "feet"  # 'feet' or 'meters'
SIMILARITY_THRESHOLD = 80  # 0-100

# Roadway field requirements
REQUIRED_COLUMNS_ROADWAY = [
    "RW_PREFIX",
    "RW_TYPE_US",
    "RW_SUFFIX",
    "RW_SUFFIX_",
    "FULLNAME",
]

DESCRIPTIONS_ROADWAY = {
    "RW_PREFIX": "Directional prefix (e.g. 'N' in 'N Washington St')",
    "RW_TYPE_US": "Street type (e.g. 'St' in 'N Washington St')",
    "RW_SUFFIX": "Directional suffix (e.g. 'SE' in 'Park St SE')",
    "RW_SUFFIX_": "Additional suffix (e.g. 'EB' in 'I-66 EB')",
    "FULLNAME": "Full street name",
}

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
arcpy.env.overwriteOutput = True

# =============================================================================
# FUNCTIONS
# =============================================================================


def create_work_gdb(base_dir: str) -> str:
    """Create <base_dir>/typo_work_<timestamp>.gdb and return its path.

    Re-use if it already exists in this session.
    """
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"typo_work_{ts}.gdb"
    gdb = os.path.join(base_dir, name)
    if not arcpy.Exists(gdb):
        arcpy.management.CreateFileGDB(base_dir, name)
        logging.info("Created workspace %s", gdb)
    else:
        logging.info("Using existing workspace %s", gdb)
    return gdb


def fgdb_path(gdb: str, fc_name: str) -> str:
    """Return full path inside the work GDB."""
    return os.path.join(gdb, fc_name)


# -----------------------------------------------------------------------------
# OTHER FUNCTIONS
# -----------------------------------------------------------------------------


def load_gtfs_stops(folder: str) -> pd.DataFrame:
    """Loads GTFS stops from stops.txt into a pandas DataFrame.

    Args:
        folder: The path to the folder containing the GTFS `stops.txt` file.

    Returns:
        A pandas DataFrame with stop data.

    Raises:
        FileNotFoundError: If `stops.txt` is not found in the specified folder.
        ValueError: If the `stops.txt` file is missing required columns.
    """
    path = os.path.join(folder, "stops.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"stops.txt not found in {folder}")
    df = pd.read_csv(path, dtype=str, low_memory=False)
    need = {"stop_id", "stop_name", "stop_lat", "stop_lon"}
    if not need.issubset(df.columns):
        raise ValueError(f"stops.txt missing columns: {', '.join(need - set(df.columns))}")
    df["stop_lat"] = df["stop_lat"].astype(float)
    df["stop_lon"] = df["stop_lon"].astype(float)
    return df


def normalize_street(name: str, mods: set[str]) -> str:
    """Cleans and standardizes a street name string.

    Removes modifiers (e.g., 'St', 'Ave'), punctuation, and extra whitespace,
    and converts the string to lowercase.

    Args:
        name: The street name to normalize.
        mods: A set of modifier words (like 'rd', 'st', 'blvd') to remove.

    Returns:
        The normalized street name string.
    """
    if not isinstance(name, str):
        return ""
    if mods:
        name = re.sub(
            r"\b(" + "|".join(map(re.escape, mods)) + r")\b",
            " ",
            name,
            flags=re.IGNORECASE,
        )
    name = re.sub(r"[^\w\s]", " ", name)
    return re.sub(r"\s+", " ", name).strip().lower()


def split_stop_name(stop_name: str, mods: set[str]) -> list[str]:
    """Splits a GTFS stop name into normalized street name components.

    Uses common intersection separators (e.g., '@', '&', '/') to divide the
    stop name and then normalizes each resulting part.

    Args:
        stop_name: The full stop name string.
        mods: A set of modifier words to be removed during normalization.

    Returns:
        A list of normalized street name fragments from the stop name.
    """
    if not isinstance(stop_name, str):
        return []
    seps = [" @ ", " and ", " & ", "/", " intersection of "]
    parts = re.split("|".join(map(re.escape, seps)), stop_name, flags=re.IGNORECASE)
    return [normalize_street(p, mods) for p in parts if p.strip()]


def dl_score(a: str, b: str) -> float:
    """Calculates the Damerau-Levenshtein similarity ratio between two strings.

    Args:
        a: The first string.
        b: The second string.

    Returns:
        A similarity score between 0.0 and 100.0.
    """
    return difflib.SequenceMatcher(None, a, b).ratio() * 100


def make_stops_fc(df: pd.DataFrame, out_fc: str, sr: int) -> None:
    """Creates a point feature class from a DataFrame of GTFS stops.

    Args:
        df: DataFrame containing stop_id, stop_name, stop_lon, stop_lat.
        out_fc: The full path for the output feature class.
        sr: The spatial reference ID (WKID) for the output feature class.
    """
    if arcpy.Exists(out_fc):
        arcpy.management.Delete(out_fc)
    arcpy.management.CreateFeatureclass(
        os.path.dirname(out_fc),
        os.path.basename(out_fc),
        "POINT",
        spatial_reference=arcpy.SpatialReference(sr),
    )
    arcpy.management.AddField(out_fc, "stop_id", "TEXT", 50)
    arcpy.management.AddField(out_fc, "stop_name", "TEXT", 255)
    with arcpy.da.InsertCursor(out_fc, ["SHAPE@XY", "stop_id", "stop_name"]) as cur:
        for r in df.itertuples(index=False):
            cur.insertRow([(r.stop_lon, r.stop_lat), r.stop_id, r.stop_name])


def safe_project_or_copy(in_fc: str, out_fc: str, out_sr: int) -> None:
    """Project `in_fc` to `out_fc`. If Project fails, fall back to CopyFeatures.

    Ensures `out_fc` exists on return.
    """
    if arcpy.Exists(out_fc):
        arcpy.management.Delete(out_fc)

    desc = arcpy.Describe(in_fc)
    src_sr = desc.spatialReference
    tgt_sr = arcpy.SpatialReference(out_sr)

    try:
        if src_sr.name and src_sr.factoryCode == tgt_sr.factoryCode:
            # Already in target SR
            arcpy.management.CopyFeatures(in_fc, out_fc)
        else:
            arcpy.management.Project(in_fc, out_fc, tgt_sr)
    except Exception as exc:
        logging.warning("Project failed (%s). Copying features instead.", exc)
        arcpy.management.CopyFeatures(in_fc, out_fc)

    if not arcpy.Exists(out_fc):
        raise RuntimeError(f"Failed to create {out_fc}")


def buffer_fc(in_fc: str, out_fc: str, dist: float, unit: str) -> None:
    """Creates a buffer around features in a feature class.

    Args:
        in_fc: The input feature class.
        out_fc: The path for the output buffer feature class.
        dist: The buffer distance.
        unit: The units for the buffer distance (e.g., 'feet', 'meters').
    """
    if arcpy.Exists(out_fc):
        arcpy.management.Delete(out_fc)
    arcpy.analysis.Buffer(in_fc, out_fc, f"{dist} {unit}", dissolve_option="NONE")


def spatial_join_fc(target: str, join: str, out_fc: str) -> None:
    """Performs a one-to-many spatial join.

    Finds all join features that intersect with each target feature.

    Args:
        target: The target feature class.
        join: The feature class to join to the target.
        out_fc: The path for the output joined feature class.
    """
    if arcpy.Exists(out_fc):
        arcpy.management.Delete(out_fc)
    arcpy.analysis.SpatialJoin(
        target,
        join,
        out_fc,
        join_operation="JOIN_ONE_TO_MANY",
        match_option="INTERSECT",
    )


def field_set(fc: str) -> set[str]:
    """Gets a set of field names from a feature class or table."""
    return {f.name for f in arcpy.ListFields(fc)}


def map_road_fields(fc: str) -> dict[str, str]:
    """Prompts user to map required roadway fields if they are not found.

    Args:
        fc: The roadway feature class to check.

    Returns:
        A dictionary mapping required field names to the actual field names
        found in the feature class.

    Raises:
        ValueError: If a user-provided alternative field does not exist, or
            if a mapping for the 'FULLNAME' field is not provided.
    """
    exists = field_set(fc)
    mapping: dict[str, str] = {}
    for col in REQUIRED_COLUMNS_ROADWAY:
        if col in exists:
            mapping[col] = col
        else:
            logging.warning("Field '%s' missing.", col)
            logging.info("Description: %s", DESCRIPTIONS_ROADWAY[col])
            logging.info("Available: %s", ", ".join(sorted(exists)))
            alt = input(f"Enter field name for '{col}' or blank to skip: ").strip()
            if alt:
                if alt in exists:
                    mapping[col] = alt
                else:
                    raise ValueError(f"Field '{alt}' not present.")
    if "FULLNAME" not in mapping:
        raise ValueError("You must supply a field for FULLNAME.")
    return mapping


def modifiers_from_roads(fc: str, fld: str) -> set[str]:
    """Extracts a set of unique string values from a feature class field.

    Used to build a set of street modifiers (e.g., 'St', 'Rd', 'N') for
    normalization.

    Args:
        fc: The feature class to query.
        fld: The field from which to extract values.

    Returns:
        A set of unique, lowercase string values from the specified field.
    """
    mods = set()
    with arcpy.da.SearchCursor(fc, [fld]) as cur:
        for (v,) in cur:
            if v:
                mods.add(str(v).strip().lower())
    return mods


def road_clean_dict(fc: str, fullname: str, mods: set[str]) -> dict[str, set[str]]:
    """Creates a lookup from normalized road names to original names.

    Args:
        fc: The roadway feature class.
        fullname: The field containing the full roadway name.
        mods: A set of modifiers to remove during normalization.

    Returns:
        A dictionary where keys are normalized road names and values are sets
        of the original, un-normalized names corresponding to each key.
    """
    d = defaultdict(set)
    with arcpy.da.SearchCursor(fc, [fullname]) as cur:
        for (full,) in cur:
            if not full:
                continue
            clean = normalize_street(full, mods)
            d[clean].add(full)
    return d


def stop_to_candidate_roads(join_fc: str, fullname: str, mods: set[str]) -> dict[str, set[str]]:
    """Maps each stop ID to the set of nearby, normalized road names.

    Args:
        join_fc: The feature class from the stop-to-road spatial join.
        fullname: The field containing the full roadway name.
        mods: A set of modifiers to remove during road name normalization.

    Returns:
        A dictionary where keys are 'stop_id's and values are sets of
        normalized names of roads that were spatially joined to that stop.
    """
    sc = defaultdict(set)
    with arcpy.da.SearchCursor(join_fc, ["stop_id", fullname]) as cur:
        for sid, full in cur:
            if full:
                sc[sid].add(normalize_street(full, mods))
    return sc


def detect_typos(
    stops_df: pd.DataFrame,
    stop2roads: dict[str, set[str]],
    road_clean: dict[str, set[str]],
    mods: set[str],
    thresh: int,
) -> pd.DataFrame:
    """Compares stop name parts to nearby road names to find likely typos.

    For each stop, it splits the stop name into parts. Each part is then
    compared against the set of nearby road names for that stop. If a part
    is very similar (but not identical) to a nearby road name, it's flagged
    as a potential typo.

    Args:
        stops_df: DataFrame of all GTFS stops.
        stop2roads: A mapping from stop_id to a set of nearby normalized road names.
        road_clean: A mapping from a normalized road name to its original form(s).
        mods: A set of modifiers to remove during name normalization.
        thresh: The similarity score (0-100) threshold for flagging a typo.

    Returns:
        A pandas DataFrame containing details of each potential typo found.
    """
    universe = set(road_clean.keys())
    out_rows = []

    for rec in stops_df.itertuples(index=False):
        # Ensure sid and sname are strings to satisfy the type checker
        sid, sname = str(rec.stop_id), str(rec.stop_name)
        pieces = split_stop_name(sname, mods)
        candidates = stop2roads.get(sid, universe)

        for frag in pieces:
            if frag in candidates:
                continue
            for match in difflib.get_close_matches(frag, candidates, n=3, cutoff=thresh / 100):
                score = dl_score(frag, match)
                if thresh <= score < 100:
                    for orig in road_clean.get(match, {match}):
                        out_rows.append(
                            {
                                "stop_id": sid,
                                "stop_name": sname,
                                "street_in_stop_name": frag,
                                "similar_road_name_clean": match,
                                "similar_road_name_orig": orig,
                                "similarity_score": round(score, 1),
                            }
                        )

    if not out_rows:
        return pd.DataFrame()

    return pd.DataFrame(out_rows).sort_values("similarity_score", ascending=False).drop_duplicates()


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Main script execution function."""
    # workspace
    WORK_GDB = create_work_gdb(OUTPUT_DIR)

    # GTFS stops
    logging.info("Loading GTFS stops …")
    stops_df = load_gtfs_stops(GTFS_FOLDER)
    stops_raw = fgdb_path(WORK_GDB, "stops_raw")
    make_stops_fc(stops_df, stops_raw, STOPS_CRS)

    # Project stops
    stops_proj = fgdb_path(WORK_GDB, "stops_proj")
    logging.info("Projecting stops → %s …", TARGET_CRS)
    safe_project_or_copy(stops_raw, stops_proj, TARGET_CRS)

    # Project roads
    roads_proj = fgdb_path(WORK_GDB, "roads_proj")
    logging.info("Projecting roads …")
    safe_project_or_copy(ROADWAYS_PATH, roads_proj, TARGET_CRS)

    # Roadway schema
    logging.info("Mapping roadway fields …")
    col_map = map_road_fields(roads_proj)
    mods = modifiers_from_roads(roads_proj, col_map.get("RW_TYPE_US", col_map["FULLNAME"]))
    logging.info("Found %d modifiers.", len(mods))

    # Buffer stops
    stops_buf = fgdb_path(WORK_GDB, "stops_buf")
    logging.info("Buffering stops (%s %s) …", BUFFER_DISTANCE, BUFFER_DISTANCE_UNIT)
    buffer_fc(stops_proj, stops_buf, BUFFER_DISTANCE, BUFFER_DISTANCE_UNIT)

    # Spatial join
    join_fc = fgdb_path(WORK_GDB, "stops_roads_join")
    logging.info("SpatialJoin buffers ↔ roads …")
    spatial_join_fc(stops_buf, roads_proj, join_fc)

    # Build lookup dictionaries
    r_clean = road_clean_dict(roads_proj, col_map["FULLNAME"], mods)
    stop2rd = stop_to_candidate_roads(join_fc, col_map["FULLNAME"], mods)

    # Detect typos
    logging.info("Running difflib matching …")
    typos = detect_typos(stops_df, stop2rd, r_clean, mods, SIMILARITY_THRESHOLD)

    # Output
    out_csv = os.path.join(OUTPUT_DIR, OUTPUT_CSV)
    if typos.empty:
        logging.info("No potential typos found.")
    else:
        typos.to_csv(out_csv, index=False)
        logging.info("Wrote %d rows → %s", len(typos), out_csv)

    logging.info("All done. Workspace retained at %s for inspection.", WORK_GDB)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Processing failed")
        sys.exit(1)
