"""
Script Name:
    stops_versus_road_name_typo_checker_arcpy.py
"""

import difflib
import logging
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import arcpy
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

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------


def create_work_gdb(base_dir: str) -> str:
    """
    Create <base_dir>/typo_work_<timestamp>.gdb and return its path.
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
    path = os.path.join(folder, "stops.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"stops.txt not found in {folder}")
    df = pd.read_csv(path, dtype=str, low_memory=False)
    need = {"stop_id", "stop_name", "stop_lat", "stop_lon"}
    if not need.issubset(df.columns):
        raise ValueError(
            f"stops.txt missing columns: {', '.join(need - set(df.columns))}"
        )
    df["stop_lat"] = df["stop_lat"].astype(float)
    df["stop_lon"] = df["stop_lon"].astype(float)
    return df


def normalize_street(name: str, mods: set[str]) -> str:
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
    if not isinstance(stop_name, str):
        return []
    seps = [" @ ", " and ", " & ", "/", " intersection of "]
    parts = re.split("|".join(map(re.escape, seps)), stop_name, flags=re.IGNORECASE)
    return [normalize_street(p, mods) for p in parts if p.strip()]


def dl_score(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio() * 100


def make_stops_fc(df: pd.DataFrame, out_fc: str, sr: int) -> None:
    """Create a simple point FC for GTFS stops."""
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
    """
    Project `in_fc` to `out_fc`. If Project fails, fall back to CopyFeatures.
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
    if arcpy.Exists(out_fc):
        arcpy.management.Delete(out_fc)
    arcpy.analysis.Buffer(in_fc, out_fc, f"{dist} {unit}", dissolve_option="NONE")


def spatial_join_fc(target: str, join: str, out_fc: str) -> None:
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
    return {f.name for f in arcpy.ListFields(fc)}


def map_road_fields(fc: str) -> dict[str, str]:
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
    mods = set()
    with arcpy.da.SearchCursor(fc, [fld]) as cur:
        for (v,) in cur:
            if v:
                mods.add(str(v).strip().lower())
    return mods


def road_clean_dict(fc: str, fullname: str, mods: set[str]) -> dict[str, set[str]]:
    d = defaultdict(set)
    with arcpy.da.SearchCursor(fc, [fullname]) as cur:
        for (full,) in cur:
            if not full:
                continue
            clean = normalize_street(full, mods)
            d[clean].add(full)
    return d


def stop_to_candidate_roads(
    join_fc: str, fullname: str, mods: set[str]
) -> dict[str, set[str]]:
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
    universe = set(road_clean.keys())
    out_rows = []

    for rec in stops_df.itertuples(index=False):
        sid, sname = rec.stop_id, rec.stop_name
        pieces = split_stop_name(sname, mods)
        candidates = stop2roads.get(sid, universe)

        for frag in pieces:
            if frag in candidates:
                continue
            for match in difflib.get_close_matches(
                frag, candidates, n=3, cutoff=thresh / 100
            ):
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

    return (
        pd.DataFrame(out_rows)
        .sort_values("similarity_score", ascending=False)
        .drop_duplicates()
    )


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
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
    mods = modifiers_from_roads(
        roads_proj, col_map.get("RW_TYPE_US", col_map["FULLNAME"])
    )
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
