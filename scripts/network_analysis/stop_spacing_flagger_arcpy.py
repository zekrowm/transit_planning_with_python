"""Generates GIS data layers and flags short inter-stop spacings from a GTFS feed.

This module processes a GTFS (General Transit Feed Specification) dataset to
produce several geospatial outputs and a report on transit segment lengths.

Outputs:
    1.  `stops.shp`: A projected point feature class representing GTFS stops.
    2.  `routes.shp`: A projected polyline feature class representing GTFS
        shapes (one feature per `shape_id`) or dissolved routes (one feature
        per `route_id`), configurable via the `ROUTE_LEVEL` parameter.
    3.  `segments.shp`: A projected polyline feature class representing
        inter-stop segments, derived from linear referencing of stops along
        routes. This feature class includes a `LENGTH_FT` field indicating
        the segment's length.
    4.  `short_spacing_segments.txt`: A tab-delimited log file (text) that
        identifies and lists any inter-stop segments found to be shorter
        than the `MIN_SPACING_FT` threshold defined in the configuration.

Requirements:
    -   An ArcGIS Pro Standard license is required.
"""

from __future__ import annotations

import csv
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import arcpy

# =============================================================================
# CONFIGURATION
# =============================================================================

GTFS_FOLDER: str = r"Path\To\Your\GTFS_Folder"
OUTPUT_FOLDER: str = r"Path\To\Your\Output_Folder"
EPSG_CODE: int = 2248  # StatePlane Maryland (US survey feet)
ROUTE_LEVEL: str = "shape"  # "shape" or "route" (dissolve to route_id)
MIN_SPACING_FT: float = 400.0  # threshold for short‐spacing log
SPACING_LOG_FILE: str = "short_spacing_segments.txt"

# Optional list of route_ids to process; leave empty ([]) for no filtering.
ROUTE_FILTER: List[str] = []

# =============================================================================
# HELPERS
# =============================================================================


def validate_config() -> None:
    """Ensure GTFS folder & required files exist; create output folder."""
    gtfs = Path(GTFS_FOLDER)
    out = Path(OUTPUT_FOLDER)
    if not gtfs.exists():
        raise FileNotFoundError(f"GTFS folder not found: {gtfs}")
    required = ["stops.txt", "shapes.txt"]
    if ROUTE_LEVEL.lower() == "route":
        required += ["trips.txt", "routes.txt"]
    missing = [f for f in required if not (gtfs / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing GTFS files: {', '.join(missing)}")
    out.mkdir(parents=True, exist_ok=True)


def make_xy_event_layer(
    table: str, x_field: str, y_field: str, out_layer: str, sr: arcpy.SpatialReference
) -> str:
    """Make an in‐memory XY event layer and return its name."""
    arcpy.AddMessage(f" → MakeXYEventLayer: {out_layer}")
    arcpy.MakeXYEventLayer_management(table, x_field, y_field, out_layer, sr)
    return out_layer


def points_to_line(point_layer: str, out_line: str, line_field: str, sort_field: str) -> None:
    """Convert points to lines, grouping by line_field and sorting by sort_field."""
    arcpy.AddMessage(" → PointsToLine")
    arcpy.PointsToLine_management(
        point_layer, out_line, Line_Field=line_field, Sort_Field=sort_field
    )


def apply_route_filter(
    raw_fc: str,
    gtfs_folder: str,
    route_field: str,
    per_shape: bool,
    filter_routes: List[str],
) -> str:
    """Return an in-memory feature class containing only the desired routes."""
    arcpy.AddMessage(f" → Applying route filter: {filter_routes!r}")
    lyr = "routes_filter_lyr"
    arcpy.MakeFeatureLayer_management(raw_fc, lyr)

    if per_shape:
        # shape_id ➜ route_id lookup
        mapping: Dict[str, str] = {}
        trips_path = Path(gtfs_folder) / "trips.txt"
        with open(trips_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                sid = row.get("shape_id")
                if sid:
                    mapping[sid] = row["route_id"]

        good_shapes = [sid for sid, rid in mapping.items() if rid in filter_routes]
        if good_shapes:
            shape_list = ",".join(f"'{s}'" for s in good_shapes)
            where = f"shape_id IN ({shape_list})"
        else:
            where = "1=0"
    else:
        if filter_routes:
            route_list = ",".join(f"'{r}'" for r in filter_routes)
            where = f"route_id IN ({route_list})"
        else:
            where = "1=1"

    arcpy.SelectLayerByAttribute_management(lyr, "NEW_SELECTION", where)
    out_lyr = "in_memory/routes_filtered"
    arcpy.CopyFeatures_management(lyr, out_lyr)
    arcpy.AddMessage(f"   ✔ Filtered routes → {out_lyr}")
    return out_lyr


def project_feature(in_fc: str, out_fc: str, target_sr: arcpy.SpatialReference) -> None:
    """Project a feature‐class or layer to target spatial reference."""
    arcpy.AddMessage(f" → Project → {out_fc}")
    arcpy.Project_management(in_fc, out_fc, target_sr)


def dissolve_routes_by_route_id(
    shapes_fc: str, trips_txt: Path, routes_txt: Path, out_fc: str
) -> None:
    """Dissolve one feature per route_id."""
    arcpy.AddMessage(" → Dissolve → by route_id")
    mapping: Dict[str, str] = {}
    with open(trips_txt, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            sid = row.get("shape_id")
            if sid:
                mapping.setdefault(sid, row["route_id"])
    arcpy.AddField_management(shapes_fc, "route_id", "TEXT", 50)
    with arcpy.da.UpdateCursor(shapes_fc, ["shape_id", "route_id"]) as ucur:
        for sid, _ in ucur:
            ucur.updateRow((sid, mapping.get(sid)))
    arcpy.Dissolve_management(shapes_fc, out_fc, "route_id")
    tmp_routes = arcpy.TableToTable_conversion(
        str(routes_txt), arcpy.env.scratchGDB, "routes_tmp"
    ).getOutput(0)
    arcpy.JoinField_management(out_fc, "route_id", tmp_routes, "route_id")


def create_routes_with_m(in_fc: str, route_id_field: str, out_fc: str) -> None:
    """Create M-enabled routes (measure = LENGTH)."""
    arcpy.AddMessage(" → CreateRoutes (M‐enabled)")
    arcpy.CheckOutExtension("linearReferencing")
    arcpy.lr.CreateRoutes(in_fc, route_id_field, out_fc, measure_source="LENGTH")


def locate_stops_along_routes(
    stops_fc: str,
    routes_fc: str,
    route_id_field: str,
    out_table: str,
    search_radius: str = "100 Feet",
) -> None:
    """Locate stops along M‐enabled routes, output event table."""
    arcpy.AddMessage(" → LocateFeaturesAlongRoutes")
    arcpy.lr.LocateFeaturesAlongRoutes(
        stops_fc,
        routes_fc,
        route_id_field,
        search_radius,
        out_table,
        f"{route_id_field} POINT MEAS",
    )


def build_segments(
    routes_fc: str,
    events_tbl: str,
    route_id_field: str,
    out_fc: str,
    spat_ref: arcpy.SpatialReference,
) -> None:
    """Build inter‐stop segments."""
    arcpy.AddMessage(" → Building segments")
    out_folder = os.path.dirname(out_fc)
    out_name = os.path.splitext(os.path.basename(out_fc))[0]
    arcpy.CreateFeatureclass_management(
        out_folder, out_name, "POLYLINE", spatial_reference=spat_ref
    )
    arcpy.AddField_management(out_fc, "route_id", "TEXT", 50)
    arcpy.AddField_management(out_fc, "from_stop", "TEXT", 50)
    arcpy.AddField_management(out_fc, "to_stop", "TEXT", 50)
    arcpy.AddField_management(out_fc, "length_ft", "DOUBLE")
    routes: Dict[str, arcpy.Polyline] = {}
    with arcpy.da.SearchCursor(routes_fc, [route_id_field, "SHAPE@"]) as scur:
        for rid, geom in scur:
            routes[rid] = geom
    measures: Dict[str, List[Tuple[float, str]]] = {}
    with arcpy.da.SearchCursor(events_tbl, [route_id_field, "MEAS", "stop_id"]) as ecur:
        for rid, meas, sid in ecur:
            measures.setdefault(rid, []).append((meas, sid))
    with arcpy.da.InsertCursor(
        out_fc, ["SHAPE@", "route_id", "from_stop", "to_stop", "length_ft"]
    ) as icur:
        for rid, pts in measures.items():
            pts.sort(key=lambda x: x[0])
            line = routes[rid]
            for (m1, s1), (m2, s2) in zip(pts, pts[1:]):
                if m2 <= m1:
                    continue
                seg = line.segmentAlongLine(m1, m2, False)
                icur.insertRow((seg, rid, s1, s2, seg.length))
    arcpy.AddMessage(f"   ✔ {out_fc}")


def flag_short_spacing(segments_fc: str, threshold: float, log_path: Path) -> None:
    """Log all segments shorter than threshold."""
    arcpy.AddMessage(" → Flagging short‐spacing segments")
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("route_id\tfrom_stop\tto_stop\tlength_ft\n")
        with arcpy.da.SearchCursor(
            segments_fc, ["route_id", "from_stop", "to_stop", "length_ft"]
        ) as scur:
            for rid, frm, to, length in scur:
                if length < threshold:
                    fh.write(f"{rid}\t{frm}\t{to}\t{length:.1f}\n")
    arcpy.AddMessage(f"   ✔ {log_path.name}")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Executes the main workflow for generating GIS data layers and identifying short inter-stop spacings from a GTFS feed.

    This function orchestrates the processing steps, including:
    - Configuration validation.
    - GTFS shape and stop processing.
    - Creation of M-enabled routes.
    - Linear referencing to build inter-stop segments.
    - Identification and logging of short spacing segments.
    """
    validate_config()
    arcpy.env.overwriteOutput = True
    arcpy.env.workspace = arcpy.env.scratchGDB
    gtfs = str(Path(GTFS_FOLDER))
    out = str(Path(OUTPUT_FOLDER))
    sr_wgs84 = arcpy.SpatialReference(4326)
    sr_target = arcpy.SpatialReference(EPSG_CODE)
    per_shape = ROUTE_LEVEL.lower() == "shape"
    route_field = "shape_id" if per_shape else "route_id"

    try:
        # ROUTES
        arcpy.AddMessage("\n=== Routes / Shapes ===")
        shapes_tbl = arcpy.TableToTable_conversion(
            os.path.join(gtfs, "shapes.txt"), "in_memory", "shapes_tbl"
        ).getOutput(0)
        shapes_xy = make_xy_event_layer(
            shapes_tbl, "shape_pt_lon", "shape_pt_lat", "shapes_xy", sr_wgs84
        )
        shapes_line = "in_memory/shapes_line"
        points_to_line(shapes_xy, shapes_line, "shape_id", "shape_pt_sequence")
        routes_raw = os.path.join(out, "routes_raw.shp")
        project_feature(shapes_line, routes_raw, sr_target)
        if ROUTE_FILTER:
            routes_fc = apply_route_filter(
                routes_raw, GTFS_FOLDER, route_field, per_shape, ROUTE_FILTER
            )
            routes_shp = os.path.join(out, "routes.shp")
            arcpy.CopyFeatures_management(routes_fc, routes_shp)
            arcpy.AddMessage(f"   ✔ Filtered routes → {routes_shp}")
        else:
            routes_fc = routes_raw
            arcpy.AddMessage(f"   ✔ All routes → {routes_fc}")

        # STOPS
        arcpy.AddMessage("\n=== Stops ===")
        stops_tbl = arcpy.TableToTable_conversion(
            os.path.join(gtfs, "stops.txt"), "in_memory", "stops_tbl"
        ).getOutput(0)
        stops_xy = make_xy_event_layer(stops_tbl, "stop_lon", "stop_lat", "stops_xy", sr_wgs84)
        stops_wgs = arcpy.CopyFeatures_management(stops_xy, "in_memory/stops_wgs").getOutput(0)
        full_stops = os.path.join(out, "stops_full.shp")
        project_feature(stops_wgs, full_stops, sr_target)
        if ROUTE_FILTER:
            arcpy.MakeFeatureLayer_management(full_stops, "stops_lyr")
            arcpy.SelectLayerByLocation_management("stops_lyr", "INTERSECT", routes_fc)
            filtered_stops = os.path.join(out, "stops.shp")
            arcpy.CopyFeatures_management("stops_lyr", filtered_stops)
            arcpy.AddMessage(f"   ✔ Filtered stops → {filtered_stops}")
            stops_fc = filtered_stops
        else:
            arcpy.AddMessage(f"   ✔ All stops → {full_stops}")
            stops_fc = full_stops

        # SEGMENTS
        arcpy.AddMessage("\n=== Inter-stop Segments ===")
        routes_m = "in_memory/routes_m"
        create_routes_with_m(routes_fc, route_field, routes_m)
        events_tbl = "in_memory/stops_events"
        locate_stops_along_routes(stops_fc, routes_m, route_field, events_tbl)
        segments_fc = os.path.join(out, "segments.shp")
        build_segments(routes_m, events_tbl, route_field, segments_fc, sr_target)
        arcpy.AddMessage(f"   ✔ Segments → {segments_fc}")

        # SHORT SPACING
        log_path = Path(OUTPUT_FOLDER) / SPACING_LOG_FILE
        flag_short_spacing(segments_fc, MIN_SPACING_FT, log_path)

        arcpy.AddMessage("\nAll outputs created successfully!")
    except Exception:
        arcpy.AddError("### Script failed ###")
        arcpy.AddError(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
