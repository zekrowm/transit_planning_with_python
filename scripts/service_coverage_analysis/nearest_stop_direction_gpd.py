"""Identifies nearby GTFS stops and associated routes for given locations or stop codes.

Supports two input modes:
- 'location': Uses point shapefile or manual lat/lon coordinates.
- 'stop_code': Uses a list of GTFS stop codes.

Applies optional route filters and returns closest routes and directions per input.
Outputs results to a CSV file with customizable location metadata.

Typical use:
    - Evaluate transit accessibility near proposed schools, facilities, or sites.
    - Summarize GTFS route presence at selected stop codes.

Inputs:
    - GTFS files: stops.txt, stop_times.txt, trips.txt, routes.txt.
    - User-defined configuration: input mode, buffer radius, filters, paths.

Outputs:
    - CSV listing stops or locations with nearby GTFS routes and directions.
"""

from __future__ import annotations

import os
import sys

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# =============================================================================
# CONFIGURATION
# =============================================================================

GTFS_FOLDER = r"Path\To\Your\GTFS\Folder"
OUTPUT_FOLDER = r"Path\To\Your\Output\Folder"

INPUT_MODE = "location"  # "location" | "stop_code"
LOCATION_SOURCE = "shapefile"  # "manual"   | "shapefile"
POINT_SHAPEFILE = r"Path\To\Your\Points.shp"
POINT_NAME_FIELD = "OBJECTID"  # column copied to 'Location'

# extra point-layer attributes you want in the CSV
LOCATION_EXTRA_FIELDS = ["SCHOOL_NAM", "SCHOOL_TYP", "WEB_ADDRES"]  # Edit

# Route filters
ROUTE_FILTER_IN: list[str] = []  # keep only these (leave empty for no “in” filter)
ROUTE_FILTER_OUT: list[str] = ["9999A", "9999B", "9999C"]  # always drop these

# location-mode specifics
MANUAL_LOCATIONS = [
    {"name": "Braddock", "latitude": 38.813545, "longitude": -77.053864},
    {"name": "Crystal City", "latitude": 38.85835, "longitude": -77.051232},
]
BUFFER_DISTANCE = 0.25  # numeric value
BUFFER_UNIT = "miles"  # 'miles' | 'feet'
PROJECTED_CRS = "EPSG:2232"  # NAD83 / DC state plane (ft)

# stop_code-mode specifics
STOP_CODE_FILTER: list[str] = []

OUTPUT_FILE_NAME = "proximity_results.csv"

# =============================================================================
# FUNCTIONS
# =============================================================================


def _check_gtfs(path: str) -> None:
    for fn in ("stops.txt", "stop_times.txt", "trips.txt", "routes.txt"):
        fp = os.path.join(path, fn)
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Required GTFS file missing → {fp!s}")


def _load_gtfs(path: str) -> dict[str, pd.DataFrame]:
    return {
        fn.split(".")[0]: pd.read_csv(os.path.join(path, fn), dtype=str)
        for fn in ("stops.txt", "stop_times.txt", "trips.txt", "routes.txt")
    }


def _load_locations(
    source: str,
    *,
    manual_list: list[dict] | None = None,
    shp_path: str | None = None,
    name_field: str = "name",
) -> gpd.GeoDataFrame:
    if source == "manual":
        if not manual_list:
            raise ValueError("manual_list must be provided when LOCATION_SOURCE='manual'")
        gdf = gpd.GeoDataFrame(
            manual_list,
            geometry=[Point(d["longitude"], d["latitude"]) for d in manual_list],
            crs="EPSG:4326",
        )
    elif source == "shapefile":
        if not shp_path:
            raise ValueError("shp_path must be provided when LOCATION_SOURCE='shapefile'")
        gdf = gpd.read_file(shp_path)
        gdf = gdf[gdf.geometry.type.isin({"Point", "MultiPoint"})]
        gdf = (
            gdf.set_crs("EPSG:4326", inplace=False) if gdf.crs is None else gdf.to_crs("EPSG:4326")
        )
        if name_field in gdf.columns:
            gdf = gdf.rename(columns={name_field: "name"})
        if "name" not in gdf.columns:
            gdf["name"] = [f"loc_{i}" for i in range(len(gdf))]
    else:
        raise ValueError("LOCATION_SOURCE must be 'manual' or 'shapefile'")
    return gdf


def _stops_to_gdf(stops: pd.DataFrame) -> gpd.GeoDataFrame:
    stops = stops.assign(
        stop_lat=stops.stop_lat.astype(float), stop_lon=stops.stop_lon.astype(float)
    )
    return gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops.stop_lon, stops.stop_lat),
        crs="EPSG:4326",
    )


def _distance_ft(value: float, unit: str) -> float:
    return value * 5280 if unit.lower() == "miles" else value


def _apply_route_filters(df: pd.DataFrame) -> pd.DataFrame:
    if ROUTE_FILTER_IN:
        df = df[df.route_short_name.isin(ROUTE_FILTER_IN)]
    if ROUTE_FILTER_OUT:
        df = df[~df.route_short_name.isin(ROUTE_FILTER_OUT)]
    return df


def _nearby_routes(
    gdf_locations: gpd.GeoDataFrame,
    gdf_stops: gpd.GeoDataFrame,
    st_trips_routes: pd.DataFrame,
    buf_ft: float,
    extra_cols: list[str],
) -> list[dict]:
    results: list[dict] = []

    for _, loc in gdf_locations.iterrows():
        base = {
            "Location": loc["name"],
            **{c: ("" if pd.isna(loc[c]) else str(loc[c])) for c in extra_cols},
        }

        # buffer + spatial filter
        nearby = gdf_stops[gdf_stops.geometry.within(loc.geometry.buffer(buf_ft))]
        if nearby.empty:
            results.append({**base, "Routes": "No routes", "Stops": "No stops"})
            continue

        stop_ids = nearby.stop_id.unique()
        df = st_trips_routes[st_trips_routes.stop_id.isin(stop_ids)]
        if df.empty:
            results.append({**base, "Routes": "No routes", "Stops": "No stops"})
            continue

        merged = (
            nearby[["stop_id", "geometry"]]
            .merge(df[["stop_id", "route_short_name", "direction_id"]], on="stop_id")
            .drop_duplicates()
        )
        merged["dist"] = merged.geometry.distance(loc.geometry)

        nearest = merged.groupby(["route_short_name", "direction_id"], as_index=False).apply(
            lambda x: x.loc[x.dist.idxmin()]
        )

        pair_set = {(r, d) for r, d in zip(nearest.route_short_name, nearest.direction_id)}
        routes = ", ".join(sorted(f"{r} (dir {d})" for r, d in pair_set))
        stops = ", ".join(sorted(nearest.stop_id.astype(str).unique()))
        results.append({**base, "Routes": routes, "Stops": stops})

    return results


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Runs the main analysis based on global CONFIGURATION variables.

    Performs either a location-based proximity search or a stop-code lookup
    and exports the results to a CSV.
    """
    try:
        _check_gtfs(GTFS_FOLDER)
        gtfs = _load_gtfs(GTFS_FOLDER)

        st_trips_routes = (
            gtfs["stop_times"]
            .merge(gtfs["trips"], on="trip_id", how="inner")
            .merge(gtfs["routes"], on="route_id", how="inner")
            .pipe(_apply_route_filters)
        )

        if st_trips_routes.empty:
            print("Route filters removed every route – nothing to analyse.")
            return

        if INPUT_MODE == "location":
            gdf_loc = _load_locations(
                LOCATION_SOURCE,
                manual_list=MANUAL_LOCATIONS,
                shp_path=POINT_SHAPEFILE,
                name_field=POINT_NAME_FIELD,
            )
            gdf_stops = _stops_to_gdf(gtfs["stops"]).to_crs(PROJECTED_CRS)
            rows = _nearby_routes(
                gdf_loc.to_crs(PROJECTED_CRS),
                gdf_stops,
                st_trips_routes,
                _distance_ft(BUFFER_DISTANCE, BUFFER_UNIT),
                LOCATION_EXTRA_FIELDS,
            )

        elif INPUT_MODE == "stop_code":
            if "stop_code" not in gtfs["stops"].columns:
                print("stops.txt lacks 'stop_code' – cannot run stop_code mode.")
                return
            stop_ids = gtfs["stops"][gtfs["stops"].stop_code.isin(STOP_CODE_FILTER)].stop_id
            if stop_ids.empty:
                print("No stops matched STOP_CODE_FILTER.")
                return

            df = (
                gtfs["stop_times"][gtfs["stop_times"].stop_id.isin(stop_ids)]
                .merge(gtfs["trips"], on="trip_id", how="inner")
                .merge(gtfs["routes"], on="route_id", how="inner")
                .pipe(_apply_route_filters)
            )

            rows = []
            for sid, grp in df.groupby("stop_id"):
                route_pairs = sorted(
                    {f"{r} (dir {d})" for r, d in zip(grp.route_short_name, grp.direction_id)}
                )
                rows.append({"Stop_ID": sid, "Routes": "; ".join(route_pairs)})

        else:
            raise ValueError("INPUT_MODE must be 'location' or 'stop_code'.")

        if not rows:
            print("No results.")
            return

        out_csv = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE_NAME)
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"✔  Results written → {out_csv}")

    except Exception as exc:  # pylint: disable=broad-except
        print(f"✖  {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
