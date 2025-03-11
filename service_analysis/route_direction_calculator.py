"""
GTFS Direction Classification Script

This script analyzes General Transit Feed Specification (GTFS) data to classify transit routes according to their geographic direction (Northbound, Southbound, Eastbound, Westbound) or as loops (Clockwise, Counter-Clockwise, or general loops).

Direction classification is based on comparing the start and end points of each route's shape geometry:

- If the start and end points are within a specified threshold distance (default: 200 meters), the shape is considered a loop. The direction of loop shapes is determined by calculating the polygon's signed area to identify clockwise or counter-clockwise orientation.
- For non-loop shapes, the script evaluates the greater absolute change in latitude versus longitude to assign a cardinal direction (NB, SB, EB, WB).

Inputs:
    - Standard GTFS files: routes.txt, trips.txt, stop_times.txt, shapes.txt, stops.txt

Outputs:
    - Excel summary file (Directions_Summary.xlsx) detailing the count of trips per route, direction, and shape.
    - Individual Excel files per route and direction containing departure times and stop information.

Configurations allow filtering routes for targeted analysis, and the script utilizes EPSG:26985 (NAD83 / Maryland) for spatial calculations by default.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString

# Configuration
GTFS_FOLDER = (
    r"/path/to/your/gtfs_folder"
)
OUTPUT_FOLDER = (
    r"/path/to/your/output_folder"
)

# Optional: Include only these route_short_name values
ROUTE_FILTER_IN = []
# Routes to exclude
ROUTE_FILTER_OUT = ['9999A', '9999B', '9999C']

# NAD83 / Maryland (meters)
PROJECTED_CRS = 'EPSG:26985'
LOOP_THRESHOLD = 200


def classify_direction(
    line_4326: LineString,
    line_projected: LineString,
    loop_threshold: int = LOOP_THRESHOLD
) -> str:
    """
    Classify the direction of a GTFS shape between its start and end points.

    If the start/end points are close (less than loop_threshold), the shape
    is treated as a loop and classified based on the polygon's signed area.
    Otherwise, we compare latitude and longitude changes to decide NB, SB,
    EB, or WB.
    """
    start_lon, start_lat = line_4326.coords[0]
    end_lon, end_lat = line_4326.coords[-1]

    start_p = line_projected.coords[0]
    end_p = line_projected.coords[-1]

    dist_start_end = (
        ((start_p[0] - end_p[0]) ** 2) +
        ((start_p[1] - end_p[1]) ** 2)
    ) ** 0.5

    if dist_start_end < loop_threshold:
        coords = list(line_projected.coords)
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        area = sum(
            (x1 * y2 - x2 * y1)
            for (x1, y1), (x2, y2) in zip(coords, coords[1:])
        )
        if area > 0:
            return "CCW"
        elif area < 0:
            return "CW"
        return "LOOP"
    else:
        lat_diff = end_lat - start_lat
        lon_diff = end_lon - start_lon
        return (
            "NB" if abs(lat_diff) > abs(lon_diff) and lat_diff > 0
            else "SB" if abs(lat_diff) > abs(lon_diff)
            else "EB" if lon_diff > 0
            else "WB"
        )


def main():
    """
    Main function to classify directions for GTFS data and export the results.
    """
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Read GTFS files
    routes = pd.read_csv(os.path.join(GTFS_FOLDER, 'routes.txt'))
    trips = pd.read_csv(os.path.join(GTFS_FOLDER, 'trips.txt'))
    stop_times = pd.read_csv(os.path.join(GTFS_FOLDER, 'stop_times.txt'))
    shapes = pd.read_csv(os.path.join(GTFS_FOLDER, 'shapes.txt'))
    stops = pd.read_csv(os.path.join(GTFS_FOLDER, 'stops.txt'))

    # Filter routes if needed
    if ROUTE_FILTER_IN:
        routes_filtered = routes[routes['route_short_name'].isin(ROUTE_FILTER_IN)]
    else:
        routes_filtered = routes.copy()

    if ROUTE_FILTER_OUT:
        routes_filtered = routes_filtered[
            ~routes_filtered['route_short_name'].isin(ROUTE_FILTER_OUT)
        ]

    # Filter trips based on filtered routes
    trips_filtered = trips[
        trips['route_id'].isin(routes_filtered['route_id'])
    ]

    trips_merged = trips_filtered.merge(
        routes_filtered[['route_id', 'route_short_name', 'route_long_name']],
        on='route_id'
    )

    shapes_grouped = shapes.sort_values(
        ['shape_id', 'shape_pt_sequence']
    ).groupby('shape_id')

    lines = [
        (sid, LineString(zip(g['shape_pt_lon'], g['shape_pt_lat'])))
        for sid, g in shapes_grouped
    ]

    gdf_shapes = gpd.GeoDataFrame(
        lines,
        columns=['shape_id', 'geometry'],
        crs="EPSG:4326"
    )
    gdf_shapes_proj = gdf_shapes.to_crs(PROJECTED_CRS)

    directions = [
        (
            row['shape_id'],
            classify_direction(row['geometry'], gdf_shapes_proj.loc[i, 'geometry'])
        )
        for i, row in gdf_shapes.iterrows()
    ]

    trips_merged = trips_merged.merge(
        pd.DataFrame(directions, columns=['shape_id', 'shape_direction']),
        on='shape_id'
    )

    # Identify first and last stops
    stop_times_sorted = stop_times.sort_values(['trip_id', 'stop_sequence'])

    first_stops = (
        stop_times_sorted
        .groupby('trip_id')
        .first()
        .reset_index()
        .rename(columns={'stop_id': 'first_stop_id'})
    )
    last_stops = (
        stop_times_sorted
        .groupby('trip_id')
        .last()
        .reset_index()
        .rename(columns={'stop_id': 'last_stop_id'})
    )

    trips_merged = (
        trips_merged
        .merge(first_stops[['trip_id', 'first_stop_id']], on='trip_id')
        .merge(last_stops[['trip_id', 'last_stop_id']], on='trip_id')
    )

    trips_merged = (
        trips_merged
        .merge(
            stops[['stop_id', 'stop_name']].rename(
                columns={'stop_id': 'first_stop_id',
                         'stop_name': 'first_stop_name'}
            ),
            on='first_stop_id'
        )
        .merge(
            stops[['stop_id', 'stop_name']].rename(
                columns={'stop_id': 'last_stop_id',
                         'stop_name': 'last_stop_name'}
            ),
            on='last_stop_id'
        )
    )

    first_departures = stop_times[
        stop_times['stop_sequence'] == 1
    ][['trip_id', 'departure_time']]

    final_data = trips_merged.merge(first_departures, on='trip_id')

    # Determine dominant shapes based on trip counts
    shape_counts = (
        final_data
        .groupby(['route_short_name', 'direction_id', 'shape_id'])
        .size()
        .reset_index(name='trip_count')
    )
    idx_max = (
        shape_counts
        .groupby(['route_short_name', 'direction_id'])['trip_count']
        .idxmax()
    )
    dominant_shapes = shape_counts.loc[idx_max]

    final_data = final_data.merge(
        dominant_shapes[['route_short_name', 'direction_id', 'shape_id']],
        on=['route_short_name', 'direction_id', 'shape_id']
    )

    summary = (
        final_data
        .groupby([
            'route_short_name',
            'direction_id',
            'shape_id',
            'shape_direction',
            'first_stop_name',
            'last_stop_name'
        ], dropna=False)
        .size()
        .reset_index(name='trip_count')
    )

    summary.to_excel(
        os.path.join(OUTPUT_FOLDER, "Directions_Summary.xlsx"),
        index=False
    )

    # Export individual route-direction files
    grouped_fd = final_data.groupby(['route_short_name', 'direction_id'])
    for (route, direction), group in grouped_fd:
        group.sort_values('departure_time').to_excel(
            os.path.join(
                OUTPUT_FOLDER,
                f"Route_{route}_Dir_{direction}_departures.xlsx"
            ),
            index=False
        )

    print("Script execution completed.")


if __name__ == '__main__':
    main()
