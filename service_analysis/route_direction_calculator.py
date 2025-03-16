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

# --------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------
GTFS_FOLDER = r"/path/to/your/gtfs_folder"
OUTPUT_FOLDER = r"/path/to/your/output_folder"

# Optional: Include only these route_short_name values
ROUTE_FILTER_IN = []
# Routes to exclude
ROUTE_FILTER_OUT = ['9999A', '9999B', '9999C']

# NAD83 / Maryland (meters)
PROJECTED_CRS = 'EPSG:26985'
LOOP_THRESHOLD = 200

# Toggles
EXPORT_XLSX = True
EXPORT_JPEG = True

# NEW CODE: Boolean to analyze only the most common (modal) shape
ANALYZE_ONLY_DOMINANT_SHAPE = True
# --------------------------------------------------------------------------------

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

def plot_route_shape(gdf_shape, route, direction, output_path):
    """
    Plot the given shape geometry and save as a .jpeg.
    Highlights the start and end of the route shape, and includes a simple
    'N' arrow for orientation in the top-left corner of the map.
    """

    if not isinstance(gdf_shape, gpd.GeoDataFrame):
        gdf_shape = gpd.GeoDataFrame([gdf_shape], columns=gdf_shape.index, crs="EPSG:4326")

    gdf_shape_4326 = gdf_shape.to_crs(epsg=4326)
    shape_line = gdf_shape_4326.iloc[0].geometry
    start_lon, start_lat = shape_line.coords[0]
    end_lon, end_lat = shape_line.coords[-1]

    fig, ax = plt.subplots(figsize=(6, 6))
    gdf_shape_4326.plot(ax=ax, color='blue', linewidth=2, alpha=0.8)

    ax.plot(start_lon, start_lat, 'go', label="Start")
    ax.plot(end_lon, end_lat, 'ro', label="End")

    ax.text(
        0.05, 0.95, 'N',
        transform=ax.transAxes,
        fontsize=14,
        fontweight='bold',
        va='top',
        ha='center',
        rotation=0
    )
    ax.annotate(
        '',
        xy=(0.05, 0.94), xytext=(0.05, 0.90),
        xycoords='axes fraction',
        arrowprops=dict(facecolor='black', width=1, headwidth=6),
    )

    ax.set_title(f"Route {route}, Direction {direction} - Most Common Shape")
    ax.legend()
    plt.axis('equal')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
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

    # Create lines from shapes
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

    # Classify each shape's direction
    directions = []
    for i, row in gdf_shapes.iterrows():
        shape_id = row['shape_id']
        geom_4326 = row['geometry']
        geom_proj = gdf_shapes_proj.loc[i, 'geometry']
        directions.append((shape_id, classify_direction(geom_4326, geom_proj)))

    direction_df = pd.DataFrame(directions, columns=['shape_id', 'shape_direction'])
    trips_merged = trips_merged.merge(direction_df, on='shape_id')

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
                columns={'stop_id': 'first_stop_id', 'stop_name': 'first_stop_name'}
            ),
            on='first_stop_id'
        )
        .merge(
            stops[['stop_id', 'stop_name']].rename(
                columns={'stop_id': 'last_stop_id', 'stop_name': 'last_stop_name'}
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
    dominant_shapes['is_dominant'] = True

    final_data = final_data.merge(
        dominant_shapes[['route_short_name', 'direction_id', 'shape_id','is_dominant']],
        on=['route_short_name', 'direction_id', 'shape_id'],
        how='left'
    )

    # NEW CODE: If set to True, limit everything to only the dominant (modal) shape.
    if ANALYZE_ONLY_DOMINANT_SHAPE:
        final_data = final_data[final_data['is_dominant'] == True]

    # Rebuild the summary based on the final_data, since we may have filtered
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

    # -------------------------------------------------------------------------
    # Export to Excel if EXPORT_XLSX is True
    # -------------------------------------------------------------------------
    if EXPORT_XLSX:
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

    # -------------------------------------------------------------------------
    # Export JPEGs if EXPORT_JPEG is True (dominant shape per route/direction)
    # -------------------------------------------------------------------------
    if EXPORT_JPEG:
        # Merge to get geometry
        # Because final_data is now possibly filtered by is_dominant,
        # we should figure out which shape_ids remain.
        remaining_shape_ids = summary['shape_id'].unique().tolist()
        # Filter only those shapes that appear in final_data
        gdf_shapes_dominant = gdf_shapes[gdf_shapes['shape_id'].isin(remaining_shape_ids)]

        # We also need the route_short_name and direction_id for labeling
        # so let's build a small lookup from final_data or summary:
        shape_info_lookup = summary[['shape_id', 'route_short_name', 'direction_id']].drop_duplicates()

        # For each shape, plot the geometry
        for shape_id in remaining_shape_ids:
            row = shape_info_lookup[shape_info_lookup['shape_id'] == shape_id].iloc[0]
            route = row['route_short_name']
            direction = row['direction_id']

            route_shape_gdf = gdf_shapes_dominant[gdf_shapes_dominant['shape_id'] == shape_id]

            output_path = os.path.join(
                OUTPUT_FOLDER,
                f"Route_{route}_Dir_{direction}_DominantShape.jpeg"
            )

            plot_route_shape(
                gdf_shape=route_shape_gdf,
                route=route,
                direction=direction,
                output_path=output_path
            )

    # -------------------------------------------------------------------------
    # Flag suspicious data based on shape_direction vs direction_id
    # -------------------------------------------------------------------------
    summary_simplified = summary[[
        'route_short_name',
        'direction_id',
        'shape_direction'
    ]].drop_duplicates()

    flags = []

    # 1) For each route, check how many direction_ids and shape_directions exist
    route_groups = summary_simplified.groupby('route_short_name')
    for route_name, grp in route_groups:
        unique_dirs = grp['direction_id'].unique()
        unique_shape_dirs = grp['shape_direction'].unique()

        if len(unique_dirs) > 1 and len(unique_shape_dirs) == 1:
            flags.append({
                'route_short_name': route_name,
                'direction_id': list(unique_dirs),
                'problem': f"Multiple direction_ids {list(unique_dirs)} but only one shape_direction '{unique_shape_dirs[0]}'"
            })

    # 2) Within the same (route_short_name, direction_id), multiple cardinal directions
    def is_cardinal_direction(d):
        return d in ("NB", "SB", "EB", "WB")

    rd_group = summary_simplified.groupby(['route_short_name', 'direction_id'])
    for (rname, did), subgrp in rd_group:
        cardinal_dirs = [d for d in subgrp['shape_direction'] if is_cardinal_direction(d)]
        if len(set(cardinal_dirs)) > 1:
            flags.append({
                'route_short_name': rname,
                'direction_id': did,
                'problem': f"Conflicting cardinal directions {list(set(cardinal_dirs))}"
            })

    flagged_df = pd.DataFrame(flags)

    if not flagged_df.empty:
        flagged_df_path = os.path.join(OUTPUT_FOLDER, "Suspicious_RouteDirections.xlsx")
        flagged_df.to_excel(flagged_df_path, index=False)
        print(f"[INFO] Suspicious combinations flagged. See {flagged_df_path}.")
    else:
        print("[INFO] No suspicious route/direction combos found.")

    print("Script execution completed.")

if __name__ == '__main__':
    main()
