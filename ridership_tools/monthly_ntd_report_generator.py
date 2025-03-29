"""
Transit Ridership Analysis and Visualization Module

This script processes and analyzes transit ridership data sourced from monthly Excel files,
organized by configured periods. It standardizes and classifies route data by service type
and corridor, computes key performance metrics (e.g., passengers per hour, per trip, per mile),
and generates aggregated summaries at both service-type and route levels.

Main features include:
- Automated data loading, cleaning, and preprocessing from Excel sheets.
- Route classification by configurable service types and corridors.
- Calculation of derived transit performance metrics and aggregations.
- Exporting of comprehensive and monthly aggregated summaries to Excel.
- Generation of time-series plots for selected performance metrics.

Configurations for file paths, service categories, and plotting options are customizable via
dictionaries defined at the start of the module.

Output:
- Excel files summarizing detailed, aggregated, and route-level ridership statistics.
- Plots visualizing transit performance metrics over time.
"""
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

###############################################################################
#                             CONFIGURATION                                   #
###############################################################################

CONFIG = {
    'periods': {
        'Jul-2024': {
            'file_path': r'\\Your\File\Path\JULY 2024 NTD RIDERSHIP BY ROUTE.XLSX',
            'sheet_name': 'Temporary_Query_N'
        },
        'Aug-2024': {
            'file_path': r'\\Your\File\Path\AUGUST 2024  NTD RIDERSHIP REPORT BY ROUTE.XLSX',
            'sheet_name': 'Temporary_Query_N'
        },
        'Sep-2024': {
            'file_path': r'\\Your\File\Path\SEPTEMBER 2024 NTD RIDERSHIP BY ROUTE.XLSX',
            'sheet_name': 'Sep.2024 Finals'
        },
        'Oct-2024': {
            'file_path': r'\\Your\File\Path\NTD RIDERSHIP BY ROUTE _ OCTOBER _2024.XLSX',
            'sheet_name': 'Temporary_Query_N'
        },
        'Nov-2024': {
            'file_path': r'\\Your\File\Path\NTD RIDERSHIP BY ROUTE-NOVEMBER 2024.xlsx',
            'sheet_name': 'Temporary_Query_N'
        },
        'Dec-2024': {
            'file_path': r'\\Your\File\Path\NTD RIDERSHIP BY MONTH_DECEMBER 2024.XLSX',
            'sheet_name': 'Dec. 2024'
        },
        'Jan-2025': {
            'file_path': r'\\Your\File\Path\NTD_files_FY25\NTD RIDERSHIP BY MONTH-JANUARY 2025.xlsx',
            'sheet_name': 'Jan. 2025'
        },
        'Feb-2025': {
            'file_path': r'\\Your\File\Path\NTD RIDERSHIP BY MONTH-FEBRUARY 2025.xlsx',
            'sheet_name': 'Feb. 2025'
        },
        'Mar-2025': {
            'file_path': r'\\Your\File\Path\MARCH 2025 NTD RIDERSHIP BY MONTH.xlsx',
            'sheet_name': 'Mar. 2025'
        },
        'Apr-2025': {
            'file_path': r'\\Your\File\Path\APRIL 2025 NTD RIDERSHIP BY MONTH.xlsx',
            'sheet_name': 'Apr. 2025'
        },
        'May-2025': {
            'file_path': r'\\Your\File\Path\MAY 2025 NTD RIDERSHIP BY MONTH.xlsx',
            'sheet_name': 'May. 2025'
        },
        'Jun-2025': {
            'file_path': r'\\Your\File\Path\JUNE 2025 NTD RIDERSHIP BY MONTH.xlsx',
            'sheet_name': 'Jun. 2025'
        }
    },
    'ordered_periods': [
        'Jul-2024', 'Aug-2024', 'Sep-2024', 'Oct-2024',
        'Nov-2024', 'Dec-2024', 'Jan-2025', 'Feb-2025',
        'Mar-2025', 'Apr-2025', 'May-2025', 'Jun-2025'
    ],

    'SERVICE_TYPE_DICT': {
        'local': [
            "101", "201", "301"
        ],
        'express': [
            "102", "202", "302"
        ]
    },

    'CORRIDOR_DICT': {
        'corridor_one': ["101","102"],
        'corridor_two': ["201","202"],
        'corridor_three': ["301","302"]
    },

    'converters': {
        'MTH_BOARD':      lambda x: float(str(x).replace(',', '')) if x else None,
        'MTH_REV_HOURS':  lambda x: float(str(x).replace(',', '')) if x else None,
        'MTH_PASS_MILES': lambda x: float(str(x).replace(',', '')) if x else None,
        'ACTUAL_TRIPS':   lambda x: float(str(x).replace(',', '')) if x else None,
        'DAYS':           lambda x: float(str(x).replace(',', '')) if x else None,
        'REV_MILES':      lambda x: float(str(x).replace(',', '')) if x else None
    },

    'SERVICE_PERIODS': ['Weekday', 'Saturday', 'Sunday'],

    'output_dir': r'\\Path\to\Your\Output_Folder'
}

# ------------------ PLOT CONFIGURATION BOOLEANS -----------------------------
# Set any of these to False if you do NOT want that particular plot generated.
PLOT_CONFIG = {
    'plot_total_ridership': True,
    'plot_weekday_avg':     True,
    'plot_saturday_avg':    False,
    'plot_sunday_avg':      False,
    'plot_revenue_hours':   False,
    'plot_trips':           False,
    'plot_revenue_miles':   False,
    'plot_pph':             True,  # passengers per hour
    'plot_ppt':             True,  # passengers per trip
    'plot_ppm':             True   # passengers per mile
}

# Matplotlib settings for plot style
PLOT_STYLE = {
    'figsize': (9, 5),
    'marker': 'o',
    'linestyle': '-',
    'rotation': 45,
    'grid': True
}


###############################################################################
#                           DATA LOADING / CLEANING                           #
###############################################################################

def read_excel_data(config: dict) -> dict:
    """
    Read each period’s file_path + sheet_name into a DataFrame,
    filter by SERVICE_PERIODS, and store in a dict keyed by period name.
    """
    converters = config['converters']
    sp_filter  = config['SERVICE_PERIODS']
    data_dict  = {}

    for period in config['ordered_periods']:
        info      = config['periods'][period]
        file_path = info['file_path']
        sheet     = info['sheet_name']

        df = pd.read_excel(file_path, sheet_name=sheet, converters=converters)

        # Drop rows that lack crucial data
        df.dropna(subset=['ROUTE_NAME','MTH_BOARD'], inplace=True)
        df = df[df['MTH_BOARD'] != 0]

        # Filter by service period (Weekday, Saturday, Sunday)
        df = df[df['SERVICE_PERIOD'].isin(sp_filter)].copy()

        # Standardize route name
        df['ROUTE_NAME'] = (
            df['ROUTE_NAME']
            .astype(str)
            .str.strip()
            .str.upper()
            .str.replace(' ', '', regex=False)
            .apply(lambda x: re.sub(r'\.0$', '', x))
        )

        data_dict[period] = df

    return data_dict


###############################################################################
#                    SERVICE TYPE & CORRIDOR CLASSIFICATION                   #
###############################################################################

def classify_route(route_name: str, cfg: dict) -> str:
    """
    Returns the first service_type in which this route is found.
    If no match, 'unknown'. If the dictionary is empty, 'SYSTEMWIDE'.
    """
    st_dict = cfg['SERVICE_TYPE_DICT']
    if not st_dict:
        return 'SYSTEMWIDE'
    for service_type, route_list in st_dict.items():
        if route_name in route_list:
            return service_type
    return 'unknown'


def classify_corridor(route_name: str, cfg: dict) -> list:
    """
    A route can belong to multiple corridors if it appears in more than one list.
    Returns that list, or ['other'] if none match.
    """
    corridor_dict = cfg['CORRIDOR_DICT']
    corridors = []
    for c_name, r_list in corridor_dict.items():
        if route_name in r_list:
            corridors.append(c_name)
    return corridors if corridors else ['other']


###############################################################################
#                   ADDING DERIVED COLUMNS & AGGREGATIONS                     #
###############################################################################

def calculate_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates:
      - TOTAL_TRIPS
      - BOARDS_PER_HOUR (MTH_BOARD / MTH_REV_HOURS)
      - PASSENGERS_PER_TRIP (MTH_BOARD / TOTAL_TRIPS)
      - MTH_REV_MILES (REV_MILES * DAYS)
      - PASSENGERS_PER_MILE (MTH_BOARD / MTH_REV_MILES)
    """
    df = df.copy()
    df['TOTAL_TRIPS'] = df['ACTUAL_TRIPS'] * df['DAYS']

    df['BOARDS_PER_HOUR'] = df.apply(
        lambda row: row['MTH_BOARD']/row['MTH_REV_HOURS'] if row['MTH_REV_HOURS'] else None,
        axis=1
    )
    df['PASSENGERS_PER_TRIP'] = df.apply(
        lambda row: row['MTH_BOARD']/row['TOTAL_TRIPS'] if row['TOTAL_TRIPS'] else None,
        axis=1
    )

    df['MTH_REV_MILES'] = df['REV_MILES'] * df['DAYS']
    df['PASSENGERS_PER_MILE'] = df.apply(
        lambda row: row['MTH_BOARD']/row['MTH_REV_MILES'] if row['MTH_REV_MILES'] else None,
        axis=1
    )

    # Rounding
    df['BOARDS_PER_HOUR']     = df['BOARDS_PER_HOUR'].round(1)
    df['PASSENGERS_PER_TRIP'] = df['PASSENGERS_PER_TRIP'].round(1)
    df['PASSENGERS_PER_MILE'] = df['PASSENGERS_PER_MILE'].round(3)
    df['TOTAL_TRIPS']         = df['TOTAL_TRIPS'].round(1)

    return df


def aggregate_by_service_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize at the service_type level:
      - sum of Boardings, Hours, Miles, Trips
      - then re-compute boards/hour, passengers/trip, etc.
      - add a TOTAL row
    """
    grouped = df.groupby('service_type').agg({
        'MTH_BOARD':       'sum',
        'MTH_REV_HOURS':   'sum',
        'MTH_PASS_MILES':  'sum',
        'MTH_REV_MILES':   'sum',
        'TOTAL_TRIPS':     'sum'
    }).reset_index()

    # Derived columns for the grouped data
    grouped['BOARDS_PER_HOUR'] = (grouped['MTH_BOARD'] / grouped['MTH_REV_HOURS']).round(1)
    grouped['PASSENGERS_PER_TRIP'] = (grouped['MTH_BOARD'] / grouped['TOTAL_TRIPS']).round(1)
    grouped['PASSENGERS_PER_MILE'] = (grouped['MTH_BOARD'] / grouped['MTH_REV_MILES']).round(3)

    # Build a TOTAL row across all service types
    sums = grouped[['MTH_BOARD','MTH_REV_HOURS','MTH_PASS_MILES','MTH_REV_MILES','TOTAL_TRIPS']].sum()
    total_row = {
        'service_type': 'TOTAL',
        'MTH_BOARD': sums['MTH_BOARD'],
        'MTH_REV_HOURS': sums['MTH_REV_HOURS'],
        'MTH_PASS_MILES': sums['MTH_PASS_MILES'],
        'MTH_REV_MILES': sums['MTH_REV_MILES'],
        'TOTAL_TRIPS': sums['TOTAL_TRIPS'],
    }

    # Re-compute ratios
    if sums['MTH_REV_HOURS']:
        total_row['BOARDS_PER_HOUR'] = round(sums['MTH_BOARD'] / sums['MTH_REV_HOURS'], 1)
    else:
        total_row['BOARDS_PER_HOUR'] = None

    if sums['TOTAL_TRIPS']:
        total_row['PASSENGERS_PER_TRIP'] = round(sums['MTH_BOARD'] / sums['TOTAL_TRIPS'], 1)
    else:
        total_row['PASSENGERS_PER_TRIP'] = None

    if sums['MTH_REV_MILES']:
        total_row['PASSENGERS_PER_MILE'] = round(sums['MTH_BOARD'] / sums['MTH_REV_MILES'], 3)
    else:
        total_row['PASSENGERS_PER_MILE'] = None

    grouped = grouped.append(total_row, ignore_index=True)
    return grouped


###############################################################################
#                      ROUTE-LEVEL SUMMARY WITH SUBTOTALS                     #
###############################################################################

def route_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize each route, grouped by service_type + route_name, then add
    a 'SUBTOTAL' row for each service_type, and a 'TOTAL' row for all.
    """
    df_route = df.groupby(['service_type','ROUTE_NAME'], as_index=False).agg({
        'MTH_BOARD':'sum',
        'MTH_REV_HOURS':'sum',
        'MTH_PASS_MILES':'sum',
        'MTH_REV_MILES':'sum',
        'TOTAL_TRIPS':'sum'
    })

    # Derived columns
    # (pd.np is deprecated, using np.inf)
    df_route['BOARDS_PER_HOUR'] = (df_route['MTH_BOARD']/df_route['MTH_REV_HOURS']).replace([np.inf, None], 0).round(1)
    df_route['PASSENGERS_PER_TRIP'] = (df_route['MTH_BOARD']/df_route['TOTAL_TRIPS']).replace([np.inf, None], 0).round(1)
    df_route['PASSENGERS_PER_MILE'] = (df_route['MTH_BOARD']/df_route['MTH_REV_MILES']).replace([np.inf, None], 0).round(3)

    output_list = []
    service_types = sorted(df_route['service_type'].unique())

    for stype in service_types:
        sub_df = df_route[df_route['service_type'] == stype].copy()
        sums = sub_df[['MTH_BOARD','MTH_REV_HOURS','MTH_PASS_MILES','MTH_REV_MILES','TOTAL_TRIPS']].sum()

        # SUBTOTAL row for this service type
        sub_row = {
            'service_type': stype,
            'ROUTE_NAME': 'SUBTOTAL',
            'MTH_BOARD': sums['MTH_BOARD'],
            'MTH_REV_HOURS': sums['MTH_REV_HOURS'],
            'MTH_PASS_MILES': sums['MTH_PASS_MILES'],
            'MTH_REV_MILES': sums['MTH_REV_MILES'],
            'TOTAL_TRIPS': sums['TOTAL_TRIPS']
        }
        if sums['MTH_REV_HOURS']:
            sub_row['BOARDS_PER_HOUR'] = round(sums['MTH_BOARD']/sums['MTH_REV_HOURS'],1)
        else:
            sub_row['BOARDS_PER_HOUR'] = 0
        if sums['TOTAL_TRIPS']:
            sub_row['PASSENGERS_PER_TRIP'] = round(sums['MTH_BOARD']/sums['TOTAL_TRIPS'],1)
        else:
            sub_row['PASSENGERS_PER_TRIP'] = 0
        if sums['MTH_REV_MILES']:
            sub_row['PASSENGERS_PER_MILE'] = round(sums['MTH_BOARD']/sums['MTH_REV_MILES'],3)
        else:
            sub_row['PASSENGERS_PER_MILE'] = 0

        output_list.append(sub_df)
        output_list.append(pd.DataFrame([sub_row]))

    df_summary = pd.concat(output_list, ignore_index=True)

    # Grand total row
    not_sub = df_summary['ROUTE_NAME'] != 'SUBTOTAL'
    grand_sums = df_summary[not_sub][['MTH_BOARD','MTH_REV_HOURS','MTH_PASS_MILES','MTH_REV_MILES','TOTAL_TRIPS']].sum()

    total_dict = {
        'service_type': 'SYSTEMWIDE',
        'ROUTE_NAME': 'TOTAL',
        'MTH_BOARD': grand_sums['MTH_BOARD'],
        'MTH_REV_HOURS': grand_sums['MTH_REV_HOURS'],
        'MTH_PASS_MILES': grand_sums['MTH_PASS_MILES'],
        'MTH_REV_MILES': grand_sums['MTH_REV_MILES'],
        'TOTAL_TRIPS': grand_sums['TOTAL_TRIPS']
    }
    if grand_sums['MTH_REV_HOURS']:
        total_dict['BOARDS_PER_HOUR'] = round(grand_sums['MTH_BOARD']/grand_sums['MTH_REV_HOURS'],1)
    else:
        total_dict['BOARDS_PER_HOUR'] = 0
    if grand_sums['TOTAL_TRIPS']:
        total_dict['PASSENGERS_PER_TRIP'] = round(grand_sums['MTH_BOARD']/grand_sums['TOTAL_TRIPS'],1)
    else:
        total_dict['PASSENGERS_PER_TRIP'] = 0
    if grand_sums['MTH_REV_MILES']:
        total_dict['PASSENGERS_PER_MILE'] = round(grand_sums['MTH_BOARD']/grand_sums['MTH_REV_MILES'],3)
    else:
        total_dict['PASSENGERS_PER_MILE'] = 0

    df_summary = df_summary.append(total_dict, ignore_index=True)
    return df_summary


###############################################################################
#                     BUILD TIME-SERIES FOR PLOTTING                          #
###############################################################################

def build_monthly_timeseries(all_data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Returns a DataFrame where each row is (period, route),
    and columns for the various metrics we might plot:
      - total_ridership      => sum of MTH_BOARD (all service_periods)
      - weekday_avg          => (weekday boardings) / (weekday DAYS) if any
      - saturday_avg         => (sat boardings) / (sat DAYS) if any
      - sunday_avg           => (sun boardings) / (sun DAYS) if any
      - revenue_hours        => sum of MTH_REV_HOURS
      - trips                => sum of TOTAL_TRIPS
      - revenue_miles        => sum of MTH_REV_MILES
      - pph (passengers/hour) => total_ridership / revenue_hours
      - ppt (passengers/trip) => total_ridership / trips
      - ppm (passengers/mile) => total_ridership / revenue_miles

    We'll produce these data for each route + month, plus a "SYSTEMWIDE" row.
    """
    # For convenience, let’s keep the original columns:
    # MTH_BOARD, DAYS, MTH_REV_HOURS, TOTAL_TRIPS, MTH_REV_MILES
    # along with SERVICE_PERIOD so we can separate weekday/sat/sun.

    # Group by (period, route_name, service_period) to sum the relevant columns
    group_cols = ['period', 'ROUTE_NAME', 'SERVICE_PERIOD']
    agg_df = (all_data
              .groupby(group_cols, as_index=False)
              .agg({'MTH_BOARD': 'sum',
                    'DAYS': 'sum',
                    'MTH_REV_HOURS': 'sum',
                    'TOTAL_TRIPS': 'sum',
                    'MTH_REV_MILES': 'sum'}))

    # We want each row to correspond to (period, route).
    # We'll pivot the daily boardings for weekday/sat/sun so we can form averages.

    def get_daytype_sum(dfsub, daytype):
        row = dfsub.loc[dfsub['SERVICE_PERIOD'] == daytype]
        if row.empty:
            return (0, 0)  # (boardings, days)
        return (row['MTH_BOARD'].values[0], row['DAYS'].values[0])

    rows = []
    for (period, route), df_grp in agg_df.groupby(['period','ROUTE_NAME']):
        # Sum across all day types for total ridership, hours, trips, miles
        total_ridership = df_grp['MTH_BOARD'].sum()
        revenue_hours   = df_grp['MTH_REV_HOURS'].sum()
        total_trips     = df_grp['TOTAL_TRIPS'].sum()
        revenue_miles   = df_grp['MTH_REV_MILES'].sum()

        # For weekday avg, sat avg, sun avg, we look specifically at each day type
        wd_board, wd_days = get_daytype_sum(df_grp, 'Weekday')
        sat_board, sat_days = get_daytype_sum(df_grp, 'Saturday')
        sun_board, sun_days = get_daytype_sum(df_grp, 'Sunday')

        # If days = 0, result is None or 0; up to you. We'll do None if no days
        def safe_div(a, b):
            return round(a/b, 1) if b else None

        weekday_avg   = safe_div(wd_board, wd_days)
        saturday_avg  = safe_div(sat_board, sat_days)
        sunday_avg    = safe_div(sun_board, sun_days)

        pph = round(total_ridership/revenue_hours, 1) if revenue_hours else None
        ppt = round(total_ridership/total_trips, 1)   if total_trips else None
        ppm = round(total_ridership/revenue_miles, 3) if revenue_miles else None

        rows.append({
            'period': period,
            'route': route,
            'total_ridership': total_ridership,
            'weekday_avg': weekday_avg,
            'saturday_avg': saturday_avg,
            'sunday_avg': sunday_avg,
            'revenue_hours': revenue_hours,
            'trips': total_trips,
            'revenue_miles': revenue_miles,
            'pph': pph,
            'ppt': ppt,
            'ppm': ppm
        })

    df_time = pd.DataFrame(rows)

    # Also build a systemwide row by summing across all routes
    # for each period.
    syswide_rows = []
    for period in config['ordered_periods']:
        df_period = df_time[df_time['period'] == period]
        # Sum columns
        total_ridership = df_period['total_ridership'].sum()
        revenue_hours   = df_period['revenue_hours'].sum()
        total_trips     = df_period['trips'].sum()
        revenue_miles   = df_period['revenue_miles'].sum()

        # Weighted daily averages for weekday/sat/sun (or we can do sum of board/days again)
        # If you prefer a simpler approach, we can sum the board/days across all routes.
        # But for now, let's just do a simple sum->divide approach, same logic as above:
        wd_sum = df_period['weekday_avg'].count()  # actually we need raw board/days from original...
        # For simplicity, let's just treat it as a system-level average daily ridership:
        # We'll re-aggregate from the original dataset to be more accurate, but let's keep it short:
        # If you truly want a system-level average, you'd sum boardings and sum days from each route
        # for that day type. That requires referencing the "agg_df" again. This is a demonstration:
        df_p = agg_df[(agg_df['period']==period) & (agg_df['SERVICE_PERIOD']=='Weekday')]
        sys_wd_board = df_p['MTH_BOARD'].sum()
        sys_wd_days  = df_p['DAYS'].sum()

        df_s = agg_df[(agg_df['period']==period) & (agg_df['SERVICE_PERIOD']=='Saturday')]
        sys_sat_board = df_s['MTH_BOARD'].sum()
        sys_sat_days  = df_s['DAYS'].sum()

        df_su = agg_df[(agg_df['period']==period) & (agg_df['SERVICE_PERIOD']=='Sunday')]
        sys_sun_board = df_su['MTH_BOARD'].sum()
        sys_sun_days  = df_su['DAYS'].sum()

        def safe_div(a, b, r=1):
            return round(a/b, r) if b else None

        weekday_avg   = safe_div(sys_wd_board, sys_wd_days, 1)
        saturday_avg  = safe_div(sys_sat_board, sys_sat_days, 1)
        sunday_avg    = safe_div(sys_sun_board, sys_sun_days, 1)

        pph = round(total_ridership/revenue_hours, 1) if revenue_hours else None
        ppt = round(total_ridership/total_trips, 1)   if total_trips else None
        ppm = round(total_ridership/revenue_miles, 3) if revenue_miles else None

        syswide_rows.append({
            'period': period,
            'route': 'SYSTEMWIDE',
            'total_ridership': total_ridership,
            'weekday_avg': weekday_avg,
            'saturday_avg': saturday_avg,
            'sunday_avg': sunday_avg,
            'revenue_hours': revenue_hours,
            'trips': total_trips,
            'revenue_miles': revenue_miles,
            'pph': pph,
            'ppt': ppt,
            'ppm': ppm
        })

    df_sys = pd.DataFrame(syswide_rows)
    df_time = pd.concat([df_time, df_sys], ignore_index=True)

    return df_time


def plot_metric_over_time(df_time: pd.DataFrame, metric: str, config: dict):
    """
    For each route, plot the given metric vs. period. Also plot a "SYSTEMWIDE"
    route. Save each route’s plot in a subfolder "plots/<metric>".
    """
    output_dir = config['output_dir']
    plot_dir = os.path.join(output_dir, 'plots', metric)
    os.makedirs(plot_dir, exist_ok=True)

    # Sort the df_time by period in the correct order
    # We rely on config['ordered_periods'] for the x-axis sequence.
    ordered_periods = config['ordered_periods']

    # Keep just relevant columns
    # df_time has columns 'period', 'route', and metric
    # Filter out rows that don't have values for this metric
    df_metric = df_time[['period','route', metric]].copy()

    # Convert metric to float, fill NaN with 0 or skip if you prefer
    # (Alternatively, we can skip plotting if all are None.)
    df_metric[metric] = pd.to_numeric(df_metric[metric], errors='coerce')

    for route in sorted(df_metric['route'].unique()):
        df_r = df_metric[df_metric['route'] == route].copy()

        # Build y-values in the correct period order
        y_vals = []
        x_labels = []
        for p in ordered_periods:
            row = df_r[df_r['period'] == p]
            if not row.empty:
                val = row[metric].values[0]
            else:
                val = None
            y_vals.append(val)
            x_labels.append(p)

        # If all Nones, skip
        if all(v is None or pd.isna(v) for v in y_vals):
            continue

        plt.figure(figsize=PLOT_STYLE['figsize'])
        plt.plot(x_labels, y_vals, marker=PLOT_STYLE['marker'], linestyle=PLOT_STYLE['linestyle'])
        plt.title(f"{metric.replace('_',' ').title()} Over Time - Route {route}")
        plt.xlabel("Month")
        plt.ylabel(metric.replace('_',' ').title())
        plt.xticks(rotation=PLOT_STYLE['rotation'])
        plt.grid(PLOT_STYLE['grid'])

        # Try to set a nice y-limit
        # If we have numeric data, let’s set bottom=0, top at 110% of max
        numeric_vals = [v for v in y_vals if v is not None and not pd.isna(v)]
        if numeric_vals:
            y_max = max(numeric_vals)*1.1
            plt.ylim(0, y_max if y_max>0 else 1)

        fname = f"{metric}_route_{route}.png"
        outpath = os.path.join(plot_dir, fname)
        plt.tight_layout()
        plt.savefig(outpath, dpi=150)
        plt.close()


def generate_all_plots(df_time: pd.DataFrame, config: dict, plot_config: dict):
    """
    Check each boolean in plot_config, and if True, generate plots
    for the corresponding metric.
    """
    # Map the boolean keys to the actual column in df_time
    # You can rename them however you'd like.
    metric_map = {
        'plot_total_ridership': 'total_ridership',
        'plot_weekday_avg':     'weekday_avg',
        'plot_saturday_avg':    'saturday_avg',
        'plot_sunday_avg':      'sunday_avg',
        'plot_revenue_hours':   'revenue_hours',
        'plot_trips':           'trips',
        'plot_revenue_miles':   'revenue_miles',
        'plot_pph':             'pph',
        'plot_ppt':             'ppt',
        'plot_ppm':             'ppm'
    }

    for config_key, col_name in metric_map.items():
        if plot_config.get(config_key, False):
            print(f"Generating plots for {col_name} ...")
            plot_metric_over_time(df_time, col_name, config)


###############################################################################
#                                 MAIN                                        #
###############################################################################

def main():
    """
    Main entry point for the monthly NTD report generator.

    This function orchestrates the entire ETL and analysis process:
    1. Reads Excel data from multiple periods.
    2. Classifies routes, computes derived metrics.
    3. Aggregates/exports data into various Excel outputs.
    4. Optionally generates plots for visualizing metrics over time.
    """
    # 1. Read all periods from config
    data_dict = read_excel_data(CONFIG)

    # 2. Classify routes & corridors + derived columns
    for period, df in data_dict.items():
        df['service_type'] = df['ROUTE_NAME'].apply(lambda r: classify_route(r, CONFIG))
        df['corridors']    = df['ROUTE_NAME'].apply(lambda r: classify_corridor(r, CONFIG))
        df = calculate_derived_columns(df)
        df['period'] = period  # keep track of which month/period
        data_dict[period] = df  # store updated DataFrame back

    # 3. Concatenate all periods into a single DataFrame
    all_data = pd.concat(data_dict.values(), ignore_index=True)

    # 3a. Check for any unknown routes (optional)
    unknown_routes = all_data.loc[all_data['service_type']=='unknown','ROUTE_NAME'].unique()
    if len(unknown_routes) > 0:
        print("\nRoutes not classified by SERVICE_TYPE_DICT:")
        print(", ".join(sorted(unknown_routes)))

    # =======================================================================
    #             FILE #1: DetailedAllPeriods + Monthly Sheets
    # =======================================================================
    output_dir = CONFIG['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    file1_path = os.path.join(output_dir, 'DetailedAllPeriods_andMonthlySheets.xlsx')
    with pd.ExcelWriter(file1_path) as writer:
        # (1) 'DetailedAllPeriods' first
        all_data.to_excel(writer, sheet_name='DetailedAllPeriods', index=False)

        # (2) Then each month's data in subsequent sheets
        for period in CONFIG['ordered_periods']:
            data_dict[period].to_excel(writer, sheet_name=period, index=False)

    print("Concatenated NTD data has been exported successfully.")


    # =======================================================================
    #       FILE #2: Aggregated by Service Type (YTD + each month)
    # =======================================================================
    file2_path = os.path.join(output_dir, 'AggByServiceType.xlsx')
    # YTD aggregator (all periods)
    service_type_agg_ytd = aggregate_by_service_type(all_data)

    with pd.ExcelWriter(file2_path) as writer:
        # First sheet: YTD aggregator
        service_type_agg_ytd.to_excel(writer, sheet_name='YTD', index=False)

        # Then one sheet per month
        for period in CONFIG['ordered_periods']:
            df_period = data_dict[period]
            monthly_agg = aggregate_by_service_type(df_period)
            monthly_agg.to_excel(writer, sheet_name=period, index=False)

    print("Summary statistics by service type have been exported successfully.")


    # =======================================================================
    #             FILE #3: Route-Level Summary (YTD Only)
    # =======================================================================
    file3_path = os.path.join(output_dir, 'RouteLevelSummary.xlsx')
    route_summary_ytd = route_level_summary(all_data)

    with pd.ExcelWriter(file3_path) as writer:
        route_summary_ytd.to_excel(writer, sheet_name='YTD_Route_Level', index=False)

    print("YTD summary statistics by route have been exported successfully.")


    # =======================================================================
    #            OPTIONAL: GENERATE TIME-SERIES PLOTS FOR METRICS
    # =======================================================================
    # Build a monthly time-series dataframe for each route & systemwide
    df_time = build_monthly_timeseries(all_data, CONFIG)
    # Then generate plots if booleans are set to True
    generate_all_plots(df_time, CONFIG, PLOT_CONFIG)

    print("All done.")


if __name__ == '__main__':
    main()
