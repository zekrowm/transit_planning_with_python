"""
Census Data Processing Script

This script processes census data by performing the following operations:
- Loads and merges shapefiles from specified input directories.
- Filters geographic data based on provided FIPS codes.
- Processes various demographic datasets including population, households, jobs,
  income, ethnicity, language proficiency, vehicle ownership, and age.
- Merges tract-level and block-level data to calculate estimates.
- Exports the processed data to CSV and shapefile formats for further analysis.

Configuration:
- Customize the full file paths in the configuration section below.
- JT00, P1, H9, and .shp files are mandatory. They can be found here:
  - https://lehd.ces.census.gov/data/
  - https://data.census.gov/table
  - https://www.census.gov/cgi-bin/geo/shapefiles/index.php
- Other tables are optional depending on the detailed information you are
  interested in.
- Output configuration is consolidated at the end of the config section.
"""

import logging
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# -----------------------------------------------------------------------------
# 1) Mandatory Inputs
# -----------------------------------------------------------------------------

# A. Shapefiles (block-level)
BLOCK_SHP_FILES = [
    r"C:\full\path\to\tl_2023_11_tabblock20.shp",
    r"C:\full\path\to\tl_2023_24_tabblock20.shp",
    r"C:\full\path\to\tl_2023_51_tabblock20.shp",
    # Add or remove as needed
]

# B. FIPS codes to filter
FIPS_TO_FILTER = [
    "51059",
    "51013",
    "51510",
    "51600",
    "51610",
    "11001",
    "24031",
    "24033",
    "51107",
    "51153",
    "51683",
    "51685",
    # Add or remove as needed
]

# C. Population data by block (P1)
P1_FILES = [
    r"C:\full\path\to\DECENNIALPL2020.P1-Data.csv",
    r"C:\full\path\to\DECENNIALPL2020.P1-Data.csv",
    r"C:\full\path\to\DECENNIALPL2020.P1-Data.csv",
]

# D. Households data by block (H9)
H9_FILES = [
    r"C:\full\path\to\DECENNIALDHC2020.H9-Data.csv",
    r"C:\full\path\to\DECENNIALDHC2020.H9-Data.csv",
    r"C:\full\path\to\DECENNIALDHC2020.H9-Data.csv",
]
DTYPES_H9 = {"GEO_ID": str, "H9_001N": "Int64"}

# E. Jobs data by block (JT00)
JT00_FILES = [
    r"C:\full\path\to\va_wac_S000_JT00_2021.csv.gz",
    r"C:\full\path\to\md_wac_S000_JT00_2021.csv.gz",
    r"C:\full\path\to\dc_wac_S000_JT00_2021.csv.gz",
]

# -----------------------------------------------------------------------------
# 2) Optional Inputs
# -----------------------------------------------------------------------------

# Income data by tract (B19001)
INCOME_B19001_FILES = [
    r"C:\full\path\to\ACSDT5Y2022.B19001-Data.csv"
    # Add more paths or remove if not used
]

# Ethnicity data by tract (P9)
ETHNICITY_P9_FILES = [
    r"C:\full\path\to\DECENNIALDHC2020.P9-Data.csv"
    # Add more paths or remove if not used
]

# Language data by tract (C16001)
LANGUAGE_C16001_FILES = [
    r"C:\full\path\to\ACSDT5Y2022.C16001-Data.csv"
    # Add more paths or remove if not used
]

# Vehicle ownership data by tract (B08201)
VEHICLE_B08201_FILES = [
    r"C:\full\path\to\ACSDT5Y2022.B08201-Data.csv"
    # Add more paths or remove if not used
]

# Age data by tract (B01001)
AGE_B01001_FILES = [
    r"C:\full\path\to\ACSDT5Y2022.B01001-Data.csv"
    # Add more paths or remove if not used
]

# -----------------------------------------------------------------------------
# 3) Output Configuration
# -----------------------------------------------------------------------------

CSV_OUTPUT_PATH = r"C:\full\path\to\output\df_joined_blocks.csv"
SHP_OUTPUT_PATH = (
    r"C:\Users\zach\Desktop\Zach\python_stuff\projects\census_data_processing_for_transit_2025_01_21"
    r"\output\va_md_dc_census_blocks_folder\va_census_blocks.shp"
)

# =============================================================================
# FUNCTIONS
# =============================================================================


def load_and_merge_shapefiles(shapefile_paths: list[str]) -> gpd.GeoDataFrame:
    """
    Load and merge multiple shapefiles into a single GeoDataFrame.

    :param shapefile_paths: A list of file paths to block-level shapefiles.
    :return: Merged GeoDataFrame of all block-level shapefiles.
    """
    logging.info("Loading shapefiles...")
    gdf_list = [gpd.read_file(shp) for shp in shapefile_paths]
    merged = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs=gdf_list[0].crs)
    logging.info("Shapefiles loaded and merged.")
    return merged


def filter_geo_data_by_fips(
    gdf: gpd.GeoDataFrame, state_col: str, county_col: str, fips_to_filter: list[str]
) -> gpd.GeoDataFrame:
    """
    Create a new FIPS column from state and county fields, then filter rows to only those FIPS codes.

    :param gdf: GeoDataFrame containing at least state and county columns.
    :param state_col: Column name holding the STATEFP data.
    :param county_col: Column name holding the COUNTYFP data.
    :param fips_to_filter: List of FIPS codes to keep.
    :return: Filtered GeoDataFrame.
    """
    logging.info("Filtering GeoDataFrame by FIPS codes...")
    gdf["FIPS"] = gdf[state_col].astype(str) + gdf[county_col].astype(str)
    filtered = gdf[gdf["FIPS"].isin(fips_to_filter)].copy()
    logging.info("GeoDataFrame filtered; remaining rows: %d", len(filtered))
    return filtered


def plot_geodataframe(gdf: gpd.GeoDataFrame, title: str) -> None:
    """
    Display a quick plot of a GeoDataFrame.

    :param gdf: GeoDataFrame to plot.
    :param title: Plot title.
    """
    logging.info("Plotting filtered shapefile for quick visualization...")
    gdf.plot()
    plt.title(title)
    plt.show()


def load_csv_data(
    file_paths: list[str],
    skiprows: list[int] | None,
    column_renames: dict[str, str],
    columns_to_keep: list[str] | None,
    dtype_map: dict[str, str] | None = None,
    compression: str | None = None,
) -> pd.DataFrame:
    """
    Load and concatenate multiple CSV files into a single DataFrame.

    :param file_paths: List of CSV (or compressed CSV) file paths.
    :param skiprows: Row indices to skip (often used for the Census 1-row metadata).
    :param column_renames: Mapping from original columns to new names.
    :param columns_to_keep: List of columns to keep in the resulting DataFrame.
    :param dtype_map: Optional dtype mapping for certain columns.
    :param compression: Compression type for reading CSV (e.g., 'gzip').
    :return: Concatenated DataFrame of all input files.
    """
    logging.info("Loading CSV data from paths: %s", file_paths)
    df_list = []
    for file in file_paths:
        df_temp = pd.read_csv(file, skiprows=skiprows, dtype=dtype_map, compression=compression)
        if column_renames:
            df_temp.rename(columns=column_renames, inplace=True)
        if columns_to_keep:
            df_temp = df_temp[columns_to_keep]
        df_list.append(df_temp)

    concatenated_df = pd.concat(df_list, ignore_index=True)
    logging.info("CSV data loaded and concatenated; final shape: %s", concatenated_df.shape)
    return concatenated_df


def merge_dataframes_on_geo_id(
    base_df: pd.DataFrame, other_df: pd.DataFrame, join_how: str = "outer"
) -> pd.DataFrame:
    """
    Merge two dataframes on 'GEO_ID'.

    :param base_df: The primary DataFrame.
    :param other_df: The secondary DataFrame.
    :param join_how: Merge strategy, e.g. 'outer', 'left', 'right', 'inner'.
    :return: Merged DataFrame.
    """
    if base_df.empty:
        return other_df
    elif other_df.empty:
        return base_df
    else:
        return pd.merge(base_df, other_df, on="GEO_ID", how=join_how)


def calculate_tract_based_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a tract-level DataFrame with known ratio columns (like perc_minority),
    compute new columns as estimated values at the block-level after merging
    with block data.

    The columns must exist in df to avoid KeyErrors.

    :param df: DataFrame that already has block-level columns and optional ratio columns.
    :return: Modified DataFrame with new estimate columns (in-place changes).
    """
    # Example columns to check
    estimate_columns = [
        ("perc_low_income", "total_hh", "est_low_income"),
        ("perc_lep", "total_pop", "est_lep"),
        ("perc_minority", "total_pop", "est_minority"),
        ("perc_lo_veh", "total_hh", "est_lo_veh"),
        ("perc_lo_veh_mod", "total_hh", "est_lo_veh_mod"),
        ("perc_youth", "total_pop", "est_youth"),
        ("perc_elderly", "total_pop", "est_elderly"),
    ]

    for perc_col, base_col, new_col in estimate_columns:
        if perc_col in df.columns and base_col in df.columns:
            df[new_col] = df[perc_col] * df[base_col]
            df[new_col] = df[new_col].round(3)

    return df


def export_dataframes_to_disk(
    df: pd.DataFrame,
    geo_df: gpd.GeoDataFrame,
    csv_output_path: str,
    shp_output_path: str,
    shapefile_merge_left: str = "GEOIDFQ20",
    shapefile_merge_right: str = "GEO_ID",
) -> None:
    """
    Export a DataFrame to CSV and merge with a GeoDataFrame to export as a shapefile.

    :param df: DataFrame with final aggregated data.
    :param geo_df: GeoDataFrame (filtered by FIPS) to merge with df before exporting shapefile.
    :param csv_output_path: Output path for the CSV file.
    :param shp_output_path: Output path for the shapefile.
    :param shapefile_merge_left: Column in 'geo_df' for merging.
    :param shapefile_merge_right: Column in 'df' for merging.
    """
    # Export CSV
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
    df.to_csv(csv_output_path, index=True)
    logging.info("CSV file saved to: %s", csv_output_path)

    # Export Shapefile
    os.makedirs(os.path.dirname(shp_output_path), exist_ok=True)

    if shapefile_merge_left in geo_df.columns and shapefile_merge_right in df.columns:
        logging.info("Merging data for shapefile export...")
        result_gdf = geo_df.merge(
            df, left_on=shapefile_merge_left, right_on=shapefile_merge_right, how="left"
        )
    else:
        logging.warning(
            "Could not merge shapefile. Check column names: '%s' or '%s' not found.",
            shapefile_merge_left,
            shapefile_merge_right,
        )
        result_gdf = geo_df.copy()

    result_gdf.to_file(shp_output_path)
    logging.info("Shapefile saved to: %s", shp_output_path)


# =============================================================================
# MAIN
# =============================================================================


def main():
    """
    Main function orchestrating the overall census data processing logic.
    """
    logging.info("Starting Census Data Processing...")

    # 1) Load and filter shapefiles by FIPS
    merged_gdf = load_and_merge_shapefiles(BLOCK_SHP_FILES)
    filtered_gdf = filter_geo_data_by_fips(merged_gdf, "STATEFP20", "COUNTYFP20", FIPS_TO_FILTER)
    plot_geodataframe(filtered_gdf, "Shapefile Plot - Filtered by FIPS")

    # 2) Load mandatory block-level data: population (P1), household (H9), jobs (JT00)
    df_population = load_csv_data(
        P1_FILES,
        skiprows=[1],
        column_renames={"GEO_ID": "GEO_ID", "NAME": "NAME", "P1_001N": "total_pop"},
        columns_to_keep=["GEO_ID", "NAME", "total_pop"],
    )

    df_household = load_csv_data(
        H9_FILES,
        skiprows=[1],
        column_renames={"H9_001N": "total_hh"},
        columns_to_keep=["GEO_ID", "total_hh"],
        dtype_map=DTYPES_H9,
    )

    df_jobs = load_csv_data(
        JT00_FILES,
        skiprows=None,
        column_renames={
            "w_geocode": "w_geocode",
            "C000": "tot_empl",
            "CE01": "low_wage",
            "CE02": "mid_wage",
            "CE03": "high_wage",
        },
        columns_to_keep=["w_geocode", "tot_empl", "low_wage", "mid_wage", "high_wage"],
        compression="gzip",
    )
    # Add the 'GEO_ID' column for merging with a prefix
    df_jobs["GEO_ID"] = "1000000US" + df_jobs["w_geocode"].astype(str)
    df_jobs.drop(columns=["w_geocode"], inplace=True)

    # 3) Concatenate block-level datasets
    df_blocks = merge_dataframes_on_geo_id(df_population, df_household)
    df_blocks = merge_dataframes_on_geo_id(df_blocks, df_jobs)

    # Create synthetic IDs
    df_blocks["tract_id_synth"] = df_blocks["GEO_ID"].str[9:20]
    df_blocks["block_id_synth"] = df_blocks["GEO_ID"].str[9:24]
    df_blocks.fillna(0, inplace=True)

    # 4) Load optional tract-level data
    #    (income, ethnicity, language, vehicle ownership, age)
    #    Each is done separately with load_csv_data() and then
    #    additional transformations as needed.

    # --- Income ---
    df_income = pd.DataFrame()
    if INCOME_B19001_FILES:
        df_income = load_csv_data(
            INCOME_B19001_FILES,
            skiprows=[1],
            column_renames={
                "GEO_ID": "GEO_ID",
                "NAME": "NAME",
                "B19001_001E": "total_hh",
                "B19001_002E": "sub_10k",
                "B19001_003E": "10k_15k",
                "B19001_004E": "15k_20k",
                "B19001_005E": "20k_25k",
                "B19001_006E": "25k_30k",
                "B19001_007E": "30k_35k",
                "B19001_008E": "35k_40k",
                "B19001_009E": "40k_45k",
                "B19001_010E": "45k_50k",
                "B19001_011E": "50k_60k",
            },
            columns_to_keep=[
                "GEO_ID",
                "NAME",
                "total_hh",
                "sub_10k",
                "10k_15k",
                "15k_20k",
                "20k_25k",
                "25k_30k",
                "30k_35k",
                "35k_40k",
                "40k_45k",
                "45k_50k",
                "50k_60k",
            ],
        )
        # Add derived columns
        df_income["low_income"] = df_income[
            [
                "sub_10k",
                "10k_15k",
                "15k_20k",
                "20k_25k",
                "25k_30k",
                "30k_35k",
                "35k_40k",
                "40k_45k",
                "45k_50k",
                "50k_60k",
            ]
        ].sum(axis=1)
        df_income["perc_low_income"] = df_income["low_income"] / df_income["total_hh"]
        df_income["FIPS_code"] = df_income["GEO_ID"].str[9:14]
        df_income.drop(["total_hh"], axis=1, inplace=True)

    # --- Ethnicity ---
    df_ethnicity = pd.DataFrame()
    if ETHNICITY_P9_FILES:
        df_ethnicity = load_csv_data(
            ETHNICITY_P9_FILES,
            skiprows=[1],
            column_renames={
                "GEO_ID": "GEO_ID",
                "NAME": "NAME",
                "P9_001N": "total_pop",
                "P9_002N": "all_hisp",
                "P9_005N": "white",
                "P9_006N": "black",
                "P9_007N": "native",
                "P9_008N": "asian",
                "P9_009N": "pac_isl",
                "P9_010N": "other",
                "P9_011N": "multi",
            },
            columns_to_keep=[
                "GEO_ID",
                "NAME",
                "total_pop",
                "all_hisp",
                "white",
                "black",
                "native",
                "asian",
                "pac_isl",
                "other",
                "multi",
            ],
        )
        df_ethnicity["minority"] = df_ethnicity[
            ["black", "native", "asian", "pac_isl", "other", "multi"]
        ].sum(axis=1)
        df_ethnicity["perc_minority"] = df_ethnicity["minority"] / df_ethnicity["total_pop"]
        df_ethnicity["FIPS_code"] = df_ethnicity["GEO_ID"].str[9:14]
        df_ethnicity.drop(["total_pop"], axis=1, inplace=True)

    # --- Language ---
    df_language = pd.DataFrame()
    if LANGUAGE_C16001_FILES:
        df_language = load_csv_data(
            LANGUAGE_C16001_FILES,
            skiprows=[1],
            column_renames={
                "C16001_001E": "total_lang_pop",
                "C16001_005E": "spanish_engnwell",
                "C16001_008E": "frenchetc_engnwell",
                "C16001_011E": "germanetc_engnwell",
                "C16001_014E": "slavicetc_engnwell",
                "C16001_017E": "indoeuroetc_engnwell",
                "C16001_020E": "korean_engnwell",
                "C16001_023E": "chineseetc_engnwell",
                "C16001_026E": "vietnamese_engnwell",
                "C16001_032E": "asiapacetc_engnwell",
                "C16001_035E": "arabic_engnwell",
                "C16001_037E": "otheretc_engnwell",
            },
            columns_to_keep=[
                "GEO_ID",
                "total_lang_pop",
                "spanish_engnwell",
                "frenchetc_engnwell",
                "germanetc_engnwell",
                "slavicetc_engnwell",
                "indoeuroetc_engnwell",
                "korean_engnwell",
                "chineseetc_engnwell",
                "vietnamese_engnwell",
                "asiapacetc_engnwell",
                "arabic_engnwell",
                "otheretc_engnwell",
            ],
        )
        lep_cols = [
            "spanish_engnwell",
            "frenchetc_engnwell",
            "germanetc_engnwell",
            "slavicetc_engnwell",
            "indoeuroetc_engnwell",
            "korean_engnwell",
            "chineseetc_engnwell",
            "vietnamese_engnwell",
            "asiapacetc_engnwell",
            "arabic_engnwell",
            "otheretc_engnwell",
        ]
        df_language[lep_cols] = (
            df_language[lep_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
        )
        df_language["all_nwell"] = df_language[lep_cols].sum(axis=1)
        df_language["perc_lep"] = df_language["all_nwell"] / df_language["total_lang_pop"]
        df_language["perc_lep"].replace([float("inf"), -float("inf")], 0, inplace=True)
        df_language["perc_lep"] = df_language["perc_lep"].fillna(0).round(3)

    # --- Vehicle Ownership ---
    df_vehicle = pd.DataFrame()
    if VEHICLE_B08201_FILES:
        df_vehicle = load_csv_data(
            VEHICLE_B08201_FILES,
            skiprows=[1],
            column_renames={
                "GEO_ID": "GEO_ID",
                "B08201_001E": "all_hhs",
                "B08201_002E": "veh_0_all_hh",
                "B08201_003E": "veh_1_all_hh",
                "B08201_008E": "veh_0_hh_1",
                "B08201_009E": "veh_1_hh_1",
                "B08201_014E": "veh_0_hh_2",
                "B08201_015E": "veh_1_hh_2",
                "B08201_020E": "veh_0_hh_3",
                "B08201_021E": "veh_1_hh_3",
                "B08201_022E": "veh_2_hh_3",
                "B08201_026E": "veh_0_hh_4p",
                "B08201_027E": "veh_1_hh_4p",
                "B08201_028E": "veh_2_hh_4p",
            },
            columns_to_keep=[
                "GEO_ID",
                "all_hhs",
                "veh_0_all_hh",
                "veh_1_all_hh",
                "veh_0_hh_1",
                "veh_1_hh_1",
                "veh_0_hh_2",
                "veh_1_hh_2",
                "veh_0_hh_3",
                "veh_1_hh_3",
                "veh_2_hh_3",
                "veh_0_hh_4p",
                "veh_1_hh_4p",
                "veh_2_hh_4p",
            ],
        )
        df_vehicle["all_lo_veh_hh"] = df_vehicle[["veh_0_all_hh", "veh_1_all_hh"]].sum(axis=1)
        df_vehicle["perc_lo_veh"] = df_vehicle["all_lo_veh_hh"] / df_vehicle["all_hhs"]
        df_vehicle["perc_0_veh"] = df_vehicle["veh_0_all_hh"] / df_vehicle["all_hhs"]
        df_vehicle["perc_1_veh"] = df_vehicle["veh_1_all_hh"] / df_vehicle["all_hhs"]
        df_vehicle["perc_veh_1_hh_1"] = df_vehicle["veh_1_hh_1"] / df_vehicle["all_hhs"]
        df_vehicle["perc_lo_veh_mod"] = df_vehicle["perc_lo_veh"] - df_vehicle["perc_veh_1_hh_1"]
        df_vehicle["perc_lo_veh_mod"] = df_vehicle["perc_lo_veh_mod"].round(3)

    # --- Age ---
    df_age = pd.DataFrame()
    if AGE_B01001_FILES:
        df_age = load_csv_data(
            AGE_B01001_FILES,
            skiprows=[1],
            column_renames={
                "GEO_ID": "GEO_ID",
                "B01001_001E": "total_pop",
                "B01001_006E": "m_15_17",
                "B01001_007E": "m_18_19",
                "B01001_008E": "m_20",
                "B01001_009E": "m_21",
                "B01001_020E": "m_65_66",
                "B01001_021E": "m_67_69",
                "B01001_022E": "m_70_74",
                "B01001_023E": "m_75_79",
                "B01001_024E": "m_80_84",
                "B01001_025E": "m_a_85",
                "B01001_030E": "f_15_17",
                "B01001_031E": "f_18_19",
                "B01001_032E": "f_20",
                "B01001_033E": "f_21",
                "B01001_044E": "f_65_66",
                "B01001_045E": "f_67_69",
                "B01001_046E": "f_70_74",
                "B01001_047E": "f_75_79",
                "B01001_048E": "f_80_84",
                "B01001_049E": "f_a_85",
            },
            columns_to_keep=[
                "GEO_ID",
                "total_pop",
                "m_15_17",
                "m_18_19",
                "m_20",
                "m_21",
                "m_65_66",
                "m_67_69",
                "m_70_74",
                "m_75_79",
                "m_80_84",
                "m_a_85",
                "f_15_17",
                "f_18_19",
                "f_20",
                "f_21",
                "f_65_66",
                "f_67_69",
                "f_70_74",
                "f_75_79",
                "f_80_84",
                "f_a_85",
            ],
        )
        df_age["all_youth"] = df_age[
            ["m_15_17", "f_15_17", "m_18_19", "f_18_19", "m_20", "f_20", "m_21", "f_21"]
        ].sum(axis=1)
        df_age["all_elderly"] = df_age[
            [
                "m_65_66",
                "f_65_66",
                "m_67_69",
                "f_67_69",
                "m_70_74",
                "f_70_74",
                "m_75_79",
                "f_75_79",
                "m_80_84",
                "f_80_84",
                "m_a_85",
                "f_a_85",
            ]
        ].sum(axis=1)
        df_age["perc_youth"] = (df_age["all_youth"] / df_age["total_pop"]).round(3)
        df_age["perc_elderly"] = (df_age["all_elderly"] / df_age["total_pop"]).round(3)
        df_age.drop(["total_pop"], axis=1, inplace=True)

    # 5) Merge all tract-level dataframes
    df_tracts = pd.DataFrame()  # Start empty
    for optional_df in [df_income, df_ethnicity, df_vehicle, df_age, df_language]:
        df_tracts = (
            optional_df
            if df_tracts.empty
            else pd.merge(df_tracts, optional_df, on="GEO_ID", how="outer")
        )
    df_tracts.fillna(0, inplace=True)

    # Cleanup columns from the optional data if desired
    columns_to_drop = [
        "sub_10k",
        "10k_15k",
        "15k_20k",
        "20k_25k",
        "25k_30k",
        "30k_35k",
        "35k_40k",
        "40k_45k",
        "45k_50k",
        "50k_60k",
        "veh_0_hh_1",
        "veh_1_hh_1",
        "veh_0_hh_2",
        "veh_1_hh_2",
        "veh_0_hh_3",
        "veh_1_hh_3",
        "veh_2_hh_3",
        "veh_0_hh_4p",
        "veh_1_hh_4p",
        "veh_2_hh_4p",
        "m_15_17",
        "m_18_19",
        "m_20",
        "m_21",
        "m_65_66",
        "m_67_69",
        "m_70_74",
        "m_75_79",
        "m_80_84",
        "m_a_85",
        "f_15_17",
        "f_18_19",
        "f_20",
        "f_21",
        "f_65_66",
        "f_67_69",
        "f_70_74",
        "f_75_79",
        "f_80_84",
        "f_a_85",
        "all_youth",
        "all_elderly",
        "all_hisp",
        "white",
        "black",
        "native",
        "asian",
        "pac_isl",
        "other",
        "multi",
        "minority",
        "total_lang_pop",
        "spanish_engnwell",
        "frenchetc_engnwell",
        "germanetc_engnwell",
        "slavicetc_engnwell",
        "indoeuroetc_engnwell",
        "korean_engnwell",
        "chineseetc_engnwell",
        "vietnamese_engnwell",
        "asiapacetc_engnwell",
        "arabic_engnwell",
        "otheretc_engnwell",
        "all_nwell",
        "low_income",
        "all_hhs",
        "veh_0_all_hh",
        "veh_1_all_hh",
        "all_lo_veh_hh",
    ]
    df_tracts.drop(
        columns=[c for c in columns_to_drop if c in df_tracts], inplace=True, errors="ignore"
    )

    # Clean tract_id for merging
    if not df_tracts.empty:
        df_tracts["tract_id_clean"] = df_tracts["GEO_ID"].str[9:]

    # 6) Merge block and tract data
    if not df_tracts.empty:
        df_combined = pd.merge(
            df_blocks, df_tracts, left_on="tract_id_synth", right_on="tract_id_clean", how="outer"
        )
    else:
        df_combined = df_blocks.copy()

    df_combined.fillna(0, inplace=True)

    # 7) Calculate block-level estimates for optional data
    df_combined = calculate_tract_based_ratios(df_combined)

    # 8) Filter final data by the relevant FIPS codes
    df_combined["FIPS_code"] = df_combined["GEO_ID"].str[9:14]
    df_filtered_blocks = df_combined[df_combined["FIPS_code"].isin(FIPS_TO_FILTER)].copy()

    # Convert ExtensionArray dtypes to float64 if needed
    for column in df_filtered_blocks.columns:
        if pd.api.types.is_extension_array_dtype(df_filtered_blocks[column]):
            df_filtered_blocks[column] = df_filtered_blocks[column].astype("float64")

    # 9) Export final data to CSV and Shapefile
    export_dataframes_to_disk(
        df_filtered_blocks,
        filtered_gdf,  # GeoDataFrame already filtered by FIPS
        CSV_OUTPUT_PATH,
        SHP_OUTPUT_PATH,
        shapefile_merge_left="GEOIDFQ20",
        shapefile_merge_right="GEO_ID",
    )

    logging.info("Census Data Processing complete.")


if __name__ == "__main__":
    main()
