"""Join and clean Census block- and tract-level data for analysis.

This module loads multiple Census CSV files at the block and tract level,
merges them into a single DataFrame, standardizes key columns, and optionally
filters by county FIPS codes. Residual raw Census code columns are dropped
unless explicitly preserved.

Typical usage is from a Jupyter notebook, either within ArcGIS Pro or a
standalone Python environment, with user-specified file paths.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Hashable, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# CSV output file path
CSV_OUTPUT_PATH: str | None = r"PATH\TO\OUTPUT\joined_blocks.csv"

# Optional county FIPS filter
# Provide 5-digit state+county codes, e.g. ["11001", "51059"].
# Leave empty ([]) to disable filtering.
COUNTY_FIPS_FILTER: list[str] = [
    # "11001",
    # "51059",
]

# --- Required block-level Census files (CSV or CSV.GZ format) ---------------
POP_FILES: list[str] = [
    r"PATH\TO\BLOCK_POPULATION_FILE.csv",  # e.g., DECENNIALPL2020.P1-Data.csv
]
HH_FILES: list[str] = [
    r"PATH\TO\BLOCK_HOUSEHOLDS_FILE.csv",  # e.g., DECENNIALDHC2020.H9-Data.csv
]
JOBS_FILES: list[str] = [
    r"PATH\TO\BLOCK_JOBS_FILE.csv.gz",  # e.g., dc_wac_S000_JT00_2022.csv.gz
]

# --- Optional tract-level inputs ---------------------------------------------
INCOME_FILES: list[str] = [
    r"PATH\TO\TRACT_INCOME_FILE.csv",  # e.g., ACSDT5Y2022.B19001-Data.csv
]
ETHNICITY_FILES: list[str] = [
    r"PATH\TO\TRACT_ETHNICITY_FILE.csv",  # e.g., DECENNIALDHC2020.P9-Data.csv
]
LANGUAGE_FILES: list[str] = [
    r"PATH\TO\TRACT_LANGUAGE_FILE.csv",  # e.g., ACSDT5Y2022.C16001-Data.csv
]
VEHICLE_FILES: list[str] = [
    r"PATH\TO\TRACT_VEHICLES_FILE.csv",  # e.g., ACSDT5Y2022.B08201-Data.csv
]
AGE_FILES: list[str] = [
    r"PATH\TO\TRACT_AGE_FILE.csv",  # e.g., ACSDT5Y2022.B01001-Data.csv
]

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s :: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

GEO_ID_COL = "GEO_ID"
# Regex to spot un-renamed Census code columns (e.g. B19001_003E, P9_007N, C000)
_UNFRIENDLY_COL_RE = re.compile(r"^[A-Z]{2,}\d{3,}.*")

# =============================================================================
# FUNCTIONS
# =============================================================================

def _fill_numeric_only(df: pd.DataFrame, value: int | float = 0) -> pd.DataFrame:
    """Replace *only* numeric NaNs with *value*; leave object columns untouched.

    Args:
        df: Frame to operate on (modified in-place and returned for convenience).
        value: Scalar used to fill missing values in numeric columns
            (defaults to ``0``).

    Returns:
        The same ``df`` reference, for call-chaining convenience.

    Notes
    -----
    This helper prevents accidental overwriting of NaN/None in string columns
    such as ``NAME`` while keeping downstream arithmetic safe for division.
    """
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(value)
    return df


def _load_and_concat(
    files: Sequence[str],
    *,
    skiprows: int | Sequence[int] | Callable[[int], bool] | None = None,
    dtype: Mapping[Hashable, str | np.dtype[Any]] | None = None,
    usecols: Sequence[Hashable] | None = None,
    rename: Mapping[str, str] | None = None,
    compression: Literal["infer", "gzip", "bz2", "zip", "xz", "zstd"] | None = None,
) -> pd.DataFrame:
    """Read multiple CSV/CSV-GZ files and return a concatenated DataFrame.

    Only explicitly selected or renamed columns are retained to minimise memory
    use and noise upstream.
    """
    frames: list[pd.DataFrame] = []

    for path in files:
        # Build kwargs dynamically so pandas receives only non-None values
        read_kwargs: dict[str, Any] = {"compression": compression}
        if skiprows is not None:
            read_kwargs["skiprows"] = skiprows
        if dtype is not None:
            read_kwargs["dtype"] = dtype
        if usecols is not None:
            read_kwargs["usecols"] = usecols

        df = pd.read_csv(path, **read_kwargs)

        if rename:
            df.rename(columns=rename, inplace=True)
            # If all columns were loaded, retain only those we explicitly renamed
            if usecols is None:
                keep = {GEO_ID_COL, "NAME", *rename.values()}
                df = df.loc[:, df.columns.intersection(keep)]

        frames.append(df)

    if not frames:  # no input files
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    LOGGER.debug("Loaded %d rows from %d file(s) [%s]", len(out), len(frames), files)
    return out


def _merge_on_geo_id(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Outer-merge two frames on GEO_ID, discarding duplicate label columns."""
    if left.empty:
        return right.copy()
    if right.empty:
        return left.copy()

    dup_cols = (set(left.columns) & set(right.columns)) - {GEO_ID_COL}
    right = right.drop(columns=dup_cols, errors="ignore")
    return left.merge(right, on=GEO_ID_COL, how="outer")


def _drop_unfriendly_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Remove any column that still looks like a raw Census code."""
    bad_cols = [c for c in df.columns if _UNFRIENDLY_COL_RE.match(c)]
    LOGGER.debug("Dropping %d unfriendly column(s): %s", len(bad_cols), bad_cols)
    return df.drop(columns=bad_cols, errors="ignore")


# -----------------------------------------------------------------------------
# BLOCK-LEVEL BUILD
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class _BlockInputs:
    pop_files: list[str]
    hh_files: list[str]
    jobs_files: list[str]


def _build_block_df(inp: _BlockInputs) -> pd.DataFrame:  # noqa: D401  (keep same sig)
    """Return a block-level DataFrame with population, households and jobs."""
    # ----- Population -------------------------------------------------------
    pop = _load_and_concat(
        inp.pop_files,
        skiprows=[1],
        rename={"P1_001N": "total_pop"},
        usecols=[GEO_ID_COL, "NAME", "P1_001N"],
    )

    # ----- Households -------------------------------------------------------
    hh = _load_and_concat(
        inp.hh_files,
        skiprows=[1],
        rename={"H9_001N": "total_hh"},
        usecols=[GEO_ID_COL, "H9_001N"],
        dtype={"H9_001N": "Int64"},
    )

    # ----- Jobs -------------------------------------------------------------
    jobs = _load_and_concat(
        inp.jobs_files,
        rename={
            "C000": "tot_empl",
            "CE01": "low_wage",
            "CE02": "mid_wage",
            "CE03": "high_wage",
        },
        usecols=["w_geocode", "C000", "CE01", "CE02", "CE03"],
        compression="gzip",
    )
    if not jobs.empty:
        jobs[GEO_ID_COL] = "1000000US" + jobs["w_geocode"].astype(str)
        jobs.drop(columns="w_geocode", inplace=True)

    # ----- Merge + tidy -----------------------------------------------------
    df = _merge_on_geo_id(pop, hh)
    df = _merge_on_geo_id(df, jobs)
    df["tract_id_synth"] = df[GEO_ID_COL].str[9:20]
    df["block_id_synth"] = df[GEO_ID_COL].str[9:24]

    _fill_numeric_only(df)  # 🔄 **replaces previous fillna(0)**
    return df


# -----------------------------------------------------------------------------
# TRACT-LEVEL BUILD & DERIVATIONS
# -----------------------------------------------------------------------------
def _derive_income(df: pd.DataFrame) -> pd.DataFrame:
    bands = [
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
    df["low_income"] = df[bands].sum(axis=1)
    df["perc_low_income"] = df["low_income"] / df["total_hh"]
    df.drop(columns="total_hh", inplace=True)
    return df


def _derive_ethnicity(df: pd.DataFrame) -> pd.DataFrame:
    minority = ["black", "native", "asian", "pac_isl", "other", "multi"]
    df["minority"] = df[minority].sum(axis=1)
    df["perc_minority"] = df["minority"] / df["total_pop"]
    df.drop(columns="total_pop", inplace=True)
    return df


def _derive_language(df: pd.DataFrame) -> pd.DataFrame:
    lep_cols = [c for c in df.columns if c.endswith("_engnwell")]
    df[lep_cols] = df[lep_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    df["all_nwell"] = df[lep_cols].sum(axis=1)
    df["perc_lep"] = (df["all_nwell"] / df["total_lang_pop"]).fillna(0).round(3)
    return df


def _derive_vehicle(df: pd.DataFrame) -> pd.DataFrame:
    df["all_lo_veh_hh"] = df[["veh_0_all_hh", "veh_1_all_hh"]].sum(axis=1)
    df["perc_lo_veh"] = df["all_lo_veh_hh"] / df["all_hhs"]
    df["perc_0_veh"] = df["veh_0_all_hh"] / df["all_hhs"]
    df["perc_1_veh"] = df["veh_1_all_hh"] / df["all_hhs"]
    df["perc_veh_1_hh_1"] = df["veh_1_hh_1"] / df["all_hhs"]
    df["perc_lo_veh_mod"] = (df["perc_lo_veh"] - df["perc_veh_1_hh_1"]).round(3)
    return df


def _derive_age(df: pd.DataFrame) -> pd.DataFrame:
    youth = ["m_15_17", "f_15_17", "m_18_19", "f_18_19", "m_20", "f_20", "m_21", "f_21"]
    elderly = [
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
    df["all_youth"] = df[[c for c in youth if c in df]].sum(axis=1)
    df["all_elderly"] = df[[c for c in elderly if c in df]].sum(axis=1)
    if "total_pop" in df.columns:
        df["perc_youth"] = (df["all_youth"] / df["total_pop"]).round(3)
        df["perc_elderly"] = (df["all_elderly"] / df["total_pop"]).round(3)
        df.drop(columns="total_pop", inplace=True)
    return df


@dataclass(slots=True)
class _TractInputs:
    income_files: list[str]
    ethnicity_files: list[str]
    language_files: list[str]
    vehicle_files: list[str]
    age_files: list[str]


def _build_tract_df(inp: _TractInputs) -> pd.DataFrame:  # noqa: D401
    """Return a tract-level DataFrame of optional socio-economic measures."""
    dfs: list[pd.DataFrame] = []

    # ----- Income -----------------------------------------------------------
    if inp.income_files:
        income = _load_and_concat(
            inp.income_files,
            skiprows=[1],
            rename={
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
        )
        dfs.append(_derive_income(income))

    # ----- Ethnicity --------------------------------------------------------
    if inp.ethnicity_files:
        ethnicity = _load_and_concat(
            inp.ethnicity_files,
            skiprows=[1],
            rename={
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
        )
        dfs.append(_derive_ethnicity(ethnicity))

    # ----- Language Proficiency --------------------------------------------
    if inp.language_files:
        language = _load_and_concat(
            inp.language_files,
            skiprows=[1],
            rename={
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
        )
        dfs.append(_derive_language(language))

    # ----- Vehicle Ownership -----------------------------------------------
    if inp.vehicle_files:
        vehicle = _load_and_concat(
            inp.vehicle_files,
            skiprows=[1],
            rename={
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
        )
        dfs.append(_derive_vehicle(vehicle))

    # ----- Age --------------------------------------------------------------
    if inp.age_files:
        age = _load_and_concat(
            inp.age_files,
            skiprows=[1],
            rename={
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
        )
        dfs.append(_derive_age(age))

    if not dfs:
        return pd.DataFrame()

    merged = dfs[0]
    for optional in dfs[1:]:
        merged = _merge_on_geo_id(merged, optional)

    _fill_numeric_only(merged)
    merged["tract_id_clean"] = merged[GEO_ID_COL].str[9:]
    return merged


# -----------------------------------------------------------------------------
# FIPS HELPERS
# -----------------------------------------------------------------------------

def _ensure_fips_column(
    df: pd.DataFrame,
    *,
    dst: str = "FIPS",
    geo_candidates: tuple[str, ...] = ("GEO_ID", "GEO_ID_blk", "GEO_ID_trt"),
    start: int = 9,
    end: int = 14,
) -> None:
    """Create a 5-digit county FIPS column *in-place*, selecting the first GEO_ID.

    Args:
        df: DataFrame to mutate.
        dst: Destination column name for the extracted FIPS code.
        geo_candidates: Ordered candidate source columns containing GEO_ID-style
            strings. The first match present in *df* is used.
        start: Start index (inclusive) of the FIPS slice within the GEO_ID.
        end: End index (exclusive) of the FIPS slice within the GEO_ID.

    Raises:
        KeyError: If none of *geo_candidates* are found in *df*.
    """
    if dst in df.columns:
        return
    source = next((c for c in geo_candidates if c in df.columns), None)
    if not source:
        raise KeyError(f"No GEO_ID column found among {geo_candidates}")
    df[dst] = df[source].astype(str).str[start:end]


def _apply_fips_filter(
    df: pd.DataFrame,
    *,
    fips: Iterable[str] | None = None,
    dst_col: str = "FIPS",
) -> pd.DataFrame:
    """Return a copy of *df* filtered to the requested county FIPS list.

    Args:
        df: Input DataFrame.
        fips: Iterable of 5-digit county FIPS codes. If ``None`` or empty,
            the function is a no-op and returns *df* unchanged.
        dst_col: Column containing the county FIPS code (created if missing).

    Returns:
        Filtered DataFrame (new copy) or the original *df* if *fips* is empty.
    """
    if not fips:
        return df
    _ensure_fips_column(df, dst=dst_col)
    wanted = {str(code).zfill(5) for code in fips}
    return df[df[dst_col].isin(wanted)].copy()


# -----------------------------------------------------------------------------
# PUBLIC API
# -----------------------------------------------------------------------------

def build_joined_table(
    *,
    pop_files: list[str],
    hh_files: list[str],
    jobs_files: list[str],
    income_files: list[str] | None = None,
    ethnicity_files: list[str] | None = None,
    language_files: list[str] | None = None,
    vehicle_files: list[str] | None = None,
    age_files: list[str] | None = None,
    county_fips_filter: Iterable[str] | None = None,
    _clean_columns: bool = True,
) -> pd.DataFrame:
    """Return a fully-joined **block + tract** DataFrame with optional FIPS filter.

    Args:
        pop_files, hh_files, jobs_files: Block-level Census file paths.
        income_files, ethnicity_files, language_files, vehicle_files, age_files:
            Optional lists of tract-level file paths.
        county_fips_filter: Optional iterable of 5-digit county FIPS codes to
            retain in the final output.
        _clean_columns: If ``True`` (default) drop any residual Census code
            columns that were not renamed.

    Returns:
        A DataFrame containing the merged block and tract records, cleaned,
        NaNs filled in numeric columns, and (optionally) filtered by county.
    """
    block_df = _build_block_df(_BlockInputs(pop_files, hh_files, jobs_files))
    tract_df = _build_tract_df(
        _TractInputs(
            income_files or [],
            ethnicity_files or [],
            language_files or [],
            vehicle_files or [],
            age_files or [],
        )
    )

    # ----- Merge block ↔ tract ---------------------------------------------
    if tract_df.empty:
        combined = block_df
    else:
        combined = block_df.merge(
            tract_df,
            left_on="tract_id_synth",
            right_on="tract_id_clean",
            how="outer",
            suffixes=("_blk", "_trt"),
        )

    # ----- Column cleanup + FIPS filter -------------------------------------
    if _clean_columns:
        combined = _drop_unfriendly_cols(combined)

    combined = _apply_fips_filter(combined, fips=county_fips_filter)
    _fill_numeric_only(combined)
    return combined


__all__ = ["build_joined_table"]

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Orchestrate join using CONFIGURATION paths and, optionally, write CSV."""
    LOGGER.info("Building joined table from configured paths …")

    df_joined = build_joined_table(
        pop_files=POP_FILES,
        hh_files=HH_FILES,
        jobs_files=JOBS_FILES,
        income_files=INCOME_FILES,
        ethnicity_files=ETHNICITY_FILES,
        language_files=LANGUAGE_FILES,
        vehicle_files=VEHICLE_FILES,
        age_files=AGE_FILES,
        county_fips_filter=COUNTY_FIPS_FILTER,
    )
    LOGGER.info("Created DataFrame with shape %s", df_joined.shape)

    if CSV_OUTPUT_PATH:
        out_path = Path(CSV_OUTPUT_PATH)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_joined.to_csv(out_path, index=False)
        LOGGER.info("CSV written to %s", out_path.resolve())


if __name__ == "__main__":
    main()
