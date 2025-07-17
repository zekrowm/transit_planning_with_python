"""Join Census block- and tract-level data into a unified DataFrame.

This module discovers, reads, and merges demographic and employment datasets
from the U.S. Census and LODES, based on GEO_ID alignment. Output includes
population, household counts, job totals, income brackets, ethnicity, language
proficiency, vehicle availability, and age group statistics.

Supports input as CSV, GZ, or ZIP files (containing '-Data.csv'), and can filter
by county FIPS codes. Output may be saved as a flat CSV.

Helpful links:
    https://data.census.gov/table
    https://lehd.ces.census.gov/data/
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Hashable, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

#: Folder that holds every Census download (plain CSV, *.csv.gz*, or ZIPs).
#: Sub-directories are searched automatically.
ROOT_DATA_DIR: str | Path = r"Path\To\Your\Census_Table_Data_Files"          # <<< EDIT ME

#: Optional output CSV (set to None to skip writing)
CSV_OUTPUT_PATH: str | None = r"Path\To\Your\Output_Folder\joined_blocks.csv"

#: Optional county FIPS filter (5-digit codes, e.g. ["11001", "51059"])
COUNTY_FIPS_FILTER: list[str] = [
    "11001", "24031", "24033",
    "51683", "51685", "51059",
    "51013", "51510", "51600",
    "51610", "51107", "51153",
]

# ----------------------------------------------------------------------------- 
# Signatures that map a file name to a *topic* variable.  ALL tokens listed for
# a topic must appear in the file name (case-insensitive).
TOPIC_SIGNATURES: dict[str, Sequence[str] | str] = {
    "POP_FILES":        ("P1",),
    "HH_FILES":         ("H9",),
    "JOBS_FILES":       ("_S000_JT00_",),   # LODES WAC
    "INCOME_FILES":     ("B19001",),
    "ETHNICITY_FILES":  ("P9",),
    "LANGUAGE_FILES":   ("C16001",),
    "VEHICLE_FILES":    ("B08201",),
    "AGE_FILES":        ("B01001",),
}

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s :: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

# =============================================================================
# FUNCTIONS
# =============================================================================

def _token_match(name: str, tokens: Sequence[str] | str) -> bool:
    """Return *True* if **all** tokens occur in *name* (case-insensitive)."""
    if isinstance(tokens, str):
        tokens = (tokens,)
    low = name.lower()
    return all(tok.lower() in low for tok in tokens)


def discover_census_files(
    root_dir: str | Path,
    signatures: Mapping[str, Sequence[str] | str] = TOPIC_SIGNATURES,
) -> dict[str, list[str]]:
    """Recursively locate Census “data” files and bucket them by topic.

    * Accepts plain **CSV**, **CSV.GZ**, or **ZIP** archives.
    * ZIPs are returned as the ZIP path itself – content is handled later.
    * File order is sorted for determinism.

    Returns
    -------
    dict[str, list[str]]
        Keys mirror *signatures* and align with downstream variable names.
    """
    buckets: dict[str, list[str]] = {k: [] for k in signatures}
    root = Path(root_dir).expanduser().resolve()

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if not path.name.lower().endswith(("-data.csv", ".csv.gz", ".zip")):
            continue

        for var, sig in signatures.items():
            if _token_match(path.name, sig):
                buckets[var].append(str(path))
                break

    for lst in buckets.values():
        lst.sort()
    return buckets


def _read_csv_any(path: str | Path, **read_kwargs) -> pd.DataFrame:
    """Read a CSV/CSV.GZ directly *or* the first “-Data.csv” member in a ZIP."""
    p = Path(path)
    suf = p.suffix.lower()

    if suf == ".zip":
        with zipfile.ZipFile(p) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith("-data.csv")]
            if not members:
                raise FileNotFoundError(f"No '*-Data.csv' inside {p}")
            with zf.open(members[0]) as fh, io.TextIOWrapper(fh, encoding="utf-8") as txt:
                return pd.read_csv(txt, **read_kwargs)

    return pd.read_csv(p, **read_kwargs)

# -----------------------------------------------------------------------------
# DATA-PROCESSING CONSTANTS & REGEXES
# -----------------------------------------------------------------------------

GEO_ID_COL = "GEO_ID"
_UNFRIENDLY_COL_RE = re.compile(r"^[A-Z]{2,}\d{3,}.*")

def _fill_numeric_only(df: pd.DataFrame, value: int | float = 0) -> pd.DataFrame:
    """Replace *only* numeric NaNs with *value*; leave object columns untouched."""
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(value)
    return df


def _clean_name_cols(df: pd.DataFrame) -> None:
    """Sanitise NAME‑like columns in place (remove CR/LF/TAB)."""
    for col in df.filter(regex=r"^NAME").columns:
        df[col] = (
            df[col]                       # Series
            .astype(str)                  # ensure string dtype
            .str.replace(r"[\r\n\t]+", " ", regex=True)  # collapse control chars
            .str.strip()                  # trim leading/trailing spaces
        )


def _load_and_concat(
    files: Sequence[str],
    *,
    skiprows: int | Sequence[int] | Callable[[int], bool] | None = None,
    dtype: Mapping[Hashable, str | np.dtype[Any]] | None = None,
    usecols: Sequence[Hashable] | None = None,
    rename: Mapping[str, str] | None = None,
    compression: Literal["infer", "gzip", "bz2", "zip", "xz", "zstd"] | None = None,
) -> pd.DataFrame:
    """Read multiple Census CSV / CSV‑GZ / ZIP files and concatenate the results.

    Embedded control characters in *NAME* columns are stripped immediately to
    guarantee that every logical record remains on a single physical line when
    the final DataFrame is exported.

    Parameters
    ----------
    files :
        Paths to source files.
    skiprows, dtype, usecols, rename, compression :
        Passed straight through to :func:`pandas.read_csv`; see pandas docs.

    Returns
    -------
    pd.DataFrame
        Concatenated frame (empty if *files* is empty).

    Notes
    -----
    * ZIP archives are handled transparently via ``_read_csv_any``.
    * Column renaming occurs **before** we prune columns via *usecols*
      (unless *usecols* is explicitly supplied).
    """
    frames: list[pd.DataFrame] = []

    for path in files:
        read_kwargs: dict[str, Any] = {"compression": compression}
        if skiprows is not None:
            read_kwargs["skiprows"] = skiprows
        if dtype is not None:
            read_kwargs["dtype"] = dtype
        if usecols is not None:
            read_kwargs["usecols"] = usecols

        # --- read, then sanitise ---
        df = _read_csv_any(path, **read_kwargs)
        _clean_name_cols(df)  # ← fix #1 applied here

        # --- optional column renaming & pruning ---
        if rename:
            df.rename(columns=rename, inplace=True)
            if usecols is None:
                keep = {GEO_ID_COL, "NAME", *rename.values()}
                df = df.loc[:, df.columns.intersection(keep)]

        frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _merge_on_geo_id(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Outer-merge two frames on GEO_ID, dropping duplicate columns."""
    if left.empty:
        return right.copy()
    if right.empty:
        return left.copy()

    dup = (set(left.columns) & set(right.columns)) - {GEO_ID_COL}
    return left.merge(right.drop(columns=dup), on=GEO_ID_COL, how="outer")


def _drop_unfriendly_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Remove any column that still looks like a raw Census code."""
    to_drop = [c for c in df.columns if _UNFRIENDLY_COL_RE.match(c)]
    return df.drop(columns=to_drop, errors="ignore")


# -----------------------------------------------------------------------------
# BLOCK-LEVEL BUILD
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class _BlockInputs:
    pop_files: list[str]
    hh_files: list[str]
    jobs_files: list[str]


def _build_block_df(inp: _BlockInputs) -> pd.DataFrame:
    """Return a block-level DataFrame with population, households, and jobs."""
    pop = _load_and_concat(
        inp.pop_files,
        skiprows=[1],
        rename={"P1_001N": "total_pop"},
        usecols=[GEO_ID_COL, "NAME", "P1_001N"],
    )
    hh = _load_and_concat(
        inp.hh_files,
        skiprows=[1],
        rename={"H9_001N": "total_hh"},
        usecols=[GEO_ID_COL, "H9_001N"],
        dtype={"H9_001N": "Int64"},
    )
    jobs = _load_and_concat(
        inp.jobs_files,
        rename={
            "C000": "tot_empl",
            "CE01": "low_wage",
            "CE02": "mid_wage",
            "CE03": "high_wage",
        },
        usecols=["w_geocode", "C000", "CE01", "CE02", "CE03"],
    )
    if not jobs.empty:
        jobs[GEO_ID_COL] = "1000000US" + jobs["w_geocode"].astype(str)
        jobs.drop(columns="w_geocode", inplace=True)

    df = _merge_on_geo_id(pop, hh)
    df = _merge_on_geo_id(df, jobs)
    df["tract_id_synth"] = df[GEO_ID_COL].str[9:20]
    df["block_id_synth"] = df[GEO_ID_COL].str[9:24]

    _fill_numeric_only(df)
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
    youth = [
        "m_15_17",
        "f_15_17",
        "m_18_19",
        "f_18_19",
        "m_20",
        "f_20",
        "m_21",
        "f_21",
    ]
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


def _build_tract_df(inp: _TractInputs) -> pd.DataFrame:
    """Return a tract-level DataFrame of optional socio-economic measures."""
    dfs: list[pd.DataFrame] = []

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
    """Create a 5-digit county FIPS column *in-place* from the first GEO_ID."""
    if dst in df.columns:
        return
    source = next((c for c in geo_candidates if c in df.columns), None)
    if source is None:
        raise KeyError(f"No GEO_ID column found among {geo_candidates}")
    df[dst] = df[source].astype(str).str[start:end]


def _apply_fips_filter(
    df: pd.DataFrame,
    *,
    fips: Iterable[str] | None = None,
    dst_col: str = "FIPS",
) -> pd.DataFrame:
    """Return a copy filtered to *fips* (or unchanged if *fips* is empty/None)."""
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
    """Return a fully joined block + tract DataFrame with optional FIPS filter."""
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

    combined = (
        block_df
        if tract_df.empty
        else block_df.merge(
            tract_df,
            left_on="tract_id_synth",
            right_on="tract_id_clean",
            how="outer",
            suffixes=("_blk", "_trt"),
        )
    )

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
    """Orchestrate discovery, join, and optional CSV export."""
    LOGGER.info("Discovering Census datasets under %s …", ROOT_DATA_DIR)
    discovered = discover_census_files(ROOT_DATA_DIR)

    df_joined = build_joined_table(
        pop_files=discovered["POP_FILES"],
        hh_files=discovered["HH_FILES"],
        jobs_files=discovered["JOBS_FILES"],
        income_files=discovered["INCOME_FILES"],
        ethnicity_files=discovered["ETHNICITY_FILES"],
        language_files=discovered["LANGUAGE_FILES"],
        vehicle_files=discovered["VEHICLE_FILES"],
        age_files=discovered["AGE_FILES"],
        county_fips_filter=COUNTY_FIPS_FILTER,
    )
    LOGGER.info("Created DataFrame with shape %s", df_joined.shape)

    if CSV_OUTPUT_PATH:
        out_path = Path(CSV_OUTPUT_PATH).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_joined.to_csv(out_path, index=False)
        LOGGER.info("CSV written to %s", out_path)


if __name__ == "__main__":
    main()
