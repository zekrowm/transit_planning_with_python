"""End-to-end ArcPy pipeline: merge Census CSVs, merge TIGER shapefiles, and join to final output.

Overview
--------
This single script performs three stages to produce a final feature class / shapefile
containing Census block geometry joined to a consolidated attribute table
(population, households, jobs, income, ethnicity, language, vehicles, age).

Stages:
    1) CSV stage (pandas): discover, read, and merge Census + LODES CSV inputs into
       a single attribute table (one row per block). Optional tract-level inputs
       are merged to blocks on tract ID.
    2) TIGER stage (ArcPy): discover, merge, and (optionally) FIPS-filter TIGER/Line
       tabblock shapefiles into a single polygon feature class.
    3) Join stage (ArcPy): join the attribute CSV from step 1 to the block polygons
       from step 2 using a normalized 15-digit block identifier, then write final output.

Notes:
-----
* ZIPped TIGER shapefiles are NOT read directly—unzip first.
* All TIGER inputs should be in the same coordinate system.
* Shapefile outputs will truncate field names to 10 chars; use a FileGDB to retain names.
* The CSV merger expects files that match the table codes configured below (e.g., P1, H9, etc.).
* GEO_ID vs GEOID differences are handled; GEO_ID is synthesized from GEOID if needed.

Helpful links
-------------
    Demographic data: https://data.census.gov/table
    Jobs data:        https://lehd.ces.census.gov/data/
    Geographic data:  https://www.census.gov/cgi-bin/geo/shapefiles/index.php
"""

from __future__ import annotations

import fnmatch
import io
import logging
import os
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final, Hashable, Iterable, Literal, Mapping, Sequence

import arcpy
import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# ---- Input roots ----
INPUT_CSV_DIR: str | Path = r"Folder\Path\To\Your\input_csvs"         # <<< EDIT ME
INPUT_SHP_DIR: str | Path = r"Folder\Path\To\Your\input_shps"         # <<< EDIT ME

# ---- TIGER shapefile discovery ----
# Glob pattern to select block shapefiles (basenames). Example: "tl_2023_*_tabblock20.shp"
TIGER_INPUT_GLOB: str = "tl_*_*_tabblock20.shp" # Edit if using block group or tract geometry

# Optional FIPS filter (5-char county codes)
# Leave empty to keep all.
# Add codes for your counties of interest to filter.
FIPS_TO_FILTER: list[str] = [
#    "11001",
#    "24031", "24033",
#    "51683", "51685", "51059", "51013", "51510", "51600", "51610", "51107", "51153",
]

# ---- Outputs ----
INTERMEDIATE_MERGED_SHP: str = r"File\Path\To\Your\output_intermediate\merged_blocks.shp"       # or ...gdb\merged_blocks
INTERMEDIATE_COMBINED_CSV: str = r"File\Path\To\Your\output_intermediate\combined_blocks.csv"
FINAL_JOINED_FEATURES: str = r"File\Path\To\Your\output_final\blocks_with_attrs.shp"     # or ...gdb\blocks_with_attrs

# ---- CSV topic signatures ----
TOPIC_SIGNATURES: dict[str, Sequence[str] | str] = {
    "POP_FILES": ("P1",),
    "HH_FILES": ("H9",),
    "JOBS_FILES": ("_S000_JT00_",),  # LODES WAC
    "INCOME_FILES": ("B19001",),
    "ETHNICITY_FILES": ("P9",),
    "LANGUAGE_FILES": ("C16001",),
    "VEHICLE_FILES": ("B08201",),
    "AGE_FILES": ("B01001",),
}

# ---- ArcPy/Join settings ----
FIPS_FIELD_NAME: str = "FIPS"
STATE_CANDIDATES: tuple[str, ...] = ("STATEFP20", "STATEFP")
COUNTY_CANDIDATES: tuple[str, ...] = ("COUNTYFP20", "COUNTYFP")
LEFT_KEY: Final[str] = "GEOID20"      # block GEOID in geometry
RIGHT_KEY: Final[str] = "GEO_ID"      # preferred ID in CSV (24-char, rightmost 15 used)
DERIVATION_SRC: Final[str] = "GEO_ID_blk"  # fallback CSV column
USE_IN_MEMORY: bool = True
FORCE_FLOAT: bool = False
arcpy.env.overwriteOutput = True

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s :: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def _gp(msg: str, level: str = "info") -> None:
    """Emit a message to ArcGIS geoprocessor console and Python logger."""
    if level == "warning":
        arcpy.AddWarning(msg)
    elif level == "error":
        arcpy.AddError(msg)
    else:
        arcpy.AddMessage(msg)


# =============================================================================
# FUNCTIONS
# =============================================================================
# -----------------------------------------------------------------------------
# CSV MERGE (pandas)
# -----------------------------------------------------------------------------

GEO_ID_COL = "GEO_ID"
_UNFRIENDLY_COL_RE = re.compile(r"^[A-Z]{2,}\d{3,}.*")


def _token_match(name: str, tokens: Sequence[str] | str) -> bool:
    """Return True if *all* tokens occur in *name* (case-insensitive)."""
    if isinstance(tokens, str):
        tokens = (tokens,)
    low = name.lower()
    return all(tok.lower() in low for tok in tokens)


def discover_census_files(
    root_dir: str | Path,
    signatures: Mapping[str, Sequence[str] | str] = TOPIC_SIGNATURES,
) -> dict[str, list[str]]:
    """Recursively locate Census “data” CSV/ZIP and bucket them by topic."""
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


def _read_csv_any(path: str | Path, **read_kwargs: Any) -> pd.DataFrame:
    """Read a CSV/CSV.GZ directly or the first “-Data.csv” member in a ZIP."""
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


def _fill_numeric_only(df: pd.DataFrame, value: int | float = 0) -> pd.DataFrame:
    """Replace only numeric NaNs with *value*; leave object columns untouched."""
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(value)
    return df


def _clean_name_cols(df: pd.DataFrame) -> None:
    """Sanitize NAME-like columns in place (remove CR/LF/TAB)."""
    for col in df.filter(regex=r"^NAME").columns:
        df[col] = df[col].astype(str).str.replace(r"[\r\n\t]+", " ", regex=True).str.strip()


def _load_and_concat(
    files: Sequence[str],
    *,
    skiprows: int | Sequence[int] | Callable[[int], bool] | None = None,
    dtype: Mapping[Hashable, str | np.dtype[Any]] | None = None,
    usecols: Sequence[Hashable] | Callable[[Hashable], bool] | None = None,
    rename: Mapping[str, str] | None = None,
    compression: Literal["infer", "gzip", "bz2", "zip", "xz", "zstd"] | None = None,
) -> pd.DataFrame:
    """Read multiple Census CSV / CSV-GZ / ZIP files and concatenate the results."""
    frames: list[pd.DataFrame] = []

    for path in files:
        read_kwargs: dict[str, Any] = {}
        if compression is not None:
            read_kwargs["compression"] = compression
        if skiprows is not None:
            read_kwargs["skiprows"] = skiprows
        if dtype is not None:
            read_kwargs["dtype"] = dtype

        if usecols is not None and not callable(usecols):
            wanted = set(usecols)
            read_kwargs["usecols"] = lambda c: c in wanted  # type: ignore[arg-type]
        elif callable(usecols):
            read_kwargs["usecols"] = usecols

        df = _read_csv_any(path, **read_kwargs)
        _clean_name_cols(df)

        if rename:
            df.rename(columns=rename, inplace=True)
            if not callable(usecols) and usecols is None:
                keep = {GEO_ID_COL, "GEOID", "NAME", *rename.values()}
                df = df.loc[:, df.columns.intersection(keep)]

        frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _ensure_geo_id(df: pd.DataFrame) -> None:
    """Ensure GEO_ID exists; derive from GEOID if needed (block-level assumption)."""
    if GEO_ID_COL in df.columns:
        return
    if "GEOID" in df.columns:
        df["GEOID"] = df["GEOID"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(15)
        df[GEO_ID_COL] = "1000000US" + df["GEOID"]


def _add_geo_derivatives(df: pd.DataFrame) -> None:
    """Create tract/block synthetic ids in-place from whichever GEO column exists."""
    if "tract_id_synth" in df.columns and "block_id_synth" in df.columns:
        return

    if GEO_ID_COL in df.columns:
        s = df[GEO_ID_COL].astype(str)
        df["tract_id_synth"] = s.str[9:20]
        df["block_id_synth"] = s.str[9:24]
        return

    if "GEOID" in df.columns:
        s = df["GEOID"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(15)
        df["tract_id_synth"] = s.str[:11]
        df["block_id_synth"] = s.str[:15]
        return

    raise KeyError("Neither GEO_ID nor GEOID present to build synthetic IDs.")


def _merge_on_geo_id(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Outer-merge two frames on GEO_ID, dropping duplicate columns."""
    if left.empty:
        return right.copy()
    if right.empty:
        return left.copy()

    _ensure_geo_id(left)
    _ensure_geo_id(right)

    if GEO_ID_COL not in left.columns or GEO_ID_COL not in right.columns:
        raise KeyError("Cannot merge: GEO_ID not present in one or both frames.")

    dup = (set(left.columns) & set(right.columns)) - {GEO_ID_COL}
    return left.merge(right.drop(columns=dup, errors="ignore"), on=GEO_ID_COL, how="outer")


def _drop_unfriendly_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Remove any column that still looks like a raw Census code."""
    to_drop = [c for c in df.columns if _UNFRIENDLY_COL_RE.match(c)]
    return df.drop(columns=to_drop, errors="ignore")


@dataclass
class _BlockInputs:
    """Container for block-level input file lists."""

    __slots__ = ("pop_files", "hh_files", "jobs_files")
    pop_files: list[str]
    hh_files: list[str]
    jobs_files: list[str]


def _build_block_df(inp: _BlockInputs) -> pd.DataFrame:
    """Return a block-level DataFrame with population, households, and jobs."""
    pop = _load_and_concat(
        inp.pop_files,
        skiprows=[1],
        rename={"P1_001N": "total_pop"},
        usecols=["GEO_ID", "GEOID", "NAME", "P1_001N"],
    )

    hh = _load_and_concat(
        inp.hh_files,
        skiprows=[1],
        rename={"H9_001N": "total_hh"},
        usecols=["GEO_ID", "GEOID", "H9_001N"],
        dtype={"H9_001N": "Int64"},
    )

    jobs = _load_and_concat(
        inp.jobs_files,
        rename={"C000": "tot_empl", "CE01": "low_wage", "CE02": "mid_wage", "CE03": "high_wage"},
        usecols=["w_geocode", "C000", "CE01", "CE02", "CE03"],
    )
    if not jobs.empty:
        geostr = jobs["w_geocode"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(15)
        jobs["GEOID"] = geostr
        jobs[GEO_ID_COL] = "1000000US" + geostr
        jobs.drop(columns="w_geocode", inplace=True)

    df = _merge_on_geo_id(pop, hh)
    df = _merge_on_geo_id(df, jobs)

    _add_geo_derivatives(df)
    _fill_numeric_only(df)
    return df


@dataclass
class _TractInputs:
    """Container for tract-level optional input file lists."""

    __slots__ = ("income_files", "ethnicity_files", "language_files", "vehicle_files", "age_files")
    income_files: list[str]
    ethnicity_files: list[str]
    language_files: list[str]
    vehicle_files: list[str]
    age_files: list[str]


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
    else:
        LOGGER.info("Optional: INCOME_FILES not found; skipping income derivations.")

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
    else:
        LOGGER.info("Optional: ETHNICITY_FILES not found; skipping ethnicity derivations.")

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
    else:
        LOGGER.info("Optional: LANGUAGE_FILES not found; skipping language derivations.")

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
    else:
        LOGGER.info("Optional: VEHICLE_FILES not found; skipping vehicle derivations.")

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
    else:
        LOGGER.info("Optional: AGE_FILES not found; skipping age derivations.")

    if not dfs:
        return pd.DataFrame()

    merged = dfs[0]
    for optional in dfs[1:]:
        merged = _merge_on_geo_id(merged, optional)

    _fill_numeric_only(merged)
    merged["tract_id_clean"] = merged[GEO_ID_COL].astype(str).str[9:]
    return merged


def _make_shapefile_safe_columns(
    df: pd.DataFrame,
    *,
    preferred_map: Mapping[str, str] | None = None,
    preserve: Iterable[str] = (GEO_ID_COL, "GEOID", "NAME"),
    verify: bool = True,
) -> pd.DataFrame:
    """Return a copy of *df* with ≤10-char, unique, shapefile-safe column names.

    The function applies a preferred explicit mapping first (for human-readable,
    stable names), then deterministically shortens any remaining columns longer
    than 10 characters using an 8-character base + 2-character checksum scheme.
    Columns listed in *preserve* are never renamed.

    Args:
        df: Input DataFrame.
        preferred_map: Mapping from long → preferred short names (≤10 chars).
        preserve: Column names that must not be renamed.
        verify: If True, validates (a) max length ≤10, (b) uniqueness.

    Returns:
        A new DataFrame with renamed columns.

    Raises:
        ValueError: If verification fails (length >10 or duplicate names).
    """
    import hashlib
    import re

    old_cols = list(df.columns)
    preserve_set = set(preserve)
    pref_map = dict(preferred_map or {})

    # Sanity check the explicit map
    for k, v in pref_map.items():
        if len(v) > 10:
            raise ValueError(f"Preferred short name '{v}' for '{k}' exceeds 10 chars.")

    def _auto_name(name: str, used: set[str]) -> str:
        """Create a deterministic ≤10-char label using 8+2 scheme."""
        # Normalize base: uppercase, alnum only
        base = re.sub(r"[^A-Za-z0-9]", "", name.upper())
        if not base:
            base = "COL"
        base8 = base[:8]
        # 2-char checksum for collision resistance (md5 hex tail)
        chk = hashlib.md5(name.encode("utf-8")).hexdigest()[-2:].upper()
        candidate = f"{base8}{chk}"
        # If still collides, perturb by appending an index (last char)
        if candidate in used:
            for i in range(36):  # 0-9A-Z
                tail = (str(i) if i < 10 else chr(ord("A") + i - 10))
                cand2 = f"{base8[:7]}{tail}{chk}"
                if cand2 not in used:
                    return cand2
            # Fallback: brute-force add hash prefix
            for i in range(256):
                cand3 = f"{base8[:6]}{i:02X}"
                if cand3 not in used:
                    return cand3
        return candidate

    new_names: dict[str, str] = {}
    used: set[str] = set()

    # First pass: apply explicit mapping and preserve list
    for col in old_cols:
        if col in preserve_set:
            new = col
        elif col in pref_map:
            new = pref_map[col]
        else:
            new = col  # possibly >10; will fix in second pass
        # Track, but allow duplicates for now—will resolve after second pass
        new_names[col] = new

    # Second pass: auto-shorten any >10 and fix collisions
    for col, proposed in list(new_names.items()):
        new = proposed
        if col not in preserve_set:
            if len(new) > 10:
                new = _auto_name(col, used)
            # Ensure uniqueness
            if new in used:
                if new == proposed:
                    # Derive a fresh auto-name if explicit/proposed collides
                    new = _auto_name(f"{col}_dup", used)
                else:
                    # Proposed was different but collided; also regenerate
                    new = _auto_name(f"{col}_dup", used)
        new_names[col] = new
        used.add(new)

    out = df.rename(columns=new_names).copy()

    if verify:
        too_long = [c for c in out.columns if len(c) > 10]
        if too_long:
            raise ValueError(f"Columns exceed 10 chars after rename: {too_long}")
        if len(set(out.columns)) != len(out.columns):
            raise ValueError("Duplicate column names produced after rename.")

    return out


def build_joined_table_from_folder(
    csv_root: str | Path,
    county_fips_filter: Iterable[str] | None = None,
    _clean_columns: bool = True,
) -> pd.DataFrame:
    """Discover CSVs under *csv_root* and produce a fully joined attributes table.

    This version applies an explicit short-name mapping followed by a deterministic
    auto-shortening for any remaining long labels, ensuring shapefile safety (≤10 chars).
    """
    # Explicit, human-readable preferred names (≤10)
    SHORT_NAMES: dict[str, str] = {
        # Core synthetic IDs (later renamed; join uses long names internally)
        "tract_id_synth": "TRCT_ID_S",
        "block_id_synth": "BLK_ID_S",
        "tract_id_clean": "TRCT_ID_C",
        "GEO_ID_blk": "GEO_ID_blk",  # already 10

        # Block-level totals
        "total_pop": "POP_TOT",
        "total_hh": "HH_TOT",
        "tot_empl": "EMP_TOT",
        "low_wage": "EMP_LO",
        "mid_wage": "EMP_MD",
        "high_wage": "EMP_HI",

        # Income (tract-derived)
        "low_income": "HH_LOWINC",
        "perc_low_income": "PCT_LOWINC",

        # Ethnicity (tract-derived)
        "minority": "MINOR_CNT",
        "perc_minority": "PCT_MINOR",

        # Language / LEP (tract-derived)
        "total_lang_pop": "LANG_TOT",
        "spanish_engnwell": "SPA_NWELL",
        "frenchetc_engnwell": "FRN_NWELL",
        "germanetc_engnwell": "GER_NWELL",
        "slavicetc_engnwell": "SLA_NWELL",
        "indoeuroetc_engnwell": "IND_NWELL",
        "korean_engnwell": "KOR_NWELL",
        "chineseetc_engnwell": "CHN_NWELL",
        "vietnamese_engnwell": "VIE_NWELL",
        "asiapacetc_engnwell": "ASP_NWELL",
        "arabic_engnwell": "ARA_NWELL",
        "otheretc_engnwell": "OTH_NWELL",
        "all_nwell": "LEP_CNT",
        "perc_lep": "PCT_LEP",

        # Vehicles (tract-derived)
        "all_hhs": "HHS_ALL",
        "veh_0_all_hh": "HH0_ALL",
        "veh_1_all_hh": "HH1_ALL",
        "veh_0_hh_1": "HH1_V0",
        "veh_1_hh_1": "HH1_V1",
        "veh_0_hh_2": "HH2_V0",
        "veh_1_hh_2": "HH2_V1",
        "veh_0_hh_3": "HH3_V0",
        "veh_1_hh_3": "HH3_V1",
        "veh_2_hh_3": "HH3_V2",
        "veh_0_hh_4p": "HH4P_V0",
        "veh_1_hh_4p": "HH4P_V1",
        "veh_2_hh_4p": "HH4P_V2",
        "all_lo_veh_hh": "HH_LOVEH",
        "perc_lo_veh": "PCT_LOVEH",
        "perc_0_veh": "PCT_VEH0",
        "perc_1_veh": "PCT_VEH1",
        "perc_veh_1_hh_1": "PCT_HH1V1",
        "perc_lo_veh_mod": "PCT_LOVEM",

        # Age (tract-derived)
        "all_youth": "YOUTH_CNT",
        "all_elderly": "ELDER_CNT",
        "perc_youth": "PCT_YOUTH",
        "perc_elderly": "PCT_ELDER",
    }

    discovered = discover_census_files(csv_root, TOPIC_SIGNATURES)
    if not discovered["POP_FILES"]:
        raise RuntimeError("Required: No POP_FILES (P1) found under the CSV root.")

    block_df = _build_block_df(
        _BlockInputs(
            pop_files=discovered["POP_FILES"],
            hh_files=discovered["HH_FILES"],
            jobs_files=discovered["JOBS_FILES"],
        )
    )

    tract_df = _build_tract_df(
        _TractInputs(
            income_files=discovered["INCOME_FILES"],
            ethnicity_files=discovered["ETHNICITY_FILES"],
            language_files=discovered["LANGUAGE_FILES"],
            vehicle_files=discovered["VEHICLE_FILES"],
            age_files=discovered["AGE_FILES"],
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

    if county_fips_filter:
        tmp = combined.copy()
        if GEO_ID_COL not in tmp.columns and "GEOID" in tmp.columns:
            tmp["GEOID"] = tmp["GEOID"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(15)
            tmp[GEO_ID_COL] = "1000000US" + tmp["GEOID"]
        fips = tmp[GEO_ID_COL].astype(str).str[9:14]
        mask = fips.isin({str(x).zfill(5) for x in county_fips_filter})
        combined = combined.loc[mask].copy()

    _fill_numeric_only(combined)

    # Apply short names LAST (after all joins/filters) to avoid breaking lookups.
    combined = _make_shapefile_safe_columns(
        combined,
        preferred_map=SHORT_NAMES,
        preserve=(GEO_ID_COL, "GEOID", "NAME"),
        verify=True,
    )
    return combined


# -----------------------------------------------------------------------------
# TIGER MERGE & FILTER (ArcPy)
# -----------------------------------------------------------------------------


def discover_tiger_datasets(root_dir: str | Path, pattern: str) -> list[str]:
    """Return absolute paths to TIGER shapefiles matching *pattern* (no ZIP)."""
    root_path = Path(root_dir).expanduser().resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(f"{root_path} is not a valid directory")

    matched: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root_path):
        for name in filenames:
            if name.lower().endswith(".shp") and fnmatch.fnmatch(name, pattern):
                matched.append(str(Path(dirpath, name).resolve()))

    if not matched:
        raise FileNotFoundError(f"No shapefiles matching '{pattern}' were found under {root_path}")

    matched.sort()
    LOGGER.info("Discovered %d shapefile(s) to merge", len(matched))
    _gp(f"Discovered {len(matched)} shapefile(s) to merge.")
    return matched


def merge_shapefiles(
    shp_paths: Sequence[str],
    *,
    workspace: str | None = None,
    out_name: str = "tiger_merge_tmp",
) -> str:
    """Merge multiple shapefiles into a temporary feature class."""
    if not shp_paths:
        raise ValueError("No input shapefiles provided to merge")

    if workspace is None:
        workspace = "in_memory" if USE_IN_MEMORY else arcpy.env.scratchGDB

    arcpy.env.overwriteOutput = True
    out_fc = os.path.join(workspace, out_name)

    LOGGER.info("Merging %d input shapefile(s) → %s", len(shp_paths), out_fc)
    _gp(f"Merging {len(shp_paths)} input shapefile(s) → {out_fc}")
    arcpy.management.Merge(shp_paths, out_fc)
    return out_fc


def _find_first_existing_field(feature_class: str, candidates: Iterable[str]) -> str | None:
    """Return the first field name from candidates that exists in feature_class."""
    existing_fields = {f.name for f in arcpy.ListFields(feature_class)}
    for cand in candidates:
        if cand in existing_fields:
            return cand
    return None


def ensure_fips_field(
    feature_class: str,
    *,
    fips_field: str = FIPS_FIELD_NAME,
    state_candidates: Sequence[str] = STATE_CANDIDATES,
    county_candidates: Sequence[str] = COUNTY_CANDIDATES,
) -> None:
    """Ensure a 5-digit FIPS field exists by concatenating state + county."""
    fields = {f.name: f for f in arcpy.ListFields(feature_class)}
    if fips_field in fields:
        LOGGER.info("Field %s already present — skipping creation", fips_field)
        _gp(f"Field {fips_field} already present — skipping creation.")
        return

    state_field = _find_first_existing_field(feature_class, state_candidates)
    county_field = _find_first_existing_field(feature_class, county_candidates)
    if state_field is None or county_field is None:
        raise KeyError(
            "Required columns not found. Expected one of "
            f"{state_candidates} and one of {county_candidates}."
        )

    LOGGER.info("Adding FIPS field %s", fips_field)
    _gp(f"Adding FIPS field {fips_field}")
    arcpy.management.AddField(
        in_table=feature_class, field_name=fips_field, field_type="TEXT", field_length=5
    )

    fields_to_update = (state_field, county_field, fips_field)
    with arcpy.da.UpdateCursor(feature_class, fields_to_update) as cursor:
        for row in cursor:
            state_val = str(row[0]).zfill(2) if row[0] is not None else ""
            county_val = str(row[1]).zfill(3) if row[1] is not None else ""
            row[2] = f"{state_val}{county_val}"
            cursor.updateRow(row)

    LOGGER.info("Populated new column %s", fips_field)
    _gp(f"Populated new column {fips_field}")


def filter_by_fips(
    feature_class: str,
    fips_values: Sequence[str],
    *,
    fips_field: str = FIPS_FIELD_NAME,
    workspace: str | None = None,
    out_name: str = "tiger_merge_fips_tmp",
) -> str:
    """Optionally filter a feature class by FIPS list. Returns input if list is empty."""
    if not fips_values:
        LOGGER.info("FIPS filter empty — exporting full dataset")
        _gp("FIPS filter empty — exporting full dataset")
        return feature_class

    workspace = workspace or ("in_memory" if USE_IN_MEMORY else arcpy.env.scratchGDB)
    out_fc = os.path.join(workspace, out_name)

    sql_values = ",".join(f"'{val}'" for val in fips_values)
    where_clause = f"{arcpy.AddFieldDelimiters(feature_class, fips_field)} IN ({sql_values})"

    LOGGER.info("Selecting by FIPS → %s", where_clause)
    _gp("Selecting by FIPS...")

    arcpy.management.MakeFeatureLayer(feature_class, "fips_lyr", where_clause)
    count = int(arcpy.management.GetCount("fips_lyr").getOutput(0))
    if count == 0:
        LOGGER.warning("No features matched the FIPS list — output will be empty")
        _gp("Warning: No features matched the FIPS list — output will be empty.", "warning")

    arcpy.management.CopyFeatures("fips_lyr", out_fc)

    LOGGER.info("Selected %d feature(s)", count)
    _gp(f"Selected {count} feature(s)")
    return out_fc


def write_output(in_fc: str, out_path: str) -> None:
    """Copy final FC to the requested output path (FGDB/shapefile)."""
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    LOGGER.info("Writing output → %s", out_path)
    _gp(f"Writing output → {out_path}")
    arcpy.management.CopyFeatures(in_fc, out_path)
    _gp("Output written successfully.")
    LOGGER.info("Output written successfully")


# -----------------------------------------------------------------------------
# JOIN (ArcPy)
# -----------------------------------------------------------------------------


def _list_field_names(dataset: str) -> list[str]:
    """Return a simple list of field names on *dataset*."""
    return [f.name for f in arcpy.ListFields(dataset)]


def _field_exists(dataset: str, field_name: str) -> bool:
    """Check whether *field_name* exists on *dataset*."""
    return field_name in _list_field_names(dataset)


def _add_text_key_field(
    dataset: str,
    source_field: str,
    target_field: str,
    length: int = 15,
) -> str:
    """Add a text field and calculate it to the rightmost *length* chars.

    Casts the source to string to avoid failures when the source field is numeric
    (e.g., some TIGER GEOIDs). Uses correct Python slice syntax.

    Args:
        dataset: Path to the feature class or table.
        source_field: Existing field name to derive from.
        target_field: Name of the text field to create/populate.
        length: Number of rightmost characters to keep.

    Returns:
        The name of the target key field (same as ``target_field``).
    """
    if not _field_exists(dataset, source_field):
        raise KeyError(f"'{source_field}' not found in {dataset}")

    if not _field_exists(dataset, target_field):
        LOGGER.info("Adding text key field '%s' to %s", target_field, dataset)
        arcpy.management.AddField(
            in_table=dataset,
            field_name=target_field,
            field_type="TEXT",
            field_length=length,
        )

    # Example when length=15 → str(!GEO_ID!)[-15:]
    expr = f"str(!{source_field}!)[-{length}:]"
    LOGGER.info(
        "Calculating %s from %s (rightmost %d chars)",
        target_field,
        source_field,
        length,
    )
    arcpy.management.CalculateField(
        in_table=dataset,
        field=target_field,
        expression=expr,
        expression_type="PYTHON3",
    )
    return target_field


def _ensure_feature_class_key(fc: str, key: str) -> str:
    """Ensure the feature class *fc* has a normalized 15-char text join key."""
    if not _field_exists(fc, key):
        raise KeyError(f"'{key}' not found in {fc}")
    norm_field = f"{key}_15"
    return _add_text_key_field(fc, key, norm_field, length=15)


def _load_csv_to_memory(csv_path: str) -> str:
    """Load a CSV into the in_memory workspace and return its table path.

    Uses the Conversion toolbox (``arcpy.conversion.TableToTable``). This avoids the
    AttributeError you can hit when trying to call TableToTable from ``arcpy.management``.

    Args:
        csv_path: Absolute path to the input CSV file.

    Returns:
        The fully qualified in_memory table path.
    """
    out_name = "attrs_csv"
    LOGGER.info("Loading CSV → in_memory table: %s → %s", csv_path, out_name)
    arcpy.conversion.TableToTable(
        in_rows=csv_path,
        out_path="in_memory",
        out_name=out_name,
    )
    return os.path.join("in_memory", out_name)


def _ensure_table_key(table: str, preferred: str, fallback: str) -> str:
    """Ensure the attribute table has a normalized 15-char text key."""
    table_fields = _list_field_names(table)

    if preferred in table_fields:
        return _add_text_key_field(table, preferred, f"{preferred}_15", length=15)
    if fallback in table_fields:
        return _add_text_key_field(table, fallback, f"{fallback}_15", length=15)

    raise KeyError(f"Neither '{preferred}' nor '{fallback}' found in attribute table {table}.")


def _maybe_force_float(fc: str) -> None:
    """Optionally coerce short/long integer fields in *fc* to double (GDB only)."""
    if not FORCE_FLOAT:
        return

    desc = arcpy.Describe(fc)
    is_gdb = desc.dataType in ("FeatureClass", "Table") and desc.path.lower().endswith(".gdb")
    if not is_gdb:
        LOGGER.warning("FORCE_FLOAT True, but '%s' is not in a file gdb. Skipping coercion.", fc)
        return

    for field in arcpy.ListFields(fc):
        if field.type in ("SmallInteger", "Integer"):
            arcpy.management.AlterField(in_table=fc, field=field.name, new_field_type="DOUBLE")


def _ensure_temp_gdb(base_path: str) -> str:
    """Ensure a scratch FGDB next to *base_path*; return its path."""
    base_dir = os.path.dirname(base_path)
    gdb_path = os.path.join(base_dir, "scratch_join.gdb")
    if not arcpy.Exists(gdb_path):
        LOGGER.info("Creating scratch GDB at %s", gdb_path)
        arcpy.management.CreateFileGDB(base_dir, "scratch_join.gdb")
    return gdb_path


def join_blocks_to_attributes(
    blocks_fc: str,
    attrs_csv: str,
    out_path: str,
    left_key: str = LEFT_KEY,
    right_key: str = RIGHT_KEY,
    derivation_src: str = DERIVATION_SRC,
) -> None:
    """Join block geometry to attribute rows and write the result.

    Steps:
        1) Copy block features to memory.
        2) Ensure normalized 15-char keys on both sides.
        3) Perform an attribute join in memory.
        4) Export to shapefile or feature class. When writing to a shapefile,
           exports via a temporary file geodatabase to reduce field-name issues.

    Args:
        blocks_fc: Input block feature class or shapefile path.
        attrs_csv: Path to the combined attributes CSV.
        out_path: Output path (shapefile or file geodatabase feature class).
        left_key: Field on the blocks feature class to normalize/join on.
        right_key: Preferred field on the CSV table to normalize/join on.
        derivation_src: Fallback CSV field if ``right_key`` is absent.

    Raises:
        KeyError: If required key fields are missing.
        arcpy.ExecuteError: If any ArcPy geoprocessing tool fails.
    """
    LOGGER.info("Start join: %s ← %s", blocks_fc, attrs_csv)

    # Copy blocks to in_memory using a safe path constructor (avoid backslash escapes).
    blocks_mem = os.path.join("in_memory", "blocks_fc")
    LOGGER.info("Copying features to memory: %s → %s", blocks_fc, blocks_mem)
    arcpy.management.CopyFeatures(blocks_fc, blocks_mem)

    # Normalize join keys
    left_join_field = _ensure_feature_class_key(blocks_mem, left_key)

    attrs_mem = _load_csv_to_memory(attrs_csv)
    right_join_field = _ensure_table_key(attrs_mem, right_key, derivation_src)

    # Perform the join in memory
    joined_view = "joined_view"
    LOGGER.info(
        "Joining in memory: %s.%s → %s.%s",
        blocks_mem,
        left_join_field,
        attrs_mem,
        right_join_field,
    )
    arcpy.management.MakeFeatureLayer(blocks_mem, joined_view)
    arcpy.management.AddJoin(
        in_layer_or_view=joined_view,
        in_field=left_join_field,
        join_table=attrs_mem,
        join_field=right_join_field,
        join_type="KEEP_ALL",
    )

    # Prepare output location
    out_path = str(Path(out_path))
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Export joined result
    if out_path.lower().endswith(".shp"):
        scratch_gdb = _ensure_temp_gdb(out_path)
        tmp_fc = os.path.join(scratch_gdb, "joined_blocks")
        LOGGER.info("Copying joined layer → %s (temp GDB)", tmp_fc)
        arcpy.management.CopyFeatures(joined_view, tmp_fc)

        LOGGER.info("Exporting temp GDB feature class → shapefile %s", out_path)
        arcpy.conversion.FeatureClassToFeatureClass(
            in_features=tmp_fc,
            out_path=str(out_dir),
            out_name=Path(out_path).name,
        )
        LOGGER.warning("Shapefile output truncates field names to 10 chars.")
    else:
        LOGGER.info("Copying joined layer → %s", out_path)
        arcpy.management.CopyFeatures(joined_view, out_path)

    _maybe_force_float(out_path)
    LOGGER.info("Finished join → %s", out_path)


# =============================================================================
# MAIN
# =============================================================================


def run_pipeline() -> None:
    """Run CSV merge → TIGER merge/filter → join → outputs."""
    try:
        # 1) CSV stage: build combined attributes
        LOGGER.info("Discovering & merging CSVs under %s ...", INPUT_CSV_DIR)
        df_joined = build_joined_table_from_folder(INPUT_CSV_DIR, county_fips_filter=FIPS_TO_FILTER)
        out_csv = Path(INTERMEDIATE_COMBINED_CSV).expanduser().resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_joined.to_csv(out_csv, index=False)
        LOGGER.info("Combined CSV written → %s (rows=%d, cols=%d)", out_csv, *df_joined.shape)

        # 2) TIGER stage: discover, merge, ensure FIPS, filter, and write intermediate merged shp/fc
        LOGGER.info("Discovering TIGER shapefiles under %s ...", INPUT_SHP_DIR)
        shp_paths = discover_tiger_datasets(INPUT_SHP_DIR, TIGER_INPUT_GLOB)
        merged_fc_tmp = merge_shapefiles(shp_paths)
        ensure_fips_field(merged_fc_tmp, fips_field=FIPS_FIELD_NAME)
        filtered_fc_tmp = filter_by_fips(merged_fc_tmp, FIPS_TO_FILTER, fips_field=FIPS_FIELD_NAME)

        # Copy merged/filtered to requested intermediate location
        write_output(filtered_fc_tmp, INTERMEDIATE_MERGED_SHP)

        # 3) Join stage: join attributes to merged geometry
        join_blocks_to_attributes(
            blocks_fc=INTERMEDIATE_MERGED_SHP,
            attrs_csv=str(out_csv),
            out_path=FINAL_JOINED_FEATURES,
            left_key=LEFT_KEY,
            right_key=RIGHT_KEY,
            derivation_src=DERIVATION_SRC,
        )

        LOGGER.info("Pipeline completed successfully.")
        _gp("Pipeline completed successfully.")

    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Pipeline failed: %s", exc)
        _gp(f"Pipeline failed: {exc}", "error")
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline()
