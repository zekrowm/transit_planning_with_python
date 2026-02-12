"""GTFS stop comparison (before vs after) with notebook-friendly execution.

Outputs (CSV):
- stops_before.csv     : all stops from before feed
- stops_after.csv      : all stops from after feed
- stops_modified.csv   : overlap stop_id where relocated > threshold and/or attributes changed
- stops_deleted.csv    : stop_id present only in before feed
- stops_new.csv        : stop_id present only in after feed
- summary.json

Also outputs:
- stops_comparison.xlsx (sheets: before, after, modified, deleted, new, summary,
  optional nearest_id_matches)
- gtfs_stop_compare.log

No arcpy / geopandas. pandas + numpy + scipy only.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# =============================================================================
# Config
# =============================================================================

BEFORE_GTFS_DIR = Path(r"Path\To\GTFS\Dir")
AFTER_GTFS_DIR = Path(r"Path\to\GTFS\Dir")
OUTPUT_DIR = Path(r"Path\To\Output\Dir")

RELOCATE_THRESHOLD_FEET = 25.0
OVERLAP_WARN_THRESHOLD = 0.10  # warn if overlap fraction < 10%

ENABLE_NEAREST_MATCHES_WHEN_LOW_OVERLAP = True
NEAREST_MATCHES_MAX_FEET = 500.0  # only report nearest matches within this distance


# =============================================================================
# Data model
# =============================================================================


@dataclass(frozen=True)
class Summary:
    """Summary metrics for the comparison."""

    before_stop_count: int
    after_stop_count: int
    overlap_stop_count: int
    overlap_fraction_of_before: float
    overlap_fraction_of_after: float
    modified_count: int
    unchanged_count: int
    new_count: int
    deleted_count: int
    relocated_count: int
    attr_changed_count: int


# =============================================================================
# Logging
# =============================================================================


def setup_logging(output_dir: Path) -> logging.Logger:
    """Create a logger that writes to console + a file in the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("gtfs_stop_compare")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(output_dir / "gtfs_stop_compare.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# =============================================================================
# IO helpers
# =============================================================================


def load_gtfs_data(
    gtfs_folder_path: str,
    files: Optional[Sequence[str]] = None,
    dtype: str | type[str] | Mapping[str, Any] = str,
) -> dict[str, pd.DataFrame]:
    """Load one or more GTFS text files into memory.

    Args:
        gtfs_folder_path: Absolute or relative path to the folder
            containing the GTFS feed.
        files: Explicit sequence of file names to load. If ``None``,
            the standard 13 GTFS text files are attempted.
        dtype: Value forwarded to :pyfunc:`pandas.read_csv(dtype=…)` to
            control column dtypes. Supply a mapping for per-column dtypes.

    Returns:
        Mapping of file stem → :class:`pandas.DataFrame`; for example,
        ``data["trips"]`` holds the parsed *trips.txt* table.

    Raises:
        OSError: Folder missing or one of *files* not present.
        ValueError: Empty file or CSV parser failure.
        RuntimeError: Generic OS error while reading a file.

    Notes:
        All columns default to ``str`` to avoid pandas’ type-inference
        pitfalls (e.g. leading zeros in IDs).
    """
    if not os.path.exists(gtfs_folder_path):
        raise OSError(f"The directory '{gtfs_folder_path}' does not exist.")

    if files is None:
        files = (
            "agency.txt",
            "stops.txt",
            "routes.txt",
            "trips.txt",
            "stop_times.txt",
            "calendar.txt",
            "calendar_dates.txt",
            "fare_attributes.txt",
            "fare_rules.txt",
            "feed_info.txt",
            "frequencies.txt",
            "shapes.txt",
            "transfers.txt",
        )

    missing = [
        file_name
        for file_name in files
        if not os.path.exists(os.path.join(gtfs_folder_path, file_name))
    ]
    if missing:
        raise OSError(f"Missing GTFS files in '{gtfs_folder_path}': {', '.join(missing)}")

    data: dict[str, pd.DataFrame] = {}
    for file_name in files:
        key = file_name.replace(".txt", "")
        file_path = os.path.join(gtfs_folder_path, file_name)
        try:
            # Cast dtype to Any because pandas-stubs is strict about dict[str, Any]
            # vs DtypeArg, even though it works at runtime.
            from typing import cast

            df = pd.read_csv(file_path, dtype=cast("Any", dtype), low_memory=False)
            data[key] = df
            logging.info("Loaded %s (%d records).", file_name, len(df))

        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"File '{file_name}' in '{gtfs_folder_path}' is empty.") from exc

        except pd.errors.ParserError as exc:
            raise ValueError(
                f"Parser error in '{file_name}' in '{gtfs_folder_path}': {exc}"
            ) from exc

        except OSError as exc:
            raise RuntimeError(
                f"OS error reading file '{file_name}' in '{gtfs_folder_path}': {exc}"
            ) from exc

    return data


def normalize_text(series: pd.Series) -> pd.Series:
    """Normalize text for comparisons."""
    return series.fillna("").astype(str).str.strip()


def coerce_float(series: pd.Series) -> pd.Series:
    """Convert a string series to float; invalid values become NaN."""
    return pd.to_numeric(series, errors="coerce").astype(float)


def validate_stop_ids_unique(df: pd.DataFrame, label: str, logger: logging.Logger) -> pd.DataFrame:
    """Ensure stop_id is unique; if not, warn and keep the first occurrence per stop_id."""
    if "stop_id" not in df.columns:
        raise ValueError(f"{label}: stops.txt is missing required column 'stop_id'.")

    dup_mask = df["stop_id"].duplicated(keep="first")
    dup_count = int(dup_mask.sum())
    if dup_count > 0:
        dup_ids = df.loc[dup_mask, "stop_id"].head(20).tolist()
        logger.warning(
            "%s: found %s duplicate stop_id values; keeping first occurrence. Sample: %s",
            label,
            dup_count,
            dup_ids,
        )
        df = df.loc[~dup_mask].copy()

    return df


def load_stops(gtfs_path: Path, label: str, logger: logging.Logger) -> pd.DataFrame:
    """Load and standardize GTFS stops using the canonical helper."""
    # load_gtfs_data expects a str path
    data = load_gtfs_data(str(gtfs_path), files=["stops.txt"])
    df = data["stops"]

    required = {"stop_id", "stop_lat", "stop_lon"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{label}: stops.txt missing required columns: {missing}")

    df = df.copy()
    df["stop_id"] = normalize_text(df["stop_id"])

    if "stop_name" in df.columns:
        df["stop_name"] = normalize_text(df["stop_name"])

    df["stop_lat"] = coerce_float(df["stop_lat"])
    df["stop_lon"] = coerce_float(df["stop_lon"])

    df = validate_stop_ids_unique(df, label=label, logger=logger)

    missing_xy = int(df["stop_lat"].isna().sum() + df["stop_lon"].isna().sum())
    if missing_xy > 0:
        logger.warning("%s: %s rows have missing/invalid stop_lat/stop_lon.", label, missing_xy)

    return df


# =============================================================================
# Distance
# =============================================================================


def haversine_meters(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Great-circle distance (meters) between arrays of lat/lon in degrees."""
    r = 6_371_000.0
    lat1r = np.deg2rad(lat1)
    lon1r = np.deg2rad(lon1)
    lat2r = np.deg2rad(lat2)
    lon2r = np.deg2rad(lon2)

    dlat = lat2r - lat1r
    dlon = lon2r - lon1r

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return r * c


def meters_to_feet(meters: np.ndarray) -> np.ndarray:
    """Convert meters to feet."""
    return meters * 3.280839895013123


# =============================================================================
# Comparison logic
# =============================================================================


def pick_attribute_columns(before: pd.DataFrame, after: pd.DataFrame) -> list[str]:
    """Columns to compare for attribute changes (only those present in both feeds)."""
    candidates = [
        "stop_name",
        "stop_code",
        "stop_desc",
        "zone_id",
        "location_type",
        "parent_station",
        "stop_timezone",
        "wheelchair_boarding",
        "platform_code",
    ]
    return [c for c in candidates if c in before.columns and c in after.columns]


def build_modified_description(
    relocated: bool, changed_fields: list[str], distance_ft: float | None
) -> str:
    """Build a compact description of what changed for a modified stop."""
    parts: list[str] = []
    if relocated:
        if distance_ft is None or not np.isfinite(distance_ft):
            parts.append("Relocated (> threshold), distance unavailable.")
        else:
            parts.append(f"Relocated {distance_ft:.1f} ft (> threshold).")
    if changed_fields:
        parts.append(f"Fields changed: {';'.join(changed_fields)}")
    return " ".join(parts).strip()


def compare_stops(
    before: pd.DataFrame,
    after: pd.DataFrame,
    relocate_threshold_ft: float,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Summary, pd.DataFrame | None]:
    """Compare stops from two GTFS feeds.

    Returns:
        modified_df, deleted_df, new_df, unchanged_df, summary, nearest_matches(optional)
    """
    before_ids = set(before["stop_id"].tolist())
    after_ids = set(after["stop_id"].tolist())
    overlap_ids = before_ids & after_ids

    overlap_fraction_of_before = (len(overlap_ids) / len(before_ids)) if before_ids else 0.0
    overlap_fraction_of_after = (len(overlap_ids) / len(after_ids)) if after_ids else 0.0

    logger.info("Before stops: %s", len(before_ids))
    logger.info("After stops:  %s", len(after_ids))
    logger.info("Overlap IDs:  %s", len(overlap_ids))
    logger.info("Overlap as %% of before: %.1f%%", 100.0 * overlap_fraction_of_before)
    logger.info("Overlap as %% of after:  %.1f%%", 100.0 * overlap_fraction_of_after)

    if min(overlap_fraction_of_before, overlap_fraction_of_after) < OVERLAP_WARN_THRESHOLD:
        logger.warning(
            "Stop_id overlap is under %.0f%%. This often means either a major system overhaul "
            "or a stop_id renumbering/rekeying.",
            100.0 * OVERLAP_WARN_THRESHOLD,
        )

    deleted_ids = before_ids - after_ids
    new_ids = after_ids - before_ids

    deleted_df = before.loc[before["stop_id"].isin(deleted_ids)].copy()
    new_df = after.loc[after["stop_id"].isin(new_ids)].copy()

    # Compare overlap stops
    before_o = before.loc[before["stop_id"].isin(overlap_ids)].copy()
    after_o = after.loc[after["stop_id"].isin(overlap_ids)].copy()

    merged = before_o.merge(after_o, on="stop_id", how="inner", suffixes=("_before", "_after"))

    # Distance
    lat_b = merged["stop_lat_before"].to_numpy(dtype=float)
    lon_b = merged["stop_lon_before"].to_numpy(dtype=float)
    lat_a = merged["stop_lat_after"].to_numpy(dtype=float)
    lon_a = merged["stop_lon_after"].to_numpy(dtype=float)

    valid_xy = ~(np.isnan(lat_b) | np.isnan(lon_b) | np.isnan(lat_a) | np.isnan(lon_a))
    dist_ft = np.full(shape=(len(merged),), fill_value=np.nan, dtype=float)
    if int(valid_xy.sum()) > 0:
        meters = haversine_meters(
            lat_b[valid_xy], lon_b[valid_xy], lat_a[valid_xy], lon_a[valid_xy]
        )
        dist_ft[valid_xy] = meters_to_feet(meters)

    merged["distance_ft"] = dist_ft
    merged["delta_lat"] = merged["stop_lat_after"] - merged["stop_lat_before"]
    merged["delta_lon"] = merged["stop_lon_after"] - merged["stop_lon_before"]

    relocated_mask = merged["distance_ft"] > relocate_threshold_ft

    # Attribute changes
    attr_cols = pick_attribute_columns(before, after)
    changed_fields_col: list[str] = []

    for _, row in merged.iterrows():
        changed_fields: list[str] = []
        for c in attr_cols:
            b = str(row.get(f"{c}_before", "") or "").strip()
            a = str(row.get(f"{c}_after", "") or "").strip()
            if b != a:
                changed_fields.append(c)
        changed_fields_col.append(";".join(changed_fields))

    merged["changed_fields"] = changed_fields_col
    attr_changed_mask = merged["changed_fields"].astype(str).str.len() > 0

    modified_mask = relocated_mask | attr_changed_mask
    unchanged_mask = ~modified_mask

    def classify_type(relocated: bool, attr_changed: bool) -> str:
        if relocated and attr_changed:
            return "relocated+attrs"
        if relocated:
            return "relocated"
        if attr_changed:
            return "attrs"
        return "unchanged"

    relocated_arr = relocated_mask.to_numpy()
    attr_changed_arr = attr_changed_mask.to_numpy()
    merged["change_type"] = [
        classify_type(bool(r), bool(a))
        for r, a in zip(relocated_arr, attr_changed_arr)  # noqa: B905
    ]

    # Friendly description
    descs: list[str] = []
    for _, row in merged.iterrows():
        relocated = row["change_type"] in {"relocated", "relocated+attrs"}
        changed_fields = [f for f in str(row.get("changed_fields", "")).split(";") if f]
        dist = row.get("distance_ft")
        dist_val = float(dist) if pd.notna(dist) else None
        descs.append(
            build_modified_description(
                relocated=relocated, changed_fields=changed_fields, distance_ft=dist_val
            )
        )

    merged["change_description"] = descs

    modified_df = merged.loc[modified_mask].copy()
    unchanged_df = merged.loc[unchanged_mask].copy()

    # Order columns for modified output (keep it readable)
    key_cols = [
        "stop_id",
        "change_type",
        "change_description",
        "distance_ft",
        "delta_lat",
        "delta_lon",
        "changed_fields",
    ]
    before_cols = [c for c in modified_df.columns if c.endswith("_before")]
    after_cols = [c for c in modified_df.columns if c.endswith("_after")]

    # Prefer to show stop_name/lat/lon early
    def sort_cols(cols: list[str]) -> list[str]:
        priority = {"stop_name": 0, "stop_lat": 1, "stop_lon": 2}
        return sorted(
            cols,
            key=lambda x: (
                priority.get(x.replace("_before", "").replace("_after", ""), 99),
                x,
            ),
        )

    before_cols = sort_cols(before_cols)
    after_cols = sort_cols(after_cols)

    keep_cols = [c for c in key_cols if c in modified_df.columns] + before_cols + after_cols
    modified_df = (
        modified_df[keep_cols].sort_values(["change_type", "stop_id"]).reset_index(drop=True)
    )

    # Sort other outputs
    deleted_df = deleted_df.sort_values("stop_id").reset_index(drop=True)
    new_df = new_df.sort_values("stop_id").reset_index(drop=True)
    unchanged_df = unchanged_df.sort_values("stop_id").reset_index(drop=True)

    summary = Summary(
        before_stop_count=len(before_ids),
        after_stop_count=len(after_ids),
        overlap_stop_count=len(overlap_ids),
        overlap_fraction_of_before=float(overlap_fraction_of_before),
        overlap_fraction_of_after=float(overlap_fraction_of_after),
        modified_count=int(modified_mask.sum()),
        unchanged_count=int(unchanged_mask.sum()),
        new_count=len(new_df),
        deleted_count=len(deleted_df),
        relocated_count=int(relocated_mask.sum()),
        attr_changed_count=int(attr_changed_mask.sum()),
    )

    nearest_matches = None
    if (
        ENABLE_NEAREST_MATCHES_WHEN_LOW_OVERLAP
        and min(overlap_fraction_of_before, overlap_fraction_of_after) < OVERLAP_WARN_THRESHOLD
    ):
        nearest_matches = try_build_nearest_matches(
            before=before,
            after=after,
            logger=logger,
            max_feet=NEAREST_MATCHES_MAX_FEET,
        )

    return modified_df, deleted_df, new_df, unchanged_df, summary, nearest_matches


def try_build_nearest_matches(
    before: pd.DataFrame,
    after: pd.DataFrame,
    logger: logging.Logger,
    max_feet: float,
) -> pd.DataFrame | None:
    """Optional helper when stop_id overlap is very low.

    For each AFTER stop, finds nearest BEFORE stop by coordinates (within max_feet).
    """
    b = before[["stop_id", "stop_lat", "stop_lon"]].dropna().copy()
    a = after[["stop_id", "stop_lat", "stop_lon"]].dropna().copy()
    if b.empty or a.empty:
        logger.info("Insufficient valid coordinates for nearest-match output.")
        return None

    lat0 = float(pd.concat([b["stop_lat"], a["stop_lat"]], ignore_index=True).mean())
    meters_per_degree = 111_320.0
    cos0 = math.cos(math.radians(lat0))

    bx = b["stop_lon"].to_numpy() * cos0 * meters_per_degree
    by = b["stop_lat"].to_numpy() * meters_per_degree
    ax = a["stop_lon"].to_numpy() * cos0 * meters_per_degree
    ay = a["stop_lat"].to_numpy() * meters_per_degree

    tree = cKDTree(np.column_stack([bx, by]))
    _, idx = tree.query(np.column_stack([ax, ay]), k=1)

    nearest_before_ids = b["stop_id"].to_numpy()[idx]
    meters = haversine_meters(
        a["stop_lat"].to_numpy(),
        a["stop_lon"].to_numpy(),
        b["stop_lat"].to_numpy()[idx],
        b["stop_lon"].to_numpy()[idx],
    )
    dist_ft = meters_to_feet(meters)

    out = pd.DataFrame(
        {
            "after_stop_id": a["stop_id"].to_numpy(),
            "nearest_before_stop_id": nearest_before_ids,
            "nearest_distance_ft": dist_ft,
        }
    )
    out = (
        out.loc[out["nearest_distance_ft"] <= max_feet]
        .sort_values("nearest_distance_ft")
        .reset_index(drop=True)
    )

    logger.info("Nearest-match output created (%s rows within %.0f ft).", len(out), max_feet)
    return out


# =============================================================================
# Export
# =============================================================================


def write_outputs(
    output_dir: Path,
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    modified_df: pd.DataFrame,
    deleted_df: pd.DataFrame,
    new_df: pd.DataFrame,
    summary: Summary,
    nearest_matches: pd.DataFrame | None,
    logger: logging.Logger,
) -> None:
    """Write CSVs + Excel workbook + summary json."""
    output_dir.mkdir(parents=True, exist_ok=True)

    before_csv = output_dir / "stops_before.csv"
    after_csv = output_dir / "stops_after.csv"
    modified_csv = output_dir / "stops_modified.csv"
    deleted_csv = output_dir / "stops_deleted.csv"
    new_csv = output_dir / "stops_new.csv"
    summary_json = output_dir / "summary.json"
    xlsx_path = output_dir / "stops_comparison.xlsx"

    before_df.to_csv(before_csv, index=False, encoding="utf-8")
    after_df.to_csv(after_csv, index=False, encoding="utf-8")
    modified_df.to_csv(modified_csv, index=False, encoding="utf-8")
    deleted_df.to_csv(deleted_csv, index=False, encoding="utf-8")
    new_df.to_csv(new_csv, index=False, encoding="utf-8")

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    # Use openpyxl engine explicitly
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        before_df.to_excel(writer, sheet_name="before", index=False)
        after_df.to_excel(writer, sheet_name="after", index=False)
        modified_df.to_excel(writer, sheet_name="modified", index=False)
        deleted_df.to_excel(writer, sheet_name="deleted", index=False)
        new_df.to_excel(writer, sheet_name="new", index=False)
        pd.DataFrame([asdict(summary)]).to_excel(writer, sheet_name="summary", index=False)

        if nearest_matches is not None and not nearest_matches.empty:
            nearest_matches.to_excel(writer, sheet_name="nearest_id_matches", index=False)

    logger.info("Wrote: %s", before_csv)
    logger.info("Wrote: %s", after_csv)
    logger.info("Wrote: %s", modified_csv)
    logger.info("Wrote: %s", deleted_csv)
    logger.info("Wrote: %s", new_csv)
    logger.info("Wrote: %s", summary_json)
    logger.info("Wrote: %s", xlsx_path)

    if nearest_matches is not None:
        nm_csv = output_dir / "nearest_id_matches.csv"
        nearest_matches.to_csv(nm_csv, index=False, encoding="utf-8")
        logger.info("Wrote: %s", nm_csv)


# =============================================================================
# Notebook-friendly entry point
# =============================================================================


def run_compare(
    before_dir: Path = BEFORE_GTFS_DIR,
    after_dir: Path = AFTER_GTFS_DIR,
    out_dir: Path = OUTPUT_DIR,
    threshold_feet: float = RELOCATE_THRESHOLD_FEET,
) -> Summary:
    """Run the comparison (notebook-friendly) and write outputs."""
    logger = setup_logging(out_dir)

    logger.info("Before GTFS: %s", before_dir)
    logger.info("After GTFS:  %s", after_dir)
    logger.info("Output dir:  %s", out_dir)
    logger.info("Relocation threshold: %.1f ft", threshold_feet)

    before_df = load_stops(before_dir, label="before", logger=logger)
    after_df = load_stops(after_dir, label="after", logger=logger)

    modified_df, deleted_df, new_df, _unchanged_df, summary, nearest_matches = compare_stops(
        before=before_df,
        after=after_df,
        relocate_threshold_ft=float(threshold_feet),
        logger=logger,
    )

    write_outputs(
        output_dir=out_dir,
        before_df=before_df,
        after_df=after_df,
        modified_df=modified_df,
        deleted_df=deleted_df,
        new_df=new_df,
        summary=summary,
        nearest_matches=nearest_matches,
        logger=logger,
    )

    logger.info(
        "Done. Modified=%s (relocated=%s, attr_changed=%s). Deleted=%s. New=%s. Unchanged=%s.",
        summary.modified_count,
        summary.relocated_count,
        summary.attr_changed_count,
        summary.deleted_count,
        summary.new_count,
        summary.unchanged_count,
    )
    return summary


# =============================================================================
# CLI (still supported; notebook ignores injected args)
# =============================================================================


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse CLI args and return (args, unknown_args)."""
    parser = argparse.ArgumentParser(description="Compare GTFS stops between two feeds.")
    parser.add_argument("--before", type=Path, default=BEFORE_GTFS_DIR, help="Before GTFS folder")
    parser.add_argument("--after", type=Path, default=AFTER_GTFS_DIR, help="After GTFS folder")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument(
        "--threshold-feet",
        type=float,
        default=RELOCATE_THRESHOLD_FEET,
        help="Relocation threshold in feet",
    )
    args, unknown = parser.parse_known_args(list(argv) if argv is not None else None)
    return args, unknown


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point (notebook-safe)."""
    args, _unknown = parse_args(argv)
    run_compare(
        before_dir=args.before,
        after_dir=args.after,
        out_dir=args.out,
        threshold_feet=args.threshold_feet,
    )


if __name__ == "__main__":
    main()
