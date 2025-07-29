"""Calculate OTP percentages by stop and publish user‑friendly outputs.

This script processes a raw OTP CSV export and generates user-friendly outputs
including:
  - Recalculated OTP percentages (% On Time, % Early, % Late)
  - Route/direction-wide pivot tables of OTP and counts by stop
  - Stop-level summary tables of OTP performance
  - Optional filtering by route, direction, and timepoint

It supports override of stop order using a custom JSON configuration file and
ensures all configured stops appear in output tables, even if missing from the
input dataset.

Outputs are written to a user-specified directory, with clear filenames for each
(route, direction) combination.

TODO (future): GTFS‑synced visualizations (see stub at bottom).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

CSV_PATH: Path | str = r"Path\To\Your\OTP by Timepoint.csv"
OUTPUT_DIR: Path | str = r"Path\To\Your\Output_Folder"

OUT_SUFFIX: str = "_processed"

SHORT_ROUTE_FILTER: List[str] = ["101"]

TIMEPOINT_FILTER: List[str] = []
RDT_FILTER: List[Tuple[str, str, str]] = []

TIMEPOINT_ORDER: Dict[str, List[str]] = {
    "EASTBOUND": [
        "DULLES AIRPORT",
        "WORLDGATE & ELDEN",
        "HERNDON METRO STATION NORTH SIDE",
        "RESTON TOWN CENTER METRO",
        "WIEHE-RESTON EAST TRANSIT CTR",
    ],
    "WESTBOUND": [
        "WIEHE-RESTON EAST TRANSIT CTR",
        "RESTON TOWN CENTER METRO",
        "HERNDON METRO STATION NORTH SIDE",
        "WORLDGATE & ELDEN",
        "DULLES AIRPORT",
    ],
}

TIMEPOINT_ORDER_FILE: Path | str | None = None

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# =============================================================================
# FUNCTIONS
# =============================================================================


def build_argparser() -> argparse.ArgumentParser:
    """Create and configure the CLI argument parser.

    Returns:
    -------
    argparse.ArgumentParser
        A parser pre‑populated with all command‑line options supported by the
        script. The parser **does not** parse the arguments yet; call
        :py:meth:`parse_known_args` or :py:meth:`parse_args` on the returned
        object.
    """
    p = argparse.ArgumentParser(
        description="Recalculate OTP percentages, apply optional filters, and "
        "output pivot+summary tables."
    )
    p.add_argument("-i", "--input", default=CSV_PATH, help="Path to the input CSV.")
    p.add_argument(
        "-t",
        "--timepoints",
        nargs="*",
        default=TIMEPOINT_FILTER,
        metavar="ID",
        help="Time‑point IDs to keep.",
    )
    p.add_argument(
        "-r",
        "--routes",
        nargs="*",
        default=SHORT_ROUTE_FILTER,
        metavar="SHORT_ROUTE",
        help="Short Route codes to keep.",
    )
    p.add_argument(
        "-g",
        "--rdt",
        default="",
        type=str,
        help=(
            "Route,Direction,Time‑point triples separated by ';', "
            "e.g. '151,NORTHBOUND,MHUS;152,SOUTHBOUND,MVES'."
        ),
    )
    p.add_argument("-d", "--outdir", default=OUTPUT_DIR, help="Folder for all output files.")
    p.add_argument(
        "-o", "--output", default=None, help="Explicit output CSV path for the long table."
    )
    p.add_argument(
        "--pattern-file",
        default=TIMEPOINT_ORDER_FILE,
        type=str,
        metavar="JSON",
        help="JSON file overriding TIMEPOINT_ORDER.",
    )
    return p


def parse_rdt_arg(arg: str) -> List[Tuple[str, str, str]]:
    """Parse the `--rdt` option into a list of *(route, direction, timepoint)* triples.

    The CLI accepts a semicolon‑delimited string such as
    `"151,NORTHBOUND,MHUS;152,SOUTHBOUND,MVES"`.
    Each triple is validated for three comma‑separated parts.

    Parameters
    ----------
    arg
        Raw string from the command line.

    Returns:
    -------
    list[tuple[str, str, str]]
        Parsed triples. If the user supplied an empty string the default
        ``RDT_FILTER`` constant is returned.

    Raises:
    ------
    SystemExit
        If a chunk does not contain exactly three comma‑separated fields.
    """
    if not arg.strip():
        return RDT_FILTER
    triples: List[Tuple[str, str, str]] = []
    for chunk in arg.split(";"):
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 3:
            sys.exit(f"ERROR: bad --rdt chunk '{chunk}'. Use ROUTE,DIRECTION,TIMEPOINT.")
        triples.append(tuple(parts))  # type: ignore[arg-type]
    return triples


def make_short_route(route_str: str) -> str:
    """Return the *short route* code (portion before the first dash, no spaces)."""
    return route_str.split("-", 1)[0].replace(" ", "").strip()


def recalc_percentages(df: pd.DataFrame) -> pd.DataFrame:
    """Add *Total Counts* and %‑columns derived from raw on‑time/early/late counts.

    The function works in‑place but also returns the modified DataFrame to allow
    chaining.
    """
    df["Total Counts"] = (df["Sum # On Time"] + df["Sum # Early"] + df["Sum # Late"]).astype(
        "Int64"
    )
    for pct_col, cnt_col in (
        ("% On Time", "Sum # On Time"),
        ("% Early", "Sum # Early"),
        ("% Late", "Sum # Late"),
    ):
        df[pct_col] = (df[cnt_col] / df["Total Counts"].replace(0, pd.NA) * 100).round(2)
    return df


def filter_basic(df: pd.DataFrame, timepoints: List[str], routes: List[str]) -> pd.DataFrame:
    """Sub‑set the DataFrame by *Timepoint ID* and *Short Route*."""
    if timepoints:
        df = df[df["Timepoint ID"].isin(timepoints)]
    if routes:
        df = df[df["Short Route"].isin(routes)]
    return df


def filter_rdt(df: pd.DataFrame, triples: List[Tuple[str, str, str]]) -> pd.DataFrame:
    """Return only rows whose *(route, direction, timepoint)* matches `triples`."""
    if not triples:
        return df
    mask = False
    for r, d, t in triples:
        mask |= (
            (df["Short Route"] == r)
            & (df["Direction"].str.upper() == d.upper())
            & (df["Timepoint ID"] == t)
        )
    return df[mask]


def construct_output_path(inp: Path, outdir: str | Path, explicit: str | None) -> Path:
    """Resolve the path for the long‑table CSV output."""
    if explicit:
        return Path(explicit)
    return Path(outdir) / f"{inp.stem}{OUT_SUFFIX}{inp.suffix}"


def dedupe_apparent_trips(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows that represent the *same* trip at the same stop."""
    df["TripStart"] = df["Trip"].str.split().str[0]
    sort_cols = ["Short Route", "Direction", "Timepoint ID", "TripStart"]
    df = df.sort_values("Total Counts", ascending=False)
    return df.drop_duplicates(subset=sort_cols, keep="first")


def load_timepoint_order(path: str | Path | None) -> Dict[str, List[str]]:
    """Load a JSON file that maps directions to an ordered list of stops."""
    if path is None:
        return {k.upper(): v for k, v in TIMEPOINT_ORDER.items()}
    fp = Path(path)
    if not fp.exists():
        sys.exit(f"ERROR: pattern-file not found – {fp}")
    try:
        obj = json.loads(fp.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        sys.exit(f"ERROR: invalid JSON in {fp} – {exc}")  # noqa: TRY003
    if not isinstance(obj, dict):
        sys.exit("ERROR: pattern-file root must be a JSON object.")
    return {k.upper(): list(map(str, v)) for k, v in obj.items()}


def enforce_timepoint_order(df: pd.DataFrame, order_map: Dict[str, List[str]]) -> pd.DataFrame:
    """Filter out rows with stops not present in the configured order list."""
    mask = df.apply(
        lambda row: row["Timepoint Description"] in order_map.get(row["Direction"], []),
        axis=1,
    )
    if not mask.all():
        unknown = (
            df.loc[~mask, ["Direction", "Timepoint Description"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        logging.warning(
            "Dropped %d rows; unknown stops: %s",
            (~mask).sum(),
            "; ".join(f"{d} – {tp}" for d, tp in unknown),
        )
    return df[mask].copy()


def pivot_route_direction(
    df: pd.DataFrame,
    metric: str,
    order_map: Dict[str, List[str]],
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """Wide table for one metric.  Guarantees every configured stop exists.

    Returns { (route, direction): DataFrame } with columns:
      TripStart | Trip | <stops…>
    """
    results: Dict[Tuple[str, str], pd.DataFrame] = {}

    for (route, direction), g in df.groupby(["Short Route", "Direction"]):
        direction_uc = direction.upper()
        cfg_stops = order_map.get(direction_uc, [])

        if not cfg_stops:
            logging.warning("Direction %s missing in TIMEPOINT_ORDER – skipped", direction)
            continue

        g = g.assign(
            TPDesc=pd.Categorical(
                g["Timepoint Description"],
                categories=cfg_stops,
                ordered=True,
            ),
            TripStart=g["Trip"].str.split().str[0],
        )

        pivot = g.pivot(index="TripStart", columns="TPDesc", values=metric)

        # Merge full Trip ID
        trip_lookup = (
            g[["TripStart", "Trip"]].drop_duplicates(subset="TripStart").set_index("TripStart")
        )
        wide = trip_lookup.join(pivot)

        # Ensure all configured stops appear
        missing_cols = [s for s in cfg_stops if s not in wide.columns]
        if missing_cols:
            wide[missing_cols] = pd.NA
            logging.warning(
                "[%s %s] No OTP data for stops: %s",
                route,
                direction_uc,
                "; ".join(missing_cols),
            )

        # Order columns
        wide = wide[["Trip"] + cfg_stops]
        wide = wide.sort_index()

        results[(route, direction_uc)] = wide

    return results


def summary_route_direction(
    df: pd.DataFrame,
    order_map: Dict[str, List[str]],
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """Create a stop‑level summary for each (route, direction).

    Columns
    -------
    Timepoint Description | AvgPct | Count
        AvgPct : mean of % On Time (simple mean – see note below)
        Count  : Σ Total Counts  (ontime + early + late events)

    Warns once if a configured stop has zero observations.
    """
    summaries: Dict[Tuple[str, str], pd.DataFrame] = {}

    for (route, direction), g in df.groupby(["Short Route", "Direction"]):
        direction_uc = direction.upper()
        cfg = order_map.get(direction_uc, [])

        summ = (
            g.groupby("Timepoint Description")
            .agg(
                AvgPct=("% On Time", "mean"),  # simple average
                Count=("Total Counts", "sum"),  # ← FIXED
            )
            .reindex(cfg)
        )

        missing = summ[summ["Count"].isna()].index.tolist()
        if missing:
            logging.warning(
                "[%s %s] No OTP data for stops: %s",
                route,
                direction_uc,
                "; ".join(missing),
            )
            summ.loc[missing, ["AvgPct", "Count"]] = [pd.NA, 0]

        summaries[(route, direction_uc)] = summ.reset_index()

    return summaries


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Entry‑point guarded by ``if __name__ == "__main__"``."""
    parser = build_argparser()
    args, _unknown = parser.parse_known_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"ERROR: input file not found – {input_path}")

    df = pd.read_csv(input_path)
    if df.empty:
        sys.exit("ERROR: input CSV is empty.")

    # config + processing ------------------------------------------------------
    rdt_triples = parse_rdt_arg(args.rdt)
    order_map = load_timepoint_order(args.pattern_file)

    df["Short Route"] = df["Route"].astype(str).apply(make_short_route)
    df = recalc_percentages(df)
    df = filter_rdt(df, rdt_triples)
    df = filter_basic(df, args.timepoints, args.routes)
    df = dedupe_apparent_trips(df)
    df = enforce_timepoint_order(df, order_map)

    # save long table ----------------------------------------------------------
    out_path = construct_output_path(input_path, args.outdir, args.output)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logging.info("Processed %d rows → %s", len(df), out_path.resolve())

    # pivots & summaries -------------------------------------------------------
    pivot_pct = pivot_route_direction(df, "% On Time", order_map)
    pivot_cnt = pivot_route_direction(df, "Total Counts", order_map)
    summaries = summary_route_direction(df, order_map)

    outdir = Path(args.outdir)
    for key in pivot_pct:
        route, direction = key
        stem = f"{route}_{direction}"
        pivot_pct[key].reset_index().to_csv(outdir / f"{stem}_pct.csv", index=False)
        pivot_cnt[key].reset_index().to_csv(outdir / f"{stem}_cnt.csv", index=False)
        summaries[key].to_csv(outdir / f"{stem}_summary.csv", index=False)
        logging.info("Wrote %s* files for %s %s", stem, route, direction)

    # ------------------------------------------------------------------------
    # TODO: integrate GTFS stop‑times and generate heatmaps or line plots
    #       of % On Time by stop (matplotlib).  Also consider headway variance.
    # ------------------------------------------------------------------------


if __name__ == "__main__":
    main()
