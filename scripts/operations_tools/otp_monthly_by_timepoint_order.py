"""Process OTP-by-timepoint exports into monthly route/direction/variation deliverables.

This script ingests an OTP-by-timepoint CSV export and produces analysis-ready outputs that treat
each unique (Short Route, Direction, Variation) as its own pattern. It converts the source data to
a monthly grain (YYYY-MM) to avoid relying on potentially meaningless day-level dates in the export.

Core steps
----------
- Normalize key fields (Short Route, Direction, Variation, Timepoint Order).
- Construct a canonical Year-Month (YYYY-MM) field:
  - Prefer an existing YYYY-MM column if present.
  - Otherwise derive from Date where parseable, and optionally backfill using Month when the year
    can be inferred from other dated rows.
- Aggregate to monthly totals per stop (Timepoint Order) and recompute OTP, early, and late
  percentages from the aggregated counts.
- Generate per-variation outputs:
  - Monthly pivot tables of % On Time and Total Counts (rows = Year-Month, columns = stop order).
  - A stop-level summary for the analysis window that retains Timepoint ID/Description as context.
  - Optional line plots showing on-time/early/late % across stop order, including a dashed
    reference line for a configurable OTP standard.

Important notes
---------------
- Variations are handled separately because different variants can serve different stops, in
  different orders, or follow different paths.
- If a route pattern changes mid-year, run separate month windows (start/end month config) to
  avoid mixing incompatible patterns.
- If the export cannot support a reliable YYYY-MM (no usable YYYY-MM field, no parseable dates,
  and no way to infer year for Month values), the script fails loudly rather than guessing.
"""

from __future__ import annotations

import argparse
import difflib
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

CSV_PATH: Path | str = r"Path\To\Your\OTP by Timepoint Aggregated.csv"
OUTPUT_DIR: Path | str = r"Path\To\Your\Output_Folder"

OUT_SUFFIX: str = "_processed"

SHORT_ROUTE_FILTER: List[str] = []  # Add your desired routes, if needed

# When filtering by timepoints, pass **Timepoint Order** values here (integers).
TIMEPOINT_FILTER: List[int] = []

# RDT triples use **Timepoint Order** as the 3rd value (ROUTE,DIRECTION,ORDER)
RDT_FILTER: List[Tuple[str, str, int]] = []

# Optional analysis window (inclusive), formatted "YYYY-MM".
ANALYSIS_START_MONTH: str | None = None
ANALYSIS_END_MONTH: str | None = None

# If not None, keep only the top N variations per (route, direction) by Total Counts.
TOP_VARIATIONS_PER_ROUTE_DIR: int | None = None

ALLOWED_DIRECTIONS: List[str] = [
    "NORTHBOUND",
    "SOUTHBOUND",
    "EASTBOUND",
    "WESTBOUND",
    "LOOP",
]

SORT_TIMEPOINT_ORDER_ASC: bool = True

# OTP standard (percent). Used only for plotting reference line.
OTP_STANDARD_PCT: float = 85.0

WRITE_TIMEPOINT_LOOKUPS: bool = True

GENERATE_LINE_PLOTS: bool = True
PLOT_DPI: int = 160

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# =============================================================================
# ARGUMENTS
# =============================================================================


def build_argparser() -> argparse.ArgumentParser:
    """Create and configure the CLI argument parser."""
    p = argparse.ArgumentParser(
        description=(
            "Recalculate OTP percentages, apply optional filters, and "
            "output pivot+summary tables."
        )
    )
    p.add_argument("-i", "--input", default=CSV_PATH, help="Path to the input CSV.")
    p.add_argument(
        "-t",
        "--timepoints",
        nargs="*",
        default=TIMEPOINT_FILTER,
        type=int,
        metavar="TIMEPOINT_ORDER",
        help="Timepoint Order values to keep (integers).",
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
            "Route,Direction,TimepointOrder triples separated by ';', "
            "e.g. '151,NORTHBOUND,3;152,SOUTHBOUND,5'. Direction is normalized."
        ),
    )
    p.add_argument("-d", "--outdir", default=OUTPUT_DIR, help="Folder for all output files.")
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help="Explicit output CSV path for the long (processed) table.",
    )
    p.add_argument(
        "--start-month",
        default=ANALYSIS_START_MONTH,
        type=str,
        metavar="YYYY-MM",
        help="Inclusive analysis start month (e.g., 2025-01). If omitted, no start bound.",
    )
    p.add_argument(
        "--end-month",
        default=ANALYSIS_END_MONTH,
        type=str,
        metavar="YYYY-MM",
        help="Inclusive analysis end month (e.g., 2025-06). If omitted, no end bound.",
    )
    p.add_argument(
        "--top-variations",
        default=TOP_VARIATIONS_PER_ROUTE_DIR,
        type=int,
        metavar="N",
        help="Keep only the top N variations per (route, direction) by Total Counts.",
    )
    p.add_argument(
        "--no-timepoint-lookups",
        action="store_true",
        help="Disable writing *_timepoints.csv lookup files.",
    )
    p.add_argument(
        "--no-line-plots",
        action="store_true",
        help="Disable writing *_otp_line.png plot files.",
    )
    return p


def parse_rdt_arg(arg: str) -> List[Tuple[str, str, int]]:
    """Parse the `--rdt` option into (route, direction, timepoint_order) triples."""
    if not arg.strip():
        return RDT_FILTER

    triples: List[Tuple[str, str, int]] = []
    for chunk in arg.split(";"):
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 3:
            sys.exit(f"ERROR: bad --rdt chunk '{chunk}'. Use ROUTE,DIRECTION,TIMEPOINT_ORDER.")
        route, direction, order_raw = parts
        try:
            order_val = int(order_raw)
        except ValueError:
            sys.exit(f"ERROR: bad --rdt chunk '{chunk}'. TIMEPOINT_ORDER must be an integer.")
        triples.append((route, direction, order_val))
    return triples


# =============================================================================
# CORE HELPERS
# =============================================================================


def make_short_route(route_str: str) -> str:
    """Return the short route code (portion before the first dash, no spaces)."""
    return str(route_str).split("-", 1)[0].replace(" ", "").strip()


def construct_output_path(inp: Path, outdir: str | Path, explicit: str | None) -> Path:
    """Resolve the path for the long-table CSV output."""
    if explicit:
        return Path(explicit)
    return Path(outdir) / f"{inp.stem}{OUT_SUFFIX}{inp.suffix}"


def normalize_timepoint_order_column(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the 'Timepoint Order' column to integer values."""
    col = "Timepoint Order"
    if col not in df.columns:
        sys.exit("ERROR: input CSV is missing required column 'Timepoint Order'.")

    coerced = pd.to_numeric(df[col], errors="coerce")
    if coerced.isna().any():
        cols = [col]
        for extra in ("Timepoint ID", "Timepoint Description", "Route", "Direction", "Variation"):
            if extra in df.columns:
                cols.append(extra)
        bad = df.loc[coerced.isna(), cols].head(10)
        sys.exit(
            "ERROR: 'Timepoint Order' contains non-numeric values. "
            f"Examples:\n{bad.to_string(index=False)}"
        )

    df[col] = coerced.astype("Int64")
    return df


def normalize_variation_column(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the 'Variation' column to a stable string label."""
    if "Variation" not in df.columns:
        df["Variation"] = "UNKNOWN"
        return df

    df["Variation"] = (
        df["Variation"]
        .astype("string")
        .fillna("UNKNOWN")
        .map(lambda s: s.strip() if s is not None else "UNKNOWN")
        .replace("", "UNKNOWN")
    )
    return df


def normalize_direction_value(value: str, allowed: List[str]) -> str:
    """Coerce a free-text direction to one of the allowed values."""
    val = (value or "").strip().upper()
    if val in allowed:
        return val

    guess = difflib.get_close_matches(val, allowed, n=1, cutoff=0.6)
    if guess:
        fixed = guess[0]
        logging.warning("Direction normalized: %r → %r", value, fixed)
        return fixed

    logging.warning("Unrecognized direction %r; defaulting to 'LOOP'", value)
    return "LOOP"


def normalize_directions_column(df: pd.DataFrame, allowed: List[str]) -> pd.DataFrame:
    """Normalize the 'Direction' column in place to the allowed list."""
    df["Direction"] = df["Direction"].astype(str).map(
        lambda s: normalize_direction_value(s, allowed)
    )
    return df


def parse_month_yyyy_mm(value: str) -> pd.Period:
    """Parse a month string in YYYY-MM format into a monthly Period."""
    try:
        return pd.Period(value, freq="M")
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"ERROR: invalid month {value!r}; expected 'YYYY-MM'. {exc}")


def month_name_to_number(value: str) -> int | None:
    """Convert common month tokens to a month number (1-12)."""
    if value is None:
        return None
    v = str(value).strip()
    if not v:
        return None

    # Accept 'Apr', 'April', '04', '4'
    m_map = {
        "jan": 1,
        "january": 1,
        "feb": 2,
        "february": 2,
        "mar": 3,
        "march": 3,
        "apr": 4,
        "april": 4,
        "may": 5,
        "jun": 6,
        "june": 6,
        "jul": 7,
        "july": 7,
        "aug": 8,
        "august": 8,
        "sep": 9,
        "sept": 9,
        "september": 9,
        "oct": 10,
        "october": 10,
        "nov": 11,
        "november": 11,
        "dec": 12,
        "december": 12,
    }

    low = v.lower()
    if low in m_map:
        return m_map[low]

    # numeric month
    try:
        n = int(low)
    except ValueError:
        return None
    if 1 <= n <= 12:
        return n
    return None


def add_year_month_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a stable monthly grain column 'Year-Month' (string 'YYYY-MM').

    Preference order:
      1) Existing 'Year-Month' / 'YearMonth' / 'YYYY-MM' (already in YYYY-MM)
      2) Parseable 'Date' → Year-Month
      3) If Date is sparse but Month exists, fill Year-Month for missing-date rows by:
         - inferring the year per Month from the rows that *do* have Date values

    If the script cannot construct Year-Month safely, it fails loudly.
    """
    candidates = ["Year-Month", "YearMonth", "YYYY-MM", "year_month", "yearmonth"]
    for c in candidates:
        if c in df.columns:
            ym = df[c].astype("string").fillna("").str.strip()
            if (ym != "").any():
                parsed = ym.map(lambda s: parse_month_yyyy_mm(str(s)) if s else pd.NaT)
                if parsed.isna().all():
                    sys.exit(
                        f"ERROR: Found {c!r} but none of its values parse as 'YYYY-MM'. "
                        "Fix the export."
                    )
                df["Year-Month"] = parsed.astype("period[M]").astype(str)
                return df

    date_parsed = None
    if "Date" in df.columns:
        date_parsed = pd.to_datetime(df["Date"], errors="coerce", infer_datetime_format=True)
        if date_parsed.notna().any():
            df["Year-Month"] = date_parsed.dt.to_period("M").astype(str)
        else:
            date_parsed = None

    # If Year-Month created and contains only real months, we're done.
    if "Year-Month" in df.columns and (df["Year-Month"] != "NaT").any():
        # But we may have many NaT rows; try to fill using Month if available.
        pass
    else:
        # No usable Date-derived Year-Month; try Month-based (requires some year source).
        df["Year-Month"] = "NaT"

    if "Month" not in df.columns:
        sys.exit(
            "ERROR: Could not derive 'Year-Month'. Provide a YYYY-MM field (recommended) or "
            "a usable 'Date' column, or include 'Month' plus at least one dated row per month."
        )

    # Attempt fill for missing Year-Month using Month + inferred year per month.
    # Step 1: Build a Month→Year lookup from rows that have a real Year-Month already.
    has_ym = df["Year-Month"].astype(str).ne("NaT")
    if not has_ym.any():
        sys.exit(
            "ERROR: Could not derive any Year-Month from Date. Add a 'Year-Month' column (YYYY-MM) "
            "to the export, or ensure Date is populated for at least some rows."
        )

    # Map Month tokens to month number
    month_num = df["Month"].map(month_name_to_number) if "Month" in df.columns else pd.Series(
        [None] * len(df), index=df.index
    )

    # Extract year + month number from existing Year-Month
    ym_period = df.loc[has_ym, "Year-Month"].map(lambda s: parse_month_yyyy_mm(str(s)))
    ym_year = ym_period.map(lambda p: int(p.year))
    ym_mon = ym_period.map(lambda p: int(p.month))

    ref = pd.DataFrame(
        {
            "MonthNum": month_num.loc[has_ym].astype("Int64"),
            "Year": ym_year.astype("Int64"),
            "Mon": ym_mon.astype("Int64"),
        }
    ).dropna()

    if ref.empty:
        sys.exit(
            "ERROR: Year-Month exists but cannot be interpreted to backfill missing months. "
            "Add a proper YYYY-MM field upstream."
        )

    # Prefer year inferred from matching month number (Month column) where possible;
    # else fall back to most common year.
    most_common_year = int(ref["Year"].mode().iloc[0])

    month_to_year: Dict[int, int] = {}
    for m in sorted(ref["Mon"].unique()):
        # use the most common year observed for that month number
        y_mode = ref.loc[ref["Mon"] == m, "Year"].mode()
        if not y_mode.empty:
            month_to_year[int(m)] = int(y_mode.iloc[0])

    # Fill missing Year-Month where MonthNum is known
    missing_ym = df["Year-Month"].astype(str).eq("NaT")
    fill_monthnum = month_num.astype("Int64")

    fill_vals: List[str] = []
    for idx in df.index:
        if not missing_ym.loc[idx]:
            fill_vals.append(str(df.loc[idx, "Year-Month"]))
            continue

        mnum = fill_monthnum.loc[idx]
        if pd.isna(mnum):
            fill_vals.append("NaT")
            continue

        year = month_to_year.get(int(mnum), most_common_year)
        fill_vals.append(f"{year:04d}-{int(mnum):02d}")

    df["Year-Month"] = pd.Series(fill_vals, index=df.index)

    # Final validation: any remaining NaT means we could not safely assign.
    if df["Year-Month"].astype(str).eq("NaT").any():
        bad_mask = df["Year-Month"].astype(str).eq("NaT")
        bad = df.loc[bad_mask, ["Month", "Route", "Direction", "Variation"]].head(10)
        sys.exit(
            "ERROR: Could not infer Year-Month for some rows "
            "(missing Month and missing Date-derived month). "
            "Fix the export. Examples:\n"
            f"{bad.to_string(index=False)}"
        )

    # Canonicalize to YYYY-MM by parsing
    df["Year-Month"] = df["Year-Month"].map(lambda s: str(parse_month_yyyy_mm(str(s))))
    return df


def filter_month_range(
    df: pd.DataFrame, start_month: str | None, end_month: str | None
) -> pd.DataFrame:
    """Filter rows by an inclusive month range using the 'Year-Month' column."""
    if start_month is None and end_month is None:
        return df

    period = df["Year-Month"].map(lambda s: parse_month_yyyy_mm(str(s)))

    mask = pd.Series(True, index=df.index)
    if start_month is not None:
        start_p = parse_month_yyyy_mm(start_month)
        mask &= period >= start_p
    if end_month is not None:
        end_p = parse_month_yyyy_mm(end_month)
        mask &= period <= end_p

    dropped = int((~mask).sum())
    if dropped:
        logging.info(
            "Month filter dropped %d rows outside [%s, %s].", dropped, start_month, end_month
        )

    return df.loc[mask].copy()


def filter_basic(df: pd.DataFrame, timepoints: List[int], routes: List[str]) -> pd.DataFrame:
    """Sub-set the DataFrame by Timepoint Order and Short Route."""
    if timepoints:
        df = df[df["Timepoint Order"].isin(timepoints)]
    if routes:
        df = df[df["Short Route"].isin(routes)]
    return df


def filter_rdt(df: pd.DataFrame, triples: List[Tuple[str, str, int]]) -> pd.DataFrame:
    """Return only rows whose (route, direction, timepoint_order) matches `triples`."""
    if not triples:
        return df

    mask = pd.Series(False, index=df.index)
    for r, d, t_ord in triples:
        mask |= (
            (df["Short Route"] == r)
            & (df["Direction"] == d.upper())
            & (df["Timepoint Order"] == t_ord)
        )
    return df[mask]


def infer_timepoint_orders_for_group(group_df: pd.DataFrame, ascending: bool) -> List[int]:
    """Infer the ordered list of Timepoint Order values for a single group."""
    orders = (
        group_df["Timepoint Order"]
        .dropna()
        .astype("Int64")
        .drop_duplicates()
        .astype(int)
        .tolist()
    )
    orders.sort(reverse=not ascending)
    return orders


def build_timepoint_lookup_for_group(group_df: pd.DataFrame) -> pd.DataFrame:
    """Build a lookup table for Timepoint Order → Timepoint ID/Description for one group."""
    if "Timepoint ID" not in group_df.columns:
        group_df = group_df.assign(**{"Timepoint ID": pd.NA})
    if "Timepoint Description" not in group_df.columns:
        group_df = group_df.assign(**{"Timepoint Description": pd.NA})

    cols = ["Timepoint Order", "Timepoint ID", "Timepoint Description", "Total Counts"]
    g = group_df[cols].copy()
    g = g.sort_values("Total Counts", ascending=False)

    lookup = (
        g.drop_duplicates(subset=["Timepoint Order"], keep="first")
        .drop(columns=["Total Counts"])
        .sort_values("Timepoint Order")
        .reset_index(drop=True)
    )
    return lookup


def aggregate_to_month(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to monthly grain by stop and recompute OTP percentages."""
    group_cols = ["Short Route", "Direction", "Variation", "Year-Month", "Timepoint Order"]

    agg = (
        df.groupby(group_cols, dropna=False)
        .agg(
            **{
                "Sum # On Time": ("Sum # On Time", "sum"),
                "Sum # Early": ("Sum # Early", "sum"),
                "Sum # Late": ("Sum # Late", "sum"),
            }
        )
        .reset_index()
    )

    agg["Total Counts"] = (agg["Sum # On Time"] + agg["Sum # Early"] + agg["Sum # Late"]).astype(
        "Int64"
    )
    denom = agg["Total Counts"].replace(0, pd.NA)

    agg["% On Time"] = (agg["Sum # On Time"] / denom * 100).round(2)
    agg["% Early"] = (agg["Sum # Early"] / denom * 100).round(2)
    agg["% Late"] = (agg["Sum # Late"] / denom * 100).round(2)

    return agg


def filter_top_variations(df_monthly: pd.DataFrame, top_n: int | None) -> pd.DataFrame:
    """Optionally keep only the top N variations per (Short Route, Direction)."""
    if top_n is None:
        return df_monthly

    totals = (
        df_monthly.groupby(["Short Route", "Direction", "Variation"], dropna=False)["Total Counts"]
        .sum()
        .reset_index()
    )
    totals["Rank"] = totals.groupby(["Short Route", "Direction"])["Total Counts"].rank(
        method="first",
        ascending=False,
    )
    keep = totals.loc[totals["Rank"] <= top_n, ["Short Route", "Direction", "Variation"]]
    out = df_monthly.merge(keep, on=["Short Route", "Direction", "Variation"], how="inner")

    dropped = len(df_monthly) - len(out)
    logging.info("Top-variation filter kept %d rows (dropped %d rows).", len(out), dropped)
    return out


def slugify_for_filename(value: str) -> str:
    """Return a filesystem-safe slug for naming output files."""
    v = (value or "").strip()
    if not v:
        return "UNKNOWN"
    v = re.sub(r"\s+", "_", v)
    v = re.sub(r"[^A-Za-z0-9_.-]+", "", v)
    return v or "UNKNOWN"


def save_variation_line_plot(
    summary_df: pd.DataFrame, outpath: Path, dpi: int, otp_standard: float
) -> None:
    """Save a line plot of On-time/Early/Late percentages across stop order."""
    import matplotlib.pyplot as plt  # local import to keep base deps light

    x = summary_df["Timepoint Order"].astype(int)

    plt.figure()
    plt.plot(x, summary_df["AvgOnTime"], marker="o", label="On-time %")
    plt.plot(x, summary_df["AvgEarly"], marker="o", label="Early %")
    plt.plot(x, summary_df["AvgLate"], marker="o", label="Late %")

    plt.axhline(otp_standard, linestyle="--", color="red", label=f"Standard ({otp_standard:.0f}%)")

    plt.xlabel("Timepoint Order")
    plt.ylabel("Percent")
    plt.title("OTP by Stop Order")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi)
    plt.close()


# =============================================================================
# OUTPUT BUILDERS
# =============================================================================


def pivot_monthly_by_stop(
    df_monthly: pd.DataFrame,
    metric: str,
    ascending: bool,
) -> Dict[Tuple[str, str, str], pd.DataFrame]:
    """Monthly pivot per (route, direction, variation)."""
    results: Dict[Tuple[str, str, str], pd.DataFrame] = {}
    group_cols = ["Short Route", "Direction", "Variation"]

    for (route, direction, variation), g in df_monthly.groupby(group_cols):
        cfg_orders = infer_timepoint_orders_for_group(g, ascending=ascending)
        if not cfg_orders:
            logging.warning(
                "[%s %s %s] No Timepoint Order values found; skipped.",
                route,
                direction,
                variation,
            )
            continue

        g = g.assign(
            TPORD=pd.Categorical(
                g["Timepoint Order"].astype("Int64"),
                categories=pd.array(cfg_orders, dtype="int64"),
                ordered=True,
            )
        )

        pivot = g.pivot(index="Year-Month", columns="TPORD", values=metric)

        missing_cols = [o for o in cfg_orders if o not in pivot.columns]
        if missing_cols:
            pivot[missing_cols] = pd.NA

        pivot = pivot[cfg_orders].sort_index()
        pivot.insert(0, "Year-Month", pivot.index.astype(str))
        pivot = pivot.reset_index(drop=True)

        results[(route, direction, variation)] = pivot

    return results


def summary_route_direction(
    df_monthly: pd.DataFrame,
    df_context: pd.DataFrame,
    ascending: bool,
    group_n: Dict[Tuple[str, str, str], int],
) -> Dict[Tuple[str, str, str], pd.DataFrame]:
    """Stop-level summary for the whole analysis window, per (route, direction, variation)."""
    summaries: Dict[Tuple[str, str, str], pd.DataFrame] = {}
    group_cols = ["Short Route", "Direction", "Variation"]

    for (route, direction, variation), g in df_monthly.groupby(group_cols):
        cfg_orders = infer_timepoint_orders_for_group(g, ascending=ascending)
        if not cfg_orders:
            logging.warning(
                "[%s %s %s] No Timepoint Order values found; skipped.",
                route,
                direction,
                variation,
            )
            continue

        n_total = group_n.get((route, direction, variation), int(g["Total Counts"].sum()))

        summ = (
            g.groupby("Timepoint Order")
            .agg(
                AvgOnTime=("% On Time", "mean"),
                AvgEarly=("% Early", "mean"),
                AvgLate=("% Late", "mean"),
                Count=("Total Counts", "sum"),
            )
            .reindex(cfg_orders)
        )
        summ.index.name = "Timepoint Order"
        out = summ.reset_index()

        ctx = df_context[
            (df_context["Short Route"] == route)
            & (df_context["Direction"] == direction)
            & (df_context["Variation"] == variation)
        ].copy()

        if ctx.empty:
            lookup = out.assign(**{"Timepoint ID": pd.NA, "Timepoint Description": pd.NA})[
                ["Timepoint Order", "Timepoint ID", "Timepoint Description"]
            ]
        else:
            # Build a contextual lookup using group Total Counts as weights
            ctx_tmp = ctx.copy()
            if "Total Counts" not in ctx_tmp.columns:
                # If context df hasn't been aggregated yet, approximate weight from processed sum
                if "Sum # Processed" in ctx_tmp.columns:
                    ctx_tmp["Total Counts"] = pd.to_numeric(
                        ctx_tmp["Sum # Processed"], errors="coerce"
                    ).fillna(0)
                else:
                    ctx_tmp["Total Counts"] = 1
            lookup = build_timepoint_lookup_for_group(ctx_tmp)

        out = out.merge(lookup, on="Timepoint Order", how="left")

        out.insert(0, "Variation", variation)
        out.insert(0, "Direction", direction)
        out.insert(0, "Short Route", route)
        out["N"] = n_total

        summaries[(route, direction, variation)] = out

    return summaries


def build_variation_index(df_monthly: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (route, direction, variation) with n and month bounds."""
    group_cols = ["Short Route", "Direction", "Variation"]
    return (
        df_monthly.groupby(group_cols, dropna=False)
        .agg(
            N=("Total Counts", "sum"),
            Rows=("Total Counts", "size"),
            StartMonth=("Year-Month", "min"),
            EndMonth=("Year-Month", "max"),
        )
        .reset_index()
        .sort_values(["Short Route", "Direction", "N"], ascending=[True, True, False])
    )


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Entry-point guarded by ``if __name__ == "__main__"``."""
    parser = build_argparser()
    args, _unknown = parser.parse_known_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"ERROR: input file not found – {input_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(input_path)

    required = [
        "Route",
        "Direction",
        "Timepoint Order",
        "Sum # On Time",
        "Sum # Early",
        "Sum # Late",
    ]
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        sys.exit(f"ERROR: missing required columns: {missing}")

    # Normalize / enrich --------------------------------------------------------
    df_raw["Short Route"] = df_raw["Route"].astype(str).apply(make_short_route)
    df_raw = normalize_variation_column(df_raw)
    df_raw = normalize_directions_column(df_raw, ALLOWED_DIRECTIONS)
    df_raw = normalize_timepoint_order_column(df_raw)

    # Monthly grain -------------------------------------------------------------
    df_raw = add_year_month_column(df_raw)
    df_raw = filter_month_range(df_raw, args.start_month, args.end_month)

    # Monthly aggregation (source of truth for analysis) ------------------------
    df_monthly = aggregate_to_month(df_raw)

    # Filters (monthly) ---------------------------------------------------------
    df_monthly = filter_basic(df_monthly, args.timepoints, args.routes)
    df_monthly = filter_rdt(df_monthly, parse_rdt_arg(args.rdt))

    if df_monthly.empty:
        sys.exit("ERROR: No rows remain after filtering; nothing to write.")

    # Optional top-variation filter (monthly) ----------------------------------
    df_monthly = filter_top_variations(df_monthly, args.top_variations)
    if df_monthly.empty:
        sys.exit("ERROR: No rows remain after top-variation filtering; nothing to write.")

    # Long table output (monthly) ----------------------------------------------
    out_path = construct_output_path(input_path, outdir, args.output)
    df_monthly.to_csv(out_path, index=False)
    logging.info("Processed %d monthly rows → %s", len(df_monthly), out_path.resolve())

    # Variation index (n / bounds) ---------------------------------------------
    variation_index = build_variation_index(df_monthly)
    variation_index_path = outdir / "variation_index.csv"
    variation_index.to_csv(variation_index_path, index=False)
    logging.info("Wrote %s", variation_index_path.resolve())

    n_map = {
        (r, d, v): int(n)
        for r, d, v, n in variation_index[
            ["Short Route", "Direction", "Variation", "N"]
        ].itertuples(index=False, name=None)
    }

    # pivots & summaries --------------------------------------------------------
    pivot_pct = pivot_monthly_by_stop(df_monthly, "% On Time", ascending=SORT_TIMEPOINT_ORDER_ASC)
    pivot_cnt = pivot_monthly_by_stop(
        df_monthly, "Total Counts", ascending=SORT_TIMEPOINT_ORDER_ASC
    )
    summaries = summary_route_direction(
        df_monthly=df_monthly,
        df_context=df_raw,
        ascending=SORT_TIMEPOINT_ORDER_ASC,
        group_n=n_map,
    )

    write_lookups = WRITE_TIMEPOINT_LOOKUPS and not args.no_timepoint_lookups
    write_plots = GENERATE_LINE_PLOTS and not args.no_line_plots

    for (route, direction, variation), wide_pct in pivot_pct.items():
        var_slug = slugify_for_filename(str(variation))
        n_total = n_map.get((route, direction, variation), 0)
        stem = f"{route}_{direction}_{var_slug}_n{n_total}"

        wide_pct.to_csv(outdir / f"{stem}_pct.csv", index=False)
        pivot_cnt[(route, direction, variation)].to_csv(outdir / f"{stem}_cnt.csv", index=False)

        summary_df = summaries[(route, direction, variation)]
        summary_df.to_csv(outdir / f"{stem}_summary.csv", index=False)

        if write_lookups:
            cols = ["Timepoint Order", "Timepoint ID", "Timepoint Description"]
            tp_lookup = summary_df[cols].copy()
            tp_lookup.to_csv(outdir / f"{stem}_timepoints.csv", index=False)

        if write_plots:
            plot_path = outdir / f"{stem}_otp_line.png"
            save_variation_line_plot(
                summary_df, plot_path, dpi=PLOT_DPI, otp_standard=OTP_STANDARD_PCT
            )

        logging.info(
            "Wrote %s* files for %s %s %s (n=%d).", stem, route, direction, variation, n_total
        )


if __name__ == "__main__":
    main()
