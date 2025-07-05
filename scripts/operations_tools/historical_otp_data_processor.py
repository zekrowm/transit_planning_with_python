"""Process historical on-time-performance (OTP) into trend plots.

Reads a monthly On-Time-Performance (OTP) CSV, computes basic
percentages, and writes one JPEG trend plot per Route × Direction.

Typical use: run inside ArcGIS Pro’s Python or any standalone
environment that has pandas, numpy, and matplotlib.

Outputs
-------
└── <PLOT_OUTPUT_DIR>/
    └── <Route>_<Dir>_on_time_percentage.jpeg
"""

from __future__ import annotations
from typing import Tuple
from io import StringIO
import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

OTP_INPUT_FILE: str = r"Path\To\Your\CLEVER_Runtime_and_OTP_Trip_Level_Data.csv"
PLOT_OUTPUT_DIR: str = r"Path\To\Your\Output\Folder"

FULL_RANGE_START: str = "2025-01-01"   # inclusive first month
FULL_RANGE_END:   str = "2025-12-31"   # inclusive last  month

# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------

ROUTE_COL      = "Route"
DIRECTION_COL  = "Direction"
YEAR_COL       = "Year"
MONTH_COL      = "Month"
ON_TIME_COL    = "Sum # On Time"
EARLY_COL      = "Sum # Early"
LATE_COL       = "Sum # Late"

# Plot look & feel
FIG_SIZE       = (12, 6)
ROTATION       = 45
PERCENT_LINES  = [95, 85, 75]          # horizontal guide lines
LINE_STYLE     = "--"
LINE_WIDTH     = 0.7
LINE_COLOUR    = "r"
FONT_SIZE      = 9
MARKER_STYLE   = "o"
LINE_STYLE_TREND = "-"

# =============================================================================
# FUNCTIONS
# =============================================================================

def flag_problem_routes(
    df: pd.DataFrame,
    *,
    twelve_month_floor: float = 80.0,
    six_month_drop: float = 10.0,
    six_month_avg_floor: float = 85.0,
) -> pd.DataFrame:
    """Identify Route × Direction pairs with sustained or sharply worsening OTP."""
    df = df.sort_values("Date")
    reasons: list[tuple[str, str, str]] = []

    for (route, direction), g in df.groupby([ROUTE_COL, DIRECTION_COL]):
        recent_12 = g.tail(12).reset_index(drop=True)
        recent_6  = g.tail(6).reset_index(drop=True)

        sustained_12 = (
            len(recent_12) == 12 and
            (recent_12["Percent On Time"] < twelve_month_floor).all()
        )

        trend_flag = False
        if len(recent_6) == 6:
            drop = recent_6["Percent On Time"].iloc[0] - recent_6["Percent On Time"].iloc[-1]
            mean6 = recent_6["Percent On Time"].mean()
            trend_flag = (drop >= six_month_drop) and (mean6 < six_month_avg_floor)

        if sustained_12 or trend_flag:
            explain = []
            if sustained_12:
                explain.append(f"all last 12 mo < {twelve_month_floor}%")
            if trend_flag:
                explain.append(
                    f"drop ≥ {six_month_drop} pp in last 6 mo & 6-mo avg < {six_month_avg_floor}%"
                )
            reasons.append((route, direction, ", ".join(explain)))

    return pd.DataFrame(reasons, columns=[ROUTE_COL, DIRECTION_COL, "Reason"])


def write_problem_log(
    problems: pd.DataFrame,
    output_dir: str,
    fname: str = "otp_problem_routes.txt",
) -> str:
    """Write *problems* DataFrame to a simple text file; return full path."""
    log_path = os.path.join(output_dir, fname)
    with open(log_path, "w", encoding="utf-8") as f:
        if problems.empty:
            f.write("No routes triggered OTP alarms.\n")
        else:
            f.write("Routes/Directions with OTP alarms\n")
            f.write("Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M") + "\n\n")
            for _, row in problems.iterrows():
                f.write(f"- {row[ROUTE_COL]} / {row[DIRECTION_COL]}: {row['Reason']}\n")
    return log_path


def load_csv(path: str) -> pd.DataFrame:
    """Return a DataFrame, or raise FileNotFoundError / ValueError."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"OTP file not found:\n{path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"OTP file is empty:\n{path}")

    print(f"✓ CSV loaded: {path}")
    return df


def clean_route_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Trim anything after the first “-” (e.g. “10A-Otis St” → “10A”)."""
    df[col] = df[col].astype(str).str.split("-").str[0].str.strip()
    return df


def enforce_int(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Force *columns* to int—missing or bad values become 0."""
    for c in columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per Route × Direction × month and recompute %."""
    agg_cols = [ON_TIME_COL, EARLY_COL, LATE_COL]
    gb = df.groupby([ROUTE_COL, DIRECTION_COL, "Date"], as_index=False)[agg_cols].sum()
    gb = add_totals_and_percents(gb)
    gb[YEAR_COL] = gb["Date"].dt.year
    gb[MONTH_COL] = gb["Date"].dt.month_name()
    return gb


def add_totals_and_percents(df: pd.DataFrame) -> pd.DataFrame:
    """Add Sum All Trips + %, returns the mutated df."""
    df["Sum All Trips"] = df[ON_TIME_COL] + df[EARLY_COL] + df[LATE_COL]

    for raw, pct_name in [
        (ON_TIME_COL, "Percent On Time"),
        (EARLY_COL,   "Percent Early"),
        (LATE_COL,    "Percent Late"),
    ]:
        df[pct_name] = (
            (df[raw] / df["Sum All Trips"].replace(0, np.nan)) * 100
        ).fillna(0)

    return df


def attach_date(df: pd.DataFrame) -> pd.DataFrame:
    """Combine Year + Month into df['Date'] (month-start)."""
    df["YY-MM"] = df[YEAR_COL].astype(str) + "-" + df[MONTH_COL].str[:3].str.capitalize()
    df["Date"]  = pd.to_datetime(df["YY-MM"] + "-01", format="%Y-%b-%d")
    return df.sort_values("Date")


def y_limits(df: pd.DataFrame) -> tuple[float, float]:
    """Global y-axis range with small padding."""
    ymin = max(df["Percent On Time"].min() - 5, 50)
    ymax = df["Percent On Time"].max() + 5
    return ymin, ymax


def sanitize(text: str) -> str:
    """Make text safe as a filename."""
    return text.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _normalise_otp_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adapt input layout to expected field names."""
    df.dropna(how="all", inplace=True)

    if "Route" not in df and "Branch" in df:
        df["Route"] = df["Branch"].astype(str).str.strip()

    if "Year" not in df and "Year Month" in df:
        ym = pd.to_numeric(df["Year Month"], errors="coerce").dropna().astype(int)
        df["Year"] = (ym // 100).astype(int)
        df["Month"] = ((ym % 100).astype(int).map(lambda m: datetime(1900, m, 1).strftime("%B")))

    return df


def plot_group(
    g: pd.DataFrame,
    route: str,
    direction: str,
    ylims: tuple[float, float],
    full_range: pd.DatetimeIndex,
) -> None:
    """Create & save one JPEG trend plot for the group."""
    plt.figure(figsize=FIG_SIZE)

    plt.plot(
        g["Date"],
        g["Percent On Time"],
        marker=MARKER_STYLE,
        linestyle=LINE_STYLE_TREND,
    )

    plt.ylim(*ylims)
    plt.xlim(full_range[0], full_range[-1])
    plt.xticks(full_range, [d.strftime("%b %Y") for d in full_range], rotation=ROTATION)

    for y in PERCENT_LINES:
        plt.axhline(y, color=LINE_COLOUR, linestyle=LINE_STYLE, linewidth=LINE_WIDTH)
        plt.text(
            full_range[-1], y, f"{y}%",
            ha="left", va="center", fontsize=FONT_SIZE, color=LINE_COLOUR,
        )

    title = f"On-Time % – Route {route} / {direction}"
    plt.title(title)
    plt.xlabel("Month")
    plt.ylabel("Percent On Time")
    plt.tight_layout()

    fname = f"{sanitize(route)}_{sanitize(direction)}_on_time_percentage.jpeg"
    plt.savefig(os.path.join(PLOT_OUTPUT_DIR, fname), format="jpeg")
    plt.close()
    print(f"  • plot saved → {fname}")


def process_otp() -> None:
    """End-to-end OTP load → normalise → aggregate → QC → plot."""
    df = load_csv(OTP_INPUT_FILE)
    df = _normalise_otp_columns(df)
    df = clean_route_column(df, ROUTE_COL)
    df = enforce_int(df, [ON_TIME_COL, EARLY_COL, LATE_COL])
    df = add_totals_and_percents(df)
    df = attach_date(df)
    df = monthly_summary(df)

    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    export_csv = os.path.join(PLOT_OUTPUT_DIR, "otp_monthly_summary.csv")
    df.to_csv(export_csv, index=False)
    print(f"✓ monthly summary written → {export_csv}")

    dupes = df.duplicated(subset=[ROUTE_COL, DIRECTION_COL, "Date"])
    if dupes.any():
        raise ValueError(
            "Duplicate Route/Direction/Date rows detected:\n"
            f"{df[dupes].head()}"
        )

    problems = flag_problem_routes(df)
    log_file = write_problem_log(problems, PLOT_OUTPUT_DIR)
    print(f"✓ OTP problem log written → {log_file}")
        
    full_range = pd.date_range(start=FULL_RANGE_START, end=FULL_RANGE_END, freq="MS")
    ylims = y_limits(df)

    var_df = df.groupby([ROUTE_COL, DIRECTION_COL])["Percent On Time"].agg(["max", "min"])
    var_df["Range"] = var_df["max"] - var_df["min"]
    print("\nRoutes with highest OTP variability:")
    print(var_df.sort_values("Range", ascending=False).head(10))

    for (route, direction), group in df.groupby([ROUTE_COL, DIRECTION_COL]):
        plot_group(group, route, direction, ylims, full_range)

    print(f"\nDone – plots saved to: {PLOT_OUTPUT_DIR}")


if __name__ == "__main__":
    process_otp()
