import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401

    HAS_MATPLOTLIB = True
except ImportError:
    mock_mpl = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock
    sys.modules["matplotlib"] = mock_mpl()
    sys.modules["matplotlib.pyplot"] = mock_mpl()
    HAS_MATPLOTLIB = False

from scripts.operations_tools.otp_monthly_trends_export import (
    _normalize_dow,
    clean_route,
    coerce_numeric,
    compute_trend_summary,
    export_trend_logs,
    format_trend_log,
    month_to_period,
    parse_current_yy_mm,
    plot_series_for_groups,
    process,
    standardize_columns,
)

FIXTURE_CSV = Path("tests/fixtures/CLEVER_Runtime_and_OTP_by_Month_sample.csv")
CURRENT_YY_MM = "25-10"  # Oct 2025


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_CSV, dtype=str)


@pytest.fixture
def processed_df(raw_df) -> pd.DataFrame:
    norm = standardize_columns(raw_df)
    return process(norm, CURRENT_YY_MM)


# ---------------------------------------------------------------------------
# parse_current_yy_mm
# ---------------------------------------------------------------------------


def test_parse_current_yy_mm_valid():
    assert parse_current_yy_mm("25-10") == (2025, 10)
    assert parse_current_yy_mm("00-01") == (2000, 1)


def test_parse_current_yy_mm_invalid_format():
    with pytest.raises(ValueError):
        parse_current_yy_mm("2025-10")


def test_parse_current_yy_mm_invalid_month():
    with pytest.raises(ValueError):
        parse_current_yy_mm("25-13")


# ---------------------------------------------------------------------------
# clean_route
# ---------------------------------------------------------------------------


def test_clean_route_strips_suffix():
    assert clean_route("   101 - Downtown") == "101"


def test_clean_route_alphanumerics_only():
    assert clean_route("  303 - Express") == "303"


def test_clean_route_uppercase():
    assert clean_route("abc - something") == "ABC"


def test_clean_route_none():
    assert clean_route(None) == ""


def test_clean_route_no_dash():
    assert clean_route("  202  ") == "202"


# ---------------------------------------------------------------------------
# month_to_period
# ---------------------------------------------------------------------------


def test_month_to_period_within_ref_year():
    # Oct 2025 is ref; Jan..Oct -> 2025
    assert month_to_period("Jan", 2025, 10) == "25-01"
    assert month_to_period("Oct", 2025, 10) == "25-10"


def test_month_to_period_prior_year():
    # Nov and Dec are after ref month Oct -> 2024
    assert month_to_period("Nov", 2025, 10) == "24-11"
    assert month_to_period("Dec", 2025, 10) == "24-12"


def test_month_to_period_case_insensitive():
    assert month_to_period("APR", 2025, 10) == "25-04"
    assert month_to_period("april", 2025, 10) == "25-04"


def test_month_to_period_invalid():
    with pytest.raises(ValueError):
        month_to_period("Xyz", 2025, 10)


# ---------------------------------------------------------------------------
# coerce_numeric
# ---------------------------------------------------------------------------


def test_coerce_numeric_handles_commas():
    s = pd.Series(["1,234", "2,029.00", "500"])
    result = coerce_numeric(s)
    assert result.tolist() == pytest.approx([1234.0, 2029.0, 500.0])


def test_coerce_numeric_handles_blanks():
    s = pd.Series(["", "100", " "])
    result = coerce_numeric(s)
    assert result[0] != result[0]  # NaN
    assert result[1] == 100.0


# ---------------------------------------------------------------------------
# _normalize_dow
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Monday", "Monday"),
        ("mon", "Monday"),
        ("MON", "Monday"),
        ("saturday", "Saturday"),
        ("Sat", "Saturday"),
        ("SUN", "Sunday"),
        ("thu", "Thursday"),
        ("FRIDAY", "Friday"),
    ],
)
def test_normalize_dow(raw, expected):
    assert _normalize_dow(raw) == expected


# ---------------------------------------------------------------------------
# standardize_columns
# ---------------------------------------------------------------------------


def test_standardize_columns_maps_names(raw_df):
    result = standardize_columns(raw_df)
    expected = {"route", "direction", "month_label", "dow", "on_time", "early", "late"}
    assert expected.issubset(set(result.columns))


def test_standardize_columns_raises_on_missing():
    bad = pd.DataFrame({"Route": ["101"], "Direction": ["EB"]})
    with pytest.raises(KeyError):
        standardize_columns(bad)


# ---------------------------------------------------------------------------
# process
# ---------------------------------------------------------------------------


def test_process_output_columns(processed_df):
    expected = {
        "route_raw",
        "route_clean",
        "direction",
        "month_label",
        "period",
        "dow",
        "on_time",
        "early",
        "late",
        "total_trips",
        "pct_on_time",
        "pct_early",
        "pct_late",
    }
    assert expected.issubset(set(processed_df.columns))


def test_process_route_clean_values(processed_df):
    routes = set(processed_df["route_clean"].unique())
    assert "101" in routes
    assert "202" in routes
    assert "303" in routes


def test_process_period_format(processed_df):
    import re

    for p in processed_df["period"].unique():
        assert re.fullmatch(r"\d{2}-\d{2}", p), f"Bad period format: {p}"


def test_process_percentages_sum_to_100(processed_df):
    total = processed_df["pct_on_time"] + processed_df["pct_early"] + processed_df["pct_late"]
    assert (total.dropna() - 100.0).abs().max() < 1e-6


def test_process_no_negative_percentages(processed_df):
    for col in ("pct_on_time", "pct_early", "pct_late"):
        assert (processed_df[col].dropna() >= 0).all()


def test_process_total_trips(processed_df):
    expected = processed_df["on_time"] + processed_df["early"] + processed_df["late"]
    diff = (processed_df["total_trips"] - expected).abs()
    assert diff.max() < 1e-6


def test_process_drops_blank_rows(raw_df):
    # Append some blank rows to the fixture
    norm = standardize_columns(raw_df)
    blank = pd.DataFrame(
        [
            {
                "route": "",
                "direction": "",
                "month_label": "",
                "dow": "",
                "on_time": "",
                "early": "",
                "late": "",
            }
        ]
    )
    norm_with_blanks = pd.concat([norm, blank], ignore_index=True)
    result = process(norm_with_blanks, CURRENT_YY_MM)
    assert len(result) == len(process(norm, CURRENT_YY_MM))


def test_process_blacklist(raw_df):
    norm = standardize_columns(raw_df)
    result = process(norm, CURRENT_YY_MM, blacklisted_routes=frozenset(["101"]))
    assert "101" not in result["route_clean"].values


def test_process_sorted_output(processed_df):
    # Output should be sorted by route_clean, direction, period
    sort_keys = processed_df[["route_clean", "direction", "period"]].values.tolist()
    assert sort_keys == sorted(sort_keys)


# ---------------------------------------------------------------------------
# compute_trend_summary
# ---------------------------------------------------------------------------


def test_compute_trend_summary_one_row_per_group(processed_df):
    summary = compute_trend_summary(processed_df, otp_standard=0.85)
    n_groups = processed_df.groupby(["route_clean", "direction"], dropna=False).ngroups
    assert len(summary) == n_groups


def test_compute_trend_summary_columns(processed_df):
    summary = compute_trend_summary(processed_df, otp_standard=0.85)
    expected_cols = {
        "route_clean",
        "direction",
        "n_periods_wd",
        "trend_wd",
        "current_wd",
        "mean_wd",
        "trend_sat",
        "current_sat",
        "trend_sun",
        "current_sun",
        "below_standard",
        "declining",
        "concern_score",
    }
    assert expected_cols.issubset(set(summary.columns))


def test_compute_trend_summary_concern_score_nonneg(processed_df):
    summary = compute_trend_summary(processed_df, otp_standard=0.85)
    assert (summary["concern_score"] >= 0).all()


def test_compute_trend_summary_percentages_in_range(processed_df):
    summary = compute_trend_summary(processed_df, otp_standard=0.85)
    for col in ("current_wd", "mean_wd", "current_sat", "current_sun"):
        vals = summary[col].dropna()
        assert (vals >= 0).all() and (vals <= 100).all(), f"{col} out of [0,100]"


# ---------------------------------------------------------------------------
# format_trend_log
# ---------------------------------------------------------------------------


def test_format_trend_log_contains_header(processed_df):
    summary = compute_trend_summary(processed_df, otp_standard=0.85)
    text = format_trend_log(
        summary,
        title="Test Title",
        current_yy_mm=CURRENT_YY_MM,
        otp_standard=0.85,
        period_min="25-01",
        period_max="25-10",
    )
    assert "Test Title" in text
    assert "ROUTE" in text
    assert "TREND_WD" in text


def test_format_trend_log_all_routes_present(processed_df):
    summary = compute_trend_summary(processed_df, otp_standard=0.85)
    text = format_trend_log(
        summary,
        title="All",
        current_yy_mm=CURRENT_YY_MM,
        otp_standard=0.85,
        period_min="25-01",
        period_max="25-10",
    )
    for route in ("101", "202", "303"):
        assert route in text


# ---------------------------------------------------------------------------
# export_trend_logs
# ---------------------------------------------------------------------------


def test_export_trend_logs_creates_files(processed_df, tmp_path):
    path_all, path_concerning = export_trend_logs(
        processed_df,
        tmp_path,
        current_yy_mm=CURRENT_YY_MM,
        otp_standard=0.85,
        concerning_pct=0.10,
    )
    assert path_all.exists()
    assert path_concerning.exists()


def test_export_trend_logs_all_has_more_routes(processed_df, tmp_path):
    path_all, path_concerning = export_trend_logs(
        processed_df,
        tmp_path,
        current_yy_mm=CURRENT_YY_MM,
        otp_standard=0.85,
        concerning_pct=0.10,
    )
    all_text = path_all.read_text(encoding="utf-8")
    concerning_text = path_concerning.read_text(encoding="utf-8")
    # "All routes" log should be at least as long as the concerning subset
    assert len(all_text) >= len(concerning_text)


# ---------------------------------------------------------------------------
# plot_series_for_groups
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
def test_plot_series_for_groups_creates_pngs(processed_df, tmp_path):
    plot_series_for_groups(processed_df, tmp_path, otp_standard=0.85)
    pngs = list(tmp_path.glob("*.png"))
    assert len(pngs) > 0


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
def test_plot_series_for_groups_weekday_saturday_sunday(processed_df, tmp_path):
    plot_series_for_groups(processed_df, tmp_path, otp_standard=0.85)
    pngs = {p.name for p in tmp_path.glob("*.png")}
    weekday_pngs = [n for n in pngs if "Weekdays" in n]
    saturday_pngs = [n for n in pngs if "Saturday" in n]
    sunday_pngs = [n for n in pngs if "Sunday" in n]
    assert len(weekday_pngs) > 0
    assert len(saturday_pngs) > 0
    assert len(sunday_pngs) > 0


# ---------------------------------------------------------------------------
# Integration: main()
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
def test_main_end_to_end(tmp_path):
    from scripts.operations_tools.otp_monthly_trends_export import main

    out_table = tmp_path / "tables"
    out_plots = tmp_path / "plots"
    main(
        [
            "--input",
            str(FIXTURE_CSV),
            "--out-table",
            str(out_table),
            "--out-plots",
            str(out_plots),
            "--current",
            CURRENT_YY_MM,
            "--otp-standard",
            "0.85",
            "--concerning-pct",
            "0.10",
        ]
    )

    assert (out_table / "otp_processed.csv").exists()
    assert (out_table / "otp_trend_summary_all.txt").exists()
    assert (out_table / "otp_trend_summary_concerning.txt").exists()
    assert len(list(out_plots.glob("*.png"))) > 0

    df_out = pd.read_csv(out_table / "otp_processed.csv")
    assert len(df_out) > 0
    assert "route_clean" in df_out.columns
    assert "pct_on_time" in df_out.columns
