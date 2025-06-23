# tests/test_fred_processor_e2e.py
"""
End-to-end smoke-test for the FRED unemployment-rate processor.

What the test does
------------------
1. Builds a _small, deterministic_ CSV that mimics a FRED download:
   • A proper ``observation_date`` column (monthly, 2019-01-01 … 2025-01-01).
   • Exactly **one** data-series column whose name is metro-area–specific
     (here: ``WASH911URN`` to match the DC example).
2. Monkey-patches the processor’s globals so it
   • reads the dummy CSV,
   • writes everything to a temporary output folder,
   • uses the date window 2020-01-01 … 2024-12-01.
3. Forces Matplotlib into the non-interactive “Agg” backend so the test runs
   headless.
4. Executes ``main()``.
5. Asserts that:
   • the Excel file exists **and** contains **exactly 60 rows**
     (5 years × 12 months),
   • all rows fall inside the requested date window, and
   • both JPEG charts were written.

The test is offline, finishes in < 2 s, and works unchanged in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# ==============================================================================
# 1. make sure the project root is importable
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


def _build_dummy_csv(csv_path: Path) -> None:
    """Create a tiny, deterministic FRED-style CSV."""
    dates = pd.date_range("2019-01-01", "2025-01-01", freq="MS")
    df = pd.DataFrame(
        {
            "observation_date": dates.strftime("%Y-%m-%d"),
            "WASH911URN": range(len(dates)),  # 0, 1, 2, … – values don’t matter
        }
    )
    df.to_csv(csv_path, index=False)

# ==============================================================================
# The test
# ==============================================================================


def test_unemployment_processor_e2e(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    GIVEN a minimal FRED-style CSV
    WHEN  the processor’s ``main()`` runs
    THEN  the expected Excel + JPEG files appear and contain the right data.
    """
    # ------------------------------------------------------------------
    # 1. Arrange – build input & locate the processor
    # ------------------------------------------------------------------
    csv_path = tmp_path / "fake_unrate.csv"
    out_dir = tmp_path / "out"
    _build_dummy_csv(csv_path)
    out_dir.mkdir()

    # Import AFTER creating the CSV so the module can read it
    #
    # Update the import below if your module lives elsewhere, e.g.:
    #   import scripts.fred_unemployment_processor as proc
    # or simply:
    #   import fred_unemployment_processor as proc
    import scripts.external_variables.fred_unemployment_processor as proc

    # ------------------------------------------------------------------
    # 2. Monkey-patch configuration
    # ------------------------------------------------------------------
    monkeypatch.setattr(proc, "CSV_FILE_PATH", str(csv_path))
    monkeypatch.setattr(proc, "OUTPUT_FOLDER", str(out_dir))
    monkeypatch.setattr(proc, "START_DATE", "2020-01-01")
    monkeypatch.setattr(proc, "END_DATE", "2024-12-01")

    # Headless Matplotlib
    proc.plt.switch_backend("Agg")

    # ------------------------------------------------------------------
    # 3. Act – run the full workflow
    # ------------------------------------------------------------------
    proc.main()

    # ------------------------------------------------------------------
    # 4. Assert – expected artefacts exist & look sane
    # ------------------------------------------------------------------
    suffix = "2020_01-2024_12"
    excel_file = out_dir / f"filtered_data_{suffix}.xlsx"
    line_jpeg = out_dir / f"line_graph_{suffix}.jpeg"
    yearly_jpeg = out_dir / f"yearly_line_graph_{suffix}.jpeg"

    # 4a. Files were written
    assert excel_file.is_file(), "Excel export missing."
    assert line_jpeg.is_file(), "Continuous-line JPEG missing."
    assert yearly_jpeg.is_file(), "Year-over-year JPEG missing."

    # 4b. Excel contains the expected 60 monthly rows, all inside the window
    df = pd.read_excel(excel_file, engine="openpyxl")
    df["observation_date"] = pd.to_datetime(df["observation_date"])

    assert len(df) == 60, "Filtered Excel should hold exactly 60 months."
    assert df["observation_date"].min() >= pd.Timestamp("2020-01-01")
    assert df["observation_date"].max() <= pd.Timestamp("2024-12-01")
