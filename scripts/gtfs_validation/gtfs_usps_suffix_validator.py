"""Validate GTFS stop names for USPS-compliant suffixes and invalid short words.

This edition is designed for interactive use in Jupyter notebooks or scripts:
update the *CONFIGURATION* section below and run the cell.

Outputs
-------
1. CSV of stops that failed validation
2. CSV of all stops with a Boolean ``has_errors`` column
3. Log summary (INFO level by default; set LOG_LEVEL = logging.DEBUG for more detail)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FOLDER = Path(r"C:\Path\To\GTFS")  # Folder containing stops.txt
OUTPUT_FOLDER = Path(r"C:\Path\To\Output")  # Where CSVs will be written
ERROR_CSV_NAME = "stop_name_suffix_errors.csv"  # Failed rows
ALL_CSV_NAME = "all_stops_validation.csv"  # Full results
LOG_LEVEL = logging.INFO  # DEBUG for verbose output

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

EXEMPT_WORDS: Tuple[str, ...] = (
    "AND",
    "VAN",
    "LA",
    "OX",
    "OLD",
    "BAY",
    "FOX",
    "LEE",
    "OAK",
    "ELM",
    "GUM",
    "MAR",
    "THE",
    "RED",
    "OWL",
    "NEW",
)

USPS_ABBREVIATIONS: Tuple[str, ...] = (
    # Full official list — keep alphabetized for readability
    "ALY",
    "ANX",
    "ARC",
    "AVE",
    "BYU",
    "BCH",
    "BND",
    "BLF",
    "BLFS",
    "BTM",
    "BLVD",
    "BR",
    "BRG",
    "BRK",
    "BRKS",
    "BG",
    "BGS",
    "BYP",
    "CP",
    "CYN",
    "CPE",
    "CSWY",
    "CTR",
    "CTRS",
    "CIR",
    "CIRS",
    "CLF",
    "CLFS",
    "CLB",
    "CMN",
    "CMNS",
    "COR",
    "CORS",
    "CRSE",
    "CT",
    "CTS",
    "CV",
    "CVS",
    "CRK",
    "CRES",
    "CRST",
    "XING",
    "XRD",
    "XRDS",
    "CURV",
    "DL",
    "DM",
    "DV",
    "DR",
    "DRS",
    "EST",
    "ESTS",
    "EXPY",
    "EXT",
    "EXTS",
    "FALL",
    "FLS",
    "FRY",
    "FLD",
    "FLDS",
    "FLT",
    "FLTS",
    "FRD",
    "FRDS",
    "FRST",
    "FRG",
    "FRGS",
    "FRK",
    "FRKS",
    "FT",
    "FWY",
    "GDN",
    "GDNS",
    "GTWY",
    "GLN",
    "GLNS",
    "GRN",
    "GRNS",
    "GRV",
    "GRVS",
    "HBR",
    "HBRS",
    "HVN",
    "HTS",
    "HWY",
    "HL",
    "HLS",
    "HOLW",
    "INLT",
    "IS",
    "ISS",
    "ISLE",
    "JCT",
    "JCTS",
    "KY",
    "KYS",
    "KNL",
    "KNLS",
    "LK",
    "LKS",
    "LAND",
    "LNDG",
    "LN",
    "LGT",
    "LGTS",
    "LF",
    "LCK",
    "LCKS",
    "LDG",
    "LOOP",
    "MALL",
    "MNR",
    "MNRS",
    "MDW",
    "MDWS",
    "MEWS",
    "ML",
    "MLS",
    "MSN",
    "MTWY",
    "MT",
    "MTN",
    "MTNS",
    "NCK",
    "ORCH",
    "OVAL",
    "OPAS",
    "PARK",
    "PKWY",
    "PASS",
    "PSGE",
    "PATH",
    "PIKE",
    "PNE",
    "PNES",
    "PL",
    "PLN",
    "PLNS",
    "PLZ",
    "PT",
    "PTS",
    "PRT",
    "PRTS",
    "PR",
    "RADL",
    "RAMP",
    "RNCH",
    "RPD",
    "RPDS",
    "RST",
    "RDG",
    "RDGS",
    "RIV",
    "RD",
    "RDS",
    "RTE",
    "ROW",
    "RUE",
    "RUN",
    "SHL",
    "SHLS",
    "SHR",
    "SHRS",
    "SKWY",
    "SPG",
    "SPGS",
    "SPUR",
    "SQ",
    "SQS",
    "STA",
    "STRA",
    "STRM",
    "ST",
    "STS",
    "SMT",
    "TER",
    "TRWY",
    "TRCE",
    "TRAK",
    "TRFY",
    "TRL",
    "TRLR",
    "TUNL",
    "TPKE",
    "UPAS",
    "UN",
    "UNS",
    "VLY",
    "VLYS",
    "VIA",
    "VW",
    "VWS",
    "VLG",
    "VLGS",
    "VL",
    "VIS",
    "WALK",
    "WALL",
    "WAY",
    "WAYS",
    "WL",
    "WLS",
)

# Pre-computed sets for constant-time look-ups
EXEMPT_WORDS_SET = {w.upper() for w in EXEMPT_WORDS}
USPS_ABBREVIATIONS_SET = {u.upper() for u in USPS_ABBREVIATIONS}
VALID_SHORT_WORDS_SET = USPS_ABBREVIATIONS_SET | EXEMPT_WORDS_SET

# =============================================================================
# FUNCTIONS
# =============================================================================

def _check_usps_suffix(stop_name: str) -> tuple[bool, str | None]:
    """Return ``False`` and a message if the last token isn’t a valid USPS suffix."""
    tokens = stop_name.split()
    if not tokens:
        return False, "Empty stop name"

    candidate = tokens[-1].upper()
    if len(candidate) in (2, 3) and candidate not in USPS_ABBREVIATIONS_SET:
        return False, f"Invalid USPS suffix: {candidate}"
    return True, None


def _find_invalid_short_words(stop_name: str) -> List[str]:
    """Return 2- or 3-letter words that are *not* in the approved sets."""
    return [
        t.upper()
        for t in stop_name.split()
        if len(t) in (2, 3) and t.upper() not in VALID_SHORT_WORDS_SET
    ]


def _validate_row(row: pd.Series) -> dict[str, Any]:
    """Apply all validation rules to a single pandas row."""
    stop_name = str(row["stop_name"]).strip()
    errors: List[str] = []

    # USPS suffix
    ok, msg = _check_usps_suffix(stop_name)
    if not ok and msg:
        errors.append(msg)

    # Other short words
    bad_words = _find_invalid_short_words(stop_name)
    if bad_words:
        errors.append(f"Invalid short words: {', '.join(bad_words)}")

    return {
        "stop_id": row["stop_id"],
        "stop_name": stop_name,
        "errors": errors,
        "has_errors": bool(errors),
    }


# -----------------------------------------------------------------------------
# WORKFLOW FUNCTIONS
# -----------------------------------------------------------------------------

def validate_stops(stops_path: Path) -> pd.DataFrame:
    """Validate every stop in *stops_path* and return the full results DataFrame.

    Raises
    ------
    FileNotFoundError
        If *stops_path* is missing.
    ValueError
        If required columns are missing.
    """
    logger = logging.getLogger(__name__)

    if not stops_path.is_file():
        raise FileNotFoundError(stops_path)

    logger.info("Reading %s", stops_path)
    df = pd.read_csv(stops_path, dtype=str)

    required = {"stop_id", "stop_name"}
    if missing := (required - set(df.columns)):
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    logger.debug("Validating %d stops …", len(df))
    return df.apply(_validate_row, axis=1, result_type="expand")


def write_outputs(results: pd.DataFrame) -> None:
    """Write both CSV outputs and emit log summaries."""
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    errors = results[results["has_errors"]]
    error_path = OUTPUT_FOLDER / ERROR_CSV_NAME
    all_path = OUTPUT_FOLDER / ALL_CSV_NAME

    if not errors.empty:
        errors.to_csv(error_path, index=False)
        logging.info("Saved %d error rows → %s", len(errors), error_path)
    else:
        logging.info("No validation errors detected.")

    results.drop(columns=["errors"]).to_csv(all_path, index=False)
    logging.info("Saved full validation results → %s", all_path)

    logging.info(
        "Finished: %d / %d stops (%.2f%%) have issues.",
        errors.shape[0],
        results.shape[0],
        (errors.shape[0] / results.shape[0] * 100) if results.shape[0] else 0.0,
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Run the GTFS stop-name validation using the CONFIGURATION constants."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(levelname)s | %(name)s | %(message)s",
    )
    stops_file = INPUT_FOLDER / "stops.txt"
    results = validate_stops(stops_file)
    write_outputs(results)


if __name__ == "__main__":  # pragma: no cover
    main()
