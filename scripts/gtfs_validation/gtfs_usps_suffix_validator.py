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
from typing import Any, Iterable, List, Tuple
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FOLDER = Path(r"C:\Path\To\GTFS")        # Folder containing stops.txt
OUTPUT_FOLDER = Path(r"C:\Path\To\Output")     # Where CSVs will be written
ERROR_CSV_NAME = "stop_name_suffix_errors.csv"  # Failed rows
ALL_CSV_NAME = "all_stops_validation.csv"       # Full results
LOG_LEVEL = logging.INFO                        # DEBUG for verbose output

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
    "ALY", "ANX", "ARC", "AVE", "BYU", "BCH", "BND", "BLF", "BLFS", "BTM", "BLVD",
    "BR", "BRG", "BRK", "BRKS", "BG", "BGS", "BYP", "CP", "CYN", "CPE", "CSWY",
    "CTR", "CTRS", "CIR", "CIRS", "CLF", "CLFS", "CLB", "CMN", "CMNS", "COR",
    "CORS", "CRSE", "CT", "CTS", "CV", "CVS", "CRK", "CRES", "CRST", "XING",
    "XRD", "XRDS", "CURV", "DL", "DM", "DV", "DR", "DRS", "EST", "ESTS", "EXPY",
    "EXT", "EXTS", "FALL", "FLS", "FRY", "FLD", "FLDS", "FLT", "FLTS", "FRD",
    "FRDS", "FRST", "FRG", "FRGS", "FRK", "FRKS", "FT", "FWY", "GDN", "GDNS",
    "GTWY", "GLN", "GLNS", "GRN", "GRNS", "GRV", "GRVS", "HBR", "HBRS", "HVN",
    "HTS", "HWY", "HL", "HLS", "HOLW", "INLT", "IS", "ISS", "ISLE", "JCT",
    "JCTS", "KY", "KYS", "KNL", "KNLS", "LK", "LKS", "LAND", "LNDG", "LN",
    "LGT", "LGTS", "LF", "LCK", "LCKS", "LDG", "LOOP", "MALL", "MNR", "MNRS",
    "MDW", "MDWS", "MEWS", "ML", "MLS", "MSN", "MTWY", "MT", "MTN", "MTNS",
    "NCK", "ORCH", "OVAL", "OPAS", "PARK", "PKWY", "PASS", "PSGE", "PATH",
    "PIKE", "PNE", "PNES", "PL", "PLN", "PLNS", "PLZ", "PT", "PTS", "PRT",
    "PRTS", "PR", "RADL", "RAMP", "RNCH", "RPD", "RPDS", "RST", "RDG", "RDGS",
    "RIV", "RD", "RDS", "RTE", "ROW", "RUE", "RUN", "SHL", "SHLS", "SHR",
    "SHRS", "SKWY", "SPG", "SPGS", "SPUR", "SQ", "SQS", "STA", "STRA", "STRM",
    "ST", "STS", "SMT", "TER", "TRWY", "TRCE", "TRAK", "TRFY", "TRL", "TRLR",
    "TUNL", "TPKE", "UPAS", "UN", "UNS", "VLY", "VLYS", "VIA", "VW", "VWS",
    "VLG", "VLGS", "VL", "VIS", "WALK", "WALL", "WAY", "WAYS", "WL", "WLS",
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
    """
    Validate every stop in *stops_path* and return the full results DataFrame.

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









"""
Validates GTFS stop names for capitalization style, USPS suffix compliance, and invalid short words.

Analyzes stop names from GTFS `stops.txt`, classifies capitalization style, checks final words
against USPS suffixes, and flags suspicious short words. Outputs include validation errors and
summary statistics for QA purposes.

Inputs:
    - GTFS 'stops.txt' file

Outputs:
    - CSV of stops with suffix or short word errors
    - CSV of all stops with capitalization style classification
    - Console summary of capitalization style distribution
"""

import os

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# Define input and output paths
INPUT_FOLDER = r"C:\Path\To\Your\System\GTFS_Data"  # Replace with your folder path
OUTPUT_FOLDER = r"C:\Path\To\Your\Output_Folder"  # Replace with your folder path
OUTPUT_FILE = "stop_name_suffix_errors.csv"  # Suffix check output file name
OUTPUT_ALL_STOPS_FILE = (
    "all_stops_by_caps_style.csv"  # Capitalization check output file
)

# Exempt words that are allowed even if they're not USPS suffixes
EXEMPT_WORDS = [
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
]
EXEMPT_WORDS_SET = {word.upper() for word in EXEMPT_WORDS}

# Approved USPS abbreviations
USPS_ABBREVIATIONS = [
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
]

USPS_ABBREVIATIONS_SET = {abbr.upper().strip() for abbr in USPS_ABBREVIATIONS}

# Combined valid short words (USPS suffixes + exempt words)
VALID_SHORT_WORDS_SET = USPS_ABBREVIATIONS_SET.union(EXEMPT_WORDS_SET)

# =============================================================================
# FUNCTIONS
# =============================================================================


def check_capitalization(stop_name):
    """
    Check the capitalization scheme of the stop name.
    Returns one of:
        - 'ALL_LOWERCASE'
        - 'ALL_UPPERCASE'
        - 'PROPER_TITLE_CASE'
        - 'FIRST_LETTER_CAPITALIZED'
        - 'MIXED_CASE'
    """
    stop_name_lower = stop_name.lower()
    stop_name_upper = stop_name.upper()

    # Check for all lowercase
    if stop_name == stop_name_lower:
        return "ALL_LOWERCASE"

    # Check for all uppercase
    if stop_name == stop_name_upper:
        return "ALL_UPPERCASE"

    # Check for proper title case with exceptions for small words
    exceptions = {
        "and",
        "or",
        "the",
        "in",
        "at",
        "by",
        "to",
        "for",
        "of",
        "on",
        "as",
        "a",
        "an",
        "but",
    }
    title_case_words = stop_name.title().split()
    for i in range(1, len(title_case_words)):
        if title_case_words[i].lower() in exceptions:
            title_case_words[i] = title_case_words[i].lower()

    title_case_normalized = " ".join(title_case_words)
    if stop_name == title_case_normalized:
        return "PROPER_TITLE_CASE"

    # Check for first letter capitalization
    if stop_name and stop_name[0].isupper() and stop_name[1:].islower():
        return "FIRST_LETTER_CAPITALIZED"

    # Otherwise, it's mixed case
    return "MIXED_CASE"


def check_usps_suffix(stop_name):
    """
    Check if the stop name ends with a valid USPS suffix, only if the suffix is a short word.
    Returns a tuple (bool, message).
    """
    stop_name_parts = stop_name.split()
    if not stop_name_parts:
        return False, "Empty stop name"

    last_part = stop_name_parts[-1].upper()
    if len(last_part) in [2, 3]:  # Only consider short words
        if last_part not in USPS_ABBREVIATIONS_SET:
            return False, f"Invalid suffix: {last_part}"

    return True, ""


def find_invalid_short_words(stop_name):
    """
    Find two or three-letter words in the stop name not in USPS abbreviations or exempt words.
    Returns a list of invalid short words.
    """
    words = stop_name.split()
    short_words = [word.upper() for word in words if len(word) in [2, 3]]
    invalid_words = [word for word in short_words if word not in VALID_SHORT_WORDS_SET]
    return invalid_words


def validate_stop(stop_row):
    """
    Validate a single stop row.
    Returns a dictionary with potential errors and the capitalization scheme.
    """
    stop_id = stop_row["stop_id"]
    stop_name = stop_row["stop_name"].strip()

    # Check capitalization
    capitalization_scheme = check_capitalization(stop_name)

    # Check USPS suffix
    is_suffix_valid, suffix_message = check_usps_suffix(stop_name)

    # Check invalid short words
    invalid_short_words = find_invalid_short_words(stop_name)

    errors = []
    if not is_suffix_valid:
        errors.append(suffix_message)
    if invalid_short_words:
        errors.append(f"Invalid short words: {', '.join(invalid_short_words)}")

    return {
        "stop_id": stop_id,
        "stop_name": stop_name,
        "capitalization_scheme": capitalization_scheme,
        "errors": errors,
    }


# =============================================================================
# MAIN
# =============================================================================


def main():
    """
    Main entry point to validate GTFS stop names for capitalization and USPS suffix usage.
    """
    output_folder_path = OUTPUT_FOLDER
    output_file_path = os.path.join(output_folder_path, OUTPUT_FILE)
    output_all_stops_file_path = os.path.join(output_folder_path, OUTPUT_ALL_STOPS_FILE)

    input_file_path = os.path.join(INPUT_FOLDER, "stops.txt")
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Load stops data
    stops_df = pd.read_csv(input_file_path, dtype=str)

    # Ensure required columns exist
    required_columns_stops = ["stop_id", "stop_name"]
    missing_columns = [
        col for col in required_columns_stops if col not in stops_df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in stops.txt: {', '.join(missing_columns)}"
        )

    # Validate each stop
    results = stops_df.apply(validate_stop, axis=1, result_type="expand")

    # Aggregate capitalization schemes
    scheme_counts = results["capitalization_scheme"].value_counts()
    total_stops = len(results)

    # Print the percentages for each scheme
    for scheme in [
        "ALL_LOWERCASE",
        "ALL_UPPERCASE",
        "PROPER_TITLE_CASE",
        "FIRST_LETTER_CAPITALIZED",
        "MIXED_CASE",
    ]:
        count = scheme_counts.get(scheme, 0)
        percent = (count / total_stops) * 100
        print(f"Percent of stops with {scheme}: {percent:.2f}%")

    # Separate out errors
    all_errors = []
    for idx, row in results.iterrows():
        if row["errors"]:
            for err in row["errors"]:
                all_errors.append(
                    {
                        "stop_id": row["stop_id"],
                        "stop_name": row["stop_name"],
                        "error": err,
                    }
                )

    # Save errors to CSV if any
    if all_errors:
        errors_df = pd.DataFrame(all_errors)
        errors_df.to_csv(output_file_path, index=False)
        print(f"Errors found. Report saved to {output_file_path}")
    else:
        print("No errors found.")

    # Export all stops with their ids, names, and capitalization scheme
    export_df = results[["stop_id", "stop_name", "capitalization_scheme"]]
    export_df.to_csv(output_all_stops_file_path, index=False)
    print(f"All stops exported to {output_all_stops_file_path}")


if __name__ == "__main__":
    main()
