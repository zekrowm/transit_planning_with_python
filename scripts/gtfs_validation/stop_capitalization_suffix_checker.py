"""
Script Name:
        gtfs_stop_capitalization_suffix_checker.py

Purpose:
        Validates stop names in GTFS data for correct capitalization,
        USPS suffix usage, and the presence of invalid short words.

Inputs:
        1. GTFS 'stops.txt' file located in the specified INPUT_FOLDER.

Outputs:
        1. A CSV file ('stop_name_suffix_errors.csv') detailing stops with
           suffix or short word errors.
        2. A CSV file ('all_stops_by_caps_style.csv') listing all stops
           with their determined capitalization style.
        3. Console printout of capitalization scheme percentages and
           error status.

Dependencies:
        os, pandas
"""

import os

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# Define input and output paths
INPUT_FOLDER = r"C:\Path\To\Your\System\GTFS_Data"  # Replace with your folder path
OUTPUT_FOLDER = r"C:\Path\To\Your\Output_Folder"    # Replace with your folder path
OUTPUT_FILE = 'stop_name_suffix_errors.csv'         # Suffix check output file name
OUTPUT_ALL_STOPS_FILE = 'all_stops_by_caps_style.csv'  # Capitalization check output file

# Exempt words that are allowed even if they're not USPS suffixes
EXEMPT_WORDS = [
    "AND", "VAN", "LA", "OX", "OLD", "BAY", "FOX", "LEE", "OAK", "ELM",
    "GUM", "MAR", "THE", "RED", "OWL", "NEW"
]
EXEMPT_WORDS_SET = {word.upper() for word in EXEMPT_WORDS}

# Approved USPS abbreviations
USPS_ABBREVIATIONS = [
    "ALY", "ANX", "ARC", "AVE", "BYU", "BCH", "BND", "BLF", "BLFS", "BTM", "BLVD",
    "BR", "BRG", "BRK", "BRKS", "BG", "BGS", "BYP", "CP", "CYN", "CPE", "CSWY", "CTR",
    "CTRS", "CIR", "CIRS", "CLF", "CLFS", "CLB", "CMN", "CMNS", "COR", "CORS", "CRSE",
    "CT", "CTS", "CV", "CVS", "CRK", "CRES", "CRST", "XING", "XRD", "XRDS", "CURV",
    "DL", "DM", "DV", "DR", "DRS", "EST", "ESTS", "EXPY", "EXT", "EXTS", "FALL", "FLS",
    "FRY", "FLD", "FLDS", "FLT", "FLTS", "FRD", "FRDS", "FRST", "FRG", "FRGS", "FRK",
    "FRKS", "FT", "FWY", "GDN", "GDNS", "GTWY", "GLN", "GLNS", "GRN", "GRNS", "GRV",
    "GRVS", "HBR", "HBRS", "HVN", "HTS", "HWY", "HL", "HLS", "HOLW", "INLT", "IS",
    "ISS", "ISLE", "JCT", "JCTS", "KY", "KYS", "KNL", "KNLS", "LK", "LKS", "LAND",
    "LNDG", "LN", "LGT", "LGTS", "LF", "LCK", "LCKS", "LDG", "LOOP", "MALL", "MNR",
    "MNRS", "MDW", "MDWS", "MEWS", "ML", "MLS", "MSN", "MTWY", "MT", "MTN", "MTNS",
    "NCK", "ORCH", "OVAL", "OPAS", "PARK", "PKWY", "PASS", "PSGE", "PATH", "PIKE",
    "PNE", "PNES", "PL", "PLN", "PLNS", "PLZ", "PT", "PTS", "PRT", "PRTS", "PR",
    "RADL", "RAMP", "RNCH", "RPD", "RPDS", "RST", "RDG", "RDGS", "RIV", "RD", "RDS",
    "RTE", "ROW", "RUE", "RUN", "SHL", "SHLS", "SHR", "SHRS", "SKWY", "SPG", "SPGS",
    "SPUR", "SQ", "SQS", "STA", "STRA", "STRM", "ST", "STS", "SMT", "TER", "TRWY",
    "TRCE", "TRAK", "TRFY", "TRL", "TRLR", "TUNL", "TPKE", "UPAS", "UN", "UNS", "VLY",
    "VLYS", "VIA", "VW", "VWS", "VLG", "VLGS", "VL", "VIS", "WALK", "WALL", "WAY",
    "WAYS", "WL", "WLS"
]

USPS_ABBREVIATIONS_SET = {abbr.upper().strip() for abbr in USPS_ABBREVIATIONS}

# Combined valid short words (USPS suffixes + exempt words)
VALID_SHORT_WORDS_SET = USPS_ABBREVIATIONS_SET.union(EXEMPT_WORDS_SET)

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------


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
        return 'ALL_LOWERCASE'

    # Check for all uppercase
    if stop_name == stop_name_upper:
        return 'ALL_UPPERCASE'

    # Check for proper title case with exceptions for small words
    exceptions = {"and", "or", "the", "in", "at", "by", "to", "for", "of",
                  "on", "as", "a", "an", "but"
    }
    title_case_words = stop_name.title().split()
    for i in range(1, len(title_case_words)):
        if title_case_words[i].lower() in exceptions:
            title_case_words[i] = title_case_words[i].lower()

    title_case_normalized = " ".join(title_case_words)
    if stop_name == title_case_normalized:
        return 'PROPER_TITLE_CASE'

    # Check for first letter capitalization
    if stop_name and stop_name[0].isupper() and stop_name[1:].islower():
        return 'FIRST_LETTER_CAPITALIZED'

    # Otherwise, it's mixed case
    return 'MIXED_CASE'


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
    stop_id = stop_row['stop_id']
    stop_name = stop_row['stop_name'].strip()

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
        'stop_id': stop_id,
        'stop_name': stop_name,
        'capitalization_scheme': capitalization_scheme,
        'errors': errors
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
    required_columns_stops = ['stop_id', 'stop_name']
    missing_columns = [col for col in required_columns_stops if col not in stops_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in stops.txt: {', '.join(missing_columns)}")

    # Validate each stop
    results = stops_df.apply(validate_stop, axis=1, result_type='expand')

    # Aggregate capitalization schemes
    scheme_counts = results['capitalization_scheme'].value_counts()
    total_stops = len(results)

    # Print the percentages for each scheme
    for scheme in ['ALL_LOWERCASE', 'ALL_UPPERCASE', 'PROPER_TITLE_CASE',
                   'FIRST_LETTER_CAPITALIZED', 'MIXED_CASE']:
        count = scheme_counts.get(scheme, 0)
        percent = (count / total_stops) * 100
        print(f"Percent of stops with {scheme}: {percent:.2f}%")

    # Separate out errors
    all_errors = []
    for idx, row in results.iterrows():
        if row['errors']:
            for err in row['errors']:
                all_errors.append({
                    'stop_id': row['stop_id'],
                    'stop_name': row['stop_name'],
                    'error': err
                })

    # Save errors to CSV if any
    if all_errors:
        errors_df = pd.DataFrame(all_errors)
        errors_df.to_csv(output_file_path, index=False)
        print(f"Errors found. Report saved to {output_file_path}")
    else:
        print("No errors found.")

    # Export all stops with their ids, names, and capitalization scheme
    export_df = results[['stop_id', 'stop_name', 'capitalization_scheme']]
    export_df.to_csv(output_all_stops_file_path, index=False)
    print(f"All stops exported to {output_all_stops_file_path}")


if __name__ == "__main__":
    main()
