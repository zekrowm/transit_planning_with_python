"""Validate short tokens in GTFS stop names against USPS abbreviations.

The script flags suspect 2‑ to 4‑letter tokens and (optionally) appends
approved tokens to an exemption list.  It logs every creation or
modification of the exemption file and every time an error report CSV
is written.
"""

import logging
import re
import sys
from pathlib import Path
from typing import Iterable, List, Set

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

STOPS_FILE = Path(r"Path\To\Your\GTFS_Folder\stops.txt")
EXEMPT_FILE = Path(
    r"Path\To\Your\approved_words.txt"
)  # Set for both existing file or desired file location and name
OUTPUT_CSV = Path(r"Path\To\Your\stop_name_suffix_errors.csv")

INTERACTIVE = True  # Ask about unknown words?
WRITE_EXEMPT = True  # Append approved words back to EXEMPT_FILE?

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("stop_name_validation.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

TOKEN_RE = re.compile(r"[A-Za-z]{1,}")
MIN_LEN = 2
MAX_LEN = 4

USPS_ABBREVIATIONS: tuple[str, ...] = (
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
USPS_SET = set(USPS_ABBREVIATIONS)

# =============================================================================
# FUNCTIONS
# =============================================================================


def tokenize(text: str) -> List[str]:
    """Split *text* into purely alphabetic tokens."""
    return TOKEN_RE.findall(text)


def load_word_list(path: Path | None) -> Set[str]:
    """Return a set of uppercase words contained (one per line) in *path*."""
    if path is None or not path.exists():
        return set()
    return {ln.strip().upper() for ln in path.read_text("utf-8").splitlines() if ln.strip()}


def interactive_classify(tokens: Iterable[str]) -> Set[str]:
    """Ask the user (stdin) to approve or reject each token."""
    approved: set[str] = set()
    for tok in sorted(set(tokens)):
        while True:
            ans = input(f"Treat '{tok}' as VALID? [y/n] ").strip().lower()
            if ans in {"y", "yes"}:
                approved.add(tok)
                break
            if ans in {"n", "no"}:
                break
            print("Please answer y or n.")
    return approved


def append_words(path: Path, words: Iterable[str]) -> None:
    """Append *words* to *path*, logging every create or update event."""
    existing: set[str] = load_word_list(path)
    new: list[str] = sorted(set(words) - existing)
    if not new:
        LOGGER.debug("No new exemptions to write → %s", path)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    action = "CREATED" if not path.exists() else "UPDATED"
    LOGGER.info("%s %s with %d new word(s): %s", action, path, len(new), ", ".join(new))

    with path.open("a", encoding="utf-8") as fh:
        for w in new:
            fh.write(f"{w}\n")


def find_offending_words(name: str, valid: Set[str]) -> List[str]:
    """Return the list of suspect tokens in *name* not contained in *valid*."""
    offenders: list[str] = []
    for tok in tokenize(name):
        u = tok.upper()
        if MIN_LEN <= len(u) <= MAX_LEN and u not in valid:
            offenders.append(u)
    return offenders


def run_validation(
    stops_path: Path,
    exempt_path: Path | None = None,
    *,
    interactive: bool = True,
    write_exempt: bool = True,
    output_csv: Path | None = None,
) -> pd.DataFrame:
    """Validate stop names and return a DataFrame of every offending token."""
    if not stops_path.is_file():
        raise FileNotFoundError(stops_path)

    df = pd.read_csv(stops_path, dtype=str)
    if {"stop_id", "stop_name"}.difference(df.columns):
        raise ValueError("stops.txt must have 'stop_id' and 'stop_name' columns.")

    exempt = load_word_list(exempt_path)
    valid_set = USPS_SET | exempt

    if interactive:
        unknown = {tok for name in df["stop_name"] for tok in find_offending_words(name, valid_set)}
        if unknown:
            print(f"{len(unknown)} unknown short words need review …")
            approved = interactive_classify(unknown)
            exempt |= approved
            if write_exempt and approved and exempt_path is not None:
                append_words(exempt_path, approved)
            valid_set = USPS_SET | exempt  # refresh

    # Build result rows
    rows: list[dict[str, str]] = []
    for _, row in df.iterrows():
        for bad in find_offending_words(str(row["stop_name"]), valid_set):
            rows.append(
                {
                    "stop_id": row["stop_id"],
                    "stop_code": row.get("stop_code", ""),
                    "stop_name": row["stop_name"],
                    "bad_word": bad,
                }
            )
    errs = pd.DataFrame(rows)

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        errs.to_csv(output_csv, index=False)
        print(f"Saved {len(errs)} offending rows → {output_csv}")
        LOGGER.info("Wrote %d offending rows → %s", len(errs), output_csv)

    return errs


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Run the stop‑name validation workflow."""
    errors_df = run_validation(
        STOPS_FILE,
        exempt_path=EXEMPT_FILE,
        interactive=INTERACTIVE,
        write_exempt=WRITE_EXEMPT,
        output_csv=OUTPUT_CSV,
    )

    # Quick peek at the first few violations
    print(errors_df.head())


if __name__ == "__main__":
    main()
