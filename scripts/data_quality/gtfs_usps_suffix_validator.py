"""Validate short tokens in GTFS stop names against USPS abbreviations.

The script flags suspect 2‑ to 4‑letter tokens and (optionally) appends
approved tokens to an exemption list.  It logs every creation or
modification of the exemption file and every time an error report CSV
is written.
"""

from __future__ import annotations

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

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

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
    """Return a set of uppercase words contained (one per line) in *path*.

    If *path* is None, does not exist, or is not a regular file, log a
    warning and return an empty set instead of crashing.
    """
    if path is None:
        logging.warning("No approved-words file provided; proceeding with USPS list only.")
        return set()

    if not path.exists():
        logging.warning(
            "Approved-words file does not exist: %s — proceeding with USPS list only.", path
        )
        return set()

    if not path.is_file():
        logging.warning(
            "Approved-words path is not a regular file: %s — proceeding with USPS list only.", path
        )
        return set()

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logging.warning(
            "Could not read approved-words file %s (%s) — proceeding with USPS list only.",
            path,
            exc,
        )
        return set()

    words = {ln.strip().upper() for ln in text.splitlines() if ln.strip()}
    logging.info("Loaded %d approved word(s) from %s", len(words), path)
    return words


def interactive_classify(tokens: Iterable[str]) -> Set[str]:
    ""r"Ask the user (stdin) to approve or reject each token.

    Prompts accept:
        y \ yes  → token is valid (added to approved set)
        n \ no   → token is not valid (skipped)
        q \ quit → stop prompting; keep everything approved so far,
                   treat all remaining tokens as 'n'.
    """
    approved: set[str] = set()
    token_list = sorted(set(tokens))
    total = len(token_list)

    for idx, tok in enumerate(token_list):
        while True:
            ans = input(f"[{idx + 1}/{total}] Treat '{tok}' as VALID? [y/n/q] ").strip().lower()
            if ans in {"y", "yes"}:
                approved.add(tok)
                break
            if ans in {"n", "no"}:
                break
            if ans in {"q", "quit"}:
                remaining = total - idx
                logging.warning(
                    "User exited interactive review early: %d of %d token(s) were not considered "
                    "and will be treated as invalid. %d token(s) approved before exit.",
                    remaining,
                    total,
                    len(approved),
                )
                return approved
            logging.info("Please answer y, n, or q.")

    return approved


def append_words(path: Path, words: Iterable[str]) -> None:
    """Append *words* to *path*, logging every create or update event.

    If *path* points at a directory, is otherwise unwritable, or cannot be
    created, log a warning and return without raising — so interactive
    approvals from this session are not lost to an exception.
    """
    words = set(words)
    if not words:
        return

    # Guard against path being a directory (e.g. Path('.') from an empty config value).
    if path.exists() and not path.is_file():
        logging.warning(
            "Cannot append approved words: %s is not a regular file. "
            "%d approval(s) from this session were NOT persisted: %s",
            path,
            len(words),
            ", ".join(sorted(words)),
        )
        return

    existing: set[str] = load_word_list(path) if path.is_file() else set()
    new: list[str] = sorted(words - existing)
    if not new:
        logging.debug("No new exemptions to write → %s", path)
        return

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        action = "CREATED" if not path.is_file() else "UPDATED"
        with path.open("a", encoding="utf-8") as fh:
            for w in new:
                fh.write(f"{w}\n")
    except (OSError, PermissionError) as exc:
        logging.warning(
            "Failed to write approved words to %s (%s). "
            "%d approval(s) from this session were NOT persisted: %s",
            path,
            exc,
            len(new),
            ", ".join(new),
        )
        return

    logging.info("%s %s with %d new word(s): %s", action, path, len(new), ", ".join(new))


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
            logging.info("%d unknown short words need review …", len(unknown))
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
        logging.info("Saved %d offending rows → %s", len(errs), output_csv)
        logging.info("Wrote %d offending rows → %s", len(errs), output_csv)

    return errs


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Run the stop‑name validation workflow."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler("stop_name_validation.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    errors_df = run_validation(
        STOPS_FILE,
        exempt_path=EXEMPT_FILE,
        interactive=INTERACTIVE,
        write_exempt=WRITE_EXEMPT,
        output_csv=OUTPUT_CSV,
    )

    # Quick peek at the first few violations
    logging.info(errors_df.head())


if __name__ == "__main__":
    main()
