"""Reusable utility functions for GTFS data processing scripts.

Includes common GTFS data loader.
"""

import logging
import os

import re
import sys
from collections import defaultdict
from typing import Any, Mapping, Optional
import pandas as pd

# -----------------------------------------------------------------------------
# REUSABLE FUNCTIONS
# -----------------------------------------------------------------------------

def load_gtfs_data(
    gtfs_folder_path: str,
    files: Optional[list[str]] = None,
    dtype: str | type[str] | Mapping[str, Any] = str,
) -> dict[str, pd.DataFrame]:
    """Load one or more GTFS text files into a dictionary of DataFrames.

    Args:
        gtfs_folder_path (str): Absolute or relative path to the directory
            containing GTFS text files.
        files (list[str] | None): Explicit list of GTFS filenames to load.
            If ``None``, the full standard GTFS set is read.
        dtype (str | Mapping[str, Any]): Value forwarded to
            :pyfunc:`pandas.read_csv` to control column dtypes;
            defaults to ``str``.

    Returns:
        dict[str, pandas.DataFrame]: Mapping of file stem → DataFrame.
        For example, ``data["trips"]`` contains *trips.txt*.

    Raises:
        OSError: The folder does not exist or a required file is missing.
        ValueError: A file is empty or malformed.
        RuntimeError: An OS-level error occurs while reading.
    """
    if not os.path.exists(gtfs_folder_path):
        raise OSError(f"The directory '{gtfs_folder_path}' does not exist.")

    if files is None:
        files = [
            "agency.txt",
            "stops.txt",
            "routes.txt",
            "trips.txt",
            "stop_times.txt",
            "calendar.txt",
            "calendar_dates.txt",
            "fare_attributes.txt",
            "fare_rules.txt",
            "feed_info.txt",
            "frequencies.txt",
            "shapes.txt",
            "transfers.txt",
        ]

    missing = [
        file_name
        for file_name in files
        if not os.path.exists(os.path.join(gtfs_folder_path, file_name))
    ]
    if missing:
        raise OSError(f"Missing GTFS files in '{gtfs_folder_path}': {', '.join(missing)}")

    data = {}
    for file_name in files:
        key = file_name.replace(".txt", "")
        file_path = os.path.join(gtfs_folder_path, file_name)
        try:
            df = pd.read_csv(file_path, dtype=dtype, low_memory=False)
            data[key] = df
            logging.info(f"Loaded {file_name} ({len(df)} records).")

        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"File '{file_name}' in '{gtfs_folder_path}' is empty.") from exc

        except pd.errors.ParserError as exc:
            raise ValueError(
                f"Parser error in '{file_name}' in '{gtfs_folder_path}': {exc}"
            ) from exc

        except OSError as exc:
            raise RuntimeError(
                f"OS error reading file '{file_name}' in '{gtfs_folder_path}': {exc}"
            ) from exc
    return data
