"""Reusable utility functions for GTFS data processing scripts.

Includes common GTFS data loaders, validators, and formatting helpers.
"""

import logging
import os

import pandas as pd

# -----------------------------------------------------------------------------
# REUSABLE FUNCTIONS
# -----------------------------------------------------------------------------


def load_gtfs_data(gtfs_folder_path: str, files: list[str] = None, dtype=str):
    """Load a GTFS feed into pandas DataFrames.

    Uses the ``logging`` module to report progress.

    Args:
        gtfs_folder_path: Absolute or relative path to the directory that
            contains the GTFS text files.
        files: Specific GTFS filenames to load (e.g. ``["stops.txt",
            "trips.txt"]``). If *None*, every standard GTFS file listed in the
            specification is attempted.
        dtype: A pandas *dtype* or *dict* of column→dtype mappings passed
            straight to :pyfunc:`pandas.read_csv`. Defaults to ``str`` for
            fully-string-typed frames.

    Returns:
        A dictionary whose keys are the filenames **without** the ``.txt``
        extension and whose values are the corresponding DataFrames.

    Raises:
        OSError: The feed folder does not exist or at least one required file
            is missing.
        ValueError: A file is empty or cannot be parsed by
            :pyfunc:`pandas.read_csv`.
        RuntimeError: Any other I/O error raised while reading a file.
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
        raise OSError(
            f"Missing GTFS files in '{gtfs_folder_path}': {', '.join(missing)}"
        )

    data = {}
    for file_name in files:
        key = file_name.replace(".txt", "")
        file_path = os.path.join(gtfs_folder_path, file_name)
        try:
            df = pd.read_csv(file_path, dtype=dtype, low_memory=False)
            data[key] = df
            logging.info(f"Loaded {file_name} ({len(df)} records).")

        except pd.errors.EmptyDataError as exc:
            raise ValueError(
                f"File '{file_name}' in '{gtfs_folder_path}' is empty."
            ) from exc

        except pd.errors.ParserError as exc:
            raise ValueError(
                f"Parser error in '{file_name}' in '{gtfs_folder_path}': {exc}"
            ) from exc

        except OSError as exc:
            raise RuntimeError(
                f"OS error reading file '{file_name}' in '{gtfs_folder_path}': {exc}"
            ) from exc

    return data
