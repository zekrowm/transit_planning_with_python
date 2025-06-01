"""
gtfs_helpers.py

Reusable utility functions for GTFS data processing scripts.

Includes common GTFS data loaders, validators, and formatting helpers.
"""

import importlib
import logging
import os
from pathlib import Path
from typing import Dict

import pandas as pd

# -----------------------------------------------------------------------------
# REUSABLE FUNCTIONS
# -----------------------------------------------------------------------------


def load_gtfs_data(gtfs_folder_path: str, files: list[str] = None, dtype=str):
    """
    Loads GTFS files into pandas DataFrames from the specified directory.
    This function uses the logging module for output.

    Parameters:
        gtfs_folder_path (str): Path to the directory containing GTFS files.
        files (list[str], optional): GTFS filenames to load. Default is all
            standard GTFS files:
            [
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
                "transfers.txt"
            ]
        dtype (str or dict, optional): Pandas dtype to use. Default is str.

    Returns:
        dict[str, pd.DataFrame]: Dictionary keyed by file name without extension.

    Raises:
        OSError: If gtfs_folder_path doesn't exist or if any required file is missing.
        ValueError: If a file is empty or there's a parsing error.
        RuntimeError: For OS errors during file reading.
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


# -----------------------------------------------------------------------------
# REUSABLE FUNCTIONS
# -----------------------------------------------------------------------------


def apply_route_filters(
    gtfs_data: Dict[str, pd.DataFrame],
    settings_module: str = "__main__",
    copy: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Filters a GTFS dataset by the route-ID lists defined in *settings_module*.

    Parameters:
        gtfs_data (dict[str, pd.DataFrame]):
            A dictionary of GTFS tables (e.g., as produced by ``load_gtfs_data``).
        settings_module (str, optional):
            Import path of a Python module containing two constants:
            ``FILTER_IN_ROUTES`` and ``FILTER_OUT_ROUTES`` (each ``list[str]``).
            Defaults to the caller’s ``__main__``.
        copy (bool, optional):
            If ``True`` (default) each DataFrame is deep-copied before the
            filters are applied; if ``False`` the original objects are modified
            in place.

    Returns:
        dict[str, pd.DataFrame]:
            A new dictionary containing only the rows that satisfy the route
            criteria.

    Raises:
        AttributeError:
            If *settings_module* does not define both required constants.
        ValueError:
            If conflicting route IDs appear in both lists or the filter would
            produce an impossible result.

    Example:
        >>> # in your main script …
        >>> FILTER_IN_ROUTES = ["10A", "10B"]
        >>> feed = load_gtfs_data("feeds/metrobus")          # doctest: +SKIP
        >>> filtered = apply_route_filters(feed)             # doctest: +SKIP
    """
    # --------------------------------------------------------------------- #
    # Retrieve the caller’s route lists
    # --------------------------------------------------------------------- #
    cfg = importlib.import_module(settings_module)
    try:
        in_routes = getattr(cfg, "FILTER_IN_ROUTES")
        out_routes = getattr(cfg, "FILTER_OUT_ROUTES")
    except AttributeError as exc:
        raise AttributeError(
            f"Module '{settings_module}' must define both "
            f"FILTER_IN_ROUTES and FILTER_OUT_ROUTES."
        ) from exc

    # --------------------------------------------------------------------- #
    # Log what we’re about to do (root logger, matching load_gtfs_data style)
    # --------------------------------------------------------------------- #
    logging.info(
        "Applying route filters – keep: %s | drop: %s",
        in_routes or "<all>",
        out_routes or "<none>",
    )

    # --------------------------------------------------------------------- #
    # Delegate the heavy lifting to the lower-level helper
    # --------------------------------------------------------------------- #
    return filter_gtfs_by_routes(
        gtfs_data,
        filter_in_routes=in_routes,
        filter_out_routes=out_routes,
        copy=copy,
        logger=logging.getLogger(),  # pass the root logger for consistency
    )
