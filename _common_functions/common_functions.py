"""
common_functions.py

Reusable utility functions for GTFS data processing scripts.

Includes common GTFS data loaders, validators, and formatting helpers.
"""

def load_gtfs_data(files=None, dtype=str):
    """
    Loads GTFS files into pandas DataFrames from a path defined externally
    (GTFS_FOLDER_PATH).

    Parameters:
        files (list[str], optional): GTFS filenames to load. Default is all
            standard GTFS files:
            [
                "agency.txt", "stops.txt", "routes.txt", "trips.txt",
                "stop_times.txt", "calendar.txt", "calendar_dates.txt",
                "fare_attributes.txt", "fare_rules.txt", "feed_info.txt",
                "frequencies.txt", "shapes.txt", "transfers.txt"
            ]
        dtype (str or dict, optional): Pandas dtype to use. Default is str.

    Returns:
        dict[str, pd.DataFrame]: Dictionary keyed by file name without extension.

    Raises:
        FileNotFoundError: If GTFS_FOLDER_PATH doesn't exist or if any required
            file is missing.
        ValueError: If a file is empty or there's a parsing error.
        Exception: For any unexpected error during loading.
    """

    # Check if GTFS_FOLDER_PATH exists
    if not os.path.exists(GTFS_FOLDER_PATH):
        raise FileNotFoundError(f"The directory '{GTFS_FOLDER_PATH}' does not exist.")

    # Default to all standard GTFS files if none were specified
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
            "transfers.txt"
        ]

    # Check for missing files
    missing = [
        file_name for file_name in files
        if not os.path.exists(os.path.join(GTFS_FOLDER_PATH, file_name))
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing GTFS files in '{GTFS_FOLDER_PATH}': {', '.join(missing)}"
        )

    # Load files into DataFrames
    data = {}
    for file_name in files:
        key = file_name.replace(".txt", "")
        file_path = os.path.join(GTFS_FOLDER_PATH, file_name)

        try:
            df = pd.read_csv(file_path, dtype=dtype)
            data[key] = df
            print(f"Loaded {file_name} ({len(df)} records).")

        except pd.errors.EmptyDataError:
            raise ValueError(f"File '{file_name}' is empty.")
        except pd.errors.ParserError as err:
            raise ValueError(f"Parser error in '{file_name}': {err}")
        except Exception as err:
            raise Exception(f"Error loading '{file_name}': {err}")

    return data
