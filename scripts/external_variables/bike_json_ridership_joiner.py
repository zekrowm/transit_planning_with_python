"""Enrich bikeshare station JSON with ridership summaries.

This script joins aggregated ridership data from a summary CSV file into
a station metadata JSON (e.g., from GBFS), and prepares an enriched output
suitable for use in ArcGIS Pro or other GIS tools that accept GeoJSON/JSON.

Functionality:
  - Reads a station JSON file with a `data.stations` array.
  - Loads a summary CSV with `station` and `total_activity` columns.
  - Joins `total_activity` into each station record by `name` or `short_name`.
  - Writes a new JSON with ridership attached to each station record.
  - Outputs a CSV listing any unmatched summary records (e.g., invalid keys).

Intended Use:
  Designed for spatial analysts and data engineers preparing bikeshare
  station-level data for GIS visualization, QA workflows, and planning studies.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ===============================================================================
# CONFIGURATION
# ===============================================================================

# Path to your GBFS station_information.json
STATION_JSON = Path(r"Path\To\Your\Station_Information.json")

# Path to your raw tripdata CSV
TRIP_CSV = Path(r"Path\To\Your\Concatenated_Bikeshare.csv")

# If True, join on 'start_station_id'; if False, on 'start_station_name'
JOIN_BY_ID: bool = True

# Optional list of region_id strings to include. If None or empty, include all regions.
# Example: REGION_IDS = ["42", "104"]
REGION_IDS: Optional[List[str]] = None

# Output files
ENRICHED_CSV = TRIP_CSV.parent / "tripdata_enriched.csv"
UNMATCHED_CSV = TRIP_CSV.parent / "unmatched_trips.csv"

# ===============================================================================
# FUNCTIONS
# ===============================================================================

def load_station_json(
    json_path: Path,
    use_id: bool,
    region_ids: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load GBFS station feed JSON and index by station key.

    Optionally filters to only those stations whose 'region_id' is in `region_ids`.

    Args:
        json_path:   Path to station_information.json.
        use_id:      If True, key on 'station_id'; else on 'name'.
        region_ids:  If provided, only include stations where record['region_id'] is in this list.

    Returns:
        Mapping of station_key → station_metadata dict.
    """
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    stations = data.get("data", {}).get("stations", [])
    key_field = "station_id" if use_id else "name"
    station_map: Dict[str, Dict[str, Any]] = {}

    for record in stations:
        key = record.get(key_field)
        if not key:
            continue
        # apply region filter
        rid = record.get("region_id")
        if region_ids and str(rid) not in region_ids:
            continue

        station_map[str(key).strip()] = record

    return station_map


def enrich_trip_csv(
    trip_csv: Path,
    station_map: Dict[str, Dict[str, Any]],
    use_id: bool,
    enriched_csv: Path,
    unmatched_csv: Path,
) -> None:
    """Join station metadata into each trip row and export results.

    Args:
        trip_csv:      Input tripdata CSV.
        station_map:   station_key → metadata dict.
        use_id:        If True, match on 'start_station_id'; else 'start_station_name'.
        enriched_csv:  Path to write enriched CSV.
        unmatched_csv: Path to write trips with no matching station.
    """
    id_field = "start_station_id" if use_id else "start_station_name"

    with trip_csv.open("r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        base_fields: List[str] = list(reader.fieldnames or [])

        sample_meta = next(iter(station_map.values()), {})
        # drop any fields whose value is a list or dict
        extra_fields = [
            fld for fld, val in sample_meta.items()
            if fld not in ("station_id", "name")
               and not isinstance(val, (list, dict))
        ]

        enriched_fields = base_fields + extra_fields
        with (
            enriched_csv.open("w", newline="", encoding="utf-8") as enf,
            unmatched_csv.open("w", newline="", encoding="utf-8") as umf,
        ):
            enriched_writer = csv.DictWriter(enf, fieldnames=enriched_fields)
            unmatched_writer = csv.DictWriter(umf, fieldnames=base_fields)

            enriched_writer.writeheader()
            unmatched_writer.writeheader()

            for row in reader:
                key = (row.get(id_field) or "").strip()
                meta = station_map.get(key)
                if meta:
                    out_row = row.copy()
                    for fld in extra_fields:
                        out_row[fld] = meta.get(fld, "")
                    enriched_writer.writerow(out_row)
                else:
                    unmatched_writer.writerow(row)

    print(f"✅ Enriched trips saved to: {enriched_csv}")
    print(f"⚠️  Unmatched trips saved to: {unmatched_csv}")


# ===============================================================================
# MAIN
# ===============================================================================

def main() -> None:
    """Main execution flow."""
    try:
        station_map = load_station_json(STATION_JSON, JOIN_BY_ID)
        enrich_trip_csv(TRIP_CSV, station_map, JOIN_BY_ID, ENRICHED_CSV, UNMATCHED_CSV)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
