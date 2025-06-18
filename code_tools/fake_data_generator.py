"""Generate a realistic-looking *fake* copy of a single relational table.

1. Load a small “real” sample from `INPUT_FILE`.
2. Infer a lightweight schema (dtype, ranges, simple semantics).
3. Produce `N_ROWS` synthetic rows with Faker + NumPy.
4. Write the result as Parquet in `OUTPUT_DIR` (same stem as input).
"""

from __future__ import annotations
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final, List, Tuple
import re
import numpy as np
import pandas as pd
from faker import Faker
from faker.providers import BaseProvider

# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

INPUT_FILE: str = r"File\Path\To\Your\stops_sample.parquet"
OUTPUT_DIR: str = r"Folder\Path\To\Your\fake_output"

N_ROWS: int = 100  # synthetic rows to create
GLOBAL_SEED: int | None = 42  # None = fresh randomness every run

#: (min_lat, max_lat, min_lon, max_lon) – default = DC metro
BBOX: Tuple[float, float, float, float] = (38.60, 39.30, -77.50, -76.80)

_LAT_RGX = re.compile(r"\b(lat|latitude|y)\b", re.I)
_LON_RGX = re.compile(r"\b(lon|lng|long|longitude|x)\b", re.I)

# --------------------------------------------------------------------------------------------------
# SYNTHETIC DATA ENGINE (from your original code)
# --------------------------------------------------------------------------------------------------

faker: Final[Faker] = Faker()  # global instance (seeded in main)


class _StopNameProvider(BaseProvider):
    """Custom provider: 'Word Street & Nth' style stop names."""

    def stop_name(self) -> str:  # noqa: D401
        word = self.generator.word().capitalize()
        ordinal = f"{np.random.randint(1, 50)}th"
        street_type = self.generator.street_suffix()
        return f"{word} {street_type} & {ordinal}"


faker.add_provider(_StopNameProvider)

# ==================================================================================================
# FUNCTIONS
# ==================================================================================================

def _max_decimal_places(
    s: pd.Series,
    default: int = 3,
    cap: int = 6,
) -> int:
    """Return the largest number of decimal places found in *s*.

    - When nothing has a decimal, returns 0.  
    - Never returns more than *cap*.  
    - Guarantees at least *default* places whenever a non-integer value is present.
    """
    numeric = pd.to_numeric(s, errors="coerce").dropna()
    if numeric.empty:
        return 0

    max_dp = 0
    for val in numeric:
        txt = f"{val}"
        if "." in txt:
            dp = len(txt.split(".")[1].rstrip("0"))
            max_dp = max(max_dp, dp)
            if max_dp >= cap:
                break

    if max_dp == 0:
        return 0
    return max(max_dp, default)

def _is_categorical(s: pd.Series) -> bool:
    uniq = s.nunique(dropna=True)
    return 2 <= uniq <= 20


def _num_range(s: pd.Series) -> Tuple[float, float]:
    numeric = pd.to_numeric(s, errors="coerce")
    return float(numeric.min()), float(numeric.max())


class _SchemaField:
    """Holds generation logic for one column."""

    def __init__(self, name: str, generator: Callable[[int], List[Any]]) -> None:
        self.name = name
        self._gen = generator

    def generate(self, n: int) -> List[Any]:  # noqa: D401
        return self._gen(n)


# ── UPDATED _Schema CLASS (only the from_sample method is changed) ────────────
class _Schema:
    """Lightweight schema built from a sample DataFrame."""

    def __init__(self, fields: dict[str, _SchemaField]) -> None:
        self._fields = fields

    # ---------------------------------------------------------------------
    @classmethod
    def from_sample(cls, df: pd.DataFrame) -> "_Schema":
        """Infer simple generation rules from a small real sample."""
        min_lat, max_lat, min_lon, max_lon = BBOX
        fields: dict[str, _SchemaField] = {}

        for col in df.columns:
            s = df[col]
            col_lc = col.lower()

            # ── Geospatial lat/long special cases ───────────────────────
            if _LAT_RGX.search(col_lc):

                def _gen_lat(
                    n: int, lo: float = min_lat, hi: float = max_lat
                ) -> list[float]:
                    return np.random.uniform(lo, hi, size=n).round(6).tolist()

                fields[col] = _SchemaField(col, _gen_lat)
                continue

            if _LON_RGX.search(col_lc):

                def _gen_lon(
                    n: int, lo: float = min_lon, hi: float = max_lon
                ) -> list[float]:
                    return np.random.uniform(lo, hi, size=n).round(6).tolist()

                fields[col] = _SchemaField(col, _gen_lon)
                continue

            # ── Numeric columns ─────────────────────────────────────────
            if pd.api.types.is_integer_dtype(s):

                low, high = map(int, _num_range(s))

                def _gen_int(n: int, lo: int = low, hi: int = high) -> list[int]:
                    return np.random.randint(lo, hi + 1, size=n).tolist()

                fields[col] = _SchemaField(col, _gen_int)
                continue

            if pd.api.types.is_float_dtype(s):
                low, high = _num_range(s)
                decimals = _max_decimal_places(s)

                def _gen_float(
                    n: int,
                    lo: float = low,
                    hi: float = high,
                    dp: int = decimals,
                ) -> list[float]:
                    return np.random.uniform(lo, hi, size=n).round(dp).tolist()

                fields[col] = _SchemaField(col, _gen_float)
                continue

            # ── Low-cardinality categoricals ───────────────────────────
            if _is_categorical(s):
                pool = s.dropna().unique().tolist()

                def _gen_cat(n: int, items: list[Any] = pool) -> list[Any]:
                    return np.random.choice(items, size=n, replace=True).tolist()

                fields[col] = _SchemaField(col, _gen_cat)
                continue

            # ── String heuristics (unchanged) ──────────────────────────
            if "name" in col_lc:
                gen = lambda n: [faker.name() for _ in range(n)]
            elif "address" in col_lc or "stop" in col_lc:
                gen = lambda n: [faker.stop_name() for _ in range(n)]
            elif "date" in col_lc:
                gen = lambda n: [faker.date() for _ in range(n)]
            else:
                gen = lambda n: [faker.word() for _ in range(n)]

            fields[col] = _SchemaField(col, gen)

        return cls(fields)

    # ─────────────────────────────────────────────────────────────
    def fake(self, n_rows: int) -> pd.DataFrame:  # noqa: D401
        data = {name: f.generate(n_rows) for name, f in self._fields.items()}
        return pd.DataFrame(data)


def mock_dataframe(
    sample_df: pd.DataFrame, n_rows: int, seed: int | None = None
) -> pd.DataFrame:
    """Return a synthetic DataFrame matching `sample_df`’s structure."""
    if seed is not None:
        Faker.seed(seed)
        np.random.seed(seed)
    schema = _Schema.from_sample(sample_df)
    return schema.fake(n_rows)


def _load_table(path: Path) -> pd.DataFrame:
    """Load a small sample, inferring loader from suffix."""
    ext = path.suffix.lower()
    if ext in {".parquet", ".pq"}:
        # You'll still need pyarrow installed to read the *sample* parquet file if INPUT_FILE is a .parquet file.
        # If your sample is also CSV/XLSX, you can remove this line and just rely on the other loaders.
        try:
            return pd.read_parquet(path, engine="pyarrow")
        except ImportError:
            raise ImportError("Please install 'pyarrow' (pip install pyarrow) to read parquet input files.")
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in {".feather", ".ft"}:
        return pd.read_feather(path)
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported sample file type: {path}")


# ==================================================================================================
# MAIN
# ==================================================================================================


def main() -> None:  # noqa: D401
    """Function serves as entry point and orchestrator for the script."""
    src = Path(INPUT_FILE)
    if not src.exists():
        raise FileNotFoundError(f"Sample file not found: {src}")

    output_root = Path(OUTPUT_DIR)
    output_root.mkdir(parents=True, exist_ok=True)

    sample = _load_table(src)
    fake_df = mock_dataframe(sample, n_rows=N_ROWS, seed=GLOBAL_SEED)

    # --- MODIFICATION STARTS HERE ---
    # Change the output file extension and save method
    output_format = "csv"  # or "xlsx"
    dest = output_root / f"{src.stem}.{output_format}"

    if output_format == "csv":
        fake_df.to_csv(dest, index=False)
    elif output_format == "xlsx":
        # You'll need openpyxl installed for this: pip install openpyxl
        try:
            fake_df.to_excel(dest, index=False)
        except ImportError:
            raise ImportError("Please install 'openpyxl' (pip install openpyxl) to write XLSX files.")
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    # --- MODIFICATION ENDS HERE ---

    print(f"✓ {src.stem}: {N_ROWS:,} fake rows → {dest}")
    print("All done.")


if __name__ == "__main__":
    main()
