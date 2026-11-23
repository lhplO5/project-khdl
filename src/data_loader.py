"""Utilities for loading configuration, reading raw data, and creating reproducible samples."""

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml


def load_config(config_path: str | Path = "configs/config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast float64/int64 columns to float32/int32 to reduce RAM usage."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype("int32")
    return df


def load_tabular_file(file_path: str | Path, downcast: bool = True) -> pd.DataFrame:
    """Load a CSV or Parquet file into a DataFrame with optional numeric downcasting."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    return downcast_numeric(df) if downcast else df


def create_dev_sample(
    df: pd.DataFrame,
    fraction: float = 0.1,
    random_seed: int = 42,
    stratify_col: Optional[str] = None,
) -> pd.DataFrame:
    """Create a reproducible development sample from a DataFrame."""
    if not 0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1].")
    if fraction == 1.0:
        return df.copy()

    if stratify_col and stratify_col in df.columns:
        return (
            df.groupby(stratify_col, group_keys=False)
            .apply(lambda x: x.sample(frac=fraction, random_state=random_seed))
            .reset_index(drop=True)
        )
    return df.sample(frac=fraction, random_state=random_seed).reset_index(drop=True)
