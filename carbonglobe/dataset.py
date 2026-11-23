"""Các hàm đọc cấu hình, tối ưu bộ nhớ và trích xuất tập dữ liệu mẫu."""

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml


def read_settings(path: str | Path = "settings.yaml") -> Dict[str, Any]:
    """Đọc file cấu hình YAML của dự án."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def shrink_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Ép float64/int64 về float32/int32 để giảm một nửa dung lượng RAM."""
    float_cols = df.select_dtypes(include=["float64"]).columns
    int_cols = df.select_dtypes(include=["int64"]).columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype("float32")
    if len(int_cols) > 0:
        df[int_cols] = df[int_cols].astype("int32")
    return df


def read_frame(file_path: str | Path, compact: bool = True) -> pd.DataFrame:
    """Nạp bảng dữ liệu từ CSV hoặc Parquet."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {p}")

    ext = p.suffix.lower()
    if ext == ".parquet":
        df = pd.read_parquet(p)
    elif ext == ".csv":
        df = pd.read_csv(p)
    else:
        raise ValueError(f"Định dạng chưa hỗ trợ: {ext}")

    return shrink_dtypes(df) if compact else df


def build_subset(
    df: pd.DataFrame,
    fraction: float = 0.1,
    seed: int = 42,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """Trích xuất tập mẫu tái lập được để phát triển nhanh trên máy cá nhân."""
    if not 0 < fraction <= 1.0:
        raise ValueError("fraction phải nằm trong khoảng (0, 1].")
    if fraction == 1.0:
        return df.copy()

    if group_col and group_col in df.columns:
        return (
            df.groupby(group_col, group_keys=False)
            .apply(lambda g: g.sample(frac=fraction, random_state=seed))
            .reset_index(drop=True)
        )
    return df.sample(frac=fraction, random_state=seed).reset_index(drop=True)
