"""Pipeline tiền xử lý và làm sạch dữ liệu CarbonGlobe."""

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.dataset import shrink_dtypes


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa tên cột về dạng snake_case chữ thường."""
    renamed = {}
    for col in df.columns:
        clean = re.sub(r"[^0-9a-zA-Z]+", "_", str(col).strip()).strip("_").lower()
        renamed[col] = clean
    return df.rename(columns=renamed)


def drop_unusable_columns(
    df: pd.DataFrame,
    max_missing_ratio: float = 0.5,
    protected_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Loại bỏ cột hằng số và cột có tỷ lệ giá trị khuyết vượt ngưỡng."""
    protected = set(protected_cols or [])
    to_drop: List[str] = []

    for col in df.columns:
        if col in protected:
            continue
        if df[col].isna().mean() > max_missing_ratio or df[col].nunique(dropna=True) <= 1:
            to_drop.append(col)

    return df.drop(columns=to_drop), to_drop


def impute_missing_values(
    df: pd.DataFrame,
    group_col: Optional[str] = None,
    time_col: Optional[str] = None,
) -> pd.DataFrame:
    """Điền khuyết dữ liệu: nội suy theo chuỗi thời gian nếu có khóa nhóm, sau đó dùng median."""
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()

    if group_col and time_col and group_col in out.columns and time_col in out.columns:
        out = out.sort_values([group_col, time_col]).reset_index(drop=True)
        out[numeric_cols] = (
            out.groupby(group_col)[numeric_cols]
            .transform(lambda s: s.interpolate(method="linear", limit_direction="both"))
        )

    for col in numeric_cols:
        if out[col].isna().any():
            median_val = out[col].median()
            fill_val = 0.0 if pd.isna(median_val) else median_val
            out[col] = out[col].fillna(fill_val)

    return out


def clip_outliers(
    df: pd.DataFrame,
    lower_q: float = 0.005,
    upper_q: float = 0.995,
    exclude_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Giới hạn giá trị ngoại lai theo phân vị (winsorizing) cho các cột số."""
    out = df.copy()
    excluded = set(exclude_cols or [])
    numeric_cols = [
        c for c in out.select_dtypes(include=[np.number]).columns if c not in excluded
    ]

    for col in numeric_cols:
        lo = out[col].quantile(lower_q)
        hi = out[col].quantile(upper_q)
        if pd.notna(lo) and pd.notna(hi) and lo < hi:
            out[col] = out[col].clip(lower=lo, upper=hi)

    return out


def clean_dataset(
    df: pd.DataFrame,
    max_missing_ratio: float = 0.5,
    group_col: Optional[str] = None,
    time_col: Optional[str] = None,
    protected_cols: Optional[List[str]] = None,
    clip_quantiles: Tuple[float, float] = (0.005, 0.995),
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Chạy toàn bộ chuỗi làm sạch dữ liệu và trả về bảng sạch kèm metadata."""
    initial_shape = df.shape
    cleaned = normalize_column_names(df)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    cleaned, dropped_cols = drop_unusable_columns(
        cleaned,
        max_missing_ratio=max_missing_ratio,
        protected_cols=protected_cols,
    )
    cleaned = impute_missing_values(cleaned, group_col=group_col, time_col=time_col)
    cleaned = clip_outliers(
        cleaned,
        lower_q=clip_quantiles[0],
        upper_q=clip_quantiles[1],
        exclude_cols=protected_cols,
    )
    cleaned = shrink_dtypes(cleaned)

    summary: Dict[str, object] = {
        "initial_shape": initial_shape,
        "final_shape": cleaned.shape,
        "dropped_cols": dropped_cols,
        "remaining_nulls": int(cleaned.isna().sum().sum()),
    }
    return cleaned, summary
