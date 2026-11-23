"""Kiểm toán chất lượng dữ liệu: thống kê missing, cột hằng số, trùng lặp và ngoại lai."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def audit_dataframe(
    df: pd.DataFrame,
    key_cols: Optional[List[str]] = None,
    iqr_factor: float = 3.0,
) -> Dict[str, Any]:
    """Tạo báo cáo kiểm toán chất lượng cho bảng dữ liệu đầu vào."""
    n_rows, n_cols = df.shape
    memory_mb = float(df.memory_usage(deep=True).sum() / 1e6)

    missing_ratio = df.isna().mean()
    cols_with_missing = {
        col: round(float(ratio), 4)
        for col, ratio in missing_ratio[missing_ratio > 0].sort_values(ascending=False).items()
    }

    constant_cols = [
        col for col in df.columns if df[col].nunique(dropna=True) <= 1
    ]

    duplicate_rows = int(df.duplicated().sum())
    duplicate_keys = (
        int(df.duplicated(subset=key_cols).sum())
        if key_cols and all(c in df.columns for c in key_cols)
        else None
    )

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_summary: Dict[str, float] = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr
        outlier_rate = float(((series < lower) | (series > upper)).mean())
        if outlier_rate > 0:
            outlier_summary[col] = round(outlier_rate, 4)

    top_outliers = dict(
        sorted(outlier_summary.items(), key=lambda x: x[1], reverse=True)[:20]
    )

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "memory_mb": round(memory_mb, 3),
        "duplicate_rows": duplicate_rows,
        "duplicate_keys": duplicate_keys,
        "n_constant_cols": len(constant_cols),
        "constant_cols": constant_cols,
        "n_cols_with_missing": len(cols_with_missing),
        "cols_with_missing": cols_with_missing,
        "top_outlier_cols": top_outliers,
    }


def save_audit_report(report: Dict[str, Any], output_path: str | Path) -> Path:
    """Lưu báo cáo kiểm toán ra file JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return out
