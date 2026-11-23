"""Kiểm toán chất lượng dữ liệu mảng đa chiều CarbonGlobe (.npz / .npy)."""

import json
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from src.dataset import TREE_BAND


def audit_array(arr: np.ndarray, name: str) -> Dict[str, Any]:
    """Kiểm tra shape, dtype, RAM, số lượng NaN/Inf và khoảng giá trị của mảng NumPy."""
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    finite_vals = arr[np.isfinite(arr)]

    return {
        "name": name,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "memory_mb": round(float(arr.nbytes / 1e6), 3),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "nan_ratio": round(float(nan_count / arr.size), 6) if arr.size else 0.0,
        "min": round(float(finite_vals.min()), 4) if finite_vals.size else None,
        "max": round(float(finite_vals.max()), 4) if finite_vals.size else None,
        "mean": round(float(finite_vals.mean()), 4) if finite_vals.size else None,
        "std": round(float(finite_vals.std()), 4) if finite_vals.size else None,
    }


def audit_targets_summary(
    y_arr: np.ndarray,
    target_names: Sequence[str] = TREE_BAND,
) -> Dict[str, Dict[str, float]]:
    """Thống kê phân phối vật lý cho từng biến trong 7 biến mục tiêu TREE_BAND."""
    summary: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(target_names):
        vals = y_arr[..., idx]
        finite = vals[np.isfinite(vals)]
        summary[name] = {
            "nan_count": int(np.isnan(vals).sum()),
            "min": round(float(finite.min()), 4) if finite.size else 0.0,
            "p50": round(float(np.median(finite)), 4) if finite.size else 0.0,
            "p95": round(float(np.percentile(finite, 95)), 4) if finite.size else 0.0,
            "max": round(float(finite.max()), 4) if finite.size else 0.0,
            "mean": round(float(finite.mean()), 4) if finite.size else 0.0,
            "std": round(float(finite.std()), 4) if finite.size else 0.0,
        }
    return summary


def save_audit_report(report: Dict[str, Any], output_path: str | Path) -> Path:
    """Lưu báo cáo kiểm toán ra file JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return out
