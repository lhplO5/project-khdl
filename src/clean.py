"""Làm sạch mảng đa chiều CarbonGlobe: xử lý NaN/Inf và giới hạn ngoại lai sau chuẩn hóa."""

from typing import Dict, Tuple

import numpy as np


def sanitize_non_finite(arr: np.ndarray, channel_fill: np.ndarray | None = None) -> Tuple[np.ndarray, int]:
    """Thay thế NaN/Inf theo giá trị trung bình từng kênh (hoặc 0.0) và trả về số phần tử đã sửa."""
    out = arr.astype(np.float32, copy=True)
    bad_mask = ~np.isfinite(out)
    n_bad = int(bad_mask.sum())
    if n_bad == 0:
        return out, 0

    if channel_fill is not None:
        fill_values = np.broadcast_to(channel_fill.astype(np.float32), out.shape)
        out[bad_mask] = fill_values[bad_mask]
    else:
        out[bad_mask] = 0.0
    return out, n_bad


def clip_normalized_outliers(
    arr: np.ndarray,
    z_min: float = -10.0,
    z_max: float = 10.0,
) -> Tuple[np.ndarray, int]:
    """Giới hạn các giá trị ngoại lai cực đoan trong không gian chuẩn hóa Z-score."""
    n_clipped = int(((arr < z_min) | (arr > z_max)).sum())
    if n_clipped == 0:
        return arr, 0
    return np.clip(arr, z_min, z_max).astype(np.float32), n_clipped


def clean_carbonglobe_arrays(
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    clip_z_range: Tuple[float, float] = (-10.0, 10.0),
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Làm sạch cặp mảng (X, Y) đã chuẩn hóa và trả về thống kê số điểm đã xử lý."""
    x_clean, x_bad = sanitize_non_finite(x_norm)
    y_clean, y_bad = sanitize_non_finite(y_norm)

    x_clean, x_clipped = clip_normalized_outliers(x_clean, clip_z_range[0], clip_z_range[1])
    y_clean, y_clipped = clip_normalized_outliers(y_clean, clip_z_range[0], clip_z_range[1])

    stats = {
        "x_non_finite_fixed": x_bad,
        "y_non_finite_fixed": y_bad,
        "x_outliers_clipped": x_clipped,
        "y_outliers_clipped": y_clipped,
    }
    return x_clean, y_clean, stats
