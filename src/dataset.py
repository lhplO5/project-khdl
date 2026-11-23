"""Nạp dữ liệu NumPy/NPZ CarbonGlobe, chuẩn hóa Z-score và ánh xạ lên lưới toàn cầu."""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import yaml


TREE_BAND = ["height", "agb", "soil", "lai", "gpp", "npp", "rh"]
AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]


def read_pipeline_config(path: str | Path = "configs/pipeline.yaml") -> Dict[str, Any]:
    """Đọc file cấu hình YAML của dự án."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_stats(stats_dir: str | Path, stats_file: str = "data_stats.npz") -> Dict[str, np.ndarray]:
    """Nạp thống kê chuẩn hóa (x_mean, x_std, y_mean, y_std) từ data_stats.npz."""
    path = Path(stats_dir) / stats_file
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file thống kê: {path}")
    data = np.load(path)
    return {
        "x_mean": data["x_mean"].astype(np.float32),
        "x_std": data["x_std"].astype(np.float32),
        "y_mean": data["y_mean"].astype(np.float32),
        "y_std": data["y_std"].astype(np.float32),
    }


def load_mask2d(stats_dir: str | Path, mask_file: str = "mask2d.npy") -> np.ndarray:
    """Nạp ma trận mặt nạ đất liền toàn cầu (360, 720)."""
    path = Path(stats_dir) / mask_file
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file mask2d: {path}")
    return np.load(path)


def load_ml_ready_npz(
    global_dir: str | Path,
    filename: str = "res_train4_test8.npz",
    max_sites: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Nạp tập train/test đóng gói sẵn từ res_train4_test8.npz."""
    path = Path(global_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")

    npz = np.load(path)
    x_train = npz["x_train"].astype(np.float32)  # (S, 40, 12, 136)
    y_train = npz["y_train"].astype(np.float32)  # (15, S, 41, 12, 7)

    if max_sites is not None and max_sites < x_train.shape[0]:
        x_train = x_train[:max_sites]
        y_train = y_train[:, :max_sites]

    out: Dict[str, np.ndarray] = {"x_train": x_train, "y_train": y_train}
    for key in ("x_test", "y_test"):
        if key in npz:
            arr = npz[key].astype(np.float32)
            if max_sites is not None:
                arr = arr[:max_sites] if key == "x_test" else arr[:, :max_sites]
            out[key] = arr
    return out


def normalize_array(arr: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Chuẩn hóa Z-score với ngưỡng an toàn cho độ lệch chuẩn."""
    safe_std = np.where(std < eps, 1.0, std).astype(np.float32)
    return ((arr - mean) / safe_std).astype(np.float32)


def denormalize_array(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Khôi phục giá trị từ không gian chuẩn hóa về đơn vị vật lý."""
    return (arr * std + mean).astype(np.float32)


def to_global_grid(
    values: np.ndarray,
    mask2d: np.ndarray,
    site_indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Ánh xạ vector giá trị theo site lên lưới 2D toàn cầu (360, 720)."""
    grid = np.full(mask2d.shape, np.nan, dtype=np.float32)
    flat_pos = np.flatnonzero(mask2d.astype(bool).ravel())
    pos = flat_pos if site_indices is None else flat_pos[np.asarray(site_indices)]
    grid.ravel()[pos] = values
    return grid
