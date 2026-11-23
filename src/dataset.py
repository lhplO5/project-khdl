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


def build_strata_from_target(
    y_train: np.ndarray,
    target_index: int,
    n_bins: int = 5,
) -> np.ndarray:
    """Phân tầng site theo phân vị của trạng thái mục tiêu ban đầu (năm 0, tháng 12).

    Dùng giá trị trung bình qua các mức tuổi để có một đại diện ổn định cho mỗi site,
    sau đó chia thành ``n_bins`` nhóm phân vị đều nhau (mặc định 5 nhóm).
    Trả về mảng nhãn tầng có shape ``(S,)``.
    """
    if n_bins < 1:
        raise ValueError("n_bins phải >= 1.")

    per_age = y_train[:, :, 0, 11, target_index]  # (A, S)
    site_value = np.nanmean(per_age, axis=0)      # (S,)

    # Nếu một site hoàn toàn thiếu dữ liệu, gán giá trị trung vị để không tạo tầng rác.
    if not np.isfinite(site_value).all():
        fill_value = np.nanmedian(site_value)
        site_value = np.where(np.isfinite(site_value), site_value, fill_value)

    quantiles = np.quantile(site_value, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    strata = np.digitize(site_value, np.unique(quantiles))
    return strata.astype(np.int32)


def select_site_indices(
    n_sites: int,
    max_sites: Optional[int] = None,
    strategy: str = "random",
    seed: int = 42,
    strata: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Chọn tập chỉ số site tái lập được cho chế độ lấy mẫu con dữ liệu thật.

    - ``random``: lấy mẫu ngẫu nhiên không hoàn lại, cố định theo ``seed``.
    - ``stratified``: lấy mẫu tỉ lệ theo ``strata`` (ví dụ phân tầng theo ``agb``),
      đảm bảo phủ đủ các dải sinh thái khác nhau; mỗi tầng có ít nhất một site
      khi ngân sách cho phép.

    Luôn trả về đúng ``max_sites`` chỉ số (hoặc toàn bộ nếu ngân sách lớn hơn),
    đã sắp xếp tăng dần để truy vấn/ánh xạ ổn định.
    """
    if max_sites is None or max_sites >= n_sites:
        return np.arange(n_sites, dtype=np.int64)
    if max_sites <= 0:
        return np.empty(0, dtype=np.int64)
    if strategy not in {"random", "stratified"}:
        raise ValueError(f"Chiến lược lấy mẫu không hợp lệ: {strategy!r}")

    rng = np.random.default_rng(seed)

    if strategy == "random":
        chosen = rng.choice(n_sites, size=max_sites, replace=False)
        return np.sort(chosen).astype(np.int64)

    if strata is None or len(strata) != n_sites:
        raise ValueError("strategy='stratified' yêu cầu mảng 'strata' có độ dài bằng n_sites.")

    strata = np.asarray(strata)
    labels, counts = np.unique(strata, return_counts=True)
    if max_sites < len(labels):
        # Không đủ ngân sách để phủ mọi tầng: lấy mẫu ngẫu nhiên toàn cục.
        chosen = rng.choice(n_sites, size=max_sites, replace=False)
        return np.sort(chosen).astype(np.int64)

    # Phân bổ tỉ lệ theo số site của từng tầng, tối thiểu 1 site/tầng.
    ideal = counts.astype(np.float64) / counts.sum() * max_sites
    alloc = np.maximum(np.floor(ideal).astype(np.int64), 1)

    # Bớt các tầng dư (ưu tiên tầng vượt chỉ tiêu nhiều nhất), giữ tối thiểu 1.
    while alloc.sum() > max_sites:
        candidates = np.flatnonzero(alloc > 1)
        idx = candidates[np.argmax(alloc[candidates] - ideal[candidates])]
        alloc[idx] -= 1

    # Thêm vào các tầng thiếu (ưu tiên tầng thiếu nhiều nhất) trong giới hạn pool.
    while alloc.sum() < max_sites:
        candidates = np.flatnonzero(alloc < counts)
        if candidates.size == 0:
            break
        idx = candidates[np.argmax(ideal[candidates] - alloc[candidates])]
        alloc[idx] += 1

    chosen_parts = []
    for label, n_take, n_pool in zip(labels, alloc, counts):
        pool = np.flatnonzero(strata == label)
        n_take = int(min(n_take, len(pool)))
        if n_take == len(pool):
            chosen_parts.append(pool)
        else:
            chosen_parts.append(rng.choice(pool, size=n_take, replace=False))

    chosen = np.sort(np.concatenate(chosen_parts)).astype(np.int64)

    # Bù các site còn thiếu nếu một tầng nhỏ không đủ chỉ tiêu.
    if chosen.size < max_sites:
        remaining = np.setdiff1d(np.arange(n_sites, dtype=np.int64), chosen)
        extra = rng.choice(remaining, size=max_sites - chosen.size, replace=False)
        chosen = np.sort(np.concatenate([chosen, extra])).astype(np.int64)
    return chosen


def load_ml_ready_npz(
    global_dir: str | Path,
    filename: str = "res_train4_test8.npz",
    max_sites: Optional[int] = None,
    strategy: str = "head",
    seed: int = 42,
    stratify_by: Optional[str] = None,
    n_bins: int = 5,
    subset_test: bool = False,
) -> Dict[str, np.ndarray]:
    """Nạp tập train/test đóng gói sẵn từ res_train4_test8.npz.

    Khi ``max_sites`` được đặt, chỉ số site **train** được chọn tái lập theo
    ``strategy`` (``random`` | ``stratified``) và ``seed``. Với
    ``strategy='stratified'``, các site được phân tầng theo phân vị của
    ``stratify_by`` (một trong ``TREE_BAND``) tại trạng thái ban đầu, đảm bảo
    phủ đủ các dải sinh thái. Phân bổ theo tỷ lệ nên không làm lệch phân phối
    toàn cầu của train.

    Mặc định **giữ nguyên toàn bộ test8** (852 site cố định) để giao thức
    rollout 40 năm luôn được đánh giá trên cùng một tập. Chỉ khi
    ``subset_test=True`` mới lấy mẫu test độc lập (seed kế tiếp) cho các
    thử nghiệm nhanh.

    Trả về kèm ``site_indices`` (và ``site_indices_test`` nếu lấy mẫu test)
    để tái lập và ánh xạ lên lưới.
    """
    path = Path(global_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")

    if stratify_by is not None and stratify_by not in TREE_BAND:
        raise ValueError(f"stratify_by phải thuộc TREE_BAND, nhận được {stratify_by!r}")
    target_index = TREE_BAND.index(stratify_by) if stratify_by is not None else None

    npz = np.load(path)
    x_train = npz["x_train"].astype(np.float32)  # (S, 40, 12, 136)
    y_train = npz["y_train"].astype(np.float32)  # (15, S, 41, 12, 7)

    train_strata = None
    if strategy == "stratified":
        if target_index is None:
            raise ValueError("strategy='stratified' yêu cầu tham số 'stratify_by'.")
        train_strata = build_strata_from_target(y_train, target_index, n_bins=n_bins)

    site_indices = select_site_indices(
        n_sites=x_train.shape[0],
        max_sites=max_sites,
        strategy=strategy,
        seed=seed,
        strata=train_strata,
    )
    out: Dict[str, np.ndarray] = {
        "x_train": x_train[site_indices],
        "y_train": y_train[:, site_indices],
        "site_indices": site_indices,
    }

    if "x_test" in npz:
        x_test = npz["x_test"].astype(np.float32)
        y_test = npz["y_test"].astype(np.float32) if "y_test" in npz else None
        if subset_test:
            test_strata = None
            if strategy == "stratified" and y_test is not None and target_index is not None:
                test_strata = build_strata_from_target(y_test, target_index, n_bins=n_bins)
            test_indices = select_site_indices(
                n_sites=x_test.shape[0],
                max_sites=max_sites,
                strategy=strategy,
                seed=seed + 1,  # tách luồng RNG tránh trùng mẫu với train
                strata=test_strata,
            )
            x_test = x_test[test_indices]
            out["site_indices_test"] = test_indices
            if y_test is not None:
                y_test = y_test[:, test_indices]
        out["x_test"] = x_test
        if y_test is not None:
            out["y_test"] = y_test
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
