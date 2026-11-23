"""Tạo đặc trưng one-step-ahead chuẩn hóa từ chuỗi 40 năm x 12 tháng của CarbonGlobe."""

from typing import Sequence, Tuple

import numpy as np

from src.dataset import AGES


def build_one_step_dataset(
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    ages: Sequence[int] = AGES,
    december_month_idx: int = 11,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Xây dựng cặp (X_feat, Y_target) cho bài toán dự báo từng bước (one-step-ahead):
      - x_norm: (S, 40, 12, 136) đầu vào đã chuẩn hóa
      - y_norm: (A, S, 41, 12, 7) đầu ra tháng đã chuẩn hóa (năm 0 là trạng thái khởi tạo)
    Trả về:
      - X_feat: (A * S * 40, 144) gồm [136 trung bình năm | 7 mục tiêu tháng 12 năm t-1 | age/500]
      - Y_target: (A * S * 40, 7) gồm 7 mục tiêu tháng 12 năm t
    """
    n_ages, n_sites, n_years_plus_one, _, _ = y_norm.shape
    n_years = n_years_plus_one - 1
    if x_norm.shape[0] != n_sites or x_norm.shape[1] != n_years:
        raise ValueError(
            f"Kích thước không khớp: x_norm={x_norm.shape}, y_norm={y_norm.shape}"
        )
    if len(ages) != n_ages:
        raise ValueError(f"Số mức tuổi ({len(ages)}) không khớp chiều đầu của y_norm ({n_ages}).")

    # Trung bình 12 tháng thành đặc trưng năm: (S, 40, 136)
    x_annual = x_norm.mean(axis=2).astype(np.float32)

    # Lấy giá trị tháng 12 của mỗi năm: (A, S, 41, 7)
    y_dec = y_norm[:, :, :, december_month_idx, :].astype(np.float32)

    feat_blocks = []
    targ_blocks = []
    for age_idx, age in enumerate(ages):
        age_col = np.full((n_sites, n_years, 1), age / 500.0, dtype=np.float32)
        prev_y = y_dec[age_idx, :, :-1, :]  # (S, 40, 7)
        next_y = y_dec[age_idx, :, 1:, :]   # (S, 40, 7)

        combined = np.concatenate([x_annual, prev_y, age_col], axis=-1)  # (S, 40, 144)
        feat_blocks.append(combined.reshape(-1, combined.shape[-1]))
        targ_blocks.append(next_y.reshape(-1, next_y.shape[-1]))

    X_feat = np.concatenate(feat_blocks, axis=0).astype(np.float32)
    Y_target = np.concatenate(targ_blocks, axis=0).astype(np.float32)
    return X_feat, Y_target
