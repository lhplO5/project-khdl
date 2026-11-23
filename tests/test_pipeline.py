"""Kiểm thử tự động toàn bộ pipeline Tuần 1 & Tuần 2 cho schema CarbonGlobe."""

import tempfile
from pathlib import Path

import numpy as np

from src.clean import clean_carbonglobe_arrays
from src.dataset import AGES, TREE_BAND, load_ml_ready_npz, load_stats, normalize_array
from src.features import build_one_step_dataset
from src.profiling import audit_array, audit_targets_summary
from src.run_sprint1_2 import create_synthetic_carbonglobe_files


def test_end_to_end_sprint1_and_sprint2() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        raw_dir = Path(tmp)
        glob_dir, stat_dir = create_synthetic_carbonglobe_files(
            raw_dir, n_sites=6, n_ages=15, n_years=40, n_months=12, n_features=136, n_targets=7
        )

        data = load_ml_ready_npz(glob_dir)
        stats = load_stats(stat_dir)
        assert data["x_train"].shape == (6, 40, 12, 136)
        assert data["y_train"].shape == (15, 6, 41, 12, 7)

        x_audit = audit_array(data["x_train"], "x_train")
        assert x_audit["nan_count"] >= 1
        target_summary = audit_targets_summary(data["y_train"], TREE_BAND)
        assert set(target_summary.keys()) == set(TREE_BAND)

        x_norm = normalize_array(data["x_train"], stats["x_mean"], stats["x_std"])
        y_norm = normalize_array(data["y_train"], stats["y_mean"], stats["y_std"])
        x_clean, y_clean, clean_stats = clean_carbonglobe_arrays(x_norm, y_norm, clip_z_range=(-10.0, 10.0))

        assert clean_stats["x_non_finite_fixed"] >= 1
        assert np.isfinite(x_clean).all() and np.isfinite(y_clean).all()
        assert x_clean.min() >= -10.0 and y_clean.max() <= 10.0

        X_feat, Y_target = build_one_step_dataset(x_clean, y_clean, ages=AGES)
        expected_rows = 15 * 6 * 40
        assert X_feat.shape == (expected_rows, 144)
        assert Y_target.shape == (expected_rows, 7)
