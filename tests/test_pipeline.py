"""Kiểm thử tự động toàn bộ pipeline Tuần 1 & Tuần 2 cho schema CarbonGlobe."""

import tempfile
from pathlib import Path

import numpy as np

from src.clean import clean_carbonglobe_arrays
from src.dataset import (
    AGES,
    TREE_BAND,
    build_strata_from_target,
    load_ml_ready_npz,
    load_stats,
    normalize_array,
    select_site_indices,
)
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


def test_select_site_indices_random_is_reproducible() -> None:
    a = select_site_indices(1000, max_sites=100, strategy="random", seed=42)
    b = select_site_indices(1000, max_sites=100, strategy="random", seed=42)
    c = select_site_indices(1000, max_sites=100, strategy="random", seed=7)

    assert len(a) == 100
    assert len(np.unique(a)) == 100
    assert np.array_equal(a, b)          # cùng seed -> giống hệt
    assert not np.array_equal(a, c)      # khác seed -> khác mẫu
    assert np.all(np.diff(a) > 0)        # đã sắp xếp tăng dần


def test_select_site_indices_stratified_covers_every_stratum() -> None:
    rng = np.random.default_rng(0)
    strata = rng.integers(0, 5, size=1000)

    chosen = select_site_indices(1000, max_sites=200, strategy="stratified", seed=42, strata=strata)

    assert len(chosen) == 200
    assert len(np.unique(chosen)) == 200
    covered = np.unique(strata[chosen])
    assert np.array_equal(covered, np.unique(strata))  # phủ đủ 5 tầng
    # tái lập được
    again = select_site_indices(1000, max_sites=200, strategy="stratified", seed=42, strata=strata)
    assert np.array_equal(chosen, again)


def test_load_ml_ready_npz_keeps_full_test8_by_default() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        raw_dir = Path(tmp)
        rng = np.random.default_rng(1)
        n_train, n_test = 80, 30
        x_train = rng.normal(size=(n_train, 40, 12, 136)).astype(np.float32)
        y_train = rng.normal(size=(15, n_train, 41, 12, 7)).astype(np.float32)
        x_test = rng.normal(size=(n_test, 40, 12, 136)).astype(np.float32)
        y_test = rng.normal(size=(15, n_test, 41, 12, 7)).astype(np.float32)
        np.savez(raw_dir / "res_train4_test8.npz", x_train=x_train, y_train=y_train,
                 x_test=x_test, y_test=y_test)

        data = load_ml_ready_npz(
            raw_dir, max_sites=20, strategy="stratified", seed=42, stratify_by="agb", n_bins=4
        )

        assert data["x_train"].shape == (20, 40, 12, 136)
        assert data["y_train"].shape == (15, 20, 41, 12, 7)
        # test8 giữ nguyên để mọi cấu hình được đánh giá trên cùng một tập.
        assert data["x_test"].shape == (n_test, 40, 12, 136)
        assert data["y_test"].shape == (15, n_test, 41, 12, 7)
        assert "site_indices_test" not in data
        assert len(np.unique(data["site_indices"])) == 20
        assert data["site_indices"].max() < n_train


def test_load_ml_ready_npz_can_subset_test_for_quick_debug() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        raw_dir = Path(tmp)
        rng = np.random.default_rng(1)
        n_train, n_test = 80, 30
        x_train = rng.normal(size=(n_train, 40, 12, 136)).astype(np.float32)
        y_train = rng.normal(size=(15, n_train, 41, 12, 7)).astype(np.float32)
        x_test = rng.normal(size=(n_test, 40, 12, 136)).astype(np.float32)
        y_test = rng.normal(size=(15, n_test, 41, 12, 7)).astype(np.float32)
        np.savez(raw_dir / "res_train4_test8.npz", x_train=x_train, y_train=y_train,
                 x_test=x_test, y_test=y_test)

        data = load_ml_ready_npz(
            raw_dir,
            max_sites=20,
            strategy="stratified",
            seed=42,
            stratify_by="agb",
            n_bins=4,
            subset_test=True,
        )

        assert data["x_test"].shape == (20, 40, 12, 136)
        assert data["y_test"].shape == (15, 20, 41, 12, 7)
        assert len(np.unique(data["site_indices_test"])) == 20
        # train/test chọn độc lập trên không gian chỉ số riêng của từng tập
        assert data["site_indices_test"].max() < n_test
