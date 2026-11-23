"""Điều phối chạy toàn bộ pipeline Tuần 1 & Tuần 2 cho bộ dữ liệu CarbonGlobe (.npz)."""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

from src.clean import clean_carbonglobe_arrays
from src.dataset import (
    AGES,
    TREE_BAND,
    load_mask2d,
    load_ml_ready_npz,
    load_stats,
    normalize_array,
    read_pipeline_config,
)
from src.features import build_one_step_dataset
from src.profiling import audit_array, audit_targets_summary, save_audit_report


def create_synthetic_carbonglobe_files(
    raw_dir: Path,
    n_sites: int = 30,
    n_ages: int = 15,
    n_years: int = 40,
    n_months: int = 12,
    n_features: int = 136,
    n_targets: int = 7,
    seed: int = 42,
) -> Tuple[Path, Path]:
    """Sinh bộ file .npz/.npy giả lập theo đúng schema CarbonGlobe để chạy thử end-to-end."""
    rng = np.random.default_rng(seed)
    glob_dir = raw_dir / "data_global"
    stat_dir = raw_dir / "data_stats"
    glob_dir.mkdir(parents=True, exist_ok=True)
    stat_dir.mkdir(parents=True, exist_ok=True)

    x_train = rng.normal(15.0, 5.0, size=(n_sites, n_years, n_months, n_features)).astype(np.float32)
    y_train = rng.normal(50.0, 15.0, size=(n_ages, n_sites, n_years + 1, n_months, n_targets)).astype(np.float32)

    # Cài sẵn một ít NaN và outlier để kiểm chứng bước làm sạch
    x_train[0, 0, 0, 0] = np.nan
    y_train[0, 0, 1, 11, 1] = 5000.0

    np.savez_compressed(glob_dir / "res_train4_test8.npz", x_train=x_train, y_train=y_train)
    np.savez(
        stat_dir / "data_stats.npz",
        x_mean=x_train.mean(axis=(0, 1, 2)),
        x_std=np.maximum(x_train.std(axis=(0, 1, 2)), 1e-3),
        y_mean=y_train.mean(axis=(0, 1, 2, 3)),
        y_std=np.maximum(y_train.std(axis=(0, 1, 2, 3)), 1e-3),
    )

    mask2d = np.zeros((360, 720), dtype=np.uint8)
    mask2d[100:110, 200 : 200 + (n_sites // 10) + 1] = 1
    np.save(stat_dir / "mask2d.npy", mask2d)
    return glob_dir, stat_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy pipeline CarbonGlobe Tuần 1 & Tuần 2.")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="File cấu hình YAML.")
    parser.add_argument("--synthetic", action="store_true", help="Sinh và chạy trên dữ liệu mô phỏng.")
    args = parser.parse_args()

    cfg = read_pipeline_config(args.config)
    raw_dir = Path(cfg["paths"]["raw_dir"])
    interim_dir = Path(cfg["paths"]["interim_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        glob_dir, stat_dir = create_synthetic_carbonglobe_files(raw_dir, seed=cfg["seed"])
        tag = "synthetic"
    else:
        glob_dir = raw_dir / cfg["paths"]["global_subdir"]
        stat_dir = raw_dir / cfg["paths"]["stats_subdir"]
        tag = "real"

    max_sites = cfg["subset"]["max_sites"] if cfg["subset"]["enabled"] else None
    data = load_ml_ready_npz(glob_dir, cfg["carbonglobe"]["ml_ready_file"], max_sites=max_sites)
    stats = load_stats(stat_dir, cfg["carbonglobe"]["stats_file"])
    mask2d = load_mask2d(stat_dir, cfg["carbonglobe"]["mask_file"])

    # Sprint 1: Kiểm toán chất lượng dữ liệu thô
    audit_report = {
        "source_tag": tag,
        "x_train_audit": audit_array(data["x_train"], "x_train"),
        "y_train_audit": audit_array(data["y_train"], "y_train"),
        "mask2d_valid_sites": int(mask2d.astype(bool).sum()),
        "target_physical_summary": audit_targets_summary(data["y_train"], TREE_BAND),
    }
    audit_path = save_audit_report(audit_report, interim_dir / f"audit_{tag}.json")

    # Sprint 2: Chuẩn hóa -> Làm sạch -> Tạo đặc trưng 1-step (144 features -> 7 targets)
    x_norm = normalize_array(data["x_train"], stats["x_mean"], stats["x_std"], eps=cfg["cleaning"]["eps_std"])
    y_norm = normalize_array(data["y_train"], stats["y_mean"], stats["y_std"], eps=cfg["cleaning"]["eps_std"])

    clip_range = tuple(cfg["cleaning"]["clip_z_range"])
    x_clean, y_clean, clean_stats = clean_carbonglobe_arrays(x_norm, y_norm, clip_z_range=clip_range)

    X_feat, Y_target = build_one_step_dataset(x_clean, y_clean, ages=AGES)
    out_npz = processed_dir / f"one_step_{tag}.npz"
    np.savez_compressed(out_npz, X_feat=X_feat, Y_target=Y_target)

    print(f"[Sprint 1 - Audit] Báo cáo kiểm toán: {audit_path}")
    print(f"[Sprint 2 - Clean] {clean_stats}")
    print(f"[Sprint 2 - Features] X_feat={X_feat.shape}, Y_target={Y_target.shape} -> {out_npz}")


if __name__ == "__main__":
    main()
