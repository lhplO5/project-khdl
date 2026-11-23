"""Điều phối chạy toàn bộ pipeline Tuần 1 & Tuần 2 cho bộ dữ liệu CarbonGlobe (.npz)."""

import argparse
from pathlib import Path

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

def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy pipeline CarbonGlobe Tuần 1 & Tuần 2.")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="File cấu hình YAML.")
    args = parser.parse_args()

    cfg = read_pipeline_config(args.config)
    raw_dir = Path(cfg["paths"]["raw_dir"])
    interim_dir = Path(cfg["paths"]["interim_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

   
    glob_dir = raw_dir / cfg["paths"]["global_subdir"]
    stat_dir = raw_dir / cfg["paths"]["stats_subdir"]
    tag = "real"

    subset_cfg = cfg.get("subset", {}) or {}
    max_sites = subset_cfg.get("max_sites") if subset_cfg.get("enabled", False) else None
    strategy = subset_cfg.get("strategy", "random")
    stratify_by = subset_cfg.get("stratify_by") if strategy == "stratified" else None
    data = load_ml_ready_npz(
        glob_dir,
        cfg["carbonglobe"]["ml_ready_file"],
        max_sites=max_sites,
        strategy=strategy,
        seed=cfg.get("seed", 42),
        stratify_by=stratify_by,
        n_bins=subset_cfg.get("n_bins", 5),
        subset_test=bool(subset_cfg.get("subset_test", False)),
    )
    stats = load_stats(stat_dir, cfg["carbonglobe"]["stats_file"])
    mask2d = load_mask2d(stat_dir, cfg["carbonglobe"]["mask_file"])

    # Sprint 1: Kiểm toán chất lượng dữ liệu thô
    audit_report = {
        "source_tag": tag,
        "subset": {
            "enabled": bool(subset_cfg.get("enabled", False)),
            "max_sites": max_sites,
            "strategy": strategy,
            "stratify_by": stratify_by,
            "seed": cfg.get("seed", 42),
            "subset_test": bool(subset_cfg.get("subset_test", False)),
            "selected_sites": int(len(data["site_indices"])),
            "site_indices": data["site_indices"].tolist(),
            "test_sites": int(data["x_test"].shape[0]) if "x_test" in data else 0,
        },
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
