"""Script điều phối chạy toàn bộ pipeline Tuần 1 (Kiểm toán + Làm sạch dữ liệu)."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.clean import clean_dataset
from src.dataset import build_subset, read_frame, read_pipeline_config
from src.profiling import audit_dataframe, save_audit_report


def generate_synthetic_carbonglobe(
    n_sites: int = 50,
    n_years: int = 40,
    n_features: int = 136,
    n_targets: int = 7,
    seed: int = 42,
) -> pd.DataFrame:
    """Sinh dữ liệu giả lập theo cấu trúc CarbonGlobe để kiểm thử pipeline khi chưa có file gốc."""
    rng = np.random.default_rng(seed)
    total_rows = n_sites * n_years

    site_ids = np.repeat(np.arange(n_sites), n_years)
    years = np.tile(np.arange(1985, 1985 + n_years), n_sites)

    data = {"site_id": site_ids, "year": years}
    for i in range(n_features):
        col = rng.normal(loc=0.0, scale=1.0, size=total_rows).astype("float32")
        # Cài sẵn một ít missing và outlier để kiểm thử bộ làm sạch
        if i % 10 == 0:
            mask = rng.random(total_rows) < 0.03
            col[mask] = np.nan
        if i % 15 == 0:
            col[0] = 999.0
        data[f"feat_{i:03d}"] = col

    for j in range(n_targets):
        data[f"target_{j:02d}"] = rng.normal(loc=10.0, scale=2.5, size=total_rows).astype("float32")

    return pd.DataFrame(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy pipeline dữ liệu Tuần 1.")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Đường dẫn file cấu hình.")
    parser.add_argument("--input-file", default=None, help="File dữ liệu trong data/raw/.")
    parser.add_argument("--synthetic", action="store_true", help="Chạy thử trên dữ liệu mô phỏng.")
    args = parser.parse_args()

    cfg = read_pipeline_config(args.config)
    interim_dir = Path(cfg["paths"]["interim_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        df = generate_synthetic_carbonglobe(
            n_features=cfg["schema"]["feature_count"],
            n_targets=cfg["schema"]["target_count"],
            n_years=cfg["schema"]["horizon_years"],
            seed=cfg["seed"],
        )
        source_tag = "synthetic"
    else:
        if not args.input_file:
            raise ValueError("Cần truyền --input-file hoặc bật cờ --synthetic để chạy thử.")
        df = read_frame(args.input_file, compact=cfg["schema"]["use_float32"])
        source_tag = Path(args.input_file).stem

    if cfg["subset"]["enabled"]:
        df = build_subset(
            df,
            fraction=cfg["subset"]["fraction"],
            seed=cfg["seed"],
            group_col=cfg["subset"]["group_col"],
        )

    audit_report = audit_dataframe(df, key_cols=["site_id", "year"])
    audit_path = save_audit_report(audit_report, interim_dir / f"audit_{source_tag}.json")

    cleaned_df, clean_summary = clean_dataset(
        df,
        group_col="site_id",
        time_col="year",
        protected_cols=["site_id", "year"],
    )
    out_parquet = processed_dir / f"clean_{source_tag}.parquet"
    cleaned_df.to_parquet(out_parquet, index=False)

    print(f"[Audit] Đã lưu báo cáo kiểm toán: {audit_path}")
    print(f"[Clean] Shape: {clean_summary['initial_shape']} -> {clean_summary['final_shape']}")
    print(f"[Clean] Missing còn lại: {clean_summary['remaining_nulls']}")
    print(f"[Output] Đã xuất dữ liệu sạch: {out_parquet}")


if __name__ == "__main__":
    main()
