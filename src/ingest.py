"""Tải và giải nén tự động dữ liệu CarbonGlobe từ Kaggle vào data/raw/."""

import argparse
import subprocess
import zipfile
from pathlib import Path
from typing import List


COMPETITION_SLUG = "ieee-bigdata-cup-2026-ai-emulation-challenge"


def download_kaggle_competition(
    output_dir: str | Path = "data/raw",
    competition: str = COMPETITION_SLUG,
    unzip: bool = True,
) -> List[Path]:
    """Tải dữ liệu cuộc thi qua Kaggle CLI và giải nén nếu cần."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        competition,
        "-p",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Lỗi khi tải dữ liệu Kaggle (mã {result.returncode}): {result.stderr.strip()}\n"
            "Hãy kiểm tra đã đặt file kaggle.json vào ~/.kaggle/ chưa."
        )

    if unzip:
        for archive in out_path.glob("*.zip"):
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(out_path)
            archive.unlink()

    return sorted(p for p in out_path.iterdir() if p.name != ".gitkeep")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tải dữ liệu CarbonGlobe từ Kaggle.")
    parser.add_argument("--output-dir", default="data/raw", help="Thư mục lưu dữ liệu thô.")
    parser.add_argument("--competition", default=COMPETITION_SLUG, help="Slug cuộc thi Kaggle.")
    parser.add_argument("--no-unzip", action="store_true", help="Không giải nén file .zip sau khi tải.")
    args = parser.parse_args()

    files = download_kaggle_competition(
        output_dir=args.output_dir,
        competition=args.competition,
        unzip=not args.no_unzip,
    )
    print(f"Đã tải {len(files)} file vào {args.output_dir}:")
    for f in files:
        size_mb = f.stat().st_size / 1e6
        print(f"  - {f.name} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
