import sys
"""Download and extract CarbonGlobe from Kaggle -> data/raw/."""

import argparse
import subprocess
import zipfile
from pathlib import Path

COMPETITION_SLUG = "ieee-bigdata-cup-2026-ai-emulation-challenge"


def get_size_mb(path: Path) -> float:
    """Calculate total size recursively in MiB."""
    if path.is_file():
        total_bytes = path.stat().st_size
    else:
        total_bytes = sum(
            f.stat().st_size
            for f in path.rglob("*")
            if f.is_file()
        )

    return total_bytes / (1024 ** 2)


def download_kaggle_competition(
    output_dir: str | Path = "data/raw",
    competition: str = COMPETITION_SLUG,
    unzip: bool = True,
) -> list[Path]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "competitions",
        "download",
        "-c", competition,
        "-p", str(out_path),
    ]
    print(f"Downloading competition: {competition}...")

    result = subprocess.run(cmd, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Kaggle download failed (exit code {result.returncode}). "
            "Check your Kaggle API credentials."
        )

    if unzip:
        archives = list(out_path.glob("*.zip"))

        for archive in archives:
            print(f"Extracting {archive.name}...")

            with zipfile.ZipFile(archive, "r") as zf:
                bad_file = zf.testzip()
                if bad_file:
                    raise RuntimeError(
                        f"Corrupted archive {archive.name}: {bad_file}"
                    )
                zf.extractall(out_path)

            archive.unlink()
            print(f"Removed archive: {archive.name}")

    return sorted(
        p for p in out_path.iterdir()
        if p.name != ".gitkeep"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download CarbonGlobe from Kaggle."
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Raw data directory.",
    )
    parser.add_argument(
        "--competition",
        default=COMPETITION_SLUG,
        help="Kaggle competition slug.",
    )
    parser.add_argument(
        "--no-unzip",
        action="store_true",
        help="Keep downloaded ZIP archives.",
    )

    args = parser.parse_args()

    files = download_kaggle_competition(
        output_dir=args.output_dir,
        competition=args.competition,
        unzip=not args.no_unzip,
    )

    print(f"\nDownloaded {len(files)} items to {args.output_dir}:")

    for path in files:
        size = get_size_mb(path)
        print(f"  - {path.name:<20} {size:>10.2f} MiB")


if __name__ == "__main__":
    main()
