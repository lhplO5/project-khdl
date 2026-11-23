#!/usr/bin/env python
"""Visualise the CarbonGlobe public test files (test_ssp126.npz / test_ssp585.npz).

    test_x  : (S, 40, 12, 136)  forcing/input features, 40 years x 12 months
    test_y0 : (S, 15, 7)        initial state per stand age (15 ages) and target (7)

Self-contained: no other project file needed.

Usage:
    uv run --with numpy --with matplotlib src/visualize_public.py \
        --public-dir data/raw/public --scenario ssp126 --out-dir figs_public

If sites_ssp.csv has latitude/longitude columns, site maps are drawn as well.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TREE_BAND = ["height", "agb", "soil", "lai", "gpp", "npp", "rh"]
AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ----------------------------------------------------------------------------- helpers
def save(fig, out_dir: Path, name: str):
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def to_global_grid(values, mask2d):
    """Place per-site values onto the (360, 720) grid (site i -> i-th land cell, row-major)."""
    grid = np.full(mask2d.shape, np.nan, dtype=np.float32)
    flat_pos = np.flatnonzero(mask2d.astype(bool).ravel())
    n = min(len(values), len(flat_pos))
    grid.ravel()[flat_pos[:n]] = values[:n]
    return grid


def zscore(a, axis=-1):
    mu = np.nanmean(a, axis=axis, keepdims=True)
    sd = np.nanstd(a, axis=axis, keepdims=True)
    return (a - mu) / np.where(sd < 1e-8, 1.0, sd)


def robust_lims(a, lo=2, hi=98):
    a = a[np.isfinite(a)]
    return (np.percentile(a, lo), np.percentile(a, hi)) if a.size else (0, 1)


def panel_grid(n, ncols=4, w=5.0, h=3.2):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    return fig, axes


# ----------------------------------------------------------------------------- feature plots
def plot_feature_summary(x, names, out_dir):
    F = x.shape[-1]
    flat = x.reshape(-1, F)
    mean, std = np.nanmean(flat, 0), np.nanstd(flat, 0)
    nan_frac = np.mean(~np.isfinite(flat), 0)
    zero_frac = np.mean(flat == 0, 0)
    idx = np.arange(F)
    fig, axes = plt.subplots(4, 1, figsize=(max(12, F * 0.12), 11), sharex=True)
    axes[0].bar(idx, mean, color="tab:blue"); axes[0].set_ylabel("mean")
    axes[1].bar(idx, std, color="tab:purple"); axes[1].set_ylabel("std")
    axes[2].bar(idx, nan_frac, color="tab:red"); axes[2].set_ylabel("NaN fraction")
    axes[3].bar(idx, zero_frac, color="tab:gray"); axes[3].set_ylabel("zero fraction")
    axes[3].set_xticks(idx[:: max(1, F // 40)])
    axes[3].set_xticklabels([names[i] for i in idx[:: max(1, F // 40)]], rotation=90, fontsize=7)
    fig.suptitle(f"Input feature summary ({F} features)")
    fig.tight_layout()
    save(fig, out_dir, "06_feature_summary.png")


def plot_feature_monthly(x, names, out_dir):
    clim = np.nanmean(x, axis=(0, 1))                 # (12, F)
    z = zscore(clim, axis=0).T                        # (F, 12), z-scored per feature across months
    fig, ax = plt.subplots(figsize=(8, max(6, z.shape[0] * 0.13)))
    im = ax.imshow(z, aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
    ax.set_xticks(range(12)); ax.set_xticklabels(MONTHS, rotation=60)
    step = max(1, len(names) // 60)
    ax.set_yticks(range(0, len(names), step)); ax.set_yticklabels(names[::step], fontsize=6)
    fig.colorbar(im, ax=ax, label="z-score across months")
    ax.set_title("Monthly climatology of input features")
    save(fig, out_dir, "07_feature_monthly.png")


def plot_feature_trend(x, names, out_dir):
    yearly = np.nanmean(x, axis=(0, 2))               # (40, F)
    z = zscore(yearly, axis=0).T                      # (F, 40)
    fig, ax = plt.subplots(figsize=(10, max(6, z.shape[0] * 0.13)))
    im = ax.imshow(z, aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
    ax.set_xlabel("year")
    step = max(1, len(names) // 60)
    ax.set_yticks(range(0, len(names), step)); ax.set_yticklabels(names[::step], fontsize=6)
    fig.colorbar(im, ax=ax, label="z-score across years")
    ax.set_title("Yearly evolution of input features (mean over sites and months)")
    save(fig, out_dir, "08_feature_trend.png")


def plot_feature_correlation(x, names, out_dir, rng, n_rows=200_000):
    F = x.shape[-1]
    flat = x.reshape(-1, F)
    if flat.shape[0] > n_rows:
        flat = flat[rng.choice(flat.shape[0], n_rows, replace=False)]
    flat = np.where(np.isfinite(flat), flat, np.nanmean(flat, axis=0, keepdims=True))
    corr = np.corrcoef(flat, rowvar=False)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    step = max(1, F // 40)
    ax.set_xticks(range(0, F, step)); ax.set_xticklabels(names[::step], rotation=90, fontsize=6)
    ax.set_yticks(range(0, F, step)); ax.set_yticklabels(names[::step], fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Input feature correlation")
    save(fig, out_dir, "09_feature_correlation.png")


def plot_feature_histograms(x, names, out_dir, rng, n_hist, n_sample=500_000):
    F = x.shape[-1]
    n_hist = min(n_hist, F)
    flat = x.reshape(-1, F)
    if flat.shape[0] > n_sample:
        flat = flat[rng.choice(flat.shape[0], n_sample, replace=False)]
    fig, axes = panel_grid(n_hist, ncols=6, w=3.2, h=2.4)
    for i in range(n_hist):
        v = flat[:, i]
        v = v[np.isfinite(v)]
        if v.size:
            lo, hi = np.percentile(v, [0.5, 99.5])
            if hi <= lo:
                hi = lo + 1
            axes[i].hist(v, bins=60, range=(lo, hi), color="tab:cyan")
        axes[i].set_title(names[i], fontsize=8)
        axes[i].tick_params(labelsize=6)
    fig.suptitle(f"Histograms of the first {n_hist} input features")
    fig.tight_layout()
    save(fig, out_dir, "10_feature_histograms.png")


def read_site_coords(csv_path):
    """Return (lat, lon) arrays from sites_ssp.csv, or None if no lat/lon-like columns."""
    if not csv_path.exists():
        return None
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    cols = list(rows[0].keys())
    print(f"  sites csv columns: {cols}")
    lat_c = next((c for c in cols if c.strip().lower() in ("lat", "latitude", "y")), None)
    lon_c = next((c for c in cols if c.strip().lower() in ("lon", "lng", "long", "longitude", "x")), None)
    if lat_c is None or lon_c is None:
        print("  no lat/lon columns found -> skipping site maps")
        return None
    return (np.array([float(r[lat_c]) for r in rows]), np.array([float(r[lon_c]) for r in rows]))


def plot_y0_distributions(y0, out_dir):
    fig, axes = panel_grid(len(TREE_BAND))
    for t, name in enumerate(TREE_BAND):
        v = y0[..., t].ravel()
        v = v[np.isfinite(v)]
        lo, hi = np.percentile(v, [0.5, 99.5])
        axes[t].hist(v, bins=80, range=(lo, hi), color="tab:green")
        axes[t].set_yscale("log")
        axes[t].set_title(f"{name} (mean={v.mean():.3g}, std={v.std():.3g})", fontsize=9)
    fig.suptitle("test_y0: initial-state distributions (all sites and ages)")
    fig.tight_layout()
    save(fig, out_dir, "01_y0_distributions.png")


def plot_y0_vs_age(y0, out_dir):
    ages = AGES[: y0.shape[1]]
    fig, axes = panel_grid(len(TREE_BAND))
    for t, name in enumerate(TREE_BAND):
        q25, q50, q75 = np.nanpercentile(y0[:, :, t], [25, 50, 75], axis=0)
        axes[t].fill_between(ages, q25, q75, alpha=0.3, color="tab:orange")
        axes[t].plot(ages, q50, "o-", color="tab:orange", ms=3)
        axes[t].set_xscale("log")
        axes[t].set_title(name)
        axes[t].set_xlabel("stand age (years)")
        axes[t].grid(alpha=0.3)
    fig.suptitle("test_y0 vs stand age (median and IQR across sites)")
    fig.tight_layout()
    save(fig, out_dir, "02_y0_vs_age.png")


def plot_y0_correlation(y0, out_dir):
    flat = y0.reshape(-1, y0.shape[-1])
    flat = flat[np.isfinite(flat).all(1)]
    c = np.corrcoef(flat, rowvar=False)
    T = len(TREE_BAND)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(c, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(T)); ax.set_xticklabels(TREE_BAND, rotation=45)
    ax.set_yticks(range(T)); ax.set_yticklabels(TREE_BAND)
    for i in range(T):
        for j in range(T):
            ax.text(j, i, f"{c[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    ax.set_title("test_y0 target correlation")
    save(fig, out_dir, "03_y0_correlation.png")


def plot_site_maps(y0, coords, out_dir):
    lat, lon = coords
    if len(lat) != y0.shape[0]:
        print(f"  sites csv has {len(lat)} rows but test_y0 has {y0.shape[0]} -> skipping maps")
        return
    fig, axes = plt.subplots(2, 4, figsize=(20, 7))
    axes = axes.ravel()
    for t, name in enumerate(TREE_BAND):
        v = np.nanmean(y0[:, :, t], axis=1)  # mean over ages
        lo, hi = np.nanpercentile(v, [2, 98])
        sc = axes[t].scatter(lon, lat, c=v, s=4, cmap="viridis", vmin=lo, vmax=hi)
        axes[t].set_title(f"{name} (initial, mean over ages)")
        fig.colorbar(sc, ax=axes[t], fraction=0.046)
    axes[7].scatter(lon, lat, s=2, color="k")
    axes[7].set_title("site locations")
    for ax in axes:
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
    fig.tight_layout()
    save(fig, out_dir, "04_site_maps.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--public-dir", required=True)
    ap.add_argument("--scenario", default="ssp126", help="ssp126 or ssp585")
    ap.add_argument("--max-sites", type=int, default=2000, help="Subsample for feature plots (0 = all)")
    ap.add_argument("--n-hist", type=int, default=24)
    ap.add_argument("--feature-names", default=None)
    ap.add_argument("--out-dir", default="figs_public")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pub, out_dir = Path(args.public_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    path = pub / f"test_{args.scenario}.npz"
    print(f"Loading {path} ...")
    d = np.load(path)
    x = d["test_x"].astype(np.float32)
    y0 = d["test_y0"].astype(np.float32)
    print(f"  test_x {x.shape}, test_y0 {y0.shape}")

    F = x.shape[-1]
    names = [f"f{i:03d}" for i in range(F)]
    if args.feature_names:
        loaded = [l.strip() for l in Path(args.feature_names).read_text().splitlines() if l.strip()]
        if len(loaded) == F:
            names = loaded

    print("Target (y0) plots ...")
    plot_y0_distributions(y0, out_dir)
    plot_y0_vs_age(y0, out_dir)
    plot_y0_correlation(y0, out_dir)
    coords = read_site_coords(pub / "sites_ssp.csv")
    if coords is not None:
        plot_site_maps(y0, coords, out_dir)

    if args.max_sites and args.max_sites < x.shape[0]:
        x = x[np.sort(rng.choice(x.shape[0], args.max_sites, replace=False))]
        print(f"  subsampled to {x.shape[0]} sites for feature plots")

    print("Feature plots ...")
    plot_feature_summary(x, names, out_dir)
    plot_feature_monthly(x, names, out_dir)
    plot_feature_trend(x, names, out_dir)
    plot_feature_correlation(x, names, out_dir, rng)
    plot_feature_histograms(x, names, out_dir, rng, args.n_hist)
    print(f"Done. Figures in {out_dir.resolve()}")


if __name__ == "__main__":
    main()