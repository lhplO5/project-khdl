#!/usr/bin/env python
"""Visualise the CarbonGlobe dataset (res_train4_test8.npz + mask2d.npy + data_stats.npz).

Array layout (taken from the loader code):
    x_train : (S, 40, 12, 136)      S sites, 40 years, 12 months, 136 input features
    y_train : (15, S, 41, 12, 7)    15 stand ages, S sites, 41 years, 12 months, 7 targets
    mask2d  : (360, 720)            global land mask (0.5 deg grid)

Figures written to --out-dir:
    01_target_maps.png          global maps of each target (initial state + 40-yr change)
    02_target_distributions.png histograms of each target
    03_target_trajectories.png  40-yr trajectories (median + 10-90% band) per target
    04_target_vs_age.png        target vs stand age
    05_target_seasonal.png      monthly cycle of each target
    06_feature_summary.png      mean / std / missing fraction of each of the 136 inputs
    07_feature_monthly.png      monthly climatology of every input feature (z-scored)
    08_feature_trend.png        yearly evolution of every input feature (z-scored)
    09_feature_correlation.png  correlation matrix between input features
    10_feature_histograms.png   histograms of the first --n-hist input features

Usage:
    python visualize_carbonglobe.py --data-dir path/to/carbonglobe
    python visualize_carbonglobe.py --data-dir path/to/carbonglobe --max-sites 5000
    python visualize_carbonglobe.py --data-dir ... --feature-names names.txt   # one name per line

Requires: numpy, matplotlib.
"""

import argparse
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


# ----------------------------------------------------------------------------- target plots
def plot_target_maps(y, mask2d, out_dir, force=False):
    if mask2d is None:
        print("  mask2d not found -> skipping maps")
        return
    S = y.shape[1]
    n_land = int(mask2d.astype(bool).sum())
    if S != n_land and not force:
        print(f"  S={S} but mask has {n_land} land cells -> site-to-grid mapping unknown; "
              "skipping maps (use --force-maps to draw them anyway).")
        return
    fig, axes = plt.subplots(2, 7, figsize=(7 * 4.2, 2 * 2.6))
    for t, name in enumerate(TREE_BAND):
        y0 = np.nanmean(y[:, :, 0, :, t], axis=(0, 2))      # initial state, mean over ages & months
        y40 = np.nanmean(y[:, :, -1, :, t], axis=(0, 2))    # final year
        delta = y40 - y0
        g0, gd = to_global_grid(y0, mask2d), to_global_grid(delta, mask2d)
        vmin, vmax = robust_lims(g0)
        im = axes[0, t].imshow(g0, cmap="viridis", vmin=vmin, vmax=vmax,
                               extent=[-180, 180, -90, 90], origin="upper")
        axes[0, t].set_title(f"{name} (year 0)", fontsize=10)
        fig.colorbar(im, ax=axes[0, t], fraction=0.046, pad=0.02, orientation="horizontal")
        m = max(abs(x) for x in robust_lims(gd)) or 1.0
        im = axes[1, t].imshow(gd, cmap="RdBu_r", vmin=-m, vmax=m,
                               extent=[-180, 180, -90, 90], origin="upper")
        axes[1, t].set_title(f"{name}: change yr 0 -> 40", fontsize=10)
        fig.colorbar(im, ax=axes[1, t], fraction=0.046, pad=0.02, orientation="horizontal")
        for r in (0, 1):
            axes[r, t].set_xticks([])
            axes[r, t].set_yticks([])
    fig.suptitle("Targets averaged over stand ages and months", y=1.0)
    save(fig, out_dir, "01_target_maps.png")


def plot_target_distributions(y, out_dir, rng, n_sample=2_000_000):
    fig, axes = panel_grid(len(TREE_BAND))
    for t, name in enumerate(TREE_BAND):
        v = y[..., t].ravel()
        if v.size > n_sample:
            v = v[rng.choice(v.size, n_sample, replace=False)]
        v = v[np.isfinite(v)]
        lo, hi = np.percentile(v, [0.5, 99.5]) if v.size else (0, 1)
        axes[t].hist(v, bins=100, range=(lo, hi), color="tab:green", alpha=0.85)
        axes[t].set_yscale("log")
        axes[t].set_title(f"{name}  (mean={v.mean():.3g}, std={v.std():.3g})", fontsize=10)
    fig.suptitle("Target distributions (all ages, years, months; log counts)")
    fig.tight_layout()
    save(fig, out_dir, "02_target_distributions.png")


def plot_target_trajectories(y, out_dir):
    years = np.arange(y.shape[2])
    fig, axes = panel_grid(len(TREE_BAND))
    for t, name in enumerate(TREE_BAND):
        ann = np.nanmean(y[..., t], axis=3)               # (A, S, 41) annual mean
        flat = ann.reshape(-1, ann.shape[-1])              # (A*S, 41)
        q10, q50, q90 = np.nanpercentile(flat, [10, 50, 90], axis=0)
        axes[t].fill_between(years, q10, q90, alpha=0.3, color="tab:blue", label="10-90%")
        axes[t].plot(years, q50, color="tab:blue", label="median")
        axes[t].set_title(name)
        axes[t].set_xlabel("year")
        axes[t].grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle("Target trajectories (pooled over ages and sites)")
    fig.tight_layout()
    save(fig, out_dir, "03_target_trajectories.png")


def plot_target_vs_age(y, out_dir):
    ages = AGES[: y.shape[0]]
    fig, axes = panel_grid(len(TREE_BAND))
    for t, name in enumerate(TREE_BAND):
        per_age = np.nanmean(y[:, :, :, :, t], axis=(2, 3))   # (A, S)
        q25, q50, q75 = np.nanpercentile(per_age, [25, 50, 75], axis=1)
        axes[t].fill_between(ages, q25, q75, alpha=0.3, color="tab:orange")
        axes[t].plot(ages, q50, "o-", color="tab:orange", ms=3)
        axes[t].set_xscale("log")
        axes[t].set_title(name)
        axes[t].set_xlabel("stand age (years)")
        axes[t].grid(alpha=0.3)
    fig.suptitle("Target vs stand age (median and IQR across sites)")
    fig.tight_layout()
    save(fig, out_dir, "04_target_vs_age.png")


def plot_target_seasonal(y, out_dir):
    fig, axes = panel_grid(len(TREE_BAND))
    for t, name in enumerate(TREE_BAND):
        monthly = np.nanmean(y[..., t], axis=(0, 2))            # (S, 12)
        q25, q50, q75 = np.nanpercentile(monthly, [25, 50, 75], axis=0)
        axes[t].fill_between(range(12), q25, q75, alpha=0.3, color="tab:red")
        axes[t].plot(range(12), q50, "o-", color="tab:red", ms=3)
        axes[t].set_xticks(range(12))
        axes[t].set_xticklabels(MONTHS, rotation=60, fontsize=8)
        axes[t].set_title(name)
        axes[t].grid(alpha=0.3)
    fig.suptitle("Seasonal cycle of targets (median and IQR across sites)")
    fig.tight_layout()
    save(fig, out_dir, "05_target_seasonal.png")


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


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="Folder containing the npz/npy files")
    ap.add_argument("--filename", default="res_train4_test8.npz")
    ap.add_argument("--mask-dir", default=None, help="Folder with mask2d.npy (default: --data-dir)")
    ap.add_argument("--mask-file", default="mask2d.npy")
    ap.add_argument("--force-maps", action="store_true",
                    help="Draw maps even if #sites != #land cells in the mask (likely misaligned)")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--max-sites", type=int, default=5000,
                    help="Random site subsample for the statistical plots (maps use all sites). 0 = all")
    ap.add_argument("--n-hist", type=int, default=24, help="Number of feature histograms")
    ap.add_argument("--feature-names", default=None, help="Text file with one feature name per line")
    ap.add_argument("--out-dir", default="carbonglobe_figs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    path = data_dir / args.filename
    if not path.exists():
        raise FileNotFoundError(path)
    print(f"Loading {path} ...")
    npz = np.load(path)
    x_key, y_key = f"x_{args.split}", f"y_{args.split}"
    x = npz[x_key].astype(np.float32)   # (S, 40, 12, 136)
    y = npz[y_key].astype(np.float32) if y_key in npz else None  # (15, S, 41, 12, 7)
    print(f"  {x_key}: {x.shape}" + (f"   {y_key}: {y.shape}" if y is not None else ""))

    mask_path = Path(args.mask_dir or data_dir) / args.mask_file
    mask2d = np.load(mask_path) if mask_path.exists() else None

    F = x.shape[-1]
    names = [f"f{i:03d}" for i in range(F)]
    if args.feature_names:
        loaded = [l.strip() for l in Path(args.feature_names).read_text().splitlines() if l.strip()]
        if len(loaded) == F:
            names = loaded
        else:
            print(f"  warning: {len(loaded)} names given but {F} features; ignoring names")

    # Maps use every site; statistical plots use a random subsample to stay fast.
    if y is not None:
        print("Target plots ...")
        plot_target_maps(y, mask2d, out_dir, args.force_maps)

    S = x.shape[0]
    if args.max_sites and args.max_sites < S:
        idx = np.sort(rng.choice(S, args.max_sites, replace=False))
        x = x[idx]
        if y is not None:
            y = y[:, idx]
        print(f"  subsampled to {len(idx)} of {S} sites for statistical plots")

    if y is not None:
        plot_target_distributions(y, out_dir, rng)
        plot_target_trajectories(y, out_dir)
        plot_target_vs_age(y, out_dir)
        plot_target_seasonal(y, out_dir)

    print("Feature plots ...")
    plot_feature_summary(x, names, out_dir)
    plot_feature_monthly(x, names, out_dir)
    plot_feature_trend(x, names, out_dir)
    plot_feature_correlation(x, names, out_dir, rng)
    plot_feature_histograms(x, names, out_dir, rng, args.n_hist)
    print(f"Done. Figures in {out_dir.resolve()}")


if __name__ == "__main__":
    main()