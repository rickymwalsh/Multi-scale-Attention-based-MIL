"""Analyse and plot the distribution of offline-extracted MIL features.

Supports all three feature storage layouts produced by offline_feature_extraction.py:

  - fpn / backbone_pyramid  : feat_dir/multi_scale/{patient}/{image}/C4_patch_features.pt
                                                                      C5_patch_features.pt
  - msp                     : feat_dir/patch_size-{scale}/{patient}/{image}/patch_features.pt
                              (one sub-directory per scale)
  - single (default)        : feat_dir/patch_size-{scale}/{patient}/{image}/patch_features.pt
                              (single scale)

Because the dataset is large (~20 k images/bags), a random subset of bags is sampled and
then a random subset of feature values is drawn from each bag so that memory usage
stays manageable.

Usage example (FPN features, default dataset)::

    python Datasets/analyse_features.py \\
        --feat_dir /data/walsh/datasets/vindr-mammo-mil-b2 \\
        --multi_scale_model fpn \\
        --n_bags 500 \\
        --output_dir /home/walsh/gaze/gazeMIL/results/feature_analysis
"""

import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless servers
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


# ── Quantile levels reported in the summary table ──────────────────────────
QUANTILE_LEVELS = [0.0, 0.05, 0.25, 0.50, 0.75, 0.95, 1.0]
QUANTILE_LABELS = ["min", "5th", "25th", "50th", "75th", "95th", "max"]


# ── Helpers ─────────────────────────────────────────────────────────────────

def _all_bag_dirs_fpn(feat_dir: Path):
    """Return list of (patient, image, bag_dir) for FPN layout."""
    base = feat_dir / "multi_scale"
    dirs = []
    for patient_dir in base.iterdir():
        if not patient_dir.is_dir():
            continue
        for image_dir in patient_dir.iterdir():
            if (image_dir / "C4_patch_features.pt").exists():
                dirs.append(image_dir)
    return dirs


def _all_bag_dirs_single(feat_dir: Path, scale: int):
    """Return list of bag_dirs for single-scale layout."""
    base = feat_dir / f"patch_size-{scale}"
    dirs = []
    for patient_dir in base.iterdir():
        if not patient_dir.is_dir():
            continue
        for image_dir in patient_dir.iterdir():
            if (image_dir / "patch_features.pt").exists():
                dirs.append(image_dir)
    return dirs


def _sample_values(tensor: torch.Tensor, max_values: int) -> np.ndarray:
    """Flatten *tensor* to 1-D float32 and randomly subsample at most *max_values* scalars."""
    flat = tensor.float().reshape(-1).numpy()
    if len(flat) > max_values:
        idx = np.random.choice(len(flat), size=max_values, replace=False)
        flat = flat[idx]
    return flat


def _collect_values(bag_dirs, load_fn, max_values_per_bag: int, desc: str):
    """Stream through *bag_dirs*, call *load_fn(bag_dir)* → dict[label, Tensor],
    and accumulate sampled values per label.  Returns dict[label, np.ndarray]."""
    accumulated = {}
    for bag_dir in tqdm(bag_dirs, desc=desc, unit="bag"):
        try:
            tensors = load_fn(bag_dir)
        except Exception as exc:
            print(f"  Warning: could not load {bag_dir}: {exc}")
            continue
        for label, tensor in tensors.items():
            vals = _sample_values(tensor, max_values_per_bag)
            if label not in accumulated:
                accumulated[label] = []
            accumulated[label].append(vals)

    return {label: np.concatenate(chunks) for label, chunks in accumulated.items()}


# ── Load functions per model type ────────────────────────────────────────────

def _load_fpn(bag_dir: Path):
    c4 = torch.load(bag_dir / "C4_patch_features.pt", weights_only=True)
    c5 = torch.load(bag_dir / "C5_patch_features.pt", weights_only=True)
    return {"C4": c4, "C5": c5}


def _load_single(bag_dir: Path):
    x = torch.load(bag_dir / "patch_features.pt", weights_only=True)
    return {"features": x}


def _load_msp(scales):
    """Return a load function that reads all scales from their respective subdirs."""
    def _fn(image_dir: Path):
        # For MSP the path passed is the image_dir under the *first* scale;
        # reconstruct sibling paths for the other scales.
        # Convention: image_dir is <feat_dir>/patch_size-<scale0>/<patient>/<image>
        # We stored (scale_label, full_path) pairs in the bag_dirs list for MSP.
        raise NotImplementedError("Use _collect_values_msp for MSP.")
    return _fn


# ── Plotting helpers ─────────────────────────────────────────────────────────

def _plot_histogram(values: np.ndarray, label: str, ax: plt.Axes, n_bins: int = 100):
    """Plot a histogram of *values* on *ax*."""
    ax.hist(values, bins=n_bins, density=True, alpha=0.75, edgecolor="none")
    ax.set_title(f"{label} — feature value distribution\n"
                 f"({len(values):,} sampled values)")
    ax.set_xlabel("Feature value")
    ax.set_ylabel("Density")


def _print_quantile_table(values: np.ndarray, label: str):
    qs = np.quantile(values, QUANTILE_LEVELS)
    print(f"\n{'─' * 50}")
    print(f"  {label}")
    print(f"{'─' * 50}")
    for name, q in zip(QUANTILE_LABELS, qs):
        print(f"  {name:>5s} : {q:+.6f}")
    print(f"  mean  : {values.mean():+.6f}")
    print(f"  std   : {values.std():+.6f}")
    print(f"{'─' * 50}")
    return qs


def _plot_quantile_summary(all_values: dict, output_dir: Path):
    """Bar chart comparing quantiles across feature sets."""
    labels = list(all_values.keys())
    n = len(QUANTILE_LEVELS)
    x = np.arange(n)
    width = 0.8 / max(len(labels), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, label in enumerate(labels):
        qs = np.quantile(all_values[label], QUANTILE_LEVELS)
        offset = (i - len(labels) / 2 + 0.5) * width
        bars = ax.bar(x + offset, qs, width=width, label=label, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(QUANTILE_LABELS)
    ax.set_ylabel("Feature value")
    ax.set_title("Quantile comparison across feature sets")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    fig.tight_layout()
    path = output_dir / "quantile_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def _save_histogram_grid(all_values: dict, output_dir: Path, n_bins: int):
    """One histogram per feature set, arranged in a grid figure."""
    n = len(all_values)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows),
                              squeeze=False)
    for ax_idx, (label, values) in enumerate(all_values.items()):
        ax = axes[ax_idx // ncols][ax_idx % ncols]
        _plot_histogram(values, label, ax, n_bins=n_bins)

    # Hide any unused axes
    for ax_idx in range(n, nrows * ncols):
        axes[ax_idx // ncols][ax_idx % ncols].set_visible(False)

    fig.tight_layout()
    path = output_dir / "feature_histograms.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def _save_per_channel_boxplots(bag_dirs, load_fn, scale_label: str,
                                output_dir: Path, n_sample_bags: int = 200):
    """Plot per-channel boxplots showing how each channel's mean activation
    varies across bags.

    For 4-D tensors (patches × C × H × W) the mean is taken over patches,
    H, and W, yielding one value per channel per bag.  For 2-D tensors
    (patches × features) the mean is taken over patches, yielding one value
    per feature dimension per bag.  The resulting (n_bags × n_channels)
    matrix is then plotted as a box-per-channel boxplot.
    """
    per_bag_means = []  # list of 1-D arrays, one per bag

    for bag_dir in tqdm(bag_dirs[:n_sample_bags],
                        desc=f"Per-channel boxplots ({scale_label})", unit="bag"):
        try:
            tensors = load_fn(bag_dir)
            tensor = tensors.get(scale_label, next(iter(tensors.values())))
        except Exception:
            continue

        t = tensor.float()
        if t.dim() == 4:
            # (num_patches, C, H, W) → mean over patches, H, W → (C,)
            per_chan = t.mean(dim=(0, 2, 3)).numpy()
        elif t.dim() == 2:
            # (num_patches, features) → mean over patches → (features,)
            per_chan = t.mean(dim=0).numpy()
        else:
            continue

        per_bag_means.append(per_chan)

    if not per_bag_means:
        return

    # Stack to (n_bags, n_channels)
    data = np.stack(per_bag_means, axis=0)
    n_channels = data.shape[1]

    # Wider figure for many channels; keep height fixed
    fig_width = max(10, n_channels // 6)
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    ax.boxplot(data, positions=np.arange(n_channels),
               widths=0.6, patch_artist=True,
               medianprops=dict(color="black", linewidth=1.5),
               boxprops=dict(facecolor="steelblue", alpha=0.6),
               flierprops=dict(marker=".", markersize=2, alpha=0.3),
               whiskerprops=dict(linewidth=0.8),
               capprops=dict(linewidth=0.8))

    ax.set_xlabel("Channel index")
    ax.set_ylabel("Mean activation (per bag)")
    ax.set_title(f"{scale_label} — per-channel activation distribution\n"
                 f"({len(per_bag_means)} bags, one box per channel)")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    # Avoid overcrowding x-axis tick labels for wide plots
    tick_step = max(1, n_channels // 30)
    ax.set_xticks(np.arange(0, n_channels, tick_step))
    ax.set_xticklabels(np.arange(0, n_channels, tick_step), fontsize=7)

    fig.tight_layout()
    path = output_dir / f"per_channel_boxplots_{scale_label}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def _save_per_channel_means(bag_dirs, load_fn, scale_label: str,
                             output_dir: Path, n_sample_bags: int = 200):
    """For 4-D feature tensors (patches × C × H × W), plot the mean activation
    per channel across a sample of bags.  Skips 2-D tensors (patches × features)
    since channels are not a separate dimension there."""
    channel_means = None
    n_seen = 0

    for bag_dir in tqdm(bag_dirs[:n_sample_bags],
                        desc=f"Per-channel means ({scale_label})", unit="bag"):
        try:
            tensors = load_fn(bag_dir)
            tensor = tensors.get(scale_label, next(iter(tensors.values())))
        except Exception:
            continue

        if tensor.dim() != 4:
            return  # 2-D features — skip per-channel plot

        # tensor: (num_patches, C, H, W) → mean over patches, H, W → (C,)
        per_chan = tensor.float().mean(dim=(0, 2, 3)).numpy()  # (C,)
        if channel_means is None:
            channel_means = per_chan
        else:
            channel_means += per_chan
        n_seen += 1

    if channel_means is None or n_seen == 0:
        return

    channel_means /= n_seen
    fig, ax = plt.subplots(figsize=(max(8, len(channel_means) // 8), 4))
    ax.bar(np.arange(len(channel_means)), channel_means, width=1.0, alpha=0.8)
    ax.set_xlabel("Channel index")
    ax.set_ylabel("Mean activation")
    ax.set_title(f"{scale_label} — mean activation per channel\n"
                 f"(averaged over {n_seen} bags)")
    fig.tight_layout()
    path = output_dir / f"per_channel_means_{scale_label}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def _save_per_channel_std_analysis(bag_dirs, load_fn, scale_label: str,
                                    output_dir: Path, n_sample_bags: int = 200,
                                    n_bins: int = 80):
    """Analyse and plot the distribution of per-channel standard deviations.

    For each sampled bag the std is computed per channel across all patches and
    spatial positions (for 4-D tensors: patches × C × H × W) or across all
    patches (for 2-D tensors: patches × features).  This yields one std value
    per (channel, bag) pair.

    Outputs
    -------
    per_channel_std_histogram_{scale_label}.png
        Histogram of *all* per-(channel, bag) std values, showing the overall
        spread of channel variability.
    per_channel_mean_std_{scale_label}.png
        Bar chart of the mean std per channel, averaged across bags, so you can
        see which channels are most and least variable.
    Quantile summary printed to stdout and appended to ``channel_std_summary.csv``.
    """
    per_bag_stds = []  # list of (C,) arrays, one per bag

    for bag_dir in tqdm(bag_dirs[:n_sample_bags],
                        desc=f"Per-channel stds ({scale_label})", unit="bag"):
        try:
            tensors = load_fn(bag_dir)
            tensor = tensors.get(scale_label, next(iter(tensors.values())))
        except Exception:
            continue

        t = tensor.float()
        if t.dim() == 4:
            # (num_patches, C, H, W) → flatten patches & spatial → std per channel
            n_patches, C, H, W = t.shape
            per_chan_std = t.permute(1, 0, 2, 3).reshape(C, -1).std(dim=1).numpy()
        elif t.dim() == 2:
            # (num_patches, features) → std per feature dim
            per_chan_std = t.std(dim=0).numpy()
        else:
            continue

        per_bag_stds.append(per_chan_std)

    if not per_bag_stds:
        return

    # (n_bags, C)
    data = np.stack(per_bag_stds, axis=0)
    n_bags_seen, n_channels = data.shape

    # ── 1. Histogram of all (channel, bag) std values ─────────────────────
    all_stds = data.ravel()

    print(f"\n=== Per-channel std distribution — {scale_label} ===")
    print(f"    {n_bags_seen} bags × {n_channels} channels = {len(all_stds):,} values")
    qs = np.quantile(all_stds, QUANTILE_LEVELS)
    for name, q in zip(QUANTILE_LABELS, qs):
        print(f"  {name:>5s} : {q:.6f}")
    print(f"  mean  : {all_stds.mean():.6f}")
    print(f"  std of stds : {all_stds.std():.6f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(all_stds, bins=n_bins, density=True, alpha=0.75, edgecolor="none")
    ax.set_xlabel("Per-channel std")
    ax.set_ylabel("Density")
    ax.set_title(f"{scale_label} — distribution of per-channel standard deviations\n"
                 f"({n_bags_seen} bags × {n_channels} channels)")
    # Mark quantiles as vertical lines
    for lvl, val in zip([0.25, 0.50, 0.75], np.quantile(all_stds, [0.25, 0.50, 0.75])):
        ax.axvline(val, color="firebrick", linewidth=1.0, linestyle="--",
                   label=f"Q{int(lvl*100)} = {val:.3f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    hist_path = output_dir / f"per_channel_std_histogram_{scale_label}.png"
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {hist_path}")

    # ── 2. Bar chart: mean std per channel across bags ────────────────────
    mean_std_per_channel = data.mean(axis=0)  # (C,)

    fig_width = max(10, n_channels // 6)
    fig, ax = plt.subplots(figsize=(fig_width, 4))
    ax.bar(np.arange(n_channels), mean_std_per_channel, width=1.0, alpha=0.8)
    ax.set_xlabel("Channel index")
    ax.set_ylabel("Mean std across bags")
    ax.set_title(f"{scale_label} — mean per-channel std\n"
                 f"(averaged over {n_bags_seen} bags)")
    tick_step = max(1, n_channels // 30)
    ax.set_xticks(np.arange(0, n_channels, tick_step))
    ax.set_xticklabels(np.arange(0, n_channels, tick_step), fontsize=7)
    fig.tight_layout()
    bar_path = output_dir / f"per_channel_mean_std_{scale_label}.png"
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {bar_path}")

    # ── 3. Append to CSV ──────────────────────────────────────────────────
    import pandas as pd
    rows = []
    for c_idx in range(n_channels):
        rows.append({
            "feature_set": scale_label,
            "channel": c_idx,
            "mean_std": float(mean_std_per_channel[c_idx]),
            **{name: float(np.quantile(data[:, c_idx], lvl))
               for name, lvl in zip(QUANTILE_LABELS, QUANTILE_LEVELS)},
        })
    csv_path = output_dir / f"channel_std_summary_{scale_label}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


# ── MSP special handling ──────────────────────────────────────────────────────

def _collect_msp(feat_dir: Path, scales: list, n_bags: int,
                 max_values_per_bag: int):
    """Collect feature values for all MSP scales."""
    # Find bags via the first scale directory
    base0 = feat_dir / f"patch_size-{scales[0]}"
    all_image_dirs = []
    for patient_dir in base0.iterdir():
        if not patient_dir.is_dir():
            continue
        for image_dir in patient_dir.iterdir():
            if (image_dir / "patch_features.pt").exists():
                all_image_dirs.append((patient_dir.name, image_dir.name))

    random.shuffle(all_image_dirs)
    sampled = all_image_dirs[:n_bags]

    accumulated = {scale: [] for scale in scales}
    for patient_id, image_id in tqdm(sampled, desc="Collecting MSP features", unit="bag"):
        for scale in scales:
            pt_path = feat_dir / f"patch_size-{scale}" / patient_id / image_id / "patch_features.pt"
            if not pt_path.exists():
                continue
            try:
                x = torch.load(pt_path, weights_only=True)
                accumulated[scale].append(_sample_values(x, max_values_per_bag))
            except Exception as exc:
                print(f"  Warning: {pt_path}: {exc}")

    return {f"scale-{s}": np.concatenate(v)
            for s, v in accumulated.items() if v}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyse and plot offline MIL feature distributions."
    )
    parser.add_argument("--feat_dir", type=str, required=True,
                        help="Root feature directory (e.g. /data/walsh/datasets/vindr-mammo-mil-b2).")
    parser.add_argument("--multi_scale_model", type=str, default='fpn',
                        choices=["fpn", "backbone_pyramid", "msp", None],
                        help="Feature storage layout. None = single-scale.")
    parser.add_argument("--scales", type=int, nargs="+", default=[16],
                        help="Patch scale(s). Used for 'msp' and single-scale layouts.")
    parser.add_argument("--n_bags", type=int, default=500,
                        help="Number of bags to randomly sample (default: 500).")
    parser.add_argument("--max_values_per_bag", type=int, default=50_000,
                        help="Max scalar values sampled per bag per scale (default: 50 000).")
    parser.add_argument("--n_bins", type=int, default=100,
                        help="Number of histogram bins (default: 100).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str,
                        default="results/feature_analysis",
                        help="Directory where plots and summary are saved.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    feat_dir = Path(args.feat_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Gather bag directories ────────────────────────────────────────────
    if args.multi_scale_model in ("fpn", "backbone_pyramid"):
        all_bag_dirs = _all_bag_dirs_fpn(feat_dir)
        random.shuffle(all_bag_dirs)
        bag_dirs = all_bag_dirs[: args.n_bags]
        print(f"Found {len(all_bag_dirs):,} FPN bags — using {len(bag_dirs):,}.")

        all_values = _collect_values(
            bag_dirs, _load_fpn, args.max_values_per_bag,
            desc="Collecting FPN features"
        )

        # Per-channel mean, boxplots, and std analysis
        for scale_label in all_values:
            _save_per_channel_means(
                bag_dirs, _load_fpn, scale_label, output_dir, n_sample_bags=200
            )
            _save_per_channel_boxplots(
                bag_dirs, _load_fpn, scale_label, output_dir, n_sample_bags=200
            )
            _save_per_channel_std_analysis(
                bag_dirs, _load_fpn, scale_label, output_dir,
                n_sample_bags=200, n_bins=args.n_bins,
            )

    elif args.multi_scale_model == "msp":
        all_values = _collect_msp(
            feat_dir, args.scales, args.n_bags, args.max_values_per_bag
        )
        # Per-channel means not applicable for MSP (2-D features)

    else:
        # Single-scale
        scale = args.scales[0]
        all_bag_dirs = _all_bag_dirs_single(feat_dir, scale)
        random.shuffle(all_bag_dirs)
        bag_dirs = all_bag_dirs[: args.n_bags]
        print(f"Found {len(all_bag_dirs):,} bags at scale {scale} — using {len(bag_dirs):,}.")

        all_values = _collect_values(
            bag_dirs, _load_single, args.max_values_per_bag,
            desc=f"Collecting scale-{scale} features"
        )
        _save_per_channel_boxplots(
            bag_dirs, _load_single, "features", output_dir, n_sample_bags=200
        )
        _save_per_channel_std_analysis(
            bag_dirs, _load_single, "features", output_dir,
            n_sample_bags=200, n_bins=args.n_bins,
        )

    if not all_values:
        print("No features were loaded. Check --feat_dir and --multi_scale_model.")
        return

    # ── Print quantile tables ─────────────────────────────────────────────
    print("\n=== Quantile summary ===")
    for label, values in all_values.items():
        _print_quantile_table(values, label)

    # Save quantile table as CSV
    import pandas as pd
    rows = []
    for label, values in all_values.items():
        qs = np.quantile(values, QUANTILE_LEVELS)
        row = {"feature_set": label}
        for name, q in zip(QUANTILE_LABELS, qs):
            row[name] = float(q)
        row["mean"] = float(values.mean())
        row["std"] = float(values.std())
        rows.append(row)
    csv_path = output_dir / "quantile_summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ── Plots ─────────────────────────────────────────────────────────────
    _save_histogram_grid(all_values, output_dir, n_bins=args.n_bins)
    _plot_quantile_summary(all_values, output_dir)

    print(f"\nAll outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
