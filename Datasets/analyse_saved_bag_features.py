"""Analyse saved bag instance-feature files for class separability.

Files are produced by ``MultiScalePyramidalMILmodel.save_collected_features()``
and have the structure::

    {
        'epoch': int,
        'scales': list[str|int],
        'labels': Tensor[num_bags],
        'examples': {scale_key: Tensor[num_bags, num_instances, feat_dim]},
    }

For each file, bag-level feature vectors are derived by pooling over instances
per scale (mean, max, and mean+max concatenated are all evaluated), then all
scales are concatenated.  Multiple classifiers are evaluated with stratified
5-fold cross-validation.  A Fisher's criterion score quantifies linear
separability without requiring a train/test split.  PCA scatter plots are saved
alongside the JSON metrics.

Usage example::

    python Datasets/analyse_saved_bag_features.py \\
        --input_dir results/my_run/examples \\
        --output_dir results/my_run/feature_analysis
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_file(
    file_path: Path,
) -> tuple[np.ndarray, list, dict[str, np.ndarray], dict[str, np.ndarray]] | None:
    """Load one .pt file.

    Returns ``(y, scale_order, per_scale_mean, per_scale_max)`` or ``None`` on failure.
    Each per_scale dict maps scale_key → ndarray of shape ``(num_bags, feat_dim)``.
    """
    payload = torch.load(file_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "examples" not in payload:
        print(f"  Skipping {file_path.name}: unexpected payload format")
        return None

    examples = payload["examples"]
    if not isinstance(examples, dict) or len(examples) == 0:
        print(f"  Skipping {file_path.name}: no examples dict")
        return None

    raw_labels = payload.get("labels", None)
    if raw_labels is None:
        print(f"  Skipping {file_path.name}: no labels")
        return None
    y = raw_labels.detach().cpu().numpy().reshape(-1)

    scale_order = sorted(examples.keys())
    per_scale_mean: dict[str, np.ndarray] = {}
    per_scale_max: dict[str, np.ndarray] = {}
    for key in scale_order:
        t = examples[key]
        if not isinstance(t, torch.Tensor) or t.dim() != 3:
            print(f"  Skipping {file_path.name}: bad tensor for scale '{key}'")
            return None
        tf = t.float()
        per_scale_mean[key] = tf.mean(dim=1).cpu().numpy()          # (bags, feat_dim)
        per_scale_max[key] = tf.max(dim=1).values.cpu().numpy()     # (bags, feat_dim)

    if len(set(arr.shape[0] for arr in per_scale_mean.values())) != 1:
        print(f"  Skipping {file_path.name}: inconsistent bag counts across scales")
        return None

    if per_scale_mean[scale_order[0]].shape[0] != y.shape[0]:
        print(f"  Skipping {file_path.name}: label count != bag count")
        return None

    return (y >= 0.5).astype(np.int64), scale_order, per_scale_mean, per_scale_max


def load_dataset(
    files: Sequence[Path],
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Aggregate all files into a single dataset.

    Returns ``(y, scale_order, per_scale_mean_X, per_scale_max_X)``.
    Each per_scale dict maps scale_key → ndarray ``(num_bags, feat_dim)``.
    """
    y_parts: list[np.ndarray] = []
    mean_parts: dict[str, list[np.ndarray]] = {}
    max_parts: dict[str, list[np.ndarray]] = {}
    scale_order: list[str] = []

    for fp in files:
        result = _load_file(fp)
        if result is None:
            continue
        y, s_order, per_scale_mean, per_scale_max = result
        if not scale_order:
            scale_order = s_order
            mean_parts = {k: [] for k in s_order}
            max_parts = {k: [] for k in s_order}

        y_parts.append(y)
        for k in scale_order:
            mean_parts[k].append(per_scale_mean[k])
            max_parts[k].append(per_scale_max[k])

    if not y_parts:
        raise RuntimeError("No valid labelled samples loaded from the provided files.")

    y = np.concatenate(y_parts, axis=0)
    per_scale_mean_X = {k: np.concatenate(v, axis=0) for k, v in mean_parts.items()}
    per_scale_max_X = {k: np.concatenate(v, axis=0) for k, v in max_parts.items()}
    return y, scale_order, per_scale_mean_X, per_scale_max_X


# ---------------------------------------------------------------------------
# Separability metrics
# ---------------------------------------------------------------------------

def fisher_criterion(X: np.ndarray, y: np.ndarray) -> float:
    """Multi-feature Fisher's criterion: ratio of between-class to within-class scatter.

    Uses the trace ratio ``tr(S_B) / tr(S_W)``.  Higher → more linearly separable.
    """
    classes = np.unique(y)
    if len(classes) < 2:
        return float("nan")
    overall_mean = X.mean(axis=0)
    S_B = np.zeros((X.shape[1], X.shape[1]))
    S_W = np.zeros((X.shape[1], X.shape[1]))
    for cls in classes:
        X_cls = X[y == cls]
        n_cls = X_cls.shape[0]
        mean_cls = X_cls.mean(axis=0)
        diff = (mean_cls - overall_mean).reshape(-1, 1)
        S_B += n_cls * (diff @ diff.T)
        centred = X_cls - mean_cls
        S_W += centred.T @ centred
    tr_sw = np.trace(S_W)
    if tr_sw < 1e-12:
        return float("inf")
    return float(np.trace(S_B) / tr_sw)


def class_mean_cosine_similarity(X: np.ndarray, y: np.ndarray) -> float:
    """Cosine similarity between class mean vectors (−1 to 1; lower = more separable)."""
    classes = np.unique(y)
    if len(classes) != 2:
        return float("nan")
    m0 = X[y == classes[0]].mean(axis=0)
    m1 = X[y == classes[1]].mean(axis=0)
    n0, n1 = np.linalg.norm(m0), np.linalg.norm(m1)
    if n0 < 1e-12 or n1 < 1e-12:
        return float("nan")
    return float(np.dot(m0, m1) / (n0 * n1))


# ---------------------------------------------------------------------------
# Cross-validated classification
# ---------------------------------------------------------------------------

def _cv_metrics(pipeline, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
    """Run stratified k-fold CV and return aggregated metrics."""
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise RuntimeError("Need at least 2 classes.")

    # Need at least n_splits samples per class for stratified splitting
    min_count = counts.min()
    effective_splits = min(n_splits, int(min_count))
    if effective_splits < 2:
        return {"error": f"Too few samples (min class count={min_count}) for CV"}

    skf = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=42)
    aucs, baccs, reports = [], [], []
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(X_tr, y_tr)
            y_pred = pipeline.predict(X_te)
            try:
                if hasattr(pipeline, "predict_proba"):
                    proba = pipeline.predict_proba(X_te)[:, 1]
                else:
                    proba = pipeline.decision_function(X_te)
                auc = roc_auc_score(y_te, proba)
            except Exception:
                auc = float("nan")
        baccs.append(balanced_accuracy_score(y_te, y_pred))
        aucs.append(auc)
        reports.append(classification_report(y_te, y_pred, output_dict=True, zero_division=0))

    # Macro-average the per-fold classification reports
    all_keys = [k for k in reports[0] if isinstance(reports[0][k], dict)]
    avg_report: dict = {}
    for k in all_keys:
        avg_report[k] = {
            metric: float(np.mean([r[k][metric] for r in reports]))
            for metric in reports[0][k]
        }

    return {
        "n_cv_folds": effective_splits,
        "auc_roc_mean": float(np.nanmean(aucs)),
        "auc_roc_std": float(np.nanstd(aucs)),
        "balanced_accuracy_mean": float(np.mean(baccs)),
        "balanced_accuracy_std": float(np.std(baccs)),
        "classification_report_avg": avg_report,
    }


def run_classifiers(X: np.ndarray, y: np.ndarray, n_cv_splits: int = 3) -> dict:
    """Evaluate classifiers on ``(X, y)`` via stratified CV."""
    classifiers = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")),
        ]),
        "linear_svm": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(max_iter=1000, random_state=42, class_weight="balanced")),
        ]),
    }

    results: dict = {}
    for name, pipeline in classifiers.items():
        print(f"    [{name}] {n_cv_splits}-fold CV ...", end=" ", flush=True)
        results[name] = _cv_metrics(pipeline, X, y, n_splits=n_cv_splits)
        res = results[name]
        if "error" in res:
            print(res["error"])
        else:
            print(f"AUC {res['auc_roc_mean']:.3f} ± {res['auc_roc_std']:.3f}")

    return results


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def rank_features(X: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Rank feature channels by Cohen's d (vectorised, no model fitting)."""
    classes = np.unique(y)
    if len(classes) != 2:
        return {}

    X0, X1 = X[y == classes[0]], X[y == classes[1]]
    mean_diff = X1.mean(axis=0) - X0.mean(axis=0)
    pooled_std = np.sqrt((X0.std(axis=0) ** 2 + X1.std(axis=0) ** 2) / 2)
    pooled_std = np.where(pooled_std < 1e-12, 1e-12, pooled_std)
    cohens_d = np.abs(mean_diff / pooled_std)

    return {
        "cohens_d": cohens_d,
        "cohens_d_rank": np.argsort(cohens_d)[::-1].astype(int),
    }


def plot_top_feature_distributions(
    X: np.ndarray,
    y: np.ndarray,
    ranked_indices: np.ndarray,
    scores: np.ndarray,
    output_path: Path,
    title: str,
    top_n: int = 20,
    score_name: str = "Cohen's d",
) -> None:
    """Overlapping density histograms for the top-N most important feature channels.

    Each subplot shows the per-class distribution of one feature, with the
    feature index and its importance score in the subplot title.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping feature distribution plot")
        return

    n_plot = min(top_n, len(ranked_indices))
    selected = ranked_indices[:n_plot]

    ncols = min(5, n_plot)
    nrows = (n_plot + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.0, nrows * 2.4))
    axes_flat = np.array(axes).reshape(-1)

    colours = {0: "#4393c3", 1: "#d6604d"}
    class_labels = {0: "Neg (0)", 1: "Pos (1)"}

    for i, feat_idx in enumerate(selected):
        ax = axes_flat[i]
        score = float(scores[feat_idx])
        for cls in np.unique(y):
            vals = X[y == cls, feat_idx]
            ax.hist(vals, bins=30, density=True, alpha=0.55,
                    color=colours[int(cls)], label=class_labels[int(cls)])
        ax.set_title(f"f{feat_idx}  {score_name}={score:.2f}", fontsize=7.5)
        ax.tick_params(labelsize=6)
        ax.set_yticks([])
        if i == 0:
            ax.legend(fontsize=6.5, loc="upper right")

    for j in range(n_plot, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title, fontsize=9, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# PCA visualisation
# ---------------------------------------------------------------------------

def _save_pca_plot(
    X: np.ndarray,
    y: np.ndarray,
    output_path: Path,
    title: str = "PCA of bag features",
) -> None:
    """Save a 2-component PCA scatter plot coloured by class."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("  matplotlib / sklearn not available — skipping PCA plot")
        return

    X_scaled = StandardScaler().fit_transform(X)
    n_components = min(2, X_scaled.shape[1], X_scaled.shape[0])
    if n_components < 2:
        return

    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    colours = {0: "#4393c3", 1: "#d6604d"}
    labels_text = {0: "Negative (cls 0)", 1: "Positive (cls 1)"}
    for cls in np.unique(y):
        mask = y == cls
        ax.scatter(Z[mask, 0], Z[mask, 1], c=colours.get(int(cls), "grey"),
                   label=labels_text.get(int(cls), str(cls)), alpha=0.55, s=20, linewidths=0)
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}% var)")
    ax.set_title(title)
    ax.legend(markerscale=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyse(
    X: np.ndarray,
    y: np.ndarray,
    scale_order: list[str],
    per_scale_X: dict[str, np.ndarray],
    output_dir: Path,
    n_cv_splits: int = 3,
    label: str = "all_scales",
    top_n_features: int = 20,
    lightweight: bool = False,
) -> dict:
    """Run analysis on a feature matrix.

    When ``lightweight=True``, skips classifier CV and plot generation —
    only Fisher's criterion and Cohen's d are computed.  Used for per-scale
    sub-sections where the all-scales result already provides the full picture.
    """
    classes, counts = np.unique(y, return_counts=True)
    mode_str = "lightweight" if lightweight else "full"
    print(f"\n  -- {label}  ({int(X.shape[0])} bags, {int(X.shape[1])} features) [{mode_str}]")

    feat_ranks = rank_features(X, y)  # fast: vectorised numpy only

    top_features_summary: dict = {}
    if feat_ranks:
        top_idx = feat_ranks["cohens_d_rank"][:top_n_features].tolist()
        top_scores = feat_ranks["cohens_d"][feat_ranks["cohens_d_rank"][:top_n_features]].tolist()
        top_features_summary["cohens_d"] = {
            "top_feature_indices": top_idx,
            "top_feature_scores": [round(s, 5) for s in top_scores],
        }

    fc = fisher_criterion(X, y)
    cs = class_mean_cosine_similarity(X, y)
    print(f"    Fisher criterion: {fc:.4f}  |  class-mean cosine similarity: {cs:.4f}")

    result: dict = {
        "label": label,
        "n_bags": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "class_distribution": {str(int(c)): int(n) for c, n in zip(classes, counts)},
        "fisher_criterion": fc,
        "class_mean_cosine_similarity": cs,
        "top_features": top_features_summary,
    }

    if not lightweight:
        result["classifiers"] = run_classifiers(X, y, n_cv_splits=n_cv_splits)

        print(f"    Saving PCA plot ...", end=" ", flush=True)
        _save_pca_plot(
            X, y,
            output_dir / f"pca_{label}.png",
            title=f"PCA — {label} ({int(X.shape[0])} bags)",
        )
        print("done")

        if feat_ranks:
            print(f"    Saving feature distribution plot ...", end=" ", flush=True)
            plot_top_feature_distributions(
                X, y,
                ranked_indices=feat_ranks["cohens_d_rank"],
                scores=feat_ranks["cohens_d"],
                output_path=output_dir / f"feat_dist_{label}.png",
                title=f"Top features by Cohen's d — {label}",
                top_n=top_n_features,
                score_name="d",
            )
            print("done")

    return result


def run_full_analysis(
    y: np.ndarray,
    scale_order: list[str],
    per_scale_mean_X: dict[str, np.ndarray],
    per_scale_max_X: dict[str, np.ndarray],
    output_dir: Path,
    n_cv_splits: int = 5,
    top_n_features: int = 20,
) -> dict:
    """Analyse all-scales combined and each scale individually, for each pooling strategy."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build per-scale mean+max concatenation
    per_scale_meanmax_X = {
        k: np.concatenate([per_scale_mean_X[k], per_scale_max_X[k]], axis=1)
        for k in scale_order
    }

    pooling_strategies = {
        "mean": per_scale_mean_X,
        "max": per_scale_max_X,
        "mean_max": per_scale_meanmax_X,
    }

    all_results: dict = {}

    n_strategies = len(pooling_strategies)
    for i, (pool_name, per_scale_X) in enumerate(pooling_strategies.items(), 1):
        print(f"\n{'='*60}")
        print(f"  Pooling: {pool_name}  ({i}/{n_strategies})")
        print(f"{'='*60}")
        pool_results: dict = {}
        X_all = np.concatenate([per_scale_X[k] for k in scale_order], axis=1)

        # Full analysis for the combined all-scales result
        pool_results["all_scales"] = analyse(
            X_all, y, scale_order, per_scale_X, output_dir, n_cv_splits,
            label=f"{pool_name}_all_scales", top_n_features=top_n_features,
            lightweight=False,
        )

        # Lightweight per-scale breakdown (Fisher + Cohen's d only, no CV or plots).
        # Skipped for mean_max since mean and max are already broken down individually.
        if len(scale_order) > 1 and pool_name != "mean_max":
            for scale_key in scale_order:
                X_s = per_scale_X[scale_key]
                safe_key = scale_key.replace("/", "_").replace(" ", "_")
                pool_results[f"scale_{safe_key}"] = analyse(
                    X_s, y, [scale_key], {scale_key: X_s}, output_dir,
                    n_cv_splits, label=f"{pool_name}_scale_{safe_key}",
                    top_n_features=top_n_features, lightweight=True,
                )

        all_results[pool_name] = pool_results

    return all_results


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _print_section(section_key: str, section: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  {section_key}  ({section.get('n_bags', '?')} bags, "
          f"{section.get('n_features', '?')} features)")
    print(f"{'='*60}")
    dist = section.get("class_distribution", {})
    print(f"  Class distribution: {dist}")
    fc = section.get("fisher_criterion", float("nan"))
    cs = section.get("class_mean_cosine_similarity", float("nan"))
    print(f"  Fisher criterion (higher = more separable): {fc:.4f}")
    print(f"  Class-mean cosine similarity (lower = more separable): {cs:.4f}")

    if "classifiers" not in section:
        return  # lightweight section — no CV results

    print()
    for clf_name, clf_res in section["classifiers"].items():
        if "error" in clf_res:
            print(f"  [{clf_name}]  {clf_res['error']}")
            continue
        n_folds = clf_res.get("n_cv_folds", "?")
        auc_m = clf_res.get("auc_roc_mean", float("nan"))
        auc_s = clf_res.get("auc_roc_std", float("nan"))
        bacc_m = clf_res.get("balanced_accuracy_mean", float("nan"))
        bacc_s = clf_res.get("balanced_accuracy_std", float("nan"))
        print(f"  [{clf_name}]  {n_folds}-fold CV")
        print(f"    ROC-AUC:           {auc_m:.4f} ± {auc_s:.4f}")
        print(f"    Balanced accuracy: {bacc_m:.4f} ± {bacc_s:.4f}")


def _print_results(results: dict) -> None:
    for pool_name, pool_results in results.items():
        print(f"\n{'#'*60}")
        print(f"  Pooling strategy: {pool_name}")
        print(f"{'#'*60}")
        for section_key, section in pool_results.items():
            _print_section(section_key, section)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse saved MIL bag features for class separability."
    )
    parser.add_argument(
        "--input_dir", type=Path, default=Path("results_debug"),
        help="Root directory to search for saved feature files.",
    )
    parser.add_argument(
        "--pattern", type=str, default="**/*encoded_instances.pt",
        help="Glob pattern (relative to input_dir) to match saved feature files.",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("results/feature_analysis"),
        help="Directory to save analysis outputs.",
    )
    parser.add_argument(
        "--n_cv_splits", type=int, default=5,
        help="Number of stratified CV folds (default 5).",
    )
    parser.add_argument(
        "--top_n_features", type=int, default=20,
        help="Number of top feature channels to visualise per ranking method (default 20).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.input_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(
            f"No files found in {args.input_dir} matching pattern '{args.pattern}'."
        )

    print(f"Found {len(files)} file(s):")
    for f in files:
        print(f"  {f}")

    y, scale_order, per_scale_mean_X, per_scale_max_X = load_dataset(files)
    classes, counts = np.unique(y, return_counts=True)
    n_bags = per_scale_mean_X[scale_order[0]].shape[0]
    feat_dim = sum(per_scale_mean_X[k].shape[1] for k in scale_order)
    print(f"\nLoaded: {n_bags} bags  |  features per bag (mean-pool, all scales): {feat_dim}")
    print(f"Class distribution: { {str(int(c)): int(n) for c, n in zip(classes, counts)} }")
    print(f"Scales: {scale_order}")

    results = run_full_analysis(
        y, scale_order, per_scale_mean_X, per_scale_max_X,
        args.output_dir, args.n_cv_splits, args.top_n_features,
    )

    _print_results(results)

    out_json = args.output_dir / "feature_separability_analysis.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics → {out_json}")
    print(f"Saved PCA plots → {args.output_dir}/pca_*.png")
    print(f"Saved feature distribution plots → {args.output_dir}/feat_dist_*.png")


if __name__ == "__main__":
    main()
