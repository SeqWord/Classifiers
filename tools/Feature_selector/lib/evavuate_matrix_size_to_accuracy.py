#!/usr/bin/env python3
"""
Evaluate how the number of top-ranked genomic features affects Random-Forest
classification accuracy.

This module is intended to be imported by select_features.py from ../lib.
"""

from __future__ import annotations

import os
import sys
from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder


def _extract_labels(sample_fields: Sequence[str]) -> np.ndarray:
    """
    Convert the first matrix column to ecological class labels.

    Supported formats include::

        Habitat|Subhabitat|GenomeID
        HDBSCAN_cluster|Habitat|Subhabitat|GenomeID

    The final field is always treated as the sample/genome identifier.  When
    all rows contain at least four pipe-delimited fields and the first field is
    an integer (including -1, as commonly used for HDBSCAN noise), that leading
    field is treated as an HDBSCAN/unsupervised-cluster label and excluded from
    the ecological class.  The remaining hierarchy fields form the RF target
    class.
    """
    parsed: List[List[str]] = []

    for value in sample_fields:
        text = str(value).strip()
        if not text:
            raise ValueError("Empty sample/group label found in the first matrix column.")
        parts = [part.strip() for part in text.split("|")]
        if any(part == "" for part in parts):
            raise ValueError(f"Empty component in sample/group label: {text}")
        parsed.append(parts)

    # Detect the matrix format used by the HDBSCAN pipeline.  Requiring this
    # pattern in every row avoids accidentally discarding a genuine ecological
    # hierarchy level in ordinary Habitat|Subhabitat|GenomeID matrices.
    def _is_integer(text: str) -> bool:
        try:
            int(text)
            return True
        except ValueError:
            return False

    has_leading_cluster = bool(parsed) and all(
        len(parts) >= 4 and _is_integer(parts[0])
        for parts in parsed
    )

    labels: List[str] = []
    no_hierarchy = 0

    for parts in parsed:
        if len(parts) == 1:
            # Allow matrices where the first column already contains class labels.
            labels.append(parts[0])
            no_hierarchy += 1
            continue

        # Exclude the last field (sample ID).  In HDBSCAN-formatted matrices,
        # also exclude the first field (unsupervised cluster ID).
        hierarchy = parts[1:-1] if has_leading_cluster else parts[:-1]
        if not hierarchy:
            raise ValueError(
                "Cannot derive an ecological class label from first-column value: "
                + "|".join(parts)
            )
        labels.append("|".join(hierarchy))

    if has_leading_cluster:
        print(
            "INFO: detected a leading integer HDBSCAN/cluster field in the first "
            "matrix column; it is excluded from RF class labels."
        )
    if no_hierarchy:
        print(
            "INFO: first matrix column is not pipe-delimited for "
            f"{no_hierarchy} row(s); the complete field was used as the class label."
        )

    labels_array = np.asarray(labels, dtype=object)
    unique, counts = np.unique(labels_array, return_counts=True)
    if len(unique) < 2:
        raise ValueError("RF accuracy evaluation requires at least two classes/groups.")

    print(
        f"INFO: RF target classes detected: {len(unique)}; "
        f"class size range: {int(counts.min())}..{int(counts.max())}."
    )
    return labels_array


def _encode_feature_columns(values: np.ndarray) -> np.ndarray:
    """Encode each SNP/feature column independently as integer categorical states."""
    if values.ndim != 2:
        raise ValueError("Feature matrix must be two-dimensional.")

    encoded = np.empty(values.shape, dtype=np.int32)
    missing_tokens = {"", "na", "nan", "none", "null", "."}

    for column in range(values.shape[1]):
        raw = np.asarray([str(v).strip().upper() for v in values[:, column]], dtype=object)
        if any(v.lower() in missing_tokens for v in raw):
            raise ValueError(
                f"Missing/undefined state detected in evaluated feature column {column + 1}."
            )
        _, inverse = np.unique(raw, return_inverse=True)
        encoded[:, column] = inverse

    return encoded


def _make_cv(labels: np.ndarray, seed: int) -> StratifiedKFold:
    counts = pd.Series(labels).value_counts()
    min_count = int(counts.min())
    n_splits = min(5, min_count)
    if n_splits < 2:
        rare = counts[counts < 2]
        raise ValueError(
            "RF cross-validation requires at least two genomes in every class. "
            f"Classes with fewer than two genomes: {', '.join(map(str, rare.index.tolist()))}"
        )
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def _plot_accuracy(feature_counts: Sequence[int], accuracies: Sequence[float], output_png: str) -> None:
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib is not installed; PNG graph cannot be created.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(feature_counts))
    bars = ax.bar(positions, accuracies)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(value) for value in feature_counts])
    ax.set_xlabel("Number of top-ranked features")
    ax.set_ylabel("Cross-validated RF accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Random-Forest accuracy versus feature-matrix size")

    for bar, accuracy in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(accuracy + 0.02, 0.98),
            f"{accuracy:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    print(f"RF accuracy graph saved to: {output_png}")

    backend = str(matplotlib.get_backend()).lower()
    noninteractive_backends = {"agg", "pdf", "svg", "ps", "cairo", "template"}
    headless = (
        not os.environ.get("DISPLAY")
        and not sys.platform.startswith(("win", "darwin"))
    )

    if backend in noninteractive_backends or headless:
        print(
            f"INFO: Matplotlib backend '{matplotlib.get_backend()}' is non-interactive/headless; "
            "PNG was saved but no plot window can be opened."
        )
    else:
        try:
            plt.show()
        except Exception as exc:
            print(f"WARNING: could not open Matplotlib window: {exc}")

    plt.close(fig)


def main(
    matrix_rows: Sequence[Sequence[str]],
    ranked_feature_ids: Sequence[str],
    feature_counts: Sequence[int],
    output_dir: str,
    output_prefix: str = "matrix",
    seed: int = 42,
    trees: int = 100,
) -> pd.DataFrame:
    """
    Calculate Random-Forest accuracy for increasing numbers of top-ranked features.

    Parameters
    ----------
    matrix_rows
        Original genome x feature matrix including a header row.  The first
        column contains a hierarchical group label followed by sample ID, or a
        class label directly.
    ranked_feature_ids
        Feature IDs ordered from best to worst after statistical, neighbourhood,
        and redundancy filtering.
    feature_counts
        Numbers of top-ranked features to evaluate.
    output_dir
        Directory for the PNG and TSV output files.
    output_prefix
        Prefix derived from the input matrix filename.
    seed
        Random seed used for cross-validation and Random Forest.
    trees
        Number of trees in the Random Forest.
    """
    if len(matrix_rows) < 3:
        raise ValueError("Matrix must contain a header and at least two genome rows.")
    if trees < 1:
        raise ValueError("trees must be >= 1.")

    header = [str(value).strip() for value in matrix_rows[0]]
    if len(header) < 2:
        raise ValueError("Matrix must contain a label/ID column and at least one feature column.")

    expected_columns = len(header)
    for row_number, row in enumerate(matrix_rows[1:], start=2):
        if len(row) != expected_columns:
            raise ValueError(
                f"Matrix row {row_number} contains {len(row)} columns; expected {expected_columns}."
            )

    feature_to_column = {}
    for column, feature_id in enumerate(header[1:], start=1):
        if not feature_id:
            raise ValueError(f"Empty feature ID in matrix column {column + 1}.")
        if feature_id in feature_to_column:
            raise ValueError(f"Duplicate feature ID in matrix header: {feature_id}")
        feature_to_column[feature_id] = column

    ranked_available = [feature for feature in ranked_feature_ids if feature in feature_to_column]
    if not ranked_available:
        raise ValueError("None of the ranked features is present in the SNP matrix.")

    counts = sorted(dict.fromkeys(int(value) for value in feature_counts))
    if not counts:
        raise ValueError("No feature counts were supplied for RF accuracy evaluation.")
    if counts[0] < 1 or counts[-1] > len(ranked_available):
        raise ValueError(
            f"Feature counts must fall within 1..{len(ranked_available)} for the supplied matrix."
        )

    labels = _extract_labels([row[0] for row in matrix_rows[1:]])
    y = LabelEncoder().fit_transform(labels)
    cv = _make_cv(labels, seed)

    clf = RandomForestClassifier(
        n_estimators=trees,
        random_state=seed,
        n_jobs=-1,
    )

    accuracies: List[float] = []
    standard_deviations: List[float] = []

    print("\n=== RF ACCURACY BY FEATURE-MATRIX SIZE ===")
    for count in counts:
        selected_ids = ranked_available[:count]
        columns = [feature_to_column[feature_id] for feature_id in selected_ids]
        raw_values = np.asarray(
            [[row[column] for column in columns] for row in matrix_rows[1:]],
            dtype=object,
        )
        X = _encode_feature_columns(raw_values)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=None)
        mean_accuracy = float(np.mean(scores))
        sd_accuracy = float(np.std(scores, ddof=0))
        accuracies.append(mean_accuracy)
        standard_deviations.append(sd_accuracy)
        print(
            f"Top {count:>6} features: accuracy = {mean_accuracy:.6f} "
            f"(SD across folds = {sd_accuracy:.6f})"
        )

    os.makedirs(output_dir, exist_ok=True)
    output_png = os.path.join(output_dir, f"{output_prefix}_rf_accuracy_by_feature_count.png")
    output_tsv = os.path.join(output_dir, f"{output_prefix}_rf_accuracy_by_feature_count.tsv")

    results = pd.DataFrame(
        {
            "feature_count": counts,
            "mean_cv_accuracy": accuracies,
            "sd_cv_accuracy": standard_deviations,
        }
    )
    results.to_csv(output_tsv, sep="\t", index=False)
    print(f"RF accuracy values saved to: {output_tsv}")

    _plot_accuracy(counts, accuracies, output_png)
    return results


if __name__ == "__main__":
    raise SystemExit(
        "This module is designed to be called from select_features.py; "
        "use --evaluate in select_features.py."
    )
