#!/usr/bin/env python3
"""
Multilevel genotype/group-separation analysis.

Combines:
  1) Random-Forest label-randomization ("bootstrap") classification test
  2) PERMANOVA on a Jaccard genomic distance matrix
  3) Mantel test between genomic distance and group-distance matrices

Input layout supported by this script
-------------------------------------
A TSV/CSV matrix in ./input/<project_folder>/ with:
  * column 1: hierarchical label + sample ID separated by '|'
              e.g. L4|L3|L2|L1|GenomeID
  * columns 2..N: binary SNP/features (0/1)

Hierarchy levels are numbered from the bottom upward:
  --level 1     -> L1 (lowest / most specific label)
  --level 2     -> L2
  --level 3     -> L3
  --level 4     -> L4 (highest label in the supplied example)
  --level 1,2   -> composite L2|L1 grouping

The last pipe-delimited field is treated as the sample/genome identifier and is
not part of the hierarchy.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import UndefinedMetricWarning

__version__ = "3.0"
DEFAULT_REPLICATES = 100
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Random-Forest bootstrap/random-label test, PERMANOVA, and Mantel "
            "test for hierarchically labelled binary genomic matrices."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "project_folder",
        help=(
            "Project subfolder. Input is read from ./input/PROJECT_FOLDER and "
            "results are written to ./output/PROJECT_FOLDER."
        ),
    )
    parser.add_argument("-i", "--input_folder", default="input")
    parser.add_argument("-o", "--output_folder", default="output")
    parser.add_argument(
        "-m", "-f", "--input_matrix_file", "--input_file",
        default=None,
        help=(
            "Optional CSV/TSV filename inside the project input folder. If omitted, "
            "the script requires exactly one *.csv or *.tsv file there."
        ),
    )
    parser.add_argument(
        "-n", "--replicates",
        type=int,
        default=DEFAULT_REPLICATES,
        help=f"Number of random/permutation replicates (default: {DEFAULT_REPLICATES}).",
    )
    parser.add_argument(
        "-L", "--level",
        default="1,2",
        help=(
            "Hierarchy level(s), counted from bottom (1) to top (n).\n"
            "Examples: --level 1 ; --level 2 ; --level 1,2 (default: 1,2)."
        ),
    )
    parser.add_argument(
        "--algorithm",
        "-a",
        default="ALL",
        choices=["ALL", "RF", "BOOTSTRAP", "PERMANOVA", "MANTEL"],
        type=str.upper,
        help="Analysis to run (default: ALL). RF and BOOTSTRAP are synonyms.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "-t", "--trees",
        type=int,
        default=100,
        help="Number of trees in the Random Forest (default: 100).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not attempt to open the matplotlib plot window; PNG is still saved.",
    )
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    return parser.parse_args()


def locate_input(input_folder: str, output_folder: str, project_folder: str, input_file: str | None) -> tuple[Path, Path]:
    base_dir = Path(__file__).resolve().parent
    input_dir = base_dir / input_folder / project_folder
    output_dir = base_dir / output_folder / project_folder

    if not input_dir.is_dir():
        raise SystemExit(f"[ERROR] Input folder does not exist: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_file:
        path = input_dir / input_file
        if not path.is_file():
            raise SystemExit(f"[ERROR] Input file does not exist: {path}")
        return path, output_dir

    candidates = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".csv", ".tsv"}
    )
    if len(candidates) == 0:
        raise SystemExit(f"[ERROR] No .csv or .tsv matrix found in {input_dir}")
    if len(candidates) > 1:
        names = "\n  ".join(p.name for p in candidates)
        raise SystemExit(
            f"[ERROR] More than one CSV/TSV file found in {input_dir}. "
            f"Use --input_matrix_file to choose one:\n  {names}"
        )
    return candidates[0], output_dir


def detect_delimiter(path: Path) -> str:
    if path.suffix.lower() == ".tsv":
        return "\t"
    if path.suffix.lower() == ".csv":
        return ","
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        sample = handle.read(8192)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
    except csv.Error:
        return "\t" if sample.count("\t") > sample.count(",") else ","


def parse_levels(level_arg: str, n_levels: int) -> list[int]:
    try:
        requested = [int(x.strip()) for x in level_arg.split(",") if x.strip()]
    except ValueError as exc:
        raise SystemExit("[ERROR] --level must contain integer levels such as 1 or 1,2.") from exc

    if not requested:
        raise SystemExit("[ERROR] --level cannot be empty.")
    if len(set(requested)) != len(requested):
        raise SystemExit("[ERROR] --level contains duplicate levels.")
    bad = [x for x in requested if x < 1 or x > n_levels]
    if bad:
        raise SystemExit(
            f"[ERROR] Requested level(s) {bad} outside valid range 1..{n_levels}."
        )
    return sorted(requested, reverse=True)  # top-to-bottom display in composite label


def load_matrix(path: Path, level_arg: str):
    sep = detect_delimiter(path)
    df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
    if df.shape[1] < 2:
        raise SystemExit("[ERROR] Matrix must contain a label/ID column plus feature columns.")

    raw = df.iloc[:, 0].astype(str).str.strip()
    split_rows = [x.split("|") for x in raw]
    lengths = sorted(set(len(x) for x in split_rows))
    if len(lengths) != 1 or lengths[0] < 2:
        raise SystemExit(
            "[ERROR] First column must consistently contain pipe-delimited hierarchy "
            "labels followed by the sample ID, e.g. L4|L3|L2|L1|GenomeID."
        )

    n_fields = lengths[0]
    n_levels = n_fields - 1
    requested = parse_levels(level_arg, n_levels)

    hierarchy = [row[:-1] for row in split_rows]
    sample_ids = np.array([row[-1].strip() for row in split_rows], dtype=object)
    if any(x == "" for x in sample_ids):
        raise SystemExit("[ERROR] Empty sample/genome identifier found in the first column.")
    if len(set(sample_ids)) != len(sample_ids):
        # Distance methods only need unique IDs conceptually; make duplicate IDs explicit instead of failing.
        seen: dict[str, int] = {}
        fixed = []
        for x in sample_ids:
            seen[x] = seen.get(x, 0) + 1
            fixed.append(x if seen[x] == 1 else f"{x}__dup{seen[x]}")
        sample_ids = np.array(fixed, dtype=object)

    # Level 1 = last hierarchy component; level n = first component.
    selected_indices = [n_levels - lvl for lvl in requested]
    labels = np.array(
        ["|".join(row[i].strip() for i in selected_indices) for row in hierarchy],
        dtype=object,
    )
    valid = np.array([lbl != "" and all(part.strip() for part in lbl.split("|")) for lbl in labels])
    if not np.all(valid):
        print(f"[WARNING] Dropping {np.sum(~valid)} rows with empty selected hierarchy labels.")
        df = df.loc[valid].reset_index(drop=True)
        labels = labels[valid]
        sample_ids = sample_ids[valid]

    feature_df = df.iloc[:, 1:].copy()
    # Missing-like values are not accepted for distance tests because their biological meaning is ambiguous.
    bad_tokens = {"", "na", "nan", "nd", "non", "none", "."}
    lowered = feature_df.apply(lambda s: s.astype(str).str.strip().str.lower())
    if lowered.isin(bad_tokens).any().any():
        locations = np.argwhere(lowered.isin(bad_tokens).to_numpy())
        r, c = locations[0]
        raise SystemExit(
            f"[ERROR] Missing/undefined feature value at row {r+2}, column "
            f"'{feature_df.columns[c]}'. Remove/impute missing values before these tests."
        )

    try:
        X = feature_df.astype(float).to_numpy()
    except ValueError as exc:
        raise SystemExit(
            "[ERROR] Feature columns must be numeric binary values (0/1) for Jaccard analyses."
        ) from exc

    if not np.isin(X, [0.0, 1.0]).all():
        vals = np.unique(X[~np.isin(X, [0.0, 1.0])])[:10]
        raise SystemExit(
            f"[ERROR] Non-binary feature values detected ({vals.tolist()}). "
            "This script uses Jaccard distance and therefore expects 0/1 features."
        )

    if len(np.unique(labels)) < 2:
        raise SystemExit("[ERROR] Selected hierarchy level(s) produce fewer than two groups.")

    return sample_ids, labels, X.astype(np.uint8), n_levels, requested, sep


def make_cv(labels: np.ndarray, seed: int):
    counts = pd.Series(labels).value_counts()
    min_count = int(counts.min())
    n_splits = min(5, len(labels))
    if n_splits < 2:
        raise SystemExit("[ERROR] At least two samples are required for RF cross-validation.")
    if min_count < n_splits:
        rare = counts[counts < n_splits]
        print(
            f"[WARNING] {len(rare)} selected group(s) contain fewer than {n_splits} genomes "
            "(minimum = %d). Stratified folds cannot contain every class in every fold; "
            "RF accuracy should be interpreted cautiously." % min_count
        )
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed), n_splits


def rf_bootstrap(X: np.ndarray, labels: np.ndarray, replicates: int, seed: int, trees: int):
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    cv, folds = make_cv(labels, seed)
    clf = RandomForestClassifier(random_state=seed, n_estimators=trees, n_jobs=-1)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The least populated class in y has only")
        observed = float(cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=None).mean())
        rng = np.random.default_rng(seed)
        null = np.empty(replicates, dtype=float)
        for i in range(replicates):
            y_perm = rng.permutation(y)
            null[i] = cross_val_score(clf, X, y_perm, cv=cv, scoring="accuracy", n_jobs=None).mean()
    p = (np.count_nonzero(null >= observed) + 1.0) / (replicates + 1.0)
    return observed, null, p, folds


def jaccard_distance_matrix(X: np.ndarray) -> np.ndarray:
    # scipy's Jaccard operates naturally on boolean presence/absence vectors.
    D = squareform(pdist(X.astype(bool), metric="jaccard"))
    # pdist can return NaN when both vectors are all-zero in some scipy versions.
    return np.nan_to_num(D, nan=0.0)


def permanova_stat(D: np.ndarray, labels: np.ndarray) -> float:
    n = len(labels)
    unique, inverse = np.unique(labels, return_inverse=True)
    g = len(unique)
    if g < 2 or n <= g:
        return np.nan

    D2 = D * D
    ss_total = D2.sum() / (2.0 * n)
    ss_within = 0.0
    for k in range(g):
        idx = np.flatnonzero(inverse == k)
        nk = len(idx)
        if nk > 0:
            block = D2[np.ix_(idx, idx)]
            ss_within += block.sum() / (2.0 * nk)
    ss_between = ss_total - ss_within
    df_between = g - 1
    df_within = n - g
    if ss_within <= 0 or df_within <= 0:
        return np.inf if ss_between > 0 else np.nan
    return (ss_between / df_between) / (ss_within / df_within)


def run_permanova(D: np.ndarray, labels: np.ndarray, replicates: int, seed: int):
    observed = float(permanova_stat(D, labels))
    rng = np.random.default_rng(seed)
    null = np.empty(replicates, dtype=float)
    for i in range(replicates):
        null[i] = permanova_stat(D, rng.permutation(labels))
    p = (np.count_nonzero(null >= observed) + 1.0) / (replicates + 1.0)
    return observed, null, p


def mantel_stat(D: np.ndarray, labels: np.ndarray) -> float:
    iu = np.triu_indices_from(D, k=1)
    genomic = D[iu]
    env = (labels[:, None] != labels[None, :]).astype(float)[iu]
    if np.std(genomic) == 0 or np.std(env) == 0:
        return np.nan
    return float(pearsonr(genomic, env).statistic)


def run_mantel(D: np.ndarray, labels: np.ndarray, replicates: int, seed: int):
    observed = mantel_stat(D, labels)
    rng = np.random.default_rng(seed)
    null = np.empty(replicates, dtype=float)
    for i in range(replicates):
        null[i] = mantel_stat(D, rng.permutation(labels))
    finite = np.isfinite(null)
    if not np.isfinite(observed) or not finite.any():
        return observed, null, np.nan
    # Match scikit-bio's default two-sided Mantel alternative.
    p = (np.count_nonzero(np.abs(null[finite]) >= abs(observed)) + 1.0) / (finite.sum() + 1.0)
    return observed, null, p


def save_distribution(
    path: Path,
    values: np.ndarray,
    value_name: str,
    observed: float,
):
    """Save the observed statistic followed by the randomized null replicates."""
    report = pd.DataFrame({
        "replicate": np.concatenate(([0], np.arange(1, len(values) + 1))),
        "grouping": ["original"] + ["randomized"] * len(values),
        value_name: np.concatenate(([observed], values)),
    })
    report.to_csv(path, sep="\t", index=False)


def plot_rf(null: np.ndarray, observed: float, output_png: Path, show: bool):
    """Always try to save PNG; show a window only when an interactive backend is available."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib is not installed; RF distribution PNG cannot be created.")
        return

    fig, ax = plt.subplots()
    ax.hist(null, bins=30, alpha=0.7, label="Null (random labels)")
    ax.axvline(observed, linestyle="--", label="Observed score")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Frequency")
    ax.set_title("Bootstrap / Random-label Accuracy Distribution\n(Multilevel grouping)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    print(f"RF plot saved: {output_png}")

    if show:
        backend = str(matplotlib.get_backend()).lower()
    
        noninteractive_backends = {
            "agg", "pdf", "svg", "ps", "cairo", "template"
        }
        noninteractive = backend in noninteractive_backends
    
        headless = (
            not os.environ.get("DISPLAY")
            and not sys.platform.startswith(("win", "darwin"))
        )
    
        if noninteractive or headless:
            print(
                f"[INFO] Matplotlib backend '{matplotlib.get_backend()}' "
                "is non-interactive/headless; PNG retained."
            )
        else:
            plt.show()


def main() -> int:
    args = parse_args()
    if args.replicates < 1:
        raise SystemExit("[ERROR] --replicates must be >= 1.")
    if args.trees < 1:
        raise SystemExit("[ERROR] --trees must be >= 1.")

    input_path, output_dir = locate_input(args.input_folder, args.output_folder, args.project_folder, args.input_matrix_file)
    sample_ids, labels, X, n_levels, levels, sep = load_matrix(input_path, args.level)

    level_tag = "-".join(str(x) for x in sorted(levels))
    base = input_path.stem
    counts = pd.Series(labels).value_counts().sort_index()

    print(f"Input: {input_path}")
    print(f"Samples: {len(labels)}; features: {X.shape[1]}; hierarchy levels detected: {n_levels}")
    print(f"Selected level(s), bottom=1: {','.join(map(str, sorted(levels)))}")
    print(f"Composite groups: {len(counts)}")
    print("Group sizes:")
    for label, count in counts.items():
        print(f"  {label}: {count}")

    # Save exact grouping used by all tests.
    pd.DataFrame({"sample_id": sample_ids, "group": labels}).to_csv(
        output_dir / f"{base}_level_{level_tag}_grouping.tsv", sep="\t", index=False
    )

    alg = "RF" if args.algorithm == "BOOTSTRAP" else args.algorithm
    run_rf = alg in {"ALL", "RF"}
    run_perm = alg in {"ALL", "PERMANOVA"}
    run_man = alg in {"ALL", "MANTEL"}

    summary = []

    if run_rf:
        print("\n=== RANDOM-FOREST BOOTSTRAP / RANDOM-LABEL TEST ===")
        try:
            observed, null, p, folds = rf_bootstrap(X, labels, args.replicates, args.seed, args.trees)
            print(f"Observed {folds}-fold CV accuracy: {observed:.6f}")
            print(f"Random-label p-value: {p:.6g}")
            
            save_distribution(
                output_dir / f"{base}_level_{level_tag}_rf_null.tsv", null, "accuracy", observed
            )
            plot_rf(
                null,
                observed,
                output_dir / f"{base}_level_{level_tag}_bootstrap_accuracy_distribution.png",
                show=not args.no_show,
            )
            summary.append(("RF", "CV_accuracy", observed, p, args.replicates))
        except SystemExit as exc:
            if alg == "RF":
                raise
            print(str(exc))
            print("[WARNING] RF test skipped; PERMANOVA/Mantel can still be run.")

    D = None
    if run_perm or run_man:
        print("\nComputing Jaccard distance matrix...")
        D = jaccard_distance_matrix(X)

    if run_perm:
        print("\n=== PERMANOVA ===")
        observed, null, p = run_permanova(D, labels, args.replicates, args.seed)
        print(f"Pseudo-F: {observed:.6f}")
        print(f"Permutation p-value: {p:.6g}")
        save_distribution(
            output_dir / f"{base}_level_{level_tag}_permanova_null.tsv",
            null,
            "pseudo_F",
            observed,
        )
        summary.append(("PERMANOVA", "pseudo_F", observed, p, args.replicates))

    if run_man:
        print("\n=== MANTEL ===")
        observed, null, p = run_mantel(D, labels, args.replicates, args.seed)
        print(f"Mantel r: {observed:.6f}")
        print(f"Two-sided permutation p-value: {p:.6g}")
        save_distribution(
            output_dir / f"{base}_level_{level_tag}_mantel_null.tsv",
            null,
            "mantel_r",
            observed,
        )
        summary.append(("MANTEL", "pearson_r", observed, p, args.replicates))

    summary_path = output_dir / f"{base}_level_{level_tag}_summary.tsv"
    pd.DataFrame(
        summary,
        columns=["analysis", "statistic", "observed", "p_value", "replicates"],
    ).to_csv(summary_path, sep="\t", index=False)
    print(f"\nSummary saved: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
