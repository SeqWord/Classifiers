#!/usr/bin/env python3
"""
phylo_ecology_overlap.py

Quantify overlap between phylogenetic structure and ecological categories encoded
in Newick tip labels:

    level1|level2|genome_name

Primary statistic:
    MPD = mean pairwise patristic distance among genomes assigned to the same
    ecological category.

A permutation test randomly shuffles complete ecological labels among tree tips
while preserving category sizes. Strong ecological/phylogenetic overlap produces
an observed MPD smaller than expected under the null.

Reported measures:
    observed_MPD
    null_mean_MPD
    SES = (null_mean_MPD - observed_MPD) / null_sd_MPD
          Positive SES means stronger phylogenetic clustering by ecology.
    P = (1 + number(null_MPD <= observed_MPD)) / (permutations + 1)
        One-sided empirical P value.
    between_category_MPD
    overlap_effect = (between_category_MPD - observed_MPD) /
                     between_category_MPD
        0 means same-category pairs are no closer than different-category pairs;
        positive values indicate ecological clustering on the phylogeny.

Dependencies:
    biopython, numpy

Example:
    python phylo_ecology_overlap.py MyProject --tree core_snps_filetered.nwk --level 1,2
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from Bio import Phylo


def parse_args():
    p = argparse.ArgumentParser(
        description="Test overlap between phylogenetic and ecological clustering."
    )
    p.add_argument("project", help="Project subfolder within input and output folders.")
    p.add_argument("-i", "--input_folder", default="input",
                   help="Input base folder (default: input).")
    p.add_argument("-o", "--output_folder", default="output",
                   help="Output base folder (default: output).")
    p.add_argument("-t", "--tree", required=True,
                   help="Newick filename within input_folder/project.")
    p.add_argument("-L", "-l", "--level", required=True,
                   help="Comma-separated ecological levels, e.g. 1, 2, or 1,2.")
    p.add_argument("-n", "--permutations", type=int, default=10000,
                   help="Number of label permutations (default: 10000).")
    p.add_argument("-s", "--seed", type=int, default=12345,
                   help="Random seed (default: 12345).")
    p.add_argument("-d", "--delimiter", type=str, default="|",
                   help="Delimiter between ecological levels in titles (default: '|').")
    p.add_argument("-f", "--output_file", default=None,
                   help="Optional output TSV filename stem (one file per level).")
    return p.parse_args()


def parse_levels(value):
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part or not part.isdecimal() for part in parts):
        raise ValueError("--level must be comma-separated positive integers (e.g. 1,2).")
    levels = [int(part) for part in parts]
    if any(level < 1 for level in levels):
        raise ValueError("--level values must be at least 1.")
    return list(dict.fromkeys(levels))


def read_tree_and_labels(tree, tips, level, delimiter="|"):
    labels = []
    genomes = []
    for tip in tips:
        fields = [field.strip() for field in tip.name.split(delimiter)]
        if not all(fields):
            raise ValueError(f"Malformed ecological/genome label: {tip.name!r}")
        labels.append("|".join(fields[:level]))
        genomes.append(fields[-1])

    counts = Counter(labels)
    singleton_categories = [k for k, v in counts.items() if v < 2]
    if singleton_categories:
        print("WARNING: singleton categories cannot contribute within-category "
              "pairs: " + ", ".join(singleton_categories), file=sys.stderr)
    if not any(v >= 2 for v in counts.values()):
        raise ValueError("No ecological category contains at least two genomes.")
    return np.asarray(labels, dtype=object), genomes, counts


def load_tree(tree_path):
    tree = Phylo.read(str(tree_path), "newick")
    tips = tree.get_terminals()
    if len(tips) < 3:
        raise ValueError("The tree must contain at least three terminal nodes.")
    if any(not tip.name for tip in tips):
        raise ValueError("A terminal node has no name.")
    widths = {len(tip.name.split("|")) for tip in tips}
    if len(widths) != 1:
        raise ValueError("Inconsistent numbers of '|' separated fields in tip labels.")
    field_count = widths.pop()
    if field_count < 2:
        raise ValueError("Tree labels must contain ecological fields and a genome name.")
    if any(not all(part.strip() for part in tip.name.split("|")) for tip in tips):
        raise ValueError("Empty ecological or genome field in a tip label.")
    return tree, tips, field_count - 1


def patristic_matrix(tree, tips):
    n = len(tips)
    matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            d = float(tree.distance(tips[i], tips[j]))
            matrix[i, j] = d
            matrix[j, i] = d

    return matrix


def within_mpd(distances, labels):
    """Mean distance across all same-category unordered pairs."""
    n = len(labels)
    i, j = np.triu_indices(n, 1)
    same = labels[i] == labels[j]

    if not np.any(same):
        raise ValueError("No within-category genome pairs are available.")

    return float(np.mean(distances[i[same], j[same]]))


def between_mpd(distances, labels):
    n = len(labels)
    i, j = np.triu_indices(n, 1)
    different = labels[i] != labels[j]

    if not np.any(different):
        raise ValueError("Only one ecological category is present.")

    return float(np.mean(distances[i[different], j[different]]))


def category_statistics(distances, labels):
    rows = []
    for category in sorted(set(labels)):
        idx = np.flatnonzero(labels == category)
        if len(idx) >= 2:
            sub = distances[np.ix_(idx, idx)]
            iu = np.triu_indices(len(idx), 1)
            mpd = float(np.mean(sub[iu]))
            pairs = len(idx) * (len(idx) - 1) // 2
        else:
            mpd = float("nan")
            pairs = 0
        rows.append((category, len(idx), pairs, mpd))
    return rows


def permutation_test(distances, labels, permutations, seed):
    rng = np.random.default_rng(seed)
    observed = within_mpd(distances, labels)

    null = np.empty(permutations, dtype=float)
    for k in range(permutations):
        permuted = rng.permutation(labels)
        null[k] = within_mpd(distances, permuted)

    null_mean = float(np.mean(null))
    null_sd = float(np.std(null, ddof=1)) if permutations > 1 else float("nan")

    # Lower MPD = stronger clustering than random.
    p_value = (1.0 + float(np.sum(null <= observed))) / (permutations + 1.0)

    if np.isfinite(null_sd) and null_sd > 0:
        ses = (null_mean - observed) / null_sd
    else:
        ses = float("nan")

    return observed, null_mean, null_sd, ses, p_value


def fmt(x):
    return "NA" if not np.isfinite(x) else f"{x:.8g}"


def main():
    args = parse_args()
    if args.permutations < 1:
        raise ValueError("--permutations must be at least 1.")
    if Path(args.project).name != args.project or args.project in (".", ".."):
        raise ValueError("project must be a single folder name.")

    tree_path = Path(args.input_folder) / args.project / args.tree
    if not tree_path.is_file():
        raise FileNotFoundError(f"Tree file not found: {tree_path}")

    tree, tips, number_of_levels = load_tree(tree_path)
    levels = parse_levels(args.level)
    if any(level > number_of_levels for level in levels):
        raise ValueError(
            f"Requested levels {levels} exceed the {number_of_levels} ecological "
            "levels found in the tree (last field is the genome name)."
        )
    output_dir = Path(args.output_folder) / args.project
    output_dir.mkdir(parents=True, exist_ok=True)
    distances = patristic_matrix(tree, tips)
    delimiter = args.delimiter

    for level in levels:
        labels, genomes, counts = read_tree_and_labels(tree, tips, level, delimiter)
        observed, null_mean, null_sd, ses, p_value = permutation_test(
            distances, labels, args.permutations, args.seed
        )
        between = between_mpd(distances, labels) if len(counts) > 1 else float("nan")
        effect = ((between - observed) / between
                  if np.isfinite(between) and between != 0 else float("nan"))
        category_rows = category_statistics(distances, labels)

        print(f"\nTree: {tree_path}")
        print(f"Ecological level: {level}")
        print(f"Genomes: {len(tips)}")
        print(f"Categories: {len(counts)}")
        print("Category sizes:")
        for category, n in sorted(counts.items()):
            print(f"  {category}: {n}")
        print("\nGlobal phylogeny-ecology overlap:")
        for name, value in (
            ("observed within-category MPD", observed),
            ("between-category MPD", between),
            ("null mean MPD", null_mean),
            ("null SD", null_sd),
            ("standardized effect (SES)", ses),
            ("overlap effect", effect),
            ("permutation P (clustering)", p_value),
        ):
            print(f"  {name:32s}: {fmt(value)}")
        print("\nPer-category observed MPD:")
        print("  category\tgenomes\tpairs\tMPD")
        for category, n, pairs, mpd in category_rows:
            print(f"  {category}\t{n}\t{pairs}\t{fmt(mpd)}")

        stem = Path(args.output_file).stem if args.output_file else tree_path.stem
        output_path = output_dir / f"{stem}_level{level}.tsv"
        with output_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, delimiter="\t")
            writer.writerow(["metric", "value"])
            for metric, value in (
                ("tree", str(tree_path)), ("level", level),
                ("genomes", len(tips)), ("categories", len(counts)),
                ("permutations", args.permutations), ("seed", args.seed),
                ("observed_within_category_MPD", observed),
                ("between_category_MPD", between), ("null_mean_MPD", null_mean),
                ("null_sd_MPD", null_sd), ("SES", ses),
                ("overlap_effect", effect), ("permutation_P", p_value),
            ):
                writer.writerow([metric, value])
            writer.writerow([])
            writer.writerow(["category", "genomes", "within_pairs", "observed_MPD"])
            writer.writerows(category_rows)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
