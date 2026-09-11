#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Command-line interface for general supervised-model recommendation."""

from __future__ import annotations

import argparse
import math
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List


# training/model_selector.py -> ../lib
BASE_DIR = Path(__file__).resolve().parent
LIB_DIR = BASE_DIR / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.append(str(LIB_DIR))

try:
    import nn__algorithm_selector as NAS
except ImportError as exc:
    raise SystemExit(
        f"ERROR: Cannot import ../lib/nn__algorithm_selector.py from {LIB_DIR}: {exc}"
    )

try:
    import NeuralNetwork as NN
except ImportError:
    NN = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a hierarchical marker matrix and recommend a supervised "
            "classification algorithm for training."
        )
    )

    parser.add_argument(
        "project_folder",
        help="Project subfolder inside input_folder. Mandatory positional argument.",
    )
    parser.add_argument(
        "-i", "--input_folder",
        default="input",
        help="Input root folder containing project folders (default: input).",
    )
    parser.add_argument(
        "-o", "--output_folder",
        default="output",
        help="Output root folder containing project folders (default: output).",
    )
    parser.add_argument(
        "-f", "--input_file",
        default="",
        help=(
            "Optional CSV/TSV matrix filename inside input_folder/project_folder. "
            "If omitted, all CSV/TSV files in the project folder are analyzed."
        ),
    )
    parser.add_argument(
        "-d", "--delimiter",
        default="|",
        help=(
            "Delimiter separating hierarchical labels and the final genome name "
            "in the first matrix column (default: '|')."
        ),
    )
    parser.add_argument(
        "--empty_symbol",
        default="",
        help="Additional symbol representing missing feature values (default: empty string).",
    )
    parser.add_argument(
        "--remove_empty_field", "--remove_empty_filed",
        dest="remove_empty_field",
        type=float,
        default=1.0,
        help=(
            "Remove feature columns whose missing-value fraction is greater than "
            "this threshold; range 0.0..1.0 (default: 1.0 = keep all)."
        ),
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="Maximum number of stratified cross-validation folds (default: 5).",
    )
    parser.add_argument(
        "--top_alternatives",
        type=int,
        default=3,
        help="Number of alternative algorithms to report (default: 3).",
    )

    return parser.parse_args()


def _format_cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "none"
    if isinstance(value, (tuple, list)):
        return ",".join(str(v) for v in value)
    return str(value)


def _build_run_command(
    project: str,
    algorithm: str,
    input_file: str,
    input_folder: str,
    output_folder: str,
    delimiter: str,
    parameters: Dict[str, Any] | None = None,
) -> str:
    """
    Build the proposed run.py syntax.

    A fixed algorithm means that run.py should use that algorithm throughout
    the hierarchy. AUTO means node-specific algorithm and parameter selection.
    """
    parts: List[str] = [
        "python3", "run.py", project, algorithm,
        "-i", input_folder,
        "-o", output_folder,
        "-f", os.path.basename(input_file),
        "-d", delimiter,
    ]

    for key, value in (parameters or {}).items():
        if value is None:
            continue
        parts.extend(["--" + key, _format_cli_value(value)])

    return " ".join(shlex.quote(str(part)) for part in parts)


def _fmt_score(value: float) -> str:
    return f"{value:.4f}" if math.isfinite(value) else "not available"


def _build_report(
    input_file: str,
    matrix: NAS.MatrixData,
    selection: NAS.HierarchySelectionResult,
    args: argparse.Namespace,
) -> str:
    lines: List[str] = []
    add = lines.append

    add("=== SUPERVISED MODEL SELECTOR ===")
    add(f"Input:                 {input_file}")
    add(f"Matrix shape:          {matrix.X.shape[0]} x {matrix.X.shape[1]}")
    add(f"Feature data type:     {matrix.data_type}")
    add(f"Hierarchy levels:      {matrix.n_levels}")
    add(f"Label delimiter:       {args.delimiter!r}")
    add(f"Evaluable levels:      {selection.evaluable_levels}")
    add(f"Removed feature cols:  {len(matrix.removed_columns)}")

    if matrix.removed_columns:
        preview = ", ".join(
            f"{name}({fraction:.1%})"
            for name, fraction in matrix.removed_columns[:10]
        )
        if len(matrix.removed_columns) > 10:
            preview += f" ... +{len(matrix.removed_columns) - 10} more"
        add(f"  {preview}")

    add("")
    add(
        "General ranking is based on the mean stratified cross-validated "
        "balanced accuracy across evaluable hierarchy levels."
    )
    add(
        "The report intentionally does not list recommendations for every hierarchy "
        "node; node-specific selection is delegated to run.py AUTO."
    )

    if selection.best is None:
        add("")
        add("No reliable general recommendation could be produced.")
        add(f"Status: {selection.status}")
        add("=" * 70)
        return "\n".join(lines) + "\n"

    best = selection.best
    add("")
    add("Best general model:")
    add(f"  {best.algorithm}  (mean balanced accuracy: {_fmt_score(best.score)})")
    add(f"  Levels contributing to score: {best.levels_evaluated}/{best.levels_total}")

    add("")
    add("Why selected:")
    for reason in best.rationale:
        add(f"  - {reason}")

    add("")
    add("Recommended parameters for a fixed-model run:")
    for key, value in best.parameters.items():
        add(f"  {key}: {value}")

    add("")
    add("Recommended fixed-model command:")
    add("  " + _build_run_command(
        project=args.project_folder,
        algorithm=best.algorithm,
        input_file=input_file,
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        delimiter=args.delimiter,
        parameters=best.parameters,
    ))

    if selection.alternatives:
        add("")
        add("Good alternatives (matrix-specific options):")
        for alt in selection.alternatives:
            add(
                f"  - {alt.algorithm}  "
                f"(mean balanced accuracy: {_fmt_score(alt.score)}, "
                f"levels {alt.levels_evaluated}/{alt.levels_total})"
            )
            add("    " + _build_run_command(
                project=args.project_folder,
                algorithm=alt.algorithm,
                input_file=input_file,
                input_folder=args.input_folder,
                output_folder=args.output_folder,
                delimiter=args.delimiter,
                parameters=alt.parameters,
            ))

    if matrix.n_levels > 1:
        add("")
        add("Recommended hierarchical training mode:")
        add(
            "  AUTO is recommended when different hierarchy nodes may require "
            "different classifiers or parameter settings."
        )
        add(
            "  During AUTO training, run.py should call "
            "nn__algorithm_selector.recommend_model() separately at each internal "
            "node using the samples in that node and its immediate child labels."
        )
        add("  " + _build_run_command(
            project=args.project_folder,
            algorithm="AUTO",
            input_file=input_file,
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            delimiter=args.delimiter,
        ))

    add("")
    add("Candidate mean CV scores:")
    for algorithm in NAS.SUPPORTED_MODELS:
        score = selection.mean_scores.get(algorithm, float("nan"))
        coverage = selection.level_coverage.get(algorithm, 0)
        add(f"  {algorithm:<9} {_fmt_score(score):<13} levels={coverage}")

    add("=" * 70)
    return "\n".join(lines) + "\n"


def process_file(input_file: str, output_folder: str, args: argparse.Namespace) -> int:
    matrix = NAS.read_hierarchical_matrix(
        file_path=input_file,
        delimiter=args.delimiter,
        empty_threshold=args.remove_empty_field,
        empty_symbol=args.empty_symbol,
    )

    selection = NAS.recommend_hierarchy(
        matrix=matrix,
        delimiter=args.delimiter,
        cv_folds=args.cv_folds,
        top_alternatives=args.top_alternatives,
    )

    report = _build_report(input_file, matrix, selection, args)

    os.makedirs(output_folder, exist_ok=True)
    report_file = os.path.join(
        output_folder,
        Path(input_file).stem + "_model_selector_report.txt",
    )
    with open(report_file, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(report)

    print("\n" + report)
    print(f"Report saved to: {report_file}\n")
    return 0


def _validate_environment() -> None:
    if NN is None:
        print(
            f"WARNING: NeuralNetwork.py could not be imported from {LIB_DIR}. "
            "The selector can still use sklearn probes, but run.py requires the "
            "NeuralNetwork module for final training.",
            file=sys.stderr,
        )
        return

    missing = NAS.validate_neuralnetwork_module(NN)
    if missing:
        print(
            "WARNING: NeuralNetwork.py does not expose expected model class(es): "
            + ", ".join(missing),
            file=sys.stderr,
        )


def main() -> int:
    args = parse_args()

    if not 0.0 <= args.remove_empty_field <= 1.0:
        raise ValueError("ERROR: --remove_empty_field must be in the range 0.0..1.0!")
    if args.cv_folds < 2:
        raise ValueError("ERROR: --cv_folds must be >= 2!")
    if args.top_alternatives < 0:
        raise ValueError("ERROR: --top_alternatives must be >= 0!")
    if not args.delimiter:
        raise ValueError("ERROR: --delimiter must not be empty!")

    _validate_environment()

    project_input = os.path.join(args.input_folder, args.project_folder)
    if not os.path.isdir(project_input):
        raise FileNotFoundError(
            f"ERROR: Project input folder does not exist: {project_input}!"
        )

    project_output = os.path.join(args.output_folder, args.project_folder)
    os.makedirs(project_output, exist_ok=True)

    input_files: List[str] = []
    if args.input_file:
        input_path = os.path.join(project_input, args.input_file)
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"ERROR: Input matrix does not exist: {input_path}!")
        input_files.append(input_path)
    else:
        for name in sorted(os.listdir(project_input)):
            if name.lower().endswith((".tsv", ".csv")):
                input_files.append(os.path.join(project_input, name))

    if not input_files:
        raise FileNotFoundError(
            f"ERROR: No CSV/TSV matrices were found in {project_input}!"
        )

    for input_file in input_files:
        process_file(input_file, project_output, args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
