#!/usr/bin/env python3
"""MTB_AMR_Classifier command-line interface.

Inference-only wrapper around NetworkParser query:

    trained model + FASTQ / FASTA / VCF sample -> predicted hierarchy path
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

try:
    from mtb_amr_classifier.inputs import VALID_INPUT_TYPES
    from mtb_amr_classifier.predict import load_config, predict_hierarchy
except ImportError:  # pragma: no cover
    from inputs import VALID_INPUT_TYPES  # type: ignore
    from predict import load_config, predict_hierarchy  # type: ignore

LOGGER = logging.getLogger("mtb_amr_classifier")


def configure_logging(verbose: bool = False, quiet: bool = False) -> None:
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _set_if_provided(config: Any, key: str, value: Any) -> None:
    if value is not None:
        setattr(config, key, value)


def build_predict_parser(
    prog: Optional[str] = None, add_help: bool = True
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Predict the MTB hierarchy path of a sample from a trained "
            "NetworkParser model. Accepts FASTQ, FASTA, or VCF input."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=add_help,
    )
    parser.add_argument(
        "--model",
        "--bundle",
        dest="model",
        required=True,
        help=(
            "Trained model: networkparser_model_bundle.npb (preferred) or a "
            "hierarchical_model_registry.json from NetworkParser training."
        ),
    )
    parser.add_argument(
        "--sample",
        "--genomic",
        dest="sample",
        required=True,
        help=(
            "Sample input: a VCF/gVCF file or directory, a FASTA file or "
            "directory, a paired-end FASTQ directory, or a feature matrix."
        ),
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for hierarchy-path predictions and query audits.",
    )
    parser.add_argument(
        "--ref_fasta",
        default=None,
        help="Reference FASTA/GenBank. Required for FASTQ; recommended for VCF/FASTA.",
    )
    parser.add_argument(
        "--input-type",
        "--query_input_type",
        dest="input_type",
        choices=list(VALID_INPUT_TYPES),
        default="auto",
        help="How to interpret --sample. auto detects FASTQ, FASTA, VCF, or matrix.",
    )
    parser.add_argument(
        "--fasta-mapping-mode",
        "--raw_sequence_mapping_mode",
        dest="fasta_mapping_mode",
        choices=["auto", "blast", "exact"],
        default="auto",
        help="How FASTA sequences are mapped onto trained marker contexts.",
    )
    parser.add_argument(
        "--max_markers",
        type=int,
        default=10,
        help="Maximum supporting markers to report per hierarchy level.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional JSON file with NetworkParserConfig overrides.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=None,
        help="Parallel workers where supported. Defaults to the config value.",
    )

    fastq = parser.add_argument_group("FASTQ preprocessing")
    fastq.add_argument("--fastq_max_parallel_samples", type=int, default=None)
    fastq.add_argument("--fastq_threads", type=int, default=None)
    fastq.add_argument("--fastq_memory_per_sample_mb", type=int, default=None)
    fastq.add_argument(
        "--fastq_clean_intermediates",
        action="store_true",
        help="Remove FASTQ intermediate working files after successful preprocessing.",
    )
    fastq.add_argument(
        "--fastq_no_auto_index_reference",
        action="store_true",
        help="Do not create missing BWA/samtools indexes for the reference.",
    )
    fastq.add_argument("--fastq_min_mapping_quality", type=int, default=None)

    review = parser.add_argument_group("Query-time review guards")
    review.add_argument(
        "--no_low_support_review",
        action="store_true",
        help="Do not replace rare-class predictions with a review-required label.",
    )
    review.add_argument(
        "--no_amr_weak_evidence_review",
        action="store_true",
        help="Disable the weak-evidence AMR guard.",
    )
    review.add_argument(
        "--amr_weak_evidence_mode",
        choices=["warn", "block"],
        default=None,
        help="warn keeps the class and flags it; block replaces a weak susceptible call.",
    )

    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mtb_amr_classifier",
        description=(
            "MTB_AMR_Classifier: predict Mycobacterium tuberculosis hierarchy "
            "paths (lineage → AMR) from a trained NetworkParser model."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")
    predict = subparsers.add_parser(
        "predict",
        parents=[build_predict_parser(add_help=False)],
        add_help=True,
        help="Load a trained model and predict the hierarchy path of a sample.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    predict.set_defaults(command="predict")
    return parser


def apply_runtime_overrides(config: Any, args: argparse.Namespace) -> Any:
    _set_if_provided(config, "n_jobs", getattr(args, "n_jobs", None))
    _set_if_provided(
        config,
        "fastq_max_parallel_samples",
        getattr(args, "fastq_max_parallel_samples", None),
    )
    _set_if_provided(config, "fastq_threads", getattr(args, "fastq_threads", None))
    _set_if_provided(
        config,
        "fastq_memory_per_sample_mb",
        getattr(args, "fastq_memory_per_sample_mb", None),
    )
    _set_if_provided(
        config,
        "fastq_min_mapping_quality",
        getattr(args, "fastq_min_mapping_quality", None),
    )
    if bool(getattr(args, "fastq_clean_intermediates", False)):
        config.fastq_clean_intermediates = True
    if bool(getattr(args, "fastq_no_auto_index_reference", False)):
        config.fastq_auto_index_reference = False
    if bool(getattr(args, "no_low_support_review", False)):
        config.low_support_review_enabled = False
    if bool(getattr(args, "no_amr_weak_evidence_review", False)):
        config.amr_weak_evidence_review_enabled = False
    _set_if_provided(
        config,
        "amr_weak_evidence_mode",
        getattr(args, "amr_weak_evidence_mode", None),
    )
    if hasattr(config, "__post_init__"):
        config.__post_init__()
    return config


def _print_hierarchy_summary(predictions: Any, output_dir: Path) -> None:
    if predictions is None or getattr(predictions, "empty", True):
        LOGGER.warning("No predictions were written.")
        return

    path_col = (
        "predicted_hierarchy_path"
        if "predicted_hierarchy_path" in predictions.columns
        else None
    )
    print(f"Wrote predictions to {output_dir}")
    print(f"Samples: {len(predictions)}")
    if path_col is None:
        LOGGER.warning(
            "Prediction table has no predicted_hierarchy_path column. "
            "Check query_predictions.csv in the output directory."
        )
        return

    for _, row in predictions.iterrows():
        sample_id = row.get("sample_id", "unknown")
        path = row.get(path_col, "")
        terminal = row.get("predicted_terminal_label", "")
        status = row.get("hierarchy_terminal_status", "")
        extra = ""
        if terminal:
            extra = f" | terminal={terminal}"
        if status:
            extra += f" | status={status}"
        print(f"  {sample_id}: {path}{extra}")


def run_predict(args: argparse.Namespace) -> int:
    config = load_config(getattr(args, "config", None))
    config = apply_runtime_overrides(config, args)
    predictions = predict_hierarchy(
        model=args.model,
        sample=args.sample,
        output_dir=args.output_dir,
        ref_fasta=args.ref_fasta,
        input_type=args.input_type,
        config=config,
        n_jobs=args.n_jobs,
        max_markers=int(args.max_markers),
        fasta_mapping_mode=args.fasta_mapping_mode,
    )
    _print_hierarchy_summary(predictions, Path(args.output_dir))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw = list(argv) if argv is not None else sys.argv[1:]
    if not raw or raw[0] in {"-h", "--help"}:
        parser = build_parser()
        parser.parse_args(raw or ["--help"])
        return 0
    if raw[0] == "predict":
        parser = build_parser()
        args = parser.parse_args(raw)
    else:
        parser = build_predict_parser(prog="mtb_amr_classifier")
        args = parser.parse_args(raw)
        args.command = "predict"

    configure_logging(
        verbose=bool(getattr(args, "verbose", False)),
        quiet=bool(getattr(args, "quiet", False)),
    )
    try:
        return run_predict(args)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
