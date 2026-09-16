"""Predict a hierarchy path for one or more MTB samples from a trained model."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

try:
    from mtb_amr_classifier.config import NetworkParserConfig
    from mtb_amr_classifier.inputs import detect_input_type
    from mtb_amr_classifier.model_bundle import query_bundle
    from mtb_amr_classifier.query_engine import NetworkParserQueryEngine
except ImportError:  # pragma: no cover
    from config import NetworkParserConfig  # type: ignore
    from inputs import detect_input_type  # type: ignore
    from model_bundle import query_bundle  # type: ignore
    from query_engine import NetworkParserQueryEngine  # type: ignore

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

HIERARCHY_PATH_COLUMNS = [
    "sample_id",
    "predicted_hierarchy_path",
    "predicted_terminal_label",
    "predicted_terminal_level",
    "hierarchy_terminal_status",
    "hierarchy_terminal_reason",
    "predicted_level1",
    "predicted_level2",
    "predicted_level1_identity",
    "predicted_level2_identity",
]


def load_config(config_path: Optional[PathLike] = None) -> NetworkParserConfig:
    config = NetworkParserConfig()
    if config_path is None:
        config.__post_init__()
        return config

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        overrides: Dict[str, Any] = json.load(handle)
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning("Ignoring unknown config key: %s", key)
    config.__post_init__()
    return config


def write_hierarchy_paths(predictions: pd.DataFrame, output_dir: Path) -> Path:
    """Write a compact table focused on the predicted hierarchy route."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "hierarchy_paths.tsv"
    columns = [col for col in HIERARCHY_PATH_COLUMNS if col in predictions.columns]
    if "sample_id" not in columns and "sample_id" in predictions.columns:
        columns = ["sample_id"] + columns
    if not columns:
        columns = list(predictions.columns)
    predictions.loc[:, columns].to_csv(path, sep="\t", index=False)
    return path


def predict_hierarchy(
    *,
    model: PathLike,
    sample: PathLike,
    output_dir: PathLike,
    ref_fasta: Optional[PathLike] = None,
    input_type: str = "auto",
    config: Optional[NetworkParserConfig] = None,
    n_jobs: Optional[int] = None,
    max_markers: int = 10,
    fasta_mapping_mode: str = "auto",
) -> pd.DataFrame:
    """Apply a trained NetworkParser model bundle or registry to a new sample.

    Parameters
    ----------
    model
        Path to ``networkparser_model_bundle.npb`` (preferred) or a hierarchy
        registry JSON from NetworkParser training.
    sample
        FASTQ directory, FASTA file/directory, VCF file/directory, or a
        precomputed feature matrix.
    output_dir
        Directory for prediction tables and query audits.
    """
    model_path = Path(model)
    sample_path = Path(sample)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample input not found: {sample_path}")

    if config is None:
        config = NetworkParserConfig()
        config.__post_init__()
    if n_jobs is not None:
        config.n_jobs = int(n_jobs)

    resolved_type = detect_input_type(sample_path, input_type)
    if resolved_type == "fastq" and not ref_fasta:
        raise ValueError(
            "FASTQ samples require --ref_fasta so reads can be aligned before prediction."
        )

    ref = str(ref_fasta) if ref_fasta is not None else None
    logger.info(
        "Predicting hierarchy path | model=%s | sample=%s | input_type=%s",
        model_path,
        sample_path,
        resolved_type,
    )

    if model_path.suffix.lower() == ".npb":
        predictions = query_bundle(
            bundle_path=model_path,
            genomic_path=str(sample_path),
            output_dir=out,
            config=config,
            ref_fasta=ref,
            max_markers=int(max_markers),
            n_jobs=n_jobs,
            query_input_type=resolved_type,
            raw_sequence_mapping_mode=fasta_mapping_mode,
        )
    else:
        engine = NetworkParserQueryEngine(
            registry_path=str(model_path),
            config=config,
        )
        predictions = engine.query(
            genomic_path=str(sample_path),
            output_dir=str(out),
            ref_fasta=ref,
            max_markers=int(max_markers),
            n_jobs=n_jobs,
            query_input_type=resolved_type,
            raw_sequence_mapping_mode=fasta_mapping_mode,
        )

    write_hierarchy_paths(predictions, out)
    return predictions
