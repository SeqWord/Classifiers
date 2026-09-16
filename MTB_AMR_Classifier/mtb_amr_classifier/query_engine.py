#!/usr/bin/env python3
# network_parser/query_engine.py
"""
NetworkParser user-facing query engine
=====================================

Purpose
-------
Apply a trained hierarchical NetworkParser model registry to new strain/sample
input and produce a user-facing prediction report.

The query engine is inference-only:

    new strain/sample -> same feature representation -> saved Level 1 model
                      -> saved Level 2 model -> report

It does not rerun RF-FDR, permutation testing, FDR correction, decision-tree
training, or bootstrap confidence. Those are training/discovery-time operations.

Expected trained input
----------------------
A hierarchical model registry produced by ``hierarchy_protocol.py`` with:

    level1.model_file
    level1.features
    level2.global_fallback.model_file / features
    level2.by_level1_group.<group>.model_file / features, where available

Outputs
-------
    query_predictions.csv
    query_predictions_compact.tsv
    query_predictions_readable.html
    query_route_audit.json
    query_report.json
    query_report.txt
    query_alignment_summary.json
"""

from __future__ import annotations

import argparse
import copy
import html
import json
import logging
import pickle
import sys
import threading
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

try:
    from mtb_amr_classifier.config import NetworkParserConfig
    from mtb_amr_classifier.data_loader import DataLoader
    from mtb_amr_classifier.sequence_query_encoder import (
        encode_raw_sequence_query,
        encode_vcf_query_from_manifest,
        load_feature_manifest,
    )
    from mtb_amr_classifier.utils import (
        normalize_sample_id,
        progress_iter,
        resolve_effective_n_jobs,
        should_run_parallel,
    )
    from mtb_amr_classifier.fastq_processor import FastqProcessor
    from mtb_amr_classifier.matrix_contract import (
        FittedMissingnessState,
        transform_with_missingness_state,
    )
    from mtb_amr_classifier.vcf_call_semantics import (
        CallState,
        VcfQCConfig,
        callability_gate_result,
    )
except ImportError:  # pragma: no cover - supports direct source-tree execution
    from config import NetworkParserConfig  # type: ignore
    from data_loader import DataLoader  # type: ignore
    from sequence_query_encoder import (  # type: ignore
        encode_raw_sequence_query,
        encode_vcf_query_from_manifest,
        load_feature_manifest,
    )
    from utils import (  # type: ignore
        normalize_sample_id,
        progress_iter,
        resolve_effective_n_jobs,
        should_run_parallel,
    )
    from fastq_processor import FastqProcessor  # type: ignore
    from matrix_contract import (  # type: ignore
        FittedMissingnessState,
        transform_with_missingness_state,
    )
    from vcf_call_semantics import (  # type: ignore
        CallState,
        VcfQCConfig,
        callability_gate_result,
    )


logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=json_default)


def load_config(config_path: Optional[str]) -> NetworkParserConfig:
    config = NetworkParserConfig()
    if config_path is None:
        config.__post_init__()
        return config

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        overrides = json.load(handle)

    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning("Ignoring unknown config key in query config: %s", key)

    config.__post_init__()
    return config


TRAINED_VCF_CONFIG_KEYS = (
    "qual_threshold",
    "min_dp_per_sample",
    "min_gq_per_sample",
    "mq_threshold",
    "mq0f_threshold",
    "biallelic_only",
    "vcf_supported_ploidies",
    "vcf_respect_filter",
    "vcf_allowed_filters",
    "assume_absent_variant_is_reference",
    "expand_gvcf_ref_blocks",
    "validate_ref_against_genome",
    "min_feature_recovery_fraction",
    "min_callable_fraction",
    "enforce_query_callability_gates",
    "contig_alias_map",
    "allow_position_only_vcf_match",
    "ancestral_allele",
)


def apply_trained_vcf_config(
    config: NetworkParserConfig, registry: Dict[str, Any]
) -> NetworkParserConfig:
    """Copy VCF encoding settings saved with the trained model.

    Variant-only AFRO VCFs were trained with min_gq=0 and absent-site = REF.
    Querying without those settings marks every sample as uncallable.
    """
    saved = registry.get("config") if isinstance(registry, dict) else None
    if not isinstance(saved, dict):
        return config
    applied: List[str] = []
    for key in TRAINED_VCF_CONFIG_KEYS:
        if key not in saved or not hasattr(config, key):
            continue
        value = saved[key]
        if getattr(config, key) != value:
            setattr(config, key, value)
            applied.append(f"{key}={value!r}")
    if applied:
        logger.info("Using trained-model VCF settings: %s", ", ".join(applied))
    if hasattr(config, "__post_init__"):
        config.__post_init__()
    return config


def resolve_path(path_value: Optional[str], base_dir: Path) -> Optional[Path]:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    return path


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_pickle(path: Path) -> Any:
    """
    Load saved model payloads robustly.

    Uses joblib first for sklearn-style models, then pickle as fallback.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model payload not found: {path}")

    try:
        from mtb_amr_classifier.pickle_compat import (
            install_network_parser_pickle_aliases,
        )
    except ImportError:  # pragma: no cover
        from pickle_compat import install_network_parser_pickle_aliases  # type: ignore

    install_network_parser_pickle_aliases()
    try:
        import joblib

        loaded = joblib.load(path)
    except Exception:
        with open(path, "rb") as handle:
            loaded = pickle.load(handle)
    try:
        from mtb_amr_classifier.model_bundle import _patch_sklearn_estimator_compat

        return _patch_sklearn_estimator_compat(loaded)
    except Exception:
        return loaded


# -----------------------------------------------------------------------------
# Matrix loading and feature alignment
# -----------------------------------------------------------------------------


def load_query_matrix(
    genomic_path: str,
    output_dir: Path,
    config: NetworkParserConfig,
    ref_fasta: Optional[str] = None,
    n_jobs: Optional[int] = None,
) -> pd.DataFrame:
    """Load/construct the query sample × genomic-feature matrix.

    Query inputs are not discovery cohorts. For VCF-derived query samples we
    therefore relax cohort-level feature-retention filters so observed query
    variants are preserved and later aligned to the trained selected-feature
    space. Missing trained features are still filled as 0 by
    align_to_training_features().
    """
    query_config = copy.copy(config)
    query_config.min_sample_presence = 1
    query_config.remove_invariant = False
    query_config.min_minor_count = 0
    query_config.matrices_min_count = 0

    loader = DataLoader(
        config=query_config,
        n_jobs=n_jobs if n_jobs is not None else getattr(query_config, "n_jobs", -1),
    )
    X = loader.load_genomic_matrix(
        file_path=genomic_path,
        output_dir=str(output_dir / "query_matrix_artifacts"),
        ref_fasta=ref_fasta,
    )
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "DataLoader.load_genomic_matrix did not return a pandas DataFrame."
        )
    X = X.copy()
    X.index = X.index.astype(str).map(normalize_sample_id)
    return X


def is_hierarchical_registry(registry: Dict[str, Any]) -> bool:
    """Return True when the registry uses the recursive hierarchy schema."""
    if not isinstance(registry, dict):
        return False
    protocol = str(registry.get("protocol", "")).strip().lower()
    hierarchy = registry.get("hierarchy", {})
    return protocol == "multi_level_hierarchy_protocol" or (
        isinstance(hierarchy, dict) and isinstance(hierarchy.get("root"), dict)
    )


def _add_unique_features(
    ordered: List[str],
    seen: set,
    features: Iterable[Any],
) -> None:
    for feature in features or []:
        f = str(feature)
        if f and f not in seen:
            seen.add(f)
            ordered.append(f)


def _collect_features_from_hierarchy_node(
    node: Dict[str, Any],
    ordered: List[str],
    seen: set,
) -> None:
    """Recursively collect selected features from all trainable hierarchy nodes."""
    if not isinstance(node, dict):
        return
    _add_unique_features(ordered, seen, node.get("features", []))
    children = node.get("children", {})
    if isinstance(children, dict):
        for child in children.values():
            if isinstance(child, dict):
                _collect_features_from_hierarchy_node(child, ordered, seen)


def _collect_manifest_candidates_from_hierarchy_node(
    node: Dict[str, Any],
    candidates: List[Optional[str]],
) -> None:
    """Recursively collect selected-feature manifest paths from hierarchy nodes."""
    if not isinstance(node, dict):
        return
    feature_manifest = node.get("feature_manifest")
    if isinstance(feature_manifest, dict):
        candidates.append(feature_manifest.get("manifest_file"))
    children = node.get("children", {})
    if isinstance(children, dict):
        for child in children.values():
            if isinstance(child, dict):
                _collect_manifest_candidates_from_hierarchy_node(child, candidates)


def collect_required_features_from_registry(registry: Dict[str, Any]) -> List[str]:
    """Collect the union of every feature required by hierarchy models."""
    ordered: List[str] = []
    seen: set = set()

    if is_hierarchical_registry(registry):
        hierarchy = registry.get("hierarchy", {}) if isinstance(registry, dict) else {}
        root = hierarchy.get("root", {}) if isinstance(hierarchy, dict) else {}
        _collect_features_from_hierarchy_node(root, ordered, seen)
        return ordered

    level1 = registry.get("level1", {}) if isinstance(registry, dict) else {}
    _add_unique_features(ordered, seen, level1.get("features", []))

    level2 = registry.get("level2", {}) if isinstance(registry, dict) else {}
    global_payload = (
        level2.get("global_fallback", {}) if isinstance(level2, dict) else {}
    )
    _add_unique_features(ordered, seen, global_payload.get("features", []))
    global_binary_payload = (
        level2.get("global_binary_fallback", {}) if isinstance(level2, dict) else {}
    )
    _add_unique_features(ordered, seen, global_binary_payload.get("features", []))

    by_group = level2.get("by_level1_group", {}) if isinstance(level2, dict) else {}
    if isinstance(by_group, dict):
        for payload in by_group.values():
            if isinstance(payload, dict):
                _add_unique_features(ordered, seen, payload.get("features", []))

    return ordered


def resolve_registry_feature_manifest(
    registry: Dict[str, Any], registry_base: Path
) -> Optional[Path]:
    """Resolve the all-feature manifest saved during training."""
    candidates: List[Optional[str]] = []
    training_matrix = (
        registry.get("training_matrix", {}) if isinstance(registry, dict) else {}
    )
    candidates.append(training_matrix.get("feature_manifest_file"))

    if is_hierarchical_registry(registry):
        hierarchy = registry.get("hierarchy", {}) if isinstance(registry, dict) else {}
        root = hierarchy.get("root", {}) if isinstance(hierarchy, dict) else {}
        _collect_manifest_candidates_from_hierarchy_node(root, candidates)

    level1 = registry.get("level1", {}) if isinstance(registry, dict) else {}
    l1_manifest = level1.get("feature_manifest", {}) if isinstance(level1, dict) else {}
    if isinstance(l1_manifest, dict):
        candidates.append(l1_manifest.get("manifest_file"))

    level2 = registry.get("level2", {}) if isinstance(registry, dict) else {}
    global_payload = (
        level2.get("global_fallback", {}) if isinstance(level2, dict) else {}
    )
    g_manifest = (
        global_payload.get("feature_manifest", {})
        if isinstance(global_payload, dict)
        else {}
    )
    if isinstance(g_manifest, dict):
        candidates.append(g_manifest.get("manifest_file"))

    global_binary_payload = (
        level2.get("global_binary_fallback", {}) if isinstance(level2, dict) else {}
    )
    gb_manifest = (
        global_binary_payload.get("feature_manifest", {})
        if isinstance(global_binary_payload, dict)
        else {}
    )
    if isinstance(gb_manifest, dict):
        candidates.append(gb_manifest.get("manifest_file"))

    by_group = level2.get("by_level1_group", {}) if isinstance(level2, dict) else {}
    if isinstance(by_group, dict):
        for payload in by_group.values():
            if not isinstance(payload, dict):
                continue
            group_manifest = payload.get("feature_manifest", {})
            if isinstance(group_manifest, dict):
                candidates.append(group_manifest.get("manifest_file"))

    for candidate in candidates:
        resolved = resolve_path(candidate, registry_base)
        if resolved is not None and resolved.exists():
            return resolved
    return None


def feature_call_metadata_by_sample(
    calls: Optional[pd.DataFrame],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    if calls is None or calls.empty:
        return {}
    if "sample_id" not in calls.columns or "feature_id" not in calls.columns:
        return {}
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for _, row in calls.iterrows():
        sample_id = str(row.get("sample_id", ""))
        feature_id = str(row.get("feature_id", ""))
        if not sample_id or not feature_id:
            continue
        out.setdefault(sample_id, {})[feature_id] = row.to_dict()
    return out


def align_to_training_features(
    X_new: pd.DataFrame,
    features: Sequence[str],
    *,
    fill_missing_as_zero: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Align new samples to a saved trained feature list.

    By default, features absent from the query matrix are filled with NaN
    (not ordinary zero biological evidence). Set ``fill_missing_as_zero=True``
    only for explicit legacy compatibility; this is audited in the summary.
    Existing NaN cells (non-callable genotypes) are preserved.
    """
    requested = [str(f) for f in features]

    if not requested:
        raise ValueError("Training feature list is empty; cannot align query matrix.")

    X = X_new.copy()
    X.columns = X.columns.astype(str)

    requested_set = set(requested)
    missing = [f for f in requested if f not in X.columns]
    extra = [f for f in X.columns if f not in requested_set]

    fill_value = 0.0 if fill_missing_as_zero else float("nan")
    if missing:
        fill_block = pd.DataFrame(fill_value, index=X.index, columns=missing)
        X = pd.concat([X, fill_block], axis=1)

    X_aligned = X.loc[:, requested].copy()
    X_aligned = X_aligned.apply(pd.to_numeric, errors="coerce")
    if fill_missing_as_zero:
        # Only fill structural absence; do not invent values for already-NaN cells
        # that represent non-callable genotypes from the encoder.
        for col in missing:
            X_aligned[col] = X_aligned[col].fillna(0.0)

    missing_fraction = float(len(missing) / max(1, len(requested)))
    noncallable_fraction = (
        float(X_aligned.isna().to_numpy().mean()) if X_aligned.size else 0.0
    )

    if missing_fraction == 0 and noncallable_fraction == 0:
        alignment_status = "complete"
        warning = None
    elif missing_fraction < 0.5 and noncallable_fraction < 0.5:
        alignment_status = "partial"
        warning = (
            "Some trained features were missing or non-callable in the query input. "
            "They are encoded as NaN (not biological zero) unless fill_missing_as_zero=True. "
            "Interpret prediction support with caution."
        )
    else:
        alignment_status = "low_feature_coverage"
        warning = (
            "Many trained features were missing or non-callable in the query input. "
            "Prediction may abstain or be marked review/unresolved under callability gates."
        )

    summary = {
        "requested_training_features": int(len(requested)),
        "features_present_in_query": int(len(requested) - len(missing)),
        "missing_training_features": int(len(missing)),
        "missing_training_features_filled_as_zero": int(len(missing))
        if fill_missing_as_zero
        else 0,
        "missing_training_features_filled_as_nan": int(len(missing))
        if not fill_missing_as_zero
        else 0,
        "missing_training_feature_fraction": missing_fraction,
        "noncallable_cell_fraction": noncallable_fraction,
        "fill_missing_as_zero": bool(fill_missing_as_zero),
        "extra_query_features_ignored": int(len(extra)),
        "alignment_status": alignment_status,
        "warning": warning,
        "missing_feature_names": missing,
    }

    if fill_missing_as_zero and missing:
        logger.warning(
            "LEGACY ALIGNMENT: %d missing trained features filled as 0 "
            "(fill_missing_as_zero=True). Prefer NaN + callability gates.",
            len(missing),
        )

    if warning:
        logger.warning(warning)

    if missing_fraction > 0.3 or noncallable_fraction > 0.3:
        logger.warning(
            "High missing/non-callable feature fraction (missing=%.2f, noncallable=%.2f). "
            "Check callability (gVCF/depth) and reference/contig naming.",
            missing_fraction,
            noncallable_fraction,
        )

    return X_aligned, summary


# -----------------------------------------------------------------------------
# Model prediction helpers
# -----------------------------------------------------------------------------


def unpack_model_payload(
    payload: Any,
) -> Tuple[Any, Optional[Any], Optional[List[str]]]:
    """
    Support the fallback payload written by hierarchy_protocol.py and plain
    sklearn-like model objects.
    """
    if isinstance(payload, dict) and "model" in payload:
        model = payload.get("model")
        label_encoder = payload.get("label_encoder")
        features = payload.get("features")
        return model, label_encoder, list(features) if features is not None else None
    return payload, None, None


def _missingness_state_from_payload(payload: Any) -> Optional[FittedMissingnessState]:
    """Return the train-fitted matrix preprocessor embedded with a model."""
    model, _, _ = unpack_model_payload(payload)
    raw_state: Any = None
    if isinstance(payload, dict):
        raw_state = payload.get("missingness_state") or payload.get(
            "preprocessing_state"
        )
    if raw_state is None:
        raw_state = getattr(model, "networkparser_missingness_state", None)
    if isinstance(raw_state, FittedMissingnessState):
        return raw_state
    if isinstance(raw_state, dict) and raw_state:
        return FittedMissingnessState.from_dict(raw_state)
    return None


def build_query_callability_gates(
    X_raw: pd.DataFrame,
    features: Sequence[str],
    *,
    config: Any,
    feature_metadata_by_sample: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Calculate per-sample recovery/callability before any prediction.

    Matrix columns establish coordinate/feature recovery. Per-call metadata, when
    available, distinguishes filtered, missing-GT, unresolved and absent states.
    Numeric non-missing matrix cells are callable evidence. This function always
    delegates the final decision to the shared VCF callability policy.
    """
    requested = [str(feature) for feature in features]
    qc = VcfQCConfig.from_config(config)
    metadata = feature_metadata_by_sample or {}
    raw_columns = set(map(str, X_raw.columns))
    gates: Dict[str, Dict[str, Any]] = {}

    for raw_sample_id, row in X_raw.iterrows():
        sample_id = normalize_sample_id(str(raw_sample_id))
        sample_meta = metadata.get(sample_id, {})
        states: List[CallState] = []
        n_assumed_reference = 0
        for feature in requested:
            meta = sample_meta.get(feature, {}) if isinstance(sample_meta, dict) else {}
            raw_state = str(meta.get("call_state", "")).strip()
            try:
                state = CallState(raw_state) if raw_state else None
            except ValueError:
                state = CallState.UNRESOLVED_OR_AMBIGUOUS

            if state is None:
                if feature not in raw_columns:
                    state = CallState.LOCUS_NOT_PRESENT
                else:
                    value = row.get(feature, float("nan"))
                    state = (
                        CallState.MISSING_OR_NO_CALL
                        if pd.isna(value)
                        else (
                            CallState.CALLED_ALTERNATE
                            if float(value) == 1.0
                            else CallState.CALLED_REFERENCE
                        )
                    )
            if bool(meta.get("assumed_reference", False)):
                n_assumed_reference += 1
            states.append(state)

        gate = callability_gate_result(
            states,
            qc=qc,
            n_assumed_reference=n_assumed_reference,
        )
        state_counts = {state.value: 0 for state in CallState}
        for state in states:
            state_counts[state.value] += 1
        reasons: List[str] = []
        if not requested:
            reasons.append("empty_required_feature_list")
        if float(gate.get("feature_recovery_fraction", 0.0)) < float(
            gate.get("min_feature_recovery_fraction", 0.0)
        ):
            reasons.append("feature_recovery_below_threshold")
        if float(gate.get("callable_fraction", 0.0)) < float(
            gate.get("min_callable_fraction", 0.0)
        ):
            reasons.append("callable_fraction_below_threshold")
        if int(gate.get("n_callable", 0)) == 0:
            reasons.append("no_callable_required_features")
        gate.update(
            {
                "sample_id": sample_id,
                "n_required_features": int(len(requested)),
                "call_state_counts": state_counts,
                "abstention_reason_codes": reasons,
            }
        )
        if not requested or int(gate.get("n_callable", 0)) == 0:
            gate.update(
                {
                    "gate_passed": False,
                    "gate_status": (
                        "empty_required_feature_list_abstain"
                        if not requested
                        else "no_callable_required_features_abstain"
                    ),
                    "prediction_action": "abstain_review_unresolved",
                }
            )
        gates[sample_id] = gate
    return gates


def _prepare_query_matrix_for_payload(payload: Any, X: pd.DataFrame) -> pd.DataFrame:
    """Apply only the preprocessor fitted with the deployment model."""
    state = _missingness_state_from_payload(payload)
    if state is None:
        if X.isna().any().any():
            raise ValueError(
                "Query contains non-callable required markers but the model has no "
                "train-fitted missingness/preprocessing state."
            )
        return X.apply(pd.to_numeric, errors="coerce")
    transformed, _ = transform_with_missingness_state(
        X,
        state,
        apply_imputation=True,
        drop_high_missing_samples=False,
    )
    return transformed


def _marker_value_for_identify(value: Any) -> str:
    """Normalize query marker values to the symbol format used during ML training.

    MLProtocolRunner trains NetworkParser-style models on string symbols such
    as "0" and "1".  Query alignment produces numeric 0/1 values, so passing
    floats directly to identify() turns them into "0.0"/"1.0", which the
    fitted OneHotEncoder treats as unseen categories.
    """
    try:
        if value is None or pd.isna(value):
            return ""
    except Exception:
        if value is None:
            return ""

    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "nd"}:
        return ""

    try:
        numeric = float(text)
        if np.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
    except Exception:
        pass

    return text


def predict_labels_and_support(
    payload: Any,
    X: pd.DataFrame,
    *,
    gate_results: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[str], List[Optional[float]], List[Dict[str, float]]]:
    """
    Predict labels plus model support scores.

    ``predict_proba`` outputs are reported as uncalibrated model support,
    not as probability confidence.

    Supports:
      - sklearn-like models with predict()/predict_proba()
      - fallback payloads containing model + label_encoder
      - NetworkParser-style models exposing identify()
    """
    model, label_encoder, _ = unpack_model_payload(payload)

    sample_ids = [normalize_sample_id(str(value)) for value in X.index]
    predict_positions: List[int] = []
    for position, sample_id in enumerate(sample_ids):
        gate = (gate_results or {}).get(sample_id)
        if gate is None or gate.get("prediction_action") == "predict":
            predict_positions.append(position)

    labels: List[str] = ["unavailable" for _ in sample_ids]
    max_support: List[Optional[float]] = [None for _ in sample_ids]
    class_support: List[Dict[str, float]] = [{} for _ in sample_ids]
    if not predict_positions:
        return labels, max_support, class_support

    X_predict = X.iloc[predict_positions].copy()
    try:
        X_model = _prepare_query_matrix_for_payload(payload, X_predict)
    except Exception as exc:
        for position in predict_positions:
            sample_id = sample_ids[position]
            if gate_results is not None:
                gate = gate_results.setdefault(sample_id, {})
                gate.update(
                    {
                        "gate_passed": False,
                        "gate_status": "missing_preprocessing_state_abstain",
                        "prediction_action": "abstain_review_unresolved",
                    }
                )
                reasons = list(gate.get("abstention_reason_codes") or [])
                if "missing_train_fitted_preprocessing_state" not in reasons:
                    reasons.append("missing_train_fitted_preprocessing_state")
                gate["abstention_reason_codes"] = reasons
                gate["preprocessing_error"] = str(exc)
        return labels, max_support, class_support

    if hasattr(model, "predict"):
        raw_pred = model.predict(X_model)

        if label_encoder is not None:
            try:
                predicted_labels = [
                    str(v) for v in label_encoder.inverse_transform(raw_pred)
                ]
            except Exception:
                predicted_labels = [str(v) for v in raw_pred]
        else:
            predicted_labels = [str(v) for v in raw_pred]
        for position, label in zip(predict_positions, predicted_labels):
            labels[position] = label

        if hasattr(model, "predict_proba"):
            try:
                proba = np.asarray(model.predict_proba(X_model), dtype=float)

                if label_encoder is not None and hasattr(label_encoder, "classes_"):
                    classes = [str(c) for c in label_encoder.classes_]
                elif hasattr(model, "classes_"):
                    classes = [str(c) for c in model.classes_]
                else:
                    classes = [str(i) for i in range(proba.shape[1])]

                predicted_support = [float(np.max(row)) for row in proba]
                predicted_class_support = [
                    {
                        classes[i]: float(row[i])
                        for i in range(min(len(classes), len(row)))
                    }
                    for row in proba
                ]
                for local, position in enumerate(predict_positions):
                    max_support[position] = predicted_support[local]
                    class_support[position] = predicted_class_support[local]

            except Exception as exc:
                logger.warning(
                    "Model exposes predict_proba but support extraction failed: %s",
                    exc,
                )

        return labels, max_support, class_support

    if hasattr(model, "identify"):
        for local_position, (_, row) in enumerate(X_model.iterrows()):
            marker_dict = {
                str(col): _marker_value_for_identify(value)
                for col, value in row.items()
            }

            result = model.identify(marker_dict)
            pred_list = (
                result.get("predictions", []) if isinstance(result, dict) else []
            )

            if not pred_list:
                continue

            first = pred_list[0]

            if isinstance(first, dict):
                label = (
                    first.get("label") or first.get("class") or first.get("prediction")
                )
                prob = (
                    first.get("probability")
                    or first.get("support")
                    or first.get("score")
                )
            elif isinstance(first, (tuple, list)):
                label = first[0] if len(first) >= 1 else "unavailable"
                prob = first[1] if len(first) >= 2 else None
            else:
                label = first
                prob = None

            position = predict_positions[local_position]
            labels[position] = str(label)

            try:
                max_support[position] = float(prob) if prob is not None else None
            except Exception:
                max_support[position] = None

            class_support[position] = (
                {str(label): max_support[position]}
                if max_support[position] is not None
                else {}
            )

        return labels, max_support, class_support

    raise TypeError(
        "Model payload does not expose predict() or identify(); cannot perform inference."
    )


def read_ranked_feature_table(
    filter_summary: Dict[str, Any], registry_base: Path
) -> Optional[pd.DataFrame]:
    artifacts = (
        filter_summary.get("artifacts", {}) if isinstance(filter_summary, dict) else {}
    )
    table_path = artifacts.get("rf_fdr_results_csv") or artifacts.get(
        "feature_results_csv"
    )
    resolved = resolve_path(table_path, registry_base)
    if resolved is None or not resolved.exists():
        return None
    try:
        df = pd.read_csv(resolved)
        if "feature" not in df.columns:
            return None
        return df
    except Exception as exc:
        logger.warning("Could not read ranked feature table %s: %s", resolved, exc)
        return None


def extract_model_importance(
    payload: Any, features: Sequence[str]
) -> Optional[pd.DataFrame]:
    model, _, _ = unpack_model_payload(payload)
    if not hasattr(model, "feature_importances_"):
        return None
    values = np.asarray(getattr(model, "feature_importances_"), dtype=float)
    if values.shape[0] != len(features):
        return None
    return pd.DataFrame(
        {"feature": list(features), "model_importance": values}
    ).sort_values("model_importance", ascending=False)


RESOLVED_ALLELE_CALLS = {"baseline_match", "alt_match", "known_nonbaseline_match"}
NONBASELINE_ALLELE_CALLS = {"alt_match", "known_nonbaseline_match"}
BASELINE_ALLELE_CALLS = {"baseline_match"}
UNRESOLVED_ALLELE_CALLS = {
    "not_called",
    "ambiguous_base",
    "not_called_multi_hit_context",
    "non_training_allele",
}
# Roles allowed as *resolved query states* in evidence tables.
# Globally important zero-valued inputs are NOT automatic support for the
# predicted class unless they appear on the model decision path / signed
# contribution list (see prediction_explanation helpers).
RESOLVED_EVIDENCE_ROLES = {
    "resolved_nonbaseline_state",
    "resolved_baseline_state",
    "resolved_trained_zero_state",
    "aligned_matrix_state",
}
# Markers that can support a non-baseline predicted class claim
SUPPORTING_EVIDENCE_ROLES = {
    "resolved_nonbaseline_state",
}


def _normalised_text(value: Any) -> str:
    return str(value or "").strip()


def _feature_evidence_role(
    *,
    feature_id: str,
    value: Any,
    metadata: Optional[Dict[str, Any]] = None,
    available_features: Optional[set] = None,
) -> str:
    """Classify whether a zero is a real trained state or a cautious fill value."""
    meta = metadata or {}
    allele_call = _normalised_text(meta.get("allele_call"))
    mapping_status = _normalised_text(meta.get("mapping_status"))
    numeric_value = _safe_float(value)
    numeric_value = 0.0 if numeric_value is None else float(numeric_value)

    if allele_call in NONBASELINE_ALLELE_CALLS:
        return "resolved_nonbaseline_state"
    if allele_call in BASELINE_ALLELE_CALLS:
        return "resolved_baseline_state"
    if allele_call in RESOLVED_ALLELE_CALLS:
        return (
            "resolved_trained_zero_state"
            if numeric_value == 0.0
            else "resolved_nonbaseline_state"
        )
    if allele_call in UNRESOLVED_ALLELE_CALLS or mapping_status:
        return "unresolved_zero_fill"

    # Matrix-only query mode has no allele-call metadata. In that setting, a
    # feature that came from the user-supplied matrix is an aligned matrix state;
    # a feature absent from the matrix and injected by alignment is a zero-fill.
    if available_features is not None:
        return (
            "aligned_matrix_state"
            if feature_id in available_features
            else "unresolved_zero_fill"
        )

    return "aligned_matrix_state"


def _is_supporting_marker_role(role: str) -> bool:
    """True only for non-baseline resolved states (not global zeros)."""
    return str(role) in SUPPORTING_EVIDENCE_ROLES


def _is_resolved_marker_role(role: str) -> bool:
    return str(role) in RESOLVED_EVIDENCE_ROLES


def decision_tree_path_for_sample(
    model: Any,
    sample_values: pd.Series,
    feature_names: Sequence[str],
) -> Optional[Dict[str, Any]]:
    """Return the actual decision path for a tree model (not global importance)."""
    try:
        from sklearn.tree import DecisionTreeClassifier
    except Exception:
        return None
    if not isinstance(model, DecisionTreeClassifier):
        return None
    feats = [str(f) for f in feature_names]
    x = np.asarray(
        [_safe_float(sample_values.get(f)) or 0.0 for f in feats], dtype=float
    ).reshape(1, -1)
    try:
        node_indicator = model.decision_path(x)
        node_index = node_indicator.indices[
            node_indicator.indptr[0] : node_indicator.indptr[1]
        ]
        tree_ = model.tree_
        steps: List[Dict[str, Any]] = []
        for node_id in node_index:
            feat_idx = int(tree_.feature[node_id])
            if feat_idx < 0:
                steps.append({"node": int(node_id), "is_leaf": True})
                continue
            thr = float(tree_.threshold[node_id])
            fname = feats[feat_idx] if feat_idx < len(feats) else str(feat_idx)
            val = float(x[0, feat_idx])
            steps.append(
                {
                    "node": int(node_id),
                    "feature": fname,
                    "threshold": thr,
                    "sample_value": val,
                    "direction": "left" if val <= thr else "right",
                    "is_leaf": False,
                }
            )
        return {
            "explanation_type": "decision_tree_path",
            "path": steps,
            "path_features": [s["feature"] for s in steps if s.get("feature")],
            "note": "Actual tree decision path for this sample; not global feature importance.",
        }
    except Exception as exc:
        logger.debug("decision_tree_path_for_sample failed: %s", exc)
        return None


def logistic_signed_contributions(
    model: Any,
    sample_values: pd.Series,
    feature_names: Sequence[str],
    predicted_label: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Class-specific signed linear contributions for logistic regression."""
    coef = getattr(model, "coef_", None)
    intercept = getattr(model, "intercept_", None)
    classes = getattr(model, "classes_", None)
    if coef is None:
        return None
    feats = [str(f) for f in feature_names]
    x = np.asarray(
        [_safe_float(sample_values.get(f)) or 0.0 for f in feats], dtype=float
    )
    coef = np.asarray(coef, dtype=float)
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)
    if coef.shape[1] != len(feats):
        return None
    class_idx = 0
    if classes is not None and predicted_label is not None:
        try:
            class_idx = list(map(str, classes)).index(str(predicted_label))
        except ValueError:
            class_idx = 0
        if coef.shape[0] == 1 and len(list(classes)) == 2:
            # binary: coef is for classes_[1]
            class_idx = 0
    row = coef[min(class_idx, coef.shape[0] - 1)]
    contribs: List[Dict[str, Any]] = []
    for i, f in enumerate(feats):
        c = float(row[i] * x[i])
        if abs(c) > 0 or abs(float(row[i])) > 0:
            contribs.append(
                {
                    "feature": f,
                    "value": float(x[i]),
                    "coefficient": float(row[i]),
                    "signed_contribution": c,
                    "supports_predicted_class": bool(c > 0 and x[i] != 0),
                }
            )
    contribs.sort(key=lambda d: abs(d["signed_contribution"]), reverse=True)
    inter = (
        float(
            np.asarray(intercept).ravel()[
                min(class_idx, len(np.asarray(intercept).ravel()) - 1)
            ]
        )
        if intercept is not None
        else 0.0
    )
    return {
        "explanation_type": "logistic_signed_contributions",
        "predicted_label": predicted_label,
        "intercept": inter,
        "contributions": contribs,
        "note": (
            "Class-specific signed contributions (coef * value). "
            "Zero-valued inputs contribute 0 and are not class-support markers."
        ),
    }


def prediction_explanation_for_sample(
    model: Any,
    sample_values: pd.Series,
    feature_names: Sequence[str],
    predicted_label: Optional[str] = None,
) -> Dict[str, Any]:
    """Model-appropriate local explanation (path / signed contrib / resolved inputs)."""
    payload = unpack_model_payload(model)[0] if not hasattr(model, "predict") else model
    # Prefer tree path
    path = decision_tree_path_for_sample(payload, sample_values, feature_names)
    if path is not None:
        return path
    lr = logistic_signed_contributions(
        payload, sample_values, feature_names, predicted_label
    )
    if lr is not None:
        return lr
    # Generic: resolved model inputs only (not importance-ranked zeros as support)
    resolved = []
    for f in feature_names:
        v = _safe_float(sample_values.get(f))
        if v is None:
            continue
        resolved.append({"feature": str(f), "value": float(v)})
    return {
        "explanation_type": "resolved_model_inputs",
        "inputs": resolved,
        "note": (
            "Model type has no decision-path or signed-coefficient explanation; "
            "listing resolved input values only. Not probability confidence."
        ),
    }


def supporting_markers_for_sample(
    sample_values: pd.Series,
    ranked_features: Optional[pd.DataFrame],
    model_importance: Optional[pd.DataFrame],
    max_markers: int = 10,
    feature_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    available_features: Optional[set] = None,
    *,
    prefer_nonbaseline_support: bool = True,
    model: Any = None,
    predicted_label: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Return top marker evidence for a sample.

    Global importance alone does **not** mark a zero-valued feature as supporting
    the predicted class. Prefer:
      - non-baseline resolved states, and/or
      - features on the decision path / with positive signed contribution.
    Unresolved zero-fills are excluded.
    """
    feature_metadata = feature_metadata or {}
    available_features = (
        {str(f) for f in available_features} if available_features is not None else None
    )

    path_features: Set[str] = set()
    positive_contrib: Set[str] = set()
    if model is not None:
        expl = prediction_explanation_for_sample(
            model, sample_values, list(map(str, sample_values.index)), predicted_label
        )
        if expl.get("explanation_type") == "decision_tree_path":
            path_features = {str(f) for f in expl.get("path_features", [])}
        if expl.get("explanation_type") == "logistic_signed_contributions":
            positive_contrib = {
                str(c["feature"])
                for c in expl.get("contributions", [])
                if c.get("supports_predicted_class")
            }

    def _attach_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
        feature_id = str(record.get("feature", ""))
        meta = feature_metadata.get(feature_id, {})
        for key in (
            "observed_allele",
            "mapping_status",
            "mapping_quality",
            "allele_call",
            "ref_allele",
            "alt_allele",
            "baseline_allele",
            "sequence",
            "position",
            "gene_annotation",
            "nucleotide_change",
            "amino_acid_change",
            "subject_id",
            "subject_position",
            "strand",
            "mapping_method",
            "n_context_hits",
            "n_blast_hits",
            "n_equivalent_best_hits",
            "blast_pident",
            "blast_query_coverage",
            "blast_bitscore",
        ):
            if key in meta and meta.get(key) not in (None, ""):
                record[key] = meta.get(key)
        role = _feature_evidence_role(
            feature_id=feature_id,
            value=record.get("value", 0),
            metadata=meta,
            available_features=available_features,
        )
        record["evidence_role"] = role
        on_path = feature_id in path_features
        pos_c = feature_id in positive_contrib
        val = float(record.get("value", 0) or 0)
        # Support for predicted class: non-baseline resolved OR path/contrib with non-zero value
        supports_class = bool(
            _is_supporting_marker_role(role)
            or (on_path and val != 0.0 and _is_resolved_marker_role(role))
            or pos_c
        )
        record["on_decision_path"] = on_path
        record["positive_signed_contribution"] = pos_c
        record["supports_trained_marker_pattern"] = bool(_is_resolved_marker_role(role))
        record["supports_predicted_class"] = supports_class
        # Do not present global importance zeros as class support
        if prefer_nonbaseline_support and val == 0.0 and not on_path and not pos_c:
            record["supports_predicted_class"] = False
        return record

    def _candidate_record(
        feature: Any, extra: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        feature_id = str(feature)
        if feature_id not in sample_values.index:
            return None
        value = _safe_float(sample_values.get(feature_id))
        value = 0.0 if value is None else float(value)
        record: Dict[str, Any] = {"feature": feature_id, "value": value}
        if extra:
            record.update(extra)
        record = _attach_metadata(record)
        if not record.get("supports_trained_marker_pattern"):
            return None
        return record

    records: List[Dict[str, Any]] = []
    seen = set()

    if ranked_features is not None and "feature" in ranked_features.columns:
        df = ranked_features.copy()
        df["feature"] = df["feature"].astype(str)
        df = df[df["feature"].isin(set(map(str, sample_values.index)))]
        for _, row in df.iterrows():
            feature = str(row["feature"])
            if feature in seen:
                continue
            rec = _candidate_record(
                feature,
                {
                    "rf_mean_importance": _safe_float(row.get("rf_mean_importance")),
                    "empirical_p_value": _safe_float(row.get("empirical_p_value")),
                    "corrected_p_value": _safe_float(row.get("corrected_p_value")),
                },
            )
            if rec is not None:
                records.append(rec)
                seen.add(feature)
            if len(records) >= max_markers:
                return records

    if model_importance is not None and "feature" in model_importance.columns:
        df = model_importance.copy()
        df["feature"] = df["feature"].astype(str)
        df = df[df["feature"].isin(set(map(str, sample_values.index)))]
        for _, row in df.iterrows():
            feature = str(row["feature"])
            if feature in seen:
                continue
            rec = _candidate_record(
                feature,
                {"model_importance": _safe_float(row.get("model_importance"))},
            )
            if rec is not None:
                records.append(rec)
                seen.add(feature)
            if len(records) >= max_markers:
                return records

    for feature in map(str, sample_values.index):
        if feature in seen:
            continue
        rec = _candidate_record(feature)
        if rec is not None:
            records.append(rec)
            seen.add(feature)
        if len(records) >= max_markers:
            break
    return records


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _status_from_unique_fraction(
    unique_fraction: Optional[float], has_mapping_metadata: bool
) -> Tuple[str, str]:
    """Classify whether a model-specific selected feature set was mapped in the query."""
    if not has_mapping_metadata:
        return (
            "feature_space_alignment_only",
            "No per-feature allele-call metadata were available; status is based on aligned matrix states.",
        )
    if unique_fraction is None:
        return (
            "unknown_marker_recovery",
            "Per-feature mapping metadata were present, but marker recovery fraction could not be computed.",
        )
    if unique_fraction >= 0.80:
        return (
            "adequate_marker_recovery",
            "Most selected markers for this model were mapped or reported in the query input.",
        )
    if unique_fraction >= 0.50:
        return (
            "partial_marker_recovery",
            "Only part of this model's selected marker space was mapped or reported; interpret this level with caution.",
        )
    return (
        "low_marker_recovery",
        "Most selected markers for this model were unresolved or ambiguous; prediction support is likely weak.",
    )


def _status_from_active_fraction(
    active_fraction: float, active_count: int
) -> Tuple[str, str]:
    """Classify non-baseline evidence for a model-specific selected feature set."""
    if active_count >= 10 or active_fraction >= 0.01:
        return (
            "active_marker_evidence_present",
            "This model's selected feature set contains multiple non-baseline query states.",
        )
    if active_count > 0:
        return (
            "very_low_active_marker_evidence",
            "Only a small number of this model's selected features are non-baseline in the query.",
        )
    return (
        "no_active_marker_evidence",
        "This model's selected features are recovered mainly as baseline/zero states in the query.",
    )


def _status_from_resolved_fraction(
    resolved_fraction: float,
    resolved_count: int,
    has_mapping_metadata: bool,
) -> Tuple[str, str]:
    """Classify resolved trained-marker pattern evidence independently of active 1s."""
    if not has_mapping_metadata:
        if resolved_count > 0:
            return (
                "aligned_matrix_evidence_present",
                "No per-feature allele-call metadata were available, but trained features were present in the aligned query matrix.",
            )
        return (
            "feature_space_alignment_only",
            "No per-feature allele-call metadata were available; resolved marker status cannot be separated from matrix alignment.",
        )
    if resolved_fraction >= 0.80:
        return (
            "resolved_marker_evidence_present",
            "Most model-specific trained markers were resolved, including baseline states encoded as 0.",
        )
    if resolved_fraction >= 0.50:
        return (
            "partial_resolved_marker_evidence",
            "A useful fraction of model-specific trained markers was resolved, but some calls remain caution states.",
        )
    if resolved_count > 0:
        return (
            "low_resolved_marker_evidence",
            "Only a small fraction of model-specific trained markers was resolved in the query input.",
        )
    return (
        "no_resolved_marker_evidence",
        "No model-specific trained markers were confirmed as resolved query states.",
    )


def summarize_feature_evidence_for_model(
    *,
    sample_values: pd.Series,
    features: Sequence[str],
    feature_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    available_features: Optional[set] = None,
) -> Dict[str, Any]:
    """Summarise query evidence for the exact feature list used by one model.

    Non-baseline evidence is still counted, but it is no longer the only evidence
    measure. Resolved baseline states encoded as 0 are counted as valid trained-
    marker evidence for the overall query pattern.
    """
    requested = [str(f) for f in features or []]
    feature_metadata = feature_metadata or {}
    available_features = (
        {str(f) for f in available_features} if available_features is not None else None
    )

    # Do not coerce missing/non-callable (NaN) to baseline 0 for evidence counts.
    values = pd.to_numeric(sample_values.reindex(requested), errors="coerce")
    active_features = [
        str(f) for f, value in values.items() if pd.notna(value) and float(value) != 0.0
    ]
    n_features = int(len(requested))
    n_active = int(len(active_features))
    active_fraction = float(n_active / max(1, n_features))

    has_mapping_metadata = any(str(f) in feature_metadata for f in requested)
    metadata_rows = [feature_metadata.get(str(f), {}) for f in requested]

    mapping_status_values = [
        str(m.get("mapping_status", "")) for m in metadata_rows if m
    ]
    allele_call_values = [str(m.get("allele_call", "")) for m in metadata_rows if m]

    mapping_status_counts = (
        {
            str(k): int(v)
            for k, v in pd.Series(mapping_status_values, dtype="object")
            .value_counts(dropna=False)
            .to_dict()
            .items()
        }
        if mapping_status_values
        else {}
    )
    allele_call_counts = (
        {
            str(k): int(v)
            for k, v in pd.Series(allele_call_values, dtype="object")
            .value_counts(dropna=False)
            .to_dict()
            .items()
        }
        if allele_call_values
        else {}
    )

    evidence_role_counts: Dict[str, int] = {}
    resolved_features: List[str] = []
    resolved_baseline_features: List[str] = []
    resolved_nonbaseline_features: List[str] = []
    unresolved_or_missing_features: List[str] = []

    for feature in requested:
        meta = feature_metadata.get(feature, {})
        role = _feature_evidence_role(
            feature_id=feature,
            value=values.get(feature, 0),
            metadata=meta,
            available_features=available_features,
        )
        evidence_role_counts[role] = evidence_role_counts.get(role, 0) + 1
        if role in {
            "resolved_baseline_state",
            "resolved_nonbaseline_state",
            "resolved_trained_zero_state",
        }:
            resolved_features.append(feature)
        elif role == "aligned_matrix_state" and not has_mapping_metadata:
            # For matrix-only queries, present matrix states are the best
            # available trained-pattern evidence even though allele resolution is absent.
            resolved_features.append(feature)
        else:
            unresolved_or_missing_features.append(feature)

        if role in {"resolved_baseline_state", "resolved_trained_zero_state"}:
            resolved_baseline_features.append(feature)
        if role == "resolved_nonbaseline_state":
            resolved_nonbaseline_features.append(feature)

    unique_mapped = int(
        sum(1 for status in mapping_status_values if status == "mapped_unique_context")
    )
    mapped_or_reported = int(len(mapping_status_values))
    unique_fraction = (
        float(unique_mapped / max(1, mapped_or_reported))
        if has_mapping_metadata
        else None
    )

    n_resolved = int(len(resolved_features))
    n_resolved_baseline = int(len(resolved_baseline_features))
    n_resolved_nonbaseline = int(len(resolved_nonbaseline_features))
    resolved_fraction = float(n_resolved / max(1, n_features))
    resolved_baseline_fraction = float(n_resolved_baseline / max(1, n_features))
    resolved_nonbaseline_fraction = float(n_resolved_nonbaseline / max(1, n_features))

    recovery_status, recovery_reason = _status_from_unique_fraction(
        unique_fraction, has_mapping_metadata
    )
    active_status, active_reason = _status_from_active_fraction(
        active_fraction, n_active
    )
    resolved_status, resolved_reason = _status_from_resolved_fraction(
        resolved_fraction=resolved_fraction,
        resolved_count=n_resolved,
        has_mapping_metadata=has_mapping_metadata,
    )

    n_multi_hit = int(
        allele_call_counts.get("not_called_multi_hit_context", 0)
        + sum(v for k, v in mapping_status_counts.items() if "multi_hit" in str(k))
    )
    n_ambiguous = int(
        allele_call_counts.get("ambiguous_base", 0)
        + sum(v for k, v in mapping_status_counts.items() if "ambiguous_base" in str(k))
    )
    n_non_training = int(
        allele_call_counts.get("non_training_allele", 0)
        + sum(
            v
            for k, v in mapping_status_counts.items()
            if "non_training_allele" in str(k)
        )
    )
    n_unresolved_or_missing = int(
        allele_call_counts.get("not_called", 0)
        + sum(
            v
            for k, v in mapping_status_counts.items()
            if "missing_context" in str(k) or "unresolved_context" in str(k)
        )
    )
    n_zero_fill_caution = int(evidence_role_counts.get("unresolved_zero_fill", 0))

    return {
        "n_selected_features": n_features,
        "n_active_features": n_active,
        "active_feature_fraction": active_fraction,
        "active_feature_ids": active_features,
        "nonbaseline_evidence_status": active_status,
        "nonbaseline_evidence_reason": active_reason,
        "has_mapping_metadata": bool(has_mapping_metadata),
        "n_features_with_mapping_metadata": int(mapped_or_reported),
        "n_unique_mapped_features": int(unique_mapped),
        "unique_mapped_fraction": unique_fraction,
        "marker_recovery_status": recovery_status,
        "marker_recovery_reason": recovery_reason,
        "active_marker_evidence_status": active_status,
        "active_marker_evidence_reason": active_reason,
        "n_resolved_features": n_resolved,
        "resolved_feature_fraction": resolved_fraction,
        "n_resolved_baseline_features": n_resolved_baseline,
        "resolved_baseline_feature_fraction": resolved_baseline_fraction,
        "n_resolved_nonbaseline_features": n_resolved_nonbaseline,
        "resolved_nonbaseline_feature_fraction": resolved_nonbaseline_fraction,
        "resolved_marker_evidence_status": resolved_status,
        "resolved_marker_evidence_reason": resolved_reason,
        "evidence_role_counts": evidence_role_counts,
        "mapping_status_counts": mapping_status_counts,
        "allele_call_counts": allele_call_counts,
        "n_baseline_match_calls": int(allele_call_counts.get("baseline_match", 0)),
        "n_alt_match_calls": int(allele_call_counts.get("alt_match", 0)),
        "n_known_nonbaseline_match_calls": int(
            allele_call_counts.get("known_nonbaseline_match", 0)
        ),
        "n_unresolved_or_missing_calls": n_unresolved_or_missing,
        "n_multi_hit_calls": n_multi_hit,
        "n_ambiguous_base_calls": n_ambiguous,
        "n_non_training_allele_calls": n_non_training,
        "n_zero_fill_caution_features": n_zero_fill_caution,
        "n_unresolved_or_missing_features": int(len(unresolved_or_missing_features)),
    }


def low_support_review_fields(
    *,
    label_column: str,
    prediction: str,
    policy: Dict[str, Any],
    config: NetworkParserConfig,
    prefix: str,
) -> Tuple[str, Dict[str, Any]]:
    """Replace rare-class predictions with a review-required endpoint when configured."""
    if not bool(getattr(config, "low_support_review_enabled", True)):
        return prediction, {}

    candidate = str(prediction or "").strip()
    if not candidate or candidate.lower() in {
        "unavailable",
        "low_support_review_required",
        str(
            getattr(config, "low_support_review_label", "low_support_review_required")
        ).lower(),
    }:
        return prediction, {}

    per_label = policy.get("per_label", {}) if isinstance(policy, dict) else {}
    block = per_label.get(str(label_column), {}) if isinstance(per_label, dict) else {}
    classes = block.get("classes", {}) if isinstance(block, dict) else {}
    class_info = classes.get(candidate)
    if not isinstance(class_info, dict):
        return prediction, {}

    train_count = int(class_info.get("training_sample_count", 0))
    train_min = int(
        block.get(
            "min_class_count_for_training", getattr(config, "level2_min_class_count", 2)
        )
    )
    review_min = int(
        block.get(
            "min_class_count_for_confident_reporting",
            getattr(config, "low_support_review_min_class_count", 10),
        )
    )
    excluded = bool(class_info.get("excluded_from_training", False))
    requires_review = bool(class_info.get("requires_manual_review", False))
    if not excluded and not requires_review:
        return prediction, {}

    if excluded:
        reason = (
            f"Candidate label '{candidate}' was excluded from training because only "
            f"{train_count} sample(s) were available (training minimum is {train_min})."
        )
    else:
        reason = (
            f"Candidate label '{candidate}' had only {train_count} training sample(s), "
            f"below the confident-reporting threshold of {review_min}."
        )

    review_label = str(
        policy.get(
            "review_label",
            getattr(config, "low_support_review_label", "low_support_review_required"),
        )
    )
    action = str(
        policy.get(
            "recommended_action",
            getattr(
                config,
                "low_support_review_action_message",
                (
                    "Manually review this sample or merge rare classes in metadata if that "
                    "grouping is biologically appropriate."
                ),
            ),
        )
    )
    return review_label, {
        f"predicted_{prefix}": review_label,
        f"{prefix}_candidate_prediction": candidate,
        f"{prefix}_prediction_status": "low_support_review_required",
        f"{prefix}_low_support_reason": reason,
        f"{prefix}_recommended_action": action,
        f"{prefix}_training_sample_count_for_candidate": train_count,
        f"{prefix}_interpretation_confidence": "low_support_review_required",
        f"{prefix}_confidence_note": (f"{reason} {action}").strip(),
    }


def resolve_amr_evidence_guard_label_columns(config: NetworkParserConfig) -> List[str]:
    """Return metadata label columns that should receive AMR evidence guarding."""
    configured = str(
        getattr(config, "amr_evidence_guard_label_columns", "") or ""
    ).strip()
    if configured:
        return [part.strip() for part in configured.split(",") if part.strip()]
    binary_col = str(getattr(config, "level2_binary_label_column", "") or "").strip()
    if binary_col:
        return [binary_col]
    return ["AMR_binary"]


def amr_branch_evidence_is_weak(
    *,
    evidence: Dict[str, Any],
    config: NetworkParserConfig,
) -> Tuple[bool, str]:
    """Detect when a branch AMR model lacks enough resolved marker evidence."""
    resolved_fraction = float(evidence.get("resolved_feature_fraction", 0.0) or 0.0)
    resolved_count = int(evidence.get("n_resolved_features", 0) or 0)
    resolved_status = (
        str(evidence.get("resolved_marker_evidence_status", "") or "").strip().lower()
    )
    min_fraction = float(
        getattr(config, "amr_weak_evidence_min_resolved_fraction", 0.15)
    )

    if resolved_count <= 0:
        return (
            True,
            "no resolved trained-marker states were available for the branch AMR model.",
        )
    if resolved_status == "low_resolved_marker_evidence":
        return (
            True,
            "branch AMR model resolved too few trained markers to support a confident susceptible call.",
        )
    if resolved_fraction < min_fraction:
        return (
            True,
            (
                f"only {resolved_fraction:.1%} of the branch AMR feature panel resolved "
                f"(minimum for confident reporting is {min_fraction:.1%})."
            ),
        )
    return False, ""


def amr_weak_evidence_review_fields(
    *,
    label_column: str,
    prediction: str,
    evidence: Dict[str, Any],
    config: NetworkParserConfig,
    prefix: str,
    escalation_source: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Guard weak-evidence susceptible AMR calls.

    Default mode (``amr_weak_evidence_mode='warn'``): keep the model class so
    TP/FP/TN/FN tables stay clean, and attach warning / reason fields.

    Legacy mode (``amr_weak_evidence_mode='block'``): replace the reported
    prediction with ``amr_weak_evidence_review_label`` (review/abstention token).
    """
    if not bool(getattr(config, "amr_weak_evidence_review_enabled", True)):
        return prediction, {}

    candidate = str(prediction or "").strip()
    review_label = str(
        getattr(
            config, "amr_weak_evidence_review_label", "amr_evidence_review_required"
        )
    )
    if (
        not candidate
        or candidate.lower() in {"unavailable", review_label.lower()}
        or candidate.lower() != "susceptible"
    ):
        return prediction, {}

    guard_columns = {
        col.lower() for col in resolve_amr_evidence_guard_label_columns(config)
    }
    if str(label_column).strip().lower() not in guard_columns:
        return prediction, {}

    is_weak, weak_reason = amr_branch_evidence_is_weak(evidence=evidence, config=config)
    if not is_weak:
        return prediction, {}

    action = str(
        getattr(
            config,
            "amr_weak_evidence_review_action_message",
            (
                "Branch AMR prediction had insufficient resolved resistance-marker evidence. "
                "Manually review the resistance phenotype or inspect marker coverage before "
                "accepting a susceptible call."
            ),
        )
    )
    mode = str(getattr(config, "amr_weak_evidence_mode", "warn") or "warn").strip().lower()
    if mode not in {"warn", "block"}:
        mode = "warn"

    if mode == "block":
        reason = (
            f"Candidate susceptible call for '{label_column}' was blocked because {weak_reason}"
        )
        status = "amr_evidence_review_required"
        reported = review_label
        confidence = "amr_evidence_review_required"
    else:
        # warn: keep phenotype class for evaluation; flag weak evidence alongside.
        reason = (
            f"Susceptible call for '{label_column}' retained for classification, "
            f"but marker evidence is weak because {weak_reason}"
        )
        status = "amr_weak_evidence_warning"
        reported = candidate
        confidence = "amr_weak_evidence_warning"

    if escalation_source:
        reason = (
            f"{reason} A terminal AMR fallback model ({escalation_source}) was also checked "
            "but did not provide a confident resistant override."
        )

    return reported, {
        f"predicted_{prefix}": reported,
        f"{prefix}_candidate_prediction": candidate,
        f"{prefix}_prediction_status": status,
        f"{prefix}_amr_evidence_reason": reason,
        f"{prefix}_recommended_action": action,
        f"{prefix}_amr_weak_evidence_mode": mode,
        f"{prefix}_interpretation_confidence": confidence,
        f"{prefix}_confidence_note": f"{reason} {action}".strip(),
    }


def interpretation_confidence_for_level(
    *,
    support: Optional[float],
    evidence: Dict[str, Any],
    n_supporting_markers: int,
) -> Tuple[str, str]:
    """
    Combine uncalibrated model support and resolved marker evidence into a
    cautious qualitative label.

    Labels retain the historical ``*_confidence`` token for compatibility but
    are **not** calibrated probabilities.
    """
    support_value = _safe_float(support)
    active_count = int(evidence.get("n_active_features", 0) or 0)
    resolved_count = int(evidence.get("n_resolved_features", 0) or 0)
    resolved_fraction = float(evidence.get("resolved_feature_fraction", 0.0) or 0.0)
    recovery_status = str(evidence.get("marker_recovery_status", ""))
    resolved_status = str(evidence.get("resolved_marker_evidence_status", ""))
    disclaimer = " Uncalibrated model support, not probability confidence."

    if recovery_status == "low_marker_recovery" and resolved_count == 0:
        return (
            "low_confidence",
            "Prediction generated, but too few model-specific selected markers were resolved in the query input."
            + disclaimer,
        )

    if resolved_count == 0:
        return (
            "low_confidence",
            "Prediction generated, but this model received no confirmed resolved trained-marker states for this sample."
            + disclaimer,
        )

    if support_value is None:
        return (
            "evidence_available_support_unavailable",
            "Resolved trained-marker pattern evidence is present, but uncalibrated model support was unavailable.",
        )

    if support_value >= 0.70 and n_supporting_markers > 0 and resolved_fraction >= 0.50:
        if active_count == 0:
            return (
                "high_confidence_baseline_pattern",
                "Strong uncalibrated model support with many resolved trained markers (mostly baseline states)."
                + disclaimer,
            )
        return (
            "high_confidence",
            "Strong uncalibrated model support and resolved non-baseline marker evidence."
            + disclaimer,
        )

    if support_value >= 0.50 and (n_supporting_markers > 0 or resolved_count > 0):
        return (
            "moderate_confidence",
            "Some uncalibrated model support and resolved markers; interpret cautiously."
            + disclaimer,
        )

    if resolved_status in {
        "resolved_marker_evidence_present",
        "partial_resolved_marker_evidence",
        "aligned_matrix_evidence_present",
    }:
        return (
            "low_to_moderate_confidence",
            "Resolved trained-marker evidence is available, but uncalibrated model support is weak."
            + disclaimer,
        )

    return (
        "low_confidence",
        "Weak uncalibrated model support despite available marker evidence."
        + disclaimer,
    )


def flatten_feature_evidence(prefix: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Small CSV-friendly subset of feature-evidence diagnostics."""
    return {
        f"{prefix}_n_selected_features": evidence.get("n_selected_features"),
        f"{prefix}_n_active_features": evidence.get("n_active_features"),
        f"{prefix}_active_feature_fraction": evidence.get("active_feature_fraction"),
        f"{prefix}_nonbaseline_evidence_status": evidence.get(
            "nonbaseline_evidence_status"
        ),
        f"{prefix}_nonbaseline_evidence_reason": evidence.get(
            "nonbaseline_evidence_reason"
        ),
        f"{prefix}_n_resolved_features": evidence.get("n_resolved_features"),
        f"{prefix}_resolved_feature_fraction": evidence.get(
            "resolved_feature_fraction"
        ),
        f"{prefix}_n_resolved_baseline_features": evidence.get(
            "n_resolved_baseline_features"
        ),
        f"{prefix}_resolved_baseline_feature_fraction": evidence.get(
            "resolved_baseline_feature_fraction"
        ),
        f"{prefix}_n_resolved_nonbaseline_features": evidence.get(
            "n_resolved_nonbaseline_features"
        ),
        f"{prefix}_resolved_nonbaseline_feature_fraction": evidence.get(
            "resolved_nonbaseline_feature_fraction"
        ),
        f"{prefix}_resolved_marker_evidence_status": evidence.get(
            "resolved_marker_evidence_status"
        ),
        f"{prefix}_resolved_marker_evidence_reason": evidence.get(
            "resolved_marker_evidence_reason"
        ),
        f"{prefix}_n_unique_mapped_features": evidence.get("n_unique_mapped_features"),
        f"{prefix}_unique_mapped_fraction": evidence.get("unique_mapped_fraction"),
        f"{prefix}_marker_recovery_status": evidence.get("marker_recovery_status"),
        f"{prefix}_marker_recovery_reason": evidence.get("marker_recovery_reason"),
        f"{prefix}_active_marker_evidence_status": evidence.get(
            "active_marker_evidence_status"
        ),
        f"{prefix}_active_marker_evidence_reason": evidence.get(
            "active_marker_evidence_reason"
        ),
        f"{prefix}_n_baseline_match_calls": evidence.get("n_baseline_match_calls"),
        f"{prefix}_n_alt_match_calls": evidence.get("n_alt_match_calls"),
        f"{prefix}_n_known_nonbaseline_match_calls": evidence.get(
            "n_known_nonbaseline_match_calls"
        ),
        f"{prefix}_n_unresolved_or_missing_calls": evidence.get(
            "n_unresolved_or_missing_calls"
        ),
        f"{prefix}_n_multi_hit_calls": evidence.get("n_multi_hit_calls"),
        f"{prefix}_n_ambiguous_base_calls": evidence.get("n_ambiguous_base_calls"),
        f"{prefix}_n_non_training_allele_calls": evidence.get(
            "n_non_training_allele_calls"
        ),
        f"{prefix}_n_zero_fill_caution_features": evidence.get(
            "n_zero_fill_caution_features"
        ),
    }


def decision_tree_path_explanation(
    payload: Any,
    X: pd.DataFrame,
    gate_results: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, List[str]]:
    """
    Best-effort explanation for sklearn decision-tree-like models.
    If unavailable, returns empty paths. This does not train a tree.
    """
    all_ids = [str(idx) for idx in X.index]
    eligible_ids = [
        sample_id
        for sample_id in all_ids
        if gate_results is None
        or (gate_results.get(sample_id) or {}).get("prediction_action") == "predict"
    ]
    paths: Dict[str, List[str]] = {sample_id: [] for sample_id in all_ids}
    if not eligible_ids:
        return paths

    model, _, features_from_payload = unpack_model_payload(payload)
    if not hasattr(model, "tree_"):
        return paths

    try:
        X = _prepare_query_matrix_for_payload(payload, X.loc[eligible_ids])
    except ValueError:
        # A failed callability/preprocessing gate has no valid model path.
        return paths

    feature_names = (
        features_from_payload if features_from_payload is not None else list(X.columns)
    )
    tree = model.tree_
    for sample_id, row in X.iterrows():
        node_id = 0
        rules: List[str] = []
        values = row.values.astype(float)
        while tree.children_left[node_id] != tree.children_right[node_id]:
            feature_idx = int(tree.feature[node_id])
            threshold = float(tree.threshold[node_id])
            feature_name = (
                str(feature_names[feature_idx])
                if feature_idx < len(feature_names)
                else f"feature_{feature_idx}"
            )
            value = float(values[feature_idx])
            if value <= threshold:
                rules.append(f"{feature_name} <= {threshold:.6g}")
                node_id = int(tree.children_left[node_id])
            else:
                rules.append(f"{feature_name} > {threshold:.6g}")
                node_id = int(tree.children_right[node_id])
        paths[str(sample_id)] = rules
    return paths


# -----------------------------------------------------------------------------
# Query engine
# -----------------------------------------------------------------------------


class NetworkParserQueryEngine:
    """Apply saved hierarchical NetworkParser models to new samples."""

    def _resolve_label_training_support_policy(self) -> Dict[str, Any]:
        """Load saved label-support policy or rebuild it from aligned training labels."""
        policy = (
            self.registry.get("label_training_support_policy", {})
            if isinstance(self.registry, dict)
            else {}
        )
        if isinstance(policy, dict) and policy.get("per_label"):
            return policy

        try:
            from mtb_amr_classifier.label_support import (
                build_label_training_support_policy,
            )
        except ImportError:  # pragma: no cover - supports direct source-tree execution
            from label_support import build_label_training_support_policy  # type: ignore

        training_matrix = (
            self.registry.get("training_matrix", {})
            if isinstance(self.registry, dict)
            else {}
        )
        labels_csv = (
            training_matrix.get("aligned_labels_csv")
            or training_matrix.get("aligned_hierarchy_labels_csv")
            or training_matrix.get("aligned_two_level_labels_csv")
        )
        labels_path = (
            resolve_path(labels_csv, self.registry_base) if labels_csv else None
        )
        if labels_path is None or not labels_path.exists():
            return {}

        labels_df = pd.read_csv(labels_path)
        if "sample_id" in labels_df.columns:
            labels_df = labels_df.set_index("sample_id")

        label_columns: List[str] = []
        hierarchy = (
            self.registry.get("hierarchy", {})
            if isinstance(self.registry, dict)
            else {}
        )
        if isinstance(hierarchy, dict) and hierarchy.get("label_columns"):
            label_columns = [str(x) for x in hierarchy.get("label_columns", [])]
        else:
            level1 = (
                self.registry.get("level1", {})
                if isinstance(self.registry, dict)
                else {}
            )
            level2 = (
                self.registry.get("level2", {})
                if isinstance(self.registry, dict)
                else {}
            )
            level1_label = (
                str(level1.get("label_column", "")).strip()
                if isinstance(level1, dict)
                else ""
            )
            level2_label = (
                str(level2.get("label_column", "")).strip()
                if isinstance(level2, dict)
                else ""
            )
            rename_map: Dict[str, str] = {}
            if level1_label and "level1_label" in labels_df.columns:
                rename_map["level1_label"] = level1_label
            if level2_label and "level2_label" in labels_df.columns:
                rename_map["level2_label"] = level2_label
            if rename_map:
                labels_df = labels_df.rename(columns=rename_map)
            if level1_label:
                label_columns.append(level1_label)
            if level2_label:
                label_columns.append(level2_label)

        if not label_columns:
            return {}

        return build_label_training_support_policy(
            labels_df=labels_df,
            label_columns=label_columns,
            config=self.config,
        )

    def _apply_low_support_review(
        self,
        *,
        label_column: str,
        prediction: str,
        prefix: str,
        support_policy: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        """Apply query-time low-support review policy to one hierarchical level."""
        return low_support_review_fields(
            label_column=label_column,
            prediction=prediction,
            policy=support_policy,
            config=self.config,
            prefix=prefix,
        )

    def _low_support_review_bundle(
        self,
        *,
        label_column: str,
        prediction: str,
        prefix: str,
        support_policy: Dict[str, Any],
    ) -> Tuple[str, str, Dict[str, Any], Dict[str, Any]]:
        """Return reported/routing predictions plus row and step review fields."""
        candidate = str(prediction or "").strip()
        final_pred, review_fields = self._apply_low_support_review(
            label_column=label_column,
            prediction=candidate,
            prefix=prefix,
            support_policy=support_policy,
        )
        if not review_fields:
            return candidate, candidate, {}, {}

        step_fields = {
            "prediction": final_pred,
            "candidate_prediction": review_fields.get(f"{prefix}_candidate_prediction"),
            "prediction_status": review_fields.get(f"{prefix}_prediction_status"),
            "low_support_reason": review_fields.get(f"{prefix}_low_support_reason"),
            "recommended_action": review_fields.get(f"{prefix}_recommended_action"),
        }
        if f"{prefix}_interpretation_confidence" in review_fields:
            step_fields["interpretation_confidence"] = review_fields[
                f"{prefix}_interpretation_confidence"
            ]
        if f"{prefix}_confidence_note" in review_fields:
            step_fields["confidence_note"] = review_fields[f"{prefix}_confidence_note"]
        return final_pred, candidate, review_fields, step_fields

    def _predict_saved_terminal_fallback(
        self,
        *,
        sample_id: str,
        X_raw: pd.DataFrame,
        fallback_payload: Dict[str, Any],
        fallback_source: str,
        alignment_by_node: Dict[str, Any],
        max_markers: int,
        raw_available_features: set,
        sample_feature_metadata: Dict[str, Dict[str, Any]],
        node_key: str,
    ) -> Dict[str, Any]:
        """Run one saved terminal-fallback model without mutating hierarchy steps."""
        features, payload, ranked, importance = self._load_hierarchy_node_payload(
            fallback_payload
        )
        X_node, alignment = align_to_training_features(X_raw.loc[[sample_id]], features)
        node_gates = self._gates_for_features(
            X_raw=X_raw.loc[[sample_id]],
            features=features,
            feature_metadata_by_sample={sample_id: sample_feature_metadata},
        )
        alignment["callability_gates"] = node_gates
        alignment_by_node.setdefault(node_key, alignment)
        pred, support, class_support = predict_labels_and_support(
            payload,
            X_node,
            gate_results=node_gates,
        )
        prediction = str(pred[0]) if pred else "unavailable"
        support_value = support[0] if support else None
        class_support_value = class_support[0] if class_support else {}
        markers = supporting_markers_for_sample(
            X_node.loc[sample_id],
            ranked_features=ranked,
            model_importance=importance,
            max_markers=max_markers,
            feature_metadata=sample_feature_metadata,
            available_features=raw_available_features,
        )
        evidence = summarize_feature_evidence_for_model(
            sample_values=X_node.loc[sample_id],
            features=features,
            feature_metadata=sample_feature_metadata,
            available_features=raw_available_features,
        )
        confidence, confidence_note = interpretation_confidence_for_level(
            support=support_value,
            evidence=evidence,
            n_supporting_markers=len(markers),
        )
        resistant_probability = None
        if isinstance(class_support_value, dict):
            for key, value in class_support_value.items():
                if str(key).strip().lower() == "resistant":
                    resistant_probability = float(value)
                    break
        return {
            "prediction": prediction,
            "support": support_value,
            "class_support": class_support_value,
            "resistant_probability": resistant_probability,
            "interpretation_confidence": confidence,
            "confidence_note": confidence_note,
            "feature_evidence": evidence,
            "supporting_markers": markers,
            "fallback_source": fallback_source,
            "callability_gate": node_gates.get(sample_id, {}),
        }

    def _maybe_escalate_amr_with_terminal_fallback(
        self,
        *,
        sample_id: str,
        X_raw: pd.DataFrame,
        hierarchy_steps: Sequence[Dict[str, Any]],
        alignment_by_node: Dict[str, Any],
        max_markers: int,
        raw_available_features: set,
        sample_feature_metadata: Dict[str, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Try lineage/global terminal AMR fallbacks when branch evidence is weak."""
        if not bool(
            getattr(self.config, "hierarchy_global_amr_fallback_on_weak_evidence", True)
        ):
            return None

        fallback_payload, fallback_source = self._select_terminal_fallback_payload(
            hierarchy_steps
        )
        if not isinstance(fallback_payload, dict):
            return None

        node_key = f"amr_evidence_escalation:{fallback_source}"
        result = self._predict_saved_terminal_fallback(
            sample_id=sample_id,
            X_raw=X_raw,
            fallback_payload=fallback_payload,
            fallback_source=fallback_source,
            alignment_by_node=alignment_by_node,
            max_markers=max_markers,
            raw_available_features=raw_available_features,
            sample_feature_metadata=sample_feature_metadata,
            node_key=node_key,
        )
        min_resistant_prob = float(
            getattr(
                self.config,
                "hierarchy_global_amr_fallback_min_resistant_probability",
                0.50,
            )
        )
        resistant_probability = result.get("resistant_probability")
        if str(result.get("prediction", "")).strip().lower() != "resistant":
            return None
        if (
            resistant_probability is None
            or float(resistant_probability) < min_resistant_prob
        ):
            return None
        return result

    def _apply_amr_evidence_guard_bundle(
        self,
        *,
        label_column: str,
        prediction: str,
        prefix: str,
        evidence: Dict[str, Any],
        sample_id: str,
        X_raw: pd.DataFrame,
        hierarchy_steps: List[Dict[str, Any]],
        alignment_by_node: Dict[str, Any],
        max_markers: int,
        raw_available_features: set,
        sample_feature_metadata: Dict[str, Dict[str, Any]],
        support_value: Optional[float],
        confidence: str,
        confidence_note: str,
    ) -> Tuple[str, str, Dict[str, Any], Dict[str, Any], Optional[float], str, str]:
        """Escalate or review weak-evidence susceptible AMR branch predictions."""
        candidate = str(prediction or "").strip()
        reported = candidate
        routing = candidate
        review_fields: Dict[str, Any] = {}
        step_fields: Dict[str, Any] = {}
        final_support = support_value
        final_confidence = confidence
        final_confidence_note = confidence_note

        guard_columns = {
            col.lower() for col in resolve_amr_evidence_guard_label_columns(self.config)
        }
        if str(label_column).strip().lower() not in guard_columns:
            return (
                reported,
                routing,
                review_fields,
                step_fields,
                final_support,
                final_confidence,
                final_confidence_note,
            )

        is_weak, _weak_reason = amr_branch_evidence_is_weak(
            evidence=evidence, config=self.config
        )
        if not is_weak or candidate.lower() != "susceptible":
            return (
                reported,
                routing,
                review_fields,
                step_fields,
                final_support,
                final_confidence,
                final_confidence_note,
            )

        escalation = self._maybe_escalate_amr_with_terminal_fallback(
            sample_id=sample_id,
            X_raw=X_raw,
            hierarchy_steps=hierarchy_steps,
            alignment_by_node=alignment_by_node,
            max_markers=max_markers,
            raw_available_features=raw_available_features,
            sample_feature_metadata=sample_feature_metadata,
        )
        if isinstance(escalation, dict):
            reported = str(escalation.get("prediction", reported))
            routing = reported
            final_support = escalation.get("support", final_support)
            final_confidence = str(
                escalation.get("interpretation_confidence", final_confidence)
            )
            final_confidence_note = (
                "Weak branch AMR evidence triggered terminal AMR fallback escalation "
                f"({escalation.get('fallback_source')}). {escalation.get('confidence_note', '')}"
            ).strip()
            step_fields = {
                "prediction": reported,
                "candidate_prediction": candidate,
                "prediction_status": "amr_terminal_fallback_escalation",
                "amr_evidence_reason": (
                    "Branch AMR model predicted susceptible with weak resolved-marker evidence; "
                    f"terminal fallback '{escalation.get('fallback_source')}' predicted resistant."
                ),
                "recommended_action": "Inspect fallback supporting markers and confirm resistance phenotype.",
                "interpretation_confidence": final_confidence,
                "confidence_note": final_confidence_note,
                "fallback_source": escalation.get("fallback_source"),
                "fallback_resistant_probability": escalation.get(
                    "resistant_probability"
                ),
            }
            review_fields = {
                f"predicted_{prefix}": reported,
                f"{prefix}_candidate_prediction": candidate,
                f"{prefix}_prediction_status": "amr_terminal_fallback_escalation",
                f"{prefix}_amr_evidence_reason": step_fields["amr_evidence_reason"],
                f"{prefix}_recommended_action": step_fields["recommended_action"],
                f"{prefix}_fallback_source": escalation.get("fallback_source"),
                f"{prefix}_fallback_resistant_probability": escalation.get(
                    "resistant_probability"
                ),
                f"{prefix}_interpretation_confidence": final_confidence,
                f"{prefix}_confidence_note": final_confidence_note,
            }
            return (
                reported,
                routing,
                review_fields,
                step_fields,
                final_support,
                final_confidence,
                final_confidence_note,
            )

        escalation_source = None
        reported, review_fields = amr_weak_evidence_review_fields(
            label_column=label_column,
            prediction=candidate,
            evidence=evidence,
            config=self.config,
            prefix=prefix,
            escalation_source=escalation_source,
        )
        if review_fields:
            routing = candidate
            final_confidence = review_fields.get(
                f"{prefix}_interpretation_confidence", final_confidence
            )
            final_confidence_note = review_fields.get(
                f"{prefix}_confidence_note", final_confidence_note
            )
            step_fields = {
                "prediction": reported,
                "candidate_prediction": review_fields.get(
                    f"{prefix}_candidate_prediction"
                ),
                "prediction_status": review_fields.get(f"{prefix}_prediction_status"),
                "amr_evidence_reason": review_fields.get(
                    f"{prefix}_amr_evidence_reason"
                ),
                "recommended_action": review_fields.get(f"{prefix}_recommended_action"),
                "interpretation_confidence": final_confidence,
                "confidence_note": final_confidence_note,
            }
        return (
            reported,
            routing,
            review_fields,
            step_fields,
            final_support,
            final_confidence,
            final_confidence_note,
        )

    def _init_query_caches(self) -> None:
        """Initialize per-query payload caches shared by registry and bundled engines."""
        self._hierarchy_payload_cache: Dict[
            str, Tuple[List[str], Any, Optional[pd.DataFrame], Optional[pd.DataFrame]]
        ] = {}
        self._hierarchy_payload_cache_lock = threading.Lock()
        self._level2_payload_cache: Dict[
            str,
            Tuple[str, List[str], Any, Optional[pd.DataFrame], Optional[pd.DataFrame]],
        ] = {}
        self._level2_payload_cache_lock = threading.Lock()

    def _gates_for_features(
        self,
        *,
        X_raw: pd.DataFrame,
        features: Sequence[str],
        feature_metadata_by_sample: Optional[
            Dict[str, Dict[str, Dict[str, Any]]]
        ] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Shared per-node callability gate for bundled and registry queries."""
        return build_query_callability_gates(
            X_raw,
            features,
            config=self.config,
            feature_metadata_by_sample=feature_metadata_by_sample,
        )

    def __init__(self, registry_path: str, config: NetworkParserConfig):
        self.registry_path = Path(registry_path)
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Model registry not found: {self.registry_path}")
        self.registry_base = self.registry_path.parent
        self.registry = load_json(self.registry_path)
        self.config = apply_trained_vcf_config(config, self.registry)
        self._init_query_caches()

    def _load_hierarchy_node_payload(
        self,
        node: Dict[str, Any],
    ) -> Tuple[List[str], Any, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load the saved model payload for one trainable hierarchy node."""
        model_path = resolve_path(node.get("model_file"), self.registry_base)
        cache_key = str(model_path) if model_path is not None else str(id(node))
        with self._hierarchy_payload_cache_lock:
            cached = self._hierarchy_payload_cache.get(cache_key)
            if cached is not None:
                return cached

        features = [str(f) for f in node.get("features", [])]
        if not features:
            raise ValueError(
                "Hierarchy node is marked trainable but has no selected features."
            )
        if model_path is None or not model_path.exists():
            raise ValueError("Hierarchy node is missing a readable model file.")
        payload = load_pickle(model_path)
        ranked = read_ranked_feature_table(node.get("filter", {}), self.registry_base)
        model_importance = extract_model_importance(payload, features)
        result = (features, payload, ranked, model_importance)
        with self._hierarchy_payload_cache_lock:
            self._hierarchy_payload_cache[cache_key] = result
        return result

    @staticmethod
    def _hierarchy_node_key(node: Dict[str, Any], path_tokens: Sequence[str]) -> str:
        level = str(node.get("level_number", "NA"))
        label = str(node.get("label_column", "label"))
        path_part = "/".join(str(p) for p in path_tokens if str(p)) or "root"
        return f"level_{level}:{label}:{path_part}"

    @staticmethod
    def _hierarchy_low_confidence_levels() -> frozenset:
        return frozenset(
            {
                "low_confidence",
                "low_to_moderate_confidence",
            }
        )

    def _get_hierarchy_global_lineage_fallback(self) -> Optional[Dict[str, Any]]:
        hierarchy = (
            self.registry.get("hierarchy", {})
            if isinstance(self.registry, dict)
            else {}
        )
        payload = (
            hierarchy.get("global_lineage_fallback")
            if isinstance(hierarchy, dict)
            else None
        )
        if isinstance(payload, dict) and payload.get("status") == "success":
            return payload
        return None

    def _global_lineage_fallback_label_column(self) -> Optional[str]:
        payload = self._get_hierarchy_global_lineage_fallback()
        if not isinstance(payload, dict):
            return None
        label = str(payload.get("target_label_column", "")).strip()
        return label or None

    def _should_use_global_lineage_fallback(
        self,
        *,
        branch_status: str,
        branch_confidence: str,
        branch_prediction: str,
        branch_support: Optional[float],
        global_prediction: str,
        global_support: Optional[float],
    ) -> Tuple[bool, str]:
        if branch_status != "success":
            return True, "branch_node_unavailable"

        if bool(
            getattr(
                self.config, "hierarchy_global_lineage_fallback_on_low_confidence", True
            )
        ):
            if branch_confidence in self._hierarchy_low_confidence_levels():
                return True, "low_branch_confidence"

        if bool(
            getattr(
                self.config, "hierarchy_global_lineage_fallback_on_disagreement", True
            )
        ):
            if (
                branch_prediction
                and global_prediction
                and branch_prediction != global_prediction
            ):
                delta = float(
                    getattr(
                        self.config,
                        "hierarchy_global_lineage_fallback_min_support_delta",
                        0.0,
                    )
                )
                if global_support is not None and (
                    branch_support is None or global_support >= branch_support + delta
                ):
                    return True, "global_lineage_disagreement_recovery"

        return False, ""

    def _predict_global_lineage_fallback(
        self,
        *,
        sample_id: str,
        X_raw: pd.DataFrame,
        fallback_payload: Dict[str, Any],
        alignment_by_node: Dict[str, Any],
        max_markers: int,
        raw_available_features: set,
        sample_feature_metadata: Dict[str, Dict[str, Any]],
        node_key: str,
    ) -> Dict[str, Any]:
        features, payload, ranked, importance = self._load_hierarchy_node_payload(
            fallback_payload
        )
        X_node, alignment = align_to_training_features(X_raw.loc[[sample_id]], features)
        node_gates = self._gates_for_features(
            X_raw=X_raw.loc[[sample_id]],
            features=features,
            feature_metadata_by_sample={sample_id: sample_feature_metadata},
        )
        alignment["callability_gates"] = node_gates
        alignment_by_node.setdefault(node_key, alignment)
        pred, support, class_support = predict_labels_and_support(
            payload,
            X_node,
            gate_results=node_gates,
        )
        prediction = str(pred[0]) if pred else "unavailable"
        support_value = support[0] if support else None
        class_support_value = class_support[0] if class_support else {}
        tree_paths = decision_tree_path_explanation(payload, X_node, node_gates)
        markers = supporting_markers_for_sample(
            X_node.loc[sample_id],
            ranked_features=ranked,
            model_importance=importance,
            max_markers=max_markers,
            feature_metadata=sample_feature_metadata,
            available_features=raw_available_features,
        )
        evidence = summarize_feature_evidence_for_model(
            sample_values=X_node.loc[sample_id],
            features=features,
            feature_metadata=sample_feature_metadata,
            available_features=raw_available_features,
        )
        confidence, confidence_note = interpretation_confidence_for_level(
            support=support_value,
            evidence=evidence,
            n_supporting_markers=len(markers),
        )
        return {
            "prediction": prediction,
            "support": support_value,
            "class_support": class_support_value,
            "interpretation_confidence": confidence,
            "confidence_note": confidence_note,
            "feature_evidence": evidence,
            "supporting_markers": markers,
            "decision_path": tree_paths.get(sample_id, []),
            "callability_gate": node_gates.get(sample_id, {}),
        }

    def _select_terminal_fallback_payload(
        self,
        hierarchy_steps: Sequence[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Return the best terminal fallback model for an incomplete hierarchy route.

        Recursive hierarchy training may intentionally skip deeper branches when
        a parent slice is too small or a label is not statistically separable.
        Query mode should still be able to use the saved terminal endpoint
        fallback rather than returning only ``unavailable`` for the downstream
        endpoint.

        Selection rule:
        1. Prefer the deepest already predicted parent label with a successful
           parent-conditioned fallback model.
        2. Otherwise use the global terminal fallback, if available.
        """
        hierarchy = (
            self.registry.get("hierarchy", {})
            if isinstance(self.registry, dict)
            else {}
        )
        fallbacks = (
            hierarchy.get("terminal_fallbacks", {})
            if isinstance(hierarchy, dict)
            else {}
        )
        if not isinstance(fallbacks, dict) or fallbacks.get("status") != "success":
            return None, "no_successful_terminal_fallbacks_in_registry"

        by_parent = fallbacks.get("by_parent_label", {})
        if isinstance(by_parent, dict):
            # Walk from deepest completed step back to root. This allows a
            # hierarchy such as Lineage -> Resistance_Profile -> AMR_binary to
            # fall back by Resistance_Profile when available, or by Lineage when
            # the profile node itself is unavailable.
            for step in reversed(list(hierarchy_steps)):
                label_column = str(step.get("label_column", "")).strip()
                prediction = str(step.get("prediction", "")).strip()
                if (
                    not label_column
                    or not prediction
                    or prediction.lower() == "unavailable"
                ):
                    continue
                block = by_parent.get(label_column)
                if not isinstance(block, dict):
                    continue
                models = block.get("models", {})
                if not isinstance(models, dict):
                    continue
                payload = models.get(prediction)
                if isinstance(payload, dict) and payload.get("status") == "success":
                    return payload, f"terminal_fallback_by_{label_column}"

        global_payload = fallbacks.get("global")
        if (
            isinstance(global_payload, dict)
            and global_payload.get("status") == "success"
        ):
            return global_payload, "global_terminal_fallback"

        return None, "no_matching_terminal_fallback_model"

    def _apply_terminal_fallback_prediction(
        self,
        *,
        sample_id: str,
        X_raw: pd.DataFrame,
        raw_available_features: set,
        sample_feature_metadata: Dict[str, Dict[str, Any]],
        hierarchy_steps: List[Dict[str, Any]],
        row: Dict[str, Any],
        label_columns: Sequence[str],
        alignment_by_node: Dict[str, Any],
        max_markers: int,
        trigger_reason: str,
        support_policy: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[str], Optional[int], str, str]:
        """Predict the terminal endpoint using a saved fallback model.

        Returns
        -------
        used, terminal_label, terminal_level, terminal_status, terminal_reason
        """
        fallback_payload, fallback_source = self._select_terminal_fallback_payload(
            hierarchy_steps
        )
        if not isinstance(fallback_payload, dict):
            return (
                False,
                None,
                None,
                "stopped",
                f"{trigger_reason}; terminal_fallback_unavailable: {fallback_source}",
            )

        target_label = str(fallback_payload.get("target_label_column", "")).strip()
        if not target_label:
            return (
                False,
                None,
                None,
                "stopped",
                f"{trigger_reason}; terminal_fallback_missing_target_label",
            )

        try:
            target_level = list(label_columns).index(target_label) + 1
        except ValueError:
            target_level = len(label_columns) if label_columns else None

        prefix = f"level{target_level}" if target_level is not None else "terminal"
        node_key = f"terminal_fallback:{fallback_source}:{target_label}"

        features, payload, ranked, importance = self._load_hierarchy_node_payload(
            fallback_payload
        )
        X_node, alignment = align_to_training_features(X_raw.loc[[sample_id]], features)
        node_gates = self._gates_for_features(
            X_raw=X_raw.loc[[sample_id]],
            features=features,
            feature_metadata_by_sample={sample_id: sample_feature_metadata},
        )
        alignment["callability_gates"] = node_gates
        alignment_by_node.setdefault(node_key, alignment)
        pred, support, class_support = predict_labels_and_support(
            payload,
            X_node,
            gate_results=node_gates,
        )
        prediction = str(pred[0]) if pred else "unavailable"
        support_value = support[0] if support else None
        class_support_value = class_support[0] if class_support else {}
        tree_paths = decision_tree_path_explanation(payload, X_node, node_gates)

        markers = supporting_markers_for_sample(
            X_node.loc[sample_id],
            ranked_features=ranked,
            model_importance=importance,
            max_markers=max_markers,
            feature_metadata=sample_feature_metadata,
            available_features=raw_available_features,
        )
        evidence = summarize_feature_evidence_for_model(
            sample_values=X_node.loc[sample_id],
            features=features,
            feature_metadata=sample_feature_metadata,
            available_features=raw_available_features,
        )
        confidence, confidence_note = interpretation_confidence_for_level(
            support=support_value,
            evidence=evidence,
            n_supporting_markers=len(markers),
        )
        decision_path = tree_paths.get(sample_id, [])

        review_policy = support_policy if isinstance(support_policy, dict) else {}
        reported, _routing, review_row, review_step = self._low_support_review_bundle(
            label_column=target_label,
            prediction=prediction,
            prefix=prefix,
            support_policy=review_policy,
        )
        prediction = reported
        if review_row:
            confidence = review_row.get(
                f"{prefix}_interpretation_confidence", confidence
            )
            confidence_note = review_row.get(
                f"{prefix}_confidence_note", confidence_note
            )

        step = {
            "level_number": target_level,
            "label_column": target_label,
            "prediction": prediction,
            "support": support_value,
            "class_support": class_support_value,
            "node_status": "terminal_fallback",
            "node_key": node_key,
            "fallback_source": fallback_source,
            "fallback_trigger_reason": trigger_reason,
            "interpretation_confidence": confidence,
            "confidence_note": confidence_note,
            "feature_evidence": evidence,
            "supporting_markers": markers,
            "decision_path": decision_path,
        }
        step.update(review_step)
        hierarchy_steps.append(step)

        row.update(
            {
                f"{prefix}_label_column": target_label,
                f"predicted_{prefix}": prediction,
                f"{prefix}_support": support_value,
                f"{prefix}_node_status": "terminal_fallback",
                f"{prefix}_fallback_source": fallback_source,
                f"{prefix}_fallback_trigger_reason": trigger_reason,
                f"{prefix}_interpretation_confidence": confidence,
                f"{prefix}_confidence_note": confidence_note,
                f"{prefix}_n_supporting_markers": int(len(markers)),
                **flatten_feature_evidence(prefix, evidence),
            }
        )
        row.update(review_row)

        return (
            True,
            prediction,
            int(target_level) if target_level is not None else None,
            "fallback_complete",
            f"{trigger_reason}; used_{fallback_source}",
        )

    def _traverse_hierarchy_sample(
        self,
        *,
        sample_id: str,
        X_raw: pd.DataFrame,
        root: Dict[str, Any],
        label_columns: List[str],
        raw_available_features: set,
        sample_feature_metadata: Dict[str, Dict[str, Any]],
        sample_mapping_quality: Dict[str, Any],
        query_input_type: str,
        max_markers: int,
        support_policy: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Traverse the hierarchy for one query sample."""
        alignment_by_node: Dict[str, Any] = {}
        sample_id = normalize_sample_id(sample_id)
        current = root
        path_tokens: List[str] = []
        hierarchy_steps: List[Dict[str, Any]] = []
        row: Dict[str, Any] = {"sample_id": sample_id}
        terminal_status = "complete"
        terminal_reason = "traversed_to_terminal_node"
        terminal_label: Optional[str] = None
        terminal_level: Optional[int] = None
        guard = 0

        sample_gate = sample_mapping_quality.get("callability_gate", {})
        if (
            isinstance(sample_gate, dict)
            and sample_gate.get("prediction_action") != "predict"
        ):
            reason_codes = list(sample_gate.get("abstention_reason_codes") or [])
            row.update(
                {
                    "predicted_hierarchy_path": "",
                    "predicted_terminal_label": "unavailable",
                    "predicted_terminal_level": None,
                    "hierarchy_terminal_status": "abstained",
                    "hierarchy_terminal_reason": "query_callability_gate_failed",
                    "query_prediction_action": "abstain_review_unresolved",
                    "query_abstention_reason_codes": reason_codes,
                    "query_callability_gate_status": sample_gate.get("gate_status"),
                    "query_feature_recovery_fraction": sample_gate.get(
                        "feature_recovery_fraction"
                    ),
                    "query_callable_fraction": sample_gate.get("callable_fraction"),
                    "query_call_state_counts": sample_gate.get("call_state_counts", {}),
                }
            )
            report_sample = {
                **row,
                "hierarchy_steps": [],
                "callability_gate": sample_gate,
                "fasta_mapping_quality": (
                    sample_mapping_quality if query_input_type == "fasta" else {}
                ),
                "vcf_mapping_quality": (
                    sample_mapping_quality
                    if query_input_type in {"vcf", "fastq"}
                    else {}
                ),
                "raw_sequence_mapping_quality": sample_mapping_quality,
            }
            return row, report_sample, alignment_by_node

        while isinstance(current, dict) and current and guard < 100:
            guard += 1
            status = str(current.get("status", ""))
            level_number = int(current.get("level_number", guard) or guard)
            label_column = str(current.get("label_column", f"level_{level_number}"))
            node_key = self._hierarchy_node_key(current, path_tokens)
            prefix = f"level{level_number}"

            if status == "constant":
                prediction = str(current.get("constant_label", "unavailable"))
                terminal_label = prediction
                terminal_level = level_number
                step = {
                    "level_number": level_number,
                    "label_column": label_column,
                    "prediction": prediction,
                    "support": 1.0,
                    "class_support": {prediction: 1.0},
                    "node_status": status,
                    "node_key": node_key,
                    "interpretation_confidence": "deterministic_branch",
                    "confidence_note": "This hierarchy branch had one observed child during training, so no model was fitted at this node.",
                    "feature_evidence": {},
                    "supporting_markers": [],
                    "decision_path": [],
                }
                hierarchy_steps.append(step)
                row.update(
                    {
                        f"{prefix}_label_column": label_column,
                        f"predicted_{prefix}": prediction,
                        f"{prefix}_support": 1.0,
                        f"{prefix}_node_status": status,
                        f"{prefix}_interpretation_confidence": "deterministic_branch",
                        f"{prefix}_n_supporting_markers": 0,
                    }
                )
                path_tokens.append(prediction)
                children = current.get("children", {})
                next_node = (
                    children.get(prediction) if isinstance(children, dict) else None
                )
                if isinstance(next_node, dict):
                    current = next_node
                    continue
                terminal_reason = "constant_terminal_branch"
                break

            if status != "success":
                lineage_fb = self._get_hierarchy_global_lineage_fallback()
                lineage_label = self._global_lineage_fallback_label_column()
                if (
                    isinstance(lineage_fb, dict)
                    and lineage_label
                    and label_column == lineage_label
                ):
                    fb_node_key = f"global_lineage_fallback:{label_column}:{node_key}"
                    global_result = self._predict_global_lineage_fallback(
                        sample_id=sample_id,
                        X_raw=X_raw,
                        fallback_payload=lineage_fb,
                        alignment_by_node=alignment_by_node,
                        max_markers=max_markers,
                        raw_available_features=raw_available_features,
                        sample_feature_metadata=sample_feature_metadata,
                        node_key=fb_node_key,
                    )
                    prediction = str(global_result.get("prediction", "unavailable"))
                    terminal_label = prediction
                    terminal_level = level_number
                    trigger_reason = (
                        f"{current.get('reason', 'hierarchy_node_not_trainable')}; "
                        "global_lineage_fallback_for_unavailable_branch"
                    )
                    step = {
                        "level_number": level_number,
                        "label_column": label_column,
                        "prediction": prediction,
                        "support": global_result.get("support"),
                        "class_support": global_result.get("class_support", {}),
                        "node_status": "global_lineage_fallback",
                        "node_key": fb_node_key,
                        "fallback_trigger_reason": trigger_reason,
                        "interpretation_confidence": global_result.get(
                            "interpretation_confidence"
                        ),
                        "confidence_note": global_result.get("confidence_note"),
                        "feature_evidence": global_result.get("feature_evidence", {}),
                        "supporting_markers": global_result.get(
                            "supporting_markers", []
                        ),
                        "decision_path": global_result.get("decision_path", []),
                    }
                    row_update = {
                        f"{prefix}_label_column": label_column,
                        f"predicted_{prefix}": prediction,
                        f"{prefix}_support": global_result.get("support"),
                        f"{prefix}_node_status": "global_lineage_fallback",
                        f"{prefix}_fallback_trigger_reason": trigger_reason,
                        f"{prefix}_interpretation_confidence": global_result.get(
                            "interpretation_confidence"
                        ),
                        f"{prefix}_confidence_note": global_result.get(
                            "confidence_note"
                        ),
                        f"{prefix}_n_supporting_markers": int(
                            len(global_result.get("supporting_markers", []))
                        ),
                        **flatten_feature_evidence(
                            prefix, global_result.get("feature_evidence", {})
                        ),
                    }
                    (
                        reported,
                        routing,
                        review_row,
                        review_step,
                    ) = self._low_support_review_bundle(
                        label_column=label_column,
                        prediction=prediction,
                        prefix=prefix,
                        support_policy=support_policy,
                    )
                    step.update(review_step)
                    row_update.update(review_row)
                    if review_row:
                        row_update[f"predicted_{prefix}"] = reported
                    hierarchy_steps.append(step)
                    row.update(row_update)
                    terminal_label = reported
                    path_tokens.append(routing)
                    children = current.get("children", {})
                    next_node = (
                        children.get(routing) if isinstance(children, dict) else None
                    )
                    if isinstance(next_node, dict):
                        current = next_node
                        continue

                    terminal_reason = "no_child_node_after_global_lineage_fallback"
                    if level_number < len(label_columns):
                        (
                            used_fallback,
                            fb_label,
                            fb_level,
                            fb_status,
                            fb_reason,
                        ) = self._apply_terminal_fallback_prediction(
                            sample_id=sample_id,
                            X_raw=X_raw,
                            raw_available_features=raw_available_features,
                            sample_feature_metadata=sample_feature_metadata,
                            hierarchy_steps=hierarchy_steps,
                            row=row,
                            label_columns=label_columns,
                            alignment_by_node=alignment_by_node,
                            max_markers=max_markers,
                            trigger_reason=terminal_reason,
                            support_policy=support_policy,
                        )
                        if used_fallback:
                            terminal_label = fb_label
                            terminal_level = fb_level
                            terminal_status = fb_status
                            terminal_reason = fb_reason
                    break

                terminal_status = "stopped"
                terminal_reason = str(
                    current.get("reason", "hierarchy_node_not_trainable")
                )
                terminal_level = level_number
                step = {
                    "level_number": level_number,
                    "label_column": label_column,
                    "prediction": "unavailable",
                    "support": None,
                    "class_support": {},
                    "node_status": status or "unavailable",
                    "node_key": node_key,
                    "interpretation_confidence": "unavailable",
                    "confidence_note": terminal_reason,
                    "feature_evidence": {},
                    "supporting_markers": [],
                    "decision_path": [],
                }
                hierarchy_steps.append(step)
                row.update(
                    {
                        f"{prefix}_label_column": label_column,
                        f"predicted_{prefix}": "unavailable",
                        f"{prefix}_support": None,
                        f"{prefix}_node_status": status or "unavailable",
                        f"{prefix}_interpretation_confidence": "unavailable",
                        f"{prefix}_n_supporting_markers": 0,
                    }
                )

                (
                    used_fallback,
                    fb_label,
                    fb_level,
                    fb_status,
                    fb_reason,
                ) = self._apply_terminal_fallback_prediction(
                    sample_id=sample_id,
                    X_raw=X_raw,
                    raw_available_features=raw_available_features,
                    sample_feature_metadata=sample_feature_metadata,
                    hierarchy_steps=hierarchy_steps,
                    row=row,
                    label_columns=label_columns,
                    alignment_by_node=alignment_by_node,
                    max_markers=max_markers,
                    trigger_reason=terminal_reason,
                    support_policy=support_policy,
                )
                if used_fallback:
                    terminal_label = fb_label
                    terminal_level = fb_level
                    terminal_status = fb_status
                    terminal_reason = fb_reason
                break

            features, payload, ranked, importance = self._load_hierarchy_node_payload(
                current
            )
            X_node, alignment = align_to_training_features(
                X_raw.loc[[sample_id]], features
            )
            node_gates = self._gates_for_features(
                X_raw=X_raw.loc[[sample_id]],
                features=features,
                feature_metadata_by_sample={sample_id: sample_feature_metadata},
            )
            alignment["callability_gates"] = node_gates
            alignment_by_node.setdefault(node_key, alignment)
            pred, support, class_support = predict_labels_and_support(
                payload,
                X_node,
                gate_results=node_gates,
            )
            node_gate = node_gates.get(sample_id, {})
            if node_gate.get("prediction_action") != "predict":
                reason_codes = list(node_gate.get("abstention_reason_codes") or [])
                terminal_status = "abstained"
                terminal_reason = "node_callability_gate_failed"
                terminal_label = "unavailable"
                terminal_level = level_number
                hierarchy_steps.append(
                    {
                        "level_number": level_number,
                        "label_column": label_column,
                        "prediction": "unavailable",
                        "support": None,
                        "class_support": {},
                        "node_status": "abstained",
                        "node_key": node_key,
                        "callability_gate": node_gate,
                        "abstention_reason_codes": reason_codes,
                        "interpretation_confidence": "unavailable",
                        "confidence_note": "Required node markers did not pass callability gates.",
                        "feature_evidence": {},
                        "supporting_markers": [],
                        "decision_path": [],
                    }
                )
                row.update(
                    {
                        f"{prefix}_label_column": label_column,
                        f"predicted_{prefix}": "unavailable",
                        f"{prefix}_support": None,
                        f"{prefix}_node_status": "abstained",
                        f"{prefix}_abstention_reason_codes": reason_codes,
                        "query_prediction_action": "abstain_review_unresolved",
                        "query_abstention_reason_codes": reason_codes,
                    }
                )
                break
            prediction = str(pred[0]) if pred else "unavailable"
            support_value = support[0] if support else None
            class_support_value = class_support[0] if class_support else {}
            tree_paths = decision_tree_path_explanation(payload, X_node, node_gates)

            markers = supporting_markers_for_sample(
                X_node.loc[sample_id],
                ranked_features=ranked,
                model_importance=importance,
                max_markers=max_markers,
                feature_metadata=sample_feature_metadata,
                available_features=raw_available_features,
            )
            evidence = summarize_feature_evidence_for_model(
                sample_values=X_node.loc[sample_id],
                features=features,
                feature_metadata=sample_feature_metadata,
                available_features=raw_available_features,
            )
            confidence, confidence_note = interpretation_confidence_for_level(
                support=support_value,
                evidence=evidence,
                n_supporting_markers=len(markers),
            )
            decision_path = tree_paths.get(sample_id, [])

            node_status = status
            fallback_trigger_reason: Optional[str] = None
            lineage_fb = self._get_hierarchy_global_lineage_fallback()
            lineage_label = self._global_lineage_fallback_label_column()
            if (
                isinstance(lineage_fb, dict)
                and lineage_label
                and label_column == lineage_label
            ):
                fb_node_key = f"global_lineage_fallback:{label_column}:{node_key}"
                global_result = self._predict_global_lineage_fallback(
                    sample_id=sample_id,
                    X_raw=X_raw,
                    fallback_payload=lineage_fb,
                    alignment_by_node=alignment_by_node,
                    max_markers=max_markers,
                    raw_available_features=raw_available_features,
                    sample_feature_metadata=sample_feature_metadata,
                    node_key=fb_node_key,
                )
                use_global, trigger = self._should_use_global_lineage_fallback(
                    branch_status=status,
                    branch_confidence=confidence,
                    branch_prediction=prediction,
                    branch_support=support_value,
                    global_prediction=str(global_result.get("prediction", "")),
                    global_support=global_result.get("support"),
                )
                if use_global:
                    prediction = str(global_result.get("prediction", prediction))
                    support_value = global_result.get("support", support_value)
                    class_support_value = global_result.get(
                        "class_support", class_support_value
                    )
                    confidence = str(
                        global_result.get("interpretation_confidence", confidence)
                    )
                    confidence_note = (
                        f"Global lineage fallback used ({trigger}). "
                        f"{global_result.get('confidence_note', '')}"
                    ).strip()
                    evidence = global_result.get("feature_evidence", evidence)
                    markers = global_result.get("supporting_markers", markers)
                    decision_path = global_result.get("decision_path", decision_path)
                    node_status = "global_lineage_fallback"
                    fallback_trigger_reason = trigger
                    node_key = fb_node_key

            step = {
                "level_number": level_number,
                "label_column": label_column,
                "prediction": prediction,
                "support": support_value,
                "class_support": class_support_value,
                "node_status": node_status,
                "node_key": node_key,
                "interpretation_confidence": confidence,
                "confidence_note": confidence_note,
                "feature_evidence": evidence,
                "supporting_markers": markers,
                "decision_path": decision_path,
            }
            if fallback_trigger_reason:
                step["fallback_trigger_reason"] = fallback_trigger_reason

            row_update = {
                f"{prefix}_label_column": label_column,
                f"predicted_{prefix}": prediction,
                f"{prefix}_support": support_value,
                f"{prefix}_node_status": node_status,
                f"{prefix}_interpretation_confidence": confidence,
                f"{prefix}_confidence_note": confidence_note,
                f"{prefix}_n_supporting_markers": int(len(markers)),
                **flatten_feature_evidence(prefix, evidence),
            }
            if fallback_trigger_reason:
                row_update[
                    f"{prefix}_fallback_trigger_reason"
                ] = fallback_trigger_reason

            (
                reported,
                routing,
                review_row,
                review_step,
            ) = self._low_support_review_bundle(
                label_column=label_column,
                prediction=prediction,
                prefix=prefix,
                support_policy=support_policy,
            )
            step.update(review_step)
            row_update.update(review_row)
            if review_row:
                row_update[f"predicted_{prefix}"] = reported
                step["prediction"] = reported
                step["support"] = support_value
            if not review_row:
                (
                    amr_reported,
                    amr_routing,
                    amr_review_row,
                    amr_review_step,
                    amr_support,
                    amr_confidence,
                    amr_confidence_note,
                ) = self._apply_amr_evidence_guard_bundle(
                    label_column=label_column,
                    prediction=reported,
                    prefix=prefix,
                    evidence=evidence,
                    sample_id=sample_id,
                    X_raw=X_raw,
                    hierarchy_steps=hierarchy_steps,
                    alignment_by_node=alignment_by_node,
                    max_markers=max_markers,
                    raw_available_features=raw_available_features,
                    sample_feature_metadata=sample_feature_metadata,
                    support_value=support_value,
                    confidence=confidence,
                    confidence_note=confidence_note,
                )
                if amr_review_row:
                    reported = amr_reported
                    routing = amr_routing
                    support_value = amr_support
                    confidence = amr_confidence
                    confidence_note = amr_confidence_note
                    step.update(amr_review_step)
                    step["prediction"] = reported
                    step["support"] = support_value
                    step["interpretation_confidence"] = confidence
                    step["confidence_note"] = confidence_note
                    row_update.update(amr_review_row)
                    row_update[f"predicted_{prefix}"] = reported
                    row_update[f"{prefix}_support"] = support_value
                    row_update[f"{prefix}_interpretation_confidence"] = confidence
                    row_update[f"{prefix}_confidence_note"] = confidence_note
            hierarchy_steps.append(step)
            row.update(row_update)

            terminal_label = reported
            terminal_level = level_number
            path_tokens.append(routing)
            children = current.get("children", {})
            next_node = children.get(routing) if isinstance(children, dict) else None
            if isinstance(next_node, dict):
                current = next_node
                continue

            terminal_reason = "no_child_node_for_predicted_label"
            if level_number < len(label_columns):
                (
                    used_fallback,
                    fb_label,
                    fb_level,
                    fb_status,
                    fb_reason,
                ) = self._apply_terminal_fallback_prediction(
                    sample_id=sample_id,
                    X_raw=X_raw,
                    raw_available_features=raw_available_features,
                    sample_feature_metadata=sample_feature_metadata,
                    hierarchy_steps=hierarchy_steps,
                    row=row,
                    label_columns=label_columns,
                    alignment_by_node=alignment_by_node,
                    max_markers=max_markers,
                    trigger_reason=terminal_reason,
                    support_policy=support_policy,
                )
                if used_fallback:
                    terminal_label = fb_label
                    terminal_level = fb_level
                    terminal_status = fb_status
                    terminal_reason = fb_reason
            break

        if guard >= 100:
            terminal_status = "stopped"
            terminal_reason = "hierarchy_traversal_guard_exceeded"

        predicted_path = " / ".join(
            f"{step.get('label_column')}={step.get('prediction')}"
            for step in hierarchy_steps
            if step.get("prediction") not in {None, ""}
        )
        row.update(
            {
                "predicted_hierarchy_path": predicted_path,
                "predicted_terminal_label": terminal_label,
                "predicted_terminal_level": terminal_level,
                "hierarchy_terminal_status": terminal_status,
                "hierarchy_terminal_reason": terminal_reason,
                "query_marker_recovery_status": sample_mapping_quality.get(
                    "marker_recovery_status"
                ),
                "query_marker_recovery_reason": sample_mapping_quality.get(
                    "marker_recovery_reason"
                ),
                "query_active_marker_evidence_status": sample_mapping_quality.get(
                    "active_marker_evidence_status"
                ),
                "query_active_marker_evidence_reason": sample_mapping_quality.get(
                    "active_marker_evidence_reason"
                ),
                "query_unique_mapped_fraction": sample_mapping_quality.get(
                    "unique_mapped_fraction"
                ),
                "query_active_feature_fraction": sample_mapping_quality.get(
                    "active_feature_fraction"
                ),
                "query_n_encoded_active_features": sample_mapping_quality.get(
                    "n_encoded_active_features"
                ),
                "query_n_resolved_features": sample_mapping_quality.get(
                    "n_resolved_features"
                ),
                "query_resolved_feature_fraction": sample_mapping_quality.get(
                    "resolved_feature_fraction"
                ),
                "query_n_resolved_baseline_features": sample_mapping_quality.get(
                    "n_resolved_baseline_features"
                ),
                "query_resolved_baseline_feature_fraction": sample_mapping_quality.get(
                    "resolved_baseline_feature_fraction"
                ),
                "query_resolved_marker_evidence_status": sample_mapping_quality.get(
                    "resolved_marker_evidence_status"
                ),
                "query_resolved_marker_evidence_reason": sample_mapping_quality.get(
                    "resolved_marker_evidence_reason"
                ),
                "query_n_unresolved_or_missing_calls": sample_mapping_quality.get(
                    "n_unresolved_or_missing_context_calls",
                    sample_mapping_quality.get("n_unresolved_or_missing_calls"),
                ),
                "query_n_multi_hit_calls": sample_mapping_quality.get(
                    "n_multi_hit_calls"
                ),
                "query_n_ambiguous_base_calls": sample_mapping_quality.get(
                    "n_ambiguous_base_calls"
                ),
                "query_n_non_training_allele_calls": sample_mapping_quality.get(
                    "n_non_training_allele_calls"
                ),
                "query_prediction_action": sample_mapping_quality.get(
                    "query_prediction_action", "predict"
                ),
                "query_abstention_reason_codes": list(
                    (sample_mapping_quality.get("callability_gate") or {}).get(
                        "abstention_reason_codes"
                    )
                    or []
                ),
                "query_callability_gate_status": (
                    sample_mapping_quality.get("callability_gate") or {}
                ).get("gate_status"),
                "query_feature_recovery_fraction": (
                    sample_mapping_quality.get("callability_gate") or {}
                ).get("feature_recovery_fraction"),
                "query_callable_fraction": (
                    sample_mapping_quality.get("callability_gate") or {}
                ).get("callable_fraction"),
                "query_call_state_counts": (
                    sample_mapping_quality.get("callability_gate") or {}
                ).get("call_state_counts", {}),
            }
        )
        report_sample = {
            **row,
            "hierarchy_steps": hierarchy_steps,
            "fasta_mapping_quality": sample_mapping_quality
            if query_input_type == "fasta"
            else {},
            "vcf_mapping_quality": sample_mapping_quality
            if query_input_type in {"vcf", "fastq"}
            else {},
            "raw_sequence_mapping_quality": sample_mapping_quality,
        }

        return row, report_sample, alignment_by_node

    def _query_hierarchy(
        self,
        *,
        X_raw: pd.DataFrame,
        raw_calls: Optional[pd.DataFrame],
        raw_mapping_summary: Optional[Dict[str, Any]],
        fastq_processing_summary: Optional[Dict[str, Any]],
        query_input_type: str,
        genomic_path: str,
        output_dir: Path,
        max_markers: int,
    ) -> pd.DataFrame:
        """Traverse an arbitrary-depth trained hierarchy without rerunning discovery."""
        output_dir = ensure_dir(Path(output_dir))
        hierarchy = (
            self.registry.get("hierarchy", {})
            if isinstance(self.registry, dict)
            else {}
        )
        root = hierarchy.get("root", {}) if isinstance(hierarchy, dict) else {}
        label_columns = (
            [str(x) for x in hierarchy.get("label_columns", [])]
            if isinstance(hierarchy, dict)
            else []
        )
        if not isinstance(root, dict) or not root:
            raise ValueError("Hierarchical registry is missing hierarchy.root.")

        raw_available_features = set(map(str, X_raw.columns))
        raw_feature_metadata = feature_call_metadata_by_sample(raw_calls)
        if raw_feature_metadata:
            raw_feature_metadata = {
                normalize_sample_id(str(sample_id)): feature_map
                for sample_id, feature_map in raw_feature_metadata.items()
            }

        raw_sample_quality: Dict[str, Dict[str, Any]] = {}
        if isinstance(raw_mapping_summary, dict):
            for item in raw_mapping_summary.get("per_sample", []) or []:
                if isinstance(item, dict) and item.get("sample_id") is not None:
                    raw_sample_quality[
                        normalize_sample_id(str(item.get("sample_id")))
                    ] = item
            gates = raw_mapping_summary.get("callability_gates") or {}
            if isinstance(gates, dict):
                for sid, gate in gates.items():
                    key = normalize_sample_id(str(sid))
                    payload = raw_sample_quality.setdefault(key, {"sample_id": key})
                    payload["callability_gate"] = gate
                    if gate.get("prediction_action") == "abstain_review_unresolved":
                        payload["query_prediction_action"] = "abstain_review_unresolved"
                        payload[
                            "marker_recovery_status"
                        ] = "insufficient_callability_abstain"
                        payload["marker_recovery_reason"] = (
                            "Feature recovery and/or callable fraction below configured gates; "
                            "prediction should be treated as review/unresolved."
                        )
                        logger.warning(
                            "Query sample %s abstains under callability gates | gate=%s",
                            key,
                            gate.get("gate_status"),
                        )

        sample_ids = [
            normalize_sample_id(str(sample_id)) for sample_id in X_raw.index.astype(str)
        ]
        support_policy = self._resolve_label_training_support_policy()

        def _dispatch_one(
            sample_id: str,
        ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
            return self._traverse_hierarchy_sample(
                sample_id=sample_id,
                X_raw=X_raw,
                root=root,
                label_columns=label_columns,
                raw_available_features=raw_available_features,
                sample_feature_metadata=raw_feature_metadata.get(sample_id, {}),
                sample_mapping_quality=raw_sample_quality.get(sample_id, {}),
                query_input_type=query_input_type,
                max_markers=max_markers,
                support_policy=support_policy,
            )

        parallel_samples = should_run_parallel(
            self.config,
            enabled_attr="query_parallel_samples",
            n_tasks=len(sample_ids),
        )
        if parallel_samples:
            n_jobs = resolve_effective_n_jobs(
                self.config,
                override=getattr(self.config, "query_parallel_n_jobs", None),
                minimum_tasks=len(sample_ids),
            )
            logger.info(
                "Running hierarchy query for %d samples in parallel | n_jobs=%d",
                len(sample_ids),
                int(n_jobs),
            )
            traversed = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_dispatch_one)(sample_id) for sample_id in sample_ids
            )
        else:
            traversed = [
                _dispatch_one(sample_id)
                for sample_id in progress_iter(
                    sample_ids,
                    desc="Hierarchy query samples",
                    unit="sample",
                    leave=False,
                )
            ]

        rows: List[Dict[str, Any]] = []
        report_samples: List[Dict[str, Any]] = []
        alignment_by_node: Dict[str, Any] = {}
        for row, report_sample, sample_alignment in traversed:
            rows.append(row)
            report_samples.append(report_sample)
            for key, value in sample_alignment.items():
                alignment_by_node.setdefault(key, value)

        predictions = pd.DataFrame(rows)
        predictions_path = output_dir / "query_predictions.csv"
        compact_path = output_dir / "query_predictions_compact.tsv"
        readable_path = output_dir / "query_predictions_readable.html"
        route_audit_path = output_dir / "query_route_audit.json"
        alignment_path = output_dir / "query_alignment_summary.json"

        predictions.to_csv(predictions_path, index=False)
        write_compact_predictions(predictions, compact_path)

        alignment_summary = {
            "mode": "multi_level_hierarchy_query_alignment",
            "alignment_by_node": alignment_by_node,
            "n_query_samples": int(X_raw.shape[0]),
            "n_query_features_raw": int(X_raw.shape[1]),
            "query_input_type": query_input_type,
            "fasta_mapping": raw_mapping_summary
            if query_input_type == "fasta"
            else None,
            "vcf_mapping": raw_mapping_summary
            if query_input_type in {"vcf", "fastq"}
            else None,
            "raw_sequence_mapping": raw_mapping_summary,
            "fastq_processing": fastq_processing_summary,
        }
        write_json(alignment_summary, alignment_path)

        route_audit = build_hierarchical_query_route_audit(
            registry_path=self.registry_path,
            query_input_type=query_input_type,
            report_samples=report_samples,
            alignment_by_node=alignment_by_node,
        )
        write_json(route_audit, route_audit_path)

        report = {
            "mode": "multi_level_hierarchy_query",
            "registry": str(self.registry_path),
            "genomic_input": str(genomic_path),
            "n_samples": int(len(report_samples)),
            "hierarchy_label_columns": label_columns,
            "samples": report_samples,
            "diagnostic_question": "Given this genomic evidence, where does the strain belong across the trained hierarchy, what phenotype is predicted at the terminal level, and which trained genomic markers support the interpretation?",
            "notes": [
                "Query mode is inference-only.",
                "Recursive hierarchy query traversal does not rerun central feature filtering, model selection, decision-tree training, or bootstrap confidence scoring.",
                "Each trainable hierarchy node aligns the query to that node's saved selected-feature space before prediction.",
                "Deterministic branches are followed when a training node had only one observed child label and therefore no model was fitted.",
                "For FASTA/VCF/FASTQ queries, NetworkParser reconstructs the saved trained feature space from the selected-feature manifest.",
                "Resolved baseline states encoded as 0 are valid trained-marker evidence for the overall query pattern.",
                "Unresolved, ambiguous, repeated, absent, or non-training allele calls remain NaN and contribute to callability abstention.",
            ],
            "artifacts": {
                "predictions_csv": str(predictions_path),
                "predictions_compact_tsv": str(compact_path),
                "predictions_readable_html": str(readable_path),
                "route_audit_json": str(route_audit_path),
                "alignment_summary_json": str(alignment_path),
                "report_json": str(output_dir / "query_report.json"),
                "report_txt": str(output_dir / "query_report.txt"),
                "fastq_processing_summary": (
                    str(
                        output_dir
                        / "fastq_query_preprocessing"
                        / "fastq_processing_summary.json"
                    )
                    if fastq_processing_summary is not None
                    else None
                ),
            },
        }
        write_json(report, output_dir / "query_report.json")
        write_hierarchical_text_report(report, output_dir / "query_report.txt")
        write_hierarchical_readable_html_report(report, readable_path)
        logger.info(
            "Hierarchy query complete | predictions=%s | compact=%s | readable=%s",
            predictions_path,
            compact_path,
            readable_path,
        )
        return predictions

    def _load_level1(
        self,
    ) -> Tuple[List[str], Any, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        level1 = self.registry.get("level1", {})
        features = [str(f) for f in level1.get("features", [])]
        model_path = resolve_path(level1.get("model_file"), self.registry_base)
        if not features:
            raise ValueError("Registry is missing Level 1 selected features.")
        if model_path is None or not model_path.exists():
            raise ValueError("Registry is missing a readable Level 1 model file.")
        payload = load_pickle(model_path)
        ranked = read_ranked_feature_table(level1.get("filter", {}), self.registry_base)
        model_importance = extract_model_importance(payload, features)
        return features, payload, ranked, model_importance

    def _select_level2_payload(
        self, predicted_level1: str
    ) -> Tuple[str, List[str], Any, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        cache_key = str(predicted_level1)
        with self._level2_payload_cache_lock:
            cached = self._level2_payload_cache.get(cache_key)
            if cached is not None:
                return cached

        level2 = self.registry.get("level2", {})
        by_group = level2.get("by_level1_group", {}) if isinstance(level2, dict) else {}
        group_payload = (
            by_group.get(str(predicted_level1), {})
            if isinstance(by_group, dict)
            else {}
        )

        source = "level1_group_specific"
        selected = group_payload
        if (
            not selected
            or selected.get("status") != "success"
            or not selected.get("model_file")
        ):
            selected = (
                level2.get("global_fallback", {}) if isinstance(level2, dict) else {}
            )
            source = "global_fallback"
            if (
                not selected
                or selected.get("status") != "success"
                or not selected.get("model_file")
            ):
                selected = (
                    level2.get("global_binary_fallback", {})
                    if isinstance(level2, dict)
                    else {}
                )
                source = "global_binary_fallback"

        features = [str(f) for f in selected.get("features", [])]
        model_path = resolve_path(selected.get("model_file"), self.registry_base)
        if not features or model_path is None or not model_path.exists():
            raise ValueError(
                "No usable Level 2 model found for predicted Level 1 group and no global fallback is available."
            )

        payload = load_pickle(model_path)
        ranked = read_ranked_feature_table(
            selected.get("filter", {}), self.registry_base
        )
        model_importance = extract_model_importance(payload, features)
        result = (source, features, payload, ranked, model_importance)
        with self._level2_payload_cache_lock:
            self._level2_payload_cache[cache_key] = result
        return result

    def query(
        self,
        genomic_path: str,
        output_dir: str,
        ref_fasta: Optional[str] = None,
        max_markers: int = 10,
        n_jobs: Optional[int] = None,
        query_input_type: str = "auto",
        raw_sequence_mapping_mode: str = "auto",
    ) -> pd.DataFrame:
        out = ensure_dir(Path(output_dir))
        query_input_type = str(query_input_type or "auto").lower()

        # Query mode must reconstruct the *trained* feature space.  For FASTA,
        # VCF, and FASTQ-derived VCF input we therefore use the selected-feature
        # manifest and the union of registry features, instead of allowing a
        # single-sample query to rediscover/filter/collapse its own feature set.
        genomic_candidate = Path(genomic_path)
        fasta_suffixes = {".fa", ".fna", ".fasta", ".fas"}
        fastq_suffixes = (".fastq", ".fq", ".fastq.gz", ".fq.gz")
        vcf_suffixes = (".vcf", ".vcf.gz")

        if query_input_type == "auto":
            if genomic_candidate.is_file():
                lower_name = genomic_candidate.name.lower()
                if genomic_candidate.suffix.lower() in fasta_suffixes:
                    query_input_type = "fasta"
                elif lower_name.endswith(vcf_suffixes):
                    query_input_type = "vcf"
            elif genomic_candidate.is_dir():
                files = [p for p in genomic_candidate.iterdir() if p.is_file()]
                names = [p.name.lower() for p in files]
                if any(
                    any(name.endswith(ext) for ext in fastq_suffixes) for name in names
                ):
                    query_input_type = "fastq"
                elif any(
                    any(name.endswith(ext) for ext in vcf_suffixes) for name in names
                ):
                    query_input_type = "vcf"
                elif any(p.suffix.lower() in fasta_suffixes for p in files):
                    query_input_type = "fasta"
            if query_input_type == "auto":
                query_input_type = "matrix"

        if query_input_type in {"raw_sequence", "raw_fasta", "sequence"}:
            logger.warning(
                "query_input_type=raw_sequence is deprecated; use query_input_type=fasta instead."
            )
            query_input_type = "fasta"

        raw_calls: Optional[pd.DataFrame] = None
        raw_mapping_summary: Optional[Dict[str, Any]] = None
        fastq_processing_summary: Optional[Dict[str, Any]] = None

        required_features = collect_required_features_from_registry(self.registry)
        manifest_path = resolve_registry_feature_manifest(
            self.registry, self.registry_base
        )

        def _require_feature_manifest(input_label: str) -> Path:
            if manifest_path is None:
                raise ValueError(
                    f"{input_label} query mode requires a selected-feature manifest in the model registry. "
                    "Retrain with a reference FASTA/GenBank so selected-feature context, REF/ALT, "
                    "and baseline allele metadata are saved."
                )
            manifest = load_feature_manifest(Path(manifest_path))
            context_columns = [
                col
                for col in (
                    "Context_sequence",
                    "Context_±40",
                    "Context",
                    "context_sequence",
                )
                if col in manifest.columns
            ]
            context_present = 0
            if context_columns:
                context_present = int(
                    manifest[context_columns]
                    .astype(str)
                    .apply(lambda row: any(value.strip() for value in row), axis=1)
                    .sum()
                )
            logger.info(
                "Loaded selected-feature manifest | features=%d | context_present=%d",
                len(manifest),
                context_present,
            )
            return manifest_path

        if query_input_type == "fasta":
            resolved_manifest = _require_feature_manifest("FASTA")
            X_raw, raw_mapping_summary, raw_calls = encode_raw_sequence_query(
                raw_sequence_path=genomic_path,
                feature_manifest_path=str(resolved_manifest),
                features=required_features,
                output_dir=str(out / "fasta_query_encoding"),
                mapping_mode=raw_sequence_mapping_mode,
            )
            X_raw.index = X_raw.index.astype(str).map(normalize_sample_id)

        elif query_input_type == "vcf":
            resolved_manifest = _require_feature_manifest("VCF")
            X_raw, raw_mapping_summary, raw_calls = encode_vcf_query_from_manifest(
                vcf_path=genomic_path,
                feature_manifest_path=str(resolved_manifest),
                features=required_features,
                output_dir=str(out / "vcf_query_encoding"),
                config=self.config,
            )
            X_raw.index = X_raw.index.astype(str).map(normalize_sample_id)

        elif query_input_type == "fastq":
            if not ref_fasta:
                raise ValueError(
                    "FASTQ query mode requires --ref_fasta because reads must be aligned "
                    "and converted to VCF-derived genomic features before inference."
                )
            resolved_manifest = _require_feature_manifest("FASTQ")
            fastq_out = ensure_dir(out / "fastq_query_preprocessing")
            # Pass trained marker IDs so panel_* call modes only pileup/call those sites.
            panel_ids: Optional[List[str]] = None
            call_mode = str(
                getattr(self.config, "fastq_call_mode", "full") or "full"
            ).strip().lower()
            if call_mode in {"panel_bcftools", "panel_majority"}:
                panel_ids = [str(f) for f in list(required_features)]
                logger.info(
                    "FASTQ panel call mode=%s | injecting %d trained feature IDs",
                    call_mode,
                    len(panel_ids),
                )
            processor = FastqProcessor(
                config=self.config,
                fastq_dir=genomic_path,
                ref_genome=ref_fasta,
                output_dir=str(fastq_out),
                n_jobs=n_jobs,
                panel_feature_ids=panel_ids,
            )
            vcf_dir, fastq_summary = processor.process_samples()
            fastq_processing_summary = asdict(fastq_summary)
            X_raw, raw_mapping_summary, raw_calls = encode_vcf_query_from_manifest(
                vcf_path=str(vcf_dir),
                feature_manifest_path=str(resolved_manifest),
                features=required_features,
                output_dir=str(out / "vcf_query_encoding"),
                config=self.config,
            )
            X_raw.index = X_raw.index.astype(str).map(normalize_sample_id)

        elif query_input_type == "matrix":
            X_raw = load_query_matrix(
                genomic_path=genomic_path,
                output_dir=out,
                config=self.config,
                ref_fasta=ref_fasta,
                n_jobs=n_jobs,
            )
        else:
            raise ValueError(
                "query_input_type must be one of: auto, matrix, vcf, fasta, fastq"
            )

        raw_available_features = set(map(str, X_raw.columns))

        raw_feature_metadata = feature_call_metadata_by_sample(raw_calls)
        if raw_feature_metadata:
            raw_feature_metadata = {
                normalize_sample_id(str(sample_id)): feature_map
                for sample_id, feature_map in raw_feature_metadata.items()
            }

        # Authoritative execution gate for every input mode. Encoder-provided
        # VCF/gVCF summaries remain audit evidence, while both bundled and
        # unbundled inference enforce this model-feature calculation.
        global_query_gates = self._gates_for_features(
            X_raw=X_raw,
            features=required_features,
            feature_metadata_by_sample=raw_feature_metadata,
        )
        if not isinstance(raw_mapping_summary, dict):
            raw_mapping_summary = {}
        raw_mapping_summary["callability_gates"] = global_query_gates
        raw_mapping_summary["samples_requiring_abstention"] = [
            sample_id
            for sample_id, gate in global_query_gates.items()
            if gate.get("prediction_action") != "predict"
        ]
        raw_sample_quality: Dict[str, Dict[str, Any]] = {}
        if isinstance(raw_mapping_summary, dict):
            for item in raw_mapping_summary.get("per_sample", []) or []:
                if isinstance(item, dict) and item.get("sample_id") is not None:
                    raw_sample_quality[
                        normalize_sample_id(str(item.get("sample_id")))
                    ] = item
            gates = raw_mapping_summary.get("callability_gates") or {}
            if isinstance(gates, dict):
                for sid, gate in gates.items():
                    key = normalize_sample_id(str(sid))
                    payload = raw_sample_quality.setdefault(key, {"sample_id": key})
                    payload["callability_gate"] = gate
                    if gate.get("prediction_action") == "abstain_review_unresolved":
                        payload["query_prediction_action"] = "abstain_review_unresolved"
                        payload[
                            "marker_recovery_status"
                        ] = "insufficient_callability_abstain"
                        payload["marker_recovery_reason"] = (
                            "Feature recovery and/or callable fraction below configured gates; "
                            "prediction should be treated as review/unresolved."
                        )
                        logger.warning(
                            "Query sample %s abstains under callability gates | gate=%s",
                            key,
                            gate.get("gate_status"),
                        )

        if is_hierarchical_registry(self.registry):
            logger.info(
                "Detected multi-level hierarchy registry; using recursive hierarchy query traversal."
            )
            return self._query_hierarchy(
                X_raw=X_raw,
                raw_calls=raw_calls,
                raw_mapping_summary=raw_mapping_summary,
                fastq_processing_summary=fastq_processing_summary,
                query_input_type=query_input_type,
                genomic_path=genomic_path,
                output_dir=out,
                max_markers=max_markers,
            )

        (
            level1_features,
            level1_payload,
            level1_ranked,
            level1_importance,
        ) = self._load_level1()
        X_l1, l1_alignment = align_to_training_features(X_raw, level1_features)
        l1_gates = self._gates_for_features(
            X_raw=X_raw,
            features=level1_features,
            feature_metadata_by_sample=raw_feature_metadata,
        )
        l1_alignment["callability_gates"] = l1_gates
        l1_pred, l1_support, l1_class_support = predict_labels_and_support(
            level1_payload,
            X_l1,
            gate_results=l1_gates,
        )
        l1_tree_paths = decision_tree_path_explanation(level1_payload, X_l1, l1_gates)
        support_policy = self._resolve_label_training_support_policy()
        level1_label_column = str(
            self.registry.get("level1", {}).get("label_column", "level1")
        ).strip()

        sample_ids = list(X_l1.index.astype(str))
        groups: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        for idx, sample_id in enumerate(sample_ids):
            if (l1_gates.get(sample_id) or {}).get("prediction_action") == "predict":
                groups[str(l1_pred[idx])].append((idx, sample_id))

        rows_by_sample_id: Dict[str, Dict[str, Any]] = {}
        report_by_sample_id: Dict[str, Dict[str, Any]] = {}
        alignment_by_level2_source: Dict[str, Any] = {}

        # A failed Level-1 gate terminates classic two-level routing immediately;
        # no Level-2 payload is loaded and no explanation path is fabricated.
        for idx, sample_id in enumerate(sample_ids):
            l1_gate = l1_gates.get(sample_id, {})
            if l1_gate.get("prediction_action") == "predict":
                continue
            reason_codes = list(l1_gate.get("abstention_reason_codes") or [])
            sample_mapping_quality = raw_sample_quality.get(sample_id, {})
            row = {
                "sample_id": sample_id,
                "predicted_level1_identity": "unavailable",
                "level1_support": None,
                "predicted_level2_identity": "unavailable",
                "level2_support": None,
                "level2_model_source": "not_routed_callability_abstention",
                "n_level1_supporting_markers": 0,
                "n_level2_supporting_markers": 0,
                "level1_interpretation_confidence": "unavailable",
                "level1_confidence_note": "Required Level-1 markers did not pass callability gates.",
                "level2_interpretation_confidence": "unavailable",
                "level2_confidence_note": "Level-2 routing was not attempted.",
                "query_marker_recovery_status": sample_mapping_quality.get(
                    "marker_recovery_status"
                ),
                "query_marker_recovery_reason": sample_mapping_quality.get(
                    "marker_recovery_reason"
                ),
                "query_prediction_action": "abstain_review_unresolved",
                "query_abstention_reason_codes": reason_codes,
                "query_callability_gate_status": l1_gate.get("gate_status"),
                "query_feature_recovery_fraction": l1_gate.get(
                    "feature_recovery_fraction"
                ),
                "query_callable_fraction": l1_gate.get("callable_fraction"),
                "query_call_state_counts": l1_gate.get("call_state_counts", {}),
            }
            rows_by_sample_id[sample_id] = row
            report_by_sample_id[sample_id] = {
                **row,
                "level1_class_support": {},
                "level2_class_support": {},
                "level1_feature_evidence": {},
                "level2_feature_evidence": {},
                "level1_supporting_markers": [],
                "level2_supporting_markers": [],
                "level1_decision_path": [],
                "level2_decision_path": [],
                "level1_callability_gate": l1_gate,
                "level2_callability_gate": {},
                "raw_sequence_mapping_quality": sample_mapping_quality,
            }

        logger.info(
            "Running hierarchy query with batched Level-2 inference | samples=%d | level1_groups=%d",
            len(sample_ids),
            len(groups),
        )

        for predicted_l1, members in groups.items():
            (
                level2_source,
                l2_features,
                l2_payload,
                l2_ranked,
                l2_importance,
            ) = self._select_level2_payload(predicted_l1)
            member_sample_ids = [sample_id for _, sample_id in members]
            X_l2, l2_alignment = align_to_training_features(
                X_raw.loc[member_sample_ids], l2_features
            )
            l2_gates = self._gates_for_features(
                X_raw=X_raw.loc[member_sample_ids],
                features=l2_features,
                feature_metadata_by_sample={
                    sample_id: raw_feature_metadata.get(sample_id, {})
                    for sample_id in member_sample_ids
                },
            )
            l2_alignment["callability_gates"] = l2_gates
            alignment_by_level2_source.setdefault(level2_source, l2_alignment)

            l2_pred, l2_support, l2_class_support = predict_labels_and_support(
                l2_payload,
                X_l2,
                gate_results=l2_gates,
            )
            l2_tree_paths = decision_tree_path_explanation(l2_payload, X_l2, l2_gates)
            level2_target_label_column = (
                self.registry.get("level2", {}).get("global_label_column")
                if level2_source == "global_fallback"
                else self.registry.get("level2", {}).get("label_column")
            )

            for local_idx, (global_idx, sample_id) in enumerate(members):
                sample_feature_metadata = raw_feature_metadata.get(sample_id, {})
                sample_mapping_quality = raw_sample_quality.get(sample_id, {})
                l1_gate = l1_gates.get(sample_id, {})
                l2_gate = l2_gates.get(sample_id, {})
                failed_gate = (
                    l1_gate
                    if l1_gate.get("prediction_action") != "predict"
                    else (
                        l2_gate if l2_gate.get("prediction_action") != "predict" else {}
                    )
                )
                l1_markers = supporting_markers_for_sample(
                    X_l1.loc[sample_id],
                    ranked_features=level1_ranked,
                    model_importance=level1_importance,
                    max_markers=max_markers,
                    feature_metadata=sample_feature_metadata,
                    available_features=raw_available_features,
                )
                l2_markers = supporting_markers_for_sample(
                    X_l2.loc[sample_id],
                    ranked_features=l2_ranked,
                    model_importance=l2_importance,
                    max_markers=max_markers,
                    feature_metadata=sample_feature_metadata,
                    available_features=raw_available_features,
                )

                l1_evidence = summarize_feature_evidence_for_model(
                    sample_values=X_l1.loc[sample_id],
                    features=level1_features,
                    feature_metadata=sample_feature_metadata,
                    available_features=raw_available_features,
                )
                l2_evidence = summarize_feature_evidence_for_model(
                    sample_values=X_l2.loc[sample_id],
                    features=l2_features,
                    feature_metadata=sample_feature_metadata,
                    available_features=raw_available_features,
                )

                l1_confidence, l1_confidence_note = interpretation_confidence_for_level(
                    support=l1_support[global_idx],
                    evidence=l1_evidence,
                    n_supporting_markers=len(l1_markers),
                )
                l2_confidence, l2_confidence_note = interpretation_confidence_for_level(
                    support=l2_support[local_idx],
                    evidence=l2_evidence,
                    n_supporting_markers=len(l2_markers),
                )

                (
                    l1_reported,
                    l1_routing,
                    l1_review_row,
                    _l1_review_step,
                ) = self._low_support_review_bundle(
                    label_column=level1_label_column,
                    prediction=str(predicted_l1),
                    prefix="level1_identity",
                    support_policy=support_policy,
                )
                if l1_review_row:
                    l1_confidence = l1_review_row.get(
                        "level1_identity_interpretation_confidence", l1_confidence
                    )
                    l1_confidence_note = l1_review_row.get(
                        "level1_identity_confidence_note", l1_confidence_note
                    )

                l2_candidate = str(l2_pred[local_idx])
                l2_label_column = str(
                    level2_target_label_column
                    or self.registry.get("level2", {}).get("label_column", "level2")
                ).strip()
                (
                    l2_reported,
                    _l2_routing,
                    l2_review_row,
                    _l2_review_step,
                ) = self._low_support_review_bundle(
                    label_column=l2_label_column,
                    prediction=l2_candidate,
                    prefix="level2_identity",
                    support_policy=support_policy,
                )
                if l2_review_row:
                    l2_confidence = l2_review_row.get(
                        "level2_identity_interpretation_confidence", l2_confidence
                    )
                    l2_confidence_note = l2_review_row.get(
                        "level2_identity_confidence_note", l2_confidence_note
                    )

                row = {
                    "sample_id": sample_id,
                    "predicted_level1_identity": l1_reported,
                    "level1_support": l1_support[global_idx],
                    "predicted_level2_identity": l2_reported,
                    "level2_support": l2_support[local_idx],
                    "level2_model_source": level2_source,
                    "level2_target_label_column": level2_target_label_column,
                    "n_level1_supporting_markers": int(len(l1_markers)),
                    "n_level2_supporting_markers": int(len(l2_markers)),
                    "level1_interpretation_confidence": l1_confidence,
                    "level1_confidence_note": l1_confidence_note,
                    "level2_interpretation_confidence": l2_confidence,
                    "level2_confidence_note": l2_confidence_note,
                    **flatten_feature_evidence("level1", l1_evidence),
                    **flatten_feature_evidence("level2", l2_evidence),
                    "query_marker_recovery_status": sample_mapping_quality.get(
                        "marker_recovery_status"
                    ),
                    "query_marker_recovery_reason": sample_mapping_quality.get(
                        "marker_recovery_reason"
                    ),
                    "query_active_marker_evidence_status": sample_mapping_quality.get(
                        "active_marker_evidence_status"
                    ),
                    "query_active_marker_evidence_reason": sample_mapping_quality.get(
                        "active_marker_evidence_reason"
                    ),
                    "query_unique_mapped_fraction": sample_mapping_quality.get(
                        "unique_mapped_fraction"
                    ),
                    "query_active_feature_fraction": sample_mapping_quality.get(
                        "active_feature_fraction"
                    ),
                    "query_n_encoded_active_features": sample_mapping_quality.get(
                        "n_encoded_active_features"
                    ),
                    "query_n_resolved_features": sample_mapping_quality.get(
                        "n_resolved_features"
                    ),
                    "query_resolved_feature_fraction": sample_mapping_quality.get(
                        "resolved_feature_fraction"
                    ),
                    "query_n_resolved_baseline_features": sample_mapping_quality.get(
                        "n_resolved_baseline_features"
                    ),
                    "query_resolved_baseline_feature_fraction": sample_mapping_quality.get(
                        "resolved_baseline_feature_fraction"
                    ),
                    "query_resolved_marker_evidence_status": sample_mapping_quality.get(
                        "resolved_marker_evidence_status"
                    ),
                    "query_resolved_marker_evidence_reason": sample_mapping_quality.get(
                        "resolved_marker_evidence_reason"
                    ),
                    "query_n_unresolved_or_missing_calls": sample_mapping_quality.get(
                        "n_unresolved_or_missing_context_calls",
                        sample_mapping_quality.get("n_unresolved_or_missing_calls"),
                    ),
                    "query_n_multi_hit_calls": sample_mapping_quality.get(
                        "n_multi_hit_calls"
                    ),
                    "query_n_ambiguous_base_calls": sample_mapping_quality.get(
                        "n_ambiguous_base_calls"
                    ),
                    "query_n_non_training_allele_calls": sample_mapping_quality.get(
                        "n_non_training_allele_calls"
                    ),
                    "query_prediction_action": (
                        "abstain_review_unresolved" if failed_gate else "predict"
                    ),
                    "query_abstention_reason_codes": list(
                        failed_gate.get("abstention_reason_codes") or []
                    ),
                    "query_callability_gate_status": failed_gate.get("gate_status"),
                    "query_feature_recovery_fraction": failed_gate.get(
                        "feature_recovery_fraction"
                    ),
                    "query_callable_fraction": failed_gate.get("callable_fraction"),
                    "query_call_state_counts": failed_gate.get("call_state_counts", {}),
                    **l1_review_row,
                    **l2_review_row,
                }
                rows_by_sample_id[sample_id] = row
                report_by_sample_id[sample_id] = {
                    **row,
                    "level1_routing_identity": l1_routing,
                    "level2_routing_identity": _l2_routing,
                    "level1_class_support": l1_class_support[global_idx],
                    "level2_class_support": l2_class_support[local_idx],
                    "level1_feature_evidence": l1_evidence,
                    "level2_feature_evidence": l2_evidence,
                    "level1_supporting_markers": l1_markers,
                    "level2_supporting_markers": l2_markers,
                    "fasta_mapping_quality": sample_mapping_quality
                    if query_input_type == "fasta"
                    else {},
                    "vcf_mapping_quality": sample_mapping_quality
                    if query_input_type in {"vcf", "fastq"}
                    else {},
                    "raw_sequence_mapping_quality": sample_mapping_quality,
                    "level1_decision_path": l1_tree_paths.get(sample_id, []),
                    "level2_decision_path": l2_tree_paths.get(sample_id, []),
                    "level1_callability_gate": l1_gate,
                    "level2_callability_gate": l2_gate,
                }

        rows = [rows_by_sample_id[sample_id] for sample_id in sample_ids]
        report_samples = [report_by_sample_id[sample_id] for sample_id in sample_ids]

        predictions = pd.DataFrame(rows)
        predictions_path = out / "query_predictions.csv"
        compact_path = out / "query_predictions_compact.tsv"
        readable_path = out / "query_predictions_readable.html"
        route_audit_path = out / "query_route_audit.json"

        predictions.to_csv(predictions_path, index=False)
        write_compact_predictions(predictions, compact_path)

        alignment_summary = {
            "level1": l1_alignment,
            "level2_by_source": alignment_by_level2_source,
            "n_query_samples": int(X_raw.shape[0]),
            "n_query_features_raw": int(X_raw.shape[1]),
            "query_input_type": query_input_type,
            "fasta_mapping": raw_mapping_summary
            if query_input_type == "fasta"
            else None,
            "vcf_mapping": raw_mapping_summary
            if query_input_type in {"vcf", "fastq"}
            else None,
            "raw_sequence_mapping": raw_mapping_summary,
            "fastq_processing": fastq_processing_summary,
        }
        write_json(alignment_summary, out / "query_alignment_summary.json")

        route_audit = build_query_route_audit(
            registry_path=self.registry_path,
            query_input_type=query_input_type,
            report_samples=report_samples,
            l1_alignment=l1_alignment,
            alignment_by_level2_source=alignment_by_level2_source,
        )
        write_json(route_audit, route_audit_path)

        report = {
            "mode": "hierarchy_query",
            "registry": str(self.registry_path),
            "genomic_input": str(genomic_path),
            "n_samples": int(len(report_samples)),
            "samples": report_samples,
            "diagnostic_question": "Given this genomic evidence, where does the strain belong, what phenotype is predicted, and which trained genomic markers support the interpretation?",
            "notes": [
                "Query mode is inference-only.",
                "RF-FDR feature selection is not rerun on query samples.",
                "For FASTA/VCF/FASTQ queries, NetworkParser reconstructs the saved trained feature space from the selected-feature manifest.",
                "Query mode does not rerun cohort-level matrix refinement, redundancy reduction, RF-FDR, model selection, or tree construction.",
                "For FASTA queries, saved context sequences are mapped to the query genome and the centre nucleotide is encoded with the saved baseline/REF/ALT rule.",
                "For VCF queries, saved feature coordinates are looked up by contig and position; absence from a variants-only VCF remains unknown unless the explicit legacy absence-as-reference option is enabled.",
                "Resolved baseline states encoded as 0 are valid trained-marker evidence for the overall query pattern.",
                "Unresolved, ambiguous, repeated, absent, or non-training allele calls remain NaN and contribute to callability abstention.",
            ],
            "artifacts": {
                "predictions_csv": str(predictions_path),
                "predictions_compact_tsv": str(compact_path),
                "predictions_readable_html": str(readable_path),
                "route_audit_json": str(route_audit_path),
                "alignment_summary_json": str(out / "query_alignment_summary.json"),
                "report_json": str(out / "query_report.json"),
                "report_txt": str(out / "query_report.txt"),
                "fastq_processing_summary": (
                    str(
                        out
                        / "fastq_query_preprocessing"
                        / "fastq_processing_summary.json"
                    )
                    if fastq_processing_summary is not None
                    else None
                ),
            },
        }
        write_json(report, out / "query_report.json")
        write_text_report(report, out / "query_report.txt")
        write_readable_html_report(report, readable_path)

        logger.info(
            "Query complete | predictions=%s | compact=%s | readable=%s",
            predictions_path,
            compact_path,
            readable_path,
        )
        return predictions


def write_compact_predictions(predictions: pd.DataFrame, path: Path) -> None:
    """Write a terminal-friendly compact TSV with key prediction/evidence fields."""
    preferred = [
        "sample_id",
        "predicted_hierarchy_path",
        "predicted_terminal_label",
        "predicted_terminal_level",
        "hierarchy_terminal_status",
        "hierarchy_terminal_reason",
        "predicted_level1",
        "level1_label_column",
        "level1_prediction_status",
        "level1_candidate_prediction",
        "level1_recommended_action",
        "level1_low_support_reason",
        "level1_support",
        "level1_interpretation_confidence",
        "level1_n_supporting_markers",
        "level1_n_resolved_features",
        "level1_resolved_feature_fraction",
        "level1_n_resolved_baseline_features",
        "level1_n_active_features",
        "level1_resolved_marker_evidence_status",
        "level1_nonbaseline_evidence_status",
        "predicted_level2",
        "level2_label_column",
        "level2_prediction_status",
        "level2_candidate_prediction",
        "level2_recommended_action",
        "level2_low_support_reason",
        "level3_prediction_status",
        "level3_candidate_prediction",
        "level3_amr_evidence_reason",
        "level3_recommended_action",
        "level3_fallback_resistant_probability",
        "level2_identity_prediction_status",
        "level2_identity_candidate_prediction",
        "level2_identity_recommended_action",
        "level2_identity_low_support_reason",
        "predicted_level1_identity",
        "level1_support",
        "level1_interpretation_confidence",
        "level1_n_supporting_markers",
        "n_level1_supporting_markers",
        "level1_n_resolved_features",
        "level1_resolved_feature_fraction",
        "level1_n_resolved_baseline_features",
        "level1_n_active_features",
        "level1_resolved_marker_evidence_status",
        "level1_nonbaseline_evidence_status",
        "level1_prediction_status",
        "level1_candidate_prediction",
        "level1_recommended_action",
        "level1_low_support_reason",
        "level1_identity_prediction_status",
        "level1_identity_candidate_prediction",
        "level1_identity_recommended_action",
        "level1_identity_low_support_reason",
        "predicted_level2_identity",
        "level2_support",
        "level2_model_source",
        "level2_interpretation_confidence",
        "n_level2_supporting_markers",
        "level2_n_resolved_features",
        "level2_resolved_feature_fraction",
        "level2_n_resolved_baseline_features",
        "level2_n_active_features",
        "level2_resolved_marker_evidence_status",
        "level2_nonbaseline_evidence_status",
        "query_marker_recovery_status",
        "query_resolved_marker_evidence_status",
        "query_n_unresolved_or_missing_calls",
        "query_n_multi_hit_calls",
        "query_n_ambiguous_base_calls",
        "query_n_non_training_allele_calls",
    ]
    cols = [col for col in preferred if col in predictions.columns]
    if not cols:
        cols = list(predictions.columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.loc[:, cols].to_csv(path, sep="\t", index=False)


def build_query_route_audit(
    *,
    registry_path: Path,
    query_input_type: str,
    report_samples: List[Dict[str, Any]],
    l1_alignment: Dict[str, Any],
    alignment_by_level2_source: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a compact audit of how each sample moved through query inference."""
    routes: List[Dict[str, Any]] = []
    for sample in report_samples:
        routes.append(
            {
                "sample_id": sample.get("sample_id"),
                "predicted_level1_identity": sample.get("predicted_level1_identity"),
                "level1_support": sample.get("level1_support"),
                "level1_interpretation_confidence": sample.get(
                    "level1_interpretation_confidence"
                ),
                "level1_resolved_marker_evidence_status": sample.get(
                    "level1_resolved_marker_evidence_status"
                ),
                "level1_nonbaseline_evidence_status": sample.get(
                    "level1_nonbaseline_evidence_status"
                ),
                "predicted_level2_identity": sample.get("predicted_level2_identity"),
                "level2_support": sample.get("level2_support"),
                "level2_model_source": sample.get("level2_model_source"),
                "level2_target_label_column": sample.get("level2_target_label_column"),
                "level2_interpretation_confidence": sample.get(
                    "level2_interpretation_confidence"
                ),
                "level2_resolved_marker_evidence_status": sample.get(
                    "level2_resolved_marker_evidence_status"
                ),
                "level2_nonbaseline_evidence_status": sample.get(
                    "level2_nonbaseline_evidence_status"
                ),
                "query_marker_recovery_status": sample.get(
                    "query_marker_recovery_status"
                ),
                "query_resolved_marker_evidence_status": sample.get(
                    "query_resolved_marker_evidence_status"
                ),
            }
        )

    return {
        "mode": "hierarchy_query_route_audit",
        "registry": str(registry_path),
        "query_input_type": query_input_type,
        "n_samples": int(len(report_samples)),
        "level1_alignment_status": l1_alignment.get("alignment_status")
        if isinstance(l1_alignment, dict)
        else None,
        "level2_alignment_by_source": alignment_by_level2_source,
        "routes": routes,
    }


def build_hierarchical_query_route_audit(
    *,
    registry_path: Path,
    query_input_type: str,
    report_samples: List[Dict[str, Any]],
    alignment_by_node: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a compact audit of recursive hierarchy traversal."""
    routes: List[Dict[str, Any]] = []
    for sample in report_samples:
        routes.append(
            {
                "sample_id": sample.get("sample_id"),
                "predicted_hierarchy_path": sample.get("predicted_hierarchy_path"),
                "predicted_terminal_label": sample.get("predicted_terminal_label"),
                "predicted_terminal_level": sample.get("predicted_terminal_level"),
                "hierarchy_terminal_status": sample.get("hierarchy_terminal_status"),
                "hierarchy_terminal_reason": sample.get("hierarchy_terminal_reason"),
                "steps": [
                    {
                        "level_number": step.get("level_number"),
                        "label_column": step.get("label_column"),
                        "prediction": step.get("prediction"),
                        "support": step.get("support"),
                        "node_status": step.get("node_status"),
                        "node_key": step.get("node_key"),
                        "interpretation_confidence": step.get(
                            "interpretation_confidence"
                        ),
                        "resolved_marker_evidence_status": (
                            step.get("feature_evidence") or {}
                        ).get("resolved_marker_evidence_status"),
                        "nonbaseline_evidence_status": (
                            step.get("feature_evidence") or {}
                        ).get("nonbaseline_evidence_status"),
                    }
                    for step in sample.get("hierarchy_steps", []) or []
                ],
                "query_marker_recovery_status": sample.get(
                    "query_marker_recovery_status"
                ),
                "query_resolved_marker_evidence_status": sample.get(
                    "query_resolved_marker_evidence_status"
                ),
            }
        )

    return {
        "mode": "multi_level_hierarchy_query_route_audit",
        "registry": str(registry_path),
        "query_input_type": query_input_type,
        "n_samples": int(len(report_samples)),
        "alignment_by_node": alignment_by_node,
        "routes": routes,
    }


def _hierarchy_marker_lines(markers: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for marker in markers[:10]:
        role = marker.get("evidence_role") or "NA"
        extra = f" | role={role}"
        if marker.get("observed_allele"):
            quality = marker.get("mapping_quality") or marker.get("allele_call") or ""
            quality_txt = f" | quality={quality}" if quality else ""
            extra += f" | observed={marker.get('observed_allele')} | status={marker.get('mapping_status', '')}{quality_txt}"
        lines.append(f"    - {marker.get('feature')} = {marker.get('value')}{extra}")
    return lines


def write_hierarchical_text_report(report: Dict[str, Any], path: Path) -> None:
    """Write a readable text report for arbitrary-depth hierarchy query."""
    lines: List[str] = []
    lines.append("NetworkParser multi-level hierarchy query report")
    lines.append("=" * 54)
    lines.append(f"Samples queried: {report.get('n_samples', 0)}")
    if report.get("hierarchy_label_columns"):
        lines.append(
            f"Hierarchy labels: {', '.join(map(str, report.get('hierarchy_label_columns', [])))}"
        )
    if report.get("diagnostic_question"):
        lines.append("")
        lines.append("Diagnostic question")
        lines.append("-------------------")
        lines.append(str(report.get("diagnostic_question")))

    for sample in report.get("samples", []) or []:
        lines.append("")
        lines.append(f"Sample: {sample.get('sample_id')}")
        lines.append("-" * (8 + len(str(sample.get("sample_id", "")))))
        if sample.get("raw_sequence_mapping_quality"):
            rq = sample.get("raw_sequence_mapping_quality") or {}
            lines.append("Query trained-feature recovery")
            lines.append(f"  Marker recovery: {rq.get('marker_recovery_status', 'NA')}")
            if rq.get("resolved_marker_evidence_status"):
                lines.append(
                    f"  Resolved marker pattern: {rq.get('resolved_marker_evidence_status', 'NA')}"
                )
            lines.append(
                f"  Nonbaseline marker evidence: {rq.get('active_marker_evidence_status', 'NA')}"
            )
            lines.append(
                f"  Resolved trained-marker calls: {rq.get('n_resolved_features', 0)}"
            )
            lines.append(
                f"  Active encoded calls: {rq.get('n_encoded_active_features', 0)}"
            )
            lines.append("")

        lines.append(
            f"Predicted hierarchy path: {sample.get('predicted_hierarchy_path', 'NA')}"
        )
        lines.append(f"Terminal label: {sample.get('predicted_terminal_label', 'NA')}")
        lines.append(
            f"Terminal status: {sample.get('hierarchy_terminal_status', 'NA')} ({sample.get('hierarchy_terminal_reason', 'NA')})"
        )

        for step in sample.get("hierarchy_steps", []) or []:
            lines.append("")
            lines.append(
                f"Level {step.get('level_number')} — {step.get('label_column')}"
            )
            lines.append(f"  Prediction: {step.get('prediction')}")
            if step.get("support") is not None:
                try:
                    lines.append(f"  Support: {float(step.get('support')):.4f}")
                except Exception:
                    lines.append(f"  Support: {step.get('support')}")
            if step.get("interpretation_confidence"):
                lines.append(
                    f"  Interpretation confidence: {step.get('interpretation_confidence')}"
                )
            if step.get("confidence_note"):
                lines.append(f"  Confidence note: {step.get('confidence_note')}")
            evidence = step.get("feature_evidence") or {}
            if evidence:
                lines.append("  Level-specific marker evidence:")
                lines.append(
                    f"    Selected features: {evidence.get('n_selected_features', 'NA')}"
                )
                lines.append(
                    f"    Nonbaseline features: {evidence.get('n_active_features', 'NA')}"
                )
                lines.append(
                    f"    Resolved trained-marker states: {evidence.get('n_resolved_features', 'NA')}"
                )
                lines.append(
                    f"    Resolved baseline states: {evidence.get('n_resolved_baseline_features', 'NA')}"
                )
                if evidence.get("resolved_feature_fraction") is not None:
                    lines.append(
                        f"    Resolved feature fraction: {float(evidence.get('resolved_feature_fraction')):.4f}"
                    )
                lines.append(
                    f"    Marker recovery: {evidence.get('marker_recovery_status', 'NA')}"
                )
                lines.append(
                    f"    Resolved marker pattern: {evidence.get('resolved_marker_evidence_status', 'NA')}"
                )
            if step.get("supporting_markers"):
                lines.append("  Supporting markers:")
                lines.extend(
                    _hierarchy_marker_lines(step.get("supporting_markers", []))
                )
            if step.get("decision_path"):
                lines.append("  Decision path:")
                for rule in step.get("decision_path", []):
                    lines.append(f"    - {rule}")

    lines.append("")
    lines.append("Notes")
    lines.append("-----")
    for note in report.get("notes", []):
        lines.append(f"- {note}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_hierarchical_readable_html_report(report: Dict[str, Any], path: Path) -> None:
    """Write a browser-readable report for arbitrary-depth hierarchy query."""

    def _marker_table(markers: List[Dict[str, Any]]) -> str:
        rows: List[str] = []
        for marker in markers[:10]:
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(marker.get('feature', '')))}</td>"
                f"<td>{html.escape(str(marker.get('value', '')))}</td>"
                f"<td>{html.escape(str(marker.get('evidence_role', '')))}</td>"
                f"<td>{html.escape(str(marker.get('allele_call', '')))}</td>"
                f"<td>{html.escape(str(marker.get('observed_allele', '')))}</td>"
                "</tr>"
            )
        if not rows:
            return ""
        return (
            "<table><thead><tr><th>Feature</th><th>Value</th><th>Evidence role</th>"
            "<th>Allele call</th><th>Observed</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

    cards: List[str] = []
    for sample in report.get("samples", []) or []:
        sid = html.escape(str(sample.get("sample_id", "NA")))
        steps_html: List[str] = []
        for step in sample.get("hierarchy_steps", []) or []:
            level = html.escape(str(step.get("level_number", "NA")))
            label_col = html.escape(str(step.get("label_column", "NA")))
            pred = html.escape(str(step.get("prediction", "NA")))
            support = step.get("support")
            support_text = (
                "NA"
                if support is None
                else html.escape(str(round(float(support), 4)))
                if isinstance(support, (int, float))
                else html.escape(str(support))
            )
            evidence = step.get("feature_evidence") or {}
            markers_html = _marker_table(step.get("supporting_markers", []) or [])
            steps_html.append(
                "<div class='step'>"
                f"<h3>Level {level}: {label_col}</h3>"
                f"<p><strong>Prediction:</strong> {pred} &nbsp; <strong>Support:</strong> {support_text}</p>"
                f"<p><strong>Confidence:</strong> {html.escape(str(step.get('interpretation_confidence', 'NA')))}</p>"
                f"<p><strong>Resolved markers:</strong> {html.escape(str(evidence.get('n_resolved_features', 'NA')))} / {html.escape(str(evidence.get('n_selected_features', 'NA')))}</p>"
                f"<p><strong>Resolved marker status:</strong> {html.escape(str(evidence.get('resolved_marker_evidence_status', 'NA')))}</p>"
                f"{markers_html}"
                "</div>"
            )
        cards.append(
            "<section class='card'>"
            f"<h2>{sid}</h2>"
            f"<p><strong>Predicted hierarchy path:</strong> {html.escape(str(sample.get('predicted_hierarchy_path', 'NA')))}</p>"
            f"<p><strong>Terminal label:</strong> {html.escape(str(sample.get('predicted_terminal_label', 'NA')))}</p>"
            f"<p><strong>Terminal status:</strong> {html.escape(str(sample.get('hierarchy_terminal_status', 'NA')))} — {html.escape(str(sample.get('hierarchy_terminal_reason', 'NA')))}</p>"
            f"{''.join(steps_html)}"
            "</section>"
        )

    html_doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>NetworkParser hierarchy query report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem; background: #fafafa; color: #1f2933; }}
.card {{ background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
.step {{ border-left: 4px solid #d1d5db; padding-left: 1rem; margin: 1rem 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
th, td {{ border: 1px solid #e5e7eb; padding: 0.35rem 0.5rem; text-align: left; }}
th {{ background: #f3f4f6; }}
</style>
</head>
<body>
<h1>NetworkParser multi-level hierarchy query report</h1>
<p>{html.escape(str(report.get('diagnostic_question', '')))}</p>
{''.join(cards)}
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_doc, encoding="utf-8")


def _html_kv(label: str, value: Any) -> str:
    value_text = "NA" if value is None else str(value)
    return (
        f"<div><strong>{html.escape(label)}:</strong> {html.escape(value_text)}</div>"
    )


def write_readable_html_report(report: Dict[str, Any], path: Path) -> None:
    """Write a browser-readable query report with one card per sample."""
    cards: List[str] = []
    for sample in report.get("samples", []) or []:
        sid = html.escape(str(sample.get("sample_id", "NA")))
        marker_items: List[str] = []
        for level_key, title in (
            ("level1_supporting_markers", "Level 1 supporting markers"),
            ("level2_supporting_markers", "Level 2 supporting markers"),
        ):
            markers = sample.get(level_key, []) or []
            rows = []
            for marker in markers[:10]:
                feature = html.escape(str(marker.get("feature", "")))
                value = html.escape(str(marker.get("value", "")))
                role = html.escape(str(marker.get("evidence_role", "")))
                call = html.escape(str(marker.get("allele_call", "")))
                observed = html.escape(str(marker.get("observed_allele", "")))
                rows.append(
                    f"<tr><td>{feature}</td><td>{value}</td><td>{role}</td><td>{call}</td><td>{observed}</td></tr>"
                )
            if rows:
                marker_items.append(
                    f"<h4>{html.escape(title)}</h4>"
                    "<table><thead><tr><th>Feature</th><th>Value</th><th>Evidence role</th><th>Allele call</th><th>Observed</th></tr></thead>"
                    f"<tbody>{''.join(rows)}</tbody></table>"
                )

        cards.append(
            "<section class='card'>"
            f"<h2>Sample: {sid}</h2>"
            "<div class='grid'>"
            "<div><h3>Level 1</h3>"
            + _html_kv("Prediction", sample.get("predicted_level1_identity"))
            + _html_kv("Support", sample.get("level1_support"))
            + _html_kv(
                "Interpretation confidence",
                sample.get("level1_interpretation_confidence"),
            )
            + _html_kv(
                "Resolved marker evidence",
                sample.get("level1_resolved_marker_evidence_status"),
            )
            + _html_kv("Resolved features", sample.get("level1_n_resolved_features"))
            + _html_kv(
                "Resolved baseline features",
                sample.get("level1_n_resolved_baseline_features"),
            )
            + _html_kv("Nonbaseline features", sample.get("level1_n_active_features"))
            + "</div>"
            "<div><h3>Level 2</h3>"
            + _html_kv("Prediction", sample.get("predicted_level2_identity"))
            + _html_kv("Support", sample.get("level2_support"))
            + _html_kv("Model source", sample.get("level2_model_source"))
            + _html_kv(
                "Interpretation confidence",
                sample.get("level2_interpretation_confidence"),
            )
            + _html_kv(
                "Resolved marker evidence",
                sample.get("level2_resolved_marker_evidence_status"),
            )
            + _html_kv("Resolved features", sample.get("level2_n_resolved_features"))
            + _html_kv(
                "Resolved baseline features",
                sample.get("level2_n_resolved_baseline_features"),
            )
            + _html_kv("Nonbaseline features", sample.get("level2_n_active_features"))
            + "</div></div>"
            + "".join(marker_items)
            + "</section>"
        )

    notes = "".join(
        f"<li>{html.escape(str(note))}</li>" for note in report.get("notes", []) or []
    )
    question = html.escape(str(report.get("diagnostic_question", "")))
    document = f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<title>NetworkParser query report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem; line-height: 1.45; background: #f7f7f7; color: #222; }}
.card {{ background: white; border: 1px solid #ddd; border-radius: 12px; padding: 1rem 1.25rem; margin: 1rem 0; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1rem; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 0.5rem; font-size: 0.92rem; }}
th, td {{ border: 1px solid #ddd; padding: 0.4rem; text-align: left; vertical-align: top; }}
th {{ background: #f0f0f0; }}
.question {{ background: #eef5ff; border-left: 4px solid #6699cc; padding: 0.75rem 1rem; }}
</style>
</head>
<body>
<h1>NetworkParser hierarchy query report</h1>
<p class=\"question\"><strong>Diagnostic question:</strong> {question}</p>
<p>Samples queried: {html.escape(str(report.get('n_samples', 0)))}</p>
{''.join(cards)}
<h2>Notes</h2>
<ul>{notes}</ul>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def write_text_report(report: Dict[str, Any], path: Path) -> None:
    lines: List[str] = []
    lines.append("NetworkParser hierarchy query report")
    lines.append("=" * 42)
    lines.append(f"Samples queried: {report.get('n_samples', 0)}")
    if report.get("diagnostic_question"):
        lines.append("")
        lines.append("Diagnostic question")
        lines.append("-------------------")
        lines.append(str(report.get("diagnostic_question")))
    lines.append("")

    def _evidence_lines(prefix: str, evidence: Dict[str, Any]) -> List[str]:
        out: List[str] = []
        if not evidence:
            return out
        out.append("  Level-specific marker evidence:")
        out.append(
            f"    Selected features: {evidence.get('n_selected_features', 'NA')}"
        )
        out.append(
            f"    Nonbaseline features: {evidence.get('n_active_features', 'NA')}"
        )
        out.append(
            f"    Resolved trained-marker states: {evidence.get('n_resolved_features', 'NA')}"
        )
        out.append(
            f"    Resolved baseline states: {evidence.get('n_resolved_baseline_features', 'NA')}"
        )
        if evidence.get("resolved_feature_fraction") is not None:
            out.append(
                f"    Resolved feature fraction: {float(evidence.get('resolved_feature_fraction')):.4f}"
            )
        if evidence.get("unique_mapped_fraction") is not None:
            out.append(
                f"    Unique recovery fraction: {float(evidence.get('unique_mapped_fraction')):.4f}"
            )
        out.append(
            f"    Marker recovery: {evidence.get('marker_recovery_status', 'NA')}"
        )
        out.append(
            f"    Resolved marker pattern: {evidence.get('resolved_marker_evidence_status', 'NA')}"
        )
        out.append(
            f"    Nonbaseline evidence: {evidence.get('nonbaseline_evidence_status', evidence.get('active_marker_evidence_status', 'NA'))}"
        )
        if evidence.get("n_zero_fill_caution_features", 0):
            out.append(
                f"    Zero-fill caution states: {evidence.get('n_zero_fill_caution_features')}"
            )
        if evidence.get("n_multi_hit_calls", 0):
            out.append(
                f"    Multi-hit caution states: {evidence.get('n_multi_hit_calls')}"
            )
        if evidence.get("n_ambiguous_base_calls", 0):
            out.append(
                f"    Ambiguous-base caution states: {evidence.get('n_ambiguous_base_calls')}"
            )
        if evidence.get("n_non_training_allele_calls", 0):
            out.append(
                f"    Non-training-allele caution states: {evidence.get('n_non_training_allele_calls')}"
            )
        return out

    def _marker_lines(markers: List[Dict[str, Any]]) -> List[str]:
        out: List[str] = []
        for marker in markers[:10]:
            role = marker.get("evidence_role") or "NA"
            extra = f" | role={role}"
            if marker.get("observed_allele"):
                quality = (
                    marker.get("mapping_quality") or marker.get("allele_call") or ""
                )
                quality_txt = f" | quality={quality}" if quality else ""
                extra += f" | observed={marker.get('observed_allele')} | status={marker.get('mapping_status', '')}{quality_txt}"
            out.append(f"    - {marker.get('feature')} = {marker.get('value')}{extra}")
        return out

    for sample in report.get("samples", []):
        lines.append(f"Sample: {sample.get('sample_id')}")
        lines.append("-" * (8 + len(str(sample.get("sample_id", "")))))
        if (
            sample.get("fasta_mapping_quality")
            or sample.get("vcf_mapping_quality")
            or sample.get("raw_sequence_mapping_quality")
        ):
            rq = (
                sample.get("fasta_mapping_quality")
                or sample.get("vcf_mapping_quality")
                or sample.get("raw_sequence_mapping_quality")
                or {}
            )
            label = (
                "FASTA context recovery"
                if sample.get("fasta_mapping_quality")
                else "VCF trained-feature recovery"
            )
            lines.append(label)
            lines.append(f"  Marker recovery: {rq.get('marker_recovery_status', 'NA')}")
            if rq.get("resolved_marker_evidence_status"):
                lines.append(
                    f"  Resolved marker pattern: {rq.get('resolved_marker_evidence_status', 'NA')}"
                )
            lines.append(
                f"  Nonbaseline marker evidence: {rq.get('active_marker_evidence_status', 'NA')}"
            )
            lines.append(
                f"  Unique mapped/resolved calls: {rq.get('n_unique_mapped_calls', 0)} / {rq.get('n_feature_calls', 0)}"
            )
            if rq.get("n_resolved_features") is not None:
                lines.append(
                    f"  Resolved trained-marker calls: {rq.get('n_resolved_features', 0)}"
                )
            if rq.get("n_resolved_baseline_features") is not None:
                lines.append(
                    f"  Resolved baseline calls encoded as 0: {rq.get('n_resolved_baseline_features', 0)}"
                )
            lines.append(
                f"  Active encoded calls: {rq.get('n_encoded_active_features', 0)}"
            )
            if rq.get("n_multi_hit_calls", 0):
                lines.append(
                    f"  Multi-hit contexts/coordinates filled as 0: {rq.get('n_multi_hit_calls')}"
                )
            if rq.get("n_ambiguous_base_calls", 0):
                lines.append(
                    f"  Ambiguous-base calls filled as 0: {rq.get('n_ambiguous_base_calls')}"
                )
            if rq.get("n_non_training_allele_calls", 0):
                lines.append(
                    f"  Non-training alleles filled as 0: {rq.get('n_non_training_allele_calls')}"
                )
            if rq.get("n_unresolved_or_missing_context_calls", 0):
                lines.append(
                    f"  Unresolved/missing contexts filled as 0: {rq.get('n_unresolved_or_missing_context_calls')}"
                )
            lines.append("")

        lines.append("Level 1 — strain/sample placement")
        lines.append(f"  Prediction: {sample.get('predicted_level1_identity')}")
        if sample.get("level1_support") is not None:
            lines.append(f"  Support: {float(sample.get('level1_support')):.4f}")
        if sample.get("level1_interpretation_confidence"):
            lines.append(
                f"  Interpretation confidence: {sample.get('level1_interpretation_confidence')}"
            )
            if sample.get("level1_confidence_note"):
                lines.append(
                    f"  Confidence note: {sample.get('level1_confidence_note')}"
                )
        lines.extend(
            _evidence_lines("level1", sample.get("level1_feature_evidence") or {})
        )
        if sample.get("level1_supporting_markers"):
            lines.append("  Supporting markers:")
            lines.extend(_marker_lines(sample.get("level1_supporting_markers", [])))
        if sample.get("level1_decision_path"):
            lines.append("  Decision path:")
            for rule in sample.get("level1_decision_path", []):
                lines.append(f"    - {rule}")

        lines.append("")
        lines.append("Level 2 — resistance profile")
        lines.append(f"  Prediction: {sample.get('predicted_level2_identity')}")
        if sample.get("level2_support") is not None:
            lines.append(f"  Support: {float(sample.get('level2_support')):.4f}")
        lines.append(f"  Model source: {sample.get('level2_model_source')}")
        if sample.get("level2_target_label_column"):
            lines.append(
                f"  Target label column: {sample.get('level2_target_label_column')}"
            )
        if sample.get("level2_interpretation_confidence"):
            lines.append(
                f"  Interpretation confidence: {sample.get('level2_interpretation_confidence')}"
            )
            if sample.get("level2_confidence_note"):
                lines.append(
                    f"  Confidence note: {sample.get('level2_confidence_note')}"
                )
        lines.extend(
            _evidence_lines("level2", sample.get("level2_feature_evidence") or {})
        )
        if sample.get("level2_supporting_markers"):
            lines.append("  Supporting markers:")
            lines.extend(_marker_lines(sample.get("level2_supporting_markers", [])))
        if sample.get("level2_decision_path"):
            lines.append("  Decision path:")
            for rule in sample.get("level2_decision_path", []):
                lines.append(f"    - {rule}")
        lines.append("")

    lines.append("Notes")
    lines.append("-----")
    for note in report.get("notes", []):
        lines.append(f"- {note}")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply trained hierarchical NetworkParser models to new strain/sample input.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--genomic", required=True, help="New genomic matrix file or VCF directory."
    )
    parser.add_argument(
        "--registry",
        required=True,
        help="Path to hierarchy/two-level/hierarchical model registry JSON from training.",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory for query outputs."
    )
    parser.add_argument(
        "--config", default=None, help="Optional JSON config override file."
    )
    parser.add_argument(
        "--ref_fasta",
        default=None,
        help="Optional reference FASTA for VCF parsing context.",
    )
    parser.add_argument(
        "--max_markers",
        type=int,
        default=10,
        help="Maximum supporting markers to show per level per sample.",
    )
    parser.add_argument(
        "--n_jobs", type=int, default=None, help="Runtime worker override."
    )
    parser.add_argument(
        "--query_input_type",
        choices=["auto", "matrix", "vcf", "fasta", "raw_sequence", "fastq"],
        default="auto",
        help="Interpret --genomic as a prebuilt matrix/VCF input, FASTA DNA, or paired FASTQ reads. raw_sequence remains a deprecated alias for fasta.",
    )
    parser.add_argument(
        "--fasta_mapping_mode",
        "--raw_sequence_mapping_mode",
        dest="raw_sequence_mapping_mode",
        choices=["auto", "blast", "exact"],
        default="auto",
        help="How FASTA query sequences should be mapped to selected feature contexts. The old --raw_sequence_mapping_mode option remains as an alias.",
    )
    parser.add_argument("--fastq_max_parallel_samples", type=int, default=None)
    parser.add_argument("--fastq_threads", type=int, default=None)
    parser.add_argument("--fastq_memory_per_sample_mb", type=int, default=None)
    parser.add_argument("--fastq_clean_intermediates", action="store_true")
    parser.add_argument("--fastq_no_auto_index_reference", action="store_true")
    parser.add_argument("--fastq_min_mapping_quality", type=int, default=None)
    return parser


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main(argv: Optional[List[str]] = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.n_jobs is not None:
        config.n_jobs = int(args.n_jobs)
    for key in [
        "fastq_max_parallel_samples",
        "fastq_threads",
        "fastq_memory_per_sample_mb",
        "fastq_min_mapping_quality",
    ]:
        value = getattr(args, key, None)
        if value is not None:
            setattr(config, key, value)
    if bool(getattr(args, "fastq_clean_intermediates", False)):
        config.fastq_clean_intermediates = True
    if bool(getattr(args, "fastq_no_auto_index_reference", False)):
        config.fastq_auto_index_reference = False
    config.__post_init__()

    engine = NetworkParserQueryEngine(registry_path=args.registry, config=config)
    engine.query(
        genomic_path=args.genomic,
        output_dir=args.output_dir,
        ref_fasta=args.ref_fasta,
        max_markers=int(args.max_markers),
        n_jobs=args.n_jobs,
        query_input_type=args.query_input_type,
        raw_sequence_mapping_mode=args.raw_sequence_mapping_mode,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
