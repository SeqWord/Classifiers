#!/usr/bin/env python3
# network_parser/model_bundle.py
"""
NetworkParser binary model bundle
=================================

Purpose
-------
Package a trained hierarchical NetworkParser registry into one portable binary
object that can be loaded for end-to-end query inference.

The bundle intentionally stores more than sklearn-style model objects.  It also
stores the selected-feature metadata needed to reconstruct the model-ready
matrix from a new query sample, including selected feature manifests, context
sequences, baseline/REF/ALT allele definitions, and feature-state evidence
needed by raw-sequence query mode.

Security / trust model
----------------------
``.npb`` bundles are **pickle-based and trusted-input-only**.  Only load bundles
produced by your own training pipeline or another fully trusted source.  Do not
open untrusted ``.npb`` files: pickle deserialization can execute arbitrary code.

Design rule
-----------
Training-time statistical decisions remain training-time decisions.  Querying a
bundle is inference-only: it does not rerun RF-FDR, chi-square/Fisher FDR,
permutation testing, model selection, decision-tree fitting, or bootstrap
confidence scoring.  It reloads the trained knowledge object and applies it to a
new sample in the saved feature space.

Portability rule
----------------
A complete bundle embeds every model payload, feature list, selected-feature
manifest, ranked annotation table, and terminal/global fallback required by
successful hierarchy nodes so query works after the training directory is moved
or removed.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
import logging
import pickle
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from mtb_amr_classifier.config import NetworkParserConfig
    from mtb_amr_classifier.query_engine import (
        NetworkParserQueryEngine,
        apply_trained_vcf_config,
        extract_model_importance,
        read_ranked_feature_table,
    )
except Exception:  # pragma: no cover - supports direct source-tree execution
    from config import NetworkParserConfig  # type: ignore
    from query_engine import (  # type: ignore
        NetworkParserQueryEngine,
        apply_trained_vcf_config,
        extract_model_importance,
        read_ranked_feature_table,
    )

logger = logging.getLogger(__name__)

# Schema 1.3 embeds and hashes the exact serialized model-file bytes. Earlier
# object re-pickling hashes were not stable for every custom model class.
BUNDLE_SCHEMA_VERSION = "1.3"
VCF_SEMANTICS_VERSION = "1.0"
SUPPORTED_BUNDLE_SCHEMA_VERSIONS = frozenset({"1.0", "1.1", "1.2", "1.3"})
# strict: reject unknown / unsupported schema versions
# permissive: warn and continue for unknown versions (not recommended)
DEFAULT_COMPATIBILITY_POLICY = "strict"
DEFAULT_BUNDLE_SUFFIX = ".npb"
# Trusted-input-only marker written into every bundle
PICKLE_TRUST_NOTE = (
    "TRUSTED-INPUT-ONLY: NetworkParser .npb bundles use Python pickle. "
    "Load only bundles from trusted training runs; never deserialize untrusted files."
)


# -----------------------------------------------------------------------------
# Small serialisation / path helpers
# -----------------------------------------------------------------------------


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_default(obj: Any) -> Any:
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
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    return str(obj)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def _resolve_path(path_value: Optional[str], base_dir: Path) -> Optional[Path]:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    return path


def _patch_sklearn_estimator_compat(obj: Any, _seen: Optional[set] = None) -> Any:
    """Repair estimators pickled under older sklearn for newer runtimes.

    sklearn>=1.4 DecisionTreeClassifier expects ``monotonic_cst``; models
    trained on 1.3.x lack the attribute and crash in ``predict_proba``.
    """
    if obj is None:
        return obj
    if _seen is None:
        _seen = set()
    try:
        oid = id(obj)
        if oid in _seen:
            return obj
        _seen.add(oid)
    except Exception:
        pass

    # Decision trees / forests (sklearn>=1.4 expects monotonic_cst on trees)
    if hasattr(obj, "tree_") and not hasattr(obj, "monotonic_cst"):
        try:
            object.__setattr__(obj, "monotonic_cst", None)
        except Exception:
            try:
                obj.monotonic_cst = None  # type: ignore[attr-defined]
            except Exception:
                pass

    # sklearn Pipeline / FeatureUnion / ColumnTransformer containers
    steps = None
    if hasattr(obj, "named_steps") and isinstance(getattr(obj, "named_steps"), dict):
        steps = list(obj.named_steps.values())
    elif hasattr(obj, "steps") and isinstance(getattr(obj, "steps"), (list, tuple)):
        steps = [s[1] if isinstance(s, (list, tuple)) and len(s) >= 2 else s for s in obj.steps]
    elif hasattr(obj, "transformers") and isinstance(getattr(obj, "transformers"), (list, tuple)):
        steps = []
        for t in obj.transformers:
            if isinstance(t, (list, tuple)) and len(t) >= 2:
                steps.append(t[1])
    if steps:
        for child in steps:
            _patch_sklearn_estimator_compat(child, _seen)

    # RandomForest / Bagging / Voting / stacking: patch each base estimator
    for attr in ("estimators_", "estimators", "estimator_list"):
        if not hasattr(obj, attr):
            continue
        try:
            children = getattr(obj, attr)
        except Exception:
            continue
        if isinstance(children, (list, tuple)):
            for child in children:
                if isinstance(child, (list, tuple)) and len(child) >= 2:
                    # VotingClassifier style (name, est)
                    _patch_sklearn_estimator_compat(child[1], _seen)
                else:
                    _patch_sklearn_estimator_compat(child, _seen)

    # NetworkParser wrapper often stores .pipeline or .model
    for attr in (
        "pipeline",
        "model",
        "estimator",
        "estimator_",
        "base_estimator",
        "base_estimator_",
        "classifier",
        "best_estimator_",
    ):
        if hasattr(obj, attr):
            try:
                child = getattr(obj, attr)
            except Exception:
                continue
            if child is not None and child is not obj:
                _patch_sklearn_estimator_compat(child, _seen)

    # dict payloads {"model": ...}
    if isinstance(obj, dict):
        for v in obj.values():
            _patch_sklearn_estimator_compat(v, _seen)

    return obj


def _load_pickle_or_joblib(path: Path) -> Any:
    """Load a trained model payload using joblib first, then pickle."""
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
        import joblib  # type: ignore

        loaded = joblib.load(path)
    except Exception:
        with open(path, "rb") as handle:
            loaded = pickle.load(handle)
    return _patch_sklearn_estimator_compat(loaded)


def _load_pickle_or_joblib_bytes(raw: bytes) -> Any:
    """Load a trusted serialized model from its exact embedded file bytes."""
    try:
        from mtb_amr_classifier.pickle_compat import (
            install_network_parser_pickle_aliases,
        )
    except ImportError:  # pragma: no cover
        from pickle_compat import install_network_parser_pickle_aliases  # type: ignore

    install_network_parser_pickle_aliases()
    try:
        import joblib  # type: ignore

        loaded = joblib.load(io.BytesIO(raw))
    except Exception:
        loaded = pickle.loads(raw)
    return _patch_sklearn_estimator_compat(loaded)


def _safe_token(value: Any, max_len: int = 80) -> str:
    raw = str(value)
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)
    cleaned = cleaned.strip("_") or "item"
    if len(cleaned) <= max_len:
        return cleaned
    digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"{cleaned[: max_len - 13]}_{digest}"


def _slot_id(tokens: Sequence[Any]) -> str:
    raw = json.dumps([str(t) for t in tokens], ensure_ascii=False)
    digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:12]
    label = "__".join(_safe_token(t, max_len=30) for t in tokens[:-1])
    return f"{label or 'slot'}__{digest}"


def _get_nested(mapping: Dict[str, Any], tokens: Sequence[Any]) -> Any:
    current: Any = mapping
    for token in tokens:
        if not isinstance(current, dict):
            return None
        current = current.get(token)
    return current


def _set_nested(mapping: Dict[str, Any], tokens: Sequence[Any], value: Any) -> None:
    current: Dict[str, Any] = mapping
    for token in tokens[:-1]:
        child = current.get(token)
        if not isinstance(child, dict):
            child = {}
            current[token] = child
        current = child
    current[tokens[-1]] = value


def _set_nested_payload_key(
    mapping: Dict[str, Any], tokens: Sequence[Any], key: str, value: Any
) -> None:
    current = _get_nested(mapping, tokens)
    if isinstance(current, dict):
        current[key] = value


def _feature_hash(features: Iterable[Any]) -> str:
    canonical = [str(f) for f in features]
    payload = "\n".join(canonical).encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _payload_content_hash(payload: Any) -> str:
    """
    Content hash for an embedded model payload.

    Schema 1.3 model payloads are exact serialized model-file bytes, which are
    hashed directly and therefore remain stable across processes and bundle
    serialization round trips. Legacy in-memory objects retain protocol-4
    hashing for compatibility with older bundle schemas.

    Does **not** fall back to ``repr()`` — unstable string forms are forbidden
    for persistent content hashes. Unpicklable objects raise.
    """
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return _sha256_bytes(bytes(payload))

    try:
        raw = pickle.dumps(payload, protocol=4)
    except Exception as exc:
        raise ValueError(
            f"Cannot compute stable content hash for model payload (pickle failed): {exc}. "
            "Bundles require picklable model objects; repr() fallback is not permitted."
        ) from exc
    return _sha256_bytes(raw)


def _node_is_successful_trainable(payload: Dict[str, Any]) -> bool:
    """True when a registry node is expected to contribute a usable model."""
    if not isinstance(payload, dict):
        return False
    status = str(payload.get("status", "success")).strip().lower()
    if status in {
        "skipped",
        "unavailable",
        "failed",
        "not_trained",
        "deterministic",
        "not_applicable",
        "underpowered",
    }:
        return False
    features = payload.get("features") or []
    model_file = payload.get("model_file")
    return bool(features) or bool(model_file)


# -----------------------------------------------------------------------------
# Registry traversal helpers
# -----------------------------------------------------------------------------


def is_hierarchical_registry(registry: Dict[str, Any]) -> bool:
    """Return True when the registry uses the recursive hierarchy schema."""
    if not isinstance(registry, dict):
        return False
    protocol = str(registry.get("protocol", "")).strip().lower()
    hierarchy = registry.get("hierarchy", {})
    return protocol == "multi_level_hierarchy_protocol" or (
        isinstance(hierarchy, dict) and isinstance(hierarchy.get("root"), dict)
    )


def _iter_hierarchy_model_slots(
    node: Dict[str, Any],
    tokens: List[Any],
) -> Iterator[Tuple[List[Any], str, Dict[str, Any]]]:
    """Yield recursive hierarchy model-file slots."""
    if not isinstance(node, dict):
        return

    label_column = str(node.get("label_column", "label"))
    level_number = str(node.get("level_number", "level"))
    path_values = []
    for item in node.get("path", []) or []:
        if isinstance(item, dict):
            path_values.append(str(item.get("value", "")))
    model_id = "hierarchy." + ".".join(
        [f"level_{level_number}", _safe_token(label_column, 40)]
        + [_safe_token(value, 40) for value in path_values]
    )

    yield tokens + ["model_file"], model_id, node

    children = node.get("children", {})
    if isinstance(children, dict):
        for child_key, child in children.items():
            if isinstance(child, dict):
                yield from _iter_hierarchy_model_slots(
                    child,
                    tokens + ["children", str(child_key)],
                )


def _iter_hierarchy_manifest_slots(
    node: Dict[str, Any],
    tokens: List[Any],
) -> Iterator[Tuple[List[Any], str]]:
    """Yield recursive hierarchy selected-feature manifest slots when present."""
    if not isinstance(node, dict):
        return

    label_column = str(node.get("label_column", "label"))
    level_number = str(node.get("level_number", "level"))
    fm = node.get("feature_manifest", {})
    if isinstance(fm, dict) and fm.get("manifest_file"):
        logical_name = f"hierarchy.level_{level_number}.{_safe_token(label_column, 40)}.feature_manifest"
        yield tokens + ["feature_manifest", "manifest_file"], logical_name

    children = node.get("children", {})
    if isinstance(children, dict):
        for child_key, child in children.items():
            if isinstance(child, dict):
                yield from _iter_hierarchy_manifest_slots(
                    child,
                    tokens + ["children", str(child_key)],
                )


def iter_model_file_slots(
    registry: Dict[str, Any]
) -> Iterator[Tuple[List[Any], str, Dict[str, Any]]]:
    """
    Yield model-file locations in a hierarchy registry.

    Each yield is: (path_tokens_to_model_file, model_id, payload_dict).
    The payload dict is the registry section holding model_file/features/status.
    """
    if is_hierarchical_registry(registry):
        hierarchy = registry.get("hierarchy", {}) if isinstance(registry, dict) else {}
        root = hierarchy.get("root", {}) if isinstance(hierarchy, dict) else {}
        if isinstance(root, dict):
            yield from _iter_hierarchy_model_slots(root, ["hierarchy", "root"])
        global_lineage_fallback = (
            hierarchy.get("global_lineage_fallback", {})
            if isinstance(hierarchy, dict)
            else {}
        )
        if isinstance(global_lineage_fallback, dict) and global_lineage_fallback.get(
            "model_file"
        ):
            yield (
                ["hierarchy", "global_lineage_fallback", "model_file"],
                "hierarchy.global_lineage_fallback",
                global_lineage_fallback,
            )
        terminal_fallbacks = (
            hierarchy.get("terminal_fallbacks", {})
            if isinstance(hierarchy, dict)
            else {}
        )
        if isinstance(terminal_fallbacks, dict):
            yield from _walk_terminal_fallback_models(
                terminal_fallbacks, ["hierarchy", "terminal_fallbacks"]
            )

    level1 = registry.get("level1", {}) if isinstance(registry, dict) else {}
    if isinstance(level1, dict):
        yield ["level1", "model_file"], "level1", level1

    level2 = registry.get("level2", {}) if isinstance(registry, dict) else {}
    if not isinstance(level2, dict):
        return

    global_fallback = level2.get("global_fallback", {})
    if isinstance(global_fallback, dict):
        yield [
            "level2",
            "global_fallback",
            "model_file",
        ], "level2.global_fallback", global_fallback

    global_binary = level2.get("global_binary_fallback", {})
    if isinstance(global_binary, dict):
        yield [
            "level2",
            "global_binary_fallback",
            "model_file",
        ], "level2.global_binary_fallback", global_binary

    by_group = level2.get("by_level1_group", {})
    if isinstance(by_group, dict):
        for group, payload in by_group.items():
            if not isinstance(payload, dict):
                continue
            model_id = f"level2.by_level1_group::{str(group)}"
            yield [
                "level2",
                "by_level1_group",
                str(group),
                "model_file",
            ], model_id, payload


def _walk_terminal_fallback_models(
    obj: Any,
    tokens: List[Any],
) -> Iterator[Tuple[List[Any], str, Dict[str, Any]]]:
    if not isinstance(obj, dict):
        return
    if _node_is_successful_trainable(obj) and (
        obj.get("model_file") or obj.get("features")
    ):
        model_id = "hierarchy.terminal_fallback." + ".".join(
            _safe_token(str(t), 30)
            for t in tokens[2:]
            if str(t) not in {"", "model_file"}
        )
        yield tokens + ["model_file"], model_id or "hierarchy.terminal_fallback", obj
    for key, child in obj.items():
        if key in {
            "model_file",
            "features",
            "feature_manifest",
            "filter",
            "artifacts",
            "config",
        }:
            continue
        if isinstance(child, dict):
            yield from _walk_terminal_fallback_models(child, tokens + [str(key)])
        elif isinstance(child, list):
            for i, item in enumerate(child):
                if isinstance(item, dict):
                    yield from _walk_terminal_fallback_models(
                        item, tokens + [str(key), str(i)]
                    )


def _walk_terminal_fallback_manifests(
    obj: Any,
    tokens: List[Any],
) -> Iterator[Tuple[List[Any], str]]:
    if not isinstance(obj, dict):
        return
    fm = obj.get("feature_manifest")
    if isinstance(fm, dict) and fm.get("manifest_file"):
        logical = "hierarchy.terminal_fallback." + ".".join(
            _safe_token(str(t), 30) for t in tokens[2:]
        )
        yield tokens + [
            "feature_manifest",
            "manifest_file",
        ], logical + ".feature_manifest"
    for key, child in obj.items():
        if key in {
            "model_file",
            "features",
            "filter",
            "artifacts",
            "config",
            "manifest_file",
        }:
            continue
        if isinstance(child, dict):
            yield from _walk_terminal_fallback_manifests(child, tokens + [str(key)])
        elif isinstance(child, list):
            for i, item in enumerate(child):
                if isinstance(item, dict):
                    yield from _walk_terminal_fallback_manifests(
                        item, tokens + [str(key), str(i)]
                    )


def iter_manifest_slots(registry: Dict[str, Any]) -> Iterator[Tuple[List[Any], str]]:
    """Yield manifest-file locations that are actually present in the registry."""
    training_matrix = (
        registry.get("training_matrix", {}) if isinstance(registry, dict) else {}
    )
    if isinstance(training_matrix, dict) and training_matrix.get(
        "feature_manifest_file"
    ):
        yield [
            "training_matrix",
            "feature_manifest_file",
        ], "training_matrix.feature_manifest"

    if is_hierarchical_registry(registry):
        hierarchy = registry.get("hierarchy", {}) if isinstance(registry, dict) else {}
        root = hierarchy.get("root", {}) if isinstance(hierarchy, dict) else {}
        if isinstance(root, dict):
            yield from _iter_hierarchy_manifest_slots(root, ["hierarchy", "root"])
        global_lineage_fallback = (
            hierarchy.get("global_lineage_fallback", {})
            if isinstance(hierarchy, dict)
            else {}
        )
        if isinstance(global_lineage_fallback, dict):
            manifest = global_lineage_fallback.get("feature_manifest", {})
            if isinstance(manifest, dict) and manifest.get("manifest_file"):
                yield (
                    [
                        "hierarchy",
                        "global_lineage_fallback",
                        "feature_manifest",
                        "manifest_file",
                    ],
                    "hierarchy.global_lineage_fallback.feature_manifest",
                )
        terminal_fallbacks = (
            hierarchy.get("terminal_fallbacks", {})
            if isinstance(hierarchy, dict)
            else {}
        )
        if isinstance(terminal_fallbacks, dict):
            yield from _walk_terminal_fallback_manifests(
                terminal_fallbacks, ["hierarchy", "terminal_fallbacks"]
            )

    level1 = registry.get("level1", {}) if isinstance(registry, dict) else {}
    if isinstance(level1, dict):
        fm = level1.get("feature_manifest", {})
        if isinstance(fm, dict) and fm.get("manifest_file"):
            yield [
                "level1",
                "feature_manifest",
                "manifest_file",
            ], "level1.feature_manifest"

    level2 = registry.get("level2", {}) if isinstance(registry, dict) else {}
    if not isinstance(level2, dict):
        return

    for name in ("global_fallback", "global_binary_fallback"):
        payload = level2.get(name, {})
        if not isinstance(payload, dict):
            continue
        fm = payload.get("feature_manifest", {})
        if isinstance(fm, dict) and fm.get("manifest_file"):
            yield (
                ["level2", name, "feature_manifest", "manifest_file"],
                f"level2.{name}.feature_manifest",
            )

    by_group = level2.get("by_level1_group", {})
    if isinstance(by_group, dict):
        for group, payload in by_group.items():
            if not isinstance(payload, dict):
                continue
            fm = payload.get("feature_manifest", {})
            if isinstance(fm, dict) and fm.get("manifest_file"):
                yield (
                    [
                        "level2",
                        "by_level1_group",
                        str(group),
                        "feature_manifest",
                        "manifest_file",
                    ],
                    f"level2.by_level1_group::{str(group)}.feature_manifest",
                )


def iter_ranked_table_slots(
    registry: Dict[str, Any]
) -> Iterator[Tuple[List[Any], str]]:
    """
    Yield ranked-feature table artifact locations where available.

    These are optional.  They improve supporting-marker ranking during bundled
    query mode, but predictions can still run without them.
    """

    def _walk(obj: Any, tokens: List[Any]) -> Iterator[Tuple[List[Any], str]]:
        if not isinstance(obj, dict):
            return
        artifacts = obj.get("artifacts")
        if isinstance(artifacts, dict):
            for key in ("rf_fdr_results_csv", "feature_results_csv"):
                if artifacts.get(key):
                    yield tokens + [
                        "artifacts",
                        key,
                    ], f"{'.'.join(map(str, tokens))}.{key}"
        for key, value in obj.items():
            if isinstance(value, dict):
                yield from _walk(value, tokens + [key])

    yield from _walk(registry, [])


def collect_required_features(registry: Dict[str, Any]) -> List[str]:
    """Collect the ordered union of all features required by bundled models."""
    ordered: List[str] = []
    seen = set()
    for _, _, payload in iter_model_file_slots(registry):
        features = payload.get("features", []) if isinstance(payload, dict) else []
        for feature in features or []:
            f = str(feature)
            if f and f not in seen:
                seen.add(f)
                ordered.append(f)
    return ordered


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------


@dataclass
class BundleFileRecord:
    """Serializable representation of a text table stored inside the bundle."""

    slot_tokens: List[Any]
    bundle_id: str
    original_path: Optional[str]
    kind: str
    columns: List[str]
    records: List[Dict[str, Any]]
    sep: str = "\t"

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.records)
        if self.columns:
            for col in self.columns:
                if col not in df.columns:
                    df[col] = ""
            df = df.loc[:, self.columns]
        return df


@dataclass
class NetworkParserModelBundle:
    """
    Portable binary NetworkParser trained-knowledge object.

    The bundle keeps the JSON registry for transparency, but also embeds the
    model payloads and query-critical biological metadata so the object remains
    usable after being moved away from the original training directory.

    Pickle trust: load only trusted bundles (see module docstring).
    """

    schema_version: str
    created_at: str
    registry: Dict[str, Any]
    registry_source: str
    model_payloads: Dict[str, Any] = field(default_factory=dict)
    model_sources: Dict[str, str] = field(default_factory=dict)
    manifest_records: List[BundleFileRecord] = field(default_factory=list)
    ranked_table_records: List[BundleFileRecord] = field(default_factory=list)
    feature_space: Dict[str, Any] = field(default_factory=dict)
    validation_evidence: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    # content integrity
    content_hashes: Dict[str, Any] = field(default_factory=dict)
    completeness: Dict[str, Any] = field(default_factory=dict)

    def materialize_runtime_files(self, output_dir: Path) -> Dict[str, Any]:
        """
        Write embedded manifests / ranked feature tables to a runtime directory
        and return a registry copy whose paths point to the materialized files.
        """
        runtime_dir = _ensure_dir(Path(output_dir))
        materialized_registry = copy.deepcopy(self.registry)

        manifest_dir = _ensure_dir(runtime_dir / "manifests")
        for record in self.manifest_records:
            path = manifest_dir / f"{_safe_token(record.bundle_id)}.tsv"
            record.to_dataframe().to_csv(path, sep=record.sep or "\t", index=False)
            _set_nested(materialized_registry, record.slot_tokens, str(path))

        table_dir = _ensure_dir(runtime_dir / "ranked_feature_tables")
        for record in self.ranked_table_records:
            path = table_dir / f"{_safe_token(record.bundle_id)}.csv"
            record.to_dataframe().to_csv(path, index=False)
            _set_nested(materialized_registry, record.slot_tokens, str(path))

        # Attach bundle model IDs to the copied registry so the bundled query
        # engine can load payloads from memory instead of from model_file paths.
        for tokens, model_id, _payload in iter_model_file_slots(materialized_registry):
            _set_nested_payload_key(
                materialized_registry, tokens[:-1], "bundle_model_id", model_id
            )

        registry_path = runtime_dir / "bundled_runtime_registry.json"
        _write_json(materialized_registry, registry_path)

        return {
            "registry": materialized_registry,
            "registry_path": registry_path,
            "runtime_dir": runtime_dir,
        }

    def to_payload(self) -> Dict[str, Any]:
        """Return a plain pickle payload for cross-entry-point portability."""
        return {
            "__networkparser_model_bundle__": True,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "registry": self.registry,
            "registry_source": self.registry_source,
            "model_payloads": self.model_payloads,
            "model_sources": self.model_sources,
            "manifest_records": [asdict(record) for record in self.manifest_records],
            "ranked_table_records": [
                asdict(record) for record in self.ranked_table_records
            ],
            "feature_space": self.feature_space,
            "validation_evidence": self.validation_evidence,
            "notes": self.notes,
            "content_hashes": self.content_hashes,
            "completeness": self.completeness,
            "pickle_trust_note": PICKLE_TRUST_NOTE,
        }

    def verify_content_hashes(self, *, strict: bool = True) -> Dict[str, Any]:
        """
        Verify stored payload hashes against re-hashed embedded objects.

        Returns a report dict. Raises ValueError when ``strict`` and any mismatch.
        """
        report: Dict[str, Any] = {
            "status": "ok",
            "model_mismatches": [],
            "manifest_mismatches": [],
            "preprocessing_mismatches": [],
            "skipped": [],
        }
        current_schema_strict = bool(
            strict and self.schema_version == BUNDLE_SCHEMA_VERSION
        )
        stored_models = (self.content_hashes or {}).get("models", {}) or {}
        stored_model_format = (self.content_hashes or {}).get("model_payload_format")
        if current_schema_strict and stored_model_format != "serialized_file_bytes":
            report["model_mismatches"].append(
                {
                    "error": "missing_or_unsupported_model_payload_format",
                    "expected": "serialized_file_bytes",
                    "actual": stored_model_format,
                }
            )
        if not stored_models and self.schema_version in {"1.0", "unknown"}:
            report["skipped"].append("no_model_hashes_legacy_schema")
        elif not stored_models and strict:
            report["model_mismatches"].append({"error": "missing_all_model_hashes"})
        for model_id, payload in (self.model_payloads or {}).items():
            if current_schema_strict and not isinstance(
                payload, (bytes, bytearray, memoryview)
            ):
                report["model_mismatches"].append(
                    {
                        "model_id": model_id,
                        "error": "model_payload_is_not_serialized_file_bytes",
                    }
                )
                continue
            expected = stored_models.get(model_id)
            if not expected:
                if strict and self.schema_version not in {"1.0", "unknown"}:
                    report["model_mismatches"].append(
                        {"model_id": model_id, "error": "missing_hash"}
                    )
                continue
            actual = _payload_content_hash(payload)
            if actual != expected:
                report["model_mismatches"].append(
                    {"model_id": model_id, "expected": expected, "actual": actual}
                )

        stored_manifests = (self.content_hashes or {}).get("manifests", {}) or {}
        for record in self.manifest_records or []:
            expected = stored_manifests.get(record.bundle_id)
            if not expected:
                if strict and self.schema_version not in {"1.0", "unknown"}:
                    report["manifest_mismatches"].append(
                        {"bundle_id": record.bundle_id, "error": "missing_hash"}
                    )
                continue
            df = record.to_dataframe()
            raw = df.to_csv(sep=record.sep or "\t", index=False).encode("utf-8")
            actual = _sha256_bytes(raw)
            if actual != expected:
                report["manifest_mismatches"].append(
                    {
                        "bundle_id": record.bundle_id,
                        "expected": expected,
                        "actual": actual,
                    }
                )

        stored_features = (self.content_hashes or {}).get("feature_lists", {}) or {}
        feature_mismatches: List[Dict[str, Any]] = []
        registry_feature_hashes: Dict[str, str] = {}
        for _tokens, model_id, payload in iter_model_file_slots(self.registry or {}):
            if not isinstance(payload, dict) or not _node_is_successful_trainable(
                payload
            ):
                continue
            features = [str(value) for value in (payload.get("features") or [])]
            if features:
                registry_feature_hashes[model_id] = _feature_hash(features)
        for model_id in self.model_payloads or {}:
            expected = stored_features.get(model_id)
            actual = registry_feature_hashes.get(model_id)
            if not expected:
                if current_schema_strict:
                    feature_mismatches.append(
                        {"model_id": model_id, "error": "missing_feature_list_hash"}
                    )
            elif not actual:
                feature_mismatches.append(
                    {"model_id": model_id, "error": "missing_registry_feature_list"}
                )
            elif expected != actual:
                feature_mismatches.append(
                    {
                        "model_id": model_id,
                        "error": "feature_list_sha256_mismatch",
                        "expected": expected,
                        "actual": actual,
                    }
                )

        expected_req = (self.content_hashes or {}).get("required_features_sha256")
        declared_req = (self.feature_space or {}).get("required_features_sha256")
        actual_req = _feature_hash(collect_required_features(self.registry or {}))
        if current_schema_strict and not expected_req:
            feature_mismatches.append({"error": "missing_required_features_hash"})
        if expected_req and expected_req != actual_req:
            feature_mismatches.append(
                {
                    "error": "required_features_sha256_mismatch",
                    "expected": expected_req,
                    "actual": actual_req,
                }
            )
        if declared_req and declared_req != actual_req:
            feature_mismatches.append(
                {
                    "error": "declared_required_features_sha256_mismatch",
                    "expected": declared_req,
                    "actual": actual_req,
                }
            )
        report["feature_list_mismatches"] = feature_mismatches

        stored_preprocessing = (self.content_hashes or {}).get(
            "preprocessing", {}
        ) or {}
        node_states = (self.validation_evidence or {}).get("node_states", {}) or {}
        for model_id in self.model_payloads or {}:
            expected = stored_preprocessing.get(model_id)
            state = (node_states.get(model_id) or {}).get("missingness_state")
            if not expected:
                if current_schema_strict:
                    report["preprocessing_mismatches"].append(
                        {"model_id": model_id, "error": "missing_preprocessing_hash"}
                    )
                continue
            if not state:
                report["preprocessing_mismatches"].append(
                    {"model_id": model_id, "error": "missing_preprocessing_state"}
                )
                continue
            actual = _sha256_bytes(
                json.dumps(state, sort_keys=True, default=str).encode("utf-8")
            )
            if actual != expected:
                report["preprocessing_mismatches"].append(
                    {"model_id": model_id, "expected": expected, "actual": actual}
                )

        semantics_version = (self.content_hashes or {}).get("vcf_semantics_version")
        if current_schema_strict and semantics_version != VCF_SEMANTICS_VERSION:
            report["preprocessing_mismatches"].append(
                {
                    "error": "missing_or_unsupported_vcf_semantics_version",
                    "expected": VCF_SEMANTICS_VERSION,
                    "actual": semantics_version,
                }
            )

        if (
            report["model_mismatches"]
            or report["manifest_mismatches"]
            or report["preprocessing_mismatches"]
            or feature_mismatches
        ):
            report["status"] = "failed"
            if strict:
                raise ValueError(
                    "Bundle content hash verification failed: "
                    f"models={len(report['model_mismatches'])} "
                    f"manifests={len(report['manifest_mismatches'])} "
                    f"preprocessing={len(report['preprocessing_mismatches'])} "
                    f"features={len(feature_mismatches)}"
                )
        return report

    def save(self, output_path: str | Path) -> Path:
        # Validate completeness before writing
        completeness = self.completeness or {}
        if completeness.get("fail_closed") and not completeness.get(
            "is_complete", True
        ):
            raise ValueError(
                "Refusing to write incomplete NetworkParser bundle. "
                f"Missing: {completeness.get('missing', [])}"
            )
        path = Path(output_path)
        if path.suffix == "":
            path = path.with_suffix(DEFAULT_BUNDLE_SUFFIX)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb") as handle:
            pickle.dump(self.to_payload(), handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Wrote NetworkParser model bundle: %s", path)
        return path


# -----------------------------------------------------------------------------
# Bundle construction / loading
# -----------------------------------------------------------------------------


def _read_manifest_record(
    tokens: List[Any], bundle_id: str, original_path: Path
) -> BundleFileRecord:
    df = pd.read_csv(original_path, sep="\t", dtype=str).fillna("")
    return BundleFileRecord(
        slot_tokens=list(tokens),
        bundle_id=bundle_id,
        original_path=str(original_path),
        kind="feature_manifest",
        columns=[str(c) for c in df.columns],
        records=df.to_dict(orient="records"),
        sep="\t",
    )


def _read_ranked_table_record(
    tokens: List[Any], bundle_id: str, original_path: Path
) -> BundleFileRecord:
    df = pd.read_csv(original_path)
    return BundleFileRecord(
        slot_tokens=list(tokens),
        bundle_id=bundle_id,
        original_path=str(original_path),
        kind="ranked_feature_table",
        columns=[str(c) for c in df.columns],
        records=df.to_dict(orient="records"),
        sep=",",
    )


def build_bundle_from_registry(
    registry_path: str | Path,
    output_path: Optional[str | Path] = None,
    *,
    include_model_payloads: bool = True,
    include_feature_manifests: bool = True,
    include_ranked_feature_tables: bool = True,
    fail_closed: bool = True,
) -> NetworkParserModelBundle:
    """
    Build a binary NetworkParser bundle from an existing hierarchy registry.

    Parameters
    ----------
    registry_path
        Path to hierarchy / two-level / hierarchical model registry JSON.
    output_path
        Optional path to write the ``.npb`` bundle immediately.
    include_model_payloads
        Embed trained model objects into the bundle (required for portability).
    include_feature_manifests
        Embed selected-feature manifests / context-sequence tables.
    include_ranked_feature_tables
        Embed optional RF-FDR / feature-result tables used to rank supporting
        markers in query reports.
    fail_closed
        If True (default), refuse to write when any successful trainable node is
        missing a required model payload or feature list.
    """
    registry_path = Path(registry_path)
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry not found: {registry_path}")

    registry_base = registry_path.parent
    registry = _read_json(registry_path)
    registry_for_bundle = copy.deepcopy(registry)

    model_payloads: Dict[str, Any] = {}
    model_sources: Dict[str, str] = {}
    model_hashes: Dict[str, str] = {}
    feature_list_hashes: Dict[str, str] = {}
    node_states: Dict[str, Any] = {}
    missing: List[Dict[str, Any]] = []
    required_model_ids: List[str] = []
    seen_model_ids: Dict[str, List[Any]] = {}

    for tokens, model_id, payload in iter_model_file_slots(registry):
        if not isinstance(payload, dict):
            continue
        if not _node_is_successful_trainable(payload):
            continue
        if model_id in seen_model_ids:
            missing.append(
                {
                    "model_id": model_id,
                    "reason": "duplicate_model_id",
                    "tokens": list(tokens),
                    "prior_tokens": list(seen_model_ids[model_id]),
                }
            )
        seen_model_ids.setdefault(model_id, list(tokens))
        required_model_ids.append(model_id)
        features = [str(f) for f in (payload.get("features") or [])]
        if not features:
            missing.append(
                {
                    "model_id": model_id,
                    "reason": "missing_feature_list",
                    "tokens": list(tokens),
                }
            )
        else:
            feature_list_hashes[model_id] = _feature_hash(features)

        # Capture threshold / calibration / missingness if present on the node
        model_summary = (
            payload.get("model") if isinstance(payload.get("model"), dict) else {}
        )
        node_states[model_id] = {
            "features": features,
            "feature_list_sha256": feature_list_hashes.get(model_id),
            "selected_decision_threshold": payload.get("selected_decision_threshold")
            or payload.get("decision_threshold")
            or (payload.get("threshold_selection") or {}).get(
                "selected_decision_threshold"
            )
            or model_summary.get("selected_decision_threshold"),
            "threshold_selection": payload.get("threshold_selection")
            or model_summary.get("threshold_selection"),
            "missingness_state": payload.get("missingness_state")
            or payload.get("preprocessing_state")
            or model_summary.get("missingness_state")
            or model_summary.get("preprocessing_state"),
            "baseline_alleles": payload.get("baseline_alleles"),
            "vcf_semantics_version": payload.get(
                "vcf_semantics_version", VCF_SEMANTICS_VERSION
            ),
            "hierarchy_path": payload.get("path") or payload.get("hierarchy_path"),
            "label_column": payload.get("label_column"),
        }

        if not include_model_payloads:
            missing.append(
                {
                    "model_id": model_id,
                    "reason": "model_payloads_disabled",
                    "tokens": list(tokens),
                }
            )
            continue

        model_file = _get_nested(registry, tokens)
        if tokens and tokens[-1] == "model_file":
            model_file = _get_nested(registry, tokens)
        else:
            model_file = payload.get("model_file")
            tokens = (
                list(tokens) + ["model_file"]
                if tokens[-1:] != ["model_file"]
                else list(tokens)
            )

        resolved = _resolve_path(str(model_file) if model_file else None, registry_base)
        if not model_file or resolved is None or not resolved.exists():
            missing.append(
                {
                    "model_id": model_id,
                    "reason": "missing_model_file",
                    "model_file": model_file,
                    "tokens": list(tokens),
                }
            )
            continue

        try:
            serialized_model = resolved.read_bytes()
            loaded = _load_pickle_or_joblib_bytes(serialized_model)
        except Exception as exc:
            missing.append(
                {
                    "model_id": model_id,
                    "reason": f"load_failed:{exc}",
                    "model_file": str(resolved),
                }
            )
            continue

        loaded_model = loaded.get("model") if isinstance(loaded, dict) else loaded
        loaded_missingness_state = (
            loaded.get("missingness_state") if isinstance(loaded, dict) else None
        ) or getattr(loaded_model, "networkparser_missingness_state", None)
        if node_states[model_id].get("missingness_state") is None:
            node_states[model_id]["missingness_state"] = loaded_missingness_state
        if not node_states[model_id].get("missingness_state"):
            missing.append(
                {
                    "model_id": model_id,
                    "reason": "missing_train_fitted_preprocessing_state",
                    "tokens": list(tokens),
                }
            )

        if model_id in model_payloads:
            missing.append(
                {
                    "model_id": model_id,
                    "reason": "duplicate_model_payload_slot",
                    "tokens": list(tokens),
                }
            )
            continue

        try:
            model_hashes[model_id] = _payload_content_hash(serialized_model)
        except ValueError as exc:
            missing.append(
                {
                    "model_id": model_id,
                    "reason": f"unhashable_payload:{exc}",
                    "model_file": str(resolved),
                }
            )
            continue

        # Preserve the exact serialized file representation. Hashing a newly
        # pickled in-memory object is not canonical for all custom estimators
        # and caused false integrity failures after a bundle round trip.
        model_payloads[model_id] = serialized_model
        model_sources[model_id] = str(resolved)
        section_tokens = (
            tokens[:-1] if tokens and tokens[-1] == "model_file" else tokens
        )
        _set_nested_payload_key(
            registry_for_bundle, section_tokens, "bundle_model_id", model_id
        )
        _set_nested_payload_key(
            registry_for_bundle, section_tokens, "features", features
        )
        _set_nested_payload_key(
            registry_for_bundle,
            section_tokens,
            "vcf_semantics_version",
            VCF_SEMANTICS_VERSION,
        )

    manifest_records: List[BundleFileRecord] = []
    manifest_hashes: Dict[str, str] = {}
    seen_manifest_content: Dict[str, str] = {}  # hash -> bundle_id
    if include_feature_manifests:
        seen_manifest_paths = set()
        for tokens, logical_name in iter_manifest_slots(registry):
            manifest_path = _get_nested(registry, tokens)
            resolved = _resolve_path(manifest_path, registry_base)
            if resolved is None or not resolved.exists():
                # Required even when path was not declared: successful nodes need manifests
                missing.append(
                    {
                        "manifest": logical_name,
                        "reason": "missing_required_manifest_file",
                        "path": manifest_path,
                        "tokens": list(tokens),
                    }
                )
                continue
            dedupe_key = (tuple(tokens), str(resolved))
            if dedupe_key in seen_manifest_paths:
                continue
            seen_manifest_paths.add(dedupe_key)
            bundle_id = f"{_slot_id(tokens)}__manifest"
            try:
                record = _read_manifest_record(tokens, bundle_id, resolved)
                raw = (
                    record.to_dataframe().to_csv(sep="\t", index=False).encode("utf-8")
                )
                mhash = _sha256_bytes(raw)
                if (
                    mhash in seen_manifest_content
                    and seen_manifest_content[mhash] != bundle_id
                ):
                    # Same content embedded under two IDs is OK; ambiguous *path* collisions flagged
                    pass
                seen_manifest_content[mhash] = bundle_id
                # Detect ambiguous different content under same logical name
                for existing in manifest_records:
                    if (
                        existing.bundle_id == bundle_id
                        and existing.original_path != str(resolved)
                    ):
                        missing.append(
                            {
                                "manifest": logical_name,
                                "reason": "ambiguous_duplicate_manifest",
                                "path": str(resolved),
                                "prior": existing.original_path,
                            }
                        )
                manifest_records.append(record)
                manifest_hashes[bundle_id] = mhash
            except Exception as exc:
                missing.append(
                    {
                        "manifest": logical_name,
                        "reason": f"embed_failed:{exc}",
                        "path": str(resolved),
                    }
                )
    ranked_table_records: List[BundleFileRecord] = []
    if include_ranked_feature_tables:
        seen_tables = set()
        for tokens, logical_name in iter_ranked_table_slots(registry):
            table_path = _get_nested(registry, tokens)
            resolved = _resolve_path(table_path, registry_base)
            if resolved is None or not resolved.exists():
                continue  # ranked tables optional
            dedupe_key = (tuple(tokens), str(resolved))
            if dedupe_key in seen_tables:
                continue
            seen_tables.add(dedupe_key)
            bundle_id = f"{_slot_id(tokens)}__ranked_table"
            try:
                ranked_table_records.append(
                    _read_ranked_table_record(tokens, bundle_id, resolved)
                )
            except Exception as exc:
                logger.warning(
                    "Could not embed ranked feature table %s: %s", resolved, exc
                )

    required_features = collect_required_features(registry)
    n_required = len(required_model_ids)
    n_unique_required = len(set(required_model_ids))
    n_embedded = len(model_payloads)
    # Complete only when every successful required node has exactly one unique payload
    models_complete = bool(
        include_model_payloads
        and n_required > 0
        and n_embedded == n_unique_required
        and n_unique_required == n_required
        and all(mid in model_payloads for mid in set(required_model_ids))
        and all(mid in model_hashes for mid in model_payloads)
        and all(
            mid in feature_list_hashes
            for mid in set(required_model_ids)
            if mid in model_payloads
        )
    )
    manifests_complete = bool(
        include_feature_manifests
        and len(manifest_records) > 0
        and len(manifest_hashes) == len(manifest_records)
    )

    # Raw-query capability requires more than a file called "manifest": every
    # required marker needs contig, position, alleles, baseline and a declared
    # reference identity/checksum. Otherwise the portable artifact is model-only.
    manifest_feature_rows: Dict[str, Dict[str, Any]] = {}
    reference_identities: set[Tuple[str, str]] = set()
    for record in manifest_records:
        frame = record.to_dataframe()
        feature_col = next(
            (
                name
                for name in ("Feature_ID", "feature", "feature_id")
                if name in frame.columns
            ),
            None,
        )
        if feature_col is None:
            continue
        for _, row in frame.iterrows():
            manifest_feature_rows.setdefault(
                str(row.get(feature_col, "")), row.to_dict()
            )
        ref_id_col = next(
            (
                name
                for name in ("Reference_id", "Reference_ID", "reference_id")
                if name in frame.columns
            ),
            None,
        )
        ref_hash_col = next(
            (
                name
                for name in (
                    "Reference_checksum_sha256",
                    "reference_checksum_sha256",
                    "Reference_checksum",
                )
                if name in frame.columns
            ),
            None,
        )
        if ref_id_col and ref_hash_col:
            for ref_id, ref_hash in zip(frame[ref_id_col], frame[ref_hash_col]):
                rid = str(ref_id).strip()
                rhash = str(ref_hash).strip().lower()
                if rid and rhash:
                    reference_identities.add((rid, rhash))

    raw_manifest_missing_features: List[str] = []
    for feature in required_features:
        row = manifest_feature_rows.get(str(feature))
        if row is None:
            raw_manifest_missing_features.append(str(feature))
            continue
        required_groups = (
            ("Sequence", "Contig", "Chromosome", "chrom"),
            ("Position", "POS", "position"),
            ("Ref_allele", "REF", "ref"),
            ("Alt_allele", "ALT", "alt"),
            ("Baseline_allele", "baseline_allele", "Baseline"),
        )
        if any(
            not any(str(row.get(column, "")).strip() for column in aliases)
            for aliases in required_groups
        ):
            raw_manifest_missing_features.append(str(feature))

    reference_identity_valid = len(reference_identities) == 1
    raw_query_manifest_complete = bool(
        manifests_complete
        and not raw_manifest_missing_features
        and reference_identity_valid
    )
    # Query-complete requires models + raw-query manifests + no missing slots.
    query_complete = bool(
        models_complete and raw_query_manifest_complete and len(missing) == 0
    )
    # Model-only includes bundles with no manifest or a manifest insufficient
    # for reconstructing raw VCF/FASTA/FASTQ marker calls.
    model_only = bool(
        include_model_payloads
        and models_complete
        and not query_complete
        and len(missing) == 0
    )
    is_complete = bool(query_complete or model_only)
    bundle_kind = (
        "query_complete"
        if query_complete
        else ("model_only" if model_only else "incomplete")
    )

    completeness = {
        "fail_closed": bool(fail_closed),
        "is_complete": bool(is_complete),
        "query_complete": bool(query_complete),
        "bundle_kind": bundle_kind,
        "n_required_successful_nodes": int(n_required),
        "n_unique_required_model_ids": int(n_unique_required),
        "n_embedded_models": int(n_embedded),
        "n_embedded_manifests": int(len(manifest_records)),
        "models_match_required": bool(models_complete),
        "raw_query_manifest_complete": raw_query_manifest_complete,
        "raw_manifest_missing_features": raw_manifest_missing_features[:100],
        "reference_identity_valid": reference_identity_valid,
        "reference_identities": sorted([list(value) for value in reference_identities]),
        "vcf_semantics_version": VCF_SEMANTICS_VERSION,
        "missing": missing,
    }

    if fail_closed and not is_complete:
        raise ValueError(
            "Bundle completeness validation failed (fail_closed=True). "
            f"required_nodes={n_required}; unique_ids={n_unique_required}; "
            f"embedded_models={n_embedded}; query_complete={query_complete}; "
            f"missing entries: {missing[:12]}" + (" ..." if len(missing) > 12 else "")
        )

    feature_space = {
        "required_feature_count": int(len(required_features)),
        "required_features": required_features,
        "required_features_sha256": _feature_hash(required_features),
        "per_model_feature_list_sha256": feature_list_hashes,
        "level1_feature_count": int(
            len(registry.get("level1", {}).get("features", []) or [])
        ),
        "n_embedded_models": int(len(model_payloads)),
        "n_embedded_manifests": int(len(manifest_records)),
        "n_embedded_ranked_tables": int(len(ranked_table_records)),
        "raw_query_capable": bool(query_complete),
        "reference_identity": (
            {
                "reference_id": next(iter(reference_identities))[0],
                "sha256": next(iter(reference_identities))[1],
            }
            if reference_identity_valid
            else None
        ),
    }

    validation_evidence = {
        "publication_summary": registry.get("publication_summary", {}),
        "level1_filter": registry.get("level1", {}).get("filter", {}),
        "level1_feature_panel_separability": registry.get("level1", {}).get(
            "feature_panel_separability", {}
        ),
        "level2_global_filter": registry.get("level2", {})
        .get("global_fallback", {})
        .get("filter", {}),
        "level2_global_feature_panel_separability": registry.get("level2", {})
        .get("global_fallback", {})
        .get("feature_panel_separability", {}),
        "node_states": node_states,
    }

    # Preprocessing hashes from stored missingness states when present
    preprocessing_hashes: Dict[str, str] = {}
    for mid, state in node_states.items():
        ms = state.get("missingness_state")
        if ms:
            preprocessing_hashes[mid] = _sha256_bytes(
                json.dumps(ms, sort_keys=True, default=str).encode("utf-8")
            )

    content_hashes = {
        "models": model_hashes,
        "model_payload_format": "serialized_file_bytes",
        "manifests": manifest_hashes,
        "feature_lists": feature_list_hashes,
        "preprocessing": preprocessing_hashes,
        "required_features_sha256": feature_space["required_features_sha256"],
        "vcf_semantics_version": VCF_SEMANTICS_VERSION,
    }

    notes = [
        "Bundle query mode is inference-only.",
        "The bundle stores trained models plus selected-feature manifests and context-sequence metadata.",
        "Statistical filtering, model screening, tree construction, and bootstrap evidence are training-time steps and are not rerun during bundled query.",
        PICKLE_TRUST_NOTE,
        "Complete bundles embed recursive hierarchy nodes, two-level models, terminal fallbacks, and global fallbacks.",
        f"vcf_semantics_version={VCF_SEMANTICS_VERSION}",
        f"bundle_kind={bundle_kind}",
    ]
    if bundle_kind == "model_only":
        notes.append(
            "MODEL-ONLY: selected-feature manifests are absent or insufficient for safe raw "
            "marker reconstruction. Raw VCF/FASTA/FASTQ query is rejected; provide a "
            "pre-aligned feature matrix matching the embedded feature lists."
        )

    bundle = NetworkParserModelBundle(
        schema_version=BUNDLE_SCHEMA_VERSION,
        created_at=_now_utc_iso(),
        registry=registry_for_bundle,
        registry_source=str(registry_path),
        model_payloads=model_payloads,
        model_sources=model_sources,
        manifest_records=manifest_records,
        ranked_table_records=ranked_table_records,
        feature_space=feature_space,
        validation_evidence=validation_evidence,
        notes=notes,
        content_hashes=content_hashes,
        completeness=completeness,
    )

    if output_path is not None:
        bundle.save(output_path)

    logger.info(
        "Built NetworkParser bundle | models=%d | manifests=%d | ranked_tables=%d | required_features=%d | complete=%s",
        len(model_payloads),
        len(manifest_records),
        len(ranked_table_records),
        len(required_features),
        is_complete,
    )
    return bundle


def save_bundle(bundle: NetworkParserModelBundle, output_path: str | Path) -> Path:
    """Save a bundle to disk (validates completeness when fail_closed)."""
    return bundle.save(output_path)


def _bundle_from_payload(payload: Dict[str, Any]) -> NetworkParserModelBundle:
    manifest_records = [
        BundleFileRecord(**record)
        for record in payload.get("manifest_records", [])
        if isinstance(record, dict)
    ]
    ranked_table_records = [
        BundleFileRecord(**record)
        for record in payload.get("ranked_table_records", [])
        if isinstance(record, dict)
    ]
    return NetworkParserModelBundle(
        schema_version=str(payload.get("schema_version", "unknown")),
        created_at=str(payload.get("created_at", "")),
        registry=payload.get("registry", {}) or {},
        registry_source=str(payload.get("registry_source", "")),
        model_payloads=payload.get("model_payloads", {}) or {},
        model_sources=payload.get("model_sources", {}) or {},
        manifest_records=manifest_records,
        ranked_table_records=ranked_table_records,
        feature_space=payload.get("feature_space", {}) or {},
        validation_evidence=payload.get("validation_evidence", {}) or {},
        notes=payload.get("notes", []) or [],
        content_hashes=payload.get("content_hashes", {}) or {},
        completeness=payload.get("completeness", {}) or {},
    )


def load_bundle(
    bundle_path: str | Path,
    *,
    compatibility_policy: str = DEFAULT_COMPATIBILITY_POLICY,
    verify_hashes: bool = True,
) -> NetworkParserModelBundle:
    """
    Load a NetworkParser binary bundle from disk.

    Parameters
    ----------
    compatibility_policy
        ``strict`` (default): reject unsupported schema versions.
        ``permissive``: warn and continue (not recommended for production).
    verify_hashes
        Re-hash embedded payloads and compare to stored content hashes.
    """
    path = Path(bundle_path)
    if not path.exists():
        raise FileNotFoundError(f"Bundle not found: {path}")

    logger.info(
        "Loading NetworkParser bundle (trusted-input-only pickle) | path=%s | policy=%s",
        path,
        compatibility_policy,
    )
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)

    # Backward compatibility for any early bundles saved as the dataclass object.
    if isinstance(payload, NetworkParserModelBundle):
        bundle = payload
    elif isinstance(payload, dict) and payload.get("__networkparser_model_bundle__"):
        bundle = _bundle_from_payload(payload)
    else:
        raise TypeError(f"File is not a NetworkParser model bundle: {path}")

    policy = str(compatibility_policy or DEFAULT_COMPATIBILITY_POLICY).strip().lower()
    schema = str(bundle.schema_version or "unknown")
    if schema not in SUPPORTED_BUNDLE_SCHEMA_VERSIONS:
        msg = (
            f"Bundle schema version '{schema}' is not in supported set "
            f"{sorted(SUPPORTED_BUNDLE_SCHEMA_VERSIONS)} (runtime={BUNDLE_SCHEMA_VERSION})."
        )
        if policy == "strict":
            raise ValueError(
                msg + " Set compatibility_policy='permissive' to override."
            )
        logger.warning("%s Continuing under permissive policy.", msg)
    elif schema != BUNDLE_SCHEMA_VERSION:
        logger.info(
            "Loading older supported bundle schema | bundle=%s | runtime=%s",
            schema,
            BUNDLE_SCHEMA_VERSION,
        )

    if verify_hashes:
        # Legacy 1.0 may lack hashes — verify_content_hashes handles that.
        strict_hashes = policy == "strict" and schema not in {"1.0", "unknown"}
        try:
            bundle.verify_content_hashes(strict=strict_hashes)
        except ValueError:
            if policy == "strict":
                raise
            logger.warning("Bundle hash verification failed under permissive policy.")

    return bundle


# -----------------------------------------------------------------------------
# Bundled query engine
# -----------------------------------------------------------------------------


class BundledNetworkParserQueryEngine(NetworkParserQueryEngine):
    """Query engine that loads models/manifests from a binary bundle only."""

    def __init__(
        self,
        bundle: NetworkParserModelBundle,
        config: NetworkParserConfig,
        runtime_dir: str | Path,
    ):
        runtime = bundle.materialize_runtime_files(Path(runtime_dir))
        self.bundle = bundle
        self.registry_path = Path(runtime["registry_path"])
        self.registry_base = Path(runtime["runtime_dir"])
        self.registry = runtime["registry"]
        self.config = apply_trained_vcf_config(config, self.registry)
        self._init_query_caches()

    def query(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Reject raw inputs when bundle evidence supports matrix inference only."""
        genomic_path = kwargs.get("genomic_path")
        if genomic_path is None and args:
            genomic_path = args[0]
        requested = str(kwargs.get("query_input_type", "auto") or "auto").lower()
        raw_types = {"vcf", "fasta", "fastq", "raw_sequence", "raw_fasta", "sequence"}
        inferred = requested
        candidate = Path(str(genomic_path)) if genomic_path is not None else None
        if requested == "auto" and candidate is not None:
            names: List[str] = []
            if candidate.is_file():
                names = [candidate.name.lower()]
            elif candidate.is_dir():
                names = [
                    path.name.lower() for path in candidate.iterdir() if path.is_file()
                ]
            if any(
                name.endswith((".fastq", ".fq", ".fastq.gz", ".fq.gz"))
                for name in names
            ):
                inferred = "fastq"
            elif any(
                name.endswith((".vcf", ".vcf.gz", ".g.vcf", ".g.vcf.gz"))
                for name in names
            ):
                inferred = "vcf"
            elif any(
                name.endswith((".fa", ".fna", ".fasta", ".fas")) for name in names
            ):
                inferred = "fasta"
            else:
                inferred = "matrix"
        if inferred in raw_types and not bool(
            self.bundle.completeness.get("query_complete")
        ):
            raise ValueError(
                "Bundle is model-only and cannot safely reconstruct required markers from "
                f"{inferred.upper()} input. Provide a pre-aligned matrix or rebuild a "
                "query-complete bundle with coordinate, baseline, reference ID/checksum, "
                "and train-fitted preprocessing state."
            )
        return super().query(*args, **kwargs)

    def _payload_from_registry_section(
        self, section: Dict[str, Any], level_name: str
    ) -> Any:
        model_id = section.get("bundle_model_id") if isinstance(section, dict) else None
        if model_id and str(model_id) in self.bundle.model_payloads:
            embedded = self.bundle.model_payloads[str(model_id)]
            if isinstance(embedded, (bytes, bytearray, memoryview)):
                return _load_pickle_or_joblib_bytes(bytes(embedded))
            return embedded

        # Fail closed for complete portable bundles: do not silently load
        # external training-directory paths that may not exist after move.
        model_file = section.get("model_file") if isinstance(section, dict) else None
        resolved = _resolve_path(model_file, self.registry_base)
        if resolved is not None and resolved.exists():
            logger.warning(
                "Loading %s model from materialized path rather than embedded payload: %s",
                level_name,
                resolved,
            )
            return _load_pickle_or_joblib(resolved)

        raise ValueError(
            f"Bundle is missing a usable {level_name} model payload "
            f"(bundle_model_id={model_id!r}). Bundle is not portable/complete."
        )

    def _load_level1(
        self,
    ) -> Tuple[List[str], Any, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        level1 = self.registry.get("level1", {})
        features = [str(f) for f in level1.get("features", [])]
        if not features:
            raise ValueError("Bundle registry is missing Level 1 selected features.")

        payload = self._payload_from_registry_section(level1, "Level 1")
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
            or selected.get("status") not in (None, "success")
            or not (selected.get("bundle_model_id") or selected.get("model_file"))
        ):
            selected = (
                level2.get("global_fallback", {}) if isinstance(level2, dict) else {}
            )
            source = "global_fallback"
            if (
                not selected
                or selected.get("status") not in (None, "success")
                or not (selected.get("bundle_model_id") or selected.get("model_file"))
            ):
                selected = (
                    level2.get("global_binary_fallback", {})
                    if isinstance(level2, dict)
                    else {}
                )
                source = "global_binary_fallback"

        features = [str(f) for f in selected.get("features", [])]
        if not features:
            raise ValueError(
                "No usable Level 2 feature set found for predicted Level 1 group "
                "and no global fallback is available in the bundle."
            )

        payload = self._payload_from_registry_section(selected, f"Level 2 ({source})")
        ranked = read_ranked_feature_table(
            selected.get("filter", {}), self.registry_base
        )
        model_importance = extract_model_importance(payload, features)
        result = (source, features, payload, ranked, model_importance)
        with self._level2_payload_cache_lock:
            self._level2_payload_cache[cache_key] = result
        return result

    def _load_hierarchy_node_payload(
        self,
        node: Dict[str, Any],
    ) -> Tuple[List[str], Any, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Load a recursive hierarchy / fallback node from embedded bundle payloads.

        This overrides the exact QueryEngine method used at query time
        (``_load_hierarchy_node_payload``). A similarly named unused method must
        not be introduced in its place.
        """
        model_id = node.get("bundle_model_id") if isinstance(node, dict) else None
        cache_key = f"bundle::{model_id}" if model_id else f"bundle::{id(node)}"
        with self._hierarchy_payload_cache_lock:
            cached = self._hierarchy_payload_cache.get(cache_key)
            if cached is not None:
                return cached

        features = [str(f) for f in node.get("features", [])]
        if not features:
            raise ValueError(
                f"Hierarchy node for label '{node.get('label_column')}' has no selected features in bundle."
            )
        payload = self._payload_from_registry_section(
            node,
            f"Hierarchy node ({node.get('label_column', 'label')})",
        )
        ranked = read_ranked_feature_table(node.get("filter", {}), self.registry_base)
        model_importance = extract_model_importance(payload, features)
        result = (features, payload, ranked, model_importance)
        with self._hierarchy_payload_cache_lock:
            self._hierarchy_payload_cache[cache_key] = result
        return result


def query_bundle(
    *,
    bundle_path: str | Path,
    genomic_path: str,
    output_dir: str | Path,
    config: NetworkParserConfig,
    ref_fasta: Optional[str] = None,
    max_markers: int = 10,
    n_jobs: Optional[int] = None,
    query_input_type: str = "auto",
    raw_sequence_mapping_mode: str = "auto",
    meta_path: Optional[str] = None,
    level1_label: Optional[str] = None,
    level2_label: Optional[str] = None,
    hierarchy_labels: Optional[Sequence[str]] = None,
    sample_id_column: Optional[str] = None,
    compatibility_policy: str = DEFAULT_COMPATIBILITY_POLICY,
    verify_hashes: bool = True,
) -> pd.DataFrame:
    """
    Run query inference from a binary NetworkParser bundle.

    Optional ``meta_path`` triggers post-query evaluation when labels are provided.
    """
    out = _ensure_dir(Path(output_dir))
    bundle = load_bundle(
        bundle_path,
        compatibility_policy=compatibility_policy,
        verify_hashes=verify_hashes,
    )
    engine = BundledNetworkParserQueryEngine(
        bundle=bundle,
        config=config,
        runtime_dir=out / "_bundle_runtime",
    )
    predictions = engine.query(
        genomic_path=genomic_path,
        output_dir=str(out),
        ref_fasta=ref_fasta,
        max_markers=int(max_markers),
        n_jobs=n_jobs,
        query_input_type=query_input_type,
        raw_sequence_mapping_mode=raw_sequence_mapping_mode,
    )

    if meta_path:
        try:
            from mtb_amr_classifier.model_evaluation import (
                evaluate_prediction_table,
                run_networkparser_evaluation,
            )
        except Exception:  # pragma: no cover
            try:
                from model_evaluation import (  # type: ignore
                    evaluate_prediction_table,
                    run_networkparser_evaluation,
                )
            except Exception:
                logger.warning(
                    "Skipping post-query evaluation: model_evaluation is not part of "
                    "MTB_AMR_Classifier. Use NetworkParser evaluate-hierarchy if needed."
                )
                return predictions

        # Evaluate exactly the dataframe returned by this invocation. This
        # cannot accidentally select a stale query_predictions file from an
        # older nested output directory.
        pred_path = out / "current_query_predictions_for_evaluation.csv"
        predictions.to_csv(pred_path, index=False)
        eval_dir = out / "evaluate"
        labels = list(hierarchy_labels) if hierarchy_labels else None
        if not labels:
            labels = [x for x in (level1_label, level2_label) if x]
        if labels and len(labels) >= 2:
            run_networkparser_evaluation(
                predictions_path=pred_path,
                meta_path=meta_path,
                hierarchy_labels=labels,
                output_dir=eval_dir,
                metadata_sample_id_column=sample_id_column,
            )
        elif labels:
            evaluate_prediction_table(
                predictions_path=pred_path,
                meta_path=meta_path,
                label_column=str(labels[0]),
                prediction_column="predicted_level1",
                output_dir=eval_dir,
                metadata_sample_id_column=sample_id_column,
            )
        else:
            logger.warning(
                "meta_path provided to query_bundle but no hierarchy/level labels; skipping evaluation."
            )

    return predictions


__all__ = [
    "NetworkParserModelBundle",
    "BundledNetworkParserQueryEngine",
    "BUNDLE_SCHEMA_VERSION",
    "SUPPORTED_BUNDLE_SCHEMA_VERSIONS",
    "DEFAULT_COMPATIBILITY_POLICY",
    "PICKLE_TRUST_NOTE",
    "build_bundle_from_registry",
    "save_bundle",
    "load_bundle",
    "query_bundle",
]
