# network_parser/utils.py
"""
Utility functions for NetworkParser.

Includes:
- Safe YAML config loading (optional dependency: PyYAML)
- CLI args → NetworkParserConfig builder
- Filesystem helpers (ensure_dir)
- Timestamp helper
- JSON save helper (save_json)

Keep this module lightweight because it is imported early by the CLI/package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from contextvars import ContextVar
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

try:
    from .config import NetworkParserConfig
except ImportError:  # pragma: no cover - supports direct source-tree execution
    from config import NetworkParserConfig  # type: ignore

logger = logging.getLogger(__name__)

# Per-task override for nested parallelism (outer workers set this so inner
# sklearn/joblib stages do not oversubscribe cores).
_parallel_inner_n_jobs: ContextVar[Optional[int]] = ContextVar(
    "mtb_amr_classifier_parallel_inner_n_jobs", default=None
)

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

# ──────────────────────────────────────────────────────────────
# General helpers expected by the pipeline
# ──────────────────────────────────────────────────────────────


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists and return it as a Path.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """
    Return a filesystem-friendly timestamp string.
    """
    return datetime.now().strftime(fmt)


def json_default(obj: Any) -> Any:
    """JSON serializer for common NetworkParser runtime objects."""
    try:
        import numpy as _np  # local import keeps utils lightweight at import time

        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, (_np.ndarray,)):
            return obj.tolist()
    except Exception:
        pass

    try:
        import pandas as _pd

        if isinstance(obj, (_pd.Series, _pd.Index)):
            return obj.tolist()
        if isinstance(obj, _pd.DataFrame):
            return obj.to_dict(orient="records")
    except Exception:
        pass

    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        try:
            return vars(obj)
        except Exception:
            pass
    return str(obj)


def save_json(data: Any, out_path: Union[str, Path], indent: int = 2) -> Path:
    """
    Save JSON to disk with sane defaults.
    Creates parent directories automatically.

    Parameters
    ----------
    data : Any
        JSON-serializable object.
    out_path : str | Path
        Output file path.
    indent : int
        JSON indent level.

    Returns
    -------
    Path
        Path to written JSON file.
    """
    out_path = Path(out_path)
    if out_path.parent:
        ensure_dir(out_path.parent)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=json_default)
        f.write("\n")

    logger.info("Wrote JSON: %s", out_path)
    return out_path


def normalize_sample_id(
    value: Any,
    strip_library_suffix: bool = True,
) -> str:
    """
    Normalize sample identifiers consistently across training, query, and
    metadata-alignment stages.

    This deliberately performs conservative filename cleanup plus the
    existing NetworkParser library-suffix cleanup. Path-like VCF sample names
    emitted by BAM-based callers are collapsed to an unambiguous ERR/SRR
    accession when the same accession is repeated in the path, for example
    ``SRR5535811/SRR5535811.sorted.rmdup.bam`` -> ``SRR5535811``.
    """
    sample = str(value).strip()

    if sample.lower() in {"", "nan", "none", "null", "na", "n/a"}:
        return ""

    if strip_library_suffix:
        accessions = re.findall(r"(?i)(?<![A-Z0-9])((?:ERR|SRR)\d+)(?!\d)", sample)
        unique_accessions = list(dict.fromkeys(match.upper() for match in accessions))
        if len(unique_accessions) == 1:
            return unique_accessions[0]

    sample = re.sub(r"(?i)(\.vcf\.gz|\.vcf|\.bcf\.gz|\.bcf|\.gz)$", "", sample)

    if strip_library_suffix:
        sample = re.sub(r"(?i)_library[0-9]+$", "", sample)

    return sample


# ──────────────────────────────────────────────────────────────
# User-facing pipeline logging helpers
# ──────────────────────────────────────────────────────────────


def _compact_log_value(value: Any) -> str:
    """Return a compact, stable representation for user-facing log fields."""
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        if len(items) > 6:
            shown = ",".join(str(x) for x in items[:6])
            return f"[{shown},...+{len(items) - 6}]"
        return "[" + ",".join(str(x) for x in items) + "]"
    return str(value)


def format_log_kv(**fields: Any) -> str:
    """Format key=value fields for concise, readable pipeline logs."""
    clean = []
    for key, value in fields.items():
        if value is None:
            continue
        clean.append(f"{key}={_compact_log_value(value)}")
    return " | ".join(clean)


def _normalize_sentence(text: str) -> str:
    """Return a single sentence with stable terminal punctuation."""
    body = str(text or "").strip()
    if not body:
        return ""
    if body[-1] not in ".!?":
        body += "."
    return body


def _join_step_narrative(happened: str, reason: str) -> str:
    """Combine action and rationale into plain prose without section labels."""
    parts = [_normalize_sentence(part) for part in (happened, reason)]
    return " ".join(part for part in parts if part)


def progress_enabled() -> bool:
    """Return whether tqdm progress bars should be shown."""
    flag = os.environ.get("NETWORKPARSER_DISABLE_PROGRESS", "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return False
    isatty = getattr(sys.stderr, "isatty", None)
    return bool(isatty and isatty())


def progress_iter(
    iterable: Iterable[Any],
    *,
    desc: str = "",
    total: int | None = None,
    unit: str = "",
    leave: bool = False,
    position: int | None = None,
):
    """Wrap an iterable with tqdm when progress display is enabled."""
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover - tqdm is an environment dependency
        return iterable

    return tqdm(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        leave=leave,
        position=position,
        disable=not progress_enabled(),
        dynamic_ncols=True,
    )


class PipelineProgress:
    """Track high-level pipeline stage completion with a progress bar."""

    def __init__(self, stages: Sequence[str], *, title: str = "Pipeline") -> None:
        self._stages = [str(stage) for stage in stages]
        self._title = str(title)
        self._bar = None
        if not self._stages:
            return
        try:
            from tqdm.auto import tqdm
        except Exception:  # pragma: no cover
            return
        self._bar = tqdm(
            total=len(self._stages),
            desc=self._title,
            unit="stage",
            leave=True,
            disable=not progress_enabled(),
            dynamic_ncols=True,
        )

    def begin_stage(self, name: str) -> None:
        if self._bar is not None:
            self._bar.set_description(f"{self._title} — {name}")

    def complete_stage(self, name: str = "") -> None:
        if self._bar is None:
            return
        if name:
            self._bar.set_postfix_str(str(name)[:48], refresh=False)
        self._bar.update(1)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def __enter__(self) -> "PipelineProgress":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def log_pipeline_header(log: logging.Logger, title: str, **fields: Any) -> None:
    """Emit a clear run-level banner without exposing feature names or labels."""
    line = "=" * 78
    log.info(line)
    log.info("%s", title)
    if fields:
        log.info("Run settings | %s", format_log_kv(**fields))
    log.info(line)


def log_stage_start(
    log: logging.Logger,
    stage: Union[int, str],
    name: str,
    *,
    progress: PipelineProgress | None = None,
    **fields: Any,
) -> None:
    """Emit a standard stage-start message."""
    if progress is not None:
        progress.begin_stage(name)
    suffix = f" | {format_log_kv(**fields)}" if fields else ""
    log.info("▶ Stage %s — %s%s", stage, name, suffix)


def log_stage_complete(
    log: logging.Logger,
    stage: Union[int, str],
    name: str,
    *,
    progress: PipelineProgress | None = None,
    **fields: Any,
) -> None:
    """Emit a standard stage-complete message."""
    suffix = f" | {format_log_kv(**fields)}" if fields else ""
    log.info("✓ Stage %s complete — %s%s", stage, name, suffix)
    if progress is not None:
        progress.complete_stage(name)


def log_branch_decision(
    log: logging.Logger, branch: str, status: str, **fields: Any
) -> None:
    """Emit a concise branch decision message for optional workflow branches."""
    fields = dict(fields)
    reason = fields.pop("reason", None)
    suffix = f" | {format_log_kv(**fields)}" if fields else ""
    if reason:
        reason_text = _normalize_sentence(str(reason)).rstrip(".")
        log.info("Branch decision — %s: %s; %s%s", branch, status, reason_text, suffix)
    else:
        log.info("Branch decision — %s: %s%s", branch, status, suffix)


def log_artifact(log: logging.Logger, label: str, path: Union[str, Path]) -> None:
    """Emit a standard artifact message."""
    log.info("Artifact written — %s | path=%s", label, str(path))


def log_final_run_summary(
    log: logging.Logger,
    *,
    title: str,
    sections: Iterable[Dict[str, Any]],
    artifacts: Optional[Dict[str, Union[str, Path]]] = None,
    warnings: Optional[Iterable[Dict[str, Any]]] = None,
) -> None:
    """Log a compact final summary without using a separate 'reason' label.

    Each section should include:
      - name: short stage label
      - message: one sentence explaining what the stage means
      - optional fields: compact movement/status values
    """
    line = "=" * 78
    log.info(line)
    log.info("%s", title)
    log.info(line)

    for section in sections:
        if not isinstance(section, dict):
            continue
        name = str(section.get("name", "Stage"))
        message = str(section.get("message", "")).strip()
        fields = section.get("fields", {})
        field_text = (
            format_log_kv(**fields) if isinstance(fields, dict) and fields else ""
        )
        suffix = f" | {field_text}" if field_text else ""
        if message:
            log.info("%s: %s%s", name, message, suffix)
        elif suffix:
            log.info("%s:%s", name, suffix)

    if artifacts:
        log.info("Outputs:")
        for label, path in artifacts.items():
            if path:
                log.info("  %s: %s", label, str(path))

    warning_list = [w for w in (warnings or []) if isinstance(w, dict)]
    if warning_list:
        log.info("Warnings captured in run_audit.json: %d", len(warning_list))

    log.info(line)


def audit_warning(
    *,
    stage: str,
    message: str,
    code: str = "warning",
    severity: str = "warning",
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a standardized warning record for run_audit.json."""
    return {
        "timestamp": timestamp(),
        "severity": str(severity),
        "stage": str(stage),
        "code": str(code),
        "message": str(message),
        "details": details or {},
    }


def collect_common_warnings(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect high-level warnings from a NetworkParser result payload.

    This intentionally avoids pre-flight validation. It only records issues that
    were observed during the actual run, so the audit trail remains faithful to
    what happened.
    """
    warnings: List[Dict[str, Any]] = []

    feature_filter = (
        results.get("feature_filtering", {}) if isinstance(results, dict) else {}
    )
    if isinstance(feature_filter, dict):
        if feature_filter.get("used_fallback_unfiltered_matrix"):
            warnings.append(
                audit_warning(
                    stage="central_feature_filtering",
                    code="exploratory_unfiltered_fallback",
                    message=(
                        "Central filtering retained no supported features and an unfiltered fallback was used. "
                        "Treat this as exploratory rather than publication-grade FDR-supported output."
                    ),
                    details={
                        "method": feature_filter.get("method"),
                        "fallback_strategy": feature_filter.get("fallback_strategy"),
                    },
                )
            )
        if feature_filter.get("status") in {"skipped", "disabled"}:
            warnings.append(
                audit_warning(
                    stage="central_feature_filtering",
                    code="central_filtering_skipped",
                    message="Central statistical filtering was skipped, so the downstream matrix is not FDR-filtered.",
                    details={"status": feature_filter.get("status")},
                )
            )

    panel = (
        results.get("feature_panel_separability", {})
        if isinstance(results, dict)
        else {}
    )
    if isinstance(panel, dict):
        status = str(panel.get("status", "")).lower()
        reason = str(panel.get("reason", "")).lower()
        if status in {"skipped", "failed"} or "fallback" in reason:
            warnings.append(
                audit_warning(
                    stage="feature_panel_selection",
                    code="feature_panel_not_cleanly_selected",
                    message="The ranked feature-panel step did not complete as a clean smallest-passing panel selection.",
                    details={
                        "status": panel.get("status"),
                        "reason": panel.get("reason"),
                    },
                )
            )

    ml = results.get("ml_protocol", {}) if isinstance(results, dict) else {}
    if isinstance(ml, dict) and ml:
        selector = (
            ml.get("selector", {}) if isinstance(ml.get("selector", {}), dict) else {}
        )
        selector_status = str(selector.get("selector_status", "")).lower()
        if selector_status and selector_status not in {"success", "ok"}:
            warnings.append(
                audit_warning(
                    stage="ml_protocol",
                    code="model_selector_status",
                    message="The model selector reported a non-standard status during model screening.",
                    details={"selector_status": selector.get("selector_status")},
                )
            )

    discovery = results.get("discovery", {}) if isinstance(results, dict) else {}
    if not discovery and results.get("pipeline_mode") in {"both", "decision_tree_only"}:
        warnings.append(
            audit_warning(
                stage="decision_tree_branch",
                code="decision_tree_not_run",
                message="Decision-tree interpretability output was not generated for this run.",
                details={"pipeline_mode": results.get("pipeline_mode")},
            )
        )

    return warnings


def write_run_audit(
    output_dir: Union[str, Path],
    audit_payload: Dict[str, Any],
    filename: str = "run_audit.json",
) -> Path:
    """Write the run audit file into the requested output directory."""
    out = ensure_dir(output_dir) / filename
    save_json(audit_payload, out)
    return out


def _safe_stage_name(stage_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(stage_name).strip())
    return safe.strip("_") or "stage"


# Schema for stage checkpoints. Bump when on-disk layout or hash keys change.
CHECKPOINT_SCHEMA_VERSION = "1.0"


def stable_json_dumps(obj: Any) -> str:
    """Deterministic JSON serialization for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Union[str, Path], *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_dataframe(df: Any) -> str:
    """Content hash of a DataFrame via stable CSV bytes (index + columns + values)."""
    if df is None:
        return sha256_text("null")
    try:
        import pandas as _pd  # local to avoid circular import patterns

        if not isinstance(df, _pd.DataFrame):
            return sha256_text(stable_json_dumps(df))
        csv_bytes = df.to_csv().encode("utf-8", errors="replace")
        return hashlib.sha256(csv_bytes).hexdigest()
    except Exception:
        return sha256_text(stable_json_dumps(str(type(df))))


def build_checkpoint_hashes(
    *,
    input_paths: Optional[Dict[str, Union[str, Path, None]]] = None,
    config_subset: Optional[Dict[str, Any]] = None,
    content_objects: Optional[Dict[str, Any]] = None,
    schema_version: str = CHECKPOINT_SCHEMA_VERSION,
) -> Dict[str, Any]:
    """
    Build the standard hash block stored with stage checkpoints.

    Keys
    ----
    schema_version : str
    input_hashes   : path → sha256 (missing paths recorded as null)
    config_hash    : sha256 of stable JSON config subset
    content_hashes : name → sha256 of dataframes / serializable objects
    """
    input_hashes: Dict[str, Optional[str]] = {}
    for key, path in (input_paths or {}).items():
        if path is None or str(path).strip() == "":
            input_hashes[str(key)] = None
            continue
        p = Path(path)
        if p.is_file():
            try:
                input_hashes[str(key)] = sha256_file(p)
            except OSError:
                input_hashes[str(key)] = None
        else:
            # Directory or non-file: hash path string + mtime listing fingerprint
            input_hashes[str(key)] = sha256_text(
                str(p.resolve()) if p.exists() else str(p)
            )

    config_hash = sha256_text(stable_json_dumps(config_subset or {}))

    content_hashes: Dict[str, str] = {}
    for name, obj in (content_objects or {}).items():
        if hasattr(obj, "to_csv"):
            content_hashes[str(name)] = sha256_dataframe(obj)
        else:
            content_hashes[str(name)] = sha256_text(stable_json_dumps(obj))

    return {
        "schema_version": str(schema_version),
        "input_hashes": input_hashes,
        "config_hash": config_hash,
        "content_hashes": content_hashes,
    }


def checkpoint_hashes_compatible(
    stored: Optional[Dict[str, Any]],
    expected: Optional[Dict[str, Any]],
) -> Tuple[bool, str]:
    """
    Return (ok, reason). Refuse reuse when schema, input, config, or content hashes diverge.
    """
    if not stored:
        return False, "missing_stored_hashes"
    if not expected:
        return True, "no_expected_hashes"

    stored_schema = str(stored.get("schema_version", ""))
    expected_schema = str(expected.get("schema_version", CHECKPOINT_SCHEMA_VERSION))
    if stored_schema != expected_schema:
        return False, f"schema_mismatch:{stored_schema}!={expected_schema}"

    if str(stored.get("config_hash", "")) != str(expected.get("config_hash", "")):
        return False, "config_hash_mismatch"

    stored_inputs = stored.get("input_hashes") or {}
    expected_inputs = expected.get("input_hashes") or {}
    if not isinstance(stored_inputs, dict) or not isinstance(expected_inputs, dict):
        return False, "input_hashes_malformed"
    for key, expected_val in expected_inputs.items():
        if key not in stored_inputs:
            return False, f"input_hash_missing:{key}"
        if stored_inputs.get(key) != expected_val:
            return False, f"input_hash_mismatch:{key}"

    stored_content = stored.get("content_hashes") or {}
    expected_content = expected.get("content_hashes") or {}
    if not isinstance(stored_content, dict) or not isinstance(expected_content, dict):
        return False, "content_hashes_malformed"
    for key, expected_val in expected_content.items():
        if key not in stored_content:
            return False, f"content_hash_missing:{key}"
        if stored_content.get(key) != expected_val:
            return False, f"content_hash_mismatch:{key}"

    return True, "compatible"


def checkpoint_dir(output_dir: Union[str, Path]) -> Path:
    return ensure_dir(Path(output_dir) / "_checkpoints")


def write_stage_checkpoint(
    output_dir: Union[str, Path],
    stage_name: str,
    payload: Dict[str, Any],
    *,
    status: str = "complete",
    hashes: Optional[Dict[str, Any]] = None,
    schema_version: str = CHECKPOINT_SCHEMA_VERSION,
) -> Path:
    """Write a stage checkpoint metadata file with optional compatibility hashes."""
    path = checkpoint_dir(output_dir) / f"{_safe_stage_name(stage_name)}.json"
    checkpoint_payload: Dict[str, Any] = {
        "stage_name": str(stage_name),
        "status": str(status),
        "timestamp": timestamp(),
        "schema_version": str(schema_version),
        "payload": payload,
    }
    if hashes is not None:
        checkpoint_payload["hashes"] = hashes
    save_json(checkpoint_payload, path)
    return path


def load_stage_checkpoint(
    output_dir: Union[str, Path],
    stage_name: str,
    *,
    expected_hashes: Optional[Dict[str, Any]] = None,
    refuse_incompatible: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Load a stage checkpoint metadata file if present.

    When ``expected_hashes`` is provided and ``refuse_incompatible`` is True,
    incompatible checkpoints are discarded (returns None) so the stage re-runs.
    """
    path = checkpoint_dir(output_dir) / f"{_safe_stage_name(stage_name)}.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    if expected_hashes is not None and refuse_incompatible:
        stored_hashes = (
            data.get("hashes") if isinstance(data.get("hashes"), dict) else None
        )
        # Legacy checkpoints without hashes are incompatible when expectation is set.
        if stored_hashes is None:
            logger.warning(
                "Refusing checkpoint %s: legacy checkpoint lacks compatibility hashes",
                path,
            )
            return None
        ok, reason = checkpoint_hashes_compatible(stored_hashes, expected_hashes)
        if not ok:
            logger.warning(
                "Refusing incompatible checkpoint %s: %s",
                path,
                reason,
            )
            return None

    return data


def log_flow_step(
    log: logging.Logger,
    *,
    step: str,
    happened: str,
    reason: str,
    before_samples: int | None = None,
    before_features: int | None = None,
    after_samples: int | None = None,
    after_features: int | None = None,
    threshold: str | None = None,
    status: str | None = None,
    artifact: Union[str, Path, None] = None,
) -> None:
    """Log a user-facing explanation block with action, rationale, and movement.

    This is intentionally INFO-level because it explains the statistical and
    preprocessing rationale. Exact feature/sample identifiers should remain in
    DEBUG logs or audit artifacts.
    """
    log.info("%s", step)
    narrative = _join_step_narrative(happened, reason)
    if narrative:
        log.info("  %s", narrative)

    movement = format_log_kv(
        input_samples=before_samples,
        input_features=before_features,
        output_samples=after_samples,
        output_features=after_features,
        threshold=threshold,
        status=status,
    )
    if movement:
        log.info("  %s", movement)
    if artifact is not None:
        log.info("  Wrote audit record to %s", str(artifact))


def log_filter_step(
    log: logging.Logger,
    *,
    filter_name: str,
    happened: str,
    reason: str,
    before_samples: int | None = None,
    before_features: int | None = None,
    after_samples: int | None = None,
    after_features: int | None = None,
    threshold: str | None = None,
    status: str | None = None,
    artifact: Union[str, Path, None] = None,
) -> None:
    """Convenience wrapper for filters/preprocessing gates."""
    log_flow_step(
        log,
        step=f"Filter — {filter_name}",
        happened=happened,
        reason=reason,
        before_samples=before_samples,
        before_features=before_features,
        after_samples=after_samples,
        after_features=after_features,
        threshold=threshold,
        status=status,
        artifact=artifact,
    )


# ──────────────────────────────────────────────────────────────
# Config handling
# ──────────────────────────────────────────────────────────────


def create_config_from_args(args: argparse.Namespace) -> NetworkParserConfig:
    """
    Create a NetworkParserConfig object from CLI args.

    Notes:
    - Only updates fields that are present and not None.
    - Keeps defaults from NetworkParserConfig for everything else.
    """
    config = NetworkParserConfig()

    # Defensive: use getattr because some args may not exist depending on CLI evolution
    for attr in [
        "bootstrap_iterations",
        "confidence_threshold",
        "max_interaction_order",
        "fdr_threshold",
        "min_group_size",
        "correction_method",
        "max_workers",
        "chunk_size",
        "cross_validation_folds",
        "stability_threshold",
        "min_bootstrap_support",
    ]:
        val = getattr(args, attr, None)
        if val is not None:
            setattr(config, attr, val)

    # Output formats
    out_fmt = getattr(args, "output_format", None)
    if out_fmt:
        setattr(
            config,
            "output_formats",
            [x.strip() for x in out_fmt.split(",") if x.strip()],
        )

    # Boolean flags
    # (Only set if present; otherwise leave config defaults as-is)
    for battr in ["memory_efficient", "include_matrices", "generate_plots"]:
        if hasattr(args, battr):
            setattr(config, battr, bool(getattr(args, battr)))

    return config


def load_config_file(config_path: Union[str, Path]) -> NetworkParserConfig:
    """
    Load configuration from a YAML file into NetworkParserConfig.

    Requires PyYAML to be installed (conda-forge: pyyaml).

    Parameters
    ----------
    config_path : str | Path
        Path to YAML config file.

    Returns
    -------
    NetworkParserConfig
    """
    if yaml is None:
        raise ImportError(
            "PyYAML is required to load YAML config files. "
            "Install it with: conda install -c conda-forge pyyaml "
            "or: pip install pyyaml"
        )

    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f) or {}

    config = NetworkParserConfig()

    # Sections are optional; use .get with defaults
    analysis_cfg: Dict[str, Any] = config_dict.get("analysis", {}) or {}
    processing_cfg: Dict[str, Any] = config_dict.get("processing", {}) or {}
    output_cfg: Dict[str, Any] = config_dict.get("output", {}) or {}
    validation_cfg: Dict[str, Any] = config_dict.get("validation", {}) or {}

    # Analysis
    setattr(
        config,
        "bootstrap_iterations",
        analysis_cfg.get(
            "bootstrap_iterations", getattr(config, "bootstrap_iterations", 100)
        ),
    )
    setattr(
        config,
        "confidence_threshold",
        analysis_cfg.get(
            "confidence_threshold", getattr(config, "confidence_threshold", 0.0)
        ),
    )
    setattr(
        config,
        "max_interaction_order",
        analysis_cfg.get(
            "max_interaction_order", getattr(config, "max_interaction_order", 2)
        ),
    )
    config.fdr_threshold = analysis_cfg.get("fdr_threshold", config.fdr_threshold)

    # Processing
    setattr(
        config,
        "max_workers",
        processing_cfg.get("max_workers", getattr(config, "max_workers", 1)),
    )
    config.memory_efficient = processing_cfg.get(
        "memory_efficient", config.memory_efficient
    )
    setattr(
        config,
        "chunk_size",
        processing_cfg.get("chunk_size", getattr(config, "chunk_size", 1000)),
    )

    # Output
    setattr(
        config,
        "output_formats",
        output_cfg.get("formats", getattr(config, "output_formats", ["json"])),
    )
    setattr(
        config,
        "include_matrices",
        output_cfg.get("include_matrices", getattr(config, "include_matrices", False)),
    )
    setattr(
        config,
        "generate_plots",
        output_cfg.get("generate_plots", getattr(config, "generate_plots", False)),
    )

    # Validation
    setattr(
        config,
        "cross_validation_folds",
        validation_cfg.get(
            "cross_validation_folds", getattr(config, "cross_validation_folds", 5)
        ),
    )
    setattr(
        config,
        "stability_threshold",
        validation_cfg.get(
            "stability_threshold", getattr(config, "stability_threshold", 0.0)
        ),
    )
    config.min_bootstrap_support = validation_cfg.get(
        "min_bootstrap_support", config.min_bootstrap_support
    )

    logger.info("Loaded config from %s", config_path)
    return config


def available_cpu_count() -> int:
    """Logical CPU count available to this process (always >= 1)."""
    return max(1, int(os.cpu_count() or 1))


def available_memory_gb() -> Optional[float]:
    """
    Best-effort available system RAM in GiB.

    Returns None when the platform cannot be queried. Used only to *cap*
    parallelism on small machines; missing data never blocks training.
    """
    # 1) psutil if installed
    try:
        import psutil  # type: ignore

        return float(psutil.virtual_memory().available) / (1024.0 ** 3)
    except Exception:
        pass

    # 2) Linux /proc
    try:
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            available_kb = None
            total_kb = None
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemAvailable:"):
                    available_kb = float(line.split()[1])
                elif line.startswith("MemTotal:"):
                    total_kb = float(line.split()[1])
            if available_kb is not None:
                return available_kb / (1024.0 ** 2)
            if total_kb is not None:
                # Prefer ~75% of total if MemAvailable is missing
                return 0.75 * total_kb / (1024.0 ** 2)
    except Exception:
        pass

    # 3) macOS sysctl
    try:
        import subprocess

        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        total = float(out) / (1024.0 ** 3)
        # Assume roughly half free when we cannot query pressure; conservative.
        return max(1.0, 0.45 * total)
    except Exception:
        pass
    return None


def resolve_effective_n_jobs(
    config: Any,
    *,
    override: Optional[int] = None,
    minimum_tasks: int = 1,
) -> int:
    """
    Resolve worker count for a parallel stage.

    Honours (in order): explicit override → context-local inner budget
    (set when outer hierarchy workers run) → config.n_jobs → CPU count.
    Always clamped to [1, available CPUs].
    """
    cpu_count = available_cpu_count()

    # Nested-parallelism budget from an outer Parallel stage.
    ctx_inner = _parallel_inner_n_jobs.get()
    if override is None and ctx_inner is not None:
        return max(1, min(int(ctx_inner), cpu_count))

    if override is not None:
        requested = int(override)
    else:
        requested = int(getattr(config, "n_jobs", -1) or -1)

    hard_cap = getattr(config, "parallel_max_workers", None)
    if hard_cap is not None:
        try:
            hard_cap_i = int(hard_cap)
            if hard_cap_i >= 1:
                cpu_count = min(cpu_count, hard_cap_i)
        except Exception:
            pass

    if requested == 0:
        return 1
    if requested < 0:
        # -1: use all CPUs when there is real parallel work; else 1.
        return cpu_count if int(minimum_tasks) > 1 else 1
    return max(1, min(int(requested), cpu_count))


def resolve_parallel_worker_budget(
    config: Any,
    *,
    n_tasks: int,
    override: Optional[int] = None,
    memory_per_worker_gb: Optional[float] = None,
    prefer_outer: bool = True,
) -> Dict[str, Any]:
    """
    Split available resources into outer (independent models) and inner
    (sklearn/CV within one model) worker counts.

    Scales up on large nodes and stays conservative on small machines:

    * 16 GB / few cores → typically outer=1–2 (safe, sequential-ish)
    * 128 GB / 24 cores → outer grows with free RAM and task count

    Parameters
    ----------
    n_tasks
        Number of independent units (child nodes, groups, fallbacks).
    memory_per_worker_gb
        Soft estimate of RAM needed per concurrent outer model fit.
        Defaults to config.parallel_memory_per_worker_gb (4 GiB).
    prefer_outer
        When True, spend cores on concurrent models first; each model then
        gets fewer inner threads. When False, prefer one model with full
        inner parallelism (similar to serial outer training).
    """
    n_tasks = max(1, int(n_tasks))
    total = resolve_effective_n_jobs(
        config, override=override, minimum_tasks=n_tasks
    )
    mem_gb = available_memory_gb()
    per_worker = memory_per_worker_gb
    if per_worker is None:
        per_worker = float(getattr(config, "parallel_memory_per_worker_gb", 4.0) or 4.0)
    per_worker = max(0.5, float(per_worker))

    # memory_efficient forces serial outer training
    if bool(getattr(config, "memory_efficient", False)):
        budget = {
            "total_workers": int(total),
            "outer_jobs": 1,
            "inner_jobs": int(total),
            "n_tasks": int(n_tasks),
            "available_memory_gb": mem_gb,
            "memory_cap_workers": None,
            "reason": "memory_efficient=True",
        }
        return budget

    mem_cap = None
    if mem_gb is not None and per_worker > 0:
        # Leave ~2 GiB headroom for OS / shared matrix
        usable = max(1.0, float(mem_gb) - 2.0)
        mem_cap = max(1, int(usable // per_worker))

    hard_cap = getattr(config, "parallel_max_workers", None)
    hard_cap_i = None
    if hard_cap is not None:
        try:
            hard_cap_i = max(1, int(hard_cap))
        except Exception:
            hard_cap_i = None

    outer = min(total, n_tasks)
    if mem_cap is not None:
        outer = min(outer, mem_cap)
    if hard_cap_i is not None:
        outer = min(outer, hard_cap_i)
    outer = max(1, int(outer))

    if not prefer_outer and n_tasks >= 1:
        # Optional: keep outer low so one fat model can use more cores
        outer = min(outer, max(1, total // 2)) if total >= 4 else 1

    # On very small RAM, force serial outer even if CPUs exist
    if mem_gb is not None and mem_gb < 10.0:
        outer = 1
    elif mem_gb is not None and mem_gb < 18.0:
        outer = min(outer, 2)

    inner = max(1, total // outer) if outer > 0 else total
    # Avoid outer*inner >> total oversubscription
    if outer * inner > total + outer:
        inner = max(1, total // outer)

    return {
        "total_workers": int(total),
        "outer_jobs": int(outer),
        "inner_jobs": int(inner),
        "n_tasks": int(n_tasks),
        "available_memory_gb": float(mem_gb) if mem_gb is not None else None,
        "memory_cap_workers": int(mem_cap) if mem_cap is not None else None,
        "memory_per_worker_gb": float(per_worker),
        "reason": "adaptive_cpu_and_memory",
    }


def should_run_parallel(
    config: Any,
    *,
    enabled_attr: str,
    n_tasks: int,
    min_tasks: int = 2,
) -> bool:
    """Return True when a parallel code path should be used."""
    if int(n_tasks) < int(min_tasks):
        return False
    if not bool(getattr(config, enabled_attr, True)):
        return False
    # Even if enabled, skip Parallel machinery when budget collapses to 1.
    budget = resolve_parallel_worker_budget(config, n_tasks=int(n_tasks))
    return int(budget["outer_jobs"]) >= 2


def run_with_inner_n_jobs(inner_jobs: int, fn, *args, **kwargs):
    """
    Execute ``fn`` with resolve_effective_n_jobs limited to ``inner_jobs``.

    Used by outer Parallel workers so concurrent model fits do not each take
    every CPU.
    """
    token = _parallel_inner_n_jobs.set(max(1, int(inner_jobs)))
    try:
        try:
            from threadpoolctl import threadpool_limits  # type: ignore

            with threadpool_limits(limits=max(1, int(inner_jobs))):
                return fn(*args, **kwargs)
        except Exception:
            return fn(*args, **kwargs)
    finally:
        _parallel_inner_n_jobs.reset(token)
