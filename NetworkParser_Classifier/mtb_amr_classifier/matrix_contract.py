#!/usr/bin/env python3
# network_parser/matrix_contract.py
"""
Binary genotype matrix contract for NetworkParser.

Encoding (everywhere)
---------------------
  0.0  : callable baseline allele
  1.0  : callable non-baseline / trained alternate state
  NaN  : non-callable, missing, filtered, absent, or unresolved

Rules
-----
- Missing must never silently become 0 or 1 without an explicit, fitted imputer.
- Missingness limits are fit on training data only and applied to val/query.
- Algorithms that cannot accept NaN use a deterministic imputer fitted on train.
- Required query markers that remain unknown contribute to abstention, not baseline evidence.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MATRIX_CONTRACT_VERSION = "1.0"
BASELINE_VALUE = 0.0
NON_BASELINE_VALUE = 1.0
NON_CALLABLE = float("nan")


@dataclass
class MissingnessPolicy:
    """Limits and imputation policy for the binary genotype matrix."""

    max_missing_fraction_per_sample: float = 0.5
    max_missing_fraction_per_feature: float = 0.5
    # When True, drop samples/features exceeding limits after train-only fit.
    drop_exceeding_samples: bool = True
    drop_exceeding_features: bool = True
    # Imputation for algorithms that cannot accept NaN.
    # none | baseline (fill 0) | feature_mode | constant
    impute_strategy: str = "baseline"
    impute_constant: float = 0.0
    add_missing_indicator: bool = False
    # If True, representing NaN as a categorical level is allowed (selector only).
    allow_missing_as_category: bool = False

    def validate(self) -> None:
        for name in (
            "max_missing_fraction_per_sample",
            "max_missing_fraction_per_feature",
        ):
            val = float(getattr(self, name))
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        strat = str(self.impute_strategy).strip().lower()
        if strat not in {"none", "baseline", "feature_mode", "constant"}:
            raise ValueError(
                "impute_strategy must be one of: none, baseline, feature_mode, constant"
            )
        self.impute_strategy = strat

    @classmethod
    def from_config(cls, config: Any) -> "MissingnessPolicy":
        if config is None:
            return cls()
        # Prefer explicit per-axis knobs; fall back to legacy max_missing_fraction.
        legacy = float(getattr(config, "max_missing_fraction", 0.5))
        samp = getattr(config, "max_missing_fraction_per_sample", None)
        feat = getattr(config, "max_missing_fraction_per_feature", None)
        return cls(
            max_missing_fraction_per_sample=float(samp if samp is not None else legacy),
            max_missing_fraction_per_feature=float(
                feat if feat is not None else legacy
            ),
            drop_exceeding_samples=bool(
                getattr(config, "drop_high_missing_samples", True)
            ),
            drop_exceeding_features=bool(
                getattr(config, "drop_high_missing_features", True)
            ),
            impute_strategy=str(
                getattr(config, "genotype_impute_strategy", "baseline")
            ),
            impute_constant=float(getattr(config, "genotype_impute_constant", 0.0)),
            add_missing_indicator=bool(
                getattr(config, "add_missing_indicator_features", False)
            ),
            allow_missing_as_category=bool(
                getattr(config, "allow_missing_as_category", False)
            ),
        )


@dataclass
class FittedMissingnessState:
    """Train-fitted missingness mask + imputer parameters (bundleable)."""

    policy: MissingnessPolicy
    retained_features: List[str] = field(default_factory=list)
    dropped_features: List[str] = field(default_factory=list)
    feature_missing_fraction_train: Dict[str, float] = field(default_factory=dict)
    # Per-feature fill values for feature_mode / constant / baseline
    feature_fill_values: Dict[str, float] = field(default_factory=dict)
    contract_version: str = MATRIX_CONTRACT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "policy": asdict(self.policy),
            "retained_features": list(self.retained_features),
            "dropped_features": list(self.dropped_features),
            "feature_missing_fraction_train": dict(self.feature_missing_fraction_train),
            "feature_fill_values": dict(self.feature_fill_values),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FittedMissingnessState":
        pol = payload.get("policy") or {}
        policy = MissingnessPolicy(
            **{k: pol[k] for k in MissingnessPolicy.__dataclass_fields__ if k in pol}
        )
        policy.validate()
        return cls(
            policy=policy,
            retained_features=list(payload.get("retained_features") or []),
            dropped_features=list(payload.get("dropped_features") or []),
            feature_missing_fraction_train=dict(
                payload.get("feature_missing_fraction_train") or {}
            ),
            feature_fill_values={
                str(k): float(v)
                for k, v in (payload.get("feature_fill_values") or {}).items()
            },
            contract_version=str(
                payload.get("contract_version", MATRIX_CONTRACT_VERSION)
            ),
        )

    def save_json(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "FittedMissingnessState":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


def coerce_binary_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Coerce to float binary matrix; invalid tokens → NaN (never silent 0)."""
    out = X.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.astype(float)


def missing_fraction_per_sample(X: pd.DataFrame) -> pd.Series:
    if X.shape[1] == 0:
        return pd.Series(0.0, index=X.index)
    return X.isna().mean(axis=1)


def missing_fraction_per_feature(X: pd.DataFrame) -> pd.Series:
    if X.shape[0] == 0:
        return pd.Series(0.0, index=X.columns)
    return X.isna().mean(axis=0)


def fit_missingness_state(
    X_train: pd.DataFrame,
    policy: Optional[MissingnessPolicy] = None,
) -> Tuple[pd.DataFrame, FittedMissingnessState, Dict[str, Any]]:
    """
    Fit missingness limits and imputer parameters on training data only.

    Returns (X_train_filtered_not_imputed, fitted_state, audit).
    Imputation is applied separately via ``transform_with_missingness_state``.
    """
    policy = policy or MissingnessPolicy()
    policy.validate()
    X = coerce_binary_matrix(X_train)

    feat_miss = missing_fraction_per_feature(X)
    drop_feats = [
        str(f)
        for f, frac in feat_miss.items()
        if policy.drop_exceeding_features
        and float(frac) > float(policy.max_missing_fraction_per_feature)
    ]
    keep_feats = [str(c) for c in X.columns if str(c) not in set(drop_feats)]
    if not keep_feats:
        raise ValueError(
            "All features exceeded max_missing_fraction_per_feature on training data."
        )
    Xf = X.loc[:, keep_feats].copy()

    drop_samples: List[str] = []
    if policy.drop_exceeding_samples:
        sample_miss_f = missing_fraction_per_sample(Xf)
        drop_samples = [
            str(s)
            for s, frac in sample_miss_f.items()
            if float(frac) > float(policy.max_missing_fraction_per_sample)
        ]
        if drop_samples:
            Xf = Xf.drop(index=drop_samples, errors="ignore")
        if Xf.shape[0] == 0:
            raise ValueError(
                "All samples exceeded max_missing_fraction_per_sample on training data."
            )

    fill_values: Dict[str, float] = {}
    for col in Xf.columns:
        col_s = str(col)
        if policy.impute_strategy == "none":
            continue
        if policy.impute_strategy == "baseline":
            fill_values[col_s] = BASELINE_VALUE
        elif policy.impute_strategy == "constant":
            fill_values[col_s] = float(policy.impute_constant)
        elif policy.impute_strategy == "feature_mode":
            observed = Xf[col].dropna()
            if observed.empty:
                fill_values[col_s] = BASELINE_VALUE
            else:
                # Mode of callable 0/1 values only
                vals, counts = np.unique(observed.to_numpy(), return_counts=True)
                fill_values[col_s] = float(vals[int(np.argmax(counts))])
        else:
            fill_values[col_s] = BASELINE_VALUE

    state = FittedMissingnessState(
        policy=policy,
        retained_features=keep_feats,
        dropped_features=drop_feats,
        feature_missing_fraction_train={str(k): float(v) for k, v in feat_miss.items()},
        feature_fill_values=fill_values,
    )
    audit = {
        "n_train_samples_in": int(X.shape[0]),
        "n_train_features_in": int(X.shape[1]),
        "n_train_samples_out": int(Xf.shape[0]),
        "n_train_features_out": int(Xf.shape[1]),
        "n_dropped_features": int(len(drop_feats)),
        "n_dropped_samples": int(len(drop_samples)),
        "dropped_samples": drop_samples[:50],
        "impute_strategy": policy.impute_strategy,
        "contract_version": MATRIX_CONTRACT_VERSION,
    }
    logger.info(
        "Missingness fit | samples %d→%d | features %d→%d | strategy=%s",
        audit["n_train_samples_in"],
        audit["n_train_samples_out"],
        audit["n_train_features_in"],
        audit["n_train_features_out"],
        policy.impute_strategy,
    )
    return Xf, state, audit


def transform_with_missingness_state(
    X: pd.DataFrame,
    state: FittedMissingnessState,
    *,
    apply_imputation: bool = True,
    drop_high_missing_samples: Optional[bool] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Align columns to train-retained features and optionally impute with train fill values.

    Does not invent baseline evidence without imputation enabled.
    """
    X = coerce_binary_matrix(X)
    feats = (
        list(state.retained_features) if state.retained_features else list(X.columns)
    )
    for f in feats:
        if f not in X.columns:
            X[f] = NON_CALLABLE
    X = X.loc[:, feats].copy()

    sample_miss = missing_fraction_per_sample(X)
    drop_samples_flag = (
        state.policy.drop_exceeding_samples
        if drop_high_missing_samples is None
        else bool(drop_high_missing_samples)
    )
    dropped: List[str] = []
    if drop_samples_flag:
        dropped = [
            str(s)
            for s, frac in sample_miss.items()
            if float(frac) > float(state.policy.max_missing_fraction_per_sample)
        ]
        if dropped:
            X = X.drop(index=dropped, errors="ignore")

    n_nan_before = int(X.isna().sum().sum())
    if apply_imputation and state.policy.impute_strategy != "none":
        for col in X.columns:
            fill = state.feature_fill_values.get(str(col), BASELINE_VALUE)
            X[col] = X[col].fillna(fill)

    if state.policy.add_missing_indicator:
        # Indicators for original missingness (before impute) need pre-impute mask;
        # recompute from remaining NaN if impute was skipped.
        pass  # optional extension; reserved

    audit = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_nan_before_impute": n_nan_before,
        "n_nan_after": int(X.isna().sum().sum()),
        "dropped_samples": dropped[:50],
        "imputed": bool(apply_imputation and state.policy.impute_strategy != "none"),
        "per_sample_missing_fraction": {
            str(k): float(v) for k, v in sample_miss.items()
        },
    }
    return X, audit


def prepare_for_sklearn(
    X: pd.DataFrame,
    state: Optional[FittedMissingnessState] = None,
    *,
    policy: Optional[MissingnessPolicy] = None,
) -> Tuple[pd.DataFrame, FittedMissingnessState, Dict[str, Any]]:
    """
    Convenience: if state is None, fit on X (training); else transform only.

    Always returns an imputation-ready matrix when strategy != none.
    """
    if state is None:
        Xf, state, audit = fit_missingness_state(X, policy=policy)
        X_out, t_audit = transform_with_missingness_state(
            Xf, state, apply_imputation=True, drop_high_missing_samples=False
        )
        audit["transform"] = t_audit
        return X_out, state, audit
    X_out, t_audit = transform_with_missingness_state(X, state, apply_imputation=True)
    return X_out, state, t_audit


def pairwise_complete_distance_mask(
    X: pd.DataFrame,
    *,
    min_pairwise_complete_fraction: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a boolean sample mask for visualisation: keep samples with enough
    pairwise-complete feature coverage relative to the median sample.
    """
    Xn = coerce_binary_matrix(X)
    miss = missing_fraction_per_sample(Xn)
    keep = miss.to_numpy() <= (1.0 - float(min_pairwise_complete_fraction))
    return keep, miss.to_numpy()
