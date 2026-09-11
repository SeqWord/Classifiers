#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reusable supervised-model selector for hierarchical categorical marker matrices.

This module contains no command-line user interface and does not print reports.
It is intended to be placed in ../lib and imported by both:

    training/model_selector.py
    training/run.py

The public API is centered on:

    read_hierarchical_matrix(...)
    recommend_model(...)
    recommend_hierarchy(...)

`recommend_model()` is the function intended for node-specific AUTO training.
At an internal hierarchy node, pass only the samples belonging to that node and
use the node's immediate child labels as y.
"""

from __future__ import annotations

import math
import os
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# Optional XGBoost support.  Keep import failure non-fatal so all sklearn
# candidates remain usable when xgboost is not installed.
try:
    import xgboost as _xgboost
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
    XGBOOST_VERSION = getattr(_xgboost, "__version__", "unknown")
    XGBOOST_IMPORT_ERROR = ""
except Exception as _xgb_exc:  # pragma: no cover - depends on local environment
    XGBClassifier = None  # type: ignore[assignment]
    XGBOOST_AVAILABLE = False
    XGBOOST_VERSION = ""
    XGBOOST_IMPORT_ERROR = f"{type(_xgb_exc).__name__}: {_xgb_exc}"


SUPPORTED_MODELS: Tuple[str, ...] = (
    "LR", "SVC", "RF", "DT", "MLP", "KNN", "NBayes", "XGBoost"
)

MISSING_TOKEN = "__MISSING__"


@dataclass
class MatrixData:
    """Parsed hierarchical matrix and matrix-level metadata."""

    X: pd.DataFrame
    genome_names: List[str]
    labels: List[List[str]]
    feature_titles: List[str]
    data_type: str
    removed_columns: List[Tuple[str, float]] = field(default_factory=list)

    @property
    def n_levels(self) -> int:
        return max((len(row) for row in self.labels), default=0)


@dataclass
class SelectionContext:
    """Properties of one supervised classification problem."""

    n_samples: int
    n_features: int
    n_classes: int
    class_counts: Dict[str, int]
    imbalance_ratio: float
    data_type: str
    missing_fraction: float


@dataclass
class ModelResult:
    """Recommendation for one candidate model."""

    algorithm: str
    score: float
    parameters: Dict[str, Any]
    rationale: List[str]
    levels_evaluated: int = 1
    levels_total: int = 1


@dataclass
class SelectionResult:
    """Ranked recommendations for one classification problem."""

    context: SelectionContext
    best: Optional[ModelResult]
    alternatives: List[ModelResult]
    scores: Dict[str, float]
    status: str = "ok"


@dataclass
class HierarchySelectionResult:
    """Concise general recommendation aggregated across hierarchy levels."""

    n_levels: int
    evaluable_levels: int
    best: Optional[ModelResult]
    alternatives: List[ModelResult]
    mean_scores: Dict[str, float]
    level_coverage: Dict[str, int]
    level_class_counts: List[int]
    status: str = "ok"


# -----------------------------------------------------------------------------
# Matrix parsing
# -----------------------------------------------------------------------------

def _infer_file_separator(file_path: str) -> str:
    low = file_path.lower()
    if low.endswith(".csv"):
        return ","
    if low.endswith((".tsv", ".txt")):
        return "\t"
    raise ValueError(f"Unsupported matrix format: {file_path}")


def _is_missing_series(series: pd.Series, empty_symbol: str) -> pd.Series:
    text = series.astype(str).str.strip()
    missing = series.isna() | (text == "")
    if str(empty_symbol).strip():
        missing |= text == str(empty_symbol).strip()
    return missing


def _normalize_value(value: Any, empty_symbol: str) -> str:
    if pd.isna(value):
        return MISSING_TOKEN
    text = str(value).strip()
    if text == "":
        return MISSING_TOKEN
    if str(empty_symbol).strip() and text == str(empty_symbol).strip():
        return MISSING_TOKEN
    return text


def remove_empty_columns(
    feature_df: pd.DataFrame,
    threshold: float = 1.0,
    empty_symbol: str = "",
) -> Tuple[pd.DataFrame, List[Tuple[str, float]]]:
    """
    Remove columns whose missing-value fraction is greater than `threshold`.

    The threshold is a fraction in the range 0..1. A threshold of 1 keeps all
    columns, including columns that are completely missing.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("empty-column threshold must be in the range 0.0..1.0")

    keep: List[str] = []
    removed: List[Tuple[str, float]] = []

    for column in feature_df.columns:
        fraction = float(_is_missing_series(feature_df[column], empty_symbol).mean())
        if fraction <= threshold:
            keep.append(column)
        else:
            removed.append((str(column), fraction))

    return feature_df.loc[:, keep].copy(), removed


def infer_data_type(X: pd.DataFrame) -> str:
    """Infer whether feature states are binary 0/1 or general characters."""
    observed: set[str] = set()

    for column in X.columns:
        for value in X[column].astype(str):
            token = value.strip()
            if token == MISSING_TOKEN:
                continue
            observed.add(token.lower())
            if len(observed) > 32:
                break
        if len(observed) > 32:
            break

    if observed and observed.issubset({"0", "1"}):
        return "binary"
    return "character"


def read_hierarchical_matrix(
    file_path: str,
    delimiter: str = "|",
    empty_threshold: float = 1.0,
    empty_symbol: str = "",
    flg_control_duplicates: bool = False,
) -> MatrixData:
    """
    Read a CSV/TSV matrix with one compound heading column.

    The first column must have the format::

        top_label|second_label|...|genome_name

    where `delimiter` is configurable. Everything after the first column is
    treated as a feature matrix. Feature states may be 0/1 or characters.
    """
    if not delimiter:
        raise ValueError("Hierarchy delimiter must not be empty.")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Input matrix does not exist: {file_path}")

    sep = _infer_file_separator(file_path)
    df = pd.read_csv(
        file_path,
        sep=sep,
        header=0,
        dtype=str,
        encoding="utf-8-sig",
        keep_default_na=False,
    )

    if df.shape[0] < 2:
        raise ValueError("Input matrix must contain at least two data rows.")
    if df.shape[1] < 2:
        raise ValueError(
            "Input matrix must contain one hierarchical heading column and "
            "at least one feature column."
        )

    heading_column = df.columns[0]
    feature_df = df.iloc[:, 1:].copy()

    feature_df, removed_columns = remove_empty_columns(
        feature_df,
        threshold=empty_threshold,
        empty_symbol=empty_symbol,
    )
    if feature_df.shape[1] == 0:
        raise ValueError("All feature columns were removed by missing-data filtering.")

    headings = df[heading_column].astype(str).tolist()
    split_headings: List[List[str]] = []

    for row_number, heading in enumerate(headings, start=2):
        parts = [part.strip() for part in heading.split(delimiter)]
        if len(parts) < 2:
            raise ValueError(
                f"Row {row_number}: first-column heading must contain at least "
                f"one class label and a genome name separated by {delimiter!r}."
            )
        if not parts[-1]:
            raise ValueError(f"Row {row_number}: genome name is empty.")
        split_headings.append(parts)

    genome_names = [parts[-1] for parts in split_headings]
    if flg_control_duplicates:
        duplicates = [name for name, n in Counter(genome_names).items() if n > 1]
        if duplicates:
            raise ValueError(
                "Genome names must be unique. Duplicate name(s): "
                + ", ".join(sorted(duplicates))
            )

    raw_labels = [parts[:-1] for parts in split_headings]
    n_levels = max((len(row) for row in raw_labels), default=0)
    labels = [row + [""] * (n_levels - len(row)) for row in raw_labels]

    X = feature_df.copy()
    for column in X.columns:
        X[column] = X[column].map(lambda value: _normalize_value(value, empty_symbol))

    return MatrixData(
        X=X,
        genome_names=genome_names,
        labels=labels,
        feature_titles=list(X.columns),
        data_type=infer_data_type(X),
        removed_columns=removed_columns,
    )


def hierarchy_level_target(
    labels: Sequence[Sequence[str]],
    level_index: int,
    delimiter: str = "|",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return row indices and cumulative class paths for one hierarchy level.

    At level 2, for example, labels become ``A|A1`` rather than just ``A1``.
    This prevents equal child names under different parents from being merged.
    """
    indices: List[int] = []
    targets: List[str] = []

    for i, row in enumerate(labels):
        if level_index >= len(row):
            continue
        path = [str(v).strip() for v in row[: level_index + 1]]
        if not path or any(v == "" for v in path):
            continue
        indices.append(i)
        targets.append(delimiter.join(path))

    return np.asarray(indices, dtype=int), np.asarray(targets, dtype=object)


# -----------------------------------------------------------------------------
# Model probes and parameter heuristics
# -----------------------------------------------------------------------------

class _NonNegativeShift:
    """sklearn-compatible transformer shifting ordinal codes by +1."""

    def fit(self, X: Any, y: Any = None) -> "_NonNegativeShift":
        return self

    def transform(self, X: Any) -> np.ndarray:
        return np.asarray(X, dtype=float) + 1.0

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {}

    def set_params(self, **params: Any) -> "_NonNegativeShift":
        return self


def _onehot_preprocessor(n_features: int) -> ColumnTransformer:
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), list(range(n_features)))],
        remainder="drop",
    )


def _suggest_knn_k(n: int) -> int:
    if n <= 3:
        return 1
    k = max(3, min(25, int(round(math.sqrt(n)))))
    if k % 2 == 0:
        k += 1
    return min(k, max(1, n - 1))


def _suggest_tree_depth(n: int, n_classes: int) -> Optional[int]:
    if n < 60:
        return max(3, min(8, int(round(math.log2(max(2, n))))))
    if n_classes >= 10:
        return 12
    return None


def _suggest_mlp_hidden_layers(n: int, p: int, n_classes: int) -> Tuple[int, ...]:
    if n < 100:
        return (32,)
    if p >= 1000 or n_classes >= 10:
        return (128, 64)
    if p >= 200:
        return (64, 32)
    return (64,)


def _context(X: pd.DataFrame, y: Sequence[Any], data_type: str) -> SelectionContext:
    labels = [str(v) for v in y]
    counts = Counter(labels)
    n_classes = len(counts)
    imbalance = (
        max(counts.values()) / max(1, min(counts.values())) if counts else float("nan")
    )
    missing_fraction = float(
        (X.astype(str) == MISSING_TOKEN).to_numpy().mean()
    ) if X.size else 0.0

    return SelectionContext(
        n_samples=X.shape[0],
        n_features=X.shape[1],
        n_classes=n_classes,
        class_counts=dict(counts),
        imbalance_ratio=imbalance,
        data_type=data_type,
        missing_fraction=missing_fraction,
    )


def model_parameters(algorithm: str, ctx: SelectionContext) -> Dict[str, Any]:
    """Return parameters matching constructors in NeuralNetwork.py."""
    algorithm = str(algorithm)
    class_weight = "balanced" if ctx.imbalance_ratio >= 2.0 else None

    if algorithm == "LR":
        return {
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 1500,
            "class_weight": class_weight,
        }

    if algorithm == "SVC":
        return {
            "C": 1.0,
            "kernel": "rbf",
            "degree": 3,
            "gamma": "scale",
            "coef0": 0.0,
            "class_weight": class_weight,
        }

    if algorithm == "RF":
        return {
            "n_estimators": 500 if ctx.n_samples >= 500 else 300,
            "max_depth": None,
            "max_features": "sqrt",
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "bootstrap": True,
            "class_weight": class_weight,
            "oob_score": False,
        }

    if algorithm == "DT":
        return {
            "criterion": "gini",
            "splitter": "best",
            "max_depth": _suggest_tree_depth(ctx.n_samples, ctx.n_classes),
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
            "class_weight": class_weight,
        }

    if algorithm == "MLP":
        return {
            "hidden_layer_sizes": _suggest_mlp_hidden_layers(
                ctx.n_samples, ctx.n_features, ctx.n_classes
            ),
            "activation": "relu",
            "solver": "adam",
            "alpha": 1e-4 if ctx.n_samples >= 200 else 1e-3,
            "batch_size": "auto",
            "learning_rate": "constant",
            "learning_rate_init": 1e-3,
            "max_iter": 1000,
            "early_stopping": ctx.n_samples >= 100,
        }

    if algorithm == "KNN":
        return {
            "n_neighbors": _suggest_knn_k(ctx.n_samples),
            "weights": "distance",
            "metric": "minkowski",
            "p": 1 if ctx.data_type == "character" else 2,
        }

    if algorithm == "NBayes":
        return {
            "alpha": 1.0,
            "fit_prior": True,
        }

    if algorithm == "XGBoost":
        # Conservative defaults for repeated CV probes.  Deeper trees and very
        # large ensembles are deliberately avoided for small hierarchy nodes.
        if ctx.n_samples < 100:
            n_estimators = 200
            max_depth = 3
        elif ctx.n_samples < 1000:
            n_estimators = 400
            max_depth = 4
        else:
            n_estimators = 600
            max_depth = 6

        return {
            "n_estimators": n_estimators,
            "learning_rate": 0.05,
            "max_depth": max_depth,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "min_child_weight": 1.0,
            "gamma": 0.0,
            # Do not infer scale_pos_weight from an unordered class mapping.
            # run.py may set it explicitly for a known binary positive class.
            "scale_pos_weight": None,
        }

    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _make_xgboost_probe(ctx: SelectionContext) -> Optional[Any]:
    """Construct an XGBClassifier probe when XGBoost is available.

    Import/construction failures are intentionally non-fatal because XGBoost is
    an optional dependency.  Call ``get_model_availability()`` to distinguish a
    missing XGBoost installation from a model that simply could not be scored
    on a particular hierarchy level.
    """
    if not XGBOOST_AVAILABLE or XGBClassifier is None:
        return None

    params = model_parameters("XGBoost", ctx)
    objective = "binary:logistic" if ctx.n_classes == 2 else "multi:softprob"

    xgb_params: Dict[str, Any] = {
        "n_estimators": params["n_estimators"],
        "learning_rate": params["learning_rate"],
        "max_depth": params["max_depth"],
        "subsample": params["subsample"],
        "colsample_bytree": params["colsample_bytree"],
        "reg_lambda": params["reg_lambda"],
        "reg_alpha": params["reg_alpha"],
        "min_child_weight": params["min_child_weight"],
        "gamma": params["gamma"],
        "objective": objective,
        "eval_metric": "logloss" if ctx.n_classes == 2 else "mlogloss",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }
    if ctx.n_classes > 2:
        xgb_params["num_class"] = ctx.n_classes

    try:
        return XGBClassifier(**xgb_params)
    except Exception:
        return None


def _probe_models(ctx: SelectionContext) -> Dict[str, Any]:
    p = ctx.n_features
    params = {name: model_parameters(name, ctx) for name in SUPPORTED_MODELS}

    def onehot(clf: Any) -> Pipeline:
        return Pipeline([
            ("pre", _onehot_preprocessor(p)),
            ("clf", clf),
        ])

    models: Dict[str, Any] = {
        "LR": onehot(LogisticRegression(
            C=params["LR"]["C"],
            penalty=params["LR"]["penalty"],
            solver=params["LR"]["solver"],
            max_iter=params["LR"]["max_iter"],
            class_weight=params["LR"]["class_weight"],
        )),
        "SVC": onehot(SVC(
            C=params["SVC"]["C"],
            kernel=params["SVC"]["kernel"],
            degree=params["SVC"]["degree"],
            gamma=params["SVC"]["gamma"],
            coef0=params["SVC"]["coef0"],
            class_weight=params["SVC"]["class_weight"],
        )),
        "RF": onehot(RandomForestClassifier(
            n_estimators=params["RF"]["n_estimators"],
            max_depth=params["RF"]["max_depth"],
            max_features=params["RF"]["max_features"],
            min_samples_split=params["RF"]["min_samples_split"],
            min_samples_leaf=params["RF"]["min_samples_leaf"],
            bootstrap=params["RF"]["bootstrap"],
            class_weight=params["RF"]["class_weight"],
            oob_score=params["RF"]["oob_score"],
            random_state=42,
            n_jobs=-1,
        )),
        "DT": onehot(DecisionTreeClassifier(
            criterion=params["DT"]["criterion"],
            splitter=params["DT"]["splitter"],
            max_depth=params["DT"]["max_depth"],
            min_samples_split=params["DT"]["min_samples_split"],
            min_samples_leaf=params["DT"]["min_samples_leaf"],
            max_features=params["DT"]["max_features"],
            class_weight=params["DT"]["class_weight"],
            random_state=42,
        )),
        "MLP": Pipeline([
            ("pre", _onehot_preprocessor(p)),
            ("scale", StandardScaler(with_mean=False)),
            ("clf", MLPClassifier(
                hidden_layer_sizes=params["MLP"]["hidden_layer_sizes"],
                activation=params["MLP"]["activation"],
                solver=params["MLP"]["solver"],
                alpha=params["MLP"]["alpha"],
                batch_size=params["MLP"]["batch_size"],
                learning_rate=params["MLP"]["learning_rate"],
                learning_rate_init=params["MLP"]["learning_rate_init"],
                max_iter=params["MLP"]["max_iter"],
                early_stopping=params["MLP"]["early_stopping"],
                random_state=42,
            )),
        ]),
        "KNN": onehot(KNeighborsClassifier(
            n_neighbors=params["KNN"]["n_neighbors"],
            weights=params["KNN"]["weights"],
            metric=params["KNN"]["metric"],
            p=params["KNN"]["p"],
            n_jobs=-1,
        )),
        "NBayes": Pipeline([
            ("pre", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-1,
            )),
            ("shift", _NonNegativeShift()),
            ("clf", CategoricalNB(
                alpha=params["NBayes"]["alpha"],
                fit_prior=params["NBayes"]["fit_prior"],
            )),
        ]),
    }

    xgb = _make_xgboost_probe(ctx)
    if xgb is not None:
        models["XGBoost"] = onehot(xgb)

    return models


def _cv_score(estimator: Any, X: np.ndarray, y: np.ndarray, cv_folds: int) -> float:
    counts = Counter(y.tolist())
    if len(counts) < 2:
        return float("nan")

    min_count = min(counts.values())
    n_splits = min(int(cv_folds), int(min_count))
    if n_splits < 2:
        return float("nan")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            scores = cross_val_score(
                estimator,
                X,
                y,
                cv=cv,
                scoring="balanced_accuracy",
                n_jobs=1,
                error_score=np.nan,
            )
        if np.all(np.isnan(scores)):
            return float("nan")
        return float(np.nanmean(scores))
    except Exception:
        return float("nan")


def _rationale(algorithm: str, ctx: SelectionContext, score: float) -> List[str]:
    out: List[str] = []
    if math.isfinite(score):
        out.append(f"Stratified CV balanced accuracy: {score:.4f}.")
    else:
        out.append("Reliable stratified cross-validation was not available for this node.")

    explanations = {
        "LR": "Linear logistic regression is simple and stable when the classes are separable after categorical encoding.",
        "SVC": "RBF SVC can capture nonlinear boundaries in the encoded marker space.",
        "RF": "Random forests handle nonlinear marker interactions and irrelevant features robustly.",
        "DT": "A decision tree provides interpretable rule-based classification but can overfit small data sets.",
        "MLP": "MLP can model complex feature interactions when the node has enough training samples.",
        "KNN": "KNN is useful when local neighborhoods in marker-state space correspond to the classes.",
        "NBayes": "Categorical Naive Bayes is efficient for discrete marker states and is often useful for small nodes.",
        "XGBoost": "Boosted trees can model complex nonlinear interactions in structured marker matrices.",
    }
    out.append(explanations[algorithm])

    if ctx.imbalance_ratio >= 2.0:
        out.append(
            f"Class imbalance is substantial (largest/smallest ratio {ctx.imbalance_ratio:.2f}); "
            "balanced accuracy is used for comparison."
        )
    return out


def _fallback_algorithm(ctx: SelectionContext) -> str:
    """Conservative fallback when stratified CV cannot be calculated."""
    if ctx.n_samples < 50:
        return "NBayes"
    return "RF"


# -----------------------------------------------------------------------------
# Public selection API
# -----------------------------------------------------------------------------

def recommend_model(
    X: pd.DataFrame | np.ndarray,
    y: Sequence[Any],
    data_type: str = "character",
    cv_folds: int = 5,
    top_alternatives: int = 3,
    candidate_models: Optional[Iterable[str]] = None,
    allow_fallback: bool = True,
) -> SelectionResult:
    """
    Recommend a model for one supervised classification problem.

    This is the primary API for ``run.py PROJECT AUTO``. For a hierarchy node,
    pass only samples belonging to that node and set ``y`` to its immediate
    child labels. The returned ``parameters`` match constructors in
    ``NeuralNetwork.py``.
    """
    if cv_folds < 2:
        raise ValueError("cv_folds must be >= 2")
    if top_alternatives < 0:
        raise ValueError("top_alternatives must be >= 0")

    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        arr = np.asarray(X, dtype=object)
        if arr.ndim != 2:
            raise ValueError("X must be a two-dimensional matrix")
        X_df = pd.DataFrame(arr, columns=[f"f{i}" for i in range(arr.shape[1])])

    y_arr = np.asarray([str(v) for v in y], dtype=object)
    if X_df.shape[0] != len(y_arr):
        raise ValueError("X and y contain different numbers of samples")

    ctx = _context(X_df, y_arr, data_type=data_type)
    if ctx.n_classes < 2:
        return SelectionResult(
            context=ctx,
            best=None,
            alternatives=[],
            scores={name: float("nan") for name in SUPPORTED_MODELS},
            status="fewer than two classes; no classifier is required",
        )

    allowed = list(candidate_models) if candidate_models is not None else list(SUPPORTED_MODELS)
    invalid = [name for name in allowed if name not in SUPPORTED_MODELS]
    if invalid:
        raise ValueError("Unsupported candidate model(s): " + ", ".join(invalid))

    # Encode labels to contiguous integers for compatibility with all probes,
    # including XGBoost.
    label_map = {label: i for i, label in enumerate(ctx.class_counts.keys())}
    y_encoded = np.asarray([label_map[str(v)] for v in y_arr], dtype=int)
    X_array = X_df.to_numpy(dtype=object)

    probes = _probe_models(ctx)
    scores: Dict[str, float] = {name: float("nan") for name in SUPPORTED_MODELS}
    for algorithm in allowed:
        estimator = probes.get(algorithm)
        if estimator is None:
            continue
        scores[algorithm] = _cv_score(estimator, X_array, y_encoded, cv_folds=cv_folds)

    ranked = sorted(
        [(a, scores[a]) for a in allowed if math.isfinite(scores[a])],
        key=lambda item: item[1],
        reverse=True,
    )

    if not ranked:
        if not allow_fallback:
            return SelectionResult(
                context=ctx,
                best=None,
                alternatives=[],
                scores=scores,
                status=(
                    "stratified CV unavailable; the smallest class must contain "
                    "at least two samples"
                ),
            )

        algorithm = _fallback_algorithm(ctx)
        best = ModelResult(
            algorithm=algorithm,
            score=float("nan"),
            parameters=model_parameters(algorithm, ctx),
            rationale=_rationale(algorithm, ctx, float("nan")) + [
                "This is a fallback recommendation; run.py should record that model selection was not CV-validated."
            ],
        )
        return SelectionResult(
            context=ctx,
            best=best,
            alternatives=[],
            scores=scores,
            status="fallback: stratified CV unavailable",
        )

    chosen = ranked[: 1 + top_alternatives]
    results = [
        ModelResult(
            algorithm=algorithm,
            score=score,
            parameters=model_parameters(algorithm, ctx),
            rationale=_rationale(algorithm, ctx, score),
        )
        for algorithm, score in chosen
    ]

    return SelectionResult(
        context=ctx,
        best=results[0],
        alternatives=results[1:],
        scores=scores,
        status="ok",
    )


def recommend_hierarchy(
    matrix: MatrixData,
    delimiter: str = "|",
    cv_folds: int = 5,
    top_alternatives: int = 3,
    candidate_models: Optional[Iterable[str]] = None,
) -> HierarchySelectionResult:
    """
    Produce one general model recommendation for a complete hierarchy.

    Each hierarchy level is evaluated using cumulative class paths, but the
    returned result is intentionally concise. Candidate scores are averaged
    across evaluable hierarchy levels. This function does *not* inspect every
    hierarchy node; node-specific selection belongs in ``run.py AUTO`` via
    ``recommend_model()``.
    """
    n_levels = matrix.n_levels
    if n_levels == 0:
        return HierarchySelectionResult(
            n_levels=0,
            evaluable_levels=0,
            best=None,
            alternatives=[],
            mean_scores={name: float("nan") for name in SUPPORTED_MODELS},
            level_coverage={name: 0 for name in SUPPORTED_MODELS},
            level_class_counts=[],
            status="no hierarchy labels were found",
        )

    allowed = list(candidate_models) if candidate_models is not None else list(SUPPORTED_MODELS)
    score_lists: Dict[str, List[float]] = {name: [] for name in allowed}
    contexts: List[SelectionContext] = []
    level_class_counts: List[int] = []
    evaluable_levels = 0

    for level_index in range(n_levels):
        indices, y = hierarchy_level_target(matrix.labels, level_index, delimiter)
        if len(indices) == 0:
            continue

        X_level = matrix.X.iloc[indices, :].reset_index(drop=True)
        ctx = _context(X_level, y, matrix.data_type)
        level_class_counts.append(ctx.n_classes)

        if ctx.n_classes < 2 or min(ctx.class_counts.values()) < 2:
            continue

        result = recommend_model(
            X=X_level,
            y=y,
            data_type=matrix.data_type,
            cv_folds=cv_folds,
            top_alternatives=0,
            candidate_models=allowed,
            allow_fallback=False,
        )
        contexts.append(ctx)
        evaluable_levels += 1

        for algorithm in allowed:
            score = result.scores.get(algorithm, float("nan"))
            if math.isfinite(score):
                score_lists[algorithm].append(score)

    mean_scores: Dict[str, float] = {
        name: (float(np.mean(values)) if values else float("nan"))
        for name, values in score_lists.items()
    }
    coverage: Dict[str, int] = {name: len(values) for name, values in score_lists.items()}

    ranked = sorted(
        [
            (name, mean_scores[name], coverage[name])
            for name in allowed
            if math.isfinite(mean_scores[name])
        ],
        key=lambda item: (item[1], item[2]),
        reverse=True,
    )

    if not ranked:
        # Build a matrix-wide fallback context for a concise recommendation.
        # It is intentionally marked as non-CV-validated.
        top_indices, top_y = hierarchy_level_target(matrix.labels, 0, delimiter)
        if len(top_indices) == 0:
            return HierarchySelectionResult(
                n_levels=n_levels,
                evaluable_levels=0,
                best=None,
                alternatives=[],
                mean_scores={name: float("nan") for name in SUPPORTED_MODELS},
                level_coverage={name: 0 for name in SUPPORTED_MODELS},
                level_class_counts=level_class_counts,
                status="no hierarchy level supports stratified CV",
            )
        fallback_ctx = _context(matrix.X.iloc[top_indices, :], top_y, matrix.data_type)
        algorithm = _fallback_algorithm(fallback_ctx)
        fallback = ModelResult(
            algorithm=algorithm,
            score=float("nan"),
            parameters=model_parameters(algorithm, fallback_ctx),
            rationale=[
                "No hierarchy level supports reliable stratified cross-validation.",
                "This is a conservative fallback recommendation rather than a validated model ranking.",
            ],
            levels_evaluated=0,
            levels_total=n_levels,
        )
        return HierarchySelectionResult(
            n_levels=n_levels,
            evaluable_levels=0,
            best=fallback,
            alternatives=[],
            mean_scores={name: float("nan") for name in SUPPORTED_MODELS},
            level_coverage={name: 0 for name in SUPPORTED_MODELS},
            level_class_counts=level_class_counts,
            status="fallback: no hierarchy level supports stratified CV",
        )

    # Use a conservative aggregate context to generate one parameter set that
    # can be applied when run.py is asked to force the same algorithm globally.
    if contexts:
        aggregate_counts: Dict[str, int] = {}
        aggregate_ctx = SelectionContext(
            n_samples=max(ctx.n_samples for ctx in contexts),
            n_features=matrix.X.shape[1],
            n_classes=max(ctx.n_classes for ctx in contexts),
            class_counts=aggregate_counts,
            imbalance_ratio=max(ctx.imbalance_ratio for ctx in contexts),
            data_type=matrix.data_type,
            missing_fraction=max(ctx.missing_fraction for ctx in contexts),
        )
    else:
        top_indices, top_y = hierarchy_level_target(matrix.labels, 0, delimiter)
        aggregate_ctx = _context(matrix.X.iloc[top_indices, :], top_y, matrix.data_type)

    selected = ranked[: 1 + top_alternatives]
    recommendations: List[ModelResult] = []
    for algorithm, score, n_covered in selected:
        recommendations.append(ModelResult(
            algorithm=algorithm,
            score=score,
            parameters=model_parameters(algorithm, aggregate_ctx),
            rationale=[
                f"Mean stratified CV balanced accuracy across {n_covered} evaluable hierarchy level(s): {score:.4f}.",
                "This is a general matrix-level recommendation; individual hierarchy nodes may prefer different models.",
            ],
            levels_evaluated=n_covered,
            levels_total=n_levels,
        ))

    return HierarchySelectionResult(
        n_levels=n_levels,
        evaluable_levels=evaluable_levels,
        best=recommendations[0],
        alternatives=recommendations[1:],
        mean_scores={name: mean_scores.get(name, float("nan")) for name in SUPPORTED_MODELS},
        level_coverage={name: coverage.get(name, 0) for name in SUPPORTED_MODELS},
        level_class_counts=level_class_counts,
        status="ok",
    )


def get_model_availability() -> Dict[str, Dict[str, Any]]:
    """Return dependency availability information for selector candidates.

    All sklearn-based models are required by this module and therefore marked
    available if the module itself imported successfully. XGBoost is optional.
    """
    info: Dict[str, Dict[str, Any]] = {
        name: {"available": True, "reason": ""}
        for name in SUPPORTED_MODELS
        if name != "XGBoost"
    }
    info["XGBoost"] = {
        "available": bool(XGBOOST_AVAILABLE),
        "version": XGBOOST_VERSION or None,
        "reason": "" if XGBOOST_AVAILABLE else (
            XGBOOST_IMPORT_ERROR or "xgboost package is not installed"
        ),
    }
    return info


def validate_neuralnetwork_module(module: Any) -> List[str]:
    """Return expected model class names missing from an imported NeuralNetwork module."""
    return [name for name in SUPPORTED_MODELS if not hasattr(module, name)]


if __name__ == "__main__":
    raise SystemExit(
        "This is a library module. Import it from model_selector.py or run.py."
    )
