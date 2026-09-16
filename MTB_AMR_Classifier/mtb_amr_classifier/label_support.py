"""Query-time label-support policy used when walking a hierarchy path.

Extracted from NetworkParser hierarchy training so MTB_AMR_Classifier can
flag rare classes for review without importing the training protocol.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

try:
    from mtb_amr_classifier.config import NetworkParserConfig
except ImportError:  # pragma: no cover
    from config import NetworkParserConfig  # type: ignore


def build_label_training_support_policy(
    labels_df: pd.DataFrame,
    label_columns: List[str],
    config: NetworkParserConfig,
) -> Dict[str, Any]:
    """Summarize cohort label support and which classes require manual review."""
    min_train = int(getattr(config, "level2_min_class_count", 2))
    drop_enabled = bool(getattr(config, "level2_drop_low_support_classes", True))
    review_enabled = bool(getattr(config, "low_support_review_enabled", True))
    review_min = int(getattr(config, "low_support_review_min_class_count", 10))
    review_label = str(
        getattr(config, "low_support_review_label", "low_support_review_required")
    )
    action_message = str(
        getattr(
            config,
            "low_support_review_action_message",
            (
                "Manually review this sample or merge rare classes in metadata if that "
                "grouping is biologically appropriate."
            ),
        )
    )

    per_label: Dict[str, Any] = {}
    for label_col in label_columns:
        col = str(label_col).strip()
        if not col or col not in labels_df.columns:
            continue

        series = labels_df[col].astype(str).str.strip()
        series = series.replace(
            {
                "": pd.NA,
                "-": pd.NA,
                "NA": pd.NA,
                "N/A": pd.NA,
                "None": pd.NA,
                "nan": pd.NA,
                "NaN": pd.NA,
            }
        ).dropna()
        counts = series.value_counts(dropna=True)

        classes: Dict[str, Any] = {}
        for cls, cnt in counts.items():
            label = str(cls).strip()
            if not label:
                continue
            sample_count = int(cnt)
            excluded = bool(drop_enabled and sample_count < min_train)
            requires_review = bool(review_enabled and sample_count < review_min)
            classes[label] = {
                "training_sample_count": sample_count,
                "excluded_from_training": excluded,
                "requires_manual_review": requires_review,
            }

        per_label[col] = {
            "label_column": col,
            "min_class_count_for_training": int(min_train),
            "min_class_count_for_confident_reporting": int(review_min),
            "classes": classes,
            "excluded_from_training": sorted(
                label
                for label, payload in classes.items()
                if bool(payload.get("excluded_from_training"))
            ),
            "review_required_classes": sorted(
                label
                for label, payload in classes.items()
                if bool(payload.get("requires_manual_review"))
            ),
        }

    return {
        "status": "active" if review_enabled else "disabled",
        "review_label": review_label,
        "recommended_action": action_message,
        "policy_summary": (
            "Classes below the confident-reporting threshold are emitted as "
            f"'{review_label}' during query so rare labels are not over-called."
        ),
        "per_label": per_label,
    }
