from __future__ import annotations

import unittest

import pandas as pd

from mtb_amr_classifier.config import NetworkParserConfig
from mtb_amr_classifier.label_support import build_label_training_support_policy
from mtb_amr_classifier.predict import write_hierarchy_paths


class TestLabelSupport(unittest.TestCase):
    def test_rare_class_is_flagged(self):
        config = NetworkParserConfig()
        config.level2_min_class_count = 2
        config.low_support_review_min_class_count = 10
        config.__post_init__()
        labels = pd.DataFrame(
            {
                "AMR_binary": ["resistant"] * 20 + ["susceptible"] * 3,
            }
        )
        policy = build_label_training_support_policy(
            labels, ["AMR_binary"], config
        )
        classes = policy["per_label"]["AMR_binary"]["classes"]
        self.assertFalse(classes["resistant"]["requires_manual_review"])
        self.assertTrue(classes["susceptible"]["requires_manual_review"])
        self.assertIn("susceptible", policy["per_label"]["AMR_binary"]["review_required_classes"])


class TestHierarchyPathTable(unittest.TestCase):
    def test_writes_compact_path_table(self):
        import tempfile
        from pathlib import Path

        predictions = pd.DataFrame(
            {
                "sample_id": ["ERR1"],
                "predicted_hierarchy_path": ["Lineage_clean=L4 / AMR_binary=resistant"],
                "predicted_terminal_label": ["resistant"],
                "other_column": ["ignored"],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_hierarchy_paths(predictions, Path(tmp))
            table = pd.read_csv(path, sep="\t")
            self.assertEqual(list(table["sample_id"]), ["ERR1"])
            self.assertIn("predicted_hierarchy_path", table.columns)
            self.assertNotIn("other_column", table.columns)
