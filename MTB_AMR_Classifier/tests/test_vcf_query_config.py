from __future__ import annotations

import unittest

from mtb_amr_classifier.config import NetworkParserConfig
from mtb_amr_classifier.query_engine import apply_trained_vcf_config
from mtb_amr_classifier.vcf_call_semantics import VcfQCConfig


class TestDefaultVcfQuerySettings(unittest.TestCase):
    def test_defaults_match_afro_variant_only_training(self):
        config = NetworkParserConfig()
        config.__post_init__()
        self.assertEqual(config.min_gq_per_sample, 0)
        self.assertTrue(config.assume_absent_variant_is_reference)
        qc = VcfQCConfig.from_config(config)
        self.assertEqual(qc.min_gq, 0)
        self.assertTrue(qc.assume_absent_variant_is_reference)

    def test_bundle_registry_overrides_unsafe_defaults(self):
        config = NetworkParserConfig()
        config.min_gq_per_sample = 20
        config.assume_absent_variant_is_reference = False
        registry = {
            "config": {
                "min_gq_per_sample": 0,
                "assume_absent_variant_is_reference": True,
            }
        }
        apply_trained_vcf_config(config, registry)
        self.assertEqual(config.min_gq_per_sample, 0)
        self.assertTrue(config.assume_absent_variant_is_reference)
