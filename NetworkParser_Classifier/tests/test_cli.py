from __future__ import annotations

import unittest
from pathlib import Path

from mtb_amr_classifier.cli import build_parser, build_predict_parser, main
from mtb_amr_classifier.predict import predict_hierarchy


class TestCLI(unittest.TestCase):
    def test_predict_parser_requires_model_and_sample(self):
        parser = build_predict_parser()
        args = parser.parse_args(
            [
                "--model",
                "bundle.npb",
                "--sample",
                "sample.vcf",
                "--output_dir",
                "out",
            ]
        )
        self.assertEqual(args.model, "bundle.npb")
        self.assertEqual(args.sample, "sample.vcf")
        self.assertEqual(args.input_type, "auto")

    def test_bundle_and_genomic_aliases(self):
        parser = build_predict_parser()
        args = parser.parse_args(
            [
                "--bundle",
                "model.npb",
                "--genomic",
                "reads/",
                "--output_dir",
                "out",
                "--input-type",
                "fastq",
            ]
        )
        self.assertEqual(args.model, "model.npb")
        self.assertEqual(args.sample, "reads/")
        self.assertEqual(args.input_type, "fastq")

    def test_subcommand_help(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "predict",
                "--model",
                "bundle.npb",
                "--sample",
                "x.fasta",
                "--output_dir",
                "out",
            ]
        )
        self.assertEqual(args.command, "predict")

    def test_missing_model_returns_error(self):
        code = main(
            [
                "--model",
                "does-not-exist.npb",
                "--sample",
                "does-not-exist.vcf",
                "--output_dir",
                "out",
            ]
        )
        self.assertEqual(code, 2)

    def test_predict_hierarchy_missing_files(self):
        with self.assertRaises(FileNotFoundError):
            predict_hierarchy(
                model=Path("missing.npb"),
                sample=Path("missing.vcf"),
                output_dir=Path("out"),
            )
