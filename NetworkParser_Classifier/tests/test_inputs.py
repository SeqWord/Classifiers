from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mtb_amr_classifier.inputs import detect_input_type


class TestDetectInputType(unittest.TestCase):
    def test_explicit_types(self):
        self.assertEqual(detect_input_type("unused", "vcf"), "vcf")
        self.assertEqual(detect_input_type("unused", "fasta"), "fasta")
        self.assertEqual(detect_input_type("unused", "fastq"), "fastq")
        self.assertEqual(detect_input_type("unused", "raw_sequence"), "fasta")

    def test_invalid_type(self):
        with self.assertRaises(ValueError):
            detect_input_type("unused", "bam")

    def test_file_suffixes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vcf = root / "sample.vcf.gz"
            fasta = root / "sample.fasta"
            matrix = root / "sample.csv"
            for path in (vcf, fasta, matrix):
                path.write_text("")
            self.assertEqual(detect_input_type(vcf), "vcf")
            self.assertEqual(detect_input_type(fasta), "fasta")
            self.assertEqual(detect_input_type(matrix), "matrix")

    def test_fastq_directory_preferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "s_R1.fastq.gz").write_text("")
            (root / "s_R2.fastq.gz").write_text("")
            (root / "other.vcf").write_text("")
            self.assertEqual(detect_input_type(root), "fastq")

    def test_vcf_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.vcf.gz").write_text("")
            self.assertEqual(detect_input_type(root), "vcf")
