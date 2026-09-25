"""Detect query sample type from a file or directory."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

FASTA_SUFFIXES = {".fa", ".fna", ".fasta", ".fas"}
FASTQ_SUFFIXES = (".fastq", ".fq", ".fastq.gz", ".fq.gz")
VCF_SUFFIXES = (".vcf", ".vcf.gz", ".g.vcf", ".g.vcf.gz")
MATRIX_SUFFIXES = {".csv", ".tsv", ".txt"}
VALID_INPUT_TYPES = ("auto", "vcf", "fasta", "fastq", "matrix")


def _name_matches(name: str, suffixes: Iterable[str]) -> bool:
    lower = name.lower()
    return any(lower.endswith(suffix) for suffix in suffixes)


def detect_input_type(sample_path: str | Path, requested: str = "auto") -> str:
    """Return vcf, fasta, fastq, or matrix.

    ``requested`` is used as-is unless it is ``auto``. Directories of mixed
    FASTQ + VCF prefer FASTQ, matching NetworkParser query detection.
    """
    requested = str(requested or "auto").strip().lower()
    if requested in {"raw_sequence", "raw_fasta", "sequence"}:
        return "fasta"
    if requested != "auto":
        if requested not in VALID_INPUT_TYPES:
            raise ValueError(
                f"input type must be one of: {', '.join(VALID_INPUT_TYPES)}"
            )
        return requested

    candidate = Path(sample_path)
    if candidate.is_file():
        name = candidate.name.lower()
        if candidate.suffix.lower() in FASTA_SUFFIXES:
            return "fasta"
        if _name_matches(name, FASTQ_SUFFIXES):
            return "fastq"
        if _name_matches(name, VCF_SUFFIXES):
            return "vcf"
        if candidate.suffix.lower() in MATRIX_SUFFIXES:
            return "matrix"
        return "matrix"

    if candidate.is_dir():
        names = [path.name.lower() for path in candidate.iterdir() if path.is_file()]
        if any(_name_matches(name, FASTQ_SUFFIXES) for name in names):
            return "fastq"
        if any(_name_matches(name, VCF_SUFFIXES) for name in names):
            return "vcf"
        if any(Path(name).suffix.lower() in FASTA_SUFFIXES for name in names):
            return "fasta"
        return "matrix"

    return "matrix"
