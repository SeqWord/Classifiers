"""
network_parser.data_loader

Build a **sample × variant** binary matrix (0/1) from per-sample VCF files.

Core responsibilities:
  - Parse per-sample VCF/VCF.GZ files.
  - Apply INFO/QUAL QC (QUAL/DP/MQ/MQ0F).
  - Enforce a cohort-level presence threshold (minimum #samples with the SNP).
  - Build an allelic matrix (REF + per-sample alleles) and a binary matrix (0/1)
    using a configurable baseline:
      * ancestral_allele='Y' → baseline = reference allele
      * ancestral_allele='N' → baseline = cohort mode allele (most common base)

Artifact responsibilities (when output_dir is provided):
  - Write outputs matching the three legacy scripts:
      vcf_counts/all_snp.txt
      fasta/<generic>_alleles.fasta
      fasta/<generic>_binary.fasta
      fasta/<generic>_filtered.tsv      (+ optional Context_±40)
      matrices/<generic>_alleles.tsv
      matrices/<generic>_binary.tsv
      matrices/<generic>_alleles.fasta
      matrices/<generic>_binary.fasta
      matrices/<generic>_filtered.tsv

Non-responsibilities:
  - Statistical validation (χ² / Fisher + FDR) must happen BEFORE tree construction
  - Decision tree building
  - Post-tree bootstrapping / confidence scoring

Returned value:
  - pandas.DataFrame: rows = samples, columns = variants, values ∈ {0,1}
"""

from __future__ import annotations

from bisect import bisect_right
import os
import csv
import gzip
import json
import logging
import time
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from concurrent.futures import ProcessPoolExecutor
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from mtb_amr_classifier.utils import (
        log_flow_step,
        log_filter_step,
        log_artifact,
        progress_iter,
    )
    from mtb_amr_classifier.vcf_call_semantics import (
        AlleleCall,
        CallSet,
        CallState,
        VcfQCConfig,
        assert_unique_sample_ids,
        encode_binary_for_feature,
        iter_sample_calls as shared_iter_sample_calls,
        resolve_feature_call,
        warn_legacy_absence_assumed_reference,
    )
except ImportError:  # pragma: no cover - supports direct source-tree execution
    try:
        from utils import log_flow_step, log_filter_step, log_artifact, progress_iter  # type: ignore
        from vcf_call_semantics import (  # type: ignore
            AlleleCall,
            CallSet,
            CallState,
            VcfQCConfig,
            assert_unique_sample_ids,
            encode_binary_for_feature,
            iter_sample_calls as shared_iter_sample_calls,
            resolve_feature_call,
            warn_legacy_absence_assumed_reference,
        )
    except ImportError:  # pragma: no cover
        log_flow_step = None  # type: ignore
        log_filter_step = None  # type: ignore
        log_artifact = None  # type: ignore

        def progress_iter(iterable, **kwargs):  # type: ignore
            return iterable

        shared_iter_sample_calls = None  # type: ignore
        resolve_feature_call = None  # type: ignore


def _minor_count_chunk(cols: List[List[str]]) -> List[Tuple[int, int]]:
    # returns [(count0, count1), ...] for each col in chunk
    out = []
    for col in cols:
        c1 = 0
        for v in col:
            if v == "1":
                c1 += 1
        c0 = len(col) - c1
        out.append((c0, c1))
    return out


def minor_count_filter_parallel(
    binary_cols: List[List[str]], min_count: int, n_jobs: int
) -> List[bool]:
    """
    Parallel minor-count filter across columns.

    Note: use ProcessPoolExecutor because Python loops are GIL-bound.
    Keep chunk sizes reasonably large to avoid overhead.
    """
    if min_count <= 0:
        return [True] * len(binary_cols)
    if n_jobs is None or n_jobs == 1 or len(binary_cols) < 5000:
        # small: parallel overhead not worth it
        return minor_count_filter(binary_cols, min_count)

    # choose chunks (tune)
    n_cols = len(binary_cols)
    n_workers = os.cpu_count() if n_jobs < 0 else n_jobs
    n_workers = max(1, int(n_workers))
    chunk_size = max(1000, n_cols // (n_workers * 4))

    chunks = [binary_cols[i : i + chunk_size] for i in range(0, n_cols, chunk_size)]

    results: List[Tuple[int, int]] = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for part in ex.map(_minor_count_chunk, chunks):
            results.extend(part)

    keep = []
    for c0, c1 in results:
        keep.append(min(c0, c1) >= min_count)
    return keep


def _fmt_n_jobs(n_jobs: Optional[int]) -> str:
    """Log-friendly n_jobs formatting."""
    if n_jobs is None:
        return "default"
    if isinstance(n_jobs, int) and n_jobs < 0:
        return "all cores"
    return str(n_jobs)


def _safe_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


try:
    from Bio import SeqIO
    from Bio.Seq import Seq

    HAVE_BIO = True
except ImportError:  # pragma: no cover
    HAVE_BIO = False


# ──────────────────────────────────────────────────────────────
# Basic file + VCF parsing helpers
# ──────────────────────────────────────────────────────────────


def open_any(path: Path):
    """Open plain-text or gzipped text files in read-text mode."""
    p = str(path)
    return (
        gzip.open(p, "rt")
        if p.endswith(".gz")
        else open(p, "r", encoding="utf-8", errors="replace")
    )


def parse_info_field(info_str: str) -> Dict[str, str]:
    """Parse the VCF INFO column into a dictionary of string values."""
    info: Dict[str, str] = {}
    if not info_str:
        return info
    for token in info_str.split(";"):
        if "=" in token:
            k, v = token.split("=", 1)
            info[k] = v
    return info


def is_snp_like(ref: str, alt_field: str, biallelic_only: bool = True) -> bool:
    """Return True if the record looks like a SNP (single-base REF and ALT(s))."""
    if not ref or not alt_field:
        return False
    if ref == "." or alt_field == ".":
        return False
    alts = alt_field.split(",")
    if biallelic_only and len(alts) != 1:
        return False
    if len(ref) != 1:
        return False
    if any(len(a) != 1 for a in alts):
        return False
    return True


def passes_info_qc(
    qual_str: str,
    info: Dict[str, str],
    qual_thresh: float,
    dp_thresh: int,
    mq_thresh: float,
    mq0f_thresh: float,
) -> bool:
    """Backward-compatible INFO/QUAL filter (delegates to shared QC)."""
    try:
        from mtb_amr_classifier.vcf_call_semantics import VcfQCConfig, record_fails_site_qc
    except ImportError:  # pragma: no cover
        from vcf_call_semantics import VcfQCConfig, record_fails_site_qc  # type: ignore

    qc = VcfQCConfig(
        qual_threshold=float(qual_thresh),
        min_dp=int(dp_thresh),
        min_gq=0,  # legacy helper did not enforce GQ
        mq_threshold=float(mq_thresh),
        mq0f_threshold=float(mq0f_thresh),
        respect_filter=False,  # legacy helper ignored FILTER
    )
    failed, _ = record_fails_site_qc(
        qual_str=qual_str,
        filter_field="PASS",
        info=info or {},
        sample_map={},
        qc=qc,
    )
    return not failed


def choose_called_allele(
    ref: str,
    alts: List[str],
    fmt: Optional[str],
    sample_field: Optional[str],
) -> str:
    """
    Shared-safe allele choice.

    Missing / empty GT never becomes ALT. Returns empty string when no callable
    allele can be determined (callers must not treat empty as REF/ALT).
    """
    try:
        from mtb_amr_classifier.vcf_call_semantics import (
            choose_called_allele as shared_choose,
        )
    except ImportError:  # pragma: no cover
        from vcf_call_semantics import choose_called_allele as shared_choose  # type: ignore

    called, state = shared_choose(ref, alts, fmt, sample_field)
    if called is None:
        return ""
    return str(called).upper()


def iter_sample_calls(
    vcf_path: Path,
    qual_thresh: float,
    dp_thresh: int,
    mq_thresh: float,
    mq0f_thresh: float,
    biallelic_only: bool = True,
    gq_thresh: Optional[int] = None,
    qc: Optional[Any] = None,
) -> CallSet:
    """
    Extract per-sample calls after shared QC/genotype semantics.

    Returns a CallSet (coordinate → AlleleCall), including FILTER/GQ handling
    when thresholds are provided via ``qc`` or keyword arguments.
    """
    if shared_iter_sample_calls is None:
        raise RuntimeError("vcf_call_semantics is required for VCF parsing")
    return shared_iter_sample_calls(
        Path(vcf_path),
        qc=qc,
        qual_thresh=qual_thresh,
        dp_thresh=dp_thresh,
        mq_thresh=mq_thresh,
        mq0f_thresh=mq0f_thresh,
        biallelic_only=biallelic_only,
        gq_thresh=gq_thresh,
    )


# ──────────────────────────────────────────────────────────────
# Reference + context helpers (used for filtered TSV context column)
# ──────────────────────────────────────────────────────────────


def load_reference_sequence(ref_path: Path) -> Optional[str]:
    """Load reference sequence from FASTA or GenBank.

    If multiple records exist, sequences are concatenated in file order.
    Requires Biopython.
    """
    if not ref_path.exists():
        return None
    if not HAVE_BIO:
        raise RuntimeError(
            "Biopython is required for reference sequence loading but is not available."
        )

    lower = ref_path.name.lower()
    fmt = "fasta" if lower.endswith((".fa", ".fna", ".fasta", ".fas")) else None
    if lower.endswith((".gb", ".gbk", ".gbff")):
        fmt = "genbank"

    seqs: List[str] = []
    if fmt:
        for rec in SeqIO.parse(str(ref_path), fmt):
            seqs.append(str(rec.seq).upper())
    else:
        # Try FASTA then GenBank
        try:
            for rec in SeqIO.parse(str(ref_path), "fasta"):
                seqs.append(str(rec.seq).upper())
        except Exception:
            seqs = []
        if not seqs:
            for rec in SeqIO.parse(str(ref_path), "genbank"):
                seqs.append(str(rec.seq).upper())

    if not seqs:
        return None
    return "".join(seqs).upper()


def load_reference_sequences(ref_path: Path) -> Dict[str, str]:
    """Load reference records as ``record_id -> sequence`` for context extraction.

    This preserves contig/chromosome identity for raw-sequence query manifests.
    The older ``load_reference_sequence`` helper concatenates records for legacy
    behaviour, but selected-feature context should be extracted from the same
    reference record named in the feature manifest whenever possible.
    """
    if not ref_path.exists():
        return {}
    if not HAVE_BIO:
        raise RuntimeError(
            "Biopython is required for reference sequence loading but is not available."
        )

    lower = ref_path.name.lower()
    formats: List[str] = []
    if lower.endswith((".fa", ".fna", ".fasta", ".fas")):
        formats = ["fasta"]
    elif lower.endswith((".gb", ".gbk", ".gbff")):
        formats = ["genbank"]
    else:
        formats = ["fasta", "genbank"]

    records: Dict[str, str] = {}
    for fmt in formats:
        try:
            parsed = list(SeqIO.parse(str(ref_path), fmt))
        except Exception:
            parsed = []
        if not parsed:
            continue

        for rec in parsed:
            seq = str(rec.seq).upper()
            if not seq:
                continue
            keys = {str(rec.id), str(rec.name)}
            description_first = (
                str(rec.description).split()[0] if str(rec.description).strip() else ""
            )
            if description_first:
                keys.add(description_first)
            for key in keys:
                key = key.strip()
                if key and key not in records:
                    records[key] = seq
        if records:
            return records

    return records


def context_around(pos_1based: int, genome: str, flank: int = 40) -> str:
    """Extract circular ±flank context around a 1-based position."""
    n = len(genome)
    if n == 0:
        return ""
    i = (pos_1based - 1) % n
    out = []
    for off in range(-flank, flank + 1):
        out.append(genome[(i + off) % n])
    return "".join(out)


# ──────────────────────────────────────────────────────────────
# Optional GenBank annotation table (all_snp.txt with annotation columns)
# ──────────────────────────────────────────────────────────────


def _complement_base(base: str) -> str:
    return (
        str(Seq(base).complement())
        if HAVE_BIO
        else {"A": "T", "T": "A", "G": "C", "C": "G"}.get(base.upper(), base)
    )


def _location_contains(feature, pos0: int) -> bool:
    """True if 0-based genomic coordinate is inside a (possibly compound) CDS."""
    try:
        return (
            bool(
                feature.location.nofuzzy_start <= pos0 < feature.location.nofuzzy_end
                and any(
                    int(part.start) <= pos0 < int(part.end)
                    for part in feature.location.parts
                )
            )
            if hasattr(feature.location, "parts")
            else (int(feature.location.start) <= pos0 < int(feature.location.end))
        )
    except Exception:
        try:
            return pos0 in feature.location
        except Exception:
            return False


def _cds_coding_index(feature, pos0: int) -> Optional[int]:
    """
    0-based index along the spliced coding sequence for a genomic coordinate,
    or None if the position is not in the CDS.
    """
    strand = int(feature.location.strand or 1)
    parts = (
        list(feature.location.parts)
        if hasattr(feature.location, "parts")
        else [feature.location]
    )
    # Walk parts in transcription order
    ordered = parts if strand >= 0 else list(reversed(parts))
    offset = 0
    for part in ordered:
        start = int(part.start)
        end = int(part.end)
        if start <= pos0 < end:
            if strand >= 0:
                return offset + (pos0 - start)
            return offset + (end - 1 - pos0)
        offset += max(0, end - start)
    return None


def _extract_coding_codon(
    sequence,
    feature,
    coding_index: int,
    *,
    codon_start: int = 1,
    table: int = 11,
) -> Tuple[str, int, str, str]:
    """
    Return (ref_codon, position_in_codon_0based, ref_aa, mut placeholder).

    codon_start is GenBank codon_start (1,2,3).
    """
    # Extract spliced CDS sequence on coding strand via Biopython
    try:
        cds_seq = feature.extract(sequence)
    except Exception:
        cds_seq = sequence[int(feature.location.start) : int(feature.location.end)]
        if int(feature.location.strand or 1) < 0 and HAVE_BIO:
            cds_seq = cds_seq.reverse_complement()

    phase = max(0, int(codon_start) - 1)
    # Align coding_index to codon frame
    rel = coding_index - phase
    if rel < 0:
        return "NNN", 0, "X", "X"
    codon_number0 = rel // 3
    pos_in_codon = rel % 3
    codon_start_idx = phase + codon_number0 * 3
    ref_codon = str(cds_seq[codon_start_idx : codon_start_idx + 3]).upper()
    if len(ref_codon) < 3:
        ref_codon = (ref_codon + "NNN")[:3]
    try:
        ref_aa = str(Seq(ref_codon).translate(table=table))
    except Exception:
        ref_aa = "X"
    return ref_codon, pos_in_codon, ref_aa, str(codon_number0 + 1)


@dataclass
class _CdsIntervalIndex:
    """Reusable interval index for the CDS features of one reference record."""

    starts: List[int]
    start_entries: List[Tuple[int, int, int, int, Any]]
    prefix_max_ends: List[int]
    ends: List[int]
    end_entries: List[Tuple[int, int, int, int, Any]]

    @classmethod
    def build(cls, record: Any) -> "_CdsIntervalIndex":
        entries: List[Tuple[int, int, int, int, Any]] = []
        features = [feature for feature in record.features if feature.type == "CDS"]
        for feature_order, feature in enumerate(features):
            parts = (
                list(feature.location.parts)
                if hasattr(feature.location, "parts")
                else [feature.location]
            )
            for part_order, part in enumerate(parts):
                start = int(part.start)
                end = int(part.end)
                if end > start:
                    entries.append(
                        (start, end, feature_order, part_order, feature)
                    )

        start_entries = sorted(
            entries, key=lambda entry: (entry[0], entry[2], entry[3], entry[1])
        )
        prefix_max_ends: List[int] = []
        running_max = -1
        for _, end, _, _, _ in start_entries:
            running_max = max(running_max, end)
            prefix_max_ends.append(running_max)

        end_entries = sorted(
            entries, key=lambda entry: (entry[1], entry[2], entry[3], entry[0])
        )
        return cls(
            starts=[entry[0] for entry in start_entries],
            start_entries=start_entries,
            prefix_max_ends=prefix_max_ends,
            ends=[entry[1] for entry in end_entries],
            end_entries=end_entries,
        )

    def overlapping(self, pos0: int) -> List[Any]:
        """Return all CDS features containing a coordinate in record order."""
        idx = bisect_right(self.starts, int(pos0)) - 1
        hits: Dict[int, Any] = {}
        while idx >= 0 and self.prefix_max_ends[idx] > pos0:
            start, end, feature_order, _, feature = self.start_entries[idx]
            if start <= pos0 < end:
                hits.setdefault(feature_order, feature)
            idx -= 1
        return [hits[order] for order in sorted(hits)]

    def nearest(
        self, pos0: int
    ) -> Tuple[Optional[Tuple[int, int, int, int, Any]], int]:
        """Return the nearest CDS part and distance using binary searches."""
        overlaps = self.overlapping(pos0)
        if overlaps:
            first = min(overlaps, key=lambda feature: self._feature_order(feature))
            for entry in self.start_entries:
                if entry[4] is first and entry[0] <= pos0 < entry[1]:
                    return entry, 0

        candidates: List[Tuple[int, int, int, Tuple[int, int, int, int, Any]]] = []

        right_idx = bisect_right(self.starts, int(pos0))
        if right_idx < len(self.start_entries):
            nearest_start = self.start_entries[right_idx][0]
            idx = right_idx
            while idx < len(self.start_entries) and self.start_entries[idx][0] == nearest_start:
                entry = self.start_entries[idx]
                candidates.append((nearest_start - pos0, entry[2], entry[3], entry))
                idx += 1

        left_idx = bisect_right(self.ends, int(pos0)) - 1
        if left_idx >= 0:
            nearest_end = self.end_entries[left_idx][1]
            idx = left_idx
            while idx >= 0 and self.end_entries[idx][1] == nearest_end:
                entry = self.end_entries[idx]
                candidates.append((pos0 - (nearest_end - 1), entry[2], entry[3], entry))
                idx -= 1

        if not candidates:
            return None, -1
        distance, _, _, entry = min(candidates, key=lambda item: item[:3])
        return entry, int(distance)

    def _feature_order(self, target: Any) -> int:
        for _, _, feature_order, _, feature in self.start_entries:
            if feature is target:
                return feature_order
        return len(self.start_entries)


def annotate_snps_genbank(
    snp_details: Dict[Tuple[str, int, str, str], int],
    ref_gbk_path: Path,
    *,
    reference_id: Optional[str] = None,
    allow_circular_wrap: bool = False,
    circular_contigs: Optional[Sequence[str]] = None,
) -> List[Dict[str, str]]:
    """Annotate SNPs using a contig-aware GenBank (or multi-record) reference.

    Input keys: (chrom, pos, ref_nt, alt_nt) -> count

    Negative-strand CDS substitutions transform both the codon and the ALT
    allele into coding-strand orientation before translation.
    Multi-record references are never concatenated for coordinate mapping.

    All overlapping CDS features are reported (not just the first hit).
    Out-of-range positions are explicitly rejected.
    ``circular_contigs`` declares which contigs may wrap. The legacy
    ``allow_circular_wrap`` switch applies to all contigs and is retained only
    for compatibility; topology is never inferred from a filename.
    """
    if not HAVE_BIO:
        raise RuntimeError(
            "Biopython is required for GenBank annotation but is not available."
        )

    import hashlib as _hashlib

    annotation_started_at = time.perf_counter()

    ref_gbk_path = Path(ref_gbk_path)
    ref_bytes = ref_gbk_path.read_bytes()
    ref_checksum = _hashlib.sha256(ref_bytes).hexdigest()
    with open(ref_gbk_path, "r", encoding="utf-8") as handle:
        records = list(SeqIO.parse(handle, "genbank"))
    if not records:
        with open(ref_gbk_path, "r", encoding="utf-8") as handle:
            records = [SeqIO.read(handle, "genbank")]
    declared_ref_id = (
        str(reference_id).strip()
        if reference_id
        else ";".join(str(record.id) for record in records)
    )
    circular_names = {
        str(value).strip() for value in (circular_contigs or []) if str(value).strip()
    }

    # Map contig/id aliases -> record
    by_id: Dict[str, Any] = {}
    for rec in records:
        for key in {str(rec.id), str(rec.name), str(rec.id).split(".")[0]}:
            if key and key not in by_id:
                by_id[key] = rec

    index_started_at = time.perf_counter()
    cds_indexes = {id(record): _CdsIntervalIndex.build(record) for record in records}
    index_seconds = time.perf_counter() - index_started_at
    indexed_parts = sum(
        len(index.start_entries) for index in cds_indexes.values()
    )
    logger.info(
        "GenBank annotation: CDS indexes built | records=%d | interval_parts=%d | seconds=%.3f",
        len(records),
        indexed_parts,
        index_seconds,
    )

    rows: List[Dict[str, str]] = []
    items: List[Tuple[str, int, str, str, int]] = []
    for (chrom, pos, ref_nt, alt_nt), count in snp_details.items():
        items.append((str(chrom), int(pos), str(ref_nt), str(alt_nt), int(count)))

    def _base_row(chrom, pos, count, ref_nt_u, alt_nt_u, **extra) -> Dict[str, str]:
        base = {
            "Feature_ID": f"{chrom}:{pos}:{ref_nt_u}:{alt_nt_u}",
            "Position": str(pos),
            "Count": str(count),
            "Sequence": chrom,
            "Ref_allele": ref_nt_u,
            "Alt_allele": alt_nt_u,
            "Reference_id": declared_ref_id,
            "Reference_build": declared_ref_id,  # not filename-as-build
            "Reference_checksum_sha256": ref_checksum,
            "Contig": chrom,
        }
        base.update({k: str(v) for k, v in extra.items()})
        return base

    for chrom, pos, ref_nt, alt_nt, count in sorted(
        items, key=lambda x: (x[0], x[1], x[2], x[3])
    ):
        ref_nt_u = ref_nt.upper()
        alt_nt_u = alt_nt.upper()
        pos0 = pos - 1
        rec = by_id.get(chrom) or by_id.get(chrom.split("|")[0])
        if rec is None and len(records) == 1:
            rec = records[0]
        if rec is None:
            rows.append(
                _base_row(
                    chrom,
                    pos,
                    count,
                    ref_nt_u,
                    alt_nt_u,
                    Region_type="unannotated_contig_mismatch",
                    Relative_pos="-1",
                    Codon_number="0",
                    Nucleotide_change=f"{ref_nt_u}|{alt_nt_u}",
                    Amino_acid_change="NA",
                    Gene_annotation=f". | . | contig_not_in_reference:{chrom} | [.]",
                )
            )
            continue

        sequence = rec.seq
        seq_len = len(sequence)
        # Explicit out-of-range rejection (circular wrap is opt-in only)
        if pos0 < 0 or pos0 >= seq_len:
            contig_is_circular = bool(
                allow_circular_wrap
                or chrom in circular_names
                or str(rec.id) in circular_names
                or str(rec.name) in circular_names
            )
            if contig_is_circular and seq_len > 0:
                pos0 = pos0 % seq_len
                pos = pos0 + 1
            else:
                rows.append(
                    _base_row(
                        chrom,
                        pos,
                        count,
                        ref_nt_u,
                        alt_nt_u,
                        Region_type="out_of_range",
                        Relative_pos="-1",
                        Codon_number="0",
                        Nucleotide_change=f"{ref_nt_u}|{alt_nt_u}",
                        Amino_acid_change="NA",
                        Gene_annotation=f". | . | position_out_of_range:{pos}>{seq_len} | [.]",
                        Contig=str(rec.id),
                    )
                )
                continue

        genome_ref = str(sequence[pos0]).upper()
        if genome_ref and ref_nt_u and genome_ref != ref_nt_u:
            rows.append(
                _base_row(
                    chrom,
                    pos,
                    count,
                    ref_nt_u,
                    alt_nt_u,
                    Region_type="ref_mismatch",
                    Relative_pos="-1",
                    Codon_number="0",
                    Nucleotide_change=f"{ref_nt_u}|{alt_nt_u}",
                    Amino_acid_change="NA",
                    Gene_annotation=f". | . | REF_mismatch_genome={genome_ref} | [.]",
                    Contig=str(rec.id),
                )
            )
            continue

        cds_index = cds_indexes[id(rec)]
        coding_hits = 0

        for feature in cds_index.overlapping(pos0):
            strand = int(feature.location.strand or 1)
            locus_tag = feature.qualifiers.get("locus_tag", ["."])[0]
            gene = feature.qualifiers.get("gene", ["."])[0]
            product = feature.qualifiers.get("product", ["."])[0]
            codon_start = int(feature.qualifiers.get("codon_start", ["1"])[0] or 1)
            table = int(feature.qualifiers.get("transl_table", ["11"])[0] or 11)
            start = (
                int(feature.location.nofuzzy_start)
                if hasattr(feature.location, "nofuzzy_start")
                else int(feature.location.start)
            )
            end = (
                int(feature.location.nofuzzy_end)
                if hasattr(feature.location, "nofuzzy_end")
                else int(feature.location.end)
            )
            label = f"{'+' if strand >= 0 else '-'}{locus_tag} | {gene} | {product} | [{start+1}..{end}]"

            if _location_contains(feature, pos0):
                coding_idx = _cds_coding_index(feature, pos0)
                if coding_idx is None:
                    continue
                coding_hits += 1
                ref_codon, pos_in_codon, ref_aa, codon_number = _extract_coding_codon(
                    sequence, feature, coding_idx, codon_start=codon_start, table=table
                )
                coding_alt = alt_nt_u if strand >= 0 else _complement_base(alt_nt_u)
                codon_list = list(ref_codon)
                if 0 <= pos_in_codon < len(codon_list):
                    codon_list[pos_in_codon] = coding_alt
                mut_codon = "".join(codon_list)
                try:
                    alt_aa = str(Seq(mut_codon).translate(table=table))
                except Exception:
                    alt_aa = "X"

                rows.append(
                    _base_row(
                        chrom,
                        pos,
                        count,
                        ref_nt_u,
                        alt_nt_u,
                        Region_type="coding",
                        Relative_pos=str(int(coding_idx) + 1),
                        Codon_number=str(codon_number),
                        Nucleotide_change=f"{ref_nt_u}|{alt_nt_u}",
                        Amino_acid_change=f"{ref_aa}|{alt_aa}",
                        Gene_annotation=label,
                        Strand="+" if strand >= 0 else "-",
                        Coding_alt_allele=coding_alt,
                        Contig=str(rec.id),
                        Transl_table=str(table),
                        Codon_start=str(codon_start),
                        Overlapping_cds_index=str(coding_hits),
                    )
                )
                # Continue: report all overlapping CDS, do not stop at first

        if coding_hits == 0:
            nearest_entry, nearest_cds_distance_bp = cds_index.nearest(pos0)
            if nearest_entry is not None:
                ps, pe, _, _, nearest_feature = nearest_entry
                strand = int(nearest_feature.location.strand or 1)
                locus_tag = nearest_feature.qualifiers.get("locus_tag", ["."])[0]
                gene = nearest_feature.qualifiers.get("gene", ["."])[0]
                product = nearest_feature.qualifiers.get("product", ["."])[0]
                label = f"{'+' if strand >= 0 else '-'}{locus_tag} | {gene} | {product} | [{ps+1}..{pe}]"
            else:
                label = ". | . | . | [.]"
                nearest_cds_distance_bp = -1

            rows.append(
                _base_row(
                    chrom,
                    pos,
                    count,
                    ref_nt_u,
                    alt_nt_u,
                    Region_type="non-coding",
                    Relative_pos=str(nearest_cds_distance_bp),
                    Nearest_cds_distance_bp=str(nearest_cds_distance_bp),
                    Codon_number="0",
                    Nucleotide_change=f"{ref_nt_u}|{alt_nt_u}",
                    Amino_acid_change="NA",
                    Gene_annotation=label,
                    Contig=str(rec.id),
                )
            )

    logger.info(
        "GenBank annotation complete | variants=%d | output_rows=%d | "
        "index_seconds=%.3f | total_seconds=%.3f",
        len(items),
        len(rows),
        index_seconds,
        time.perf_counter() - annotation_started_at,
    )
    return rows


# ──────────────────────────────────────────────────────────────
# FASTA matrix I/O (for allele/binary matrices)
# ──────────────────────────────────────────────────────────────


def write_fasta_matrix(
    path: Path, ref_seq: str, sample_map: Dict[str, str], ref_name: str = "REF"
) -> None:
    """Write a REF + sample sequences FASTA that represents a column-aligned matrix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as out:
        out.write(f">{ref_name}\n{ref_seq}\n")
        for name in sorted(sample_map):
            out.write(f">{name}\n{sample_map[name]}\n")


def read_fasta_matrix(path: Path) -> OrderedDict:
    """Read a FASTA matrix into an OrderedDict[name] -> list(chars)."""
    records: OrderedDict = OrderedDict()
    with open(path, "r", encoding="utf-8") as f:
        name = None
        seq_chunks: List[str] = []
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records[name] = list("".join(seq_chunks))
                name = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line.strip())
        if name is not None:
            records[name] = list("".join(seq_chunks))

    if not records:
        raise ValueError(f"FASTA file appears empty: {path}")

    lengths = {len(v) for v in records.values()}
    if len(lengths) != 1:
        raise ValueError(
            f"Sequences in {path} have different lengths: {sorted(lengths)}"
        )

    return records


def write_fasta_matrix_wrapped(
    path: Path, matrix: OrderedDict, line_width: int = 80
) -> None:
    """Write FASTA from OrderedDict[name]->list(chars) with wrapping."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as out:
        for gid, chars in matrix.items():
            out.write(f">{gid}\n")
            seq = "".join(chars)
            for i in range(0, len(seq), line_width):
                out.write(seq[i : i + line_width] + "\n")


# ──────────────────────────────────────────────────────────────
# Matrix conversion + filtering utilities (for matrices/* outputs)
# ──────────────────────────────────────────────────────────────


def transpose_rows_to_columns(matrix: OrderedDict) -> Tuple[List[str], List[List[str]]]:
    """Convert row-oriented FASTA matrix to a column list for per-marker filtering."""
    genomes = list(matrix.keys())
    if not genomes:
        raise ValueError("Empty FASTA matrix.")
    row_len = len(next(iter(matrix.values())))
    cols: List[List[str]] = [[] for _ in range(row_len)]
    for gid in genomes:
        row = matrix[gid]
        if len(row) != row_len:
            raise ValueError("Row lengths differ in FASTA matrix.")
        for j, ch in enumerate(row):
            cols[j].append(ch)
    return genomes, cols


def minor_count_filter(binary_cols: List[List[str]], min_count: int) -> List[bool]:
    """Keep column j only if min(count_0, count_1) >= min_count."""
    keep: List[bool] = []
    for col in binary_cols:
        c0 = sum(1 for x in col if x == "0")
        c1 = sum(1 for x in col if x == "1")
        keep.append(min(c0, c1) >= min_count)
    return keep


def type_filter(annotation_rows: List[Dict[str, str]], typ: str) -> List[bool]:
    """Filter mask by annotation type: all | coding | sense-mutations."""
    if typ == "all":
        return [True] * len(annotation_rows)

    def is_coding(r: Dict[str, str]) -> bool:
        return (r.get("Region_type", "") or "").strip().lower() == "coding"

    def aa_changed(r: Dict[str, str]) -> bool:
        field = (r.get("Amino_acid_change", "") or "").strip()
        if "|" in field:
            left, right = [x.strip() for x in field.split("|", 1)]
            if left and right and left != "-" and right != "-":
                return left != right
        return False

    if typ == "coding":
        return [is_coding(r) for r in annotation_rows]
    if typ == "sense-mutations":
        return [is_coding(r) and aa_changed(r) for r in annotation_rows]
    raise ValueError(f"Unknown type filter: {typ}")


def combine_masks(*masks: List[bool]) -> List[bool]:
    """Combine same-length boolean masks via AND."""
    if not masks:
        return []
    mlen = len(masks[0])
    for m in masks:
        if len(m) != mlen:
            raise ValueError("Mask lengths differ.")
    return [all(m[i] for m in masks) for i in range(mlen)]


def even_pick_indices(sorted_indices: List[int], k: int) -> List[int]:
    """Pick k indices spaced across the sorted list."""
    n = len(sorted_indices)
    if k == 0 or k >= n:
        return sorted_indices[:]
    if k == 1:
        return [sorted_indices[n // 2]]
    chosen = set()
    for i in range(k):
        pos = round(i * (n - 1) / (k - 1))
        chosen.add(pos)
    return [sorted_indices[i] for i in sorted(chosen)]


def group_and_reduce_by_pattern(
    binary_cols: List[List[str]],
    annotation_rows: List[Dict[str, str]],
    repeat_number: int,
    sample_threshold: int = 2000,
    sample_size: int = 256,
    pattern_keys: Optional[Sequence[bytes]] = None,
) -> List[bool]:
    """
    Group identical 0/1 patterns and keep up to repeat_number columns per pattern.

    Each full pattern key is constructed once and then reused for uniqueness
    reporting and exact grouping. Missing values receive their own state and
    are never folded into the ordinary zero/baseline state.
    """
    n_cols = len(binary_cols)
    if n_cols == 0:
        return []

    repeat_number = max(1, int(repeat_number))
    _ = sample_threshold, sample_size  # retained for API compatibility

    positions: List[Optional[int]] = []
    for r in annotation_rows:
        pos_raw = (r.get("Position", "") or "").strip()
        try:
            positions.append(int(pos_raw))
        except ValueError:
            try:
                positions.append(int(float(pos_raw)))
            except Exception:
                positions.append(None)

    if len(positions) < n_cols:
        positions.extend([None] * (n_cols - len(positions)))

    def _sort_cols(cols: List[int]) -> List[int]:
        return sorted(
            cols,
            key=lambda idx: (
                positions[idx] is None,
                positions[idx] if positions[idx] is not None else idx,
            ),
        )

    keep = [False] * n_cols

    if pattern_keys is None:
        pattern_keys = [
            bytes(
                0 if str(value) == "0" else 1 if str(value) == "1" else 2
                for value in column
            )
            for column in binary_cols
        ]
    if len(pattern_keys) != n_cols:
        raise ValueError(
            "pattern_keys length must match the number of binary columns"
        )

    groups: Dict[bytes, List[int]] = defaultdict(list)
    for idx, key in enumerate(pattern_keys):
        groups[bytes(key)].append(idx)

    for cols in groups.values():
        cols_sorted = _sort_cols(cols)
        picked = even_pick_indices(cols_sorted, repeat_number)
        for idx in picked:
            keep[idx] = True

    return keep


def binary_pattern_keys(binary_values: np.ndarray) -> List[bytes]:
    """Encode each matrix column once as an exact reusable 0/1/missing key."""
    values = np.asarray(binary_values, dtype=float)
    if values.ndim != 2:
        raise ValueError("binary_values must be a two-dimensional matrix")
    encoded = np.full(values.shape, 2, dtype=np.uint8)
    encoded[values == 0.0] = 0
    encoded[values == 1.0] = 1
    columns = np.ascontiguousarray(encoded.T)
    return [column.tobytes() for column in columns]


def parse_fix_positions(fix_arg: str, total_cols: int) -> Tuple[List[int], List[str]]:
    """Parse comma/space-separated 1-based positions; return (0-based indices, warnings)."""
    if not fix_arg:
        return [], []
    raw = fix_arg.replace(",", " ").split()
    vals: List[int] = []
    warnings: List[str] = []
    seen = set()
    for token in raw:
        try:
            v = int(token)
        except ValueError:
            warnings.append(f"Ignored non-integer token in --fix: '{token}'")
            continue
        if v <= 0:
            warnings.append(f"Ignored non-positive position in --fix: {v}")
            continue
        if v > total_cols:
            warnings.append(
                f"Ignored out-of-range position in --fix: {v} (max {total_cols})"
            )
            continue
        if v in seen:
            continue
        seen.add(v)
        vals.append(v - 1)
    vals.sort()
    return vals, warnings


def apply_mask_to_char_rows(matrix: OrderedDict, mask: List[bool]) -> OrderedDict:
    """Apply a column mask to every sequence row in a FASTA matrix."""
    out = OrderedDict()
    for gid, chars in matrix.items():
        out[gid] = [ch for ch, k in zip(chars, mask) if k]
    return out


def read_annotation_tsv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    """Read a tab-separated annotation TSV into rows + header (robust encodings)."""
    encodings_to_try = ["utf-8", "utf-8-sig", "cp1251", "cp1252", "latin-1"]
    last_error = None
    rows: List[Dict[str, str]] = []
    header: List[str] = []

    for enc in encodings_to_try:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                header = list(reader.fieldnames or [])
                rows = list(reader)
            break
        except UnicodeDecodeError as e:
            last_error = e
            continue

    if not rows and last_error is not None:
        raise UnicodeError(
            f"Failed to decode annotation file {path}. Last error: {last_error}"
        )

    return rows, header


def write_annotation_tsv(
    path: Path, rows: List[Dict[str, str]], header: List[str]
) -> None:
    """Write annotation TSV in a fixed header order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(
            out, fieldnames=header, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_matrix_tsv(
    path: Path, genomes: List[str], positions: List[str], data_cols: List[List[str]]
) -> None:
    """Write a matrix TSV: Genome + positions header, one row per genome."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, delimiter="\t", lineterminator="\n")
        writer.writerow(["Genome"] + positions)
        for i, gid in enumerate(genomes):
            row = [gid] + [col[i] for col in data_cols]
            writer.writerow(row)


def write_matrix_tsv_rows(
    path: Path,
    positions: List[str],
    rows: "OrderedDict[str, List[str]]",
) -> None:
    """Write an already row-oriented character matrix without transposing it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, delimiter="\t", lineterminator="\n")
        writer.writerow(["Genome"] + positions)
        for genome, values in rows.items():
            writer.writerow([genome] + list(values))


# ──────────────────────────────────────────────────────────────
# DataLoader
# ──────────────────────────────────────────────────────────────


class DataLoader:
    """Build a clean binary feature matrix from per-sample VCFs."""

    def _log_stage1_reconciliation(
        self,
        *,
        n_samples: int,
        candidate_sites: int,
        kept_sites_n: int,
        df_shape: tuple,
        out_root: Optional[Path],
        matrices_final_markers: Optional[int] = None,
    ) -> None:
        """
        Reconcile counts across:
        - candidate cohort sites
        - presence-filtered sites
        - downstream matrix features (after preprocessing)
        - curated matrices/* outputs
        """
        downstream_features = int(df_shape[1])
        removed_features = max(0, kept_sites_n - downstream_features)

        logger.info(
            "DataLoader: Stage 1 reconciliation\n"
            "  samples=%d\n"
            "  candidate sites=%d\n"
            "  kept sites after sample presence filter=%d\n"
            "  downstream features after preprocess=%d (removed=%d invariant features)\n"
            "  matrices curated markers=%s\n"
            "  artifacts out=%s",
            n_samples,
            candidate_sites,
            kept_sites_n,
            downstream_features,
            removed_features,
            str(matrices_final_markers)
            if matrices_final_markers is not None
            else "n/a",
            str(out_root) if out_root is not None else "n/a",
        )

    def __init__(self, config=None, n_jobs: Optional[int] = None):
        self.config = config
        self.n_jobs = n_jobs

        # Shared VCF QC / callability (training == query semantics)
        self.vcf_qc = (
            VcfQCConfig.from_config(config) if config is not None else VcfQCConfig()
        )
        if config is None:
            # Allow explicit constructor defaults without config object.
            self.vcf_qc = VcfQCConfig()

        # INFO-level site QC thresholds (mirrored from vcf_qc for logging/compat)
        self.qual_threshold = float(self.vcf_qc.qual_threshold)
        self.dp_threshold = int(self.vcf_qc.min_dp)
        self.gq_threshold = int(self.vcf_qc.min_gq)
        self.mq_threshold = float(self.vcf_qc.mq_threshold)
        self.mq0f_threshold = float(self.vcf_qc.mq0f_threshold)
        self.assume_absent_variant_is_reference = bool(
            self.vcf_qc.assume_absent_variant_is_reference
        )

        # Cohort-level presence threshold (minimum number of samples that must contain the SNP)
        self.min_sample_presence = (
            int(getattr(config, "min_sample_presence", 3)) if config else 3
        )

        # Binary baseline strategy: 'Y' (reference) or 'N' (cohort mode)
        self.ancestral_allele = (
            str(getattr(config, "ancestral_allele", "Y")) if config else "Y"
        )

        # Variant scope
        self.biallelic_only = bool(self.vcf_qc.biallelic_only)

        # DataLoader lightweight preprocessing (kept separate from statistical validation)
        self.remove_invariant = (
            bool(getattr(config, "remove_invariant", True)) if config else True
        )
        self.min_minor_count = (
            int(getattr(config, "min_minor_count", 0)) if config else 0
        )

        # Output naming (kept consistent across artifacts)
        self.generic_name = (
            str(getattr(config, "generic_name", "matrix")) if config else "matrix"
        )

        # Fasta2matrices-style filter knobs for matrices/* outputs
        self.matrices_min_count = (
            int(getattr(config, "matrices_min_count", 3)) if config else 3
        )
        self.matrices_repeat_number = (
            int(getattr(config, "matrices_repeat_number", 1)) if config else 1
        )
        self.matrices_type = (
            str(getattr(config, "matrices_type", "all")) if config else "all"
        )
        self.matrices_fix = str(getattr(config, "matrices_fix", "")) if config else ""

        # Optional: shrink column names in returned DataFrame (not affecting artifacts)
        self.use_integer_variant_ids = (
            bool(getattr(config, "use_integer_variant_ids", False)) if config else False
        )

        if self.assume_absent_variant_is_reference:
            warn_legacy_absence_assumed_reference()

    def load_genomic_matrix(
        self,
        file_path: str,
        output_dir: Optional[str] = None,
        ref_fasta: Optional[str] = None,
        label_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load genomic features from a VCF directory or a prebuilt matrix file."""
        _ = label_column
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Genomic input not found: {path}")

        logger.info("DataLoader input resolved | path=%s", str(path))

        if path.is_dir():
            if log_flow_step is not None:
                log_flow_step(
                    logger,
                    step="Input interpretation — VCF directory",
                    happened="Detected a directory of per-sample VCF/VCF.GZ files and will construct a sample-by-feature binary variant matrix.",
                    reason="VCF input must be quality-controlled, merged across samples, encoded against a baseline allele, and converted into the matrix representation used by downstream statistical filtering.",
                    threshold=(
                        f"QUAL>={self.qual_threshold}; DP>={self.dp_threshold}; "
                        f"MQ>={self.mq_threshold}; MQ0F<={self.mq0f_threshold}; "
                        f"min_sample_presence={self.min_sample_presence}"
                    ),
                    status="vcf_directory_mode",
                )
            else:
                logger.info("DataLoader: mode=vcf_directory")
            return self._load_vcf_directory(
                path, output_dir=output_dir, ref_path=ref_fasta
            )

        suffix = "".join(path.suffixes).lower()
        if suffix.endswith((".csv", ".tsv", ".tab")):
            if log_flow_step is not None:
                log_flow_step(
                    logger,
                    step="Input interpretation — prebuilt matrix",
                    happened="Detected an existing sample-by-feature matrix and will preserve it as the starting feature representation.",
                    reason="A prebuilt matrix is assumed to have already gone through upstream variant calling or matrix construction, so DataLoader only performs safe loading and reports the matrix shape before supervised alignment and statistical filtering.",
                    status="matrix_mode",
                )
            else:
                logger.info("DataLoader: mode=prebuilt_matrix")
            return self._load_matrix_file(path)

        raise ValueError(
            "This DataLoader expects either a directory of per-sample VCF/VCF.GZ files "
            "or a prebuilt matrix (.csv/.tsv). "
            f"Got: {path}"
        )

    def load_metadata(
        self, meta_path: str, output_dir: Optional[str] = None
    ) -> pd.DataFrame:
        """Load a metadata table and index it by sample identifier."""
        path = Path(meta_path)
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")

        sep = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
        df = pd.read_csv(path, sep=sep)

        if df.shape[1] < 2:
            raise ValueError(
                f"Metadata file looks invalid (needs ≥2 columns): {path} (shape={df.shape})"
            )

        idx_col = "Sample" if "Sample" in df.columns else df.columns[0]
        df[idx_col] = df[idx_col].astype(str)
        df = df.set_index(idx_col, drop=True)
        df.index.name = "Sample"

        if output_dir:
            outdir = Path(output_dir)
            outdir.mkdir(parents=True, exist_ok=True)
            df.to_csv(outdir / "metadata.normalized.csv")

        return df

    def load_known_markers(
        self, known_markers_path: str, output_dir: Optional[str] = None
    ) -> List[str]:
        """Load a list of marker identifiers from a .txt or .csv/.tsv file."""
        path = Path(known_markers_path)
        if not path.exists():
            logger.error("Known markers file not found: %s", path)
            raise FileNotFoundError(f"Known markers file not found: {path}")

        logger.info("Loading known markers from: %s", path)

        suffix = "".join(path.suffixes).lower()
        markers: List[str] = []

        if suffix.endswith(".txt"):
            logger.info("Detected plain text file (.txt) – reading line by line")
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    markers.append(line)

        elif suffix.endswith((".csv", ".tsv", ".tab")):
            sep = "\t" if suffix.endswith((".tsv", ".tab")) else ","
            logger.info(
                "Detected tabular file (%s) – reading from first column or 'marker' column",
                suffix,
            )
            df = pd.read_csv(path, sep=sep)
            col = "marker" if "marker" in df.columns else df.columns[0]
            markers = [str(x).strip() for x in df[col].tolist() if str(x).strip()]

        else:
            logger.error("Unsupported known markers file format: %s", suffix)
            raise ValueError(f"Unsupported known markers file type: {path}")

        # Deduplicate while preserving order
        seen = set()
        uniq_markers: List[str] = []
        for m in markers:
            if m not in seen:
                uniq_markers.append(m)
                seen.add(m)

        removed = len(markers) - len(uniq_markers)
        if removed > 0:
            logger.info(
                "Removed %d duplicate markers (total unique: %d)",
                removed,
                len(uniq_markers),
            )
        else:
            logger.info(
                "Loaded %d unique markers (no duplicates found)", len(uniq_markers)
            )

        if output_dir:
            outdir = Path(output_dir)
            outdir.mkdir(parents=True, exist_ok=True)
            out_path = outdir / "known_markers.normalized.txt"
            with open(out_path, "w", encoding="utf-8") as f:
                for m in uniq_markers:
                    f.write(m + "\n")
            logger.info("Saved normalized known markers list to: %s", out_path)

        return uniq_markers

    # ──────────────────────────────────────────────────────────
    # VCF directory → allele + binary matrices → returned DataFrame
    # ──────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────
    # Helper to process one VCF file (used in parallel)
    # ──────────────────────────────────────────────────────────────
    def _load_vcf_directory(
        self, vcf_dir: Path, output_dir: Optional[str], ref_path: Optional[str]
    ) -> pd.DataFrame:
        """Scan a directory of per-sample VCFs and build allele/binary matrices."""
        load_started_at = time.perf_counter()
        stage_timings: Dict[str, float] = {}
        vcfs = sorted(
            [p for p in vcf_dir.iterdir() if p.name.endswith((".vcf", ".vcf.gz"))]
        )
        if not vcfs:
            raise ValueError(f"No .vcf/.vcf.gz files found in: {vcf_dir}")

        # 1) Discovery (what is about to be parsed?)
        logger.info(
            "DataLoader: discovered %d VCF(s) in %s (biallelic_only=%s)",
            len(vcfs),
            str(vcf_dir),
            self.biallelic_only,
        )

        # 2) Explicitly describe what parsing does + what gets kept
        logger.info(
            "DataLoader: per-sample parsing plan\n"
            "  Each VCF will be scanned record-by-record using iter_sample_calls().\n"
            "  Retained calls: SNP-like, biallelic (if enabled), passing QC thresholds.\n"
            "  Per-sample output: an in-memory dict mapping site→(REF, CALLED) for retained sites.",
        )

        # 3) Record-level QC configuration (applied during parsing)
        logger.info(
            "DataLoader: record-level QC thresholds (shared vcf_call_semantics)\n"
            "  QUAL>=%.1f | DP>=%d | GQ>=%d | INFO/MQ>=%.1f | INFO/MQ0F<=%.3f | "
            "respect_filter=%s | assume_absent_as_ref=%s",
            float(self.qual_threshold),
            int(self.dp_threshold),
            int(self.gq_threshold),
            float(self.mq_threshold),
            float(self.mq0f_threshold),
            bool(self.vcf_qc.respect_filter),
            bool(self.assume_absent_variant_is_reference),
        )

        # 4) Cohort / matrix-level configuration (applied AFTER parsing/merge)
        logger.info(
            "DataLoader: cohort + matrix settings (applied after parsing)\n"
            "  min_sample_presence=%d | baseline=%s | min_minor_count=%d",
            int(self.min_sample_presence),
            ("REF" if self.ancestral_allele.upper() == "Y" else "MODE"),
            int(self.min_minor_count),
        )

        # Helper for parallel processing
        def process_vcf(vcf_path: Path) -> tuple[str, CallSet]:
            sample = vcf_path.name
            if sample.endswith(".vcf.gz"):
                sample = sample[:-7]
            elif sample.endswith(".vcf"):
                sample = sample[:-4]
            elif sample.endswith(".g.vcf.gz"):
                sample = sample[: -len(".g.vcf.gz")]
            elif sample.endswith(".g.vcf"):
                sample = sample[: -len(".g.vcf")]

            calls = iter_sample_calls(
                vcf_path,
                qual_thresh=self.qual_threshold,
                dp_thresh=self.dp_threshold,
                mq_thresh=self.mq_threshold,
                mq0f_thresh=self.mq0f_threshold,
                biallelic_only=self.biallelic_only,
                gq_thresh=self.gq_threshold,
                qc=self.vcf_qc,
            )
            # Prefer sample name from VCF header when present.
            if getattr(calls, "sample_name", None):
                sample = str(calls.sample_name)

            logger.debug(
                "DataLoader: per-sample parse result\n"
                "  sample=%s\n"
                "  parsed_records=%d\n"
                "  storage=CallSet (site→AlleleCall + optional gVCF ref blocks)",
                sample,
                len(calls),
            )
            return sample, calls

        # 5) Execute parsing
        n_jobs = getattr(self, "n_jobs", -1)
        logger.info(
            "DataLoader: starting parallel per-sample parsing (n_jobs=%s)",
            _fmt_n_jobs(n_jobs),
        )

        parsing_started_at = time.perf_counter()
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_vcf)(vcf)
            for vcf in progress_iter(
                vcfs, desc="Parsing VCF samples", unit="sample", leave=False
            )
        )
        stage_timings["vcf_parsing"] = time.perf_counter() - parsing_started_at

        # 6) Parsing summary + clarify where it is stored
        audit_started_at = time.perf_counter()
        n_samples = len(results)
        total_records = sum(len(calls) for _, calls in results)
        mean_records = total_records / max(1, n_samples)
        parsed_state_counts: Counter[str] = Counter()
        parsed_qc_reasons: Counter[str] = Counter()
        for _, calls in results:
            for allele_call in calls.by_pos.values():
                parsed_state_counts[allele_call.state.value] += 1
                parsed_qc_reasons.update(allele_call.qc_reasons)

        state_summary = ", ".join(
            f"{state}={count}"
            for state, count in sorted(parsed_state_counts.items())
        ) or "none"
        qc_reason_summary = ", ".join(
            f"{reason}={count}"
            for reason, count in parsed_qc_reasons.most_common(8)
        ) or "none"
        stage_timings["parse_audit"] = time.perf_counter() - audit_started_at

        logger.info(
            "DataLoader: parsing complete\n"
            "  samples=%d\n"
            "  total_parsed_records=%d\n"
            "  mean_parsed_records_per_sample=%.2f\n"
            "  call_states=%s\n"
            "  top_qc_reasons=%s\n"
            "  storage=results list of (sample_id, calls_dict) in memory",
            n_samples,
            total_records,
            mean_records,
            state_summary,
            qc_reason_summary,
        )
        logger.info(
            "DataLoader timing: VCF parsing | seconds=%.3f | samples_per_second=%.2f | "
            "records_per_second=%.2f",
            stage_timings["vcf_parsing"],
            n_samples / max(stage_timings["vcf_parsing"], 1e-12),
            total_records / max(stage_timings["vcf_parsing"], 1e-12),
        )

        logger.info(
            "DataLoader: cohort merge starting\n"
            "  The per-sample call dictionaries will now be aggregated into:\n"
            "    (1) per_sample_calls[sample] → calls_dict\n"
            "    (2) site_counts[site] → carrier_count for allele-specific events\n"
            "  Alternate alleles at the same genomic position will be retained as separate features.",
        )

        # 7) Merge parsed results into cohort-wide site universe
        per_sample_calls: Dict[str, CallSet] = {}
        site_counts: Dict[Tuple[str, int, str, str], int] = {}

        carrier_events = 0
        sample_ids = [sample for sample, _ in results]
        assert_unique_sample_ids(sample_ids, context="VCF training directory")

        merge_started_at = time.perf_counter()
        for sample, calls in results:
            per_sample_calls[sample] = calls
            for (chrom, pos), allele_call in calls.items():
                if not isinstance(allele_call, AlleleCall):
                    continue
                if allele_call.state != CallState.CALLED_ALTERNATE:
                    continue
                called = (allele_call.called_allele or "").upper()
                ref = (allele_call.ref or "").upper()
                if (
                    not called
                    or not ref
                    or called == ref
                    or len(called) != 1
                    or len(ref) != 1
                ):
                    continue
                allele_key = (chrom, pos, ref, called)
                site_counts[allele_key] = site_counts.get(allele_key, 0) + 1
                carrier_events += 1
        stage_timings["cohort_merge"] = time.perf_counter() - merge_started_at

        logger.info(
            "DataLoader: cohort merge complete\n"
            "  per_sample_calls now holds %d per-genome call maps.\n"
            "  site_counts now defines the cohort-wide polymorphic site universe (pre-filter).",
            len(per_sample_calls),
        )

        # ---- Cohort universe established (pre-filter boundary) ----
        candidate_sites = len(site_counts)
        logger.info(
            "DataLoader: cohort variant landscape (pre-filter)\n"
            "  The cohort comprises %d genomes.\n"
            "  A total of %d unique allele-specific polymorphic features were observed.\n"
            "  These correspond to %d total mutation occurrences across genomes.\n"
            "  Alternate alleles at the same genomic position were retained as separate features.",
            len(per_sample_calls),
            candidate_sites,
            carrier_events,
        )

        # Clarify artifact timing + location (what “snapshot” means)
        if output_dir:
            out = Path(output_dir)
            logger.info(
                "DataLoader: cohort artifacts\n"
                "  Output will be written to: %s\n"
                "  Writing occurs after presence filtering and encoding (matrices + FASTA + annotation tables).",
                str(out),
            )

        # 8) Cohort-level filtering: min sample presence
        kept_sites: Dict[Tuple[str, int, str, str], int] = {
            key: cnt
            for key, cnt in site_counts.items()
            if cnt >= self.min_sample_presence
        }

        kept_n = len(kept_sites)
        retention_rate = kept_n / max(1, candidate_sites)

        if log_filter_step is not None:
            log_filter_step(
                logger,
                filter_name="cohort sample presence",
                happened="Retained only polymorphic features observed in at least the configured number of samples.",
                reason="Features seen in too few samples provide weak cohort-level support and can make downstream association testing unstable.",
                before_samples=int(len(per_sample_calls)),
                before_features=int(candidate_sites),
                after_samples=int(len(per_sample_calls)),
                after_features=int(kept_n),
                threshold=f"min_sample_presence >= {self.min_sample_presence}",
                status="applied",
            )
        else:
            logger.info(
                "DataLoader: cohort presence filtering\n"
                "  A minimum of %d genomes per site was required.\n"
                "  %d of %d polymorphic sites were retained (%.2f%% retained).\n"
                "  Sites failing this threshold were removed from the cohort feature space.",
                int(self.min_sample_presence),
                kept_n,
                candidate_sites,
                retention_rate * 100,
            )

        if not kept_sites:
            diagnostic = (
                f"call_states=[{state_summary}]; "
                f"top_qc_reasons=[{qc_reason_summary}]"
            )
            if candidate_sites == 0:
                raise ValueError(
                    "No callable alternate SNPs remained after record-level QC. "
                    f"{diagnostic}. Check whether the configured QC fields exist "
                    "in the VCF FORMAT/INFO columns (for example, GQ_missing means "
                    "min_gq_per_sample is active for VCFs without GQ)."
                )
            raise ValueError(
                "No polymorphic sites retained after QC + min-sample-presence filter. "
                f"Observed {candidate_sites} callable allele-specific SNPs, but none "
                f"occurred in at least {self.min_sample_presence} samples. {diagnostic}."
            )

        # 9) Sort sites deterministically and build allele strings
        ordered_keys = sorted(kept_sites.keys(), key=lambda x: (x[0], x[1], x[2], x[3]))
        ref_bases = [k[2] for k in ordered_keys]

        samples_sorted = sorted(per_sample_calls.keys())
        per_pos_counts: List[Counter[str]] = [Counter() for _ in ordered_keys]
        sample_allele_strings: Dict[str, str] = {}
        # Parallel binary matrix cells: 0 / 1 / NaN (missing is never ordinary 0)
        sample_binary_values: Dict[str, List[float]] = {}
        call_state_audit: Dict[str, Counter] = {}

        ref_line = "".join(ref_bases)
        logger.debug(
            "DataLoader: building allele strings for %d sample(s)",
            len(per_sample_calls),
        )

        matrix_started_at = time.perf_counter()
        for sample in samples_sorted:
            calls = per_sample_calls[sample]
            alleles: List[str] = []
            bin_vals: List[float] = []
            state_counts: Counter = Counter()
            for j, (chrom, pos, ref, alt) in enumerate(ordered_keys):
                resolved = resolve_feature_call(
                    chrom=chrom,
                    pos=pos,
                    feature_ref=ref,
                    feature_alt=alt,
                    calls=calls,
                    qc=self.vcf_qc,
                )
                state_counts[resolved.state.value] += 1
                if resolved.assumed_reference:
                    state_counts["assumed_reference"] += 1
                encoded, _allele_label = encode_binary_for_feature(
                    resolved,
                    feature_ref=ref,
                    feature_alt=alt,
                    baseline_allele=ref
                    if self.ancestral_allele.upper() == "Y"
                    else None,
                )
                if (
                    resolved.state == CallState.CALLED_ALTERNATE
                    and (resolved.called_allele or "").upper() == alt
                ):
                    base = alt
                elif resolved.state == CallState.CALLED_REFERENCE:
                    base = ref
                elif np.isnan(encoded):
                    base = "N"  # non-callable; not REF evidence
                else:
                    base = alt if encoded == 1.0 else ref
                alleles.append(base)
                bin_vals.append(float(encoded))
                if base in {"A", "C", "G", "T"}:
                    per_pos_counts[j][base] += 1
            sample_allele_strings[sample] = "".join(alleles)
            sample_binary_values[sample] = bin_vals
            call_state_audit[sample] = state_counts

        # 10) Baseline selection
        if self.ancestral_allele.upper() == "Y":
            baseline = list(ref_line)
            baseline_strategy = "REF"
        else:
            baseline = [
                per_pos_counts[j].most_common(1)[0][0]
                if per_pos_counts[j]
                else ref_bases[j]
                for j in range(len(ordered_keys))
            ]
            baseline_strategy = "MODE"

        baseline_diff_from_ref = sum(
            1 for i, ch in enumerate(baseline) if ch != ref_line[i]
        )

        if log_flow_step is not None:
            baseline_reason = (
                "Reference-baseline encoding was requested, so 0 represents the reference allele and 1 represents a non-reference allele."
                if baseline_strategy == "REF"
                else "Cohort-mode baseline encoding was requested, so 0 represents the cohort-majority allele and 1 represents the minority/non-baseline allele."
            )
            log_flow_step(
                logger,
                step="Preprocessing checkpoint — binary baseline encoding",
                happened="Converted retained polymorphic features into the binary representation required by downstream statistical filtering and model training.",
                reason=baseline_reason,
                before_samples=int(len(per_sample_calls)),
                before_features=int(len(ordered_keys)),
                after_samples=int(len(per_sample_calls)),
                after_features=int(len(ordered_keys)),
                threshold=f"baseline_strategy={baseline_strategy}",
                status=(
                    "reference_baseline"
                    if baseline_strategy == "REF"
                    else f"cohort_mode_baseline; mode_differs_from_ref_at={baseline_diff_from_ref}"
                ),
            )
        elif baseline_strategy == "REF":
            logger.info(
                "DataLoader: baseline encoding\n"
                "  The reference allele was used as the baseline.\n"
                "  Encoding definition: 0 indicates the reference allele; 1 indicates a non-reference allele.\n"
                "  A total of %d polymorphic sites were encoded.",
                len(ordered_keys),
            )
        else:
            logger.info(
                "DataLoader: baseline encoding\n"
                "  The most frequent allele across the cohort was used as the baseline.\n"
                "  Encoding definition: 0 indicates the cohort-majority allele; 1 indicates a minority allele.\n"
                "  A total of %d polymorphic sites were encoded.\n"
                "  At %d sites, the cohort-majority allele differed from the reference allele.",
                len(ordered_keys),
                baseline_diff_from_ref,
            )

        # 11) Binary encoding (baseline → 0/1/NaN orientation)
        # Re-encode relative to chosen baseline when cohort MODE is requested.
        if self.ancestral_allele.upper() != "Y":
            for sample in samples_sorted:
                seq = sample_allele_strings[sample]
                re_bin: List[float] = []
                for i, ch in enumerate(seq):
                    if ch == "N" or ch not in {"A", "C", "G", "T"}:
                        re_bin.append(float("nan"))
                    else:
                        re_bin.append(0.0 if ch == baseline[i] else 1.0)
                sample_binary_values[sample] = re_bin

        ref_binary = "".join(
            "0" if ref_line[i] == baseline[i] else "1" for i in range(len(ref_line))
        )
        sample_binary_strings = {
            s: "".join(
                "?"
                if (
                    i >= len(vals) or (isinstance(vals[i], float) and np.isnan(vals[i]))
                )
                else ("1" if vals[i] == 1.0 else "0")
                for i in range(len(ref_line))
            )
            for s, vals in sample_binary_values.items()
        }

        expected_len = len(ref_line)
        for s, binseq in sample_binary_strings.items():
            if len(binseq) != expected_len:
                raise ValueError(
                    f"Binary encoding length mismatch for sample {s}: expected {expected_len}, got {len(binseq)}"
                )

        # 12) Feature IDs (variant-centric identifiers)
        variant_ids = [f"{c}:{p}:{r}:{a}" for (c, p, r, a) in ordered_keys]
        # 13) Final matrix assembly (sample-centric orientation)
        # Missing / unresolved / absent-without-callability → NaN (not ordinary 0).
        data_bin = [sample_binary_values[s] for s in samples_sorted]

        df = pd.DataFrame(
            data_bin,
            index=samples_sorted,
            columns=variant_ids,
            dtype=float,
        )
        df.index.name = "Sample"
        stage_timings["matrix_construction"] = (
            time.perf_counter() - matrix_started_at
        )
        matrix_cells = int(df.shape[0] * df.shape[1])
        logger.info(
            "DataLoader timing: matrix construction | seconds=%.3f | samples=%d | "
            "features=%d | cells=%d | cells_per_second=%.2f",
            stage_timings["matrix_construction"],
            int(df.shape[0]),
            int(df.shape[1]),
            matrix_cells,
            matrix_cells / max(stage_timings["matrix_construction"], 1e-12),
        )

        # 14) Raw matrix stats (post-encoding, pre-preprocessing)
        n_samples, raw_feature_count = df.shape
        values = df.to_numpy(dtype=float)
        total_ones = int(np.nansum(values == 1.0))
        total_zeros = int(np.nansum(values == 0.0))
        total_missing = int(np.isnan(values).sum())
        total_cells = n_samples * raw_feature_count

        matrix_density = total_ones / max(1, total_cells)
        mean_ones_per_feature = total_ones / max(1, raw_feature_count)
        mean_ones_per_sample = total_ones / max(1, n_samples)
        missing_fraction = total_missing / max(1, total_cells)

        # Aggregate call-state audit
        global_state: Counter[str] = Counter()
        for c in call_state_audit.values():
            global_state.update(c)

        logger.info(
            "DataLoader: raw binary matrix (post-encoding, pre-preprocessing)\n"
            "  The matrix contains %d genomes (rows) and %d polymorphic sites (columns).\n"
            "  This corresponds to %d total genotype entries.\n"
            "  Encoded as 1 (non-baseline)=%d | 0 (callable baseline)=%d | NaN (non-callable)=%d.\n"
            "  Matrix density (fraction of 1s) = %.6f | missing_fraction=%.6f.\n"
            "  Mean carrier count per site = %.2f genomes.\n"
            "  Mean variant burden per genome = %.2f sites.\n"
            "  Call-state tallies (sample×feature): %s",
            n_samples,
            raw_feature_count,
            total_cells,
            total_ones,
            total_zeros,
            total_missing,
            matrix_density,
            missing_fraction,
            mean_ones_per_feature,
            mean_ones_per_sample,
            dict(global_state),
        )

        # 15) Variant frequency summary (carriers per feature; NaN skipped by sum)
        carrier_counts = df.eq(1.0).sum(axis=0).astype(int)
        singleton_sites = int((carrier_counts == 1).sum())
        doubleton_sites = int((carrier_counts == 2).sum())

        logger.info(
            "DataLoader: variant frequency summary (carriers per feature)\n"
            "  singleton_sites=%d (carriers==1)\n"
            "  doubleton_sites=%d (carriers==2)",
            singleton_sites,
            doubleton_sites,
        )

        # ─────────────────────────────────────────────
        # Preprocessing
        # ─────────────────────────────────────────────
        preprocessing_started_at = time.perf_counter()
        df, prep_stats = self._preprocess_binary_matrix(df)
        stage_timings["matrix_preprocessing"] = (
            time.perf_counter() - preprocessing_started_at
        )
        logger.info(
            "DataLoader: matrix preprocessing (post-encoding)\n"
            "  The preprocessing stage applied invariant-site removal and minor allele count filtering.\n"
            "  Invariant removal enabled: %s.\n"
            "  Rationale: remove markers with no variation across samples\n"
            "  Purpose: eliminate non-informative genomic features before downstream analysis\n"
            "  Minimum minor allele count threshold: %d.\n"
            "  Rationale: remove markers where minority state appears\n"
            "  Purpose: avoids instability from extremely rare variants in small cohorts\n"
            "  The matrix contained %d features prior to preprocessing.\n"
            "  %d invariant features were removed.\n"
            "  %d features were removed due to insufficient minor allele count.\n"
            "  %d features remain after preprocessing.",
            self.remove_invariant,
            self.min_minor_count,
            prep_stats["features_before"],
            prep_stats["removed_invariant"],
            prep_stats["removed_low_minor_count"],
            prep_stats["features_after"],
        )
        retained_variant_ids_for_artifacts = df.columns.astype(str).tolist()
        artifact_matrix_df = df

        lookup = None
        if self.use_integer_variant_ids:
            df, lookup = self._convert_to_integer_variant_ids(df)
            logger.info("DataLoader: compacted variant IDs and created lookup table.")

        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)

            if self.config is not None:
                cfg_path = out / "dataloader_config.snapshot.json"
                payload = (
                    asdict(self.config)
                    if hasattr(self.config, "__dataclass_fields__")
                    else vars(self.config)
                )
                with open(cfg_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                logger.info("DataLoader: wrote config snapshot %s", str(cfg_path))

            (
                artifact_kept_sites,
                artifact_ordered_keys,
                artifact_positions_1based,
                artifact_ref_line,
                artifact_sample_allele_strings,
                artifact_ref_binary,
                artifact_sample_binary_strings,
            ) = self._filter_artifact_inputs_to_retained_features(
                kept_sites=kept_sites,
                ordered_keys=ordered_keys,
                ref_line=ref_line,
                sample_allele_strings=sample_allele_strings,
                ref_binary=ref_binary,
                sample_binary_strings=sample_binary_strings,
                retained_variant_ids=retained_variant_ids_for_artifacts,
            )

            logger.info(
                "DataLoader: writing artifacts\n" "  output_dir=%s",
                output_dir,
            )

            artifacts_started_at = time.perf_counter()
            matrices_final_markers = self._write_all_artifacts(
                out_root=out,
                kept_sites=artifact_kept_sites,
                ordered_keys=artifact_ordered_keys,
                positions_1based=artifact_positions_1based,
                ref_line=artifact_ref_line,
                sample_allele_strings=artifact_sample_allele_strings,
                ref_binary=artifact_ref_binary,
                sample_binary_strings=artifact_sample_binary_strings,
                matrix_df=artifact_matrix_df,
                ref_path=Path(ref_path) if ref_path else None,
                integer_id_lookup=lookup,
            )
            stage_timings["artifact_processing"] = (
                time.perf_counter() - artifacts_started_at
            )

            logger.info(
                "DataLoader: artifact writing complete | matrices_final_markers=%s",
                str(matrices_final_markers)
                if matrices_final_markers is not None
                else "n/a",
            )

        else:
            stage_timings["artifact_processing"] = 0.0
            logger.info(
                "DataLoader: output_dir not provided; skipping artifact writing and returning in-memory matrix."
            )

        total_seconds = time.perf_counter() - load_started_at
        logger.info(
            "DataLoader timing summary | vcf_parsing=%.3fs | parse_audit=%.3fs | "
            "cohort_merge=%.3fs | matrix_construction=%.3fs | "
            "matrix_preprocessing=%.3fs | artifact_processing=%.3fs | total=%.3fs",
            stage_timings.get("vcf_parsing", 0.0),
            stage_timings.get("parse_audit", 0.0),
            stage_timings.get("cohort_merge", 0.0),
            stage_timings.get("matrix_construction", 0.0),
            stage_timings.get("matrix_preprocessing", 0.0),
            stage_timings.get("artifact_processing", 0.0),
            total_seconds,
        )
        return df

    def _preprocess_binary_matrix(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Apply lightweight, non-statistical preprocessing to the binary matrix.

        Steps:
        1) Remove invariant features (all 0 or all 1)
        2) Enforce minimum minor allele count per site

        Returns:
        df_filtered, stats_dict
        """
        if df.empty:
            return df, {
                "features_before": 0,
                "removed_invariant": 0,
                "removed_low_minor_count": 0,
                "features_after": 0,
            }

        features_before = df.shape[1]
        removed_invariant = 0
        removed_low_minor_count = 0

        # ─────────────────────────────────────────────
        # 1) Remove invariant features
        # ─────────────────────────────────────────────
        if self.remove_invariant:
            before_filter_features = int(df.shape[1])
            before_filter_samples = int(df.shape[0])
            nunique = df.nunique(axis=0, dropna=False)
            invariant_mask = nunique <= 1
            removed_invariant = int(invariant_mask.sum())

            df = df.loc[:, ~invariant_mask]

            if log_filter_step is not None:
                log_filter_step(
                    logger,
                    filter_name="invariant genomic features",
                    happened="Removed features with only one observed state across the cohort.",
                    reason="Invariant features carry no discriminatory signal for AMR phenotypes or lineage placement, and keeping them only increases memory/runtime without improving robust inference.",
                    before_samples=before_filter_samples,
                    before_features=before_filter_features,
                    after_samples=int(df.shape[0]),
                    after_features=int(df.shape[1]),
                    threshold="unique_states_per_feature > 1",
                    status="applied",
                )
            else:
                logger.info(
                    "Invariant-feature filter applied | retained_features=%d / %d",
                    int(df.shape[1]),
                    before_filter_features,
                )

            if df.empty:
                raise ValueError(
                    "All polymorphic sites were removed during invariant filtering. "
                    "Check input data or relax remove_invariant setting."
                )
        else:
            if log_filter_step is not None:
                log_filter_step(
                    logger,
                    filter_name="invariant genomic features",
                    happened="Skipped invariant-feature removal because remove_invariant=False.",
                    reason="The user configuration requested that all encoded features remain available for downstream stages.",
                    before_samples=int(df.shape[0]),
                    before_features=int(df.shape[1]),
                    after_samples=int(df.shape[0]),
                    after_features=int(df.shape[1]),
                    threshold="disabled",
                    status="skipped",
                )

        # ─────────────────────────────────────────────
        # 2) Enforce minimum minor allele count
        # ─────────────────────────────────────────────
        if self.min_minor_count > 0 and not df.empty:
            arr = df.to_numpy(copy=False)
            count_1 = (arr == 1).sum(axis=0)
            count_0 = (arr == 0).sum(axis=0)
            keep_mask = np.minimum(count_0, count_1) >= self.min_minor_count

            before_minor = int(df.shape[1])
            before_minor_samples = int(df.shape[0])
            df = df.loc[:, np.asarray(keep_mask, dtype=bool)]
            removed_low_minor_count = before_minor - df.shape[1]

            if log_filter_step is not None:
                log_filter_step(
                    logger,
                    filter_name="low-support minor state",
                    happened="Removed features whose minority state did not meet the configured cohort-support threshold.",
                    reason="Extremely rare binary states are unstable in small cohorts and can inflate apparent associations before statistically defensible filtering.",
                    before_samples=before_minor_samples,
                    before_features=before_minor,
                    after_samples=int(df.shape[0]),
                    after_features=int(df.shape[1]),
                    threshold=f"min_minor_count >= {self.min_minor_count}",
                    status="applied",
                )
            else:
                logger.info(
                    "Minor-count filter applied | retained_features=%d / %d",
                    int(df.shape[1]),
                    before_minor,
                )

            if df.empty:
                raise ValueError(
                    "All sites removed by minor allele count filter. "
                    f"Try lowering min_minor_count (current: {self.min_minor_count}) "
                    "or verify binary encoding."
                )
        elif log_filter_step is not None:
            log_filter_step(
                logger,
                filter_name="low-support minor state",
                happened="Skipped minor-count filtering because min_minor_count is not enabled.",
                reason="No cohort-support threshold was requested at this lightweight preprocessing gate; downstream statistical filtering still controls feature retention.",
                before_samples=int(df.shape[0]),
                before_features=int(df.shape[1]),
                after_samples=int(df.shape[0]),
                after_features=int(df.shape[1]),
                threshold=f"min_minor_count={self.min_minor_count}",
                status="skipped",
            )

        features_after = df.shape[1]

        stats = {
            "features_before": features_before,
            "removed_invariant": removed_invariant,
            "removed_low_minor_count": removed_low_minor_count,
            "features_after": features_after,
        }

        return df, stats

    def _filter_artifact_inputs_to_retained_features(
        self,
        *,
        kept_sites: Dict[Tuple[str, int, str, str], int],
        ordered_keys: List[Tuple[str, int, str, str]],
        ref_line: str,
        sample_allele_strings: Dict[str, str],
        ref_binary: str,
        sample_binary_strings: Dict[str, str],
        retained_variant_ids: List[str],
    ) -> Tuple[
        Dict[Tuple[str, int, str, str], int],
        List[Tuple[str, int, str, str]],
        List[int],
        str,
        Dict[str, str],
        str,
        Dict[str, str],
    ]:
        """
        Restrict artifact-writing inputs to the exact post-preprocessing feature set.

        This keeps the returned DataFrame, FASTA artifacts, annotation tables, and
        downstream matrices/* outputs synchronized after invariant and low-count
        feature removal.
        """
        retained_set = set(str(x) for x in retained_variant_ids)

        keep_indices: List[int] = []
        filtered_ordered_keys: List[Tuple[str, int, str, str]] = []
        filtered_kept_sites: Dict[Tuple[str, int, str, str], int] = {}

        for idx, key in enumerate(ordered_keys):
            chrom, pos, ref, alt = key
            feature_id = f"{chrom}:{pos}:{ref}:{alt}"

            if feature_id in retained_set:
                keep_indices.append(idx)
                filtered_ordered_keys.append(key)
                filtered_kept_sites[key] = int(kept_sites[key])

        if len(filtered_ordered_keys) != len(retained_set):
            missing_n = len(retained_set) - len(filtered_ordered_keys)
            raise ValueError(
                "Artifact synchronization failed: post-preprocessing feature set "
                f"does not match ordered VCF-derived feature keys. Missing={missing_n}. "
                "Check feature ID construction before writing artifacts."
            )

        def _slice_string(seq: str) -> str:
            return "".join(seq[i] for i in keep_indices)

        filtered_ref_line = _slice_string(ref_line)
        filtered_ref_binary = _slice_string(ref_binary)

        filtered_sample_alleles = {
            sample: _slice_string(seq) for sample, seq in sample_allele_strings.items()
        }

        filtered_sample_binary = {
            sample: _slice_string(seq) for sample, seq in sample_binary_strings.items()
        }

        filtered_positions_1based = [pos for _, pos, _, _ in filtered_ordered_keys]

        logger.info(
            "DataLoader: synchronized artifact inputs to post-preprocessing feature set\n"
            "  retained_features=%d\n"
            "  removed_from_artifact_inputs=%d\n"
            "  purpose=prevent preprocessed-away markers from re-entering downstream matrices",
            len(filtered_ordered_keys),
            len(ordered_keys) - len(filtered_ordered_keys),
        )

        return (
            filtered_kept_sites,
            filtered_ordered_keys,
            filtered_positions_1based,
            filtered_ref_line,
            filtered_sample_alleles,
            filtered_ref_binary,
            filtered_sample_binary,
        )

    def _convert_to_integer_variant_ids(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Replace long variant IDs with compact IDs and return a lookup."""
        lookup: Dict[str, str] = {}
        new_cols: List[str] = []
        for i, col in enumerate(df.columns):
            vid = f"v{i}"
            new_cols.append(vid)
            lookup[vid] = col
        df2 = df.copy()
        df2.columns = new_cols
        return df2, lookup

    # ──────────────────────────────────────────────────────────
    # Artifact generation
    # ──────────────────────────────────────────────────────────

    def _write_all_artifacts(
        self,
        out_root: Path,
        kept_sites: Dict[Tuple[str, int, str, str], int],
        ordered_keys: List[Tuple[str, int, str, str]],
        positions_1based: List[int],
        ref_line: str,
        sample_allele_strings: Dict[str, str],
        ref_binary: str,
        sample_binary_strings: Dict[str, str],
        matrix_df: pd.DataFrame,
        ref_path: Optional[Path],
        integer_id_lookup: Optional[Dict[str, str]],
    ) -> Optional[int]:
        """Write vcf_counts/*, fasta/*, and matrices/* outputs (with detailed timing logs)."""
        def _fmt_size(p: Path) -> str:
            try:
                b = p.stat().st_size
                if b < 1024:
                    return f"{b} B"
                if b < 1024**2:
                    return f"{b/1024:.1f} KB"
                if b < 1024**3:
                    return f"{b/1024**2:.1f} MB"
                return f"{b/1024**3:.2f} GB"
            except Exception:
                return "?"

        def _log_written(p: Path, extra: str = "") -> None:
            if extra:
                logger.info("Artifacts: wrote %s (%s) %s", str(p), _fmt_size(p), extra)
            else:
                logger.info("Artifacts: wrote %s (%s)", str(p), _fmt_size(p))

        out_root.mkdir(parents=True, exist_ok=True)

        vcf_counts_dir = out_root / "vcf_counts"
        fasta_dir = out_root / "fasta"
        matrices_dir = out_root / "matrices"

        vcf_counts_dir.mkdir(parents=True, exist_ok=True)
        fasta_dir.mkdir(parents=True, exist_ok=True)
        matrices_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Artifacts: start (out=%s)", str(out_root))
        logger.info(
            "Artifacts: inputs kept_sites=%d | ordered_keys=%d | samples=%d | ref_path=%s",
            len(kept_sites),
            len(ordered_keys),
            len(sample_allele_strings),
            str(ref_path) if ref_path else "<none>",
        )

        t_total = time.perf_counter()

        # 1) vcf_counts/all_snp.txt
        t = time.perf_counter()
        all_snp_path = vcf_counts_dir / "all_snp.txt"
        baseline_metadata: Dict[str, Dict[str, str]] = {}
        for i, (chrom, pos, ref_nt, alt_nt) in enumerate(ordered_keys):
            feature_id = f"{chrom}:{pos}:{ref_nt}:{alt_nt}"
            ref_binary_value = ref_binary[i] if i < len(ref_binary) else "0"
            baseline_nt = ref_nt if str(ref_binary_value) == "0" else alt_nt
            baseline_metadata[feature_id] = {
                "Baseline_allele": str(baseline_nt).upper(),
                "Binary_reference_value": str(ref_binary_value),
                "Encoding": "1_if_allele_differs_from_baseline",
            }

        all_snp_rows, all_snp_header = self._write_all_snp_table(
            path=all_snp_path,
            kept_sites=kept_sites,
            ref_path=ref_path,
            baseline_metadata=baseline_metadata,
        )
        dt = time.perf_counter() - t
        n_rows = len(all_snp_rows) if all_snp_rows is not None else -1
        n_cols = len(all_snp_header) if all_snp_header is not None else -1
        logger.info(
            "Artifacts: all_snp.txt done (%.2fs) rows=%d cols=%d", dt, n_rows, n_cols
        )
        _log_written(all_snp_path)

        # 2) fasta/<generic>_{alleles,binary}.fasta
        t = time.perf_counter()
        alleles_fa = fasta_dir / f"{self.generic_name}_alleles.fasta"
        binary_fa = fasta_dir / f"{self.generic_name}_binary.fasta"

        # Note: write_fasta_matrix typically writes a REF row + sample rows.
        write_fasta_matrix(alleles_fa, ref_line, sample_allele_strings, ref_name="REF")
        write_fasta_matrix(binary_fa, ref_binary, sample_binary_strings, ref_name="REF")

        dt = time.perf_counter() - t
        logger.info(
            "Artifacts: fasta write done (%.2fs) sequences=%d (includes REF) | length=%d",
            dt,
            len(sample_allele_strings) + 1,
            len(ref_line),
        )
        _log_written(alleles_fa)
        _log_written(binary_fa)

        # 3) fasta/<generic>_filtered.tsv (filtered copy of all_snp.txt; optional Context_±flank)
        t = time.perf_counter()
        filtered_tsv = fasta_dir / f"{self.generic_name}_filtered.tsv"
        self._write_filtered_copy_with_context(
            input_rows=all_snp_rows,
            input_header=all_snp_header,
            output_path=filtered_tsv,
            kept_positions=set(positions_1based),
            ref_path=ref_path,
        )
        dt = time.perf_counter() - t
        logger.info(
            "Artifacts: filtered.tsv done (%.2fs) feature_rows=%d unique_positions=%d context_ref=%s",
            dt,
            len(all_snp_rows),
            len(set(positions_1based)),
            "yes" if ref_path else "no",
        )
        _log_written(filtered_tsv)

        # 4) matrices/* outputs produced by filtering (minor-count + type + redundancy + fix)
        t = time.perf_counter()
        logger.info("Artifacts: matrices/* start (this is often the slow step)")
        matrices_final_markers = self._write_matrices_outputs(
            fasta_alleles=alleles_fa,
            fasta_binary=binary_fa,
            annotation_tsv=filtered_tsv,
            out_dir=matrices_dir,
            matrix_df=matrix_df,
            ref_line=ref_line,
            sample_allele_strings=sample_allele_strings,
            ref_binary=ref_binary,
        )
        dt = time.perf_counter() - t
        logger.info("Artifacts: matrices/* done (%.2fs)", dt)

        # 5) optional lookup used by returned df
        if integer_id_lookup is not None:
            t = time.perf_counter()
            lookup_path = out_root / "variant_id_lookup.json"
            with open(lookup_path, "w", encoding="utf-8") as f:
                json.dump(integer_id_lookup, f, indent=2)
            dt = time.perf_counter() - t
            logger.info(
                "Artifacts: variant_id_lookup.json done (%.2fs) entries=%d",
                dt,
                len(integer_id_lookup),
            )
            _log_written(lookup_path)

        logger.info("Artifacts: done (total %.2fs)", time.perf_counter() - t_total)
        return matrices_final_markers

    def _write_all_snp_table(
        self,
        path: Path,
        kept_sites: Dict[Tuple[str, int, str, str], int],
        ref_path: Optional[Path],
        baseline_metadata: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> Tuple[List[Dict[str, str]], List[str]]:
        """Write the SNP summary table, with annotation columns if GenBank is provided."""
        path.parent.mkdir(parents=True, exist_ok=True)

        header_annot = [
            "Feature_ID",
            "Position",
            "Count",
            "Sequence",
            "Ref_allele",
            "Alt_allele",
            "Baseline_allele",
            "Binary_reference_value",
            "Encoding",
            "Region_type",
            "Relative_pos",
            "Codon_number",
            "Nucleotide_change",
            "Amino_acid_change",
            "Gene_annotation",
        ]

        # Decide whether to annotate (GenBank) or write minimal table (Position, Count).
        do_annotate = False
        if ref_path and ref_path.exists():
            lower = ref_path.name.lower()
            if lower.endswith((".gb", ".gbk", ".gbff")):
                do_annotate = True

        if do_annotate:
            circular_config = getattr(self.config, "reference_circular_contigs", None)
            if isinstance(circular_config, str):
                circular_config = [
                    value.strip()
                    for value in circular_config.split(",")
                    if value.strip()
                ]
            rows = annotate_snps_genbank(
                kept_sites,
                ref_path,
                reference_id=getattr(self.config, "reference_id", None),
                circular_contigs=circular_config,
            )  # type: ignore[arg-type]
            for row in rows:
                feature_id = str(row.get("Feature_ID", ""))
                if baseline_metadata and feature_id in baseline_metadata:
                    row.update(baseline_metadata[feature_id])
                else:
                    row.setdefault("Baseline_allele", row.get("Ref_allele", ""))
                    row.setdefault("Binary_reference_value", "0")
                    row.setdefault("Encoding", "1_if_allele_differs_from_baseline")
            # Annotation rows intentionally carry reference identity, overlap,
            # strand, and distance fields. Preserve the full auditable schema.
            for row in rows:
                for column in row:
                    if column not in header_annot:
                        header_annot.append(column)
            with open(path, "w", encoding="utf-8", newline="") as out:
                w = csv.DictWriter(
                    out, fieldnames=header_annot, delimiter="\t", lineterminator="\n"
                )
                w.writeheader()
                for r in rows:
                    w.writerow(r)
            return rows, header_annot

        # Minimal output keeps enough marker metadata for query-time raw-sequence encoding.
        header_min = [
            "Feature_ID",
            "Position",
            "Count",
            "Sequence",
            "Ref_allele",
            "Alt_allele",
            "Baseline_allele",
            "Binary_reference_value",
            "Encoding",
            "Nucleotide_change",
            "Reference_id",
            "Reference_build",
            "Reference_checksum_sha256",
            "Contig",
        ]
        rows_min: List[Dict[str, str]] = []

        reference_checksum = ""
        reference_identity = str(
            getattr(self.config, "reference_id", None) or ""
        ).strip()
        if ref_path and ref_path.exists():
            import hashlib as _hashlib

            reference_checksum = _hashlib.sha256(ref_path.read_bytes()).hexdigest()
            if not reference_identity:
                try:
                    reference_identity = ";".join(
                        load_reference_sequences(ref_path).keys()
                    )
                except Exception:
                    reference_identity = ""

        for chrom, pos, ref_nt, alt_nt in sorted(
            kept_sites.keys(), key=lambda x: (x[0], x[1], x[2], x[3])
        ):
            count = kept_sites[(chrom, pos, ref_nt, alt_nt)]
            feature_id = f"{chrom}:{pos}:{ref_nt}:{alt_nt}"
            baseline_row = (
                baseline_metadata.get(feature_id, {}) if baseline_metadata else {}
            )
            rows_min.append(
                {
                    "Feature_ID": feature_id,
                    "Position": str(pos),
                    "Count": str(count),
                    "Sequence": chrom,
                    "Ref_allele": ref_nt.upper(),
                    "Alt_allele": alt_nt.upper(),
                    "Baseline_allele": baseline_row.get(
                        "Baseline_allele", ref_nt.upper()
                    ),
                    "Binary_reference_value": baseline_row.get(
                        "Binary_reference_value", "0"
                    ),
                    "Encoding": baseline_row.get(
                        "Encoding", "1_if_allele_differs_from_baseline"
                    ),
                    "Nucleotide_change": f"{ref_nt}|{alt_nt}",
                    "Reference_id": reference_identity,
                    "Reference_build": reference_identity,
                    "Reference_checksum_sha256": reference_checksum,
                    "Contig": chrom,
                }
            )

        with open(path, "w", encoding="utf-8", newline="") as out:
            w = csv.DictWriter(
                out, fieldnames=header_min, delimiter="\t", lineterminator="\n"
            )
            w.writeheader()
            for r in rows_min:
                w.writerow(r)

        return rows_min, header_min

    def _write_filtered_copy_with_context(
        self,
        input_rows: List[Dict[str, str]],
        input_header: List[str],
        output_path: Path,
        kept_positions: set,
        ref_path: Optional[Path],
    ) -> None:
        """Write a filtered copy of the SNP table, optionally appending Context_±40."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # If we can load reference sequence records, append context columns.
        # For multi-contig references, context is extracted from the record named
        # in the manifest Sequence column. If no matching record is found, the
        # legacy concatenated-reference fallback is used and explicitly recorded.
        ref_records: Dict[str, str] = {}
        ref_seq_fallback: Optional[str] = None
        if ref_path and ref_path.exists():
            lower = ref_path.name.lower()
            if lower.endswith(
                (".fa", ".fna", ".fasta", ".fas", ".gb", ".gbk", ".gbff")
            ):
                try:
                    ref_records = load_reference_sequences(ref_path)
                    ref_seq_fallback = (
                        "".join(ref_records.values()).upper()
                        if ref_records
                        else load_reference_sequence(ref_path)
                    )
                except Exception as e:
                    logger.warning(
                        f"Reference loading failed; context column skipped. Reason: {e}"
                    )
                    ref_records = {}
                    ref_seq_fallback = None

        out_header = list(input_header)
        add_context = ref_seq_fallback is not None and "Position" in input_header
        if add_context:
            out_header.extend(
                [
                    "Context_±40",
                    "Context_flank",
                    "Context_center_offset",
                    "Context_reference_record",
                ]
            )

        with open(output_path, "w", encoding="utf-8", newline="") as out:
            w = csv.writer(out, delimiter="\t", lineterminator="\n")
            w.writerow(out_header)

            for r in input_rows:
                try:
                    pos = int(str(r.get("Position", "")).strip())
                except Exception:
                    continue
                if pos not in kept_positions:
                    continue

                row_vals = [str(r.get(col, "")) for col in input_header]
                if add_context and ref_seq_fallback is not None:
                    flank = 40
                    seq_name = str(r.get("Sequence", "")).strip()
                    ref_seq_for_row = ref_records.get(seq_name) if seq_name else None
                    context_source = (
                        seq_name
                        if ref_seq_for_row is not None
                        else "concatenated_reference_fallback"
                    )
                    if ref_seq_for_row is None:
                        ref_seq_for_row = ref_seq_fallback
                    row_vals.extend(
                        [
                            context_around(pos, ref_seq_for_row, flank=flank),
                            str(flank),
                            str(flank),
                            context_source,
                        ]
                    )
                w.writerow(row_vals)

    def _write_matrices_outputs(
        self,
        fasta_alleles: Path,
        fasta_binary: Path,
        annotation_tsv: Path,
        out_dir: Path,
        *,
        matrix_df: Optional[pd.DataFrame] = None,
        ref_line: Optional[str] = None,
        sample_allele_strings: Optional[Dict[str, str]] = None,
        ref_binary: Optional[str] = None,
    ) -> int:
        """
        Generate final cohort-level matrices (TSV + FASTA + filtered annotation).

        This stage refines the encoded matrix before downstream modeling.
        It operates strictly at the feature level and does NOT perform statistical
        association testing. Instead, it ensures that the final matrix is
        biologically interpretable, structurally stable, and non-redundant.

        Filtering stages (conceptual flow):

        1) Minor-count filter (signal stability control)
           - Removes markers where the minority state is too rare across genomes.
           - In small cohorts, extremely rare states introduce instability and
             can distort downstream tree splits or interaction mining.

        2) Annotation-driven type filter (biological subset selection)
           - Retains only markers whose functional annotation matches a requested category.
           - Skipped if annotation rows do not align with marker count.

        3) Redundancy reduction via pattern grouping (feature de-duplication)
           - Identifies markers that share an identical 0/1 pattern across all genomes.
           - These markers represent the exact same cohort-level signal.
           - Keeps at most `repeat_number` representatives per identical-pattern group.
           - Purpose: collapse perfectly collinear features to reduce redundancy and prevent
             one signal from being over-represented by multiple duplicate columns.

        4) Forced retention of specified positions (controlled override)
           - User-specified marker indices are force-kept even if filtered out.
           - Ensures known markers remain present for downstream reporting.

        Important methodological distinction:
        - Minor-count and type filters are biological inclusion criteria.
        - Redundancy reduction is a feature-engineering / de-duplication step.
        - None of these constitute statistical hypothesis testing.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        base = self.generic_name

        # ─────────────────────────────────────────────
        # 0) Announce what we're about to parse/write
        # ─────────────────────────────────────────────
        logger.info("Matrices/*: start (base=%s)", base)
        logger.info(
            "Matrices stage: feature refinement prior to downstream modeling\n"
            "Purpose: structural filtering + redundancy control\n"
        )

        logger.info(
            "Matrices configuration:\n"
            "min_minor_count=%d\n"
            "annotation_type_filter=%s\n"
            "repeat_number=%d\n"
            "forced_positions='%s'",
            self.matrices_min_count,
            self.matrices_type,
            self.matrices_repeat_number,
            self.matrices_fix,
        )
        direct_matrix_input = matrix_df is not None
        logger.info(
            "Matrices/*: inputs source=%s | alleles_fasta=%s | binary_fasta=%s | annotation_tsv=%s",
            "in_memory_dataframe" if direct_matrix_input else "fasta_fallback",
            str(fasta_alleles),
            str(fasta_binary),
            str(annotation_tsv),
        )
        logger.info("Matrices/*: output_dir=%s", str(out_dir))

        required_inputs = [(annotation_tsv, "annotation TSV")]
        if not direct_matrix_input:
            required_inputs.extend(
                [
                    (fasta_alleles, "alleles FASTA"),
                    (fasta_binary, "binary FASTA"),
                ]
            )
        for p, label in required_inputs:
            if not p.exists():
                raise FileNotFoundError(f"Matrices/*: missing {label}: {p}")

        # ─────────────────────────────────────────────
        # 1) Load (parse) inputs
        # ─────────────────────────────────────────────
        load_started_at = time.perf_counter()
        annot_rows, annot_header = read_annotation_tsv(annotation_tsv)
        allele_sequences: "OrderedDict[str, str]" = OrderedDict()

        if direct_matrix_input:
            if ref_line is None or ref_binary is None or sample_allele_strings is None:
                raise ValueError(
                    "Direct matrix artifact processing requires ref_line, ref_binary, "
                    "and sample_allele_strings."
                )
            assert matrix_df is not None
            sample_ids = [str(sample) for sample in matrix_df.index]
            S = int(matrix_df.shape[1])
            if len(ref_line) != S or len(ref_binary) != S:
                raise ValueError(
                    "Direct matrix/reference widths differ before artifact filtering."
                )

            allele_sequences["REF"] = str(ref_line)
            for sample in sample_ids:
                if sample not in sample_allele_strings:
                    raise ValueError(
                        f"Missing in-memory allele sequence for sample {sample!r}."
                    )
                sequence = str(sample_allele_strings[sample])
                if len(sequence) != S:
                    raise ValueError(
                        f"Allele sequence width mismatch for sample {sample!r}."
                    )
                allele_sequences[sample] = sequence

            sample_values = (
                matrix_df.apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype=float, copy=False)
            )
            ref_values = np.asarray(
                [
                    0.0 if value == "0" else 1.0 if value == "1" else np.nan
                    for value in ref_binary
                ],
                dtype=float,
            )
            binary_values = np.vstack([ref_values, sample_values])
            genomes_order = ["REF"] + sample_ids
        else:
            alleles = read_fasta_matrix(fasta_alleles)
            binary = read_fasta_matrix(fasta_binary)
            if not alleles or not binary:
                raise ValueError(
                    "Matrices/*: empty FASTA matrix detected (alleles or binary)."
                )
            if list(alleles.keys()) != list(binary.keys()):
                raise ValueError(
                    "Matrices/*: genome order differs between alleles and binary FASTA files."
                )

            genomes_order = [str(genome) for genome in alleles.keys()]
            S = len(next(iter(alleles.values())))
            if len(next(iter(binary.values()))) != S:
                raise ValueError(
                    "Matrices/*: alleles and binary FASTA have different marker counts."
                )
            allele_sequences = OrderedDict(
                (str(genome), "".join(values))
                for genome, values in alleles.items()
            )
            binary_values = np.asarray(
                [
                    [
                        0.0
                        if value == "0"
                        else 1.0
                        if value == "1"
                        else np.nan
                        for value in values
                    ]
                    for values in binary.values()
                ],
                dtype=float,
            )

        load_seconds = time.perf_counter() - load_started_at

        logger.info(
            "Matrices/*: inputs ready | source=%s | genomes=%d | markers=%d | seconds=%.3f",
            "in_memory_dataframe" if direct_matrix_input else "fasta_fallback",
            len(genomes_order),
            S,
            load_seconds,
        )

        annotation_matches = len(annot_rows) == S
        if annotation_matches:
            logger.info(
                "Matrices/*: annotation rows match markers (rows=%d)", len(annot_rows)
            )
        else:
            logger.warning(
                "Matrices/*: annotation rows do NOT match markers (rows=%d vs markers=%d). "
                "Type-filter will be skipped and annotation will be copied as-is.",
                len(annot_rows),
                S,
            )

        # ─────────────────────────────────────────────
        # 2) Announce filter parameters
        # ─────────────────────────────────────────────
        logger.info(
            "Matrices/*: params min_count=%d | type=%s | repeat_number=%d | fix='%s'",
            self.matrices_min_count,
            self.matrices_type,
            self.matrices_repeat_number,
            self.matrices_fix,
        )

        # ─────────────────────────────────────────────
        # 3) Filter 1: minor-count
        #
        # Keeps a marker only if the minority state (usually '1' in a 0/1 column)
        # appears at least `min_count` times across genomes.
        #
        # Intuition: columns where only 1 genome differs can be unstable in tiny cohorts
        # and can inflate downstream splits or interactions by acting as "singletons".
        # ─────────────────────────────────────────────
        logger.info(
            "Applying minor-count sanity filter:\n"
            "  Purpose: enforce low-count stability on the actual artifact matrix\n"
            "  Note: this should remove nothing if DataLoader artifact synchronization worked correctly."
        )

        filter_started_at = time.perf_counter()
        count_0 = np.sum(binary_values == 0.0, axis=0)
        count_1 = np.sum(binary_values == 1.0, axis=0)
        mask_minor = list(
            np.minimum(count_0, count_1) >= int(self.matrices_min_count)
        )
        kept_minor = int(sum(mask_minor))

        logger.info(
            "Minor-count sanity filter result: kept=%d/%d (removed=%d)",
            kept_minor,
            S,
            S - kept_minor,
        )
        logger.info(
            "Minor-count filter result: kept=%d/%d (removed=%d)",
            kept_minor,
            S,
            S - kept_minor,
        )

        # ─────────────────────────────────────────────
        # 4) Filter 2: type filter (annotation-driven)
        #
        # Keeps markers whose annotation indicates the requested category (e.g., coding/non-coding/etc).
        # This is only meaningful if annotation rows align 1:1 with marker columns.
        # If annotation does not match, we skip this filter rather than risk mislabeling columns.
        # ─────────────────────────────────────────────
        if annotation_matches and annot_header and self.matrices_type != "all":
            logger.info(
                "Applying annotation-driven type filter:\n"
                "  Keeping markers matching type='%s'\n"
                "  Purpose: biological subset selection",
                self.matrices_type,
            )
            mask_type = type_filter(annot_rows, self.matrices_type)
        else:
            mask_type = [True] * S
            logger.info(
                "Type filter skipped (either type='all' or annotation mismatch)."
            )

        kept_type = int(sum(mask_type))

        logger.info(
            "Type filter result: kept=%d/%d",
            kept_type,
            S,
        )

        # ─────────────────────────────────────────────
        # 5) Combined mask (minor AND type)
        #
        # This is the inclusion mask after the "content" filters:
        #   - minor-count: statistical stability
        #   - type: biological subset selection
        # ─────────────────────────────────────────────
        mask_12 = combine_masks(mask_minor, mask_type)
        kept_12 = int(sum(mask_12))
        logger.info("Matrices/*: combined (minor AND type) kept=%d/%d", kept_12, S)

        idx12 = [i for i, k in enumerate(mask_12) if k]
        binary_values_12 = binary_values[:, np.asarray(mask_12, dtype=bool)]
        annot_rows_12 = (
            [r for r, k in zip(annot_rows, mask_12) if k] if annotation_matches else []
        )

        # ─────────────────────────────────────────────
        # 6) Filter 3: redundancy reduction by identical binary pattern
        #
        # What "redundancy" means here:
        #   Two (or more) markers can produce the *exact same* 0/1 vector across all genomes.
        #   Example (conceptual):
        #       marker A: 0 0 1 0 1 0 ...
        #       marker B: 0 0 1 0 1 0 ...
        #
        # In that case, A and B are perfectly collinear:
        #   - Any model/split using A could use B interchangeably.
        #   - Keeping all of them can over-represent one signal and inflate apparent importance.
        #
        # This step groups markers by their full-sample binary pattern and keeps only a limited
        # number of representatives per group.
        #
        # How `repeat_number` is used:
        #   - repeat_number = 1  → keep only one representative marker per identical-pattern group
        #   - repeat_number = k  → keep up to k representatives per group (useful if you want
        #                          a small amount of redundancy retained for downstream reporting)
        #
        # What this step is NOT:
        #   - It does not remove markers because they are rare (minor-count already handled that).
        #   - It does not change the 0/1 encoding.
        #   - It does not merge markers into a new synthetic feature; it simply drops duplicates.
        #
        # Why it is "same as other filters" structurally:
        #   - It produces another boolean mask (keep/drop) and logs kept counts.
        # Why it is different conceptually:
        #   - It is a de-duplication / collinearity control step, not a biological inclusion filter.
        #
        # Interpretable output:
        #   - The logging includes `unique_patterns` as a compact summary:
        #       *many columns* collapsing to *few patterns* suggests strong redundancy in the matrix.
        # ─────────────────────────────────────────────
        if binary_values_12.shape[1] > 0:
            pattern_keys = binary_pattern_keys(binary_values_12)
            unique_patterns = len(set(pattern_keys))

            logger.info(
                "Applying redundancy reduction (pattern grouping):\n"
                "  unique_binary_patterns=%d\n"
                "  repeat_number=%d\n"
                "  Purpose: collapse perfectly collinear markers\n"
                "           (markers sharing identical cohort-level signals)",
                unique_patterns,
                self.matrices_repeat_number,
            )

            annotation_for_grouping = (
                annot_rows_12
                if annotation_matches
                else [{"Position": str(i + 1)} for i in range(binary_values_12.shape[1])]
            )

            mask_group = group_and_reduce_by_pattern(
                [[] for _ in pattern_keys],
                annotation_for_grouping,
                self.matrices_repeat_number,
                sample_threshold=int(
                    getattr(self.config, "matrices_redundancy_sample_threshold", 2000)
                ),
                sample_size=int(
                    getattr(self.config, "matrices_redundancy_sample_size", 256)
                ),
                pattern_keys=pattern_keys,
            )

            kept_group = int(sum(mask_group))

            logger.info(
                "Redundancy reduction result:\n"
                "  retained=%d/%d\n"
                "  removed_duplicate_representations=%d",
                kept_group,
                binary_values_12.shape[1],
                binary_values_12.shape[1] - kept_group,
            )
        else:
            mask_group = []
            kept_group = 0
            logger.info(
                "Redundancy reduction skipped (no markers after biological filtering)."
            )

        # Convert redundancy mask back to full length
        mask_final = [False] * S
        for kept_flag, original_idx in zip(mask_group, idx12):
            if kept_flag:
                mask_final[original_idx] = True
        # ─────────────────────────────────────────────
        # 7) Force-keep fixed positions
        # ─────────────────────────────────────────────
        fix_idx0, fix_warnings = parse_fix_positions(self.matrices_fix, S)

        for w in fix_warnings:
            logger.warning(w)

        if fix_idx0:
            logger.info(
                "Applying forced retention of %d user-specified marker(s).",
                len(fix_idx0),
            )

        for idx in fix_idx0:
            if 0 <= idx < S:
                mask_final[idx] = True

        kept_final = int(sum(mask_final))

        logger.info(
            "Final marker count after all filters:\n" "  %d/%d retained",
            kept_final,
            S,
        )

        # Apply the final mask directly to the in-memory row representation.
        keep_indices = np.flatnonzero(np.asarray(mask_final, dtype=bool))
        alleles_filt: "OrderedDict[str, List[str]]" = OrderedDict(
            (
                genome,
                [sequence[index] for index in keep_indices],
            )
            for genome, sequence in allele_sequences.items()
        )
        binary_filt: "OrderedDict[str, List[str]]" = OrderedDict()
        for row_index, genome in enumerate(genomes_order):
            binary_filt[genome] = [
                "0"
                if binary_values[row_index, index] == 0.0
                else "1"
                if binary_values[row_index, index] == 1.0
                else "?"
                for index in keep_indices
            ]

        # Filter annotation rows if possible; otherwise keep original
        if annotation_matches:
            annot_filt = [r for r, k in zip(annot_rows, mask_final) if k]
        else:
            annot_filt = annot_rows

        if annotation_matches:
            # Carry the full marker identity into the matrix columns.  Position-only
            # columns are ambiguous once alternate alleles and query-time raw
            # sequence encoding are supported.
            positions_f = [
                (r.get("Feature_ID") or r.get("Position", "") or "").strip()
                for r in annot_filt
            ]
        else:
            positions_f = [str(i + 1) for i in range(kept_final)]

        filter_seconds = time.perf_counter() - filter_started_at
        logger.info(
            "Matrices/*: filtering complete | markers_in=%d | markers_out=%d | seconds=%.3f",
            S,
            kept_final,
            filter_seconds,
        )

        # ─────────────────────────────────────────────
        # 8) Write outputs
        # ─────────────────────────────────────────────
        out_alleles_tsv = out_dir / f"{base}_alleles.tsv"
        out_binary_tsv = out_dir / f"{base}_binary.tsv"
        out_alleles_fa = out_dir / f"{base}_alleles.fasta"
        out_binary_fa = out_dir / f"{base}_binary.fasta"
        out_filtered_tsv = out_dir / f"{base}_filtered.tsv"
        out_feature_manifest_tsv = out_dir / f"{base}_feature_manifest.tsv"

        write_started_at = time.perf_counter()
        write_matrix_tsv_rows(out_alleles_tsv, positions_f, alleles_filt)
        write_matrix_tsv_rows(out_binary_tsv, positions_f, binary_filt)
        write_fasta_matrix_wrapped(out_alleles_fa, alleles_filt)
        write_fasta_matrix_wrapped(out_binary_fa, binary_filt)

        if annotation_matches and annot_header:
            write_annotation_tsv(out_filtered_tsv, annot_filt, annot_header)
            write_annotation_tsv(out_feature_manifest_tsv, annot_filt, annot_header)
        else:
            copied_annotation = annotation_tsv.read_text(
                encoding="utf-8", errors="replace"
            )
            out_filtered_tsv.write_text(copied_annotation, encoding="utf-8")
            out_feature_manifest_tsv.write_text(copied_annotation, encoding="utf-8")

        write_seconds = time.perf_counter() - write_started_at

        logger.info(
            "Matrices/*: wrote outputs: %s | %s | %s | %s | %s | %s",
            str(out_alleles_tsv),
            str(out_binary_tsv),
            str(out_alleles_fa),
            str(out_binary_fa),
            str(out_filtered_tsv),
            str(out_feature_manifest_tsv),
        )
        logger.info(
            "Matrices/*: done | load_seconds=%.3f | filter_seconds=%.3f | "
            "write_seconds=%.3f | total_seconds=%.3f",
            load_seconds,
            filter_seconds,
            write_seconds,
            load_seconds + filter_seconds + write_seconds,
        )
        return kept_final

    # ──────────────────────────────────────────────────────────
    # Prebuilt matrix loader
    # ──────────────────────────────────────────────────────────

    def _load_matrix_file(self, path: Path) -> pd.DataFrame:
        """Load a prebuilt matrix from CSV/TSV with row index in the first column."""
        sep = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
        df = pd.read_csv(path, sep=sep, index_col=0)

        raw_samples = int(df.shape[0])
        raw_features = int(df.shape[1])
        missing_cells = int(df.isna().sum().sum())

        df.index = df.index.astype(str)
        df.index.name = "Sample"

        if log_flow_step is not None:
            log_flow_step(
                logger,
                step="Preprocessing checkpoint — prebuilt matrix loading",
                happened="Loaded the supplied matrix and normalized the row index as sample identifiers.",
                reason="Downstream metadata alignment depends on stable sample identifiers; feature values are preserved here so central statistical filtering receives the user-supplied feature representation.",
                before_samples=raw_samples,
                before_features=raw_features,
                after_samples=int(df.shape[0]),
                after_features=int(df.shape[1]),
                threshold="none at load step",
                status="missing_cells_detected" if missing_cells else "complete",
            )
        else:
            logger.info(
                "Loaded prebuilt matrix | samples=%d | features=%d",
                df.shape[0],
                df.shape[1],
            )

        if missing_cells > 0:
            logger.info(
                "Preprocessing note — detected %d missing cells; values are preserved at matrix loading "
                "so downstream stages can apply their configured baseline/imputation rules",
                missing_cells,
            )

        return df
