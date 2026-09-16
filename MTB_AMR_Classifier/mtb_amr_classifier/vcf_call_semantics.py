#!/usr/bin/env python3
# network_parser/vcf_call_semantics.py
"""
Shared VCF genotype / QC / callability semantics for training and query.

Training (DataLoader) and query (sequence_query_encoder) must use the same
rules so that GT, FILTER, QUAL, DP, GQ, MQ, and absence of a site mean the
same thing in both paths.

Call states
-----------
CALLED_REFERENCE          Explicit callable reference (GT all-0, or gVCF ref block)
CALLED_ALTERNATE          Explicit alternate allele call
MISSING_OR_NO_CALL        GT is ., ./., empty, or sample field missing GT
FILTERED_OR_LOW_QUALITY   Site fails FILTER or QC thresholds
LOCUS_NOT_PRESENT         No record at coordinate in a variant-only VCF
UNRESOLVED_OR_AMBIGUOUS   Heterozygous / multi-allelic / malformed / mixed ploidy
                          that cannot be reduced safely to the binary encoding

Binary encoding
---------------
  CALLED_REFERENCE (matching feature baseline/ref) → 0.0
  CALLED_ALTERNATE (matching feature alt)          → 1.0
  anything else                                    → NaN  (not ordinary zero evidence)

Legacy "absence means reference" is available only via the explicit config flag
``assume_absent_variant_is_reference`` (default False). When enabled, a
prominent warning and per-call audit field ``assumed_reference=True`` are set.
"""

from __future__ import annotations

from bisect import bisect_right
import gzip
import logging
import warnings
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

try:
    from mtb_amr_classifier.utils import normalize_sample_id
except ImportError:  # pragma: no cover - supports direct source-tree execution
    from utils import normalize_sample_id  # type: ignore

logger = logging.getLogger(__name__)


class CallState(str, Enum):
    CALLED_REFERENCE = "CALLED_REFERENCE"
    CALLED_ALTERNATE = "CALLED_ALTERNATE"
    MISSING_OR_NO_CALL = "MISSING_OR_NO_CALL"
    FILTERED_OR_LOW_QUALITY = "FILTERED_OR_LOW_QUALITY"
    LOCUS_NOT_PRESENT = "LOCUS_NOT_PRESENT"
    UNRESOLVED_OR_AMBIGUOUS = "UNRESOLVED_OR_AMBIGUOUS"


# Sentinel float for non-callable encodings (also use np.nan in matrices).
NON_CALLABLE = float("nan")


@dataclass
class VcfQCConfig:
    """QC and callability policy shared by training and query."""

    qual_threshold: float = 30.0
    min_dp: int = 10
    min_gq: int = 20
    mq_threshold: float = 40.0
    mq0f_threshold: float = 0.1
    biallelic_only: bool = True
    # The binary microbial encoder supports haploid and homozygous diploid
    # calls. Higher ploidies are retained as unresolved instead of being
    # silently reduced to a binary state.
    supported_ploidies: Tuple[int, ...] = (1, 2)
    respect_filter: bool = True
    allowed_filters: Tuple[str, ...] = ("PASS", ".")
    # Safe default: do NOT treat variant-only absence as reference.
    assume_absent_variant_is_reference: bool = False
    # When True, compare VCF REF to provided reference genome at POS.
    validate_ref_against_genome: bool = False
    # gVCF END block support
    expand_gvcf_ref_blocks: bool = True
    # Query recovery gates (0–1 fractions of selected features)
    min_feature_recovery_fraction: float = 0.5
    min_callable_fraction: float = 0.5
    # If True, fail query prediction when recovery/callability below thresholds.
    enforce_query_callability_gates: bool = True
    # Explicit contig aliases (query → training/manifest). Never position-only by default.
    contig_alias_map: Optional[Dict[str, str]] = None
    allow_position_only_match: bool = False

    @classmethod
    def from_config(cls, config: Any) -> "VcfQCConfig":
        if config is None:
            return cls()
        allowed = getattr(config, "vcf_allowed_filters", ("PASS", "."))
        if isinstance(allowed, str):
            allowed = tuple(x.strip() for x in allowed.split(",") if x.strip())
        raw_aliases = getattr(config, "contig_alias_map", None)
        alias_map: Optional[Dict[str, str]] = None
        if isinstance(raw_aliases, dict):
            alias_map = {str(k): str(v) for k, v in raw_aliases.items()}
        elif isinstance(raw_aliases, str) and raw_aliases.strip():
            import json as _json

            text = raw_aliases.strip()
            if text.startswith("{"):
                alias_map = {str(k): str(v) for k, v in _json.loads(text).items()}
            else:
                alias_map = {}
                for part in text.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    if "=" not in part:
                        raise ValueError(
                            f"contig_alias_map entry must be query=train (got {part!r})"
                        )
                    src, dst = part.split("=", 1)
                    alias_map[src.strip()] = dst.strip()
        raw_supported_ploidies = getattr(config, "vcf_supported_ploidies", (1, 2))
        if isinstance(raw_supported_ploidies, str):
            raw_supported_ploidies = [
                value.strip()
                for value in raw_supported_ploidies.split(",")
                if value.strip()
            ]
        return cls(
            qual_threshold=float(getattr(config, "qual_threshold", 30.0)),
            min_dp=int(getattr(config, "min_dp_per_sample", 10)),
            min_gq=int(getattr(config, "min_gq_per_sample", 20)),
            mq_threshold=float(getattr(config, "mq_threshold", 40.0)),
            mq0f_threshold=float(getattr(config, "mq0f_threshold", 0.1)),
            biallelic_only=bool(getattr(config, "biallelic_only", True)),
            supported_ploidies=tuple(int(value) for value in raw_supported_ploidies),
            respect_filter=bool(getattr(config, "vcf_respect_filter", True)),
            allowed_filters=tuple(allowed),
            assume_absent_variant_is_reference=bool(
                getattr(config, "assume_absent_variant_is_reference", False)
            ),
            validate_ref_against_genome=bool(
                getattr(config, "validate_ref_against_genome", False)
            ),
            expand_gvcf_ref_blocks=bool(
                getattr(config, "expand_gvcf_ref_blocks", True)
            ),
            min_feature_recovery_fraction=float(
                getattr(config, "min_feature_recovery_fraction", 0.5)
            ),
            min_callable_fraction=float(getattr(config, "min_callable_fraction", 0.5)),
            enforce_query_callability_gates=bool(
                getattr(config, "enforce_query_callability_gates", True)
            ),
            contig_alias_map=alias_map,
            allow_position_only_match=bool(
                getattr(config, "allow_position_only_vcf_match", False)
            ),
        )


@dataclass
class AlleleCall:
    """One classified call at a genomic coordinate (or gVCF block)."""

    state: CallState
    chrom: str
    pos: int
    ref: str
    alts: List[str] = field(default_factory=list)
    called_allele: Optional[str] = None
    allele_index: Optional[int] = None
    filter_status: str = ""
    qc_reasons: List[str] = field(default_factory=list)
    ploidy: Optional[int] = None
    is_phased: Optional[bool] = None
    is_gvcf_ref_block: bool = False
    end_pos: Optional[int] = None
    assumed_reference: bool = False
    sample_name: str = ""
    source_vcf: str = ""
    gt_raw: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        return d


@dataclass
class ParsedGT:
    """Parsed genotype field."""

    tokens: List[str]
    indices: List[Optional[int]]
    ploidy: int
    is_phased: bool
    all_missing: bool
    any_missing: bool
    raw: str


def open_text_maybe_gzip(path: Path):
    path = Path(path)
    if str(path).lower().endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r", encoding="utf-8", errors="replace")


def parse_info_field(info_str: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not info_str or info_str == ".":
        return out
    for item in info_str.split(";"):
        if not item:
            continue
        if "=" in item:
            k, v = item.split("=", 1)
            out[k] = v
        else:
            out[item] = "1"
    return out


def parse_sample_fields(
    fmt: Optional[str], sample_field: Optional[str]
) -> Dict[str, str]:
    if not fmt or not sample_field or sample_field == ".":
        return {}
    keys = fmt.split(":")
    vals = sample_field.split(":")
    return {k: (vals[i] if i < len(vals) else "") for i, k in enumerate(keys)}


def parse_gt(gt: str) -> ParsedGT:
    """Parse GT values including haploid, diploid, phased, unphased, missing."""
    raw = "" if gt is None else str(gt).strip()
    if raw in {"", "."}:
        return ParsedGT(
            tokens=["."],
            indices=[None],
            ploidy=1,
            is_phased=False,
            all_missing=True,
            any_missing=True,
            raw=raw,
        )

    is_phased = "|" in raw and "/" not in raw
    if "/" in raw:
        tokens = raw.split("/")
        is_phased = False
    elif "|" in raw:
        tokens = raw.split("|")
        is_phased = True
    else:
        tokens = [raw]

    indices: List[Optional[int]] = []
    for tok in tokens:
        t = tok.strip()
        if t in {"", "."}:
            indices.append(None)
            continue
        try:
            indices.append(int(t))
        except ValueError:
            indices.append(None)

    any_missing = any(i is None for i in indices)
    all_missing = all(i is None for i in indices) or raw in {".", "./.", ".|."}
    return ParsedGT(
        tokens=tokens,
        indices=indices,
        ploidy=len(tokens),
        is_phased=is_phased,
        all_missing=all_missing,
        any_missing=any_missing,
        raw=raw,
    )


def is_snp_like(ref: str, alt_field: str, *, biallelic_only: bool = True) -> bool:
    ref = (ref or "").upper()
    alt_field = (alt_field or "").upper()
    if not ref or ref == "." or not alt_field or alt_field == ".":
        return False
    # gVCF non-ref symbolic
    if (
        alt_field in {"<*>", "<NON_REF>", ".,<*>"}
        or "<*>" in alt_field
        or "<NON_REF>" in alt_field
    ):
        # Allow gVCF ref-block style records even when ALT is symbolic
        return True
    alts = [a.strip() for a in alt_field.split(",") if a.strip() and a.strip() != "."]
    if not alts:
        return False
    if biallelic_only and len(alts) != 1:
        # Keep multi-allelic for classification as UNRESOLVED later; still parse.
        pass
    if len(ref) != 1:
        return False
    for a in alts:
        if a in {"<*>", "<NON_REF>"}:
            continue
        if len(a) != 1:
            return False
    return True


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "" or value == ".":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "" or value == ".":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> Optional[float]:
    """Return float if present and parseable; None if field is missing/empty/."""
    if value is None or value == "" or value == ".":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> Optional[int]:
    """Return int if present and parseable; None if field is missing/empty/."""
    f = _optional_float(value)
    if f is None:
        return None
    try:
        return int(f)
    except (TypeError, ValueError):
        return None


def record_fails_site_qc(
    *,
    qual_str: str,
    filter_field: str,
    info: Dict[str, str],
    sample_map: Dict[str, str],
    qc: VcfQCConfig,
) -> Tuple[bool, List[str]]:
    """
    Return (failed, reasons) for FILTER + QUAL/DP/GQ/MQ/MQ0F.

    Missing QC fields are audited as ``*_missing`` and treated as failed when
    the corresponding threshold is active — never silently equated to a measured
    zero (which is a real observation that may pass or fail on its own).
    """
    reasons: List[str] = []

    if qc.respect_filter:
        filt = (filter_field or ".").strip()
        # Multi-filter: all tokens must be allowed, or PASS/.
        tokens = [t.strip() for t in filt.split(";") if t.strip()]
        if not tokens:
            tokens = ["."]
        allowed = set(qc.allowed_filters)
        if not all(t in allowed for t in tokens):
            reasons.append(f"FILTER={filt}")

    qual = _optional_float(qual_str)
    if qual is None:
        reasons.append("QUAL_missing")
    elif qual < float(qc.qual_threshold):
        reasons.append(f"QUAL={qual}<{qc.qual_threshold}")

    # Prefer sample-level DP; fall back to INFO; distinguish missing from zero.
    dp = _optional_int(sample_map.get("DP"))
    if dp is None:
        dp = _optional_int(info.get("DP"))
    if dp is None:
        reasons.append("DP_missing")
    elif dp < int(qc.min_dp):
        reasons.append(f"DP={dp}<{qc.min_dp}")

    # GQ is required when threshold > 0; missing ≠ measured zero.
    if int(qc.min_gq) > 0:
        gq = _optional_int(sample_map.get("GQ"))
        if gq is None:
            gq = _optional_int(info.get("GQ"))
        if gq is None:
            reasons.append("GQ_missing")
        elif gq < int(qc.min_gq):
            reasons.append(f"GQ={gq}<{qc.min_gq}")

    if "MQ" in info:
        mq = _optional_float(info.get("MQ"))
        if mq is None:
            reasons.append("MQ_missing")
        elif mq < float(qc.mq_threshold):
            reasons.append(f"MQ={mq}<{qc.mq_threshold}")
    elif float(qc.mq_threshold) > 0:
        # MQ not provided but threshold active
        reasons.append("MQ_missing")

    if "MQ0F" in info:
        mq0f = _optional_float(info.get("MQ0F"))
        if mq0f is None:
            reasons.append("MQ0F_missing")
        elif mq0f > float(qc.mq0f_threshold):
            reasons.append(f"MQ0F={mq0f}>{qc.mq0f_threshold}")

    return (len(reasons) > 0, reasons)


def classify_genotype(
    *,
    ref: str,
    alts: Sequence[str],
    gt_raw: str,
    sample_map: Dict[str, str],
    fmt_present: bool,
    sample_field_present: bool,
    supported_ploidies: Sequence[int] = (1, 2),
) -> Tuple[CallState, Optional[str], Optional[int], ParsedGT]:
    """
    Classify GT into a CallState and optional called allele.

    Rules:
    - Missing / empty / malformed GT → MISSING_OR_NO_CALL (never ALT)
    - Missing FORMAT or sample → MISSING_OR_NO_CALL (never ALT)
    - All-zero GT → CALLED_REFERENCE
    - Single non-ref allele index (haploid or homozygous) → CALLED_ALTERNATE
    - Heterozygous / mixed / multi-alt unresolved for binary pipeline → UNRESOLVED
    """
    ref_u = (ref or "").upper()
    alt_list = [
        str(a).upper() for a in alts if str(a).strip() and str(a).strip() != "."
    ]

    empty_gt = ParsedGT(
        tokens=[],
        indices=[],
        ploidy=0,
        is_phased=False,
        all_missing=True,
        any_missing=True,
        raw="",
    )

    if not fmt_present or not sample_field_present:
        return CallState.MISSING_OR_NO_CALL, None, None, empty_gt

    if "GT" not in sample_map:
        return CallState.MISSING_OR_NO_CALL, None, None, empty_gt

    parsed = parse_gt(sample_map.get("GT", ""))
    if parsed.all_missing:
        return CallState.MISSING_OR_NO_CALL, None, None, parsed

    allowed_ploidies = {int(value) for value in supported_ploidies}
    if parsed.ploidy not in allowed_ploidies:
        return CallState.UNRESOLVED_OR_AMBIGUOUS, None, None, parsed

    # Partial missing e.g. 0/. or ./1
    if parsed.any_missing:
        return CallState.UNRESOLVED_OR_AMBIGUOUS, None, None, parsed

    indices = [i for i in parsed.indices if i is not None]
    if not indices:
        return CallState.MISSING_OR_NO_CALL, None, None, parsed

    # Out-of-range allele indices
    max_idx = len(alt_list)
    for idx in indices:
        if idx < 0 or idx > max_idx:
            return CallState.UNRESOLVED_OR_AMBIGUOUS, None, None, parsed

    unique = sorted(set(indices))
    if unique == [0]:
        return CallState.CALLED_REFERENCE, ref_u, 0, parsed

    # Homozygous or haploid alternate
    if len(unique) == 1 and unique[0] > 0:
        idx = unique[0]
        if idx > len(alt_list):
            return CallState.UNRESOLVED_OR_AMBIGUOUS, None, None, parsed
        allele = alt_list[idx - 1]
        if allele in {"<*>", "<NON_REF>"}:
            # gVCF non-ref symbolic without a concrete allele → unresolved
            return CallState.UNRESOLVED_OR_AMBIGUOUS, None, idx, parsed
        return CallState.CALLED_ALTERNATE, allele, idx, parsed

    # Heterozygous or multi-allelic mixture — not safely binary
    return CallState.UNRESOLVED_OR_AMBIGUOUS, None, None, parsed


def classify_vcf_record(
    parts: Sequence[str],
    *,
    qc: VcfQCConfig,
    sample_name: str = "",
    source_vcf: str = "",
    ref_genome: Optional[Dict[str, str]] = None,
) -> Optional[AlleleCall]:
    """
    Classify one VCF data line into an AlleleCall.

    Returns None only when the line is not a parseable variant/gVCF site for
    this pipeline (e.g. structural variants when SNP-like check fails hard).
    """
    if len(parts) < 8:
        return None

    chrom = parts[0]
    try:
        pos = int(parts[1])
    except (TypeError, ValueError):
        return None

    ref = str(parts[3]).upper()
    alt_field = str(parts[4]).upper()
    qual_str = parts[5]
    filter_field = parts[6] if len(parts) > 6 else "."
    info = parse_info_field(parts[7])

    # Determine if gVCF-style
    is_gvcf_symbolic = (
        "<*>" in alt_field
        or "<NON_REF>" in alt_field
        or alt_field in {".", "<*>", "<NON_REF>"}
    )
    end_pos = _safe_int(info.get("END"), default=pos)
    if end_pos < pos:
        end_pos = pos

    alts = [
        a.strip().upper()
        for a in alt_field.split(",")
        if a.strip() and a.strip() != "."
    ]
    # Strip symbolic for allele list used in GT indexing where possible
    concrete_alts = [a for a in alts if a not in {"<*>", "<NON_REF>"}]

    if not is_gvcf_symbolic and not is_snp_like(ref, alt_field, biallelic_only=False):
        return AlleleCall(
            state=CallState.UNRESOLVED_OR_AMBIGUOUS,
            chrom=chrom,
            pos=pos,
            ref=ref,
            alts=alts,
            filter_status=filter_field,
            qc_reasons=["non_snp_like_or_complex_allele"],
            sample_name=sample_name,
            source_vcf=source_vcf,
        )

    # biallelic_only=True: multiallelic records are unresolved (not silently kept).
    if qc.biallelic_only and len(concrete_alts) > 1:
        return AlleleCall(
            state=CallState.UNRESOLVED_OR_AMBIGUOUS,
            chrom=chrom,
            pos=pos,
            ref=ref,
            alts=alts,
            filter_status=filter_field,
            qc_reasons=["multiallelic_rejected_biallelic_only"],
            sample_name=sample_name,
            source_vcf=source_vcf,
        )

    fmt = parts[8] if len(parts) >= 9 else None
    sample_field = parts[9] if len(parts) >= 10 else None
    sample_map = parse_sample_fields(fmt, sample_field)

    failed, reasons = record_fails_site_qc(
        qual_str=qual_str,
        filter_field=filter_field,
        info=info,
        sample_map=sample_map,
        qc=qc,
    )
    if failed:
        return AlleleCall(
            state=CallState.FILTERED_OR_LOW_QUALITY,
            chrom=chrom,
            pos=pos,
            ref=ref,
            alts=alts,
            filter_status=filter_field,
            qc_reasons=reasons,
            is_gvcf_ref_block=is_gvcf_symbolic and end_pos > pos,
            end_pos=end_pos if end_pos != pos else None,
            sample_name=sample_name,
            source_vcf=source_vcf,
            gt_raw=sample_map.get("GT", ""),
        )

    if qc.validate_ref_against_genome and ref_genome is not None:
        seq = ref_genome.get(chrom) or ref_genome.get(chrom.split("|")[0])
        if seq and 1 <= pos <= len(seq):
            expected = seq[pos - 1].upper()
            if ref and expected and ref[0] != expected:
                return AlleleCall(
                    state=CallState.UNRESOLVED_OR_AMBIGUOUS,
                    chrom=chrom,
                    pos=pos,
                    ref=ref,
                    alts=alts,
                    filter_status=filter_field,
                    qc_reasons=[f"REF_mismatch_genome={expected}"],
                    sample_name=sample_name,
                    source_vcf=source_vcf,
                    gt_raw=sample_map.get("GT", ""),
                )

    state, called, allele_idx, parsed = classify_genotype(
        ref=ref,
        alts=concrete_alts if concrete_alts else alts,
        gt_raw=sample_map.get("GT", ""),
        sample_map=sample_map,
        fmt_present=bool(fmt),
        sample_field_present=bool(sample_field) and sample_field != ".",
        supported_ploidies=qc.supported_ploidies,
    )

    # gVCF reference blocks with GT=0 or 0/0
    is_ref_block = bool(
        qc.expand_gvcf_ref_blocks
        and state == CallState.CALLED_REFERENCE
        and (is_gvcf_symbolic or end_pos > pos)
    )

    return AlleleCall(
        state=state,
        chrom=chrom,
        pos=pos,
        ref=ref,
        alts=alts,
        called_allele=called,
        allele_index=allele_idx,
        filter_status=filter_field,
        qc_reasons=list(reasons),
        ploidy=parsed.ploidy if parsed.ploidy else None,
        is_phased=parsed.is_phased if parsed.raw else None,
        is_gvcf_ref_block=is_ref_block,
        end_pos=end_pos if end_pos != pos else None,
        sample_name=sample_name,
        source_vcf=source_vcf,
        gt_raw=parsed.raw,
    )


def sample_id_from_vcf_path(path: Path) -> str:
    """
    Strip VCF-like suffixes from a path basename.

    Longer suffixes (``.g.vcf.gz``) are checked before shorter ones
    (``.vcf.gz``) so sample IDs are not left with a trailing ``.g``.
    """
    name = Path(path).name
    lower = name.lower()
    for suffix in (".g.vcf.gz", ".g.vcf", ".vcf.gz", ".vcf", ".bcf.gz", ".bcf"):
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return Path(path).stem


@dataclass
class _RefBlockIndex:
    """Per-contig gVCF interval index with insertion-order tie breaking."""

    starts: List[int]
    entries: List[Tuple[int, int, int, AlleleCall]]
    prefix_max_ends: List[int]

    def find(self, pos: int) -> Optional[AlleleCall]:
        """Return the first input block covering ``pos`` in near O(log n)."""
        idx = bisect_right(self.starts, int(pos)) - 1
        best_order: Optional[int] = None
        best_call: Optional[AlleleCall] = None

        # Normally gVCF blocks are disjoint, so this loop examines one entry.
        # The prefix maximum keeps overlapping/non-canonical inputs correct
        # without reverting every lookup to a full linear scan.
        while idx >= 0 and self.prefix_max_ends[idx] >= pos:
            start, end, input_order, template = self.entries[idx]
            if start <= pos <= end and (best_order is None or input_order < best_order):
                best_order = input_order
                best_call = template
            idx -= 1
        return best_call


@dataclass
class CallSet:
    """Parsed per-sample call set with optional gVCF reference blocks."""

    by_pos: Dict[Tuple[str, int], AlleleCall] = field(default_factory=dict)
    # (chrom, start, end, template_call) inclusive 1-based intervals
    ref_blocks: List[Tuple[str, int, int, AlleleCall]] = field(default_factory=list)
    sample_name: str = ""
    source_vcf: str = ""
    _ref_block_index: Dict[str, _RefBlockIndex] = field(
        default_factory=dict, repr=False
    )
    _ref_block_index_size: int = field(default=-1, repr=False)

    def __len__(self) -> int:
        return len(self.by_pos)

    def items(self):
        return self.by_pos.items()

    def get(self, key: Tuple[str, int], default: Any = None) -> Any:
        return self.by_pos.get(key, default)

    def keys(self):
        return self.by_pos.keys()

    def values(self):
        return self.by_pos.values()

    def __contains__(self, key: object) -> bool:
        return key in self.by_pos

    def build_ref_block_index(self) -> None:
        """Build one sorted interval index per contig."""
        grouped: Dict[str, List[Tuple[int, int, int, AlleleCall]]] = {}
        for input_order, (chrom, start, end, template) in enumerate(self.ref_blocks):
            grouped.setdefault(str(chrom), []).append(
                (int(start), int(end), int(input_order), template)
            )

        indexes: Dict[str, _RefBlockIndex] = {}
        for chrom, entries in grouped.items():
            entries.sort(key=lambda item: (item[0], item[2]))
            prefix_max_ends: List[int] = []
            running_max = -1
            for _, end, _, _ in entries:
                running_max = max(running_max, int(end))
                prefix_max_ends.append(running_max)
            indexes[chrom] = _RefBlockIndex(
                starts=[entry[0] for entry in entries],
                entries=entries,
                prefix_max_ends=prefix_max_ends,
            )

        self._ref_block_index = indexes
        self._ref_block_index_size = len(self.ref_blocks)

    def find_ref_block(self, chrom: str, pos: int) -> Optional[AlleleCall]:
        """Look up a covering gVCF block using the lazily built index."""
        if getattr(self, "_ref_block_index_size", -1) != len(
            self.ref_blocks
        ) or not hasattr(self, "_ref_block_index"):
            self.build_ref_block_index()
        index = self._ref_block_index.get(str(chrom))
        return index.find(int(pos)) if index is not None else None


def parse_vcf_calls(
    path: Union[str, Path],
    *,
    qc: Optional[VcfQCConfig] = None,
    ref_genome: Optional[Dict[str, str]] = None,
) -> CallSet:
    """
    Parse one VCF/VCF.GZ/gVCF into a CallSet.

    Explicit site records are stored in ``by_pos``. gVCF reference blocks
    (END > POS with callable reference GT) are stored as intervals and resolved
    on lookup without expanding every base of large genomes.
    """
    qc = qc or VcfQCConfig()
    path = Path(path)
    callset = CallSet(
        sample_name=normalize_sample_id(sample_id_from_vcf_path(path)),
        source_vcf=str(path),
    )
    sample_name = callset.sample_name

    with open_text_maybe_gzip(path) as handle:
        for raw in handle:
            if not raw:
                continue
            line = raw.rstrip("\n")
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                parts = line.split("\t")
                if len(parts) >= 10 and parts[9].strip():
                    header_sample = normalize_sample_id(parts[9].strip())
                    if header_sample:
                        sample_name = header_sample
                        callset.sample_name = sample_name
                continue
            if line.startswith("#"):
                continue

            parts = line.split("\t")
            call = classify_vcf_record(
                parts,
                qc=qc,
                sample_name=sample_name,
                source_vcf=str(path),
                ref_genome=ref_genome,
            )
            if call is None:
                continue

            if (
                qc.expand_gvcf_ref_blocks
                and call.is_gvcf_ref_block
                and call.end_pos
                and call.end_pos > call.pos
                and call.state == CallState.CALLED_REFERENCE
            ):
                callset.ref_blocks.append(
                    (call.chrom, call.pos, int(call.end_pos), call)
                )
                # Also record the block start coordinate as an explicit site.
                callset.by_pos[(call.chrom, call.pos)] = call
            else:
                callset.by_pos[(call.chrom, call.pos)] = call

    callset.build_ref_block_index()
    return callset


def _call_from_ref_block(
    callset: CallSet,
    chrom: str,
    pos: int,
) -> Optional[AlleleCall]:
    template = callset.find_ref_block(chrom, pos)
    if template is None:
        return None
    return AlleleCall(
        state=CallState.CALLED_REFERENCE,
        chrom=chrom,
        pos=pos,
        ref=template.ref,
        alts=list(template.alts),
        called_allele=template.ref
        if len(template.ref) == 1
        else template.called_allele,
        allele_index=0,
        filter_status=template.filter_status,
        qc_reasons=list(template.qc_reasons) + ["gvcf_ref_block"],
        ploidy=template.ploidy,
        is_phased=template.is_phased,
        is_gvcf_ref_block=True,
        end_pos=template.end_pos,
        sample_name=template.sample_name,
        source_vcf=template.source_vcf,
        gt_raw=template.gt_raw,
    )


def normalize_contig_alias_map(
    alias_map: Optional[Dict[str, str]],
) -> Dict[str, str]:
    """
    Validate contig alias map: query_contig → training/manifest contig.

    Ambiguous many-to-one mappings that collapse distinct query contigs onto
    the same target while also being invertible incorrectly are accepted as
    many-to-one (explicit collapse), but empty keys fail closed.
    """
    if not alias_map:
        return {}
    out: Dict[str, str] = {}
    for raw_src, raw_dst in alias_map.items():
        src = str(raw_src).strip()
        dst = str(raw_dst).strip()
        if not src or not dst:
            raise ValueError(
                "contig_alias_map entries must be non-empty strings "
                f"(got {raw_src!r} → {raw_dst!r})"
            )
        if src in out and out[src] != dst:
            raise ValueError(
                f"Ambiguous contig alias for {src!r}: {out[src]!r} vs {dst!r}"
            )
        out[src] = dst
    return out


def candidate_contigs_for_lookup(
    chrom: str,
    *,
    contig_alias_map: Optional[Dict[str, str]] = None,
) -> List[str]:
    """
    Ordered contig names to try for lookup: exact chrom, then aliases.

    Position-only (contig-ignoring) matching is intentionally not used.
    """
    chrom = str(chrom)
    aliases = normalize_contig_alias_map(contig_alias_map)
    candidates = [chrom]
    # Direct alias: query name → training name
    if chrom in aliases and aliases[chrom] not in candidates:
        candidates.append(aliases[chrom])
    # Reverse: if map stores training→query style, also try keys that map to chrom
    for src, dst in aliases.items():
        if dst == chrom and src not in candidates:
            candidates.append(src)
    return candidates


def lookup_call(
    calls: Union[CallSet, Dict[Tuple[str, int], AlleleCall]],
    chrom: str,
    pos: int,
    *,
    position_index: Optional[Dict[int, List[AlleleCall]]] = None,
    contig_alias_map: Optional[Dict[str, str]] = None,
    allow_position_only_match: bool = False,
) -> Tuple[Optional[AlleleCall], str]:
    """
    Look up a call by contig+position, with gVCF block coverage.

    Contig-name differences are resolved only through ``contig_alias_map``.
    Unsafe position-only matching across contigs is disabled by default and
    only available when ``allow_position_only_match=True`` (not recommended).

    Returns (call_or_None, match_mode).
    """
    pos = int(pos)
    by_pos = calls.by_pos if isinstance(calls, CallSet) else calls

    candidates = candidate_contigs_for_lookup(chrom, contig_alias_map=contig_alias_map)

    # Exact contig identity always wins. Alias candidates are evaluated as a
    # set so an explicit map cannot silently choose between two contigs that
    # both contain the same position.
    exact_call = by_pos.get((chrom, pos))
    if exact_call is not None:
        return exact_call, "exact_sequence_position"
    if isinstance(calls, CallSet):
        exact_block = _call_from_ref_block(calls, chrom, pos)
        if exact_block is not None:
            return exact_block, "gvcf_ref_block"

    alias_hits: List[Tuple[AlleleCall, str]] = []
    for cand in candidates[1:]:
        call = by_pos.get((cand, pos))
        if call is not None:
            alias_hits.append((call, "contig_alias"))
            continue
        if isinstance(calls, CallSet):
            block = _call_from_ref_block(calls, cand, pos)
            if block is not None:
                alias_hits.append((block, "gvcf_ref_block_contig_alias"))
    if len(alias_hits) == 1:
        return alias_hits[0]
    if len(alias_hits) > 1:
        return None, "ambiguous_contig_alias"

    if not allow_position_only_match:
        return None, "absent"

    # Explicit legacy path only — never the default.
    if position_index is not None:
        hits = position_index.get(pos, [])
    else:
        hits = [
            allele_call
            for (_, call_pos), allele_call in by_pos.items()
            if call_pos == pos
        ]
    if len(hits) == 1:
        return hits[0], "position_only_fallback"
    if len(hits) > 1:
        return None, "ambiguous_position_only"
    return None, "absent"


def build_position_index(
    calls: Union[CallSet, Dict[Tuple[str, int], AlleleCall]],
) -> Dict[int, List[AlleleCall]]:
    index: Dict[int, List[AlleleCall]] = {}
    items = calls.by_pos.items() if isinstance(calls, CallSet) else calls.items()
    for (chrom, pos), call in items:
        index.setdefault(int(pos), []).append(call)
    return index


_LEGACY_ABSENCE_WARNING_EMITTED = False


def warn_legacy_absence_assumed_reference() -> None:
    global _LEGACY_ABSENCE_WARNING_EMITTED
    msg = (
        "LEGACY CALLABILITY MODE: assume_absent_variant_is_reference=True. "
        "Sites absent from a variant-only VCF are encoded as reference (0). "
        "This is not demonstrated callability; prefer gVCF/reference blocks or "
        "depth/callability evidence. Set assume_absent_variant_is_reference=False "
        "for safe behaviour."
    )
    if not _LEGACY_ABSENCE_WARNING_EMITTED:
        warnings.warn(msg, UserWarning, stacklevel=2)
        logger.warning(msg)
        _LEGACY_ABSENCE_WARNING_EMITTED = True


def resolve_feature_call(
    *,
    chrom: str,
    pos: int,
    feature_ref: str,
    feature_alt: str,
    calls: Union[CallSet, Dict[Tuple[str, int], AlleleCall]],
    qc: VcfQCConfig,
    position_index: Optional[Dict[int, List[AlleleCall]]] = None,
    contig_alias_map: Optional[Dict[str, str]] = None,
    allow_position_only_match: bool = False,
) -> AlleleCall:
    """
    Resolve the call state for one trained feature coordinate.

    If no VCF record exists:
      - LOCUS_NOT_PRESENT (safe default), or
      - CALLED_REFERENCE with assumed_reference=True when legacy flag is set.
    """
    feature_ref = (feature_ref or "").upper()
    feature_alt = (feature_alt or "").upper()
    call, match = lookup_call(
        calls,
        chrom,
        pos,
        position_index=position_index,
        contig_alias_map=contig_alias_map,
        allow_position_only_match=allow_position_only_match,
    )

    if call is None and match in {"ambiguous_position_only", "ambiguous_contig_alias"}:
        return AlleleCall(
            state=CallState.UNRESOLVED_OR_AMBIGUOUS,
            chrom=chrom,
            pos=pos,
            ref=feature_ref,
            alts=[feature_alt] if feature_alt else [],
            qc_reasons=[
                "ambiguous_position_only_contig_match"
                if match == "ambiguous_position_only"
                else "ambiguous_explicit_contig_alias_match"
            ],
        )

    if call is None:
        if qc.assume_absent_variant_is_reference:
            warn_legacy_absence_assumed_reference()
            return AlleleCall(
                state=CallState.CALLED_REFERENCE,
                chrom=chrom,
                pos=pos,
                ref=feature_ref,
                alts=[feature_alt] if feature_alt else [],
                called_allele=feature_ref,
                allele_index=0,
                assumed_reference=True,
                qc_reasons=["absent_from_variant_vcf_assumed_reference"],
            )
        return AlleleCall(
            state=CallState.LOCUS_NOT_PRESENT,
            chrom=chrom,
            pos=pos,
            ref=feature_ref,
            alts=[feature_alt] if feature_alt else [],
            qc_reasons=["absent_from_variant_vcf_no_callability"],
        )

    # Propagate existing state; annotate match mode
    out = AlleleCall(
        state=call.state,
        chrom=call.chrom,
        pos=call.pos,
        ref=call.ref,
        alts=list(call.alts),
        called_allele=call.called_allele,
        allele_index=call.allele_index,
        filter_status=call.filter_status,
        qc_reasons=list(call.qc_reasons)
        + ([f"match={match}"] if match != "exact_sequence_position" else []),
        ploidy=call.ploidy,
        is_phased=call.is_phased,
        is_gvcf_ref_block=call.is_gvcf_ref_block,
        end_pos=call.end_pos,
        assumed_reference=call.assumed_reference,
        sample_name=call.sample_name,
        source_vcf=call.source_vcf,
        gt_raw=call.gt_raw,
    )
    return out


def encode_binary_for_feature(
    call: AlleleCall,
    *,
    feature_ref: str,
    feature_alt: str,
    baseline_allele: Optional[str] = None,
) -> Tuple[float, str]:
    """
    Encode one feature as binary value.

    Returns (value, allele_call_label) where value is 0.0, 1.0, or NaN.
    Missing/filtered/unresolved/absent are NEVER encoded as ordinary 0.
    Assumed reference (legacy) is 0.0 but allele_call marks assumed_reference.

    gVCF reference blocks demonstrate callability without a per-position allele
    string for interior bases; when a trained feature falls inside such a block
    it is encoded as callable baseline (0), not as the block-start REF base.
    """
    feature_ref = (feature_ref or "").upper()
    feature_alt = (feature_alt or "").upper()
    baseline = (baseline_allele or feature_ref or "").upper()

    if call.state == CallState.CALLED_REFERENCE:
        if call.assumed_reference:
            # Legacy absence→REF: encode as baseline only when assumed allele == baseline.
            allele = (call.called_allele or feature_ref or "").upper()
            if allele == baseline:
                return 0.0, "assumed_reference"
            if allele and allele != baseline:
                return 1.0, "assumed_reference_non_baseline"
            return 0.0, "assumed_reference"
        # gVCF reference blocks assert the genome is REF at this coordinate.
        # Encode relative to the stored feature baseline: if cohort-mode baseline
        # is ALT, callable REF is non-baseline (1), not always zero.
        if call.is_gvcf_ref_block or any(
            "gvcf_ref_block" in r for r in call.qc_reasons
        ):
            allele = (feature_ref or call.ref or "").upper()
        else:
            allele = (call.called_allele or call.ref or feature_ref or "").upper()
        if allele == baseline:
            return 0.0, "baseline_match"
        if allele == feature_alt:
            return 1.0, "alt_match"
        # Callable allele differs from baseline (e.g. REF when baseline is ALT).
        if allele and baseline and allele != baseline:
            return 1.0, "non_baseline_callable"
        return NON_CALLABLE, "callable_non_training_allele"

    if call.state == CallState.CALLED_ALTERNATE:
        allele = (call.called_allele or "").upper()
        if allele == feature_alt:
            return 1.0, "alt_match"
        if allele == baseline or allele == feature_ref:
            return 0.0, "baseline_match"
        return NON_CALLABLE, "non_training_allele"

    if call.state == CallState.MISSING_OR_NO_CALL:
        return NON_CALLABLE, "missing_or_no_call"
    if call.state == CallState.FILTERED_OR_LOW_QUALITY:
        return NON_CALLABLE, "filtered_or_low_quality"
    if call.state == CallState.LOCUS_NOT_PRESENT:
        return NON_CALLABLE, "locus_not_present"
    return NON_CALLABLE, "unresolved_or_ambiguous"


def is_callable_state(state: CallState) -> bool:
    return state in {CallState.CALLED_REFERENCE, CallState.CALLED_ALTERNATE}


def summarize_call_states(states: Iterable[CallState]) -> Dict[str, int]:
    counts: Dict[str, int] = {s.value: 0 for s in CallState}
    n = 0
    for st in states:
        counts[st.value if isinstance(st, CallState) else str(st)] = (
            counts.get(st.value if isinstance(st, CallState) else str(st), 0) + 1
        )
        n += 1
    counts["total"] = n
    return counts


def recovery_and_callability(
    states: Sequence[CallState],
) -> Dict[str, float]:
    n = max(1, len(states))
    callable_n = sum(1 for s in states if is_callable_state(s))
    resolved_n = callable_n  # explicit calls only
    absent_n = sum(1 for s in states if s == CallState.LOCUS_NOT_PRESENT)
    missing_n = sum(1 for s in states if s == CallState.MISSING_OR_NO_CALL)
    filtered_n = sum(1 for s in states if s == CallState.FILTERED_OR_LOW_QUALITY)
    unresolved_n = sum(1 for s in states if s == CallState.UNRESOLVED_OR_AMBIGUOUS)
    assumed_n = 0  # filled by caller if tracking assumed_reference flags
    return {
        "n_features": float(len(states)),
        "n_callable": float(callable_n),
        "n_resolved": float(resolved_n),
        "callable_fraction": float(callable_n / n),
        "resolved_fraction": float(resolved_n / n),
        "absent_fraction": float(absent_n / n),
        "missing_fraction": float(missing_n / n),
        "filtered_fraction": float(filtered_n / n),
        "unresolved_fraction": float(unresolved_n / n),
        "assumed_reference_fraction": float(assumed_n / n),
    }


def callability_gate_result(
    states: Sequence[CallState],
    *,
    qc: VcfQCConfig,
    n_assumed_reference: int = 0,
) -> Dict[str, Any]:
    """Decide whether query recovery/callability is sufficient for prediction."""
    stats = recovery_and_callability(states)
    n = max(1, len(states))
    stats["assumed_reference_fraction"] = float(n_assumed_reference / n)
    stats["n_assumed_reference"] = float(n_assumed_reference)

    # Recovery = fraction with any explicit VCF evidence at the coordinate
    # (callable, missing GT, filtered, unresolved) — not LOCUS_NOT_PRESENT.
    present_n = sum(1 for s in states if s != CallState.LOCUS_NOT_PRESENT)
    recovery_fraction = float(present_n / n)
    stats["feature_recovery_fraction"] = recovery_fraction

    ok_recovery = recovery_fraction >= float(qc.min_feature_recovery_fraction)
    ok_callable = stats["callable_fraction"] >= float(qc.min_callable_fraction)
    passed = bool(ok_recovery and ok_callable)

    if not qc.enforce_query_callability_gates:
        status = "callability_gates_disabled"
        passed = True
    elif passed:
        status = "adequate_callability"
    else:
        status = "insufficient_callability_abstain"

    return {
        **stats,
        "gate_passed": passed,
        "gate_status": status,
        "min_feature_recovery_fraction": float(qc.min_feature_recovery_fraction),
        "min_callable_fraction": float(qc.min_callable_fraction),
        "prediction_action": "predict" if passed else "abstain_review_unresolved",
    }


def detect_duplicate_sample_ids(sample_ids: Sequence[str]) -> List[str]:
    """Return sorted list of normalized sample IDs that appear more than once."""
    counts: Dict[str, int] = {}
    for s in sample_ids:
        key = str(s).strip()
        counts[key] = counts.get(key, 0) + 1
    return sorted([k for k, c in counts.items() if k and c > 1])


def assert_unique_sample_ids(
    sample_ids: Sequence[str], *, context: str = "samples"
) -> None:
    dups = detect_duplicate_sample_ids(sample_ids)
    if dups:
        raise ValueError(
            f"Duplicate sample IDs detected in {context}: {dups[:20]}"
            + (" ..." if len(dups) > 20 else "")
            + ". Refusing to overwrite or silently merge samples."
        )


# ---------------------------------------------------------------------------
# Backward-compatible thin wrappers used by older call sites
# ---------------------------------------------------------------------------


def choose_called_allele(
    ref: str,
    alts: List[str],
    fmt: Optional[str],
    sample_field: Optional[str],
    *,
    qc: Optional[VcfQCConfig] = None,
) -> Tuple[Optional[str], CallState]:
    """
    Shared replacement for legacy choose_called_allele.

    Returns (called_allele_or_None, state). Never invents ALT for missing GT.
    """
    qc = qc or VcfQCConfig()
    sample_map = parse_sample_fields(fmt, sample_field)
    state, called, _, _ = classify_genotype(
        ref=ref,
        alts=alts,
        gt_raw=sample_map.get("GT", ""),
        sample_map=sample_map,
        fmt_present=bool(fmt),
        sample_field_present=bool(sample_field) and sample_field != ".",
        supported_ploidies=qc.supported_ploidies,
    )
    return called, state


def iter_sample_calls(
    vcf_path: Path,
    *,
    qc: Optional[VcfQCConfig] = None,
    qual_thresh: Optional[float] = None,
    dp_thresh: Optional[int] = None,
    mq_thresh: Optional[float] = None,
    mq0f_thresh: Optional[float] = None,
    biallelic_only: bool = True,
    gq_thresh: Optional[int] = None,
) -> CallSet:
    """
    Parse one sample VCF into a CallSet (shared training path).

    Legacy keyword thresholds override qc fields when provided.
    """
    base = qc or VcfQCConfig()
    cfg = VcfQCConfig(
        qual_threshold=float(
            qual_thresh if qual_thresh is not None else base.qual_threshold
        ),
        min_dp=int(dp_thresh if dp_thresh is not None else base.min_dp),
        min_gq=int(gq_thresh if gq_thresh is not None else base.min_gq),
        mq_threshold=float(mq_thresh if mq_thresh is not None else base.mq_threshold),
        mq0f_threshold=float(
            mq0f_thresh if mq0f_thresh is not None else base.mq0f_threshold
        ),
        biallelic_only=bool(biallelic_only),
        supported_ploidies=base.supported_ploidies,
        respect_filter=base.respect_filter,
        allowed_filters=base.allowed_filters,
        assume_absent_variant_is_reference=base.assume_absent_variant_is_reference,
        validate_ref_against_genome=base.validate_ref_against_genome,
        expand_gvcf_ref_blocks=base.expand_gvcf_ref_blocks,
        min_feature_recovery_fraction=base.min_feature_recovery_fraction,
        min_callable_fraction=base.min_callable_fraction,
        enforce_query_callability_gates=base.enforce_query_callability_gates,
        contig_alias_map=base.contig_alias_map,
        allow_position_only_match=base.allow_position_only_match,
    )
    return parse_vcf_calls(vcf_path, qc=cfg)
