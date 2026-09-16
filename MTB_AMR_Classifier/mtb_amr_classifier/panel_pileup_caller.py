#!/usr/bin/env python3
"""
Panel-restricted pileup majority / median allele calling.

For NetworkParser FASTQ query, only trained marker positions are genotyped:

    BAM + panel sites → samtools mpileup → majority/median base → compact VCF

This avoids whole-genome bcftools calling of sites that will never be used.
Haploid (microbial) default: one allele state per position.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_FEATURE_RE = re.compile(
    r"^(?P<contig>[^:]+):(?P<pos>\d+):(?P<ref>[ACGTNacgtn]+):(?P<alt>[ACGTNacgtn.*]+)$"
)


@dataclass(frozen=True)
class PanelSite:
    contig: str
    position: int  # 1-based
    ref: str
    alt: str
    feature_id: str

    @property
    def is_snp(self) -> bool:
        return len(self.ref) == 1 and len(self.alt) == 1 and self.alt not in {".", "*"}


@dataclass
class AlleleCall:
    contig: str
    position: int
    ref: str
    called_base: Optional[str]
    depth: int
    counts: Dict[str, int]
    majority_fraction: float
    status: str  # called_ref | called_alt | low_depth | ambiguous | no_coverage


def parse_feature_id(feature_id: str) -> Optional[PanelSite]:
    fid = str(feature_id).strip()
    m = _FEATURE_RE.match(fid)
    if not m:
        return None
    return PanelSite(
        contig=m.group("contig"),
        position=int(m.group("pos")),
        ref=m.group("ref").upper(),
        alt=m.group("alt").upper(),
        feature_id=fid,
    )


def load_panel_sites_from_feature_ids(
    feature_ids: Sequence[str],
) -> List[PanelSite]:
    sites: List[PanelSite] = []
    seen = set()
    skipped = 0
    for fid in feature_ids:
        site = parse_feature_id(fid)
        if site is None:
            skipped += 1
            continue
        key = (site.contig, site.position, site.ref, site.alt)
        if key in seen:
            continue
        seen.add(key)
        sites.append(site)
    if skipped:
        logger.warning(
            "Skipped %d feature IDs that do not look like Contig:Pos:Ref:Alt",
            skipped,
        )
    if not sites:
        raise ValueError("No valid panel sites parsed from feature IDs")
    return sites


def load_panel_sites_from_manifest_tsv(path: Path) -> List[PanelSite]:
    """Load sites from a selected-feature manifest TSV (Feature_ID column)."""
    import csv

    path = Path(path)
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"Empty manifest: {path}")
        # Flexible column names
        fid_col = None
        for c in reader.fieldnames:
            if c.lower() in {"feature_id", "feature", "id"}:
                fid_col = c
                break
        if fid_col is None and "Feature_ID" in reader.fieldnames:
            fid_col = "Feature_ID"
        if fid_col is None:
            # try first column
            fid_col = reader.fieldnames[0]
        ids = [row[fid_col] for row in reader if row.get(fid_col)]
    return load_panel_sites_from_feature_ids(ids)


def load_panel_sites_from_bed(path: Path, default_ref: str = "N") -> List[PanelSite]:
    """BED (0-based) with optional name column Contig:Pos:Ref:Alt."""
    sites: List[PanelSite] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            contig = parts[0]
            start0 = int(parts[1])
            end1 = int(parts[2])
            pos = end1  # 1-based end for single-base
            name = parts[3] if len(parts) > 3 else ""
            site = parse_feature_id(name) if name else None
            if site is None:
                # single-base bed without alleles — ref/alt unknown; use N/.
                site = PanelSite(
                    contig=contig,
                    position=pos if end1 - start0 == 1 else start0 + 1,
                    ref=default_ref,
                    alt=".",
                    feature_id=f"{contig}:{pos}:{default_ref}:.",
                )
            sites.append(site)
    if not sites:
        raise ValueError(f"No sites loaded from BED: {path}")
    return sites


def write_panel_bed(sites: Sequence[PanelSite], bed_path: Path) -> Path:
    """Write 0-based BED with feature id in name column (unique positions)."""
    bed_path = Path(bed_path)
    bed_path.parent.mkdir(parents=True, exist_ok=True)
    # unique by contig+position for pileup; keep first feature name
    uniq: Dict[Tuple[str, int], PanelSite] = {}
    for s in sites:
        key = (s.contig, s.position)
        if key not in uniq:
            uniq[key] = s
    with bed_path.open("w") as f:
        for (contig, pos), s in sorted(uniq.items(), key=lambda x: (x[0][0], x[0][1])):
            f.write(f"{contig}\t{pos - 1}\t{pos}\t{s.feature_id}\n")
    logger.info("Wrote panel BED with %d unique positions → %s", len(uniq), bed_path)
    return bed_path


def _parse_pileup_bases(raw: str, ref: str) -> List[str]:
    """Expand samtools mpileup bases field into a list of ACGT alleles."""
    bases: List[str] = []
    ref_u = (ref or "N").upper()
    i = 0
    s = raw
    n = len(s)
    while i < n:
        c = s[i]
        if c == "^":
            # start of read + mapping quality char
            i += 2
            continue
        if c == "$":
            i += 1
            continue
        if c in "+-":
            # indel: +3ACG / -2AT
            i += 1
            num = ""
            while i < n and s[i].isdigit():
                num += s[i]
                i += 1
            skip = int(num) if num else 0
            i += skip
            continue
        if c == "*":
            # deleted base placeholder
            i += 1
            continue
        if c in ".,":
            bases.append(ref_u)
            i += 1
            continue
        if c in "ACGTNacgtn":
            bases.append(c.upper())
            i += 1
            continue
        # ignore other chars
        i += 1
    return bases


def majority_call_from_bases(
    bases: Sequence[str],
    *,
    min_depth: int = 10,
    min_majority_fraction: float = 0.7,
) -> Tuple[Optional[str], int, Dict[str, int], float, str]:
    """
    Median/majority allele from pileup bases.

    For odd depth n, the median is the ((n+1)/2)-th base in sorted order
    among ACGT counts we use majority base (equivalent for single-base alleles).
    """
    filtered = [b for b in bases if b in {"A", "C", "G", "T"}]
    depth = len(filtered)
    counts = dict(Counter(filtered))
    if depth == 0:
        return None, 0, counts, 0.0, "no_coverage"
    if depth < min_depth:
        # still report majority for diagnostics
        base, n = Counter(filtered).most_common(1)[0]
        frac = n / depth
        return base, depth, counts, frac, "low_depth"
    base, n = Counter(filtered).most_common(1)[0]
    frac = n / depth
    if frac < min_majority_fraction:
        return base, depth, counts, frac, "ambiguous"
    return base, depth, counts, frac, "called"


def call_panel_from_bam(
    *,
    bam_path: Path,
    ref_fasta: Path,
    sites: Sequence[PanelSite],
    bed_path: Path,
    min_mapping_quality: int = 20,
    min_base_quality: int = 20,
    min_depth: int = 10,
    min_majority_fraction: float = 0.7,
) -> Dict[Tuple[str, int], AlleleCall]:
    """Run samtools mpileup on panel BED and majority-call each position."""
    if shutil.which("samtools") is None:
        raise RuntimeError("samtools not found on PATH (required for panel pileup calling)")

    write_panel_bed(sites, bed_path)
    # Map position -> representative ref (first site at that pos)
    ref_at: Dict[Tuple[str, int], str] = {}
    for s in sites:
        key = (s.contig, s.position)
        if key not in ref_at:
            ref_at[key] = s.ref

    cmd = [
        "samtools",
        "mpileup",
        "-f",
        str(ref_fasta),
        "-l",
        str(bed_path),
        "-q",
        str(int(min_mapping_quality)),
        "-Q",
        str(int(min_base_quality)),
        "-x",  # no overlapping read fragment double-count when possible
        str(bam_path),
    ]
    logger.info("Panel pileup: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode not in (0,):
        # samtools mpileup sometimes returns 0 with empty; non-zero is real failure
        if proc.returncode != 0 and not proc.stdout.strip():
            raise RuntimeError(
                f"samtools mpileup failed ({proc.returncode}): {proc.stderr[-2000:]}"
            )

    calls: Dict[Tuple[str, int], AlleleCall] = {}
    for line in proc.stdout.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        contig, pos_s, ref, depth_s, base_str = (
            parts[0],
            parts[1],
            parts[2],
            parts[3],
            parts[4],
        )
        pos = int(pos_s)
        key = (contig, pos)
        ref_u = (ref_at.get(key) or ref or "N").upper()
        bases = _parse_pileup_bases(base_str, ref_u)
        called, depth, counts, frac, status = majority_call_from_bases(
            bases,
            min_depth=min_depth,
            min_majority_fraction=min_majority_fraction,
        )
        if status == "called":
            status = "called_ref" if called == ref_u else "called_alt"
        calls[key] = AlleleCall(
            contig=contig,
            position=pos,
            ref=ref_u,
            called_base=called,
            depth=depth,
            counts=counts,
            majority_fraction=frac,
            status=status,
        )

    # positions with no pileup line → no_coverage
    for s in sites:
        key = (s.contig, s.position)
        if key not in calls:
            calls[key] = AlleleCall(
                contig=s.contig,
                position=s.position,
                ref=s.ref,
                called_base=None,
                depth=0,
                counts={},
                majority_fraction=0.0,
                status="no_coverage",
            )
    return calls


def write_calls_vcf(
    *,
    calls: Dict[Tuple[str, int], AlleleCall],
    sites: Sequence[PanelSite],
    sample_id: str,
    ref_fasta: Path,
    output_vcf: Path,
    emit_reference_sites: bool = True,
) -> Path:
    """
    Write a minimal VCF for NetworkParser encoding.

    - ALT calls: REF/ALT with GT=1
    - REF calls: if emit_reference_sites, GT=0 at panel positions
    - Unresolved: omitted (encoder may treat as missing / absence-as-ref)
    """
    output_vcf = Path(output_vcf)
    output_vcf.parent.mkdir(parents=True, exist_ok=True)

    # Group panel alts by position for multi-allelic features
    alts_at: Dict[Tuple[str, int], List[PanelSite]] = {}
    for s in sites:
        alts_at.setdefault((s.contig, s.position), []).append(s)

    contigs = sorted({s.contig for s in sites})
    lines: List[str] = [
        "##fileformat=VCFv4.2",
        f"##reference={ref_fasta}",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">',
        '##FORMAT=<ID=ADF,Number=1,Type=Float,Description="Majority allele fraction">',
        '##INFO=<ID=PANEL,Number=0,Type=Flag,Description="NetworkParser panel site">',
        '##INFO=<ID=STATUS,Number=1,Type=String,Description="pileup majority status">',
    ]
    for c in contigs:
        lines.append(f"##contig=<ID={c}>")
    lines.append(
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + sample_id
    )

    for key in sorted(calls.keys(), key=lambda x: (x[0], x[1])):
        call = calls[key]
        panel_here = alts_at.get(key, [])
        ref = call.ref
        status = call.status
        info = f"PANEL;STATUS={status}"
        dp = int(call.depth)
        adf = f"{call.majority_fraction:.4f}"

        if status in {"no_coverage", "low_depth", "ambiguous"}:
            # skip — leave missing for encoder / absence-as-ref policy
            continue

        called = (call.called_base or "").upper()
        if called == ref:
            if not emit_reference_sites:
                continue
            # represent REF with ALT=. and GT=0 (some parsers prefer no row;
            # we emit GT=0/REF for explicitness when emit_reference_sites)
            # VCF requires ALT; use symbolic .
            lines.append(
                f"{call.contig}\t{call.position}\t.\t{ref}\t.\t.\tPASS\t{info}\tGT:DP:ADF\t0:{dp}:{adf}"
            )
            continue

        # Match called base to a panel ALT at this position if possible
        matching = [s for s in panel_here if s.alt == called and s.is_snp]
        if matching:
            alt = matching[0].alt
            fid = matching[0].feature_id
        else:
            # non-panel allele at panel coordinate
            alt = called
            fid = f"{call.contig}:{call.position}:{ref}:{alt}"
            info += ";NON_PANEL_ALT"

        lines.append(
            f"{call.contig}\t{call.position}\t{fid}\t{ref}\t{alt}\t.\tPASS\t{info}\tGT:DP:ADF\t1:{dp}:{adf}"
        )

    raw = output_vcf.with_suffix("")  # strip .gz if present carefully
    if str(output_vcf).endswith(".vcf.gz"):
        text_vcf = Path(str(output_vcf)[:-3])  # .vcf
    else:
        text_vcf = output_vcf

    text_vcf.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # bgzip + index if bcftools/bgzip available
    if str(output_vcf).endswith(".vcf.gz"):
        if shutil.which("bgzip"):
            subprocess_bgzip = __import__("subprocess")
            subprocess_bgzip.run(
                ["bgzip", "-f", str(text_vcf)], check=True, capture_output=True
            )
            out = Path(str(text_vcf) + ".gz")
            if out != output_vcf:
                out.replace(output_vcf)
        elif shutil.which("bcftools"):
            import subprocess as sp

            sp.run(
                ["bcftools", "view", "-Oz", "-o", str(output_vcf), str(text_vcf)],
                check=True,
                capture_output=True,
            )
            text_vcf.unlink(missing_ok=True)
        else:
            # leave uncompressed; rename expectation
            logger.warning(
                "bgzip/bcftools not available; wrote uncompressed VCF %s", text_vcf
            )
            return text_vcf

        if shutil.which("bcftools") and Path(output_vcf).exists():
            import subprocess as sp

            sp.run(
                ["bcftools", "index", "-t", "-f", str(output_vcf)],
                check=False,
                capture_output=True,
            )
        return Path(output_vcf)

    return text_vcf


def genotype_bam_to_panel_vcf(
    *,
    bam_path: Path,
    ref_fasta: Path,
    sites: Sequence[PanelSite],
    output_vcf: Path,
    work_dir: Path,
    sample_id: str,
    min_mapping_quality: int = 20,
    min_base_quality: int = 20,
    min_depth: int = 10,
    min_majority_fraction: float = 0.7,
    emit_reference_sites: bool = False,
) -> Path:
    """End-to-end: BAM + panel → majority-call VCF.GZ."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    bed = work_dir / f"{sample_id}.panel.bed"
    calls = call_panel_from_bam(
        bam_path=Path(bam_path),
        ref_fasta=Path(ref_fasta),
        sites=sites,
        bed_path=bed,
        min_mapping_quality=min_mapping_quality,
        min_base_quality=min_base_quality,
        min_depth=min_depth,
        min_majority_fraction=min_majority_fraction,
    )
    n_alt = sum(1 for c in calls.values() if c.status == "called_alt")
    n_ref = sum(1 for c in calls.values() if c.status == "called_ref")
    n_miss = sum(
        1
        for c in calls.values()
        if c.status in {"no_coverage", "low_depth", "ambiguous"}
    )
    logger.info(
        "Panel majority calls | sample=%s | alt=%d ref=%d unresolved=%d total_pos=%d",
        sample_id,
        n_alt,
        n_ref,
        n_miss,
        len(calls),
    )
    return write_calls_vcf(
        calls=calls,
        sites=sites,
        sample_id=sample_id,
        ref_fasta=Path(ref_fasta),
        output_vcf=Path(output_vcf),
        emit_reference_sites=emit_reference_sites,
    )
