#!/usr/bin/env python3
# network_parser/fastq_processor.py
"""
FASTQ-to-VCF preprocessing for NetworkParser query mode.

Purpose
-------
Convert paired-end FASTQ reads into per-sample VCF.GZ files that can be
consumed by the existing DataLoader VCF-directory pathway.

This module does NOT perform statistical filtering, model training,
decision-tree construction, bootstrapping, or confidence scoring. It is a
query-time preprocessing bridge:

    FASTQ reads -> alignment -> sorted BAM -> VCF.GZ -> DataLoader -> query matrix

External command-line tools are required on PATH:
    - bwa
    - samtools
    - bcftools
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import re
import shlex
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from mtb_amr_classifier.utils import progress_iter
except ImportError:  # pragma: no cover - package vs source-tree layout
    try:
        from utils import progress_iter  # type: ignore
    except ImportError:  # pragma: no cover

        def progress_iter(iterable, **kwargs):  # type: ignore
            return iterable


FASTQ_EXTENSIONS = (
    ".fastq.gz",
    ".fq.gz",
    ".fastq",
    ".fq",
)


@dataclass
class FastqProcessingSummary:
    """Compact provenance summary for one FASTQ preprocessing run."""

    status: str
    input_fastq_dir: str
    reference_genome: str
    output_dir: str
    final_vcf_dir: str
    discovered_pairs: int
    successful_samples: int
    failed_samples: int
    successful_vcfs: List[str]
    failed_sample_errors: Dict[str, str]
    tool_versions: Dict[str, str]
    max_parallel_samples: int
    threads_total: int
    threads_per_sample: int


class FastqProcessor:
    """
    Convert paired FASTQ files into per-sample VCF.GZ files.

    Parameters
    ----------
    config
        NetworkParserConfig-like object. Only FASTQ-related attributes are read;
        sensible defaults are used when fields are absent.
    fastq_dir
        Directory containing paired FASTQ files.
    ref_genome
        Reference FASTA used for alignment and variant calling.
    output_dir
        Output directory for FASTQ preprocessing artifacts.
    n_jobs
        Optional total thread override. If absent, config.fastq_threads or
        config.n_jobs is used.
    """

    def __init__(
        self,
        config: Any,
        fastq_dir: str,
        ref_genome: str,
        output_dir: str,
        n_jobs: Optional[int] = None,
        panel_feature_ids: Optional[List[str]] = None,
    ) -> None:
        self.config = config
        self.fastq_dir = Path(fastq_dir)
        self.ref_genome = Path(ref_genome)
        self.output_dir = Path(output_dir)
        # Optional trained-marker feature IDs (Contig:Pos:Ref:Alt) for panel calling.
        self.panel_feature_ids: Optional[List[str]] = (
            list(panel_feature_ids) if panel_feature_ids else None
        )

        self.max_parallel_samples = int(
            getattr(config, "fastq_max_parallel_samples", 1) or 1
        )
        self.max_parallel_samples = max(1, self.max_parallel_samples)

        self.total_threads = self._resolve_total_threads(n_jobs=n_jobs)
        self.threads_per_sample = max(
            1, self.total_threads // self.max_parallel_samples
        )

        self.memory_per_sample_mb = getattr(config, "fastq_memory_per_sample_mb", None)
        if self.memory_per_sample_mb is not None:
            self.memory_per_sample_mb = int(self.memory_per_sample_mb)
            if self.memory_per_sample_mb < 256:
                raise ValueError("fastq_memory_per_sample_mb must be >= 256 or None")

        self.clean_intermediates = bool(
            getattr(config, "fastq_clean_intermediates", False)
        )
        self.auto_index_reference = bool(
            getattr(config, "fastq_auto_index_reference", True)
        )
        self.min_mapping_quality = int(getattr(config, "fastq_min_mapping_quality", 20))
        self.sample_platform = str(
            getattr(config, "fastq_sample_platform", "ILLUMINA") or "ILLUMINA"
        )
        # Legacy alias: fastq_sort_memory → per-thread if new key absent
        self.sort_memory_per_thread = str(
            getattr(config, "fastq_sort_memory_per_thread", None)
            or getattr(config, "fastq_sort_memory", "512M")
            or "512M"
        )
        self.sort_memory = self.sort_memory_per_thread  # backward-compatible attr
        # Prefer gVCF / callable reference so absence is not treated as REF later.
        self.emit_gvcf = bool(getattr(config, "fastq_emit_gvcf", True))
        self.gvcf_min_dp = int(getattr(config, "fastq_gvcf_min_dp", 10) or 10)
        multi_lane = (
            str(getattr(config, "fastq_multi_lane_policy", "fail") or "fail")
            .strip()
            .lower()
        )
        if multi_lane not in {"fail", "merge"}:
            raise ValueError("fastq_multi_lane_policy must be 'fail' or 'merge'")
        self.multi_lane_policy = multi_lane
        self.ploidy = int(getattr(config, "fastq_ploidy", 1) or 1)
        if self.ploidy not in {1, 2}:
            raise ValueError("fastq_ploidy must be 1 (haploid, microbial default) or 2")
        self.write_alignment_stats = bool(
            getattr(config, "fastq_write_alignment_stats", True)
        )
        self.normalize_vcf = bool(getattr(config, "fastq_normalize_vcf", True))
        self.call_mode = str(
            getattr(config, "fastq_call_mode", "full") or "full"
        ).strip().lower()
        if self.call_mode not in {"full", "panel_bcftools", "panel_majority"}:
            raise ValueError(
                "fastq_call_mode must be one of: full, panel_bcftools, panel_majority"
            )
        self.panel_sites_bed = getattr(config, "fastq_panel_sites_bed", None)
        self.panel_manifest = getattr(config, "fastq_panel_manifest", None)
        self.panel_min_depth = int(getattr(config, "fastq_panel_min_depth", 10) or 10)
        self.panel_min_majority_fraction = float(
            getattr(config, "fastq_panel_min_majority_fraction", 0.7) or 0.7
        )
        self.panel_min_base_quality = int(
            getattr(config, "fastq_panel_min_base_quality", 20) or 20
        )
        self.panel_emit_reference_sites = bool(
            getattr(config, "fastq_panel_emit_reference_sites", False)
        )
        self._panel_sites = None  # lazy-loaded

        # Concurrent pipeline stages share one per-sample CPU budget.
        (
            self.bwa_threads,
            self.view_threads,
            self.sort_threads,
        ) = self._allocate_stage_threads(self.threads_per_sample)

        self.intermediate_dir = self.output_dir / "intermediate"
        self.bam_dir = self.output_dir / "bams"
        self.vcf_dir = self.output_dir / "final" / "vcf"
        self.stats_dir = self.output_dir / "stats"
        self.logs_dir = self.output_dir / "logs"
        self.provenance_dir = self.output_dir / "provenance"

        self.failed_sample_errors: Dict[str, str] = {}
        self.tool_versions: Dict[str, str] = {}
        self.command_log: List[Dict[str, Any]] = []

        self._validate_inputs()
        self._setup_directory_structure()
        self._check_required_tools()
        self._prepare_reference_indexes()

        logger.info(
            "FASTQ processor initialized | fastq_dir=%s | ref=%s | call_mode=%s | "
            "parallel_samples=%d | total_threads=%d | threads_per_sample=%d",
            self.fastq_dir,
            self.ref_genome,
            self.call_mode,
            self.max_parallel_samples,
            self.total_threads,
            self.threads_per_sample,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_samples(self) -> Tuple[Path, FastqProcessingSummary]:
        """
        Process all paired FASTQ samples.

        Returns
        -------
        tuple
            (final_vcf_dir, summary)
        """
        fastq_pairs = self.find_fastq_pairs()
        if not fastq_pairs:
            raise RuntimeError(f"No valid paired FASTQ files found in {self.fastq_dir}")

        logger.info(
            "FASTQ preprocessing started | pairs=%d | output_vcf_dir=%s",
            len(fastq_pairs),
            self.vcf_dir,
        )

        successful_vcfs: List[Path] = []
        with ThreadPoolExecutor(max_workers=self.max_parallel_samples) as executor:
            futures = {
                executor.submit(self._process_one_sample, r1, r2, sample): sample
                for r1, r2, sample in fastq_pairs
            }

            for future in progress_iter(
                as_completed(futures),
                total=len(futures),
                desc="FASTQ preprocessing",
                unit="sample",
                leave=False,
            ):
                sample = futures[future]
                try:
                    vcf_path = future.result()
                    successful_vcfs.append(vcf_path)
                    logger.info(
                        "FASTQ sample complete | sample=%s | vcf=%s", sample, vcf_path
                    )
                except Exception as exc:
                    self.failed_sample_errors[sample] = str(exc)
                    logger.error(
                        "FASTQ sample failed | sample=%s | error=%s", sample, exc
                    )

        if not successful_vcfs:
            raise RuntimeError(
                "FASTQ preprocessing failed for every sample; no VCFs were produced."
            )

        summary = FastqProcessingSummary(
            status="success" if not self.failed_sample_errors else "partial_success",
            input_fastq_dir=str(self.fastq_dir),
            reference_genome=str(self.ref_genome),
            output_dir=str(self.output_dir),
            final_vcf_dir=str(self.vcf_dir),
            discovered_pairs=int(len(fastq_pairs)),
            successful_samples=int(len(successful_vcfs)),
            failed_samples=int(len(self.failed_sample_errors)),
            successful_vcfs=[str(p) for p in sorted(successful_vcfs)],
            failed_sample_errors=dict(self.failed_sample_errors),
            tool_versions=dict(self.tool_versions),
            max_parallel_samples=int(self.max_parallel_samples),
            threads_total=int(self.total_threads),
            threads_per_sample=int(self.threads_per_sample),
        )
        summary_dict = asdict(summary)
        summary_dict["provenance"] = {
            "tool_versions": dict(self.tool_versions),
            "ploidy": int(self.ploidy),
            "call_mode": str(self.call_mode),
            "panel_n_sites": (
                len(self._panel_sites) if self._panel_sites is not None else None
            ),
            "emit_gvcf": bool(self.emit_gvcf),
            "gvcf_min_dp": int(self.gvcf_min_dp),
            "normalize_vcf": bool(self.normalize_vcf),
            "write_alignment_stats": bool(self.write_alignment_stats),
            "panel_min_depth": int(self.panel_min_depth),
            "panel_min_majority_fraction": float(self.panel_min_majority_fraction),
            "cpu_budget_per_sample": {
                "threads_per_sample": int(self.threads_per_sample),
                "bwa_threads": int(self.bwa_threads),
                "samtools_view_threads": int(self.view_threads),
                "samtools_sort_threads": int(self.sort_threads),
                "sort_memory_per_thread": str(self.sort_memory_per_thread),
                "sort_memory_note": (
                    "samtools sort -m is memory per sort thread; "
                    f"approx total sort RAM ≈ {self.sort_threads} × {self.sort_memory_per_thread}"
                ),
                "concurrent_stage_sum": int(
                    self.bwa_threads + self.view_threads + self.sort_threads
                ),
            },
            "commands": list(self.command_log),
            "reference_fasta": str(self.ref_genome),
            "multi_lane_policy": str(self.multi_lane_policy),
        }
        self._write_json(
            summary_dict, self.output_dir / "fastq_processing_summary.json"
        )
        self._write_json(
            summary_dict["provenance"],
            self.provenance_dir / "fastq_provenance.json",
        )

        logger.info(
            "FASTQ preprocessing complete | successful=%d | failed=%d | vcf_dir=%s",
            summary.successful_samples,
            summary.failed_samples,
            summary.final_vcf_dir,
        )
        return self.vcf_dir, summary

    def find_fastq_pairs(self) -> List[Tuple[Path, Path, str]]:
        """
        Find paired-end FASTQ files using common R1/R2 naming patterns.

        Multi-lane R1/R2 files are paired by **parsed lane key** (not by
        independently sorting R1 and R2 path lists, which can cross-pair lanes).

        Policy (config.fastq_multi_lane_policy):
          - fail: raise with a clear listing of colliding lane files
          - merge: concatenate lanes in sorted lane-key order
        """
        files = [
            p for p in self.fastq_dir.iterdir() if p.is_file() and self._is_fastq(p)
        ]
        # sample -> lane_key -> {"R1": Path, "R2": Path}
        sample_lanes: Dict[str, Dict[str, Dict[str, Path]]] = {}

        for path in files:
            parsed = self._parse_fastq_name(path.name)
            if parsed is None:
                logger.debug(
                    "Ignoring FASTQ file with unsupported pair naming: %s", path.name
                )
                continue
            sample, read, lane_key = parsed
            lane_map = sample_lanes.setdefault(sample, {})
            slot = lane_map.setdefault(lane_key, {})
            if read in slot:
                raise RuntimeError(
                    f"Duplicate {read} FASTQ for sample={sample!r} lane={lane_key!r}: "
                    f"{slot[read]} and {path}"
                )
            slot[read] = path

        multi_lane_samples = sorted(
            s for s, lanes in sample_lanes.items() if len(lanes) > 1
        )
        if multi_lane_samples and self.multi_lane_policy == "fail":
            details = []
            for sample in multi_lane_samples:
                lane_bits = []
                for lane_key in sorted(sample_lanes[sample]):
                    slot = sample_lanes[sample][lane_key]
                    lane_bits.append(
                        f"{lane_key}: R1={slot.get('R1')}, R2={slot.get('R2')}"
                    )
                details.append(f"{sample}: " + "; ".join(lane_bits))
            raise RuntimeError(
                "Multiple FASTQ lanes detected for the same sample id(s). "
                "Refusing to silently discard later lanes. Set "
                "fastq_multi_lane_policy='merge' to concatenate lanes, or rename "
                "files so each sample has a single R1/R2 pair.\n" + "\n".join(details)
            )

        pairs: List[Tuple[Path, Path, str]] = []
        for sample in sorted(sample_lanes):
            lanes = sample_lanes[sample]
            # Pair R1/R2 by lane key; reject unpaired or unbalanced lanes.
            ordered_keys = sorted(lanes.keys())
            r1_paths: List[Path] = []
            r2_paths: List[Path] = []
            for lane_key in ordered_keys:
                slot = lanes[lane_key]
                r1 = slot.get("R1")
                r2 = slot.get("R2")
                if r1 is None and r2 is None:
                    continue
                if r1 is None:
                    logger.warning(
                        "Missing R1 FASTQ for sample=%s lane=%s", sample, lane_key
                    )
                    continue
                if r2 is None:
                    logger.warning(
                        "Missing R2 FASTQ for sample=%s lane=%s", sample, lane_key
                    )
                    continue
                r1_paths.append(r1)
                r2_paths.append(r2)

            if not r1_paths:
                logger.warning("No complete R1/R2 lane pairs for sample=%s", sample)
                continue
            if len(r1_paths) == 1:
                pairs.append((r1_paths[0], r2_paths[0], sample))
            else:
                r1_merged, r2_merged = self._merge_lane_fastqs(
                    sample, r1_paths, r2_paths, lane_keys=ordered_keys[: len(r1_paths)]
                )
                pairs.append((r1_merged, r2_merged, sample))

        self._assert_no_sanitize_collisions([sample for _, _, sample in pairs])
        return pairs

    def _merge_lane_fastqs(
        self,
        sample: str,
        r1_paths: List[Path],
        r2_paths: List[Path],
        lane_keys: Optional[List[str]] = None,
    ) -> Tuple[Path, Path]:
        """Concatenate multi-lane R1/R2 FASTQs in the same lane-key order."""
        if len(r1_paths) != len(r2_paths):
            raise RuntimeError(
                f"Internal multi-lane merge imbalance for sample={sample}: "
                f"n_R1={len(r1_paths)} n_R2={len(r2_paths)}"
            )
        safe = self._safe_sample_name(sample)
        merge_dir = self.intermediate_dir / safe / "merged_lanes"
        merge_dir.mkdir(parents=True, exist_ok=True)
        r1_out = merge_dir / f"{safe}.R1.merged.fastq.gz"
        r2_out = merge_dir / f"{safe}.R2.merged.fastq.gz"
        logger.info(
            "Merging %d FASTQ lanes for sample=%s | lane_keys=%s",
            len(r1_paths),
            sample,
            list(lane_keys) if lane_keys is not None else [p.name for p in r1_paths],
        )
        self._concat_fastqs(r1_paths, r1_out)
        self._concat_fastqs(r2_paths, r2_out)
        return r1_out, r2_out

    @staticmethod
    def _concat_fastqs(sources: List[Path], dest: Path) -> None:
        """Binary-concatenate FASTQ(.gz) files. Order is caller-defined."""
        with open(dest, "wb") as out_fh:
            for src in sources:
                with open(src, "rb") as in_fh:
                    shutil.copyfileobj(in_fh, out_fh)

    def _assert_no_sanitize_collisions(self, samples: List[str]) -> None:
        """Fail if distinct sample ids collapse to the same sanitized name."""
        safe_to_raw: Dict[str, List[str]] = {}
        for sample in samples:
            safe = self._safe_sample_name(sample)
            safe_to_raw.setdefault(safe, []).append(sample)
        collisions = {
            safe: raws for safe, raws in safe_to_raw.items() if len(set(raws)) > 1
        }
        if collisions:
            detail = "; ".join(
                f"{safe!r} <- {sorted(set(raws))}"
                for safe, raws in sorted(collisions.items())
            )
            raise RuntimeError(
                "FASTQ sample-name sanitization collisions detected. Distinct sample "
                f"ids map to the same filesystem-safe name: {detail}. Rename inputs "
                "so sanitized names remain unique."
            )

    # ------------------------------------------------------------------
    # Validation and setup
    # ------------------------------------------------------------------
    def _resolve_total_threads(self, n_jobs: Optional[int]) -> int:
        cpu_count = max(1, multiprocessing.cpu_count())

        if n_jobs is not None:
            requested = int(n_jobs)
        else:
            cfg_threads = getattr(self.config, "fastq_threads", None)
            if cfg_threads is not None:
                requested = int(cfg_threads)
            else:
                cfg_n_jobs = int(getattr(self.config, "n_jobs", -1) or -1)
                requested = cpu_count if cfg_n_jobs < 0 else cfg_n_jobs

        if requested < 1:
            requested = cpu_count
        return max(1, min(requested, cpu_count))

    def _validate_inputs(self) -> None:
        if not self.fastq_dir.exists() or not self.fastq_dir.is_dir():
            raise FileNotFoundError(f"FASTQ directory not found: {self.fastq_dir}")
        if not self.ref_genome.exists() or not self.ref_genome.is_file():
            raise FileNotFoundError(f"Reference genome not found: {self.ref_genome}")
        if self.max_parallel_samples < 1:
            raise ValueError("fastq_max_parallel_samples must be >= 1")
        if self.min_mapping_quality < 0:
            raise ValueError("fastq_min_mapping_quality must be >= 0")

    def _setup_directory_structure(self) -> None:
        paths = [
            self.output_dir,
            self.intermediate_dir,
            self.bam_dir,
            self.vcf_dir,
            self.stats_dir,
            self.logs_dir,
            self.provenance_dir,
        ]
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _allocate_stage_threads(budget: int) -> Tuple[int, int, int]:
        """
        Partition a per-sample CPU budget across concurrent piped stages:
        ``bwa mem | samtools view | samtools sort``.

        Because these stages run concurrently in a pipe, their thread counts
        **sum** against the budget (they do not each get the full budget).
        Each stage needs at least 1 thread; budgets < 3 therefore
        unavoidably oversubscribe slightly and are logged as such.
        """
        budget = max(1, int(budget))
        if budget <= 2:
            return 1, 1, 1
        # Prefer ~50% aligner, ~15% view, remainder sort.
        bwa = max(1, budget // 2)
        rem = budget - bwa
        view = max(1, rem // 3)
        sort_n = max(1, rem - view)
        # Clamp sum to budget without dropping any stage below 1.
        while bwa + view + sort_n > budget and bwa > 1:
            bwa -= 1
        while bwa + view + sort_n > budget and sort_n > 1:
            sort_n -= 1
        while bwa + view + sort_n > budget and view > 1:
            view -= 1
        return int(bwa), int(view), int(sort_n)

    def _check_required_tools(self) -> None:
        missing: List[str] = []
        required = ["bwa", "samtools"]
        if self.call_mode in {"full", "panel_bcftools"}:
            required.append("bcftools")
        elif self.call_mode == "panel_majority":
            # bcftools optional (bgzip/index); samtools required for mpileup
            if shutil.which("bcftools") is not None:
                required.append("bcftools")
        for tool in required:
            if shutil.which(tool) is None:
                missing.append(tool)
            else:
                self.tool_versions[tool] = self._tool_version(tool)
        if missing:
            raise RuntimeError(
                "FASTQ query mode requires external tools on PATH: "
                + ", ".join(missing)
            )

    def _prepare_reference_indexes(self) -> None:
        bwa_ok = all(
            self.ref_genome.with_suffix(self.ref_genome.suffix + ext).exists()
            for ext in [".amb", ".ann", ".bwt", ".pac", ".sa"]
        )
        fai_ok = Path(str(self.ref_genome) + ".fai").exists()

        if bwa_ok and fai_ok:
            return

        if not self.auto_index_reference:
            missing = []
            if not bwa_ok:
                missing.append("BWA index")
            if not fai_ok:
                missing.append("samtools FASTA index")
            raise RuntimeError(
                "Reference indexes missing: "
                + ", ".join(missing)
                + ". Enable fastq_auto_index_reference or index the reference manually."
            )

        if not bwa_ok:
            logger.info(
                "BWA reference index missing; creating index for %s", self.ref_genome
            )
            self._execute(["bwa", "index", str(self.ref_genome)], "bwa_index")

        if not fai_ok:
            logger.info(
                "samtools FASTA index missing; creating index for %s", self.ref_genome
            )
            self._execute(["samtools", "faidx", str(self.ref_genome)], "samtools_faidx")

    # ------------------------------------------------------------------
    # FASTQ naming
    # ------------------------------------------------------------------
    @staticmethod
    def _is_fastq(path: Path) -> bool:
        name = path.name.lower()
        return any(name.endswith(ext) for ext in FASTQ_EXTENSIONS)

    @staticmethod
    def _strip_fastq_extension(name: str) -> str:
        lower = name.lower()
        for ext in FASTQ_EXTENSIONS:
            if lower.endswith(ext):
                return name[: -len(ext)]
        return Path(name).stem

    @classmethod
    def _parse_fastq_name(cls, name: str) -> Optional[Tuple[str, str, str]]:
        """
        Parse FASTQ basename into ``(sample_id, read, lane_key)``.

        ``lane_key`` identifies the lane (or residual suffix) so R1/R2 files
        are paired by key rather than by independent lexical path order.
        Single-lane files use lane_key ``"L000"``.
        """
        stem = cls._strip_fastq_extension(name)
        # Illumina Casava/BCL: Sample_S1_L001_R1_001
        illumina = re.match(
            r"^(?P<sample>.+?)(?:_S\d+)?_(?P<lane>L\d{3})_R(?P<read>[12])(?:_\d+)?$",
            stem,
            flags=re.IGNORECASE,
        )
        if illumina:
            sample = illumina.group("sample").strip("._-")
            read = "R1" if illumina.group("read") == "1" else "R2"
            lane = illumina.group("lane").upper()
            if sample:
                return sample, read, lane

        # Generic: sample_R1, sample_R1_001, sample_R1_L001, sample.R1.laneA
        patterns = [
            r"^(?P<sample>.+?)[._-]R(?P<read>[12])(?:[._-](?P<rest>.+))?$",
            r"^(?P<sample>.+?)[._-](?P<read>[12])(?:[._-](?P<rest>.+))?$",
        ]
        for pattern in patterns:
            match = re.match(pattern, stem, flags=re.IGNORECASE)
            if not match:
                continue
            sample = match.group("sample").strip("._-")
            read = "R1" if match.group("read") == "1" else "R2"
            rest = (match.group("rest") or "").strip("._-")
            if not sample:
                continue
            # Lane key priority:
            # 1) explicit Illumina L00N in residual
            # 2) non-empty residual (including pure 001 file indices) so multiple
            #    R1 files for one sample remain distinct keys and pair with R2
            # 3) default L000 for a bare sample_R1 / sample_R2 pair
            if rest:
                lane_m = re.search(r"(L\d{3})", rest, flags=re.IGNORECASE)
                if lane_m:
                    lane_key = lane_m.group(1).upper()
                else:
                    lane_key = rest
            else:
                lane_key = "L000"
            # If sample still embeds an Illumina lane token, peel it for grouping.
            sample_lane = re.search(
                r"(?:^|[._-])(L\d{3})$", sample, flags=re.IGNORECASE
            )
            if sample_lane and lane_key == "L000":
                lane_key = sample_lane.group(1).upper()
                sample = re.sub(
                    r"(?:[._-]?L\d{3})$", "", sample, flags=re.IGNORECASE
                ).strip("._-")
            return sample, read, lane_key
        return None

    # ------------------------------------------------------------------
    # Per-sample processing
    # ------------------------------------------------------------------
    def _process_one_sample(self, r1: Path, r2: Path, sample: str) -> Path:
        safe_sample = self._safe_sample_name(sample)
        sample_dir = self.intermediate_dir / safe_sample
        sample_dir.mkdir(parents=True, exist_ok=True)

        sorted_bam = self.bam_dir / f"{safe_sample}.sorted.bam"
        sorted_bam_bai = Path(str(sorted_bam) + ".bai")
        final_vcf = self.vcf_dir / f"{safe_sample}.vcf.gz"
        ploidy_file = self.intermediate_dir / f"{safe_sample}.ploidy.txt"

        # Track files actually produced so cleanup removes real artifacts
        # (not only the sample intermediate directory, which may be empty).
        produced_paths: List[Path] = [
            sorted_bam,
            sorted_bam_bai,
            ploidy_file,
        ]

        try:
            self._align_sort_index_sample(
                r1=r1,
                r2=r2,
                sample=safe_sample,
                sorted_bam=sorted_bam,
            )
            self._call_variants(
                sample=safe_sample,
                sorted_bam=sorted_bam,
                final_vcf=final_vcf,
                ploidy_file=ploidy_file,
            )
            self._write_sample_metrics(
                sample=safe_sample, sorted_bam=sorted_bam, final_vcf=final_vcf
            )
        finally:
            if self.clean_intermediates:
                self._clean_sample_intermediates(
                    sample_dir=sample_dir,
                    produced_paths=produced_paths,
                )

        if not final_vcf.exists():
            raise RuntimeError(f"Expected final VCF was not created: {final_vcf}")
        return final_vcf

    def _align_sort_index_sample(
        self, r1: Path, r2: Path, sample: str, sorted_bam: Path
    ) -> None:
        # Concurrent pipe stages share threads_per_sample (sum, not each-full).
        bwa_t = int(self.bwa_threads)
        view_t = int(self.view_threads)
        sort_t = int(self.sort_threads)
        # samtools sort -m is memory *per sort thread*.
        sort_mem = str(self.sort_memory_per_thread)
        rg = f"@RG\\tID:{sample}\\tSM:{sample}\\tPL:{self.sample_platform}"

        cmd = (
            "set -euo pipefail; "
            f"bwa mem -t {bwa_t} -R {shlex.quote(rg)} "
            f"{shlex.quote(str(self.ref_genome))} {shlex.quote(str(r1))} {shlex.quote(str(r2))} "
            f"| samtools view -@ {view_t} -b - "
            f"| samtools sort -@ {sort_t} -m {shlex.quote(sort_mem)} "
            f"-o {shlex.quote(str(sorted_bam))} -"
        )
        self._execute_shell(cmd, f"{sample}_alignment_sort")
        self._execute(["samtools", "index", str(sorted_bam)], f"{sample}_bam_index")

    def _load_panel_sites(self):
        """Load trained panel sites for panel_* call modes."""
        if self._panel_sites is not None:
            return self._panel_sites
        try:
            from mtb_amr_classifier.panel_pileup_caller import (
                load_panel_sites_from_bed,
                load_panel_sites_from_feature_ids,
                load_panel_sites_from_manifest_tsv,
            )
        except ImportError:  # pragma: no cover
            from panel_pileup_caller import (  # type: ignore
                load_panel_sites_from_bed,
                load_panel_sites_from_feature_ids,
                load_panel_sites_from_manifest_tsv,
            )

        if self.panel_feature_ids:
            self._panel_sites = load_panel_sites_from_feature_ids(self.panel_feature_ids)
        elif self.panel_manifest:
            self._panel_sites = load_panel_sites_from_manifest_tsv(
                Path(str(self.panel_manifest))
            )
        elif self.panel_sites_bed:
            self._panel_sites = load_panel_sites_from_bed(Path(str(self.panel_sites_bed)))
        else:
            raise RuntimeError(
                f"fastq_call_mode={self.call_mode} requires panel sites. Provide "
                "fastq_panel_sites_bed, fastq_panel_manifest, or pass panel_feature_ids "
                "from the query selected-feature manifest."
            )
        logger.info(
            "Loaded %d panel sites for call_mode=%s",
            len(self._panel_sites),
            self.call_mode,
        )
        return self._panel_sites

    def _call_variants(
        self,
        sample: str,
        sorted_bam: Path,
        final_vcf: Path,
        ploidy_file: Optional[Path] = None,
    ) -> None:
        """
        Call variants with explicit ploidy from config (haploid=1 default) and
        optional gVCF-style callable reference blocks.

        call_mode:
          - full: whole-genome bcftools mpileup|call
          - panel_bcftools: bcftools restricted to trained panel BED
          - panel_majority: samtools mpileup on panel + median/majority base

        When ``emit_gvcf`` is True (full / panel_bcftools), use ``bcftools call -m -g``.
        When ``normalize_vcf`` is True, run ``bcftools norm -f`` after calling.
        """
        if self.call_mode == "panel_majority":
            self._call_variants_panel_majority(sample, sorted_bam, final_vcf)
            return

        threads = max(1, self.threads_per_sample)
        # bcftools requires a ploidy file for non-human assemblies.
        if ploidy_file is None:
            ploidy_file = (
                self.intermediate_dir / f"{self._safe_sample_name(sample)}.ploidy.txt"
            )
        # CHROM FROM TO SEX PLOIDY — explicit config ploidy, not hard-coded 1.
        ploidy_file.write_text(f"* * * * {int(self.ploidy)}\n", encoding="utf-8")
        ploidy_arg = f"--ploidy-file {shlex.quote(str(ploidy_file))}"
        raw_vcf = final_vcf
        if self.normalize_vcf:
            raw_vcf = (
                self.intermediate_dir / f"{self._safe_sample_name(sample)}.raw.vcf.gz"
            )

        if self.emit_gvcf:
            # -g DP: emit gVCF blocks with min depth for callable reference.
            call_args = (
                f"bcftools call -m {ploidy_arg} -g {int(self.gvcf_min_dp)} "
                f"--threads {threads} -Oz -o {shlex.quote(str(raw_vcf))}"
            )
        else:
            # Variants-only (legacy). Downstream encoding will not treat absence as REF
            # unless assume_absent_variant_is_reference is explicitly enabled.
            call_args = (
                f"bcftools call -m -v {ploidy_arg} "
                f"--threads {threads} -Oz -o {shlex.quote(str(raw_vcf))}"
            )
        # gVCF mode (bcftools call -g) requires FORMAT/DP in the mpileup stream.
        # Without -a FORMAT/DP, bcftools >=1.1x fails with:
        #   "--gvcf output mode requires FORMAT/DP tag, which is not present"
        mpileup_annots = ""
        if self.emit_gvcf:
            mpileup_annots = "-a FORMAT/DP,FORMAT/AD "

        region_arg = ""
        if self.call_mode == "panel_bcftools":
            sites = self._load_panel_sites()
            try:
                from mtb_amr_classifier.panel_pileup_caller import write_panel_bed
            except ImportError:  # pragma: no cover
                from panel_pileup_caller import write_panel_bed  # type: ignore
            bed = self.intermediate_dir / f"{self._safe_sample_name(sample)}.panel.bed"
            write_panel_bed(sites, bed)
            region_arg = f"-R {shlex.quote(str(bed))} "
            logger.info(
                "Panel bcftools call | sample=%s | sites=%d | bed=%s",
                sample,
                len(sites),
                bed,
            )

        cmd = (
            "set -euo pipefail; "
            f"bcftools mpileup -f {shlex.quote(str(self.ref_genome))} "
            f"{region_arg}"
            f"{mpileup_annots}"
            f"-Ou --min-MQ {int(self.min_mapping_quality)} {shlex.quote(str(sorted_bam))} "
            f"| {call_args}"
        )
        self._execute_shell(cmd, f"{sample}_variant_calling")
        self._execute(
            ["bcftools", "index", "-t", str(raw_vcf)], f"{sample}_raw_vcf_index"
        )

        if self.normalize_vcf:
            # Left-align and split multiallelics against the documented reference.
            self._execute(
                [
                    "bcftools",
                    "norm",
                    "-f",
                    str(self.ref_genome),
                    "-m",
                    "-both",
                    "-Oz",
                    "-o",
                    str(final_vcf),
                    str(raw_vcf),
                ],
                f"{sample}_bcftools_norm",
            )
            self._execute(
                ["bcftools", "index", "-t", str(final_vcf)], f"{sample}_vcf_index"
            )
            if self.clean_intermediates:
                try:
                    raw_vcf.unlink(missing_ok=True)
                    Path(str(raw_vcf) + ".tbi").unlink(missing_ok=True)
                except OSError:
                    pass
        else:
            if raw_vcf != final_vcf:
                shutil.move(str(raw_vcf), str(final_vcf))
            self._execute(
                ["bcftools", "index", "-t", str(final_vcf)], f"{sample}_vcf_index"
            )

    def _call_variants_panel_majority(
        self,
        sample: str,
        sorted_bam: Path,
        final_vcf: Path,
    ) -> None:
        """Median/majority pileup genotyping at trained panel positions only."""
        try:
            from mtb_amr_classifier.panel_pileup_caller import genotype_bam_to_panel_vcf
        except ImportError:  # pragma: no cover
            from panel_pileup_caller import genotype_bam_to_panel_vcf  # type: ignore

        sites = self._load_panel_sites()
        safe = self._safe_sample_name(sample)
        work = self.intermediate_dir / safe / "panel_majority"
        work.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Panel majority call | sample=%s | sites=%d | min_depth=%d | min_frac=%.2f",
            sample,
            len(sites),
            self.panel_min_depth,
            self.panel_min_majority_fraction,
        )
        genotype_bam_to_panel_vcf(
            bam_path=sorted_bam,
            ref_fasta=self.ref_genome,
            sites=sites,
            output_vcf=final_vcf,
            work_dir=work,
            sample_id=safe,
            min_mapping_quality=self.min_mapping_quality,
            min_base_quality=self.panel_min_base_quality,
            min_depth=self.panel_min_depth,
            min_majority_fraction=self.panel_min_majority_fraction,
            emit_reference_sites=self.panel_emit_reference_sites,
        )
        self.command_log.append(
            {
                "step": f"{sample}_panel_majority",
                "command": (
                    f"panel_majority_pileup sites={len(sites)} "
                    f"min_depth={self.panel_min_depth} "
                    f"min_frac={self.panel_min_majority_fraction}"
                ),
                "returncode": 0,
                "log_path": str(self.logs_dir / f"{safe}_panel_majority.log"),
            }
        )
        (self.logs_dir / f"{safe}_panel_majority.log").write_text(
            f"panel_majority complete | sites={len(sites)} | vcf={final_vcf}\n",
            encoding="utf-8",
        )

    def _write_sample_metrics(
        self, sample: str, sorted_bam: Path, final_vcf: Path
    ) -> None:
        if self.write_alignment_stats:
            self._execute_shell(
                f"samtools flagstat {shlex.quote(str(sorted_bam))} > {shlex.quote(str(self.stats_dir / (sample + '.flagstat.txt')))}",
                f"{sample}_flagstat",
            )
            self._execute_shell(
                f"samtools stats {shlex.quote(str(sorted_bam))} > {shlex.quote(str(self.stats_dir / (sample + '.alignment.stats.txt')))}",
                f"{sample}_samtools_stats",
            )
            self._execute_shell(
                f"bcftools stats {shlex.quote(str(final_vcf))} > {shlex.quote(str(self.stats_dir / (sample + '.vcf.stats.txt')))}",
                f"{sample}_bcftools_stats",
            )

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------
    def _execute(self, cmd: List[str], step_name: str) -> None:
        quoted = " ".join(shlex.quote(str(x)) for x in cmd)
        self._execute_shell(quoted, step_name)

    def _execute_shell(self, cmd: str, step_name: str) -> None:
        log_path = self.logs_dir / f"{self._safe_sample_name(step_name)}.log"
        if self.memory_per_sample_mb is not None:
            cmd = f"ulimit -v {int(self.memory_per_sample_mb) * 1024}; {cmd}"

        logger.debug("Executing %s: %s", step_name, cmd)
        result = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        self.command_log.append(
            {
                "step": str(step_name),
                "command": str(cmd),
                "returncode": int(result.returncode),
                "log_path": str(log_path),
            }
        )

        log_path.write_text(
            "COMMAND\n=======\n"
            f"{cmd}\n\n"
            "STDOUT\n======\n"
            f"{result.stdout}\n\n"
            "STDERR\n======\n"
            f"{result.stderr}\n",
            encoding="utf-8",
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"{step_name} failed with exit code {result.returncode}. See log: {log_path}"
            )

    def _tool_version(self, tool: str) -> str:
        version_commands = {
            "bwa": ["bwa"],
            "samtools": ["samtools", "--version"],
            "bcftools": ["bcftools", "--version"],
        }
        cmd = version_commands.get(tool, [tool, "--version"])
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
            )
            text = (result.stdout or result.stderr or "").strip().splitlines()
            return text[0] if text else "available"
        except Exception:
            return "available"

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_sample_name(sample: str) -> str:
        sample = str(sample).strip()
        sample = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample)
        return sample.strip("._-") or "sample"

    @staticmethod
    def _write_json(payload: Dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)

    def _clean_sample_intermediates(
        self,
        sample_dir: Path,
        produced_paths: Optional[List[Path]] = None,
    ) -> None:
        """
        Remove intermediate files actually produced for one sample.

        Always removes tracked BAM/BAI/ploidy paths. Also clears the sample
        intermediate directory (merged lanes, temp files). Never deletes final
        VCFs or stats/callability products under final/.
        """
        try:
            for path in produced_paths or []:
                try:
                    if path.is_file():
                        path.unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning(
                        "Could not remove intermediate file %s: %s", path, exc
                    )

            if sample_dir.exists():
                for path in sorted(sample_dir.rglob("*"), reverse=True):
                    try:
                        if path.is_file() or path.is_symlink():
                            path.unlink(missing_ok=True)
                        elif path.is_dir():
                            path.rmdir()
                    except OSError:
                        pass
                try:
                    sample_dir.rmdir()
                except OSError:
                    pass
        except Exception as exc:
            logger.warning(
                "Could not clean FASTQ intermediates for %s: %s", sample_dir, exc
            )
