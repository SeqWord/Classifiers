#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Classify FASTA/GenBank genomes with a hierarchical model bundle."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_DIR = SCRIPT_DIR / "lib"
if not LIB_DIR.is_dir():
    raise SystemExit(
        f"ERROR: Required classifier library directory does not exist: {LIB_DIR}"
    )
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

try:
    from model_bundle import ModelBundle
    import blast as custom_blast
except Exception as exc:
    raise SystemExit(
        f"ERROR: Cannot import model_bundle.py and blast.py from {LIB_DIR}: {exc}"
    )

TERMINAL_LABEL = "__TERMINAL__"
DNA_BASES = {"A", "C", "G", "T"}
MISSING_STATE = "n"


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a Boolean value, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Call allelic states from genome sequences and classify them through "
            "a stored hierarchical model tree."
        )
    )
    parser.add_argument(
        "project_folder",
        help="Folder under input_folder containing FASTA/GenBank genome files."
    )
    parser.add_argument("-i", "--input_folder", default="input")
    parser.add_argument("-o", "--output_folder", default="output")
    parser.add_argument(
        "--model_file", default="model/model.pkl",
        help="Model PKL path relative to classifier.py (default: model/model.pkl)."
    )
    parser.add_argument(
        "-m", "--mismatch", type=int, default=6, metavar="INT",
        help="Maximum BLAST substitutions in a full-length context hit (default: 6).",
    )
    parser.add_argument(
        "-T", "--threads", type=int, default=8, metavar="INT",
        help="Number of concurrent locus/BLAST worker threads (default: 8).",
    )
    parser.add_argument(
        "-c", "--concatenate_records",
        type=str2bool, default=True,
        help=(
            "For multi-record genomes, concatenate records with an N spacer "
            "(default: True). If False, use only the longest record."
        ),
    )
    parser.add_argument(
        "--max_no_data_for_NN", type=float, default=20.0,
        help=(
            "Maximum percentage of missing marker calls at which an NN/ML model "
            "may still be used. Above this value allele-profile similarity is used "
            "(default: 20)."
        ),
    )
    parser.add_argument(
        "--max_no_data_allowed", type=float, default=50.0,
        help=(
            "Maximum percentage of missing marker calls allowed for any classification. "
            "Above this value the sequence is reported as not identified (default: 50)."
        ),
    )
    parser.add_argument(
        "--sensitivity", type=float, default=0.85,
        help="Minimum score required for a terminal-node identification (0..1; default: 0.85)."
    )
    parser.add_argument(
        "--specificity", type=float, default=0.70,
        help=(
            "Minimum score required to continue along an intermediate branch "
            "(0..1; default: 0.70)."
        ),
    )
    parser.add_argument(
        "--report",
        choices=["short", "detailed"],
        default="short",
        help=(
            "Report format: 'short' for compact user-oriented hierarchical "
            "identifications, or 'detailed' for the full diagnostic table "
            "(default: short)."
        ),
    )
    parser.add_argument(
        "--report_level",
        type=int,
        default=2,
        help=(
            "Number of lowest hierarchy labels shown in short-report "
            "identifications (must be > 0; default: 2)."
        ),
    )
    return parser.parse_args()


def load_bundle(path: Path) -> ModelBundle:
    if not path.is_file():
        raise FileNotFoundError(f"Model bundle does not exist: {path}")
    return ModelBundle.load(path)


def iter_genome_files(folder: Path) -> Iterable[Path]:
    fasta_ext = {".fa", ".fas", ".fasta", ".fna", ".fst"}
    gb_ext = {".gb", ".gbk", ".gbf", ".gbff"}

    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        name = path.name.lower()
        plain_suffix = path.suffix.lower()
        if plain_suffix in fasta_ext | gb_ext:
            yield path
        elif name.endswith(".gz"):
            stem_suffix = Path(path.stem).suffix.lower()
            if stem_suffix in fasta_ext | gb_ext:
                yield path


def _sequence_format(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".gz"):
        name = name[:-3]
    suffix = Path(name).suffix.lower()
    if suffix in {".fa", ".fas", ".fasta", ".fna", ".fst"}:
        return "fasta"
    if suffix in {".gb", ".gbk", ".gbf", ".gbff"}:
        return "genbank"
    raise ValueError(f"Unsupported sequence format: {path}")


def read_genome_sequence(path: Path, concatenate_records: bool) -> Tuple[str, int]:
    fmt = _sequence_format(path)
    opener = gzip.open if path.name.lower().endswith(".gz") else open

    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        records = list(SeqIO.parse(handle, fmt))

    if not records:
        raise ValueError(f"No sequence records found in {path}")

    sequences = [str(record.seq).upper() for record in records if len(record.seq) > 0]
    if not sequences:
        raise ValueError(f"No non-empty sequence records found in {path}")

    if concatenate_records:
        # Prevent a marker context from being reconstructed across contig boundaries.
        sequence = ("N" * 100).join(sequences)
    else:
        sequence = max(sequences, key=len)

    return sequence, len(records)


def reverse_complement(sequence: str) -> str:
    """Return the reverse complement, including standard ambiguous bases."""
    table = str.maketrans(
        "ACGTRYMKBDHVNacgtrymkbdhvn",
        "TGCAYRKMVHDBNtgcayrkmvhdbn",
    )
    return sequence.translate(table)[::-1]


def _bundled_executable(name: str) -> Optional[str]:
    """Return a platform-compatible executable from ./lib/bin, if present."""
    bin_dir = LIB_DIR / "bin"
    candidates = [bin_dir / f"{name}.exe"] if sys.platform == "win32" else [bin_dir / name]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _find_blast_backend() -> Dict[str, Any]:
    """Select the same BLAST backend strategy used by feature_calle.py."""
    if sys.platform == "win32":
        bin_dir = LIB_DIR / "bin"
        blastall_exe = bin_dir / "blastall.exe"
        formatdb_exe = bin_dir / "formatdb.exe"
        missing = [str(x) for x in (blastall_exe, formatdb_exe) if not x.is_file()]
        if missing:
            raise RuntimeError(
                "Required Windows BLAST executable(s) not found: " + ", ".join(missing)
            )
        return {
            "kind": "custom",
            "module": custom_blast,
            "binpath": str(bin_dir),
            "description": "Windows legacy BLAST (blastall.exe/formatdb.exe)",
        }

    blastn = shutil.which("blastn")
    makeblastdb = shutil.which("makeblastdb")
    if blastn and makeblastdb:
        return {
            "kind": "native",
            "blastn": blastn,
            "makeblastdb": makeblastdb,
            "description": "system NCBI BLAST+",
        }

    blastn = _bundled_executable("blastn")
    makeblastdb = _bundled_executable("makeblastdb")
    if blastn and makeblastdb:
        try:
            os.chmod(blastn, 0o755)
            os.chmod(makeblastdb, 0o755)
        except OSError:
            pass
        return {
            "kind": "native",
            "blastn": blastn,
            "makeblastdb": makeblastdb,
            "description": "bundled Linux NCBI BLAST+",
        }

    bin_dir = LIB_DIR / "bin"
    return {
        "kind": "custom",
        "module": custom_blast,
        "binpath": str(bin_dir),
        "description": "legacy custom BLAST wrapper",
    }


def _run_checked(command: List[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n    " + 
            " ".join(command) + 
            f"\nSTDOUT:\n{result.stdout}" +
            f"\nSTDERR:\n{result.stderr}"
            )
    return result


def _create_native_blast_db(
    sequence: str,
    tmp_directory: Path,
    makeblastdb_exe: str,
) -> str:
    input_fasta = tmp_directory / "genome_for_blast.fasta"
    input_fasta.write_text(f">genome\n{sequence}\n", encoding="utf-8")
    dbname = str(tmp_directory / "genome_db")
    _run_checked([
        makeblastdb_exe,
        "-in", str(input_fasta),
        "-dbtype", "nucl",
        "-out", dbname,
    ])
    return dbname


def _query_base_from_alignment(
    qseq: str, sseq: str, qstart: int, center: int
) -> Optional[str]:
    query_pos = qstart - 1
    for qchar, schar in zip(qseq, sseq):
        if qchar != "-":
            if query_pos == center:
                base = schar.upper()
                return base if base in DNA_BASES else None
            query_pos += 1
    return None


def _native_blast_context(
    context: str,
    center: int,
    mismatch: int,
    blastn_exe: str,
    dbname: str,
    tmp_directory: Path,
) -> Optional[Tuple[str, str, int]]:
    """Search one complete context with blastn-short and return the best full hit."""
    query_file = tmp_directory / f"context_query_{threading.get_ident()}.fasta"
    query_file.write_text(f">context\n{context}\n", encoding="utf-8")

    outfmt = "6 qstart qend sstart send length mismatch gaps bitscore qseq sseq"
    result = _run_checked([
        blastn_exe,
        "-query", str(query_file),
        "-db", dbname,
        "-task", "blastn-short",
        "-dust", "no",
        "-word_size", "7",
        "-evalue", "1000",
        "-max_target_seqs", "50",
        "-outfmt", outfmt,
    ])

    candidates: List[Tuple[int, float, int, str, str]] = []
    qlen = len(context)
    for line in result.stdout.splitlines():
        fields = line.rstrip("\n").split("\t")
        if len(fields) != 10:
            continue
        try:
            qstart, qend, sstart, send, length, mismatches, gaps = map(
                int, fields[:7]
            )
            bitscore = float(fields[7])
        except ValueError:
            continue
        qseq, sseq = fields[8].upper(), fields[9].upper()
        if qstart != 1 or qend != qlen or length != qlen or gaps != 0:
            continue
        if mismatches > mismatch:
            continue
        called_base = _query_base_from_alignment(qseq, sseq, qstart, center)
        if called_base is None:
            continue
        genomic_center = (
            sstart - 1 + center if sstart <= send else sstart - 1 - center
        )
        candidates.append((mismatches, -bitscore, genomic_center, sseq, called_base))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    best = candidates[0]
    return best[4], best[3], best[2]


def _custom_blast_context(
    sequence: str,
    context: str,
    center: int,
    mismatch: int,
    blast_object: Any,
    dbname: str,
    tmp_directory: Path,
) -> Optional[Tuple[str, str, int]]:

    query_file = (
        tmp_directory
        / f"context_query_{threading.get_ident()}.fasta"
    )

    query_file.write_text(
        f">context\n{context}\n",
        encoding="utf-8",
    )

    blast_object.execute(
        str(query_file.resolve()),
        dbname,
    )

    matches = blast_object.get_matches(
        query_length=len(context),
        mismatches=mismatch,
    )

    matches = [
        match
        for match in matches
        if getattr(match, "alignment_length", 0) == len(context)
    ]

    if not matches:
        return None

    matches.sort(
        key=lambda match: (
            -int(getattr(match, "identities", 0)),
            min(
                int(match.sbjct_start),
                int(match.sbjct_end),
            ),
        )
    )

    match = matches[0]

    sstart = int(match.sbjct_start)
    send = int(match.sbjct_end)

    start0 = min(sstart, send) - 1
    end0 = max(sstart, send)

    subject_context = sequence[start0:end0].upper()

    if (
        sstart > send
        or getattr(match, "strand", "") == "Plus/Minus"
    ):
        subject_context = reverse_complement(
            subject_context
        ).upper()

        genomic_center = sstart - 1 - center

    else:
        genomic_center = sstart - 1 + center

    if (
        len(subject_context) != len(context)
        or center >= len(subject_context)
    ):
        return None

    called_base = subject_context[center].upper()

    if called_base not in DNA_BASES:
        return None

    return (
        called_base,
        subject_context,
        genomic_center,
    )


def _blast_allelic_state(
    sequence: str,
    context: str,
    center: int,
    mismatch: int,
    native_blastn: Optional[str],
    dbname: str,
    tmp_directory: Path,
    custom_blast_object: Any = None,
) -> Optional[str]:
    """Call the A/C/G/T base aligned to `center` in a full-length BLAST hit."""
    context = str(context).strip().upper()
    if not context or center < 0 or center >= len(context):
        return None

    hit: Optional[Tuple[str, str, int]] = None
    if native_blastn and dbname:
        hit = _native_blast_context(
            context=context, center=center, mismatch=mismatch,
            blastn_exe=native_blastn, dbname=dbname, tmp_directory=tmp_directory,
        )
    elif custom_blast_object is not None and dbname:
        hit = _custom_blast_context(
            sequence=sequence, context=context, center=center, mismatch=mismatch,
            blast_object=custom_blast_object, dbname=dbname, tmp_directory=tmp_directory,
        )
    return None if hit is None else hit[0]


def _record_value(record: Dict[str, str], name: str) -> str:
    for key, value in record.items():
        if str(key).strip().lower() == name.lower():
            return str(value).strip()
    return ""


def parse_allowed_alleles(text: str) -> set[str]:
    return set(re.findall(r"[ACGT]", str(text).upper()))


def call_feature_states(
    sequence: str,
    bundle: ModelBundle,
    backend: Dict[str, Any],
    tmp_directory: Path,
    mismatch: int,
    threads: int,
) -> Dict[str, str]:
    """Call marker states with full-length BLAST context searches in parallel."""
    titles = [str(v) for v in bundle["feature_titles"]]
    records = bundle.get("feature_records") or []
    by_id = {_record_value(record, "SNP"): record for record in records}
    data_type = str(bundle.get("data_type", "character")).lower()
    common = bundle.get("common_alleles") or {}

    native_blastn: Optional[str] = None
    dbname = ""

    if backend["kind"] == "native":
        dbname = _create_native_blast_db(
            sequence=sequence,
            tmp_directory=tmp_directory,
            makeblastdb_exe=backend["makeblastdb"],
        )
        native_blastn = backend["blastn"]
    else:
        db_creator = backend["module"].BLAST(
            seqtype="dna", binpath=backend["binpath"]
        )
        input_fasta = tmp_directory / "genome_for_blast.fasta"
        input_fasta.write_text(f">genome\n{sequence}\n", encoding="utf-8")
        dbname = str(tmp_directory / "genome_db")
        db_creator.create_db(fasta_file=str(input_fasta), dbname=dbname)

    thread_state = threading.local()

    def call_one(title: str) -> Tuple[str, str]:
        record = by_id.get(title)
        if record is None:
            return title, MISSING_STATE

        context = _record_value(record, "context").strip().upper()
        if not context or len(context) % 2 != 1:
            return title, MISSING_STATE
        center = len(context) // 2

        blast_object = None
        if backend["kind"] == "custom":
            blast_object = getattr(thread_state, "blast_object", None)
            if blast_object is None:
                blast_object = backend["module"].BLAST(
                    seqtype="dna", binpath=backend["binpath"]
                )
                thread_state.blast_object = blast_object

        base = _blast_allelic_state(
            sequence=sequence, context=context, center=center, mismatch=mismatch,
            native_blastn=native_blastn, dbname=dbname,
            tmp_directory=tmp_directory, custom_blast_object=blast_object,
        )
        if base is None:
            return title, MISSING_STATE

        if data_type in {"binary", "digit"}:
            common_base = str(common.get(title, "")).strip().upper()
            if common_base not in DNA_BASES:
                return title, MISSING_STATE
            return title, "0" if base == common_base else "1"

        return title, base if base in DNA_BASES else MISSING_STATE

    with ThreadPoolExecutor(max_workers=threads) as executor:
        called = list(executor.map(call_one, titles))

    return dict(called)


def missing_fraction(markers: Dict[str, str]) -> float:
    if not markers:
        return 1.0
    missing = sum(str(v).strip().lower() == MISSING_STATE for v in markers.values())
    return missing / len(markers)


def _impute_for_nn(markers: Dict[str, str], bundle: ModelBundle) -> Dict[str, str]:
    """Impute only for NN/ML prediction; raw marker calls remain unchanged."""
    data_type = str(bundle.get("data_type", "character"))
    common = bundle.get("common_alleles") or {}

    result: Dict[str, str] = {}
    for title in bundle["feature_titles"]:
        value = str(markers.get(title, MISSING_STATE)).strip()

        if value.lower() != MISSING_STATE:
            result[title] = value
            continue

        if data_type == "binary":
            result[title] = "1"
        else:
            common_state = str(common.get(title, "")).strip().upper()
            result[title] = common_state if common_state else "N"

    return result


def _softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values - np.max(values)
    exp_values = np.exp(values)
    total = float(exp_values.sum())
    return exp_values / total if total > 0 else np.ones_like(values) / len(values)


def _normalize_scores(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr[~np.isfinite(arr)] = 0.0
    arr = np.maximum(arr, 0.0)
    total = float(arr.sum())
    if total <= 0:
        return np.ones(len(arr), dtype=float) / max(1, len(arr))
    return arr / total


def normalized_entropy(scores: Sequence[float]) -> float:
    """Normalized Shannon entropy in the range 0..1."""
    probs = _normalize_scores(scores)
    if len(probs) <= 1:
        return 0.0
    positive = probs[probs > 0]
    entropy = -float(np.sum(positive * np.log2(positive)))
    return entropy / math.log2(len(probs))


def predict_stored_model(
    model_info: Dict[str, Any],
    markers: Dict[str, str],
    bundle: ModelBundle,
) -> Tuple[Dict[str, float], float]:
    """Return all child scores normalized to 0..1 and normalized entropy."""
    pipeline = model_info["pipeline"]
    feature_titles = [str(v) for v in model_info["feature_titles"]]
    classes = [str(v) for v in model_info["classes"]]
    imputed = _impute_for_nn(markers, bundle)

    row = {title: str(imputed.get(title, MISSING_STATE)) for title in feature_titles}

    known_categories = model_info.get("known_categories")
    if known_categories:
        for i, title in enumerate(feature_titles):
            if i >= len(known_categories):
                continue
            known = [str(v) for v in known_categories[i]]
            if row[title] not in known and known:
                row[title] = "1" if "1" in known else known[0]

    X = pd.DataFrame([row], columns=feature_titles).astype(str)

    if hasattr(pipeline, "predict_proba"):
        raw = np.asarray(pipeline.predict_proba(X)[0], dtype=float)
        probs = _normalize_scores(raw)

    elif hasattr(pipeline, "decision_function"):
        decision = np.asarray(pipeline.decision_function(X), dtype=float)
        if decision.ndim > 1:
            decision = decision[0]
        if decision.size == 1 and len(classes) == 2:
            p1 = 1.0 / (1.0 + math.exp(-float(decision[0])))
            probs = np.asarray([1.0 - p1, p1], dtype=float)
        else:
            probs = _softmax(decision)

    else:
        prediction = pipeline.predict(X)[0]
        probs = np.zeros(max(1, len(classes)), dtype=float)
        try:
            idx = int(prediction)
            if 0 <= idx < len(probs):
                probs[idx] = 1.0
            else:
                raise ValueError
        except Exception:
            label = str(prediction)
            if label in classes:
                probs[classes.index(label)] = 1.0
            else:
                probs[:] = 1.0 / len(probs)

    if len(probs) != len(classes):
        n = min(len(probs), len(classes))
        probs = _normalize_scores(probs[:n])
        classes = classes[:n]

    result = {label: float(score) for label, score in zip(classes, probs)}
    return result, normalized_entropy(list(result.values()))


def profile_child_scores(
    node: Dict[str, Any],
    markers: Dict[str, str],
    data_type: str,
) -> Dict[str, float]:
    """Compare a query to each child allele profile.

    Binary score:
        1 - mean(abs(query_state - child_mean_state))

    Character score:
        mean(child_frequency_of_observed_query_state)

    Missing query states are ignored in both calculations.
    """
    child_profiles = node.get("child_profiles") or {}
    scores: Dict[str, float] = {}

    for child, profile in child_profiles.items():
        components: List[float] = []

        for feature, raw_value in markers.items():
            value = str(raw_value).strip()
            if value.lower() == MISSING_STATE:
                continue

            if data_type == "binary":
                try:
                    q = float(value)
                    mean_state = profile.get(feature)
                    if mean_state is None:
                        continue
                    mean_state = float(mean_state)
                except (ValueError, TypeError):
                    continue
                components.append(1.0 - abs(q - mean_state))
            else:
                ratios = profile.get(feature) or {}
                if not isinstance(ratios, dict):
                    continue
                components.append(float(ratios.get(value, 0.0)))

        score = float(sum(components) / len(components)) if components else 0.0
        scores[str(child)] = max(0.0, min(1.0, score))

    return scores


def node_scores(
    node: Dict[str, Any],
    markers: Dict[str, str],
    bundle: ModelBundle,
    max_no_data_for_nn: float,
) -> Tuple[Dict[str, float], str, float | None]:
    """Choose NN/ML or profile scoring for one splitting node."""
    default_child = node.get("default_child")
    if default_child:
        return {str(default_child): 1.0}, "single-child", 0.0

    missing_pct = 100.0 * missing_fraction(markers)
    model_info = node.get("model")
    data_type = str(bundle.get("data_type", "character"))

    if model_info is None:
        return profile_child_scores(node, markers, data_type), "profile:model-none", None

    if missing_pct > max_no_data_for_nn:
        return profile_child_scores(node, markers, data_type), "profile:missing-data", None

    nn_scores, entropy = predict_stored_model(model_info, markers, bundle)
    max_entropy = float(bundle.get("max_entropy", 0.80))

    if entropy > max_entropy:
        return profile_child_scores(node, markers, data_type), "profile:high-entropy", entropy

    return nn_scores, "NN", entropy



def format_report_path(
    full_path: str,
    report_level: int,
    delimiter: str,
) -> Tuple[str, str]:
    """Return the visible classification and the omitted upper hierarchy path.

    ``report_level`` is the number of lowest hierarchy labels displayed in the
    prediction text.  Upper labels that are hidden from the visible prediction
    are represented in the path column beginning with ``Root``.

    Examples for ``2|Food chain|Chicken products``:
        level 1 -> Chicken products          / Root.2.Food chain
        level 2 -> Food chain|Chicken products / Root.2
        level 3 -> 2|Food chain|Chicken products / Root.0
    """
    parts = [
        part.strip()
        for part in str(full_path).split(delimiter)
        if part.strip()
    ]

    if not parts:
        return "", "Root.0"

    visible_count = min(report_level, len(parts))
    visible = delimiter.join(parts[-visible_count:])
    omitted = parts[:-visible_count]

    root_path = (
        "Root." + ".".join(omitted)
        if omitted
        else "Root.0"
    )

    return visible, root_path


def format_short_prediction(
    prediction: Dict[str, Any],
    report_level: int,
    delimiter: str,
) -> str:
    """Format one indented prediction line for the short report."""
    visible, root_path = format_report_path(
        full_path=str(prediction["path"]),
        report_level=report_level,
        delimiter=delimiter,
    )

    terminal = bool(prediction.get("terminal", False))
    score = float(prediction.get("score", 0.0))

    return (
        f"     {visible}, {score:.4f}\t"
        f"{root_path}\t{terminal}"
    )


def format_short_record(
    title: str,
    predictions: Sequence[Dict[str, Any]],
    report_level: int,
    delimiter: str,
) -> str:
    """Format one query and its identification results."""
    lines = [title]

    if predictions:
        for prediction in sorted(
            predictions,
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        ):
            lines.append(
                format_short_prediction(
                    prediction=prediction,
                    report_level=report_level,
                    delimiter=delimiter,
                )
            )
    else:
        lines.append("     Not identified")

    return "\n".join(lines)

def classify_markers(
    markers: Dict[str, str],
    bundle: ModelBundle,
    sensitivity: float,
    specificity: float,
    max_no_data_for_nn: float,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Traverse every sufficiently supported branch.

    Path score is the minimum edge score along the path. This keeps scores in
    0..1 and prevents a terminal node from receiving high confidence when an
    upstream branch was weak.
    """
    delimiter = str(bundle.get("delimiter", "|"))
    terminal_predictions: List[Dict[str, Any]] = []
    intermediate_predictions: List[Dict[str, Any]] = []
    trace: List[str] = []

    # (node, path_tuple, path_score)
    stack: List[Tuple[Dict[str, Any], Tuple[str, ...], float]] = [
        (bundle["tree"], tuple(), 1.0)
    ]

    while stack:
        node, path, path_score = stack.pop()
        children = node.get("children", {})

        # A reached node with no descendants is terminal.
        if not children and path:
            # A true terminal cluster must satisfy sensitivity.  If it does
            # not, its nearest eligible ancestor can still be returned later
            # as an intermediate-level identification.
            if path_score > sensitivity:
                terminal_predictions.append({
                    "path": delimiter.join(path),
                    "score": path_score,
                    "terminal": True,
                })
            continue

        scores, method, entropy = node_scores(
            node=node,
            markers=markers,
            bundle=bundle,
            max_no_data_for_nn=max_no_data_for_nn,
        )
        node_name = delimiter.join(path) if path else "<root>"
        entropy_text = "" if entropy is None else f", entropy={entropy:.4f}"
        trace.append(f"{node_name}: {method}{entropy_text}")

        for child, local_score in sorted(
            scores.items(), key=lambda item: item[1], reverse=True
        ):
            local_score = max(0.0, min(1.0, float(local_score)))
            new_path_score = min(path_score, local_score)

            if child == TERMINAL_LABEL:
                if path and new_path_score > sensitivity:
                    terminal_predictions.append({
                        "path": delimiter.join(path),
                        "score": new_path_score,
                        "terminal": True,
                    })
                elif path and new_path_score > specificity:
                    intermediate_predictions.append({
                        "path": delimiter.join(path),
                        "score": new_path_score,
                        "terminal": False,
                    })
                continue

            if local_score <= specificity:
                continue

            new_path = path + (child,)
            child_node = children.get(child)

            if child_node is not None and child_node.get("children"):
                intermediate_predictions.append({
                    "path": delimiter.join(new_path),
                    "score": new_path_score,
                    "terminal": False,
                })

            if child_node is not None:
                stack.append((child_node, new_path, new_path_score))

    # Deduplicate, retaining best score for each path.
    def deduplicate(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        best: Dict[str, Dict[str, Any]] = {}
        for record in records:
            path = record["path"]
            if path not in best or record["score"] > best[path]["score"]:
                best[path] = record
        return sorted(best.values(), key=lambda r: r["score"], reverse=True)

    terminals = deduplicate(terminal_predictions)
    if terminals:
        return terminals, trace

    intermediates = [
        record
        for record in deduplicate(intermediate_predictions)
        if float(record.get("score", 0.0)) > sensitivity
    ]
    if intermediates:
        max_depth = max(
            record["path"].count(delimiter) + 1
            for record in intermediates
        )
        deepest = [
            record
            for record in intermediates
            if record["path"].count(delimiter) + 1 == max_depth
        ]
        return sorted(
            deepest,
            key=lambda record: record["score"],
            reverse=True,
        ), trace

    return [], trace


def genome_name_from_path(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".gz"):
        name = name[:-3]
    return Path(name).stem


def main() -> int:
    args = parse_args()

    for name, value in (
        ("--max_no_data_for_NN", args.max_no_data_for_NN),
        ("--max_no_data_allowed", args.max_no_data_allowed),
    ):
        if not 0.0 <= value <= 100.0:
            raise ValueError(f"{name} must be in the range 0..100")

    for name, value in (
        ("--sensitivity", args.sensitivity),
        ("--specificity", args.specificity),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in the range 0..1")

    if args.max_no_data_for_NN > args.max_no_data_allowed:
        raise ValueError(
            "--max_no_data_for_NN cannot exceed --max_no_data_allowed"
        )
    if args.mismatch < 0:
        raise ValueError("--mismatch must be >= 0")
    if args.threads < 1:
        raise ValueError("--threads must be >= 1")
    if args.report_level <= 0:
        raise ValueError("--report_level must be > 0")

    model_path = Path(args.model_file)
    if not model_path.is_absolute():
        model_path = SCRIPT_DIR / model_path
    bundle = load_bundle(model_path)
    backend = _find_blast_backend()
    print(f"BLAST backend: {backend['description']}")
    print(f"Concurrent locus/BLAST workers: {args.threads}")
    print(f"Maximum BLAST substitutions per context: {args.mismatch}")

    input_root = Path(args.input_folder)
    if not input_root.is_absolute():
        input_root = SCRIPT_DIR / input_root
    genome_folder = input_root / args.project_folder
    if not genome_folder.is_dir():
        raise FileNotFoundError(f"Genome project folder does not exist: {genome_folder}")

    output_root = Path(args.output_folder)
    if not output_root.is_absolute():
        output_root = SCRIPT_DIR / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{args.project_folder}.txt"

    genome_files = list(iter_genome_files(genome_folder))
    if not genome_files:
        raise FileNotFoundError(
            f"No FASTA/GenBank files or gzip archives were found in {genome_folder}"
        )

    results: List[Dict[str, Any]] = []
    short_report_records: List[str] = []

    for path in genome_files:
        sequence, record_count = read_genome_sequence(
            path, concatenate_records=args.concatenate_records
        )
        with tempfile.TemporaryDirectory(
            prefix="classifier_blast_", dir=str(output_root)
        ) as tmp_name:
            markers = call_feature_states(
                sequence=sequence,
                bundle=bundle,
                backend=backend,
                tmp_directory=Path(tmp_name),
                mismatch=args.mismatch,
                threads=args.threads,
            )

        missing_count = sum(
            str(value).strip().lower() == MISSING_STATE
            for value in markers.values()
        )
        total_features = len(markers)
        missing_pct = (
            100.0 * missing_count / total_features if total_features else 100.0
        )
        called = total_features - missing_count

        predictions: List[Dict[str, Any]] = []

        row: Dict[str, Any] = {
            "genome": genome_name_from_path(path),
            "file": path.name,
            "records": record_count,
            "called_features": called,
            "missing_features": missing_count,
            "missing_percent": f"{missing_pct:.2f}",
            "best_prediction": "",
            "best_score": "",
            "predictions": "",
            "status": "",
            "method_trace": "",
        }

        if missing_pct > args.max_no_data_allowed:
            row["best_prediction"] = "Not identified"
            row["status"] = (
                "Not identified because too few classifier markers were recovered. "
                "The sequence may be low quality, highly incomplete, too divergent from "
                "the training set, or may belong to an organism outside the scope of this classifier."
            )
        else:
            predictions, trace = classify_markers(
                markers=markers,
                bundle=bundle,
                sensitivity=args.sensitivity,
                specificity=args.specificity,
                max_no_data_for_nn=args.max_no_data_for_NN,
            )
            row["method_trace"] = " | ".join(trace)

            if predictions:
                predictions = sorted(
                    predictions, key=lambda record: record["score"], reverse=True
                )
                row["best_prediction"] = predictions[0]["path"]
                row["best_score"] = f"{predictions[0]['score']:.6f}"
                row["predictions"] = "; ".join(
                    f"{record['path']} ({record['score']:.6f})"
                    for record in predictions
                )
                row["status"] = (
                    "Terminal identification"
                    if predictions[0].get("terminal")
                    else "Intermediate-level identification"
                )
            else:
                row["best_prediction"] = "Not identified"
                row["status"] = (
                    "Not identified: no terminal or intermediate cluster reached "
                    "the requested sensitivity/specificity thresholds."
                )

        results.append(row)

        if args.report == "short":
            # For classifier.py the title is always the original query filename.
            short_report_records.append(
                format_short_record(
                    title=path.name,
                    predictions=predictions,
                    report_level=args.report_level,
                    delimiter=str(bundle.get("delimiter", "|")),
                )
            )

        print(
            f"{path.name}: called {called}/{total_features} markers "
            f"({100.0 - missing_pct:.1f}% recovered) -> {row['best_prediction']}"
        )

    columns = [
        "genome", "file", "records", "called_features", "missing_features",
        "missing_percent", "best_prediction", "best_score", "predictions",
        "status", "method_trace",
    ]
    if args.report == "detailed":
        pd.DataFrame(results, columns=columns).to_csv(
            output_path,
            sep="\t",
            index=False,
            encoding="utf-8-sig",
        )
    else:
        output_path.write_text(
            "\n".join(short_report_records) + "\n",
            encoding="utf-8-sig",
        )

    print(f"Results saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
