# MTB_AMR_Classifier

Predict the **hierarchy path** of a *Mycobacterium tuberculosis* sample from a trained NetworkParser model.

This is the inference-only counterpart of NetworkParser `query`. It does not train models. It loads a portable `.npb` bundle (or a training registry) and walks the saved lineage → AMR route for a new sample.

Supported sample types:

| Type | Input |
|---|---|
| `vcf` | One VCF/gVCF, or a directory of VCFs |
| `fasta` | FASTA file or directory |
| `fastq` | Directory of paired-end FASTQ files |
| `matrix` | CSV/TSV feature matrix |
| `auto` | Detect the type from the input |

> Predictions are research outputs, not validated clinical diagnoses. Model bundles contain Python pickle objects — only load `.npb` files from trusted training runs.

## Workflow

```text
trained NetworkParser model + sample (FASTQ / FASTA / VCF)
        → encode into the trained marker space
        → walk the hierarchy
        → predicted path (for example Lineage=L4.3 / AMR_binary=resistant)
```

## Install

```bash
cd MTB_AMR_Classifier
conda env create -f environment.yml
conda activate mtb_amr_classifier
```

FASTQ mode also needs `bwa`, `samtools`, and `bcftools` on `PATH` (included in the Conda environment). FASTA BLAST mapping needs `blast`.

## Example: bundled model and VCFs

This repository includes a trained hierarchy model and five holdout VCFs that
the model predicted correctly (lineage → AMR binary → resistance profile).

| Path | Contents |
|---|---|
| `model/networkparser_model_bundle.npb` | Trained NetworkParser hierarchy bundle |
| `input/*.vcf.gz` | Five example samples |
| `input/manifest.tsv` | True and previously predicted paths |
| `data/reference/H37Rv.fasta` | H37Rv reference used by this model |

From the repository root:

```bash
python -m mtb_amr_classifier predict \
  --model model/networkparser_model_bundle.npb \
  --sample input \
  --ref_fasta data/reference/H37Rv.fasta \
  --output_dir results/example
```

`--sample input` treats the whole directory as VCF input. To run one sample:

```bash
python -m mtb_amr_classifier predict \
  --model model/networkparser_model_bundle.npb \
  --sample input/ERR038739.vcf.gz \
  --ref_fasta data/reference/H37Rv.fasta \
  --output_dir results/example_one
```

Open `results/example/hierarchy_paths.tsv` for the predicted routes. These
samples should match `input/manifest.tsv`:

| Sample | Expected path |
|---|---|
| `ERR036226` | lineage 4 / Sensitive / Sensitive |
| `ERR038739` | lineage 4 / resistant / MDR |
| `ERR108129` | lineage 2 / Sensitive / Sensitive |
| `ERR1873395` | lineage 2 / resistant / Pre_XDR |
| `ERR037491` | lineage 1 / resistant / Mono |

The reference is `data/reference/H37Rv.fasta` (H37Rv, the same genome this
model was trained on). Keep query samples on that coordinate system.

## Predict other samples

```bash
python -m mtb_amr_classifier predict \
  --model model/networkparser_model_bundle.npb \
  --sample /path/to/sample.vcf.gz \
  --ref_fasta data/reference/H37Rv.fasta \
  --output_dir /path/to/results
```

Paired FASTQ directory:

```bash
python -m mtb_amr_classifier predict \
  --model /path/to/networkparser_model_bundle.npb \
  --sample /path/to/fastq_dir \
  --input-type fastq \
  --ref_fasta data/reference/H37Rv.fasta \
  --output_dir /path/to/results
```

FASTA:

```bash
python -m mtb_amr_classifier predict \
  --model /path/to/networkparser_model_bundle.npb \
  --sample /path/to/sample.fasta \
  --input-type fasta \
  --ref_fasta data/reference/H37Rv.fasta \
  --output_dir /path/to/results
```

`--input-type auto` is the default. `--model` / `--bundle` and `--sample` / `--genomic` are aliases, matching NetworkParser query flags.

Use `--help` for FASTQ, review-guard, and config options:

```bash
python -m mtb_amr_classifier predict --help
```

## Python API

```python
from mtb_amr_classifier import predict_hierarchy

predictions = predict_hierarchy(
    model="model/networkparser_model_bundle.npb",
    sample="input",
    output_dir="results/example",
    ref_fasta="data/reference/H37Rv.fasta",
)
print(predictions[["sample_id", "predicted_hierarchy_path"]])
```

## Outputs

| File | Purpose |
|---|---|
| `hierarchy_paths.tsv` | Compact predicted route per sample |
| `query_predictions.csv` | Full prediction table |
| `query_predictions_compact.tsv` | Compact prediction table |
| `query_predictions_readable.html` | Human-readable report |
| `query_route_audit.json` | Hierarchy route and fallback audit |
| `query_alignment_summary.json` | Marker recovery / callability summary |

The path column looks like:

```text
Lineage_clean=L4.3.3 / AMR_binary=resistant / Resistance_Profile_Collapsed=HRZE
```

## Model source

Train the model with NetworkParser:

```bash
python -m network_parser.cli train-hierarchy \
  --genomic /path/to/training_vcfs \
  --meta /path/to/metadata.csv \
  --hierarchy_labels Lineage_clean AMR_binary \
  --ref_fasta /path/to/H37Rv.fasta \
  --output_dir /path/to/training_results
```

That run writes `networkparser_model_bundle.npb`. The example in this repository
uses the `Hierarchy_Lineage_AMR_Resistance_Profile_01` bundle copied to
`model/networkparser_model_bundle.npb`.

## Tests

```bash
pytest -q
```
