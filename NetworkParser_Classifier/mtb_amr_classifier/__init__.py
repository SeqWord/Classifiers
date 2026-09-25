"""MTB_AMR_Classifier: predict MTB hierarchy paths from a trained model.

This package is the inference-only counterpart of NetworkParser query.
It loads a trained model bundle (or registry) and predicts the hierarchy
path of a sample from FASTQ, FASTA, or VCF input.
"""

from .config import NetworkParserConfig
from .inputs import detect_input_type
from .predict import load_config, predict_hierarchy, write_hierarchy_paths
from .query_engine import NetworkParserQueryEngine
from .model_bundle import load_bundle, query_bundle

ClassifierConfig = NetworkParserConfig

__version__ = "0.1.0"

__all__ = [
    "NetworkParserConfig",
    "ClassifierConfig",
    "detect_input_type",
    "load_config",
    "predict_hierarchy",
    "write_hierarchy_paths",
    "NetworkParserQueryEngine",
    "load_bundle",
    "query_bundle",
    "__version__",
]
