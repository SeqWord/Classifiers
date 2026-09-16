"""Map pickled NetworkParser class names onto this package.

Trained .npb bundles store estimators as ``network_parser.neural_network.LR``
(and RF, DT, ...). MTB_AMR_Classifier keeps those classes but under a new
package name, so unpickling requires this alias.
"""

from __future__ import annotations

import logging
import sys
import warnings

logger = logging.getLogger(__name__)
_EXPECTED_WARNING_FILTERS_INSTALLED = False
_SKLEARN_VERSION_NOTICE_EMITTED = False


def silence_expected_runtime_warnings() -> None:
    """Hide warnings that are expected for this bundled model."""
    global _EXPECTED_WARNING_FILTERS_INSTALLED
    if _EXPECTED_WARNING_FILTERS_INSTALLED:
        return

    warnings.filterwarnings(
        "ignore",
        message=r"LEGACY CALLABILITY MODE:.*",
        category=UserWarning,
    )
    try:
        from sklearn.exceptions import InconsistentVersionWarning
    except Exception:  # pragma: no cover
        InconsistentVersionWarning = None  # type: ignore
    if InconsistentVersionWarning is not None:
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

    _EXPECTED_WARNING_FILTERS_INSTALLED = True


def note_sklearn_unpickle_version_once() -> None:
    global _SKLEARN_VERSION_NOTICE_EMITTED
    if _SKLEARN_VERSION_NOTICE_EMITTED:
        return
    _SKLEARN_VERSION_NOTICE_EMITTED = True
    logger.info(
        "Loading sklearn 1.3 estimators on a newer sklearn runtime. "
        "This is expected for the bundled model."
    )


def install_network_parser_pickle_aliases() -> None:
    import mtb_amr_classifier as pkg
    from mtb_amr_classifier import neural_network

    silence_expected_runtime_warnings()
    note_sklearn_unpickle_version_once()
    sys.modules.setdefault("network_parser", pkg)
    sys.modules.setdefault("network_parser.neural_network", neural_network)
    if not hasattr(pkg, "neural_network"):
        pkg.neural_network = neural_network
