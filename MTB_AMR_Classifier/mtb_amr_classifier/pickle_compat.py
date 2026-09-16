"""Map pickled NetworkParser class names onto this package.

Trained .npb bundles store estimators as ``network_parser.neural_network.LR``
(and RF, DT, ...). MTB_AMR_Classifier keeps those classes but under a new
package name, so unpickling requires this alias.
"""

from __future__ import annotations

import sys


def install_network_parser_pickle_aliases() -> None:
    import mtb_amr_classifier as pkg
    from mtb_amr_classifier import neural_network

    sys.modules.setdefault("network_parser", pkg)
    sys.modules.setdefault("network_parser.neural_network", neural_network)
    if not hasattr(pkg, "neural_network"):
        pkg.neural_network = neural_network
