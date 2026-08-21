#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Portable interface for hierarchical classifier PKL bundles.

Place this module in ``../lib`` next to ``NeuralNetwork.py`` and
``nn__algorithm_selector.py``.  ``training/run.py`` serializes a ModelBundle
instance, and the deployed classifier package receives a copy of this module so
the PKL can be loaded there as well.
"""

from __future__ import annotations

import copy
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


BUNDLE_INTERFACE_VERSION = 2


def _unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


class ModelBundle:
    """Data + interface stored in ``model.pkl``.

    The object deliberately exposes both dictionary-like access (for simple
    scripts) and explicit methods (for stable inter-program communication).
    """

    def __init__(self, data: Dict[str, Any]):
        self.data = dict(data)
        self.data.setdefault("interface_version", BUNDLE_INTERFACE_VERSION)
        self.data.setdefault("created_at", datetime.now().astimezone().isoformat(timespec="seconds"))

    # ------------------------------------------------------------------
    # Minimal mapping compatibility
    # ------------------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    # ------------------------------------------------------------------
    # Requested public interface
    # ------------------------------------------------------------------
    def get_creation_date(self) -> str:
        """Return ISO-formatted creation date/time."""
        return str(self.data.get("created_at", ""))

    def get_project_name(self) -> str:
        """Return the training project name."""
        return str(self.data.get("project", ""))

    def get_author(self) -> str:
        """Return model author metadata."""
        return str(self.data.get("author", ""))

    def get_description(self) -> str:
        """Return model description metadata."""
        return str(self.data.get("description", ""))

    def get_version(self) -> str:
        """Return user-supplied model/data version metadata."""
        return str(self.data.get("version", ""))

    def get_contact_details(self) -> str:
        """Return author/contact metadata."""
        return str(self.data.get("contact_details", ""))

    def get_metadata(self) -> Dict[str, str]:
        """Return the human-facing model metadata as one dictionary."""
        return {
            "creation_date": self.get_creation_date(),
            "project": self.get_project_name(),
            "author": self.get_author(),
            "description": self.get_description(),
            "version": self.get_version(),
            "contact_details": self.get_contact_details(),
        }

    def get_training_genomes(self) -> List[str]:
        """Return all genome identifiers used for training."""
        return [str(v) for v in self.data.get("training", {}).get("genome_names", [])]

    def get_clusters(self, delimiter: str = "|") -> List[str]:
        """Return all unique hierarchical cluster paths.

        Both internal and terminal cluster paths are returned.  The root is not
        included.  Cluster components are re-joined with the caller-supplied
        delimiter.
        """
        if delimiter == "":
            raise ValueError("delimiter cannot be empty")

        paths: List[str] = []
        for labels in self.data.get("label_paths", []):
            labels = [str(v) for v in labels if str(v) != ""]
            for depth in range(1, len(labels) + 1):
                paths.append(delimiter.join(labels[:depth]))
        return _unique_preserve_order(paths)

    def get_cluster_genomes(self, delimiter: str = "|") -> List[str]:
        """Return ``label_A|label_B|...|genome_name`` strings."""
        if delimiter == "":
            raise ValueError("delimiter cannot be empty")

        genomes = self.get_training_genomes()
        label_paths = self.data.get("label_paths", [])
        if len(genomes) != len(label_paths):
            raise ValueError(
                "Bundle metadata is inconsistent: genome_names and label_paths "
                "have different lengths."
            )

        result: List[str] = []
        for labels, genome in zip(label_paths, genomes):
            parts = [str(v) for v in labels if str(v) != ""]
            parts.append(str(genome))
            result.append(delimiter.join(parts))
        return result

    def get_training_matrix(self) -> List[List[str]]:
        """Return a deep copy of the initial training matrix."""
        return copy.deepcopy(self.data.get("training_matrix", []))

    def get_model_tree(self) -> Dict[str, Any]:
        """Return a deep copy of the hierarchical model/profile tree."""
        return copy.deepcopy(self.data.get("tree", {}))

    def get_cluster_profile(
        self,
        cluster_path: str | Sequence[str] | None = None,
        delimiter: str = "|",
    ) -> Dict[str, Any]:
        """Return the stored allele-profile matrix for one cluster node.

        ``cluster_path=None`` or an empty path returns the root profile.
        """
        tree = self.data.get("tree") or {}
        if cluster_path is None or cluster_path == "":
            node = tree
        else:
            path = self._normalize_path(cluster_path, delimiter)
            node = self._find_node(tree, path)
        return copy.deepcopy(node.get("similarity_profile", {}))

    def get_hierarchy(self, format: str = "text", delimiter: str = "|") -> str:
        """Return hierarchy as ``text`` or ``newick``.

        Splitting nodes are annotated with the neural-network / classifier
        algorithm stored at that node.
        """
        fmt = str(format).strip().lower()
        if fmt in {"text", "txt", "tab", "indented"}:
            return self._hierarchy_text(delimiter=delimiter)
        if fmt in {"newick", "nwk"}:
            return self._hierarchy_newick()
        raise ValueError("format must be 'text' or 'newick'")

    def get_terminal_cluster_count(self) -> int:
        """Return number of distinct terminal cluster paths in training data."""
        paths = {
            tuple(str(v) for v in labels if str(v) != "")
            for labels in self.data.get("label_paths", [])
        }
        paths.discard(tuple())
        return len(paths)

    def get_interface_info(self) -> Dict[str, Any]:
        """Return a compact description of the bundle API/data."""
        return {
            "interface_version": self.data.get("interface_version", BUNDLE_INTERFACE_VERSION),
            "created_at": self.get_creation_date(),
            "project": self.get_project_name(),
            "author": self.get_author(),
            "description": self.get_description(),
            "version": self.get_version(),
            "contact_details": self.get_contact_details(),
            "training_genomes": len(self.get_training_genomes()),
            "terminal_clusters": self.get_terminal_cluster_count(),
            "hierarchy_levels": int(self.data.get("hierarchy_levels", 0)),
            "algorithm_mode": self.data.get("algorithm_mode"),
            "min_cluster_size": self.data.get("min_cluster_size"),
            "max_entropy": self.data.get("max_entropy"),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, file_path: str | Path) -> str:
        """Save this bundle to a PKL file and return its path."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return str(path)

    @classmethod
    def load(cls, file_path: str | Path) -> "ModelBundle":
        """Load a ModelBundle from a PKL file."""
        path = Path(file_path)
        with path.open("rb") as handle:
            obj = pickle.load(handle)
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            # Backward compatibility with early dictionary bundles.
            return cls(obj)
        raise TypeError(f"{path} does not contain a compatible ModelBundle")

    # ------------------------------------------------------------------
    # Hierarchy presentation
    # ------------------------------------------------------------------
    @staticmethod
    def _node_algorithm(node: Dict[str, Any]) -> str:
        model = node.get("model")
        if isinstance(model, dict):
            return str(model.get("algorithm", "") or "")
        return ""

    def _hierarchy_text(self, delimiter: str = "|") -> str:
        tree = self.data.get("tree") or {}
        lines: List[str] = []

        def node_suffix(node: Dict[str, Any]) -> str:
            algorithm = self._node_algorithm(node)
            if algorithm:
                return f"  [algorithm={algorithm}]"
            if node.get("children"):
                reason = str(node.get("model_skip_reason", "") or "")
                return f"  [algorithm=None; fallback=allele-profile{'; ' + reason if reason else ''}]"
            return ""

        def walk(node: Dict[str, Any], depth: int, label: str) -> None:
            lines.append("\t" * depth + f"{label}{node_suffix(node)}")
            for child_label, child_node in node.get("children", {}).items():
                walk(child_node, depth + 1, str(child_label))

        lines.append(f"<root>{node_suffix(tree)}")
        for child_label, child_node in tree.get("children", {}).items():
            walk(child_node, 1, str(child_label))
        return "\n".join(lines)

    @staticmethod
    def _newick_escape(label: str) -> str:
        text = str(label)
        if any(ch in text for ch in " ,:;()[]'"):
            return "'" + text.replace("'", "''") + "'"
        return text

    def _hierarchy_newick(self) -> str:
        tree = self.data.get("tree") or {}

        def render(node: Dict[str, Any], label: str) -> str:
            children = [
                render(child_node, str(child_label))
                for child_label, child_node in node.get("children", {}).items()
            ]
            algorithm = self._node_algorithm(node)
            node_label = str(label)
            if algorithm:
                node_label = f"{node_label}[{algorithm}]"
            elif children:
                node_label = f"{node_label}[None/profile]"
            escaped = self._newick_escape(node_label)
            if children:
                return "(" + ",".join(children) + ")" + escaped
            return escaped

        children = [
            render(child_node, str(child_label))
            for child_label, child_node in tree.get("children", {}).items()
        ]
        root_algorithm = self._node_algorithm(tree)
        if root_algorithm:
            root_label = f"<root>[{root_algorithm}]"
        elif children:
            root_label = "<root>[None/profile]"
        else:
            root_label = "<root>"
        if children:
            return "(" + ",".join(children) + ")" + self._newick_escape(root_label) + ";"
        return self._newick_escape(root_label) + ";"

    # ------------------------------------------------------------------
    # Grafting
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_path(path: str | Sequence[str], delimiter: str) -> Tuple[str, ...]:
        if isinstance(path, str):
            parts = [part.strip() for part in path.split(delimiter) if part.strip()]
        else:
            parts = [str(part).strip() for part in path if str(part).strip()]
        if not parts:
            raise ValueError("terminal_path cannot be empty")
        return tuple(parts)

    @staticmethod
    def _find_node(tree: Dict[str, Any], path: Sequence[str]) -> Dict[str, Any]:
        node = tree
        for part in path:
            children = node.get("children", {})
            if part not in children:
                raise KeyError(f"Cluster path does not exist: {'|'.join(path)}")
            node = children[part]
        return node

    @staticmethod
    def _prefix_tree_paths(node: Dict[str, Any], prefix: Tuple[str, ...]) -> Dict[str, Any]:
        node = copy.deepcopy(node)

        def walk(current: Dict[str, Any], current_path: Tuple[str, ...]) -> None:
            current["path"] = list(current_path)
            for child_label, child_node in current.get("children", {}).items():
                walk(child_node, current_path + (str(child_label),))

        walk(node, prefix)
        return node

    def _validate_graft_compatibility(self, child: "ModelBundle") -> None:
        if self.data.get("data_type") != child.data.get("data_type"):
            raise ValueError("Cannot graft bundles with different data types.")

        if list(self.data.get("feature_titles", [])) != list(child.data.get("feature_titles", [])):
            raise ValueError(
                "Cannot graft bundles with different feature-title sets/order. "
                "This restriction keeps allelic-state calling and stored pipelines coherent."
            )

        parent_ref = str(self.data.get("reference_name", "") or "")
        child_ref = str(child.data.get("reference_name", "") or "")
        if parent_ref and child_ref and parent_ref != child_ref:
            raise ValueError(
                f"Cannot graft bundles built against different references: "
                f"{parent_ref!r} != {child_ref!r}"
            )

    def graft(
        self,
        child_bundle: "ModelBundle" | str | Path,
        terminal_path: str | Sequence[str],
        *,
        delimiter: str = "|",
        new_project_name: str | None = None,
    ) -> "ModelBundle":
        """Refine a terminal cluster with the hierarchy from another bundle.

        The child bundle is expected to have been trained on the SAME genomes
        that currently occupy ``terminal_path`` in the parent bundle.  Their
        old terminal labels are replaced by ``terminal_path + child_labels``.

        This strict rule prevents duplicated genomes and prevents a cluster from
        becoming both terminal and internal after grafting.  Both source bundles
        remain unchanged; a NEW ModelBundle is returned.
        """
        child = (
            ModelBundle.load(child_bundle)
            if isinstance(child_bundle, (str, Path))
            else child_bundle
        )
        if not isinstance(child, ModelBundle):
            raise TypeError("child_bundle must be a ModelBundle or path to one")

        self._validate_graft_compatibility(child)
        target_path = self._normalize_path(terminal_path, delimiter)

        parent_genomes = self.get_training_genomes()
        parent_labels = [list(v) for v in self.data.get("label_paths", [])]
        if len(parent_genomes) != len(parent_labels):
            raise ValueError(
                "Parent bundle metadata is inconsistent: genome_names and "
                "label_paths have different lengths."
            )

        target_indices = [
            i
            for i, labels in enumerate(parent_labels)
            if tuple(str(v) for v in labels if str(v) != "") == target_path
        ]
        if not target_indices:
            raise ValueError(
                f"No training genomes terminate at cluster {delimiter.join(target_path)!r}."
            )

        target_genomes = [parent_genomes[i] for i in target_indices]
        child_genomes = child.get_training_genomes()
        child_labels = [list(v) for v in child.data.get("label_paths", [])]

        if len(child_genomes) != len(child_labels):
            raise ValueError(
                "Child bundle metadata is inconsistent: genome_names and "
                "label_paths have different lengths."
            )

        if set(target_genomes) != set(child_genomes):
            only_parent = sorted(set(target_genomes) - set(child_genomes))
            only_child = sorted(set(child_genomes) - set(target_genomes))
            raise ValueError(
                "The child bundle must contain exactly the genomes present in "
                f"terminal cluster {delimiter.join(target_path)!r}. "
                f"Missing from child: {only_parent[:10]}; "
                f"unexpected in child: {only_child[:10]}."
            )

        child_label_by_genome = {
            genome: [str(v) for v in labels if str(v) != ""]
            for genome, labels in zip(child_genomes, child_labels)
        }

        result = copy.deepcopy(self)
        target = self._find_node(result.data["tree"], target_path)

        if target.get("children"):
            raise ValueError(
                f"Target {delimiter.join(target_path)} is not terminal; "
                "it already has children."
            )

        child_root = copy.deepcopy(child.data.get("tree") or {})
        target["model"] = child_root.get("model")
        target["model_skip_reason"] = child_root.get("model_skip_reason", "")
        target["default_child"] = child_root.get("default_child")
        target["similarity_profile"] = copy.deepcopy(
            child_root.get("similarity_profile", target.get("similarity_profile", {}))
        )
        target["child_profiles"] = copy.deepcopy(child_root.get("child_profiles", {}))
        target["child_sample_counts"] = copy.deepcopy(
            child_root.get("child_sample_counts", {})
        )
        target["children"] = {}
        for child_label, child_node in child_root.get("children", {}).items():
            target["children"][child_label] = self._prefix_tree_paths(
                child_node,
                target_path + (str(child_label),),
            )

        # Replace the terminal membership paths of the target genomes.
        refined_paths = [list(v) for v in parent_labels]
        for i in target_indices:
            genome = parent_genomes[i]
            refined_paths[i] = list(target_path) + child_label_by_genome[genome]
        result.data["label_paths"] = refined_paths

        # Update the first-column hierarchy headings in the stored initial
        # training matrix, while preserving all feature cells as originally
        # supplied by the parent training matrix.
        combined_matrix = result.get_training_matrix()
        if combined_matrix and len(combined_matrix) == len(parent_genomes) + 1:
            stored_delimiter = str(result.data.get("delimiter", "|"))
            for i in target_indices:
                row_index = i + 1
                if row_index >= len(combined_matrix) or not combined_matrix[row_index]:
                    continue
                genome = parent_genomes[i]
                heading = refined_paths[i] + [genome]
                combined_matrix[row_index][0] = stored_delimiter.join(heading)
            result.data["training_matrix"] = combined_matrix

        result.data["hierarchy_levels"] = max(
            int(result.data.get("hierarchy_levels", 0)),
            len(target_path) + int(child.data.get("hierarchy_levels", 0)),
        )
        result.data["created_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        if new_project_name is not None:
            result.data["project"] = str(new_project_name)

        result.data.setdefault("graft_history", []).append({
            "parent_project": self.get_project_name(),
            "child_project": child.get_project_name(),
            "terminal_path": delimiter.join(target_path),
            "refined_genomes": len(target_genomes),
            "created_at": result.data["created_at"],
        })
        return result

