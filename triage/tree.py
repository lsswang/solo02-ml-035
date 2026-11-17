"""Binary decision tree representation, validation and inference.

Tree format (shared by the trainer and external callers):

* leaf:   {"label": 0} or {"label": 1}
* split:  {"feature": f, "threshold": t, "missing_left": bool,
           "left": <tree>, "right": <tree>}

A split routes row ``z`` with ``z[f] is not None`` to the left subtree iff
``z[f] <= t``.  A missing value (``None``) goes left when ``missing_left`` is
true, right otherwise.  ``missing_left`` is optional and defaults to ``True``
for backward compatibility with the starter code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Optional

MAX_FEATURES = 6
MIN_FEATURES = 2
THRESH_MIN = -20
THRESH_MAX = 20
MAX_TRAIN_DEPTH = 3
MAX_EXTERNAL_DEPTH = 6


class TreeValidationError(ValueError):
    """Raised for malformed tree references, labels, thresholds or depth."""


@dataclass(frozen=True)
class Leaf:
    label: int

    def to_dict(self) -> dict:
        return {"label": self.label}


@dataclass(frozen=True)
class Split:
    feature: int
    threshold: int
    missing_left: bool
    left: "Node"
    right: "Node"

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "threshold": self.threshold,
            "missing_left": self.missing_left,
            "left": tree_to_dict(self.left),
            "right": tree_to_dict(self.right),
        }


Node = Leaf | Split


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def parse_tree(obj: Any, n_features: int, max_depth: int) -> Node:
    """Validate an external dict tree and return an immutable node tree.

    Validates feature references, integer thresholds within range, leaf labels,
    node shape, acyclicity and depth.  Missing ``missing_left`` defaults to true.
    """
    if not _is_int(n_features) or not (MIN_FEATURES <= n_features <= MAX_FEATURES):
        raise TreeValidationError(
            f"n_features must be an integer in [{MIN_FEATURES},{MAX_FEATURES}]"
        )
    if not _is_int(max_depth) or max_depth < 0 or max_depth > MAX_EXTERNAL_DEPTH:
        raise TreeValidationError(
            f"max_depth must be an integer in [0,{MAX_EXTERNAL_DEPTH}]"
        )

    seen: set[int] = set()

    def walk(node: Any, depth_left: int, path: str) -> Node:
        if not isinstance(node, dict):
            raise TreeValidationError(f"tree node at {path} must be a dict")
        if "label" in node:
            label = node["label"]
            if not _is_int(label) or label not in (0, 1):
                raise TreeValidationError(f"leaf label at {path} must be 0 or 1")
            if any(k != "label" for k in node):
                raise TreeValidationError(f"leaf at {path} carries split fields")
            return Leaf(label)
        # split node
        if depth_left <= 0:
            raise TreeValidationError(f"tree exceeds max depth {max_depth} at {path}")
        node_id = id(node)
        if node_id in seen:
            raise TreeValidationError("tree contains a shared/cyclic reference")
        seen.add(node_id)
        for key in ("feature", "threshold", "left", "right"):
            if key not in node:
                raise TreeValidationError(f"split at {path} missing field {key!r}")
        feature = node["feature"]
        threshold = node["threshold"]
        if not _is_int(feature) or not (0 <= feature < n_features):
            raise TreeValidationError(
                f"feature at {path} must be an integer in [0,{n_features - 1}]"
            )
        if not _is_int(threshold) or not (THRESH_MIN <= threshold <= THRESH_MAX):
            raise TreeValidationError(
                f"threshold at {path} must be an integer in "
                f"[{THRESH_MIN},{THRESH_MAX}]"
            )
        missing_left = node.get("missing_left", True)
        if not isinstance(missing_left, bool):
            raise TreeValidationError(f"missing_left at {path} must be a bool")
        left = walk(node["left"], depth_left - 1, path + "/left")
        right = walk(node["right"], depth_left - 1, path + "/right")
        return Split(feature, threshold, missing_left, left, right)

    return walk(obj, max_depth, "root")


def tree_to_dict(node: Node) -> dict:
    return node.to_dict()


def tree_depth(node: Node) -> int:
    """Number of split levels on the longest root-to-leaf path."""
    if isinstance(node, Leaf):
        return 0
    return 1 + max(tree_depth(node.left), tree_depth(node.right))


def count_internal(node: Node) -> int:
    if isinstance(node, Leaf):
        return 0
    return 1 + count_internal(node.left) + count_internal(node.right)


def predict(node: Node, row: list[Optional[int]]) -> int:
    """Classify one row.  Entries may be ``None`` (missing); action rows never are."""
    cur = node
    while isinstance(cur, Split):
        value = row[cur.feature]
        if value is None:
            go_left = cur.missing_left
        else:
            go_left = value <= cur.threshold
        cur = cur.left if go_left else cur.right
    return cur.label


def iter_splits(node: Node) -> Iterator[Split]:
    if isinstance(node, Split):
        yield node
        yield from iter_splits(node.left)
        yield from iter_splits(node.right)


def target_leaves(node: Node, target: int) -> list[tuple[Node, list[tuple[int, str, int]]]]:
    """Return ``(leaf, path)`` for every leaf whose label equals ``target``.

    Each path entry is ``(feature, "<=" | ">=", bound)`` where the integer
    condition on the (never missing) action vector is ``z[f] <= t`` (left) or
    ``z[f] >= t+1`` (right).  Repeated features on a path simply accumulate.
    """
    out: list[tuple[Node, list[tuple[int, str, int]]]] = []

    def walk(cur: Node, path: list[tuple[int, str, int]]) -> None:
        if isinstance(cur, Leaf):
            if cur.label == target:
                out.append((cur, list(path)))
            return
        path.append((cur.feature, "<=", cur.threshold))
        walk(cur.left, path)
        path[-1] = (cur.feature, ">=", cur.threshold + 1)
        walk(cur.right, path)
        path.pop()

    walk(node, [])
    return out
