"""Exact optimal small binary-tree learning by subset dynamic programming.

For every reachable subproblem ``(mask, depth_left)`` -- a set of training rows
that reach a node and the depth still available -- we enumerate *every* legal
split (feature, threshold, missing-value direction) whose two children each
receive at least one row, and take the minimum of the best leaf and the best
split.  Memoisation makes the recursion an exact shortest-path computation over
the lattice of row subsets, so the result is a global optimum of

    sum_i cost[i, predict(x_i)]  +  alpha * number_of_internal_nodes

This is deliberately not a greedy top-down heuristic: a split is scored through
the optimal subtrees below it, and repeated features on a root-to-leaf path are
allowed.  Floating-point comparisons use a 1e-12 gap (well inside the required
1e-9 optimality tolerance); costs are summed with :func:`math.fsum`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .tree import Leaf, Node, Split, count_internal, predict
from .validate import InputError, validate_training

# Strict-improvement gap; tighter than the required 1e-9 optimality tolerance.
EPS = 1e-12


@dataclass(frozen=True)
class TrainingResult:
    model: Node
    objective: float
    predictions: list[int]
    internal_nodes: int
    n_states: int


def _bits(mask: int):
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def leaf_cost(mask: int, costs: list[list[float]]) -> tuple[float, int]:
    c0 = math.fsum(costs[i][0] for i in _bits(mask))
    c1 = math.fsum(costs[i][1] for i in _bits(mask))
    # Exact float equality is the tie (both labels then have identical loss);
    # break it deterministically toward label 0 without tolerating a real gap.
    return (c0, 0) if c0 <= c1 else (c1, 1)


def fit(rows, costs, thresholds, max_depth=2, alpha=0.05) -> TrainingResult:
    rows, costs, thresholds, max_depth, alpha, n_features = validate_training(
        rows, costs, thresholds, max_depth, alpha
    )
    n = len(rows)

    # Precompute, per candidate split, the full-dataset left/right masks.
    # Missing rows are parked separately so each node can pick its own
    # missing-value direction independently.
    split_table: list[tuple[int, int, bool, int, int]] = []
    for f, cand in enumerate(thresholds):
        missing_mask = 0
        present_left = 0
        for i, row in enumerate(rows):
            v = row[f]
            if v is None:
                missing_mask |= 1 << i
        for t in cand:
            present_left = 0
            for i, row in enumerate(rows):
                v = row[f]
                if v is not None and v <= t:
                    present_left |= 1 << i
            present_right = ((1 << n) - 1) ^ present_left ^ missing_mask
            split_table.append((f, t, True, present_left | missing_mask, present_right))
            split_table.append(
                (f, t, False, present_left, present_right | missing_mask)
            )
    # Deterministic candidate order: feature, threshold, then missing-left first.
    split_table.sort(key=lambda s: (s[0], s[1], 0 if s[2] else 1))

    cache: dict[tuple[int, int], tuple[Node, float]] = {}

    def solve(mask: int, depth_left: int) -> tuple[Node, float]:
        key = (mask, depth_left)
        cached = cache.get(key)
        if cached is not None:
            return cached

        best_cost, best_label = leaf_cost(mask, costs)
        best_node: Node = Leaf(best_label)

        if depth_left > 0:
            for f, t, missing_left, base_left, base_right in split_table:
                left_mask = mask & base_left
                right_mask = mask & base_right
                if not left_mask or not right_mask:
                    continue  # both child subtrees need >= 1 arriving row
                left_node, left_val = solve(left_mask, depth_left - 1)
                right_node, right_val = solve(right_mask, depth_left - 1)
                val = left_val + right_val + alpha
                if val < best_cost - EPS:
                    best_cost = val
                    best_node = Split(f, t, missing_left, left_node, right_node)

        cache[key] = (best_node, best_cost)
        return best_node, best_cost

    root_mask = (1 << n) - 1
    model, objective_value = solve(root_mask, max_depth)
    predictions = [predict(model, row) for row in rows]
    return TrainingResult(
        model=model,
        objective=objective_value,
        predictions=predictions,
        internal_nodes=count_internal(model),
        n_states=len(cache),
    )


def objective(model: Node, rows, costs, alpha) -> float:
    """Recompute the objective from predictions (independent accounting path)."""
    total = math.fsum(costs[i][predict(model, row)] for i, row in enumerate(rows))
    return total + float(alpha) * count_internal(model)


# --------------------------------------------------------------------------- #
# Independent verifier: brute-force enumeration of every legal tree.
# Used by the test suite on small instances; no memoisation, no shared code
# with the DP recurrence beyond leaf accounting and the objective definition.
# --------------------------------------------------------------------------- #
def brute_force_optimum(
    rows, costs, thresholds, max_depth, alpha
) -> tuple[float, int]:
    """Enumerate all legal trees up to ``max_depth`` and return (objective, count).

    Serves as an independent mathematical check of the DP optimum.  Tractable for
    the small instances used in verification (few candidate splits, depth <= 2,
    or very small row counts).
    """
    rows, costs, thresholds, max_depth, alpha, _ = validate_training(
        rows, costs, thresholds, max_depth, alpha
    )
    n = len(rows)
    all_rows = (1 << n) - 1

    candidates: list[tuple[int, int, bool, int, int]] = []
    for f, cand in enumerate(thresholds):
        missing_mask = 0
        for i, row in enumerate(rows):
            if row[f] is None:
                missing_mask |= 1 << i
        for t in cand:
            pl = 0
            for i, row in enumerate(rows):
                v = row[f]
                if v is not None and v <= t:
                    pl |= 1 << i
            pr = all_rows ^ pl ^ missing_mask
            candidates.append((f, t, True, pl | missing_mask, pr))
            candidates.append((f, t, False, pl, pr | missing_mask))

    best = math.inf
    counted = 0

    def enumerate_trees(mask: int, depth_left: int) -> float:
        nonlocal best, counted
        local_best, _ = leaf_cost(mask, costs)
        counted += 1
        if local_best < best:
            best = local_best
        if depth_left == 0:
            return local_best
        for f, t, ml, bl, br in candidates:
            lm, rm = mask & bl, mask & br
            if not lm or not rm:
                continue
            lv = enumerate_trees(lm, depth_left - 1)
            rv = enumerate_trees(rm, depth_left - 1)
            val = lv + rv + alpha
            counted += 1
            if val < local_best:
                local_best = val
            if val < best:
                best = val
        return local_best

    root_best = enumerate_trees(all_rows, max_depth)
    return root_best, counted
