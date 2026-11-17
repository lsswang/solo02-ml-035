"""Tests for exact optimal-tree learning, with independent brute-force checks."""
import math
import random

import pytest

from triage import brute_force_optimum, fit, objective, predict, tree_to_dict
from triage.trainer import leaf_cost
from triage.tree import Leaf, Split, count_internal
from triage.validate import InputError


def _recomputed(res, rows, costs, alpha):
    return objective(res.model, rows, costs, alpha)


def test_xor_greedy_trap_is_solved_globally():
    # Greedy top-down cannot represent XOR from these splits; the globally
    # optimal depth-2 tree has zero misclassification cost.
    rows = [[0, 0], [0, 1], [1, 0], [1, 1]]
    costs = [[0, 1], [1, 0], [1, 0], [0, 1]]
    thresholds = [[0], [0]]
    res = fit(rows, costs, thresholds, 2, 0.05)
    assert res.predictions == [0, 1, 1, 0]
    assert res.internal_nodes == 3
    assert res.objective == pytest.approx(0.15, abs=1e-9)
    assert _recomputed(res, rows, costs, 0.05) == pytest.approx(res.objective, abs=1e-12)
    # independent full-tree enumeration reaches the same optimum
    bf, _ = brute_force_optimum(rows, costs, thresholds, 2, 0.05)
    assert bf == pytest.approx(res.objective, abs=1e-12)


def test_heterogeneous_costs_choose_cheaper_misclassification():
    # second feature is constant and offers no useful split.
    rows = [[0, 0], [1, 0]]
    # row 0 wants class 0, row 1 wants class 1 (cross-loss 10 each).
    costs = [[0, 10], [10, 0]]
    # A perfect split costs alpha=2.0 and removes all 10 of loss, so it wins
    # over the best single leaf (which must pay 10).
    res = fit(rows, costs, [[0], []], 1, 2.0)
    assert isinstance(res.model, Split)
    assert res.predictions == [0, 1]
    assert res.objective == pytest.approx(2.0, abs=1e-9)
    bf, _ = brute_force_optimum(rows, costs, [[0], []], 1, 2.0)
    assert bf == pytest.approx(2.0, abs=1e-12)
    # With the largest allowed alpha the leaf becomes optimal when the total
    # separable loss is smaller than one split's penalty.
    cheap = [[0, 1], [1, 0]]  # best single leaf pays only 1
    res_leaf = fit(rows, cheap, [[0], []], 1, 2.0)
    assert isinstance(res_leaf.model, Leaf)
    assert res_leaf.objective == pytest.approx(1.0, abs=1e-9)


def test_heterogeneous_costs_split_worthwhile():
    rows = [[0, 0], [5, 0]]
    costs = [[0, 10], [10, 0]]
    # one split costs alpha=0.5 but removes 10 of loss -> worth it.
    res = fit(rows, costs, [[2], []], 1, 0.5)
    assert isinstance(res.model, Split)
    assert res.model.feature == 0
    assert res.predictions == [0, 1]
    assert res.objective == pytest.approx(0.5, abs=1e-9)
    bf, _ = brute_force_optimum(rows, costs, [[2], []], 1, 0.5)
    assert bf == pytest.approx(0.5, abs=1e-12)


def test_zero_depth_always_returns_best_leaf():
    rows = [[-1, 2], [3, -4], [0, 0]]
    costs = [[1, 5], [4, 1], [2, 3]]
    res = fit(rows, costs, [[0], [0]], 0, 0.5)
    assert isinstance(res.model, Leaf)
    assert res.internal_nodes == 0
    expected, label = leaf_cost(0b111, costs)
    assert res.model.label == label
    assert res.objective == pytest.approx(expected, abs=1e-12)
    assert all(p == label for p in res.predictions)
    bf, _ = brute_force_optimum(rows, costs, [[0], [0]], 0, 0.5)
    assert bf == pytest.approx(expected, abs=1e-12)


def test_missing_direction_is_chosen_per_node():
    # Present rows both satisfy x <= 2 and want class 0; the missing row wants
    # class 1.  Routing None to the right (with the high present row) keeps the
    # left child pure class 0 and is strictly cheaper than missing-left.
    rows = [[None, 0], [0, 0], [5, 0]]
    costs = [[4, 0], [0, 4], [4, 0]]
    res = fit(rows, costs, [[2], []], 1, 0.01)
    split = res.model
    assert isinstance(split, Split)
    assert split.feature == 0
    assert split.missing_left is False  # None goes right -> class 1
    assert predict(split, [None, 0]) == 1
    assert predict(split, [0, 0]) == 0
    assert predict(split, [5, 0]) == 1
    # missing-left would force a 4-cost misclassification; optimum is just alpha
    assert res.objective == pytest.approx(0.01, abs=1e-12)
    bf, _ = brute_force_optimum(rows, costs, [[2], []], 1, 0.01)
    assert bf == pytest.approx(res.objective, abs=1e-12)


def test_missing_left_default_routing():
    rows = [[None, 0], [0, 0], [5, 0]]
    costs = [[0, 5], [0, 5], [5, 0]]  # missing & low -> 0, high -> 1
    res = fit(rows, costs, [[2], []], 1, 0.01)
    assert isinstance(res.model, Split)
    assert res.model.missing_left is True
    assert predict(res.model, [None, 0]) == 0
    bf, _ = brute_force_optimum(rows, costs, [[2], []], 1, 0.01)
    assert bf == pytest.approx(res.objective, abs=1e-12)


def test_repeated_feature_on_a_path_is_allowed():
    # Only one informative feature; nested splits on it at two thresholds carve
    # three regions, which a single split cannot.  The second feature is constant.
    rows = [[-5, 0], [0, 0], [5, 0]]
    costs = [[0, 9], [9, 0], [0, 9]]  # labels 0,1,0 along the feature
    res = fit(rows, costs, [[-2, 2], []], 2, 0.01)
    assert res.internal_nodes == 2
    features = {s.feature for s in _all_splits(res.model)}
    assert features == {0}
    assert res.predictions == [0, 1, 0]
    bf, _ = brute_force_optimum(rows, costs, [[-2, 2], []], 2, 0.01)
    assert bf == pytest.approx(res.objective, abs=1e-12)


def test_split_requires_both_children_nonempty():
    # Threshold above every value cannot split anything (right child empty);
    # the trainer must never emit such a node.
    rows = [[0, 0], [1, 0]]
    costs = [[0, 5], [5, 0]]
    res = fit(rows, costs, [[20], []], 1, 0.01)
    assert isinstance(res.model, Leaf)
    bf, _ = brute_force_optimum(rows, costs, [[20], []], 1, 0.01)
    assert bf == pytest.approx(res.objective, abs=1e-12)


def test_no_candidate_thresholds_means_leaf():
    rows = [[0, 0], [1, 1]]
    costs = [[0, 5], [5, 0]]
    res = fit(rows, costs, [[], []], 3, 0.01)
    assert isinstance(res.model, Leaf)


def test_predictions_match_objective_accounting():
    rng = random.Random(7)
    rows, costs = [], []
    for _ in range(10):
        rows.append([rng.choice([None, *range(-5, 6)]) for _ in range(3)])
        costs.append([round(rng.random() * 20, 3), round(rng.random() * 20, 3)])
    thresholds = [sorted(rng.sample(range(-5, 6), 3)) for _ in range(3)]
    res = fit(rows, costs, thresholds, 3, 0.37)
    manual = math.fsum(costs[i][p] for i, p in enumerate(res.predictions))
    assert manual + 0.37 * res.internal_nodes == pytest.approx(res.objective, abs=1e-9)
    assert res.predictions == [predict(res.model, r) for r in rows]


@pytest.mark.parametrize("seed", range(25))
def test_random_small_instances_match_brute_force(seed):
    rng = random.Random(seed)
    nf = rng.randint(2, 3)
    n = rng.randint(3, 7)
    rows = [[rng.choice([None, *range(-3, 4)]) for _ in range(nf)] for _ in range(n)]
    costs = [[round(rng.random() * 20, 4), round(rng.random() * 20, 4)] for _ in range(n)]
    thresholds = [sorted(rng.sample(range(-3, 4), rng.randint(1, 3))) for _ in range(nf)]
    depth = rng.randint(0, 2)
    alpha = round(0.01 + rng.random() * 1.99, 4)
    res = fit(rows, costs, thresholds, depth, alpha)
    bf, _ = brute_force_optimum(rows, costs, thresholds, depth, alpha)
    assert res.objective == pytest.approx(bf, abs=1e-9)
    assert _recomputed(res, rows, costs, alpha) == pytest.approx(bf, abs=1e-12)
    # structural validity: every internal node has two non-empty child row sets
    _assert_children_nonempty(res.model, rows)


def _all_splits(node):
    if isinstance(node, Leaf):
        return []
    return [node] + _all_splits(node.left) + _all_splits(node.right)


def _assert_children_nonempty(node, rows):
    if isinstance(node, Leaf):
        return
    left = [r for r in rows if (r[node.feature] is None and node.missing_left)
            or (r[node.feature] is not None and r[node.feature] <= node.threshold)]
    right = [r for r in rows if r not in left]
    assert left and right
    _assert_children_nonempty(node.left, left)
    _assert_children_nonempty(node.right, right)


def test_max_size_instance_runs_and_is_consistent():
    rng = random.Random(11)
    nf, n = 6, 32
    rows = [[rng.choice([None, *range(-20, 21)]) for _ in range(nf)] for _ in range(n)]
    costs = [[round(rng.random() * 20, 3), round(rng.random() * 20, 3)] for _ in range(n)]
    thresholds = [sorted(rng.sample(range(-20, 21), 4)) for _ in range(nf)]
    res = fit(rows, costs, thresholds, 3, 0.5)
    assert _recomputed(res, rows, costs, 0.5) == pytest.approx(res.objective, abs=1e-9)
    assert tree_to_dict(res.model)  # serialisable
    assert count_internal(res.model) <= 7


def test_training_input_validation():
    good_rows = [[0, 1], [1, 0]]
    good_costs = [[0.0, 1.0], [1.0, 0.0]]
    good_th = [[0], [0]]
    with pytest.raises(InputError):
        fit([], good_costs, good_th, 2, 0.1)                       # no rows
    with pytest.raises(InputError):
        fit(good_rows, good_costs, good_th, 4, 0.1)                # depth too large
    with pytest.raises(InputError):
        fit(good_rows, good_costs, good_th, 2, 0.0)                # alpha too small
    with pytest.raises(InputError):
        fit(good_rows, good_costs, good_th, 2, 2.5)               # alpha too large
    with pytest.raises(InputError):
        fit([[21, 0], [1, 0]], good_costs, good_th, 2, 0.1)       # value out of range
    with pytest.raises(InputError):
        fit([[1.5, 0], [1, 0]], good_costs, good_th, 2, 0.1)      # non-integer value
    with pytest.raises(InputError):
        fit(good_rows, [[-1.0, 1.0], [1.0, 0.0]], good_th, 2, 0.1)  # negative cost
    with pytest.raises(InputError):
        fit(good_rows, good_costs, good_th, 2, float("nan"))      # non-finite alpha
    with pytest.raises(InputError):
        fit(good_rows, good_costs, [[0, 0], [0]], 2, 0.1)         # duplicate threshold
    with pytest.raises(InputError):
        fit(good_rows, good_costs, [[21], [0]], 2, 0.1)           # bad threshold
    with pytest.raises(InputError):
        fit([[0], [1]], [[0.0, 1.0], [1.0, 0.0]], [[0]], 2, 0.1)  # one feature
    with pytest.raises(InputError):
        fit([[0, 0, 0, 0, 0, 0, 0]], [[0.0, 1.0]],
            [[0]] * 7, 2, 0.1)                                    # seven features
