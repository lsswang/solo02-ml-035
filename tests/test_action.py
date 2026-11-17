"""Tests for actionable counterfactuals: exact B&B vs enumeration vs MILP."""
import itertools
import random
from fractions import Fraction

import pytest

from triage import counterfactual, parse_tree, predict, tree_to_dict
from triage.action import (
    ActionError,
    _frac,
    _path_constraint,
    solve_leaf_enumeration,
    solve_leaf_exact,
    solve_leaf_milp,
)
from triage.tree import target_leaves
from triage.validate import InputError

ATOL, RTOL = 1e-8, 1e-9


def _scalar(z, x, up, down, fee):
    if z == x:
        return 0.0
    return up * max(z - x, 0) + down * max(x - z, 0) + fee


def _total_cost(z, req):
    x = req["current"]
    return sum(
        _scalar(z[j], x[j], req["up_cost"][j], req["down_cost"][j], req["change_fee"][j])
        for j in range(len(x))
    )


def _domain_lists(req):
    out = []
    for dom in req["domains"]:
        if "interval" in dom:
            lo, hi = dom["interval"]
            out.append(list(range(lo, hi + 1)))
        else:
            out.append(list(dom["values"]))
    return out


def _brute_force(tree, req):
    model = parse_tree(tree, len(req["current"]), 6)
    target = req["target"]
    domains = _domain_lists(req)
    best = None
    for z in itertools.product(*domains):
        z = list(z)
        if predict(model, z) != target:
            continue
        if any(sum(a * v for a, v in zip(row, z)) > b
               for row, b in zip(req["constraints"]["A"], req["constraints"]["b"])):
            continue
        c = _total_cost(z, req)
        if best is None or c < best[0] or (c == best[0] and z < best[1]):
            best = (c, z)
    return best


# A small external tree on two features (depth 2), with a known routing.
TREE = {
    "feature": 0, "threshold": 0, "missing_left": True,
    "left": {
        "feature": 1, "threshold": 0, "missing_left": False,
        "left": {"label": 0}, "right": {"label": 1},
    },
    "right": {
        "feature": 1, "threshold": 2, "missing_left": True,
        "left": {"label": 1}, "right": {"label": 0},
    },
}


def test_discrete_domain_and_basic_action():
    req = {
        "current": [0, 0], "target": 1,
        "domains": [{"interval": [0, 3]}, {"values": [0, 2, 3]}],
        "up_cost": [1, 2], "down_cost": [2, 1], "change_fee": [0.2, 0.3],
        "constraints": {"A": [], "b": []},
    }
    res = counterfactual(TREE, req)
    bf = _brute_force(TREE, req)
    assert res["status"] == "optimal"
    assert res["after_prediction"] == 1
    assert res["total_cost"] == pytest.approx(bf[0], abs=ATOL)
    assert res["action"] == bf[1]
    # per-feature costs reconcile with the total under the required tolerances
    assert sum(res["per_feature_cost"]) == pytest.approx(res["total_cost"], abs=ATOL)
    for j, c in enumerate(res["per_feature_cost"]):
        assert c == pytest.approx(_scalar(res["action"][j], req["current"][j],
                                          req["up_cost"][j], req["down_cost"][j],
                                          req["change_fee"][j]), abs=ATOL)


def test_joint_resource_constraint_changes_the_optimum():
    # Without the constraint the cheapest action is (1,0); the constraint
    # z0 - z1 <= -1 rules that out and forces (0,2).
    base = {
        "current": [0, 0], "target": 1,
        "domains": [{"interval": [0, 3]}, {"values": [0, 2, 3]}],
        "up_cost": [1, 2], "down_cost": [2, 1], "change_fee": [0.2, 0.3],
    }
    free = dict(base, constraints={"A": [], "b": []})
    limited = dict(base, constraints={"A": [[1, -1]], "b": [-1]})
    r_free = counterfactual(TREE, free)
    r_lim = counterfactual(TREE, limited)
    assert r_free["action"] == [1, 0]
    assert r_lim["action"] == [0, 2]
    assert r_lim["total_cost"] == pytest.approx(4.3, abs=ATOL)
    # the returned action really obeys the joint constraint
    z = r_lim["action"]
    assert z[0] - z[1] <= -1
    assert r_lim["total_cost"] == pytest.approx(_brute_force(TREE, limited)[0], abs=ATOL)


def test_infeasible_within_domains_is_reported_not_rounded():
    # Every target leaf conflicts with the resource constraint inside the box.
    req = {
        "current": [0, 0], "target": 1,
        "domains": [{"values": [0, 1]}, {"values": [0, 1]}],
        "up_cost": [1, 1], "down_cost": [1, 1], "change_fee": [0, 0],
        # force z1 = 0 via z1 <= 0, then z0 <= -1 impossible for z0 in {0,1}
        "constraints": {"A": [[0, 1], [1, -1]], "b": [0, -1]},
    }
    res = counterfactual(TREE, req)
    assert res["status"] == "infeasible"
    assert res["infeasible"] is True
    assert res["action"] is None
    assert _brute_force(TREE, req) is None
    # solver status is distinguished from an unfinished search: every leaf is
    # independently confirmed infeasible by both MILP and exact enumeration.
    for report in res["leaf_reports"]:
        assert report["milp"] == "infeasible"
        assert report["enumeration"] == "infeasible"
        assert report["branch_bound"] == "infeasible"


def test_zero_action_is_optimal_when_current_is_already_target():
    req = {
        "current": [0, 3], "target": 1,  # routed left (f0<=0) then f1>0 -> leaf 1
        "domains": [{"interval": [-5, 5]}, {"values": [1, 3, 5]}],
        "up_cost": [3, 4], "down_cost": [2, 1], "change_fee": [1, 1],
        "constraints": {"A": [[1, 1]], "b": [10]},
    }
    assert predict(parse_tree(TREE, 2, 6), req["current"]) == 1
    res = counterfactual(TREE, req)
    assert res["action"] == [0, 3]
    assert res["total_cost"] == 0.0
    assert res["per_feature_cost"] == [0.0, 0.0]


def test_current_point_need_not_lie_in_domain():
    req = {
        "current": [10, 10], "target": 0,
        "domains": [{"interval": [0, 2]}, {"values": [0, 2]}],
        "up_cost": [1, 1], "down_cost": [5, 5], "change_fee": [2, 2],
        "constraints": {"A": [], "b": []},
    }
    res = counterfactual(TREE, req)
    bf = _brute_force(TREE, req)
    assert res["status"] == "optimal"
    assert res["action"] == bf[1]
    assert res["total_cost"] == pytest.approx(bf[0], abs=ATOL)


def test_fixed_increase_only_decrease_only_domains():
    # feature 0 fixed, feature 1 increase-only, feature 2 decrease-only.
    tree3 = {
        "feature": 0, "threshold": 0, "missing_left": True,
        "left": {
            "feature": 1, "threshold": 1, "missing_left": True,
            "left": {"label": 0},
            "right": {
                "feature": 2, "threshold": 5, "missing_left": True,
                "left": {"label": 1}, "right": {"label": 0},
            },
        },
        "right": {"label": 0},
    }
    req = {
        "current": [0, 0, 9], "target": 1,
        "domains": [
            {"values": [0]},          # fixed
            {"interval": [0, 20]},    # increase-only relative to 0
            {"interval": [-20, 9]},   # decrease-only relative to 9
        ],
        "up_cost": [1, 1, 1], "down_cost": [1, 1, 1], "change_fee": [0, 0, 0],
        "constraints": {"A": [[1, 1, 1]], "b": [12]},
    }
    res = counterfactual(tree3, req)
    bf = _brute_force(tree3, req)
    assert res["action"][0] == 0            # fixed feature never moves
    assert res["action"][1] >= 0 and res["action"][2] <= 9
    assert res["action"] == bf[1]
    assert res["total_cost"] == pytest.approx(bf[0], abs=ATOL)


def test_three_solvers_agree_on_every_leaf_randomly():
    rng = random.Random(23)
    for _ in range(40):
        m = rng.randint(2, 4)
        current = [rng.randint(-5, 5) for _ in range(m)]
        domains = []
        for j in range(m):
            if rng.random() < 0.5:
                lo = rng.randint(-6, 0)
                hi = rng.randint(0, 6)
                domains.append(list(range(lo, hi + 1)))
            else:
                k = rng.randint(1, 5)
                domains.append(sorted(rng.sample(range(-6, 7), k)))
        ups = [_frac(round(rng.random() * 5, 2)) for _ in range(m)]
        downs = [_frac(round(rng.random() * 5, 2)) for _ in range(m)]
        fees = [_frac(round(rng.random() * 2, 2)) for _ in range(m)]
        n_cons = rng.randint(0, 3)
        constraints = []
        for _ in range(n_cons):
            coef = [rng.randint(-3, 3) for _ in range(m)]
            constraints.append((coef, rng.randint(-6, 6)))
        # keep enumeration tractable
        if sum(len(d) for d in domains) > 12:
            continue
        s1 = solve_leaf_exact(domains, constraints, current, ups, downs, fees)
        s2 = solve_leaf_enumeration(domains, constraints, current, ups, downs, fees)
        s3 = solve_leaf_milp(domains, constraints, current, ups, downs, fees)
        assert s1[0] == s2[0] == s3[0]
        if s1[2] is not None:
            assert abs(float(s1[2]) - float(s2[2])) <= ATOL
            assert abs(float(s1[2]) - float(s3[2])) <= ATOL
            # the B&B point is genuinely feasible and integral
            z = s1[1]
            assert all(isinstance(v, int) for v in z)
            assert all(z[j] in domains[j] for j in range(m))
            for coef, b in constraints:
                assert sum(a * v for a, v in zip(coef, z)) <= b


def test_path_constraint_conversion():
    assert _path_constraint(1, "<=", 3, 4) == ([0, 1, 0, 0], 3)
    assert _path_constraint(1, ">=", 2, 4) == ([0, -1, 0, 0], -2)


def test_external_tree_without_training_data_and_missing_semantics():
    # missing_left defaults to True when omitted; action vectors have no None,
    # but the parsed tree still validates the full format.
    tree = {"feature": 1, "threshold": 5,
            "left": {"label": 0}, "right": {"label": 1}}
    model = parse_tree(tree, 3, 6)
    assert predict(model, [9, 5, None]) == 0
    assert predict(model, [9, 6, None]) == 1
    req = {
        "current": [0, 0, 0], "target": 1,
        "domains": [{"values": [0]}, {"interval": [0, 10]}, {"values": [0]}],
        "up_cost": [1, 2, 1], "down_cost": [1, 1, 1], "change_fee": [0, 0.5, 0],
        "constraints": {"A": [], "b": []},
    }
    res = counterfactual(tree, req)
    assert res["action"][1] == 6
    assert res["after_prediction"] == 1


def test_depth_six_external_tree_accepted():
    # build a depth-6 chain tree, all valid thresholds/features
    node = {"label": 1}
    for d in range(6):
        node = {"feature": d % 3, "threshold": d, "missing_left": True,
                "left": {"label": 0}, "right": node}
    model = parse_tree(node, 3, 6)
    req = {
        "current": [0, 0, 0], "target": 1,
        "domains": [{"interval": [0, 10]} for _ in range(3)],
        "up_cost": [1, 1, 1], "down_cost": [1, 1, 1], "change_fee": [0, 0, 0],
        "constraints": {"A": [], "b": []},
    }
    res = counterfactual(node, req)
    assert res["status"] == "optimal"
    assert predict(model, res["action"]) == 1


def test_returned_action_is_legal_integer_vector():
    req = {
        "current": [-2, 3], "target": 1,
        "domains": [{"interval": [-4, 4]}, {"values": [-3, 0, 3, 4]}],
        "up_cost": [1.5, 0.7], "down_cost": [2.5, 1.2], "change_fee": [0.3, 0.9],
        "constraints": {"A": [[2, -1]], "b": [1]},
    }
    res = counterfactual(TREE, req)
    z = res["action"]
    assert all(isinstance(v, int) for v in z)
    assert z[0] in range(-4, 5) and z[1] in [-3, 0, 3, 4]
    assert 2 * z[0] - z[1] <= 1
    bf = _brute_force(TREE, req)
    assert res["total_cost"] == pytest.approx(bf[0], abs=ATOL)


def test_cost_tolerance_checks_are_float_exact_for_decimals():
    # 0.1-style decimals must be handled without float drift; Fraction path.
    assert _frac(0.1) + _frac(0.2) == Fraction(3, 10)
    req = {
        "current": [0, 0], "target": 1,
        "domains": [{"interval": [0, 2]}, {"values": [0, 1]}],
        "up_cost": [0.1, 0.1], "down_cost": [0.1, 0.1], "change_fee": [0.1, 0.1],
        "constraints": {"A": [], "b": []},
    }
    res = counterfactual(TREE, req)
    bf = _brute_force(TREE, req)
    assert abs(res["total_cost"] - bf[0]) <= ATOL + RTOL * abs(bf[0])


def test_action_input_validation():
    good = {
        "current": [0, 0], "target": 1,
        "domains": [{"interval": [0, 3]}, {"values": [0, 2]}],
        "up_cost": [1, 2], "down_cost": [2, 1], "change_fee": [0.2, 0.3],
        "constraints": {"A": [[1, -1]], "b": [-1]},
    }
    with pytest.raises((InputError, ActionError)):
        bad = dict(good, target=2)
        counterfactual(TREE, bad)
    with pytest.raises((InputError, ActionError)):
        bad = dict(good, current=[0, 21])  # current out of range
        counterfactual(TREE, bad)
    with pytest.raises((InputError, ActionError)):
        bad = dict(good, domains=[{"interval": [3, 0]}, {"values": [0, 2]}])
        counterfactual(TREE, bad)
    with pytest.raises((InputError, ActionError)):
        bad = dict(good, domains=[{"interval": [0, 10001]}, {"values": [0, 2]}])
        counterfactual(TREE, bad)
    with pytest.raises((InputError, ActionError)):
        bad = dict(good, up_cost=[21, 2])
        counterfactual(TREE, bad)
    with pytest.raises((InputError, ActionError)):
        bad = dict(good, change_fee=[float("inf"), 0.3])
        counterfactual(TREE, bad)
    with pytest.raises((InputError, ActionError)):
        bad = dict(good, constraints={"A": [[1, -1, 0]], "b": [-1]})  # wrong width
        counterfactual(TREE, bad)
    with pytest.raises((InputError, ActionError)):
        bad = dict(good, constraints={"A": [[101, -1]], "b": [-1]})  # coeff out of range
        counterfactual(TREE, bad)
    with pytest.raises((InputError, ActionError)):
        bad = dict(good, domains=[{"values": [1, 1]}, {"values": [0, 2]}])  # duplicate
        counterfactual(TREE, bad)


def test_tree_validation_errors():
    req = {
        "current": [0, 0], "target": 1,
        "domains": [{"interval": [0, 3]}, {"values": [0, 2]}],
        "up_cost": [1, 2], "down_cost": [2, 1], "change_fee": [0.2, 0.3],
        "constraints": {"A": [], "b": []},
    }
    with pytest.raises(ActionError):
        counterfactual({"feature": 5, "threshold": 0,  # bad feature ref
                        "left": {"label": 0}, "right": {"label": 1}}, req)
    with pytest.raises(ActionError):
        counterfactual({"feature": 0, "threshold": 21,  # bad threshold
                        "left": {"label": 0}, "right": {"label": 1}}, req)
    with pytest.raises(ActionError):
        counterfactual({"feature": 0, "threshold": 0,
                        "left": {"label": 2}, "right": {"label": 1}}, req)  # bad label
    with pytest.raises(ActionError):
        leaf = {"label": 0}
        deep = leaf
        for _ in range(7):  # exceeds depth 6
            deep = {"feature": 0, "threshold": 0, "missing_left": True,
                    "left": {"label": 0}, "right": deep}
        counterfactual(deep, req)
    with pytest.raises(ActionError):
        # a shared split node makes a DAG rather than a tree and must be refused
        shared = {"feature": 0, "threshold": 0, "missing_left": True,
                  "left": {"label": 0}, "right": {"label": 1}}
        counterfactual({"feature": 1, "threshold": 0, "missing_left": True,
                        "left": shared, "right": shared}, req)


def test_depth_zero_external_tree():
    leaf0 = {"label": 0}
    req = {
        "current": [0, 0], "target": 1,
        "domains": [{"interval": [0, 3]}, {"values": [0, 2]}],
        "up_cost": [1, 2], "down_cost": [2, 1], "change_fee": [0.2, 0.3],
        "constraints": {"A": [], "b": []},
    }
    assert counterfactual(leaf0, req)["status"] == "infeasible"
    ok = counterfactual({"label": 1}, req)
    assert ok["action"] == [0, 0] and ok["total_cost"] == 0.0


def test_constraints_key_optional_and_further_validation():
    base = {
        "current": [0, 0], "target": 1,
        "domains": [{"interval": [0, 3]}, {"values": [0, 2]}],
        "up_cost": [1, 2], "down_cost": [2, 1], "change_fee": [0.2, 0.3],
    }
    # missing constraints key is treated as no resource limits
    res = counterfactual(TREE, base)
    assert res["status"] == "optimal"

    def must_fail(mut):
        req = dict(base)
        mut(req)
        with pytest.raises((InputError, ActionError)):
            counterfactual(TREE, req)

    must_fail(lambda r: r.update(domains=[{"values": []}, {"values": [0, 2]}]))
    must_fail(lambda r: r.update(target=True))
    must_fail(lambda r: r.update(domains=[{"values": list(range(9))}, {"values": [0]}]))
    must_fail(lambda r: r.update(up_cost=[float("nan"), 2]))
    must_fail(lambda r: r.update(down_cost=[1, float("inf")]))
    must_fail(lambda r: r.update(constraints={"A": [[1, 1]], "b": []}))
    must_fail(lambda r: r.update(constraints={"A": [[1, 0]] * 9, "b": [0] * 9}))
    must_fail(lambda r: r.update(constraints={"A": [[1.5, 0]], "b": [0]}))
    must_fail(lambda r: r.update(domains=[{"values": [10001]}, {"values": [0]}]))
    with pytest.raises(ActionError):
        counterfactual([1, 2], base)  # tree not a dict


def test_unknown_status_distinct_from_infeasible(monkeypatch):
    # When the MILP engine hits a limit (status 1) the public result must be
    # "unknown", never a false "infeasible".
    import triage.action as mod

    def forced_unknown(domains, constraints, current, ups, downs, fees):
        return "unknown", None, None

    monkeypatch.setattr(mod, "solve_leaf_milp", forced_unknown)
    # disable the exact cross-check path which would otherwise re-solve
    monkeypatch.setattr(mod, "CROSSCHECK_LIMIT", -1)
    req = {
        "current": [0, 0], "target": 1,
        "domains": [{"interval": [0, 5]}, {"interval": [0, 5]}],
        "up_cost": [1, 1], "down_cost": [1, 1], "change_fee": [0, 0],
        "constraints": {"A": [], "b": []},
    }
    res = counterfactual(TREE, req)
    assert res["status"] == "unknown"
    assert res["infeasible"] is False
    assert res["action"] is None


def test_branch_bound_budget_returns_unknown_not_infeasible():
    # The exact B&B distinguishes an unfinished search from proven infeasibility.
    import triage.action as mod
    original = mod.NODE_BUDGET
    mod.NODE_BUDGET = 0
    try:
        status, z, cost, nodes = mod.solve_leaf_exact(
            [[0, 1, 2]], [([1], 10)], [0],
            [mod._frac(1)], [mod._frac(1)], [mod._frac(0)])
        assert status == "unknown"
    finally:
        mod.NODE_BUDGET = original
