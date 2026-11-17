"""Reproducible demonstration of exact tree learning and counterfactual actions.

Run with::

    python3 demo.py

It exercises, against the fixtures in ``fixtures/demo_cases.json``:

1. ordinary global-optimal training (XOR greedy trap) with the objective and
   training predictions, cross-checked by an independent full-tree enumeration;
2. an actionable counterfactual on the freshly learned model with discrete
   domains and a joint resource constraint, cross-checked by Cartesian
   enumeration and a SciPy HiGHS MILP;
3. an external tree used directly for an action, with no training data;
4. a combined scenario where the action domains, costs and resource limit stay
   fixed but the training evidence changes: the learned tree changes and the
   minimum-cost action changes with it, re-verified by predicting on each tree.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

from triage import (
    brute_force_optimum,
    counterfactual,
    fit,
    objective,
    predict,
    tree_to_dict,
)
from triage.tree import parse_tree
from triage.validate import validate_action_request

HERE = Path(__file__).parent
ATOL, RTOL = 1e-8, 1e-9


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def independent_enumerate(tree_dict, request: dict):
    """Full Cartesian enumeration: an independent optimum for small domains."""
    model = parse_tree(tree_dict, len(request["current"]), 6)
    clean = validate_action_request(request, len(request["current"]))
    best = None
    for z in itertools.product(*clean["domains"]):
        z = list(z)
        if predict(model, z) != clean["target"]:
            continue
        if any(sum(a * v for a, v in zip(row, z)) > b
               for row, b in zip(clean["A"], clean["b"])):
            continue
        cost = sum(
            (clean["up_cost"][j] * max(z[j] - clean["current"][j], 0)
             + clean["down_cost"][j] * max(clean["current"][j] - z[j], 0)
             + clean["change_fee"][j] * (z[j] != clean["current"][j]))
            for j in range(len(z))
        )
        if best is None or cost < best[0] or (cost == best[0] and z < best[1]):
            best = (cost, z)
    return best


def main() -> None:
    cases = json.loads((HERE / "fixtures" / "demo_cases.json").read_text())

    # ------------------------------------------------------------------ #
    section("1. Ordinary global-optimal training on the XOR greedy trap")
    # ------------------------------------------------------------------ #
    ord_case = cases["ordinary"]
    result = fit(ord_case["rows"], ord_case["costs"], ord_case["thresholds"],
                 ord_case["max_depth"], ord_case["alpha"])
    learned = tree_to_dict(result.model)
    print("learned tree      :", json.dumps(learned, ensure_ascii=False))
    print("objective (DP)    :", result.objective)
    print("internal nodes    :", result.internal_nodes)
    print("predictions       :", result.predictions)
    recomputed = objective(result.model, ord_case["rows"], ord_case["costs"],
                           ord_case["alpha"])
    print("objective (re-sum):", recomputed)
    assert abs(recomputed - result.objective) <= 1e-12
    bf_value, bf_count = brute_force_optimum(
        ord_case["rows"], ord_case["costs"], ord_case["thresholds"],
        ord_case["max_depth"], ord_case["alpha"])
    print(f"independent brute-force optimum: {bf_value} "
          f"(enumerated {bf_count} leaf/tree subproblems)")
    assert abs(bf_value - result.objective) <= 1e-9

    # ------------------------------------------------------------------ #
    section("2. Counterfactual on the learned tree (discrete + joint limit)")
    # ------------------------------------------------------------------ #
    request = ord_case["request"]
    answer = counterfactual(learned, request)
    print("request           :", json.dumps(request, ensure_ascii=False))
    print("solver status     :", answer["status"])
    print("action z          :", answer["action"])
    print("prediction x -> z :", answer["before_prediction"], "->",
          answer["after_prediction"])
    print("total / per-feature cost:", answer["total_cost"],
          answer["per_feature_cost"])
    print("per-leaf MILP / branch&bound / enumeration cross-check:")
    for report in answer["leaf_reports"]:
        print("  ", json.dumps(report))
    enum_best = independent_enumerate(learned, request)
    print("independent Cartesian enumeration:", enum_best)
    assert enum_best is not None
    assert abs(enum_best[0] - answer["total_cost"]) <= ATOL + RTOL * abs(enum_best[0])
    assert answer["action"] == enum_best[1]
    # explicit legality re-check of the returned integer vector
    assert predict(result.model, answer["action"]) == request["target"]
    for j, v in enumerate(answer["action"]):
        assert isinstance(v, int) and v in validate_action_request(request, 2)["domains"][j]
    assert sum(a * v for a, v in zip([1, -1], answer["action"])) <= -1

    # ------------------------------------------------------------------ #
    section("3. External tree action without any training samples")
    # ------------------------------------------------------------------ #
    external = cases["external_tree"]
    ext_request = {
        "current": [-2, 3], "target": 1,
        "domains": [{"interval": [-4, 4]}, {"values": [-3, 0, 3, 4]}],
        "up_cost": [1.5, 0.7], "down_cost": [2.5, 1.2], "change_fee": [0.3, 0.9],
        "constraints": {"A": [[2, -1]], "b": [1]},
    }
    ext_answer = counterfactual(external, ext_request)
    print("external tree     :", json.dumps(external, ensure_ascii=False))
    print("action            :", ext_answer["action"], "cost",
          ext_answer["total_cost"], "prediction",
          ext_answer["before_prediction"], "->", ext_answer["after_prediction"])
    ext_enum = independent_enumerate(external, ext_request)
    print("enumeration check :", ext_enum)
    assert abs(ext_enum[0] - ext_answer["total_cost"]) <= ATOL
    assert ext_answer["action"] == ext_enum[1]

    # ------------------------------------------------------------------ #
    section("4. Combined demo: fixed action rules, changed training evidence")
    # ------------------------------------------------------------------ #
    combo = cases["combined"]
    req = combo["request"]
    print("fixed action request (identical in both runs):")
    print(json.dumps(req, ensure_ascii=False))
    outcomes = {}
    for name, cost_key in (("A", "evidence_A_costs"), ("B", "evidence_B_costs")):
        trained = fit(combo["rows"], combo[cost_key], combo["thresholds"],
                      combo["max_depth"], combo["alpha"])
        bf, _ = brute_force_optimum(combo["rows"], combo[cost_key],
                                    combo["thresholds"], combo["max_depth"],
                                    combo["alpha"])
        assert abs(bf - trained.objective) <= 1e-9
        ans = counterfactual(tree_to_dict(trained.model), req)
        # re-verify the action by predicting on this run's actual learned tree
        verified = predict(trained.model, ans["action"])
        outcomes[name] = (trained, ans, verified)
        print(f"\nevidence {name}:")
        print("  learned tree    :", json.dumps(tree_to_dict(trained.model)))
        print("  objective / BF  :", trained.objective, "/", bf)
        print("  current predicts:", predict(trained.model, req["current"]))
        print("  best action     :", ans["action"], "cost", ans["total_cost"],
              "per-feature", ans["per_feature_cost"])
        print("  re-predict on learned tree:", verified, "(target", req["target"], ")")
        assert verified == req["target"]
        # resource and domain legality
        assert sum(ans["action"]) <= 6
        clean = validate_action_request(req, 2)
        assert all(v in d for v, d in zip(ans["action"], clean["domains"]))
    assert outcomes["A"][1]["action"] != outcomes["B"][1]["action"]
    print("\nThe action rules never changed; the different evidence learned a")
    print("different target region, so the minimum-cost action moved from",
          outcomes["A"][1]["action"], "to", outcomes["B"][1]["action"], ".")

    section("All demonstration checks passed.")


if __name__ == "__main__":
    main()
