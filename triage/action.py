"""Actionable counterfactuals: exact minimum-cost integer action.

Given an external (or freshly trained) binary tree and an action request, find
an integer vector ``z`` that simultaneously

  1. lies in every per-feature actionable domain (closed interval or a small
     set of discrete integer values -- fixed / increase-only / decrease-only
     features are expressed purely through these domains),
  2. satisfies every joint resource constraint ``A z <= b`` on the *full*
     post-action vector,
  3. is classified by the tree as the requested target class,
  4. minimises the separable change cost.

Method
------
Every target-class leaf induces a conjunction of integer path constraints on
``z`` (``z[f] <= t`` or ``z[f] >= t+1``); combined with the resource limits and
the per-feature domains this is one integer optimisation per leaf, and the
overall answer is the cheapest leaf.

The primary solver is a mixed-integer program handed to SciPy's bundled HiGHS
engine (:func:`solve_leaf_milp`): integer ``z_j``, non-negative up/down amounts,
binary change indicators linearised with a big-M, and binary selectors for
discrete set domains.  Two independent exact routes cross-check it on tractable
search spaces:

* :func:`solve_leaf_exact` -- a hand-written branch-and-bound using box-
  consistency (BC4) bound propagation, separable Fraction-cost lower bounds and
  smallest-domain branching, with a node budget that reports ``unknown`` rather
  than a false infeasibility;
* :func:`solve_leaf_enumeration` -- full Cartesian enumeration for tiny spaces.

All cost arithmetic uses :class:`fractions.Fraction` parsed from the exact
decimal input, so the optimum and every cost check are free of float rounding;
the returned integer vector is re-verified against each domain, resource limit,
target prediction and the exact cost before being returned.  Unfinished
searches (``unknown``) are always distinguished from proven ``infeasible``.
"""
from __future__ import annotations

import heapq
import itertools
from fractions import Fraction
from typing import Optional

from .tree import Node, parse_tree, predict, target_leaves
from .validate import InputError, validate_action_request

NODE_BUDGET = 3_000_000


class ActionError(ValueError):
    """Raised for malformed action requests or trees."""


def _frac(x: float) -> Fraction:
    # Exact decimal representation of the validated finite input number.
    return Fraction(repr(float(x)))


# --------------------------------------------------------------------------- #
# Domains: [kind, lo, hi, values].  kind "i" = integer interval, "s" = set.
# --------------------------------------------------------------------------- #
def _initial_domains(domains: list[list[int]]) -> list[list]:
    out = []
    for values in domains:
        lo, hi = values[0], values[-1]
        if len(values) == hi - lo + 1:
            out.append(["i", lo, hi, None])
        else:
            out.append(["s", lo, hi, list(values)])
    return out


def _domain_size(dom: list) -> int:
    if dom[0] == "i":
        return dom[2] - dom[1] + 1
    return len(dom[3])


def _scalar_cost(z: int, x: int, up: Fraction, down: Fraction, fee: Fraction) -> Fraction:
    if z == x:
        return Fraction(0)
    if z > x:
        return up * (z - x) + fee
    return down * (x - z) + fee


def _feature_cost_min(dom: list, x: int, up: Fraction, down: Fraction, fee: Fraction) -> Fraction:
    """Exact minimum feature cost over the domain.

    The piecewise-linear cost has its only kink at ``x``; on an interval the
    minimum is at lo, hi or x.  On a set every candidate is inspected.
    """
    if dom[0] == "i":
        lo, hi = dom[1], dom[2]
        candidates = {lo, hi}
        if lo <= x <= hi:
            candidates.add(x)
        return min(_scalar_cost(z, x, up, down, fee) for z in candidates)
    return min(_scalar_cost(z, x, up, down, fee) for z in dom[3])


def _lower_bound(doms, current, ups, downs, fees) -> Fraction:
    total = Fraction(0)
    for j, dom in enumerate(doms):
        total += _feature_cost_min(dom, current[j], ups[j], downs[j], fees[j])
    return total


# --------------------------------------------------------------------------- #
# Exact constraint propagation for sum_j a_j z_j <= b with integer z_j.
# --------------------------------------------------------------------------- #
def _floor_div(num: int, den: int) -> int:
    """Floor division with a positive denominator."""
    assert den > 0
    return num // den


def _ceil_div(num: int, den: int) -> int:
    """Ceiling division with a positive denominator."""
    assert den > 0
    return -((-num) // den)


def _propagate(doms, constraints) -> bool:
    """Tighten domain bounds until a fixpoint.  Return False if infeasible.

    Classic interval bounds propagation (box consistency) for linear
    inequalities ``sum a_k z_k <= b``.  Isolating variable j,
    ``a_j z_j <= b - sum_{k!=j} a_k z_k``; a value of z_j is feasible when
    *some* assignment to the others satisfies the inequality, so the right-hand
    side is evaluated at its maximum, i.e. with the other terms at their minima:

    * a_j > 0: prune the upper bound
      ``z_j <= floor((b - min_other) / a_j)``;
    * a_j = -p < 0: prune the lower bound
      ``z_j >= ceil((min_other - b) / p)``;

    where ``min_other`` is the minimum of the other linear terms over the
    current boxes.  Every prune only removes values that cannot satisfy that
    constraint under any assignment to the remaining variables, bounds move
    inward only, and finite domains guarantee termination.
    """
    while True:
        snapshot = [(d[1], d[2]) for d in doms]
        new_lo = [lo for lo, _ in snapshot]
        new_hi = [hi for _, hi in snapshot]
        for coeffs, bnd in constraints:
            min_sum = 0
            min_contrib = [0] * len(doms)
            for k, (lo, hi) in enumerate(snapshot):
                a = coeffs[k]
                if a > 0:
                    min_contrib[k] = a * lo
                elif a < 0:
                    min_contrib[k] = a * hi  # negative coefficient: min at hi
                min_sum += min_contrib[k]
            for j, a in enumerate(coeffs):
                if a == 0:
                    continue
                min_other = min_sum - min_contrib[j]
                if a > 0:
                    cand_hi = _floor_div(bnd - min_other, a)
                    if cand_hi < new_hi[j]:
                        new_hi[j] = cand_hi
                else:
                    p = -a
                    cand_lo = _ceil_div(min_other - bnd, p)
                    if cand_lo > new_lo[j]:
                        new_lo[j] = cand_lo

        moved = False
        for j, d in enumerate(doms):
            lo, hi = d[1], d[2]
            if new_lo[j] > lo:
                lo = new_lo[j]
                moved = True
            if new_hi[j] < hi:
                hi = new_hi[j]
                moved = True
            if lo > hi:
                return False
            d[1], d[2] = lo, hi
            if d[0] == "s":
                filtered = [v for v in d[3] if lo <= v <= hi]
                if not filtered:
                    return False
                if len(filtered) != len(d[3]):
                    d[3] = filtered
                    d[1], d[2] = filtered[0], filtered[-1]
                    moved = True
        if not moved:
            return True


def _feature_argmin(dom: list, x: int, up: Fraction, down: Fraction, fee: Fraction) -> int:
    """A cheapest value in the (already bounded) feature domain."""
    if dom[0] == "i":
        lo, hi = dom[1], dom[2]
        best_v, best_c = lo, _scalar_cost(lo, x, up, down, fee)
        for v in (hi, x if lo <= x <= hi else None):
            if v is None:
                continue
            c = _scalar_cost(v, x, up, down, fee)
            if c < best_c:
                best_v, best_c = v, c
        return best_v
    return min(dom[3], key=lambda v: _scalar_cost(v, x, up, down, fee))


def _greedy_feasible(doms, constraints, current, ups, downs, fees):
    """Construct a complete feasible integer assignment greedily.

    Starting from each feature's cheapest value, repeatedly pick a violated
    constraint and move the variable whose unit adjustment relieves it at the
    lowest marginal cost (staying inside its domain).  This either returns a
    feasible point -- used as an initial incumbent -- or ``None``; a ``None``
    proves nothing and the exact search still decides feasibility.
    """
    work = [list(d) if d[0] == "i" else list(d) for d in doms]
    for d in work:
        if d[0] == "s":
            d[3] = list(d[3])
    z = [_feature_argmin(d, current[j], ups[j], downs[j], fees[j])
         for j, d in enumerate(work)]

    def violations():
        out = []
        for r, (coeffs, bnd) in enumerate(constraints):
            slack = bnd - sum(a * v for a, v in zip(coeffs, z))
            if slack < 0:
                out.append((r, coeffs, bnd, slack))
        return out

    for _ in range(10000):
        viol = violations()
        if not viol:
            return list(z)
        _, coeffs, bnd, _ = viol[0]
        best_move = None  # (marginal cost, j, new value)
        for j, a in enumerate(coeffs):
            if a == 0:
                continue
            d = work[j]
            options = range(d[1], d[2] + 1) if d[0] == "i" else d[3]
            # To reduce sum a_k z_k, decrease z_j when a_j>0, increase when a_j<0.
            for v in options:
                if a > 0 and v >= z[j]:
                    continue
                if a < 0 and v <= z[j]:
                    continue
                relief = a * (z[j] - v)  # positive reduction of the lhs
                if relief <= 0:
                    continue
                delta = (_scalar_cost(v, current[j], ups[j], downs[j], fees[j])
                         - _scalar_cost(z[j], current[j], ups[j], downs[j], fees[j]))
                marginal = delta / relief
                if best_move is None or marginal < best_move[0]:
                    best_move = (marginal, j, v)
        if best_move is None:
            return None
        _, j, v = best_move
        z[j] = v
    return None


def _branch(doms, current, ups, downs, fees):
    """Split one non-singleton domain around a cheapest value.

    Choosing the variable with the smallest domain and branching at its
    argmin-cost value produces a strong incumbent quickly; both children still
    partition the domain exhaustively, so completeness is preserved.
    """
    best_j, best_size = -1, None
    for j, dom in enumerate(doms):
        size = _domain_size(dom)
        if size <= 1:
            continue
        if best_size is None or size < best_size or (
            size == best_size and dom[0] == "s" and doms[best_j][0] == "i"
        ):
            best_j, best_size = j, size
    if best_j < 0:
        return []  # every domain is a singleton: nothing to split
    dom = doms[best_j]
    pivot = _feature_argmin(dom, current[best_j], ups[best_j], downs[best_j],
                            fees[best_j])
    if dom[0] == "i":
        lo, hi = dom[1], dom[2]
        if pivot <= lo:
            splits = (["i", lo, lo, None], ["i", lo + 1, hi, None])
        elif pivot >= hi:
            splits = (["i", lo, hi - 1, None], ["i", hi, hi, None])
        else:
            splits = (["i", lo, pivot, None], ["i", pivot + 1, hi, None])
    else:
        vals = dom[3]
        k = vals.index(pivot)
        left_vals, right_vals = vals[: k + 1], vals[k + 1 :]
        if not right_vals:  # pivot is the last value
            left_vals, right_vals = vals[:k], vals[k:]
        splits = (
            ["s", left_vals[0], left_vals[-1], list(left_vals)],
            ["s", right_vals[0], right_vals[-1], list(right_vals)],
        )
    children = []
    for split in splits:
        # Full deep copy: interval bounds and set value lists must be owned.
        child = [
            ["i", d[1], d[2], None] if d[0] == "i"
            else ["s", d[1], d[2], list(d[3])]
            for d in doms
        ]
        child[best_j] = split
        children.append(child)
    return children


def solve_leaf_exact(domains, constraints, current, ups, downs, fees):
    """Exact branch-and-bound for one leaf's path + resource constraints.

    A greedy feasible assignment seeds the incumbent so most of the tree is
    pruned by lower bounds; branch-and-bound then proves global optimality (or
    infeasibility).  Returns (status, z|None, exact_cost|None, nodes_visited).
    """
    doms = _initial_domains(domains)
    if not _propagate(doms, constraints):
        return "infeasible", None, None, 1

    best_z: Optional[list[int]] = None
    best_cost: Optional[Fraction] = None
    seed = _greedy_feasible(doms, constraints, current, ups, downs, fees)
    if seed is not None and all(v in _values(doms[j]) for j, v in enumerate(seed)):
        if all(sum(a * w for a, w in zip(coef, seed)) <= b
               for coef, b in constraints):
            best_z = seed
            best_cost = sum(
                _scalar_cost(seed[j], current[j], ups[j], downs[j], fees[j])
                for j in range(len(seed))
            )

    counter = itertools.count()
    heap = [(_lower_bound(doms, current, ups, downs, fees), next(counter), doms)]
    nodes = 0

    while heap:
        lb, _, node_doms = heapq.heappop(heap)
        nodes += 1
        if nodes > NODE_BUDGET:
            return "unknown", best_z, best_cost, nodes
        if best_cost is not None and lb > best_cost:
            continue
        if all(_domain_size(d) == 1 for d in node_doms):
            z = [d[1] for d in node_doms]
            cost = sum(
                _scalar_cost(z[j], current[j], ups[j], downs[j], fees[j])
                for j in range(len(z))
            )
            if best_cost is None or cost < best_cost:
                best_cost, best_z = cost, z
            continue
        for child in _branch(node_doms, current, ups, downs, fees):
            if _propagate(child, constraints):
                child_lb = _lower_bound(child, current, ups, downs, fees)
                # Equal lower bounds must stay open: a partial domain whose
                # separable lower bound equals the incumbent can still contain
                # an equally optimal full integer assignment.
                if best_cost is None or child_lb <= best_cost:
                    heapq.heappush(heap, (child_lb, next(counter), child))

    if best_z is None:
        return "infeasible", None, None, nodes
    return "optimal", best_z, best_cost, nodes


def _values(dom: list) -> list[int]:
    if dom[0] == "i":
        return list(range(dom[1], dom[2] + 1))
    return list(dom[3])


# --------------------------------------------------------------------------- #
# Independent cross-check 1: full Cartesian enumeration (small domains only).
# --------------------------------------------------------------------------- #
def solve_leaf_enumeration(domains, constraints, current, ups, downs, fees):
    best_z, best_cost = None, None
    for z in itertools.product(*domains):
        if all(sum(a * v for a, v in zip(row, z)) <= b for row, b in constraints):
            cost = sum(
                _scalar_cost(z[j], current[j], ups[j], downs[j], fees[j])
                for j in range(len(z))
            )
            if best_cost is None or cost < best_cost or (
                cost == best_cost and list(z) < best_z
            ):
                best_cost, best_z = cost, list(z)
    if best_z is None:
        return "infeasible", None, None
    return "optimal", best_z, best_cost


# --------------------------------------------------------------------------- #
# Independent cross-check 2: SciPy HiGHS MILP with linearised change indicators.
# --------------------------------------------------------------------------- #
def solve_leaf_milp(domains, constraints, current, ups, downs, fees):
    """HiGHS MILP, independent cross-check of the branch-and-bound solver.

    Variables per feature: ``z_j`` integer, ``d_j, e_j >= 0`` the up/down
    amounts, ``q_j`` binary change indicator.  A set domain gets one binary
    selector per allowed value with exactly one selected, so discrete domains
    are modelled exactly (bounds alone would admit the gaps).
    """
    from scipy.optimize import milp, LinearConstraint, Bounds
    import numpy as np

    m = len(current)
    base = 4 * m
    selector: list[list[int]] = []  # selector[j] = variable indices of its binaries
    extra = 0
    for values in domains:
        if len(values) == values[-1] - values[0] + 1:
            selector.append([])
        else:
            selector.append(list(range(base + extra, base + extra + len(values))))
            extra += len(values)
    n_vars = base + extra

    c = np.zeros(n_vars)
    lb = np.full(n_vars, -np.inf)
    ub = np.full(n_vars, np.inf)
    integrality = np.zeros(n_vars)
    for j in range(m):
        lb[j], ub[j] = domains[j][0], domains[j][-1]
        integrality[j] = 1
        d, e, q = m + j, 2 * m + j, 3 * m + j
        lb[d] = lb[e] = 0.0
        lb[q], ub[q] = 0.0, 1.0
        integrality[q] = 1.0
        c[d], c[e], c[q] = float(ups[j]), float(downs[j]), float(fees[j])
        for k in selector[j]:
            lb[k], ub[k] = 0.0, 1.0
            integrality[k] = 1.0

    rows, lows, highs = [], [], []

    def add(row_coeffs: dict[int, float], low, high) -> None:
        row = np.zeros(n_vars)
        for idx, val in row_coeffs.items():
            row[idx] = val
        rows.append(row)
        lows.append(float(low))
        highs.append(float(high))

    for j in range(m):  # d_j - e_j - z_j = -x_j
        add({j: 1.0, m + j: -1.0, 2 * m + j: 1.0}, current[j], current[j])
    big_m = 20000.0  # >= max |z_j - x_j| over [-10000,10000] domains
    for j in range(m):  # |z_j - x_j| <= M q_j
        add({j: 1.0, 3 * m + j: -big_m}, -np.inf, current[j])
        add({j: -1.0, 3 * m + j: -big_m}, -np.inf, -current[j])
    for j, sel in enumerate(selector):  # discrete set: exactly one value picked
        if sel:
            add({k: 1.0 for k in sel}, 1.0, 1.0)
            add({j: 1.0, **{k: -float(v) for k, v in zip(sel, domains[j])}}, 0.0, 0.0)
    for coeffs, bnd in constraints:  # resource / path inequalities on z
        add({j: float(a) for j, a in enumerate(coeffs) if a != 0}, -np.inf, float(bnd))

    res = milp(
        c=c,
        constraints=LinearConstraint(np.array(rows), np.array(lows), np.array(highs)),
        integrality=integrality,
        bounds=Bounds(lb, ub),
        options={"time_limit": 60},
    )
    # scipy.optimize.milp status: 0 optimum, 1 limit reached (unknown),
    # 2 infeasible, 3 unbounded.
    if res.status == 2:
        return "infeasible", None, None
    if res.status != 0 or res.x is None:
        return "unknown", None, None
    raw_z = res.x[:m]
    # Defensive guard: an MILP optimum must be integral; never round a
    # fractional relaxation into a possibly infeasible point.
    if any(abs(v - round(v)) > 1e-6 for v in raw_z):
        return "unknown", None, None
    z = [int(round(v)) for v in raw_z]
    if any(z[j] not in domains[j] for j in range(m)):
        return "unknown", None, None
    cost = sum(
        _scalar_cost(z[j], current[j], ups[j], downs[j], fees[j]) for j in range(m)
    )
    return "optimal", z, cost


# --------------------------------------------------------------------------- #
# Public entry point.
# --------------------------------------------------------------------------- #
def _path_constraint(feature: int, sense: str, bound: int, n_features: int) -> tuple[list[int], int]:
    """Convert a path condition to a single linear inequality ``a z <= b``.

    ``z[f] <= t`` becomes ``+z[f] <= t``; ``z[f] >= t`` becomes ``-z[f] <= -t``.
    """
    coef = [0] * n_features
    if sense == "<=":
        coef[feature] = 1
        return coef, bound
    coef[feature] = -1
    return coef, -bound


def _verify_solution(model, req, z, target, exact_cost, ups, downs, fees) -> None:
    """Re-check every legality condition and the exact cost before returning."""
    current = req["current"]
    for j, v in enumerate(z):
        if not isinstance(v, int) or v not in req["domains"][j]:
            raise ActionError(f"internal error: z[{j}]={v} outside its domain")
    for row, b in zip(req["A"], req["b"]):
        if sum(a * v for a, v in zip(row, z)) > b:
            raise ActionError("internal error: action violates a resource constraint")
    if predict(model, z) != target:
        raise ActionError("internal error: action does not reach the target class")
    recomputed = sum(
        _scalar_cost(z[j], current[j], ups[j], downs[j], fees[j])
        for j in range(len(z))
    )
    if recomputed != exact_cost:
        raise ActionError("internal error: cost accounting mismatch")


# A leaf problem is cross-checked with the exact branch-and-bound (and, when
# small enough, full enumeration) whenever its search space is at most this big.
CROSSCHECK_LIMIT = 200_000


def _domain_cardinality(domains: list[list[int]]) -> int:
    total = 1
    for values in domains:
        total *= len(values)
        if total > CROSSCHECK_LIMIT:
            return total
    return total


def counterfactual(tree_dict, request: dict, max_depth: int = 6) -> dict:
    """Solve the minimum-cost actionable counterfactual for ``tree_dict``.

    Each target-class leaf is one mixed-integer program (the conjunction of its
    path constraints, the resource limits and the domains), solved to proven
    optimality by HiGHS.  On small search spaces the hand-written exact
    branch-and-bound -- and full Cartesian enumeration when tiny -- independently
    re-solves the same leaf and must agree.  The returned integer vector is then
    re-checked against every domain, resource limit, target prediction and the
    exact Fraction cost before it is handed back.
    """
    current_raw = request.get("current") if isinstance(request, dict) else None
    if not isinstance(current_raw, list) or not current_raw:
        raise ActionError("request.current must be a non-empty list")
    n_features = len(current_raw)
    try:
        model: Node = parse_tree(tree_dict, n_features, max_depth)
    except ValueError as exc:
        raise ActionError(str(exc)) from exc
    try:
        req = validate_action_request(request, n_features)
    except InputError as exc:
        raise ActionError(str(exc)) from exc

    current = req["current"]
    target = req["target"]
    ups = [_frac(v) for v in req["up_cost"]]
    downs = [_frac(v) for v in req["down_cost"]]
    fees = [_frac(v) for v in req["change_fee"]]
    resource = [(list(row), b) for row, b in zip(req["A"], req["b"])]
    cardinality = _domain_cardinality(req["domains"])

    before = predict(model, current)
    overall_status = "infeasible"
    best = None  # (Fraction cost, z, leaf_index)
    leaf_reports = []
    for leaf_idx, (_leaf, path) in enumerate(target_leaves(model, target)):
        constraints = resource + [
            _path_constraint(f, sense, bound, n_features)
            for (f, sense, bound) in path
        ]
        # Primary solver: mature MILP engine.
        status, z, cost = solve_leaf_milp(
            req["domains"], constraints, current, ups, downs, fees
        )
        report = {"leaf": leaf_idx, "milp": status}

        # Independent exact cross-checks on tractable search spaces.
        if cardinality <= CROSSCHECK_LIMIT:
            bb_status, bb_z, bb_cost, bb_nodes = solve_leaf_exact(
                req["domains"], constraints, current, ups, downs, fees
            )
            report["branch_bound"] = bb_status
            report["bb_nodes"] = bb_nodes
            # An unfinished B&B (node budget) cannot contradict the MILP; only
            # a completed B&B optimum/infeasibility is an exact cross-check.
            if bb_status == "optimal":
                if status != "optimal" or float(bb_cost) != float(cost):
                    raise ActionError("internal error: B&B/MILP disagreement")
            elif bb_status == "infeasible" and status != "infeasible":
                raise ActionError("internal error: B&B proved infeasible but MILP did not")
            if cardinality <= 50_000:
                en_status, en_z, en_cost = solve_leaf_enumeration(
                    req["domains"], constraints, current, ups, downs, fees
                )
                report["enumeration"] = en_status
                if en_status != status:
                    raise ActionError("internal error: enumeration/MILP status disagreement")
                if en_status == "optimal" and float(en_cost) != float(cost):
                    raise ActionError("internal error: enumeration/MILP cost disagreement")

        leaf_reports.append(report)
        if status == "unknown":
            overall_status = "unknown"
        elif status == "optimal" and (
            best is None or cost < best[0] or (cost == best[0] and z < best[1])
        ):
            best = (cost, z, leaf_idx)

    if best is None:
        return {
            "status": overall_status,
            "infeasible": overall_status == "infeasible",
            "action": None,
            "target": target,
            "before_prediction": before,
            "after_prediction": None,
            "total_cost": None,
            "per_feature_cost": None,
            "target_leaf": None,
            "leaf_reports": leaf_reports,
        }

    cost, z, leaf_idx = best
    after = predict(model, z)
    _verify_solution(model, req, z, target, cost, ups, downs, fees)
    per_feature = [
        float(_scalar_cost(z[j], current[j], ups[j], downs[j], fees[j]))
        for j in range(n_features)
    ]
    return {
        "status": "optimal",
        "infeasible": False,
        "action": z,
        "target": target,
        "before_prediction": before,
        "after_prediction": after,
        "total_cost": float(cost),
        "per_feature_cost": per_feature,
        "target_leaf": leaf_idx,
        "leaf_reports": leaf_reports,
    }
