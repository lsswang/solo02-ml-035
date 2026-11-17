"""Input validation for training data and action requests.

All malformed shapes, references, thresholds, costs, domains or non-finite
numbers raise :class:`InputError` (a ``ValueError``) with a clear message.
"""
from __future__ import annotations

import math
from typing import Any

from .tree import (
    MAX_EXTERNAL_DEPTH,
    MAX_FEATURES,
    MAX_TRAIN_DEPTH,
    MIN_FEATURES,
    THRESH_MAX,
    THRESH_MIN,
)

VALUE_MIN = -20
VALUE_MAX = 20
COST_MIN = 0.0
COST_MAX = 20.0
ALPHA_MIN = 0.01
ALPHA_MAX = 2.0
DOMAIN_LIMIT = 10000
MAX_DOMAIN_VALUES = 8
MAX_RESOURCE_ROWS = 8
COEFF_MIN = -100
COEFF_MAX = 100


class InputError(ValueError):
    """Raised whenever a training or action request fails validation."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{name} must be a finite number, got {value!r}")
    fv = float(value)
    if not math.isfinite(fv):
        raise InputError(f"{name} must be finite, got {value!r}")
    return fv


def validate_training(
    rows: Any, costs: Any, thresholds: Any, max_depth: Any, alpha: Any
) -> tuple[list[list[int | None]], list[list[float]], list[list[int]], int, float, int]:
    if not isinstance(rows, list) or not (1 <= len(rows) <= 32):
        raise InputError("rows must be a list of 1..32 entries")
    n = len(rows)

    n_features: int | None = None
    clean_rows: list[list[int | None]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, list):
            raise InputError(f"row {i} must be a list")
        if n_features is None:
            n_features = len(row)
            if not (MIN_FEATURES <= n_features <= MAX_FEATURES):
                raise InputError(
                    f"feature count must be in [{MIN_FEATURES},{MAX_FEATURES}]"
                )
        if len(row) != n_features:
            raise InputError(f"row {i} has {len(row)} features, expected {n_features}")
        clean: list[int | None] = []
        for f, value in enumerate(row):
            if value is None:
                clean.append(None)
            elif _is_int(value) and VALUE_MIN <= value <= VALUE_MAX:
                clean.append(value)
            else:
                raise InputError(
                    f"rows[{i}][{f}] must be None or an integer in "
                    f"[{VALUE_MIN},{VALUE_MAX}], got {value!r}"
                )
        clean_rows.append(clean)
    assert n_features is not None

    if not isinstance(costs, list) or len(costs) != n:
        raise InputError(f"costs must have {n} rows")
    clean_costs: list[list[float]] = []
    for i, pair in enumerate(costs):
        if not isinstance(pair, list) or len(pair) != 2:
            raise InputError(f"costs[{i}] must be a [cost0, cost1] pair")
        vals: list[float] = []
        for y in range(2):
            c = _finite_number(pair[y], f"costs[{i}][{y}]")
            if not (COST_MIN <= c <= COST_MAX):
                raise InputError(
                    f"costs[{i}][{y}]={c} outside [{COST_MIN},{COST_MAX}]"
                )
            vals.append(c)
        vals.append(0.0)  # placeholder to keep indexing explicit
        clean_costs.append(vals[:2])

    if not isinstance(thresholds, list) or len(thresholds) != n_features:
        raise InputError(f"thresholds must list {n_features} feature candidate lists")
    clean_thresholds: list[list[int]] = []
    for f, cand in enumerate(thresholds):
        if not isinstance(cand, list) or not (0 <= len(cand) <= 4):
            raise InputError(
                f"thresholds[{f}] must be a list of 0..4 integer candidates"
            )
        seen: set[int] = set()
        clean_t: list[int] = []
        for t in cand:
            if not _is_int(t) or not (THRESH_MIN <= t <= THRESH_MAX):
                raise InputError(
                    f"thresholds[{f}] entry {t!r} must be an integer in "
                    f"[{THRESH_MIN},{THRESH_MAX}]"
                )
            if t in seen:
                raise InputError(f"thresholds[{f}] contains duplicate threshold {t}")
            seen.add(t)
            clean_t.append(t)
        clean_thresholds.append(clean_t)

    if not _is_int(max_depth) or not (0 <= max_depth <= MAX_TRAIN_DEPTH):
        raise InputError(
            f"max_depth must be an integer in [0,{MAX_TRAIN_DEPTH}], got {max_depth!r}"
        )
    alpha_f = _finite_number(alpha, "alpha")
    if not (ALPHA_MIN <= alpha_f <= ALPHA_MAX):
        raise InputError(f"alpha must be in [{ALPHA_MIN},{ALPHA_MAX}], got {alpha_f}")

    return clean_rows, clean_costs, clean_thresholds, int(max_depth), alpha_f, n_features


def validate_tree_depth(max_depth: Any) -> int:
    if not _is_int(max_depth) or not (0 <= max_depth <= MAX_EXTERNAL_DEPTH):
        raise InputError(
            f"max_depth must be an integer in [0,{MAX_EXTERNAL_DEPTH}]"
        )
    return int(max_depth)


def validate_action_request(
    request: Any, n_features: int
) -> dict:
    """Normalize and validate an action request into a plain dict of clean arrays."""
    if not isinstance(request, dict):
        raise InputError("action request must be a dict")

    def req_array(key: str, length: int) -> list:
        arr = request.get(key)
        if not isinstance(arr, list) or len(arr) != length:
            raise InputError(f"{key} must be a list of length {length}")
        return arr

    current = req_array("current", n_features)
    clean_current: list[int] = []
    for j, x in enumerate(current):
        if not _is_int(x) or not (VALUE_MIN <= x <= VALUE_MAX):
            raise InputError(
                f"current[{j}] must be an integer in [{VALUE_MIN},{VALUE_MAX}]"
            )
        clean_current.append(x)

    target = request.get("target")
    if not _is_int(target) or target not in (0, 1):
        raise InputError("target must be 0 or 1")

    domains_raw = req_array("domains", n_features)
    domains: list[list[int]] = []
    for j, dom in enumerate(domains_raw):
        if not isinstance(dom, dict):
            raise InputError(f"domains[{j}] must be an object")
        if "interval" in dom and "values" in dom:
            raise InputError(f"domains[{j}] cannot specify both interval and values")
        if "interval" in dom:
            iv = dom["interval"]
            if (
                not isinstance(iv, list)
                or len(iv) != 2
                or not _is_int(iv[0])
                or not _is_int(iv[1])
            ):
                raise InputError(f"domains[{j}].interval must be [lo, hi] integers")
            lo, hi = iv
            if not (-DOMAIN_LIMIT <= lo <= hi <= DOMAIN_LIMIT):
                raise InputError(
                    f"domains[{j}] interval must satisfy -10000 <= lo <= hi <= 10000"
                )
            domains.append(list(range(lo, hi + 1)))
        elif "values" in dom:
            vals = dom["values"]
            if not isinstance(vals, list) or not (1 <= len(vals) <= MAX_DOMAIN_VALUES):
                raise InputError(
                    f"domains[{j}].values must list 1..{MAX_DOMAIN_VALUES} integers"
                )
            seen: set[int] = set()
            clean_vals: list[int] = []
            for v in vals:
                if not _is_int(v) or not (-DOMAIN_LIMIT <= v <= DOMAIN_LIMIT):
                    raise InputError(
                        f"domains[{j}] values must be integers in [-10000,10000]"
                    )
                if v in seen:
                    raise InputError(f"domains[{j}].values has duplicate {v}")
                seen.add(v)
                clean_vals.append(v)
            domains.append(sorted(clean_vals))
        else:
            raise InputError(f"domains[{j}] needs 'interval' or 'values'")

    coef_keys = ("up_cost", "down_cost", "change_fee")
    clean_coefs: dict[str, list[float]] = {}
    for key in coef_keys:
        arr = req_array(key, n_features)
        out: list[float] = []
        for j, c in enumerate(arr):
            fv = _finite_number(c, f"{key}[{j}]")
            if not (COST_MIN <= fv <= COST_MAX):
                raise InputError(f"{key}[{j}]={fv} outside [0,20]")
            out.append(fv)
        clean_coefs[key] = out

    cons = request.get("constraints", {"A": [], "b": []})
    if not isinstance(cons, dict) or "A" not in cons or "b" not in cons:
        raise InputError("constraints must be an object with 'A' and 'b'")
    A, b = cons["A"], cons["b"]
    if not isinstance(A, list) or len(A) > MAX_RESOURCE_ROWS:
        raise InputError(f"A must be a list of at most {MAX_RESOURCE_ROWS} rows")
    if not isinstance(b, list) or len(b) != len(A):
        raise InputError("b must have the same number of entries as A")
    clean_A: list[list[int]] = []
    for r, row in enumerate(A):
        if not isinstance(row, list) or len(row) != n_features:
            raise InputError(f"A[{r}] must have {n_features} integer entries")
        clean_row: list[int] = []
        for j, a in enumerate(row):
            if not _is_int(a) or not (COEFF_MIN <= a <= COEFF_MAX):
                raise InputError(f"A[{r}][{j}] must be an integer in [-100,100]")
            clean_row.append(a)
        clean_A.append(clean_row)
    clean_b: list[int] = []
    for r, bv in enumerate(b):
        if not _is_int(bv) or not (COEFF_MIN <= bv <= COEFF_MAX):
            raise InputError(f"b[{r}] must be an integer in [-100,100]")
        clean_b.append(bv)

    return {
        "current": clean_current,
        "target": target,
        "domains": domains,
        "up_cost": clean_coefs["up_cost"],
        "down_cost": clean_coefs["down_cost"],
        "change_fee": clean_coefs["change_fee"],
        "A": clean_A,
        "b": clean_b,
    }
