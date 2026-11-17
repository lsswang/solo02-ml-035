# Small classifier and action explanation

Python 3.11. NumPy/SciPy are used (SciPy's bundled HiGHS MILP solver). See
[`METHODS.md`](METHODS.md) for the full method, precision and cost discussion.

## Run

```bash
python3 demo.py            # end-to-end demonstration with independent checks
python3 -m pytest tests/   # 53 tests
```

## What is delivered

* **Exact optimal small-tree learning** (`triage/trainer.py`): subset dynamic
  programming over row bit-masks. Every legal `(feature, threshold,
  missing-direction)` split is evaluated through its optimal subtrees, so the
  result globally minimises `sum of prediction costs + alpha * internal nodes`
  — unlike the old greedy CART in `starter/`, which is kept only for reference.
  An independent full-tree brute-force enumerator verifies the optimum on small
  instances.
* **Actionable counterfactuals** (`triage/action.py`): given a learned or
  external binary tree (depth ≤ 6), find the minimum-cost integer action whose
  result lies in every actionable domain, satisfies every joint resource limit
  and is classified as the target class. A HiGHS MILP is the primary solver;
  an exact hand-written branch-and-bound and full Cartesian enumeration
  cross-check it on tractable instances, and every returned integer vector is
  re-verified for legality and exact (`Fraction`) cost. Unfinished searches are
  reported as `unknown`, never as `infeasible`.
* Strict input validation for shapes, tree references, thresholds, costs,
  domains and non-finite numbers.

The legacy `starter/` package (greedy tree + plain box closest-leaf tool) is
retained unchanged for comparison; the new contract is implemented in `triage/`.
