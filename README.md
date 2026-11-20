# Sinkhorn Eliminant

**`sinkhorn-eliminant`** is a self-contained Python solver for exact Sinkhorn / Kruithof scaling of matrices and tensors, accompanying:

> Shankar Sivarajan, *An explicit determinantal formula for Sinkhorn limits of matrices and tensors*, [hexagon:2611.00433](https://hexagonmath.org/2611.00433) (2026).

## Overview

Given a table (matrix) of positive numbers, [**Sinkhorn scaling**](https://en.wikipedia.org/wiki/Sinkhorn%27s_theorem)—also known as Kruithof's method, RAS, or [iterative proportional fitting](https://en.wikipedia.org/wiki/Iterative_proportional_fitting)—multiplies each row and column by positive weights until all row sums and column sums match prescribed targets (for instance, turning a matrix doubly stochastic, or updating a traffic/contingency table). The same problem extends naturally to higher-dimensional arrays (tensors), where each mode's 1-slice sums are scaled to match target margins.

Normally, this limit is approximated numerically by alternating row and column normalizations until convergence. Yet the scaling conditions are a system of polynomial equations: every entry of the exact limit $P^*$ is an algebraic number satisfying a single univariate polynomial $E(x)$ with exact integer coefficients.

Instead of iterating—or running black-box elimination—this package computes $E(x)$ directly from an **explicit matrix determinant** derived in the paper, finds its roots to high precision, and recovers both the unique positive Sinkhorn limit $P^*$ and all other complex solutions to the scaling equations from a single left null vector.

The core library uses **only the Python standard library** (no third-party packages required), computing exact polynomials via multimodular CRT with Schur-complement reduction and Newton interpolation and polishing roots in arbitrary-precision `Decimal` arithmetic. Optional `numpy`, `scipy`, `mpmath`, and `python-flint` backends are supported for larger generalized eigenvalue problems.

## Quick Start

Requirements: Python 3.10+ (standard library only). Optionally install into your environment with:

```bash
pip install -e .
```

### Python API

```python
from sinkhorn_eliminant import eliminant_polynomial, solve_scaling, matrix_diagonal_polynomial

# 3x3 matrix (Appendix A.1 / Rowland-Wu Example 1)
T = [
    [3, 9, 1],
    [3, 2, 9],
    [5, 3, 4],
]
margins = [[1, 1, 1], [1, 1, 1]]

# Exact polynomial computation
elim = eliminant_polynomial(T, margins)
print("Primitive E(x) =", elim.format("primitive"))
print("Theorem 4.6 E_tilde(x) =", elim.format("E_tilde"))

# Full solve: roots, left null-vector recovery, positivity selection, and Sinkhorn check
res = solve_scaling(T, margins, recover_all=True)
print("Limit P[0, 0] =", res.limit_root)
print("Recovered limit P* =", res.limit_P)
print("Scaling factors u^(1), u^(2) =", res.limit_scalings)

# For 2D matrices, Section 5's N x N diagonal pencil F(x) = det(diag(Gamma_hat) + x diag(Delta_hat) Q)
diag = matrix_diagonal_polynomial(T, margins)
assert diag.coeffs == elim.coeffs
```

Higher-order tensors work with the exact same calls:

```python
# 2x2x3 tensor with general margins (Appendix A.2, degree N = 12)
T_tensor = [
    [[4, 3, 8], [1, 6, 4]],
    [[7, 9, 6], [9, 5, 4]],
]
margins_tensor = [[9, 15], [7, 17], [7, 7, 10]]

res = solve_scaling(T_tensor, margins_tensor)
print("Degree:", res.eliminant.degree, "Limit P[0,0,0]:", res.limit_root)
```

To target an entry other than the top-left corner `(0, ..., 0)`, pass `entry=(i_1, ..., i_r)` (0-indexed):

```python
res_102 = solve_scaling(T_tensor, margins_tensor, entry=(1, 0, 2))
```

Rational entries and margins are supported via `fractions.Fraction`, and standard Rowland-Wu / uniform Sinkhorn margins can be generated with `sinkhorn_margins(dims)`.

### Command-Line Interface

```bash
# 3x3 matrix with uniform Sinkhorn margins
python -m sinkhorn_eliminant --tensor '[[3,9,1],[3,2,9],[5,3,4]]' --sinkhorn --matrix-diagonal --all-roots

# 2x2x3 tensor with general margins
python -m sinkhorn_eliminant \
  --tensor '[[[4,3,8],[1,6,4]],[[7,9,6],[9,5,4]]]' \
  --margins '[[9,15],[7,17],[7,7,10]]' \
  --all-roots

# Polynomial only, machine-readable JSON
python -m sinkhorn_eliminant --tensor '[[1,2],[3,4]]' --sinkhorn --poly-only --json
```

## Paper-to-Code Guide

| Paper section / result                                                                            | Module                                                    | Key functions                                                                                            |
| ------------------------------------------------------------------------------------------------- | --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| §2–§3, Lemma 3.1: Segre Sylvester matrix $\Phi(x) = [M_0 \mid K - x J]$                           | [`sylvester.py`](sinkhorn_eliminant/sylvester.py)         | `sylvester_blocks`, `sylvester_matrix`, `sz_multidegree`, `monomials`, `multinomial_degree`              |
| §3, Thm 3.2: $\det \Phi(x) = \pm \mathrm{Res}_\Sigma(\ell_1, \dots, \ell_d, \ell_x)$                                           | [`resultant.py`](sinkhorn_eliminant/resultant.py)         | `eliminant_polynomial`, `restricted_slab_resultant`                                                      |
| §4, Prop 4.4–Thm 4.6: Face discriminants, content $c(T)$, $\tilde E_T(x)$                                                 | [`discriminants.py`](sinkhorn_eliminant/discriminants.py) | `face_discriminant`, `cayley_hyperdeterminant`, `theorem46_factors`                                      |
| §5, Thm 5.3–Prop 5.4: Matrix diagonal form $F(x) = \det(\mathrm{diag}\,\hat{\Gamma} + x\,\mathrm{diag}\,\hat{\Delta}\,Q)$ | [`matrix_pencil.py`](sinkhorn_eliminant/matrix_pencil.py) | `build_matrix_pencil`, `matrix_diagonal_polynomial`                                                      |
| §3 & §6, Cor 3.3, Thm 3.4, Prop 6.1: Roots, null-vector recovery, positivity selection            | [`solver.py`](sinkhorn_eliminant/solver.py)               | `solve_scaling`, `recover_solution`, `find_polynomial_roots`, `solve_pencil_numerical`, `sinkhorn_scale` |

Note on degenerate cases (Remark 4.7): if a face of $T$ avoiding the corner has vanishing discriminant (e.g., a $3\times 3$ matrix with a vanishing $2\times 2$ minor in rows $\{2,3\}$ and columns $\{2,3\}$), then $c(T) = 0$ and $\det \Phi(x) \equiv 0$. For matrices, `matrix_diagonal_polynomial` avoids $c(T)$ and still returns the true eliminant.

## Reproducing Paper Results

```bash
# Reproduce Appendix A (Examples A.1-A.4), Rowland-Wu Example 13, Kruithof (1937),
# and the paper's theorem/remark identities
python reproduce.py

# Or print only the worked examples (add --full-polys for full degree > 12 coefficients)
python reproduce.py --examples-only
python reproduce.py --examples-only --full-polys

# Also runnable via stdlib unittest
python -m unittest -v reproduce.py
```

## Related

The case of two-way margins with a binary variable ($2\times J\times K$ tables under the no-three-way-interaction model) is treated in the companion repository [`ipf-binary-eliminant`](https://github.com/shankarsivarajan/ipf-binary-eliminant) and the paper

> Shankar Sivarajan, *An explicit determinantal formula for limits of iterative proportional fitting with a binary variable*, [hexagon:2611.00705](https://hexagonmath.org/2611.00705) (2026).
