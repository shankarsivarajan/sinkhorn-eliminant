"""Segre-variety Sylvester matrix construction for the Sinkhorn/Kruithof eliminant.

Implements Sections 2 and 3 of:
    S. Sivarajan, "An explicit determinantal formula for Sinkhorn limits of
    matrices and tensors" (HexagonMath 2611.00433).

Given a positive tensor T of format n = (n_1, ..., n_r) (all n_j >= 2) and
target one-dimensional margins mu^(1), ..., mu^(r) with a common total tau != 0,
the margin forms and the corner form

    l_{k,i}(y) = sum_c (delta_{c_k, i} - mu^(k)_i / tau) t_c y_c   (2 <= i <= n_k),
    l_x(y)     = t_1 y_1 - (x / tau) sum_c t_c y_c,

are multilinear forms of multidegree (1, ..., 1) on Sigma = P^{n_1-1} x ... x P^{n_r-1}.
The Sylvester map

    Phi(x) : S(m - 1)^{oplus (d+1)} -> S(m),    m_j = 1 + sum_{i < j} (n_i - 1),

is square of size (d+1)N, where d = sum_j (n_j - 1) and N = multinomial(d; n_1-1, ..., n_r-1).
The parameter x enters only the last N columns, linearly: Phi(x) = [M_0 | K - x J].
"""

from __future__ import annotations

import itertools
from fractions import Fraction
from math import comb, gcd, lcm, prod
from typing import Any, Sequence

Number = int | Fraction
TensorInput = Any  # nested list/tuple, numpy.ndarray, or 1-based cell dict


def _is_exact_scalar(v: Any) -> bool:
    return isinstance(v, (int, Fraction)) and not isinstance(v, bool)


def _to_fraction(v: Any) -> Fraction:
    if isinstance(v, Fraction):
        return v
    if isinstance(v, int) and not isinstance(v, bool):
        return Fraction(v)
    if isinstance(v, str):
        return Fraction(v)
    # Handle numpy integers/floats or Python floats cleanly
    if hasattr(v, "numerator") and hasattr(v, "denominator") and not isinstance(v, float):
        try:
            return Fraction(int(v.numerator), int(v.denominator))
        except Exception:
            pass
    if isinstance(v, float) or type(v).__name__.startswith("float"):
        return Fraction(str(float(v)))
    if type(v).__name__.startswith("int"):
        return Fraction(int(v))
    raise TypeError(f"Unsupported scalar type for exact conversion: {type(v)!r}")


def _infer_nested_dims(seq: Any) -> tuple[int, ...]:
    if hasattr(seq, "shape") and hasattr(seq, "tolist"):
        return tuple(int(d) for d in seq.shape)
    if not isinstance(seq, (list, tuple)):
        return ()
    if len(seq) == 0:
        raise ValueError("Tensor cannot have an empty axis.")
    sub = _infer_nested_dims(seq[0])
    for item in seq[1:]:
        if _infer_nested_dims(item) != sub:
            raise ValueError("Tensor must be rectangular (ragged nested sequences are not allowed).")
    return (len(seq),) + sub


def _nested_get(seq: Any, zero_idx: tuple[int, ...]) -> Any:
    cur = seq
    for i in zero_idx:
        cur = cur[i]
    return cur


def normalize_tensor(
    T: TensorInput,
    dims: tuple[int, ...] | None = None,
) -> tuple[tuple[int, ...], dict[tuple[int, ...], Any]]:
    """Normalize an array-like tensor or 1-based cell dict into (dims, T_dict).

    T_dict maps 1-based cell tuples c = (c_1, ..., c_r) in [1..n_1] x ... x [1..n_r]
    to their entries.
    """
    if isinstance(T, dict):
        if not T:
            raise ValueError("Tensor dictionary cannot be empty.")
        keys = list(T.keys())
        r = len(keys[0])
        if any(len(k) != r for k in keys):
            raise ValueError("All cell keys in tensor dictionary must have the same rank.")
        inferred = tuple(max(k[j] for k in keys) for j in range(r))
        if dims is None:
            dims = inferred
        else:
            dims = tuple(int(d) for d in dims)
            if len(dims) != r:
                raise ValueError(f"Expected rank {len(dims)} keys, got {r}.")
        cells = list(itertools.product(*[range(1, d + 1) for d in dims]))
        if len(T) != len(cells):
            missing = set(cells) - set(T.keys())
            if missing:
                raise ValueError(f"Tensor dictionary is missing cells, e.g. {next(iter(missing))}.")
        out = {c: T[c] for c in cells}
    else:
        inferred = _infer_nested_dims(T)
        if dims is not None and tuple(int(d) for d in dims) != inferred:
            raise ValueError(f"Provided dims {dims} do not match tensor shape {inferred}.")
        dims = inferred
        if hasattr(T, "tolist"):
            T = T.tolist()
        out = {}
        for idx in itertools.product(*[range(d) for d in dims]):
            out[tuple(i + 1 for i in idx)] = _nested_get(T, idx)

    if len(dims) < 2:
        raise ValueError(f"Tensor order r must be >= 2, got shape {dims}.")
    if any(d < 2 for d in dims):
        raise ValueError(f"All mode dimensions must be >= 2, got shape {dims}.")
    return dims, out


def normalize_margins(
    dims: tuple[int, ...],
    margins: Sequence[Sequence[Any]],
    tol: float = 1e-9,
) -> tuple[list[list[Any]], Any]:
    """Validate and normalize target margins mu^(1), ..., mu^(r) and return (mus, total)."""
    if len(margins) != len(dims):
        raise ValueError(f"Expected {len(dims)} margin vectors for shape {dims}, got {len(margins)}.")
    mus: list[list[Any]] = []
    for k, (d, mk) in enumerate(zip(dims, margins)):
        mk_list = list(mk.tolist() if hasattr(mk, "tolist") else mk)
        if len(mk_list) != d:
            raise ValueError(f"Mode {k} margin has length {len(mk_list)}, expected {d}.")
        mus.append(mk_list)

    exact = all(_is_exact_scalar(v) for mk in mus for v in mk)
    if exact:
        totals = [sum(mk) for mk in mus]
        if any(t != totals[0] for t in totals[1:]):
            raise ValueError(f"All margin vectors must have the same total; got {totals}.")
        tot = totals[0]
        if tot == 0:
            raise ValueError("Common margin total cannot be zero.")
    else:
        totals = [sum(float(v) for v in mk) for mk in mus]
        tot = totals[0]
        if abs(tot) == 0:
            raise ValueError("Common margin total cannot be zero.")
        max_diff = max(abs(t - tot) for t in totals[1:])
        if max_diff > tol * max(1.0, abs(tot)):
            raise ValueError(f"All margin vectors must share a common total; got {totals}.")
    return mus, tot


def sinkhorn_margins(
    dims: Sequence[int],
    total: Number | None = None,
) -> list[list[Fraction]]:
    """Return uniform (Rowland-Wu style) Sinkhorn margins for a given shape.

    For an m x n matrix with total=None, returns rho = (1, ..., 1) and
    kappa = (m/n, ..., m/n) with common total m, matching Rowland-Wu (arXiv:2409.02789).
    In general, returns uniform margins with common total `total` (defaulting to dims[0]
    for matrices and prod(dims) for higher-order tensors).
    """
    dims = tuple(int(d) for d in dims)
    if total is None:
        total = dims[0] if len(dims) == 2 else prod(dims)
    tot_f = _to_fraction(total)
    return [[tot_f / d for _ in range(d)] for d in dims]


def permute_to_corner(
    dims: tuple[int, ...],
    T_dict: dict[tuple[int, ...], Any],
    mus: list[list[Any]],
    entry: tuple[int, ...],
    *,
    one_based: bool = False,
) -> tuple[dict[tuple[int, ...], Any], list[list[Any]], list[list[int]]]:
    """Permute indices in each mode so `entry` moves to the corner (1, ..., 1).

    Returns (T_perm, mus_perm, mode_maps) where mode_maps[k][new_idx_0] = old_idx_0.
    """
    if len(entry) != len(dims):
        raise ValueError(f"Entry {entry} must have length {len(dims)} for shape {dims}.")
    target_0 = tuple((e - 1 if one_based else e) for e in entry)
    for k, (idx, d) in enumerate(zip(target_0, dims)):
        if not (0 <= idx < d):
            raise ValueError(f"Entry index {entry[k]} out of bounds for mode {k} of size {d}.")

    mode_maps: list[list[int]] = []
    for k, d in enumerate(dims):
        idx = target_0[k]
        if idx == 0:
            mode_maps.append(list(range(d)))
        else:
            order = [idx] + [i for i in range(d) if i != idx]
            mode_maps.append(order)

    mus_perm = [[mus[k][old_i] for old_i in mode_maps[k]] for k in range(len(dims))]
    T_perm: dict[tuple[int, ...], Any] = {}
    for c_new in itertools.product(*[range(1, d + 1) for d in dims]):
        c_old = tuple(mode_maps[k][c_new[k] - 1] + 1 for k in range(len(dims)))
        T_perm[c_new] = T_dict[c_old]
    return T_perm, mus_perm, mode_maps


def cells_of(dims: Sequence[int]) -> list[tuple[int, ...]]:
    """All 1-based cell tuples in lexicographical order."""
    return list(itertools.product(*[range(1, d + 1) for d in dims]))


def multinomial_degree(dims: Sequence[int]) -> int:
    """Generic degree N = deg(Sigma) = multinomial(d; n_1 - 1, ..., n_r - 1)."""
    n = sum(d - 1 for d in dims)
    out = 1
    for d in dims:
        out *= comb(n, d - 1)
        n -= d - 1
    return out


def sz_multidegree(dims: Sequence[int], perm: Sequence[int] | None = None) -> tuple[int, ...]:
    """Sturmfels-Zelevinsky Sylvester multidegree m^pi.

    For the identity permutation (default), m_j = 1 + sum_{i < j} (n_i - 1).
    In general, m^pi_j = 1 + sum_{pi(i) < pi(j)} (n_i - 1).
    """
    r = len(dims)
    ks = [n - 1 for n in dims]
    if perm is None:
        return tuple(1 + sum(ks[:j]) for j in range(r))
    if sorted(perm) != list(range(r)):
        raise ValueError(f"Invalid mode permutation {perm} for rank {r}.")
    pos = {mode: idx for idx, mode in enumerate(perm)}
    return tuple(1 + sum(ks[i] for i in range(r) if pos[i] < pos[j]) for j in range(r))


def monomials(dims: Sequence[int], m: Sequence[int]) -> list[tuple[tuple[int, ...], ...]]:
    """Basis of S(m): tuples of per-mode exponent vectors summing to m_k."""
    per_mode = []
    for k, d in enumerate(dims):
        mk = m[k]
        per_mode.append(
            [e for e in itertools.product(range(mk + 1), repeat=d) if sum(e) == mk]
        )
    return list(itertools.product(*per_mode))


def clear_denominators(
    T_dict: dict[tuple[int, ...], Any],
    mus: list[list[Any]],
) -> tuple[dict[tuple[int, ...], int], list[list[int]], int, int]:
    """Convert rational/integer T and margins to integers by clearing denominators.

    Returns (T_int, mus_int, D_T, D_mu) such that T_int = D_T * T and mus_int = D_mu * mus.
    """
    T_frac = {c: _to_fraction(v) for c, v in T_dict.items()}
    mus_frac = [[_to_fraction(v) for v in mk] for mk in mus]

    D_T = 1
    for v in T_frac.values():
        D_T = lcm(D_T, v.denominator)
    D_mu = 1
    for mk in mus_frac:
        for v in mk:
            D_mu = lcm(D_mu, v.denominator)

    # Also divide out common integer factor of T_int if helpful? Keep exact D_T so det scaling is tracked.
    T_int = {c: int(v * D_T) for c, v in T_frac.items()}
    mus_int = [[int(v * D_mu) for v in mk] for mk in mus_frac]
    return T_int, mus_int, D_T, D_mu


def _build_block(
    dims: tuple[int, ...],
    src: list[tuple[tuple[int, ...], ...]],
    didx: dict[tuple[tuple[int, ...], ...], int],
    form: dict[tuple[int, ...], Any],
    zero: Any,
) -> list[list[Any]]:
    r = len(dims)
    rows = len(didx)
    cols = len(src)
    B = [[zero] * cols for _ in range(rows)]
    for j, g in enumerate(src):
        for c, coef in form.items():
            if coef == zero:
                continue
            h = tuple(
                tuple(g[k][t] + (1 if c[k] == t + 1 else 0) for t in range(dims[k]))
                for k in range(r)
            )
            B[didx[h]][j] += coef
    return B


def sylvester_blocks(
    dims: tuple[int, ...],
    T_dict: dict[tuple[int, ...], Any],
    mus: list[list[Any]],
    m: Sequence[int] | None = None,
    *,
    integer_scaled: bool = True,
) -> tuple[list[list[Any]], list[list[Any]], list[list[Any]]]:
    """Build the block decomposition (M_0, K, J) of Phi(x) = [M_0 | K - x J].

    When integer_scaled=True and inputs are exact (int/Fraction), entries are
    exact integers corresponding to tau * l_{k,i} and tau * l_x (after clearing
    any input denominators via clear_denominators).
    When integer_scaled=False, builds exact Fraction (if exact inputs) or float
    matrices directly for l_{k,i} and l_x as written in Section 2 of the paper.
    """
    if m is None:
        m = sz_multidegree(dims)
    cells = cells_of(dims)
    corner = (1,) * len(dims)
    src = monomials(dims, [mi - 1 for mi in m])
    dst = monomials(dims, m)
    didx = {g: i for i, g in enumerate(dst)}
    rows = len(dst)
    N = len(src)

    if integer_scaled:
        T_use, mus_use, _, _ = clear_denominators(T_dict, mus)
        tot = sum(mus_use[0])
        zero: Any = 0
        forms_0 = []
        for k, d in enumerate(dims):
            for i in range(2, d + 1):
                forms_0.append(
                    {c: (tot * (1 if c[k] == i else 0) - mus_use[k][i - 1]) * T_use[c] for c in cells}
                )
        form_K = {c: (tot * T_use[c] if c == corner else 0) for c in cells}
        form_J = {c: T_use[c] for c in cells}
    else:
        exact = all(_is_exact_scalar(v) for v in T_dict.values()) and all(
            _is_exact_scalar(v) for mk in mus for v in mk
        )
        if exact:
            T_use = {c: _to_fraction(v) for c, v in T_dict.items()}
            mus_use = [[_to_fraction(v) for v in mk] for mk in mus]
            tot = sum(mus_use[0])
            zero = Fraction(0)
            forms_0 = []
            for k, d in enumerate(dims):
                for i in range(2, d + 1):
                    mu_ratio = mus_use[k][i - 1] / tot
                    forms_0.append(
                        {c: ((1 if c[k] == i else 0) - mu_ratio) * T_use[c] for c in cells}
                    )
            form_K = {c: (T_use[c] if c == corner else Fraction(0)) for c in cells}
            form_J = {c: T_use[c] / tot for c in cells}
        else:
            T_use = {c: float(v) for c, v in T_dict.items()}
            mus_use = [[float(v) for v in mk] for mk in mus]
            tot = sum(mus_use[0])
            zero = 0.0
            forms_0 = []
            for k, d in enumerate(dims):
                for i in range(2, d + 1):
                    mu_ratio = mus_use[k][i - 1] / tot
                    forms_0.append(
                        {c: ((1.0 if c[k] == i else 0.0) - mu_ratio) * T_use[c] for c in cells}
                    )
            form_K = {c: (T_use[c] if c == corner else 0.0) for c in cells}
            form_J = {c: T_use[c] / tot for c in cells}

    dN = len(forms_0) * N
    M0 = [[zero] * dN for _ in range(rows)]
    for bi, form in enumerate(forms_0):
        B = _build_block(dims, src, didx, form, zero)
        off = bi * N
        for r in range(rows):
            M0[r][off : off + N] = B[r]
    K = _build_block(dims, src, didx, form_K, zero)
    J = _build_block(dims, src, didx, form_J, zero)
    return M0, K, J


def sylvester_matrix(
    T: TensorInput,
    margins: Sequence[Sequence[Any]],
    x: Any,
    *,
    entry: tuple[int, ...] | None = None,
    m: Sequence[int] | None = None,
    integer_scaled: bool = False,
) -> list[list[Any]]:
    """Build the square Sylvester matrix Phi(x) = [M_0 | K - x J] at parameter x."""
    dims, T_dict = normalize_tensor(T)
    mus, _ = normalize_margins(dims, margins)
    if entry is not None:
        T_dict, mus, _ = permute_to_corner(dims, T_dict, mus, entry)
    M0, K, J = sylvester_blocks(dims, T_dict, mus, m=m, integer_scaled=integer_scaled)
    rows = len(M0)
    N = len(K[0])
    return [M0[r] + [K[r][c] - x * J[r][c] for c in range(N)] for r in range(rows)]
