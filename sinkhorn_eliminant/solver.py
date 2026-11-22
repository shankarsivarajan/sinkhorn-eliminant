"""Polynomial root-finding, null-vector recovery, and Sinkhorn limit selection.

Implements Corollary 3.3 (N x N pencil & full solution recovery from the left
null vector of Phi(x)), Theorem 3.4 (uniqueness of the positive null vector /
positive scaling limit), and Proposition 6.1(4) (sign-twist vectors for real
solutions) of:
    S. Sivarajan, "An explicit determinantal formula for Sinkhorn limits of
    matrices and tensors" (HexagonMath 2611.00433).
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from fractions import Fraction
from math import prod
from typing import Any, Sequence

from .resultant import EliminantResult, eliminant_polynomial
from .sylvester import (
    TensorInput,
    cells_of,
    monomials,
    normalize_margins,
    normalize_tensor,
    permute_to_corner,
    sylvester_blocks,
    sz_multidegree,
)


# ---------------------------------------------------------------------------
# Pure-Python arbitrary-precision complex Decimal helpers (no external deps)
# ---------------------------------------------------------------------------

def _to_dec(v: Any) -> Decimal:
    if isinstance(v, Decimal):
        return v
    if isinstance(v, int) and not isinstance(v, bool):
        return Decimal(v)
    if isinstance(v, Fraction):
        return Decimal(v.numerator) / Decimal(v.denominator)
    return Decimal(str(float(v)))


def _cd_eval_poly_deriv(coeffs_asc: list[Decimal], zr: Decimal, zi: Decimal) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    """Evaluate p(z) and p'(z) for z = zr + i*zi using Horner's method."""
    pr, pi = Decimal(0), Decimal(0)
    dpr, dpi = Decimal(0), Decimal(0)
    for c in reversed(coeffs_asc):
        # dp = dp * z + p
        dpr, dpi = dpr * zr - dpi * zi + pr, dpr * zi + dpi * zr + pi
        # p = p * z + c
        pr, pi = pr * zr - pi * zi + c, pr * zi + pi * zr
    return pr, pi, dpr, dpi


def _cd_div(ar: Decimal, ai: Decimal, br: Decimal, bi: Decimal) -> tuple[Decimal, Decimal]:
    den = br * br + bi * bi
    return (ar * br + ai * bi) / den, (ai * br - ar * bi) / den


def _real_newton_polish(c_dec: list[Decimal], zr: Decimal, dps: int, steps: int = 4) -> Decimal:
    """Refine a real root with real Decimal Newton iterations."""
    tol = Decimal(10) ** (-(dps - 6))
    for _ in range(steps):
        p = Decimal(0)
        dp = Decimal(0)
        for c in reversed(c_dec):
            dp = dp * zr + p
            p = p * zr + c
        if dp == 0:
            break
        step = p / dp
        zr -= step
        if abs(step) <= tol * max(Decimal(1), abs(zr)):
            break
    return zr


def _polish_root_decimal(
    coeffs_asc: list[int | Fraction],
    z0: complex,
    dps: int = 60,
    steps: int = 20,
) -> tuple[Decimal, Decimal]:
    """Polish a simple root z0 of an exact polynomial using Decimal Newton iterations."""
    work_prec = dps + max(40, 2 * len(coeffs_asc))
    with localcontext() as ctx:
        ctx.prec = work_prec
        c_dec = [_to_dec(c) for c in coeffs_asc]
        zr = Decimal(repr(z0.real))
        zi = Decimal(repr(z0.imag))
        step_tol = Decimal(10) ** (-(dps - 8))
        for _ in range(steps):
            pr, pi, dpr, dpi = _cd_eval_poly_deriv(c_dec, zr, zi)
            if dpr == 0 and dpi == 0:
                break
            dr, di = _cd_div(pr, pi, dpr, dpi)
            zr -= dr
            zi -= di
            scale = max(Decimal(1), abs(zr) + abs(zi))
            if abs(dr) + abs(di) < step_tol * scale:
                break
        real_tol = Decimal(10) ** (-(dps - 15)) * max(Decimal(1), abs(zr))
        if abs(zi) < real_tol:
            zi = Decimal(0)
            zr = _real_newton_polish(c_dec, zr, dps=dps)
        return zr, zi


def _strip_and_factor_zeros(coeffs_asc: Sequence[int | Fraction | float]) -> tuple[list[Any], int]:
    """Strip leading zero high-order coefficients and factor out x = 0 roots."""
    c = list(coeffs_asc)
    while len(c) > 1 and c[-1] == 0:
        c.pop()
    deg = len(c) - 1
    if deg <= 0:
        return [], 0
    k0 = 0
    while k0 < deg and c[k0] == 0:
        k0 += 1
    return c[k0:], k0


def _aberth_ehrlich(coeffs_asc: Sequence[int | Fraction | float]) -> list[complex]:
    """Find all complex roots of a polynomial in pure Python via float64 Aberth-Ehrlich."""
    c, k0 = _strip_and_factor_zeros(coeffs_asc)
    if not c:
        return []
    roots: list[complex] = [0.0 + 0.0j] * k0
    deg = len(c) - 1
    if deg == 0:
        return roots
    if deg == 1:
        if isinstance(c[0], (int, Fraction)) and isinstance(c[1], (int, Fraction)):
            roots.append(complex(-Fraction(c[0]) / Fraction(c[1])))
        else:
            roots.append(complex(-c[0] / c[1]))
        return roots

    # Rescale x = r_scale * w so |c_0| and |c_d| are balanced, then divide by max|c_j|
    # Using Decimal to avoid overflow on 200+ digit integer coefficients
    with localcontext() as ctx:
        ctx.prec = 40
        cd_dec = [_to_dec(v) for v in c]
        r_scale = float((abs(cd_dec[0]) / abs(cd_dec[-1])) ** (Decimal(1) / Decimal(deg)))
        if r_scale == 0 or not math.isfinite(r_scale):
            r_scale = 1.0
        r_dec = Decimal(repr(r_scale))
        scaled_dec = [cd_dec[j] * (r_dec**j) for j in range(deg + 1)]
        max_mag = max(abs(v) for v in scaled_dec)
        cf = [float(v / max_mag) for v in scaled_dec]

    lead = cf[-1]
    cf = [v / lead for v in cf]

    # Upper bound on roots in w-plane (Cauchy bound)
    radius = 1.0 + max(abs(v) for v in cf[:-1])
    radius = min(radius, 2.0 + sum(abs(v) for v in cf[:-1]) ** (1.0 / deg))
    # Initialize on two slightly perturbed circles to break symmetry
    z = [
        (0.5 + 0.4 * (j % 2)) * radius * cmath.exp(2j * math.pi * (j + 0.17) / deg)
        for j in range(deg)
    ]

    for _ in range(400):
        max_step = 0.0
        for i in range(deg):
            zi = z[i]
            # Evaluate p(zi) and p'(zi)
            p = cf[-1]
            dp = 0.0 + 0.0j
            for coef in reversed(cf[:-1]):
                dp = dp * zi + p
                p = p * zi + coef
            if abs(p) < 1e-300:
                continue
            if dp == 0:
                z[i] += 1e-7 * (1.0 + 0.5j)
                continue
            w = p / dp
            s = sum(1.0 / (zi - z[j]) for j in range(deg) if j != i and zi != z[j])
            denom = 1.0 - w * s
            delta = w / denom if denom != 0 else w
            z[i] = zi - delta
            step = abs(delta) / max(1.0, abs(z[i]))
            if step > max_step:
                max_step = step
        if max_step < 1e-14:
            break

    return roots + [zi * r_scale for zi in z]


def _aberth_ehrlich_decimal(c: list[int | Fraction], *, dps: int = 60) -> list[tuple[Decimal, Decimal]]:
    """High-precision Decimal Aberth-Ehrlich root finder for reduced polynomial c[0] != 0, deg >= 2."""
    deg = len(c) - 1
    work_prec = dps + max(40, 2 * deg)
    with localcontext() as ctx:
        ctx.prec = work_prec
        cd_dec = [_to_dec(v) for v in c]
        r_scale = (abs(cd_dec[0]) / abs(cd_dec[-1])) ** (Decimal(1) / Decimal(deg))
        if r_scale == 0:
            r_scale = Decimal(1)
        scaled = [cd_dec[j] * (r_scale**j) for j in range(deg + 1)]
        lead = scaled[-1]
        cf = [v / lead for v in scaled]

        cauchy = Decimal(1) + max(abs(v) for v in cf[:-1])
        sum_bound = Decimal(2) + (sum((abs(v) for v in cf[:-1]), Decimal(0)) ** (Decimal(1) / Decimal(deg)))
        radius = min(cauchy, sum_bound)

        zr: list[Decimal] = []
        zi: list[Decimal] = []
        for j in range(deg):
            ang = 2.0 * math.pi * (j + 0.17) / deg
            rad = Decimal(repr(0.5 + 0.4 * (j % 2))) * radius
            zr.append(rad * Decimal(repr(math.cos(ang))))
            zi.append(rad * Decimal(repr(math.sin(ang))))

        conv_tol = Decimal(10) ** (-(dps + 5))
        for _ in range(400):
            max_step = Decimal(0)
            for i in range(deg):
                xr, xi = zr[i], zi[i]
                pr, pi, dpr, dpi = _cd_eval_poly_deriv(cf, xr, xi)
                if pr == 0 and pi == 0:
                    continue
                if dpr == 0 and dpi == 0:
                    zr[i] += Decimal("1e-7")
                    zi[i] += Decimal("5e-8")
                    continue
                wr, wi = _cd_div(pr, pi, dpr, dpi)
                sr, si = Decimal(0), Decimal(0)
                for j in range(deg):
                    if j == i:
                        continue
                    dr, di = xr - zr[j], xi - zi[j]
                    d2 = dr * dr + di * di
                    if d2 == 0:
                        continue
                    sr += dr / d2
                    si -= di / d2
                denom_r = Decimal(1) - (wr * sr - wi * si)
                denom_i = -(wr * si + wi * sr)
                if denom_r == 0 and denom_i == 0:
                    delr, deli = wr, wi
                else:
                    delr, deli = _cd_div(wr, wi, denom_r, denom_i)
                zr[i] -= delr
                zi[i] -= deli
                step = (abs(delr) + abs(deli)) / max(Decimal(1), abs(zr[i]) + abs(zi[i]))
                if step > max_step:
                    max_step = step
            if max_step < conv_tol:
                break

        out: list[tuple[Decimal, Decimal]] = []
        for i in range(deg):
            xr = zr[i] * r_scale
            xi = zi[i] * r_scale
            if abs(xi) < Decimal(10) ** (-(dps - 15)) * max(Decimal(1), abs(xr)):
                xi = Decimal(0)
                xr = _real_newton_polish(cd_dec, xr, dps=dps)
            out.append((xr, xi))
        return out


def _roots_need_decimal_fallback(
    c: list[int | Fraction],
    roots: list[tuple[Decimal, Decimal]],
    *,
    dps: int,
) -> bool:
    """Return True if float-seeded polished roots failed residual, distinctness, or conjugacy checks."""
    deg = len(c) - 1
    work_prec = dps + max(40, 2 * deg)
    with localcontext() as ctx:
        ctx.prec = work_prec
        c_dec = [_to_dec(v) for v in c]
        res_tol = Decimal(10) ** (-max(15, dps - 20))
        conj_tol = Decimal(10) ** (-max(10, dps // 2))
        dup_tol = Decimal("1e-10")
        dp_zero_tol = Decimal("1e-8")

        mags: list[Decimal] = []
        dp_vals: list[tuple[Decimal, Decimal]] = []
        for zr, zi in roots:
            pr, pi, dpr, dpi = _cd_eval_poly_deriv(c_dec, zr, zi)
            r = (zr * zr + zi * zi).sqrt()
            mags.append(r)
            dp_vals.append(((dpr * dpr + dpi * dpi).sqrt(), r))
            p_norm = sum((abs(ck) * (r**k) for k, ck in enumerate(c_dec)), Decimal(0))
            if p_norm > 0 and (pr * pr + pi * pi).sqrt() / p_norm > res_tol:
                return True

        # Complex roots of a real polynomial must appear in conjugate pairs
        nonreal = [(zr, zi, mags[i]) for i, (zr, zi) in enumerate(roots) if zi != 0]
        if len(nonreal) % 2 != 0:
            return True
        pos_im = sorted([t for t in nonreal if t[1] > 0], key=lambda t: (t[0], t[1]))
        neg_im = sorted([(t[0], -t[1], t[2]) for t in nonreal if t[1] < 0], key=lambda t: (t[0], t[1]))
        if len(pos_im) != len(neg_im):
            return True
        for (xr1, yi1, r1), (xr2, yi2, r2) in zip(pos_im, neg_im):
            scale = max(Decimal(1), r1, r2)
            if abs(xr1 - xr2) + abs(yi1 - yi2) > conj_tol * scale:
                return True

        # Check for collapsed simple roots (excluding genuine multiple roots where p'(z) ~ 0)
        for i in range(len(roots)):
            zr1, zi1 = roots[i]
            dp_mag, r1 = dp_vals[i]
            scale = max(Decimal(1), r1)
            for j in range(i + 1, len(roots)):
                zr2, zi2 = roots[j]
                if abs(zr1 - zr2) + abs(zi1 - zi2) < dup_tol * scale:
                    dp_norm = sum((Decimal(k) * abs(c_dec[k]) * (r1 ** (k - 1)) for k in range(1, deg + 1)), Decimal(0))
                    if dp_norm > 0 and dp_mag > dp_zero_tol * dp_norm:
                        return True
    return False


def _find_roots_high_precision(
    coeffs_asc: Sequence[int | Fraction],
    *,
    dps: int = 60,
) -> list[tuple[Decimal, Decimal]]:
    """Find all roots as high-precision (real, imag) Decimal pairs."""
    c_all = list(coeffs_asc)
    while len(c_all) > 1 and c_all[-1] == 0:
        c_all.pop()
    if len(c_all) <= 1:
        return []

    try:
        import mpmath as mp

        with mp.workdps(dps):
            roots_mp = mp.polyroots(
                [mp.mpf(v.numerator) / mp.mpf(v.denominator) if isinstance(v, Fraction) else mp.mpf(v) for v in reversed(c_all)],
                maxsteps=2000,
                extraprec=max(400, 10 * len(c_all)),
            )
            out = []
            for z in roots_mp:
                re_s = mp.nstr(mp.re(z), dps)
                im_s = mp.nstr(mp.im(z), dps)
                zr, zi = Decimal(re_s), Decimal(im_s)
                if abs(zi) < Decimal(10) ** (-(dps - 15)) * max(Decimal(1), abs(zr)):
                    zi = Decimal(0)
                out.append((zr, zi))
            return out
    except ImportError:
        pass

    c_red, k0 = _strip_and_factor_zeros(c_all)
    zero_roots = [(Decimal(0), Decimal(0))] * k0
    deg = len(c_red) - 1
    if deg == 0:
        return zero_roots
    if deg == 1:
        with localcontext() as ctx:
            ctx.prec = dps + 20
            return zero_roots + [(-_to_dec(c_red[0]) / _to_dec(c_red[1]), Decimal(0))]

    approx = _aberth_ehrlich(c_red)
    polished = [_polish_root_decimal(c_red, z0, dps=dps) for z0 in approx]
    if _roots_need_decimal_fallback(c_red, polished, dps=dps):
        polished = _aberth_ehrlich_decimal(c_red, dps=dps)
    return zero_roots + polished


def find_polynomial_roots(
    coeffs_asc: Sequence[int | Fraction | float],
    *,
    dps: int = 60,
) -> list[complex]:
    """Compute all complex roots of a polynomial (coeffs_asc: low-to-high).

    Uses mpmath.polyroots if available; otherwise uses pure-Python Aberth-Ehrlich
    followed by `dps`-digit Decimal Newton polishing (with automatic Decimal
    Aberth-Ehrlich fallback when float64 seeding loses precision).
    """
    c = list(coeffs_asc)
    while len(c) > 1 and c[-1] == 0:
        c.pop()
    if len(c) <= 1:
        return []

    if all(isinstance(v, (int, Fraction)) for v in c):
        hp = _find_roots_high_precision(c, dps=dps)
        return [complex(float(zr), float(zi)) for zr, zi in hp]

    try:
        import mpmath as mp

        with mp.workdps(dps):
            roots_mp = mp.polyroots([mp.mpf(v) for v in reversed(c)], maxsteps=2000, extraprec=max(400, 10 * len(c)))
            return [complex(mp.re(z), mp.im(z)) for z in roots_mp]
    except ImportError:
        pass

    return _aberth_ehrlich(c)



# ---------------------------------------------------------------------------
# Pure-Python linear algebra for left null vector recovery
# ---------------------------------------------------------------------------

def _null_vector_complete_pivot_float(M: list[list[Any]]) -> list[Any]:
    """Compute a 1D left null vector v of M (v^T M = 0) via complete pivoting on M^T."""
    rows = len(M)
    cols = len(M[0])
    # Transpose: A is cols x rows; we seek v of length rows with A v = 0
    A = [[M[r][c] for r in range(rows)] for c in range(cols)]
    col_perm = list(range(rows))
    rank = min(cols, rows - 1)

    for k in range(rank):
        # Complete pivot in submatrix A[k:, k:]
        best_val = -1.0
        best_r, best_c = k, k
        for r in range(k, cols):
            ar = A[r]
            for c in range(k, rows):
                av = abs(ar[c])
                if av > best_val:
                    best_val = av
                    best_r, best_c = r, c
        if best_val == 0.0:
            break
        if best_r != k:
            A[k], A[best_r] = A[best_r], A[k]
        if best_c != k:
            for r in range(cols):
                A[r][k], A[r][best_c] = A[r][best_c], A[r][k]
            col_perm[k], col_perm[best_c] = col_perm[best_c], col_perm[k]

        piv = A[k][k]
        prow = A[k]
        for r in range(k + 1, cols):
            ar = A[r]
            f = ar[k]
            if f != 0:
                factor = f / piv
                for c in range(k + 1, rows):
                    ar[c] -= factor * prow[c]
                ar[k] = 0

    # Back-substitution setting free variable (last permuted coordinate) = 1
    x_perm = [0.0 * M[0][0]] * rows
    x_perm[rows - 1] = 1.0 if isinstance(M[0][0], float) else (1.0 + 0.0j)
    for k in range(rows - 2, -1, -1):
        if k >= cols or A[k][k] == 0:
            x_perm[k] = 0.0
            continue
        s = sum(A[k][c] * x_perm[c] for c in range(k + 1, rows))
        x_perm[k] = -s / A[k][k]

    v = [0.0 * M[0][0]] * rows
    for p_idx, orig_idx in enumerate(col_perm):
        v[orig_idx] = x_perm[p_idx]
    return v


def _null_vector_complete_pivot_decimal(M: list[list[Decimal]]) -> list[Decimal]:
    """60-digit Decimal left null vector of real matrix M via complete pivoting on M^T."""
    rows = len(M)
    cols = len(M[0])
    A = [[M[r][c] for r in range(rows)] for c in range(cols)]
    col_perm = list(range(rows))
    rank = min(cols, rows - 1)

    for k in range(rank):
        best_val = Decimal(-1)
        best_r, best_c = k, k
        for r in range(k, cols):
            ar = A[r]
            for c in range(k, rows):
                av = abs(ar[c])
                if av > best_val:
                    best_val = av
                    best_r, best_c = r, c
        if best_val == 0:
            break
        if best_r != k:
            A[k], A[best_r] = A[best_r], A[k]
        if best_c != k:
            for r in range(cols):
                A[r][k], A[r][best_c] = A[r][best_c], A[r][k]
            col_perm[k], col_perm[best_c] = col_perm[best_c], col_perm[k]

        piv = A[k][k]
        prow = A[k]
        for r in range(k + 1, cols):
            ar = A[r]
            f = ar[k]
            if f != 0:
                factor = f / piv
                for c in range(k + 1, rows):
                    ar[c] -= factor * prow[c]
                ar[k] = Decimal(0)

    x_perm = [Decimal(0)] * rows
    x_perm[rows - 1] = Decimal(1)
    for k in range(rows - 2, -1, -1):
        if k >= cols or A[k][k] == 0:
            continue
        s = sum((A[k][c] * x_perm[c] for c in range(k + 1, rows)), Decimal(0))
        x_perm[k] = -s / A[k][k]

    v = [Decimal(0)] * rows
    for p_idx, orig_idx in enumerate(col_perm):
        v[orig_idx] = x_perm[p_idx]
    return v


def _extract_scalings_from_nullvec(
    dims: tuple[int, ...],
    rows_monos: list[tuple[tuple[int, ...], ...]],
    idx_map: dict[tuple[tuple[int, ...], ...], int],
    ev: list[Any],
) -> list[list[Any]]:
    """Extract u^(1), ..., u^(r) (normalized with u^(k)_1 = 1) from ev_y."""
    u = []
    for k, d in enumerate(dims):
        # Find a monomial with nonzero exponent on variable (k, 0) and nonzero entry if possible
        cands = [g for g in rows_monos if g[k][0] >= 1]
        base = max(cands, key=lambda g: abs(ev[idx_map[g]]))
        denom_val = ev[idx_map[base]]
        uk = [1]
        for i in range(1, d):
            g2 = [list(e) for e in base]
            g2[k][0] -= 1
            g2[k][i] += 1
            g2_t = tuple(tuple(e) for e in g2)
            uk.append(ev[idx_map[g2_t]] / denom_val)
        u.append(uk)
    return u


def _to_nested(dims: tuple[int, ...], P_dict: dict[tuple[int, ...], Any]) -> Any:
    if len(dims) == 1:
        return [P_dict[(i + 1,)] for i in range(dims[0])]

    def build(prefix: tuple[int, ...], depth: int) -> Any:
        if depth == len(dims) - 1:
            return [P_dict[prefix + (i + 1,)] for i in range(dims[depth])]
        return [build(prefix + (i + 1,), depth + 1) for i in range(dims[depth])]

    return build((), 0)


def _unpermute_P(
    dims: tuple[int, ...],
    P_perm: dict[tuple[int, ...], Any],
    mode_maps: list[list[int]],
) -> dict[tuple[int, ...], Any]:
    P_orig: dict[tuple[int, ...], Any] = {}
    for c_new, val in P_perm.items():
        c_old = tuple(mode_maps[k][c_new[k] - 1] + 1 for k in range(len(dims)))
        P_orig[c_old] = val
    return P_orig


def _unpermute_u(
    dims: tuple[int, ...],
    u_perm: list[list[Any]],
    mode_maps: list[list[int]],
) -> list[list[Any]]:
    u_orig = []
    for k, d in enumerate(dims):
        uk = [None] * d
        for new_i, old_i in enumerate(mode_maps[k]):
            uk[old_i] = u_perm[k][new_i]
        u_orig.append(uk)
    return u_orig


# ---------------------------------------------------------------------------
# Solution recovery & Sinkhorn iteration
# ---------------------------------------------------------------------------

@dataclass
class ScalingSolution:
    """One (complex or real) solution of the scaling equations recovered from a root."""

    root: complex
    real: bool
    feasible: bool  # True if real and 0 < x < min_k mu^(k)_1
    nullvec_positive: bool
    positive: bool  # True iff all entries of P are real and positive (selects the limit)
    margin_residual: float
    corner_check: float
    P: Any  # Nested lists matching original tensor shape
    scalings: list[list[Any]]  # u^(1), ..., u^(r)
    signs: list[list[int]] | None = None  # Sign vectors s^(1), ..., s^(r) for real solutions


@dataclass
class ScalingSolveResult:
    """Full solution set for a Sinkhorn/Kruithof scaling problem."""

    eliminant: EliminantResult
    limit_root: float
    limit_P: Any
    limit_scalings: list[list[float]]
    solutions: list[ScalingSolution] = field(default_factory=list)
    sinkhorn_P: Any | None = None
    sinkhorn_diff: float | None = None


def recover_solution(
    T: TensorInput,
    margins: Sequence[Sequence[Any]],
    x: complex | float | Decimal,
    *,
    entry: tuple[int, ...] | None = None,
    dps: int = 60,
    high_precision: bool = False,
) -> ScalingSolution:
    """Recover the full scaling solution P_xi and factors u^(k) from a root x (Corollary 3.3)."""
    dims, T_dict_orig = normalize_tensor(T)
    mus_orig, _ = normalize_margins(dims, margins)
    target_entry = (0,) * len(dims) if entry is None else tuple(int(i) for i in entry)
    if entry is not None:
        T_dict, mus, mode_maps = permute_to_corner(dims, T_dict_orig, mus_orig, target_entry)
    else:
        T_dict, mus = T_dict_orig, mus_orig
        mode_maps = [list(range(d)) for d in dims]

    cells = cells_of(dims)
    corner = (1,) * len(dims)
    m = sz_multidegree(dims)
    rows_monos = monomials(dims, m)
    idx_map = {g: i for i, g in enumerate(rows_monos)}
    tot_f = sum(float(v) for v in mus[0])
    ub = min(float(mk[0]) for mk in mus)

    is_real = isinstance(x, (int, float, Fraction, Decimal)) or abs(complex(x).imag) < 1e-12
    x_c = complex(x) if not isinstance(x, Decimal) else complex(float(x), 0.0)

    def build_and_recover_float(x_val: complex) -> tuple[list[Any], dict[tuple[int, ...], complex], list[list[complex]], float]:
        M0, K, J = sylvester_blocks(dims, T_dict, mus, m=m, integer_scaled=False)
        rows = len(M0)
        N = len(K[0])
        if is_real:
            xv: Any = float(x_val.real)
            M = [[float(v) for v in M0[r]] + [float(K[r][c]) - xv * float(J[r][c]) for c in range(N)] for r in range(rows)]
        else:
            M = [[complex(v) for v in M0[r]] + [complex(K[r][c]) - x_val * complex(J[r][c]) for c in range(N)] for r in range(rows)]
        try:
            import numpy as np

            U, _, _ = np.linalg.svd(np.array(M))
            ev = np.conj(U[:, -1]).tolist()
        except ImportError:
            ev = _null_vector_complete_pivot_float(M)

        u = _extract_scalings_from_nullvec(dims, rows_monos, idx_map, ev)
        P_raw = {c: complex(T_dict[c]) * prod(u[k][c[k] - 1] for k in range(len(dims))) for c in cells}
        s = sum(P_raw.values())
        scale = tot_f / s if s != 0 else 1.0
        P_p = {c: v * scale for c, v in P_raw.items()}
        # Normalize u^(1) to absorb scale
        u[0] = [ui * scale for ui in u[0]]
        res = max(
            abs(sum(P_p[c] for c in cells if c[k] == i + 1) - float(mus[k][i]))
            for k in range(len(dims))
            for i in range(dims[k])
        )
        return ev, P_p, u, res

    ev_f, P_p, u_p, res = build_and_recover_float(x_c)

    if is_real and (high_precision or res > 1e-6):
        with localcontext() as ctx:
            ctx.prec = dps
            T_dec = {c: _to_dec(v) for c, v in T_dict.items()}
            mus_dec = [[_to_dec(v) for v in mk] for mk in mus]
            tot_dec = sum(mus_dec[0])
            x_dec = x if isinstance(x, Decimal) else _to_dec(x_c.real)
            M0, K, J = sylvester_blocks(dims, T_dict, mus, m=m, integer_scaled=False)
            rows = len(M0)
            N = len(K[0])
            M_dec = [
                [_to_dec(v) for v in M0[r]] + [_to_dec(K[r][c]) - x_dec * _to_dec(J[r][c]) for c in range(N)]
                for r in range(rows)
            ]
            ev_dec = _null_vector_complete_pivot_decimal(M_dec)
            u_dec = _extract_scalings_from_nullvec(dims, rows_monos, idx_map, ev_dec)
            P_raw_d = {c: T_dec[c] * prod((u_dec[k][c[k] - 1] for k in range(len(dims))), start=Decimal(1)) for c in cells}
            s_d = sum(P_raw_d.values(), Decimal(0))
            if s_d != 0:
                scale_d = tot_dec / s_d
                P_dec = {c: v * scale_d for c, v in P_raw_d.items()}
                u_dec[0] = [ui * scale_d for ui in u_dec[0]]
                res_d = float(
                    max(
                        abs(sum((P_dec[c] for c in cells if c[k] == i + 1), Decimal(0)) - mus_dec[k][i])
                        for k in range(len(dims))
                        for i in range(dims[k])
                    )
                )
                if res_d < res or high_precision:
                    ev_f = [float(v) for v in ev_dec]
                    P_p = {c: complex(float(v), 0.0) for c, v in P_dec.items()}
                    u_p = [[complex(float(v), 0.0) for v in uk] for uk in u_dec]
                    res = res_d

    # Normalize null vector positivity check
    if is_real:
        ev_real = [complex(v).real for v in ev_f]
        max_abs = max(abs(v) for v in ev_real) or 1.0
        tol_nv = 1e-10 * max_abs
        nullvec_pos = all(v > tol_nv for v in ev_real) or all(v < -tol_nv for v in ev_real)
    else:
        nullvec_pos = False

    positive = all(abs(v.imag) < 1e-7 and v.real > 0 for v in P_p.values())
    corner_err = float(abs(P_p[corner] - x_c))

    # Sign vectors for real solutions (Proposition 6.1(4), normalized s^(k)_1 = +1 for k >= 2)
    signs = None
    if is_real:
        P_orig_c = _unpermute_P(dims, {c: v.real for c, v in P_p.items()}, mode_maps)
        # Wait: in Appendix A, sign vectors are reported in the permuted / original mode order;
        # compute in original coordinates normalized so s^(k)_1 = +1 for k >= 2
        sgn = lambda v: 1 if v > 0 else -1
        corner_orig = tuple(idx + 1 for idx in target_entry)
        s1 = sgn(P_orig_c[corner_orig])
        svec = []
        for k, d in enumerate(dims):
            vec = []
            for i in range(1, d + 1):
                c = list(corner_orig)
                c[k] = i
                vec.append(sgn(P_orig_c[tuple(c)]) * (1 if k == 0 else s1))
            svec.append(vec)
        signs = svec
    else:
        P_orig_c = _unpermute_P(dims, P_p, mode_maps)

    u_orig = _unpermute_u(dims, u_p, mode_maps)
    P_out = _to_nested(dims, {c: (v.real if is_real else v) for c, v in P_orig_c.items()})
    u_out = [[(v.real if is_real else v) for v in uk] for uk in u_orig]

    return ScalingSolution(
        root=complex(x_c.real, 0.0) if is_real else x_c,
        real=is_real,
        feasible=bool(is_real and 0.0 < x_c.real < ub),
        nullvec_positive=nullvec_pos,
        positive=positive,
        margin_residual=float(res),
        corner_check=corner_err,
        P=P_out,
        scalings=u_out,
        signs=signs,
    )


def sinkhorn_scale(
    T: TensorInput,
    margins: Sequence[Sequence[Any]],
    *,
    tol: float = 1e-13,
    max_iters: int = 50000,
) -> tuple[Any, float]:
    """Direct Sinkhorn/Kruithof iterative scaling (cyclic mode rescaling).

    Returns (P_nested, max_margin_error).
    """
    dims, T_dict = normalize_tensor(T)
    mus, _ = normalize_margins(dims, margins)
    cells = cells_of(dims)
    P = {c: float(T_dict[c]) for c in cells}
    mus_f = [[float(v) for v in mk] for mk in mus]

    err = float("inf")
    for _ in range(max_iters):
        for k, d in enumerate(dims):
            slice_sums = [0.0] * d
            for c in cells:
                slice_sums[c[k] - 1] += P[c]
            factors = [mus_f[k][i] / slice_sums[i] for i in range(d)]
            for c in cells:
                P[c] *= factors[c[k] - 1]
        err = max(
            abs(sum(P[c] for c in cells if c[k] == i + 1) - mus_f[k][i])
            for k, d in enumerate(dims)
            for i in range(d)
        )
        if err < tol:
            break
    return _to_nested(dims, P), err


def solve_scaling(
    T: TensorInput,
    margins: Sequence[Sequence[Any]],
    *,
    entry: tuple[int, ...] | None = None,
    recover_all: bool = True,
    check_sinkhorn: bool = True,
    dps: int = 60,
    backend: str = "auto",
) -> ScalingSolveResult:
    """Compute the exact eliminant, find its roots, recover solutions, and select the limit.

    Implements Theorems 3.2, 3.4, and Corollary 3.3.
    """
    elim = eliminant_polynomial(T, margins, entry=entry, backend=backend)
    if not any(elim.coeffs):
        hint = (
            " For 2D matrices, use matrix_diagonal_polynomial(T, margins) from Section 5,"
            " which avoids the content factor c(T)."
            if len(elim.dims) == 2
            else ""
        )
        raise RuntimeError(
            "Sylvester determinant det Phi(x) vanishes identically (c(T) = 0, "
            "meaning a face not containing the target entry has vanishing discriminant)."
            + hint
        )
    hp_roots = _find_roots_high_precision(elim.coeffs, dps=dps)

    dims, _ = normalize_tensor(T)
    mus, _ = normalize_margins(dims, margins)
    target_entry = (0,) * len(dims) if entry is None else tuple(int(i) for i in entry)
    ub = min(float(mus[k][target_entry[k]]) for k in range(len(dims)))

    # Sort: real roots first (ascending), then complex pairs
    hp_roots.sort(key=lambda pair: (pair[1] != 0, pair[0], pair[1]))

    solutions: list[ScalingSolution] = []
    roots_used: list[Any] = []
    for zr, zi in hp_roots:
        is_real = zi == 0
        if not recover_all:
            if not is_real:
                continue
            xr = float(zr)
            if not (0.0 < xr < ub):
                continue
        x_arg: Any = zr if is_real else complex(float(zr), float(zi))
        sol = recover_solution(T, margins, x_arg, entry=entry, dps=dps)
        solutions.append(sol)
        roots_used.append(x_arg)

    pos = [i for i, s in enumerate(solutions) if s.positive]
    if not pos:
        raise RuntimeError("No positive scaling solution found among the polynomial roots.")
    # The float null vector is accurate only to about 1e-11 for larger Sylvester matrices; recover the limit
    # itself from the high-precision root with a high-precision null vector.
    limit_sol = recover_solution(T, margins, roots_used[pos[0]], entry=entry, dps=dps, high_precision=True)
    solutions[pos[0]] = limit_sol

    sink_P = None
    sink_diff = None
    if check_sinkhorn:
        sink_P, _ = sinkhorn_scale(T, margins)
        # Flatten and compare max entrywise diff
        cells = cells_of(dims)
        _, lim_dict = normalize_tensor(limit_sol.P, dims=dims)
        _, s_dict = normalize_tensor(sink_P, dims=dims)
        sink_diff = max(abs(float(lim_dict[c]) - float(s_dict[c])) for c in cells)

    return ScalingSolveResult(
        eliminant=elim,
        limit_root=float(limit_sol.root.real),
        limit_P=limit_sol.P,
        limit_scalings=limit_sol.scalings,
        solutions=solutions,
        sinkhorn_P=sink_P,
        sinkhorn_diff=sink_diff,
    )


def solve_pencil_numerical(
    T: TensorInput,
    margins: Sequence[Sequence[Any]],
    *,
    entry: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Fast numerical N x N generalized eigenvalue solver (Corollary 3.3).

    Requires numpy and scipy. Builds Phi(x) = [M_0 | K - x J] in float64,
    finds an orthonormal basis Z_0 of ker(M_0^T), and solves the N x N
    generalized eigenproblem (Z_0^T K) v = x (Z_0^T J) v.
    """
    import numpy as np
    import scipy.linalg as sla

    dims, T_dict = normalize_tensor(T)
    mus, _ = normalize_margins(dims, margins)
    target_entry = (0,) * len(dims) if entry is None else tuple(int(i) for i in entry)
    if entry is not None:
        T_dict, mus, _ = permute_to_corner(dims, T_dict, mus, target_entry)

    m = sz_multidegree(dims)
    M0, K, J = sylvester_blocks(dims, T_dict, mus, m=m, integer_scaled=False)
    M0_np = np.array(M0, dtype=float)
    K_np = np.array(K, dtype=float)
    J_np = np.array(J, dtype=float)

    U, sv, _ = np.linalg.svd(M0_np, full_matrices=True)
    Z0 = U[:, M0_np.shape[1] :]
    ev = sla.eigvals(Z0.T @ K_np, Z0.T @ J_np)
    ev = ev[np.isfinite(ev)]

    sink_P, _ = sinkhorn_scale(T, margins)
    _, s_dict = normalize_tensor(sink_P, dims=dims)
    target_corner = tuple(idx + 1 for idx in target_entry)
    x_sink = float(s_dict[target_corner])
    nearest = ev[int(np.argmin(np.abs(ev - x_sink)))]
    return {
        "dims": dims,
        "entry": target_entry,
        "N": len(ev),
        "eigenvalues": ev.tolist(),
        "sinkhorn_entry": x_sink,
        "nearest_eigenvalue": complex(nearest),
        "rel_diff": float(abs(nearest - x_sink) / max(1e-300, abs(x_sink))),
        "sigma_ratio": float(sv[-1] / sv[0]),
    }
