"""Exact computation of the Sinkhorn/Kruithof eliminant polynomial.

Implements Theorem 3.2 (det Phi(x) = +-Res_Sigma) and Theorem 4.6 (content
factorization R = c(T) * E_tilde) of:
    S. Sivarajan, "An explicit determinantal formula for Sinkhorn limits of
    matrices and tensors" (HexagonMath 2611.00433).

By default uses a pure-Python multimodular CRT solver with Schur-complement
reduction of the x-independent block M_0 (and Newton forward-difference
interpolation), requiring zero third-party packages and guaranteed exact via
Hadamard bounds. An optional `python-flint` backend is also supported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import gcd, isqrt, lcm
from typing import Any, Sequence

from .discriminants import theorem46_factors
from .sylvester import (
    TensorInput,
    cells_of,
    clear_denominators,
    monomials,
    multinomial_degree,
    normalize_margins,
    normalize_tensor,
    permute_to_corner,
    sylvester_blocks,
    sz_multidegree,
)


def _is_prime_64(n: int) -> bool:
    """Deterministic Miller-Rabin primality test for all 64-bit integers."""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _trim_poly(coeffs: list[Any]) -> list[Any]:
    out = list(coeffs)
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def _primitive_int_poly(coeffs: Sequence[Fraction | int]) -> list[int]:
    """Return primitive integer polynomial (gcd = 1, positive leading coeff)."""
    trimmed = _trim_poly(list(coeffs))
    if not any(trimmed):
        return [0]
    fracs = [Fraction(c) for c in trimmed]
    den_lcm = 1
    for f in fracs:
        den_lcm = lcm(den_lcm, f.denominator)
    ints = [int(f * den_lcm) for f in fracs]
    g = 0
    for c in ints:
        g = gcd(g, abs(c))
    if g > 1:
        ints = [c // g for c in ints]
    if ints[-1] < 0:
        ints = [-c for c in ints]
    return ints


def _newton_interpolate_int(vals: list[int]) -> list[int]:
    """Interpolate polynomial p(x) from integer values p(0), p(1), ..., p(N)."""
    N = len(vals) - 1
    diff = list(vals)
    falling = [1]
    poly = [Fraction(diff[0])]
    fact = 1
    for k in range(1, N + 1):
        fact *= k
        diff = [diff[i + 1] - diff[i] for i in range(len(diff) - 1)]
        new_f = [0] * (k + 1)
        for j, coef in enumerate(falling):
            new_f[j] -= (k - 1) * coef
            new_f[j + 1] += coef
        falling = new_f
        poly.append(Fraction(0))
        c = Fraction(diff[0], fact)
        for j in range(k + 1):
            poly[j] += c * falling[j]
    assert all(c.denominator == 1 for c in poly)
    return [int(c) for c in poly]


def _det_pencil_mod_p(
    M0: list[list[int]],
    K: list[list[int]],
    J: list[list[int]],
    p: int,
) -> list[int]:
    """Evaluate det([M_0 | K - x J]) mod p at x = 0, 1, ..., N.

    Reduces the x-independent block M_0 once via Gaussian elimination mod p,
    then evaluates N x N determinants for each x.
    """
    rows = len(M0)
    dN = len(M0[0])
    N = len(K[0])

    A = [[v % p for v in M0[r]] + [v % p for v in K[r]] + [v % p for v in J[r]] for r in range(rows)]
    piv_prod = 1
    for col in range(dN):
        piv = -1
        for r in range(col, rows):
            if A[r][col] != 0:
                piv = r
                break
        if piv == -1:
            # First dN columns are linearly dependent mod p => det Phi(x) == 0 (mod p) for all x
            return [0] * (N + 1)
        if piv != col:
            A[col], A[piv] = A[piv], A[col]
            piv_prod = (p - piv_prod) % p
        pv = A[col][col]
        piv_prod = (piv_prod * pv) % p
        inv = pow(pv, -1, p)
        prow = A[col]
        for r in range(col + 1, rows):
            ar = A[r]
            f = ar[col]
            if f:
                f = (f * inv) % p
                for c in range(col + 1, dN + 2 * N):
                    if prow[c]:
                        ar[c] = (ar[c] - f * prow[c]) % p

    Kred = [A[r][dN : dN + N] for r in range(dN, rows)]
    Jred = [A[r][dN + N : dN + 2 * N] for r in range(dN, rows)]
    vals = []
    for xv in range(N + 1):
        B = [[(Kred[r][c] - xv * Jred[r][c]) % p for c in range(N)] for r in range(N)]
        det = piv_prod
        for i in range(N):
            piv = -1
            for r in range(i, N):
                if B[r][i] != 0:
                    piv = r
                    break
            if piv == -1:
                det = 0
                break
            if piv != i:
                B[i], B[piv] = B[piv], B[i]
                det = (p - det) % p
            pv = B[i][i]
            det = (det * pv) % p
            inv = pow(pv, -1, p)
            prow = B[i]
            for r in range(i + 1, N):
                br = B[r]
                f = br[i]
                if f:
                    f = (f * inv) % p
                    for c in range(i + 1, N):
                        br[c] = (br[c] - f * prow[c]) % p
        vals.append(det)
    return vals


def _hadamard_pencil_bound(
    M0: list[list[int]],
    K: list[list[int]],
    J: list[list[int]],
) -> int:
    """Upper bound on max_{0 <= x <= N} |det([M_0 | K - x J])| via Hadamard's inequality."""
    rows = len(M0)
    dN = len(M0[0])
    N = len(K[0])
    H0 = 1
    for c in range(dN):
        s2 = sum(M0[r][c] * M0[r][c] for r in range(rows))
        H0 *= isqrt(s2) + 1
    Hmax = 1
    for xv in range(N + 1):
        Hx = 1
        for c in range(N):
            s2 = sum((K[r][c] - xv * J[r][c]) ** 2 for r in range(rows))
            Hx *= isqrt(s2) + 1
        if Hx > Hmax:
            Hmax = Hx
    return H0 * Hmax


def _eval_pencil_crt(
    M0: list[list[int]],
    K: list[list[int]],
    J: list[list[int]],
) -> list[int]:
    """Compute exact integer determinants det([M_0 | K - x J]) at x = 0..N via CRT."""
    N = len(K[0])
    bound = 2 * _hadamard_pencil_bound(M0, K, J)
    Mprod = 1
    acc = [0] * (N + 1)
    p = (1 << 61) - 1
    while Mprod <= bound:
        while not _is_prime_64(p):
            p -= 2
        vs = _det_pencil_mod_p(M0, K, J, p)
        if Mprod == 1:
            acc = vs
            Mprod = p
        else:
            invM = pow(Mprod % p, -1, p)
            for i in range(N + 1):
                t = ((vs[i] - acc[i]) % p) * invM % p
                acc[i] += Mprod * t
            Mprod *= p
        p -= 2
    half = Mprod // 2
    return [v - Mprod if v > half else v for v in acc]


def _eval_pencil_flint(
    M0: list[list[int]],
    K: list[list[int]],
    J: list[list[int]],
) -> list[int]:
    import flint

    rows = len(M0)
    N = len(K[0])
    vals = []
    for xv in range(N + 1):
        M = [M0[r] + [K[r][c] - xv * J[r][c] for c in range(N)] for r in range(rows)]
        vals.append(int(flint.fmpz_mat(M).det()))
    return vals


def format_polynomial(coeffs: Sequence[Any], var: str = "x", descending: bool = True) -> str:
    """Format a polynomial given low-to-high coefficients into human-readable math notation."""
    coeffs_l = list(coeffs)
    if not any(coeffs_l):
        return "0"
    indices = range(len(coeffs_l) - 1, -1, -1) if descending else range(len(coeffs_l))
    parts = []
    for k in indices:
        c = coeffs_l[k]
        if c == 0:
            continue
        sign = "-" if c < 0 else "+"
        ac = -c if c < 0 else c
        if k == 0:
            term = str(ac)
        elif k == 1:
            term = var if ac == 1 else f"{ac}*{var}"
        else:
            term = f"{var}^{k}" if ac == 1 else f"{ac}*{var}^{k}"
        if not parts:
            parts.append(f"-{term}" if sign == "-" else term)
        else:
            parts.append(f" {sign} {term}")
    return "".join(parts)


@dataclass
class EliminantResult:
    """Result of computing the Sinkhorn/Kruithof eliminant polynomial."""

    dims: tuple[int, ...]
    entry: tuple[int, ...]
    expected_degree: int
    sylvester_size: int
    multidegree: tuple[int, ...]
    coeffs: list[int]  # Primitive integer polynomial in Z[x], low-to-high
    R_coeffs: list[Fraction]  # Exact det Phi(x) over Q[x], low-to-high
    R_int_coeffs: list[int]  # Integer-scaled Sylvester determinant, low-to-high
    E_tilde_coeffs: list[Any] | None = None  # Theorem 4.6 normalization when available
    theorem46: dict[str, Any] = field(default_factory=dict)

    @property
    def degree(self) -> int:
        return len(self.coeffs) - 1 if any(self.coeffs) else -1

    @property
    def coeffs_desc(self) -> list[int]:
        return list(reversed(self.coeffs))

    @property
    def is_degenerate(self) -> bool:
        return self.degree < self.expected_degree

    def evaluate(self, x: Any, which: str = "primitive") -> Any:
        """Evaluate the polynomial at x using Horner's method."""
        seq = {
            "primitive": self.coeffs,
            "R": self.R_coeffs,
            "R_int": self.R_int_coeffs,
            "E_tilde": self.E_tilde_coeffs,
        }[which]
        if seq is None:
            raise ValueError(f"Polynomial variant {which!r} is not available for this tensor.")
        acc: Any = 0
        for c in reversed(seq):
            acc = acc * x + c
        return acc

    def format(self, which: str = "primitive", var: str = "x") -> str:
        seq = {
            "primitive": self.coeffs,
            "R": self.R_coeffs,
            "R_int": self.R_int_coeffs,
            "E_tilde": self.E_tilde_coeffs,
        }[which]
        if seq is None:
            raise ValueError(f"Polynomial variant {which!r} is not available for this tensor.")
        return format_polynomial(seq, var=var)


def eliminant_polynomial(
    T: TensorInput,
    margins: Sequence[Sequence[Any]],
    *,
    entry: tuple[int, ...] | None = None,
    multidegree_perm: Sequence[int] | None = None,
    backend: str = "auto",
) -> EliminantResult:
    """Compute the exact Sinkhorn/Kruithof eliminant polynomial for T[entry].

    Parameters
    ----------
    T : nested list/tuple, numpy.ndarray, or 1-based cell dict
        Positive (or nonnegative/general exact) input matrix or tensor of format
        (n_1, ..., n_r) with n_j >= 2.
    margins : sequence of r 1D sequences
        Target margins mu^(1), ..., mu^(r) with a common nonzero total tau.
    entry : tuple of int, optional
        0-based indices of the target entry P[entry] (defaults to corner (0, ..., 0)).
    multidegree_perm : sequence of int, optional
        Permutation pi of {0, ..., r-1} selecting one of the r! admissible
        Sturmfels-Zelevinsky multidegrees m^pi (Theorem 3.2 / Remark 3.5).
    backend : {"auto", "python", "flint"}
        Determinant computation backend. "python" uses pure-Python multimodular
        CRT; "flint" uses python-flint fmpz_mat; "auto" uses python CRT by default.

    Returns
    -------
    EliminantResult
        Contains the primitive integer polynomial `coeffs` (low-to-high), exact
        Sylvester determinant `R_coeffs`, and Theorem 4.6 `E_tilde_coeffs` when
        all face discriminants are supported.
    """
    dims, T_dict = normalize_tensor(T)
    mus, _ = normalize_margins(dims, margins)
    target_entry = (0,) * len(dims) if entry is None else tuple(int(i) for i in entry)
    if entry is not None:
        T_dict, mus, _ = permute_to_corner(dims, T_dict, mus, target_entry)

    T_int, mus_int, D_T, D_mu = clear_denominators(T_dict, mus)
    tau_int = sum(mus_int[0])
    m = sz_multidegree(dims, perm=multidegree_perm)
    N = multinomial_degree(dims)
    d = sum(n - 1 for n in dims)

    M0, K, J = sylvester_blocks(dims, T_int, mus_int, m=m, integer_scaled=True)

    use_flint = False
    if backend == "flint":
        use_flint = True
    elif backend != "auto" and backend != "python":
        raise ValueError(f"Unknown backend {backend!r}; choose 'auto', 'python', or 'flint'.")

    if use_flint:
        vals = _eval_pencil_flint(M0, K, J)
    else:
        vals = _eval_pencil_crt(M0, K, J)

    R_int_raw = _newton_interpolate_int(vals)  # polynomial in x_int = D_mu * x

    # Map back to x = x_int / D_mu and exact det Phi(x):
    # Phi_int(D_mu * x) = (D_T * tau_int) * Phi(x)
    scale_entry = D_T * tau_int
    denom_det = Fraction(scale_entry) ** ((d + 1) * N)
    R_exact = [Fraction(c * (D_mu**j), 1) / denom_det for j, c in enumerate(R_int_raw)]
    R_int_x = [c * (D_mu**j) for j, c in enumerate(R_int_raw)]

    prim = _primitive_int_poly(R_exact)

    # Exact Theorem 4.6 decomposition using original (Fraction or int) T and mus
    T_orig = {c: (v if isinstance(v, int) and not isinstance(v, bool) else Fraction(str(v)) if isinstance(v, float) else Fraction(v)) for c, v in T_dict.items()}
    mus_orig = [[(v if isinstance(v, int) and not isinstance(v, bool) else Fraction(str(v)) if isinstance(v, float) else Fraction(v)) for v in mk] for mk in mus]
    th46 = theorem46_factors(dims, T_orig, mus_orig)

    E_tilde: list[Any] | None = None
    if th46.get("supported") and th46["content"] != 0 and any(R_exact):
        content = th46["content"]
        E1 = [c / content for c in R_exact]
        lead_pred = th46["lead_pred"]
        const_pred = th46["const_pred"]
        if lead_pred != 0 and len(E1) > N and E1[N] != 0:
            ratio = Fraction(E1[N]) / Fraction(lead_pred)
            E_tilde_q = [Fraction(e) / ratio for e in E1[: N + 1]]
        elif const_pred != 0 and E1[0] != 0:
            ratio = Fraction(E1[0]) / Fraction(const_pred)
            E_tilde_q = [Fraction(e) / ratio for e in _trim_poly(E1)]
        else:
            E_tilde_q = None
        if E_tilde_q is not None:
            E_tilde_q = _trim_poly(E_tilde_q)
            if all(c.denominator == 1 for c in E_tilde_q):
                E_tilde = [int(c) for c in E_tilde_q]
            else:
                E_tilde = E_tilde_q

    return EliminantResult(
        dims=dims,
        entry=target_entry,
        expected_degree=N,
        sylvester_size=(d + 1) * N,
        multidegree=m,
        coeffs=prim,
        R_coeffs=_trim_poly(R_exact),
        R_int_coeffs=_trim_poly(R_int_x),
        E_tilde_coeffs=E_tilde,
        theorem46=th46,
    )


def restricted_slab_resultant(
    T: TensorInput,
    margins: Sequence[Sequence[Any]],
    mode: int,
    *,
    integer_scaled: bool = False,
) -> int | Fraction:
    """Compute Res_{Sigma_k}(l_1|_{Sigma_k}, ..., l_d|_{Sigma_k}) from Proposition 4.5.

    Restricts the d margin forms to the coordinate hyperplane {u^(mode)_1 = 0},
    i.e., to the slab T^(mode) = {t_c : c_{mode} >= 2} of format
    n_1 x ... x (n_{mode} - 1) x ... x n_r.

    If integer_scaled=False (default), returns the resultant of the normalized
    forms l_1, ..., l_d over Q, satisfying |R(0)| = t_1^N * prod_k |R_k|.
    If integer_scaled=True, uses the tau-scaled integer forms tau * l_s.
    """
    from .discriminants import _det_exact

    dims, T_dict = normalize_tensor(T)
    mus, _ = normalize_margins(dims, margins)
    T_int, mus_int, D_T, D_mu = clear_denominators(T_dict, mus)
    tot = sum(mus_int[0])
    if not (0 <= mode < len(dims)):
        raise ValueError(f"Mode {mode} out of range for shape {dims}.")
    if dims[mode] < 2:
        raise ValueError("Cannot restrict a mode of size < 2.")

    sub = list(dims)
    sub[mode] -= 1
    sub_t = tuple(sub)
    cells = cells_of(dims)

    if all(d == 1 for d in sub_t):
        raise ValueError("Slab cannot have all dimensions equal to 1 when rank >= 2.")

    forms = []
    for kk, d in enumerate(dims):
        for i in range(2, d + 1):
            form: dict[tuple[int, ...], int] = {}
            for c in cells:
                if c[mode] == 1:
                    continue
                c2 = list(c)
                c2[mode] -= 1
                form[tuple(c2)] = (tot * (1 if c[kk] == i else 0) - mus_int[kk][i - 1]) * T_int[c]
            forms.append(form)

    m = sz_multidegree(sub_t)
    src = monomials(sub_t, [mi - 1 for mi in m])
    dst = monomials(sub_t, m)
    didx = {g: i for i, g in enumerate(dst)}
    rows = len(dst)
    N_k = len(src)
    M = [[0] * (len(forms) * N_k) for _ in range(rows)]
    col = 0
    for form in forms:
        for g in src:
            for c, coef in form.items():
                if coef:
                    h = tuple(
                        tuple(g[k][t] + (1 if c[k] == t + 1 else 0) for t in range(sub_t[k]))
                        for k in range(len(sub_t))
                    )
                    M[didx[h]][col] += coef
            col += 1
    det_int = _det_exact(M)
    if integer_scaled:
        if D_T == 1 and D_mu == 1:
            return det_int
        return Fraction(det_int, (D_T * D_mu) ** len(dst))
    scale = Fraction(D_T * tot) ** len(dst)
    return Fraction(det_int, 1) / scale
