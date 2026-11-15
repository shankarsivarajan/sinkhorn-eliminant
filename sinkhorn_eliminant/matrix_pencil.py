"""Section 5: Matrix diagonal pencil form via Plucker coordinates.

Implements Theorem 5.3 (Fang's diagonal form via Plucker derivations) of:
    S. Sivarajan, "An explicit determinantal formula for Sinkhorn limits of
    matrices and tensors" (HexagonMath 2611.00433).

For an m x n matrix T with row margins rho = mu^(1) and column margins kappa = mu^(2):
    F(x) = det( diag(Gamma_hat) + x * diag(Delta_hat) * Q ),
where:
    - link L = { m-subsets J of {1..m, 1'..n'} : 1' in J, 1 not in J } (|L| = N = C(m+n-2, m-1)),
    - Delta_hat_J = omega_J(T),  Gamma_hat_J = s_J * t_{11} * omega_{J*}(T),
    - Q depends only on the margins (rho, kappa), with eigenvalues -kappa_1 and -rho_1.
Up to a unit, F(x) = t_{11} * det(Q) * E_tilde_T(x).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from fractions import Fraction
from math import comb, gcd, isqrt, lcm
from typing import Any, Sequence

from .discriminants import _det_exact
from .resultant import _is_prime_64, _newton_interpolate_int, _primitive_int_poly, _trim_poly, format_polynomial
from .sylvester import (
    TensorInput,
    clear_denominators,
    normalize_margins,
    normalize_tensor,
    permute_to_corner,
)


def _plucker_minor(T_mat: list[list[Any]], m: int, J: tuple[int, ...]) -> Any:
    """Maximal minor omega_J(T) of [I_m | T] on 0-based columns J."""
    R_kept = [j for j in J if j < m]
    I_rows = [i for i in range(m) if i not in R_kept]
    K_cols = [j - m for j in J if j >= m]
    inv_cnt = sum(1 for r in R_kept for i in I_rows if i < r)
    sign = -1 if inv_cnt % 2 else 1
    sub = [[T_mat[i][k] for k in K_cols] for i in I_rows]
    return sign * _det_exact(sub)


@dataclass
class MatrixPencil:
    """N x N diagonal pencil diag(gamma_hat) + x * diag(delta_hat) @ Q for an m x n matrix."""

    m: int
    n: int
    entry: tuple[int, int]
    link: list[tuple[int, ...]]
    Q: list[list[Any]]
    delta_hat: list[Any]
    gamma_hat: list[Any]
    T_mat: list[list[Any]]
    rho: list[Any]
    kappa: list[Any]

    @property
    def degree(self) -> int:
        return len(self.link)


@dataclass
class MatrixDiagonalResult:
    """Exact polynomial from Theorem 5.3's diagonal matrix pencil F(x)."""

    m: int
    n: int
    entry: tuple[int, int]
    expected_degree: int
    coeffs: list[int]  # Primitive integer polynomial in Z[x], low-to-high
    F_coeffs: list[Any]  # Exact coefficients of F(x), low-to-high
    det_Q: Any

    @property
    def degree(self) -> int:
        return len(self.coeffs) - 1 if any(self.coeffs) else -1

    @property
    def coeffs_desc(self) -> list[int]:
        return list(reversed(self.coeffs))

    def format(self, which: str = "primitive", var: str = "x") -> str:
        seq = self.coeffs if which == "primitive" else self.F_coeffs
        return format_polynomial(seq, var=var)


def build_matrix_pencil(
    T: TensorInput,
    margins: Sequence[Sequence[Any]],
    *,
    entry: tuple[int, int] | None = None,
    exact: bool = True,
) -> MatrixPencil:
    """Construct the N x N matrix diagonal pencil of Theorem 5.3."""
    dims, T_dict = normalize_tensor(T)
    if len(dims) != 2:
        raise ValueError(f"Matrix diagonal pencil applies only to 2D matrices, got shape {dims}.")
    mus, _ = normalize_margins(dims, margins)
    target_entry = (0, 0) if entry is None else (int(entry[0]), int(entry[1]))
    if entry is not None:
        T_dict, mus, _ = permute_to_corner(dims, T_dict, mus, target_entry)

    m, n = dims
    if exact:
        T_use = {c: (v if isinstance(v, int) and not isinstance(v, bool) else Fraction(str(v)) if isinstance(v, float) else Fraction(v)) for c, v in T_dict.items()}
        rho = [(v if isinstance(v, int) and not isinstance(v, bool) else Fraction(str(v)) if isinstance(v, float) else Fraction(v)) for v in mus[0]]
        kappa = [(v if isinstance(v, int) and not isinstance(v, bool) else Fraction(str(v)) if isinstance(v, float) else Fraction(v)) for v in mus[1]]
        one: Any = 1
        zero: Any = 0
    else:
        T_use = {c: float(v) for c, v in T_dict.items()}
        rho = [float(v) for v in mus[0]]
        kappa = [float(v) for v in mus[1]]
        one = 1.0
        zero = 0.0

    T_mat = [[T_use[(i + 1, j + 1)] for j in range(n)] for i in range(m)]

    c1 = m
    alpha = [one] * m + list(kappa)
    beta = [-r for r in rho] + [one] * n
    E_rows = list(range(1, m))
    E_cols = list(range(m + 1, m + n))
    E = set(E_rows + E_cols)

    link: list[tuple[int, ...]] = []
    for k in range(min(m, n)):
        for R in itertools.combinations(E_rows, k):
            for C in itertools.combinations(E_cols, k):
                rows = [i for i in E_rows if i not in R]
                link.append(tuple(sorted(rows + [c1] + list(C))))
    pos = {J: idx for idx, J in enumerate(link)}
    N = len(link)

    Q = [[zero] * N for _ in range(N)]
    for i, J in enumerate(link):
        Js = set(J)
        Q[i][i] = sum((kappa[c - m] for c in J if c > m), zero) - sum((rho[r] for r in J if r < m), zero) - rho[0]
        for b in J:
            if b == c1:
                continue
            for a in E - Js:
                Jp = tuple(sorted((Js - {b}) | {a}))
                lo, hi = (a, b) if a < b else (b, a)
                between = sum(1 for z in Js if z != b and lo < z < hi)
                sign = -1 if between % 2 else 1
                Q[i][pos[Jp]] = sign * alpha[b] * beta[a]

    delta_hat = [_plucker_minor(T_mat, m, J) for J in link]
    gamma_hat = []
    t11 = T_mat[0][0]
    for J in link:
        n_rows_in_J = sum(1 for z in J if z < m)
        s_J = (-1 if n_rows_in_J % 2 else 1) * rho[0] * kappa[0]
        J_star = tuple(sorted((set(J) - {c1}) | {0}))
        gamma_hat.append(s_J * t11 * _plucker_minor(T_mat, m, J_star))

    return MatrixPencil(
        m=m,
        n=n,
        entry=target_entry,
        link=link,
        Q=Q,
        delta_hat=delta_hat,
        gamma_hat=gamma_hat,
        T_mat=T_mat,
        rho=rho,
        kappa=kappa,
    )


def _eval_diag_pencil_crt(
    A_int: list[list[int]],
    B_int: list[list[int]],
) -> list[int]:
    """Compute det(A_int + x * B_int) at x = 0..N via multimodular CRT."""
    N = len(A_int)
    Hmax = 1
    for xv in range(N + 1):
        Hx = 1
        for c in range(N):
            s2 = sum((A_int[r][c] + xv * B_int[r][c]) ** 2 for r in range(N))
            Hx *= isqrt(s2) + 1
        if Hx > Hmax:
            Hmax = Hx
    bound = 2 * Hmax

    Mprod = 1
    acc = [0] * (N + 1)
    p = (1 << 61) - 1
    while Mprod <= bound:
        while not _is_prime_64(p):
            p -= 2
        Am = [[v % p for v in row] for row in A_int]
        Bm = [[v % p for v in row] for row in B_int]
        vs = []
        for xv in range(N + 1):
            M = [[(Am[r][c] + xv * Bm[r][c]) % p for c in range(N)] for r in range(N)]
            det = 1
            for i in range(N):
                piv = -1
                for r in range(i, N):
                    if M[r][i] != 0:
                        piv = r
                        break
                if piv == -1:
                    det = 0
                    break
                if piv != i:
                    M[i], M[piv] = M[piv], M[i]
                    det = (p - det) % p
                pv = M[i][i]
                det = (det * pv) % p
                inv = pow(pv, -1, p)
                prow = M[i]
                for r in range(i + 1, N):
                    mr = M[r]
                    f = mr[i]
                    if f:
                        f = (f * inv) % p
                        for c in range(i + 1, N):
                            mr[c] = (mr[c] - f * prow[c]) % p
            vs.append(det)

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


def matrix_diagonal_polynomial(
    T: TensorInput,
    margins: Sequence[Sequence[Any]],
    *,
    entry: tuple[int, int] | None = None,
) -> MatrixDiagonalResult:
    """Compute exact F(x) = det(diag(Gamma_hat) + x diag(Delta_hat) Q) and primitive E(x)."""
    dims, T_dict = normalize_tensor(T)
    if len(dims) != 2:
        raise ValueError(f"Matrix diagonal pencil applies only to 2D matrices, got shape {dims}.")
    mus, _ = normalize_margins(dims, margins)
    target_entry = (0, 0) if entry is None else (int(entry[0]), int(entry[1]))
    if entry is not None:
        T_dict, mus, _ = permute_to_corner(dims, T_dict, mus, target_entry)

    T_int, mus_int, D_T, D_mu = clear_denominators(T_dict, mus)
    m, n = dims
    pencil = build_matrix_pencil(
        [[T_int[(i + 1, j + 1)] for j in range(n)] for i in range(m)],
        mus_int,
        exact=True,
    )
    N = pencil.degree

    A_int = [[0] * N for _ in range(N)]
    B_int = [[0] * N for _ in range(N)]
    for i in range(N):
        A_int[i][i] = int(pencil.gamma_hat[i])
        di = int(pencil.delta_hat[i])
        for j in range(N):
            B_int[i][j] = di * int(pencil.Q[i][j])

    vals = _eval_diag_pencil_crt(A_int, B_int)
    F_int_raw = _newton_interpolate_int(vals)  # polynomial in x_int = D_mu * x

    # Map back to original (T, mus) scaling:
    # gamma_hat scales as D_mu^2 * D_T^{size(J*)}, delta_hat as D_T^{size(J)}, Q as D_mu.
    # For each row i of diag(gamma_hat) + x diag(delta_hat) Q:
    # wait: size of minor in delta_hat_J vs gamma_hat_J is NOT uniform across J,
    # Wait: let's compute F(x) directly using the rational scaling per row!
    # Actually: let r_i = size(I_i) + 1 (since minor in delta_hat_J has size k+1, minor in gamma_hat_J has size k, and t11 adds 1!).
    # So EVERY entry in row i of diag(gamma_hat) + x diag(delta_hat) Q scales by D_T^{k_i + 1} * D_mu^2 (with x_int = D_mu x)!
    # Checking total exponent of D_T across rows: sum_i (k_i + 1).
    row_k = [sum(1 for r in range(1, m) if r not in J) for J in pencil.link]
    exp_D_T = sum(k + 1 for k in row_k)
    denom = Fraction(D_T) ** exp_D_T * Fraction(D_mu) ** (2 * N)

    F_exact_q = [Fraction(c * (D_mu**j), 1) / denom for j, c in enumerate(F_int_raw)]
    if all(c.denominator == 1 for c in F_exact_q):
        F_exact: list[Any] = [int(c) for c in F_exact_q]
    else:
        F_exact = F_exact_q

    prim = _primitive_int_poly(F_exact_q)

    # det(Q) from Proposition 5.4: +- kappa_1^C(m+n-3, m-2) * rho_1^C(m+n-3, m-1)
    # Compute exact det(Q) using actual Q if needed, or Proposition 5.4 formula
    e_k = comb(m + n - 3, m - 2) if m >= 2 else 0
    e_r = comb(m + n - 3, m - 1)
    rho1 = Fraction(mus_int[0][0], D_mu)
    kap1 = Fraction(mus_int[1][0], D_mu)
    det_Q_mag = (kap1**e_k) * (rho1**e_r)

    return MatrixDiagonalResult(
        m=m,
        n=n,
        entry=target_entry,
        expected_degree=N,
        coeffs=prim,
        F_coeffs=_trim_poly(F_exact),
        det_Q=int(det_Q_mag) if det_Q_mag.denominator == 1 else det_Q_mag,
    )
