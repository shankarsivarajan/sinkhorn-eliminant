"""Face A-discriminants, content, and Theorem 4.6 normalization.

Implements Section 4 of S. Sivarajan, "An explicit determinantal formula for
Sinkhorn limits of matrices and tensors" (HexagonMath 2611.00433).

For a tensor T of format n = (n_1, ..., n_r):
- Faces are Cartesian products F = F_1 x ... x F_r with nonempty F_k subset [1..n_k].
- Disc_F(T) is the A-discriminant of the subtensor T|_F (t_c for a cell, det for a
  square matrix face, Cayley's hyperdeterminant for 2x2x2, boundary resultant for
  2x2x3, Schlafli binary-form discriminant for 2xnxn).
- Theorem 4.6:
      R(x) = c(T) * E_tilde(x)   (up to a unit q * tau^e),
  where c(T) = t_1 * prod_{F not containing 1} Disc_F(T).
"""

from __future__ import annotations

import itertools
from fractions import Fraction
from typing import Any

from .sylvester import multinomial_degree


def _det_exact(M: list[list[Any]]) -> Any:
    """Exact determinant over Z or Q via Bareiss / fraction-free or Fraction elimination."""
    n = len(M)
    if n == 0:
        return 1
    if n == 1:
        return M[0][0]
    if n == 2:
        return M[0][0] * M[1][1] - M[0][1] * M[1][0]

    use_frac = any(isinstance(v, Fraction) for row in M for v in row)
    if not use_frac:
        # Bareiss integer determinant
        A = [[int(v) for v in row] for row in M]
        sign = 1
        prev = 1
        for i in range(n - 1):
            if A[i][i] == 0:
                swap = -1
                for r in range(i + 1, n):
                    if A[r][i] != 0:
                        swap = r
                        break
                if swap == -1:
                    return 0
                A[i], A[swap] = A[swap], A[i]
                sign = -sign
            piv = A[i][i]
            for r in range(i + 1, n):
                for c in range(i + 1, n):
                    A[r][c] = (A[r][c] * piv - A[r][i] * A[i][c]) // prev
            prev = piv
        return sign * A[n - 1][n - 1]

    A_q = [[Fraction(v) for v in row] for row in M]
    det = Fraction(1)
    for i in range(n):
        piv = -1
        for r in range(i, n):
            if A_q[r][i] != 0:
                piv = r
                break
        if piv == -1:
            return Fraction(0)
        if piv != i:
            A_q[i], A_q[piv] = A_q[piv], A_q[i]
            det = -det
        pv = A_q[i][i]
        det *= pv
        for r in range(i + 1, n):
            if A_q[r][i] != 0:
                f = A_q[r][i] / pv
                for c in range(i + 1, n):
                    A_q[r][c] -= f * A_q[i][c]
    return det


def cayley_hyperdeterminant(a: Any) -> Any:
    """Cayley's 2x2x2 hyperdeterminant for a 2x2x2 nested sequence."""
    a000, a001 = a[0][0][0], a[0][0][1]
    a010, a011 = a[0][1][0], a[0][1][1]
    a100, a101 = a[1][0][0], a[1][0][1]
    a110, a111 = a[1][1][0], a[1][1][1]
    return (
        a000**2 * a111**2
        + a001**2 * a110**2
        + a010**2 * a101**2
        + a100**2 * a011**2
        - 2
        * (
            a000 * a001 * a110 * a111
            + a000 * a010 * a101 * a111
            + a000 * a100 * a011 * a111
            + a001 * a010 * a101 * a110
            + a001 * a100 * a011 * a110
            + a010 * a100 * a011 * a101
        )
        + 4 * (a000 * a011 * a101 * a110 + a001 * a010 * a100 * a111)
    )


def _poly_discriminant(coeffs_desc: list[Any]) -> Any:
    """Exact discriminant of univariate polynomial a_n x^n + ... + a_0 (coeffs_desc)."""
    while len(coeffs_desc) > 1 and coeffs_desc[0] == 0:
        coeffs_desc = coeffs_desc[1:]
    n = len(coeffs_desc) - 1
    if n <= 0:
        return 0
    if n == 1:
        return 1
    if n == 2:
        a, b, c = coeffs_desc
        return b * b - 4 * a * c
    # Disc(f) = (-1)^{n(n-1)/2} / a_n * Res(f, f')
    deriv = [(n - i) * coeffs_desc[i] for i in range(n)]
    size = 2 * n - 1
    zero = Fraction(0) if any(isinstance(v, Fraction) for v in coeffs_desc) else 0
    S = [[zero] * size for _ in range(size)]
    for r in range(n - 1):
        for j, v in enumerate(coeffs_desc):
            S[r][r + j] = v
    for r in range(n):
        for j, v in enumerate(deriv):
            S[n - 1 + r][r + j] = v
    res = _det_exact(S)
    sign = -1 if ((n * (n - 1)) // 2) % 2 else 1
    a_n = coeffs_desc[0]
    if isinstance(res, int) and isinstance(a_n, int) and res % a_n == 0:
        return sign * (res // a_n)
    return sign * Fraction(res, a_n)


def _schlafli_2xnxn(A0: list[list[Any]], A1: list[list[Any]]) -> Any:
    """Discriminant of the homogeneous binary form det(X A0 + Y A1) for 2 x n x n."""
    n = len(A0)
    # Interpolate g(x) = det(x A0 + A1) at x = 0..n
    xs = list(range(n + 1))
    vals = []
    for xv in xs:
        M = [[xv * A0[i][j] + A1[i][j] for j in range(n)] for i in range(n)]
        vals.append(_det_exact(M))
    # Lagrange interpolation over Q -> low to high
    coeffs = [Fraction(0)] * (n + 1)
    for i, xi in enumerate(xs):
        basis = [Fraction(1)]
        den = 1
        for j, xj in enumerate(xs):
            if j != i:
                basis = [Fraction(0)] + basis
                for t in range(len(basis) - 1):
                    basis[t] -= xj * basis[t + 1]
                den *= xi - xj
        for t in range(n + 1):
            coeffs[t] += Fraction(vals[i]) * basis[t] / den
    if all(c.denominator == 1 for c in coeffs) and not any(
        isinstance(v, Fraction) for row in A0 + A1 for v in row
    ):
        coeffs_desc: list[Any] = [int(c) for c in reversed(coeffs)]
    else:
        coeffs_desc = list(reversed(coeffs))
    if coeffs_desc[0] == 0:
        return 0
    return _poly_discriminant(coeffs_desc)


def faces_of(dims: tuple[int, ...]) -> list[tuple[tuple[int, ...], ...]]:
    """All nonempty faces F = F_1 x ... x F_r of the index box [1..n_1] x ... x [1..n_r]."""
    per_mode = [
        [tuple(S) for r_len in range(1, d + 1) for S in itertools.combinations(range(1, d + 1), r_len)]
        for d in dims
    ]
    return list(itertools.product(*per_mode))


def face_discriminant(
    T_dict: dict[tuple[int, ...], Any],
    face: tuple[tuple[int, ...], ...],
) -> tuple[Any | None, str | None]:
    """Compute Disc_F(T) and a short description for a face F.

    Returns (None, None) if the face discriminant is trivial (1D slice or
    non-hyperdeterminant rectangular matrix) and raises NotImplementedError
    for higher-order tensor formats whose hyperdeterminant is not implemented.
    """
    keep = [k for k, S in enumerate(face) if len(S) > 1]
    if not keep:
        c = tuple(S[0] for S in face)
        return T_dict[c], "entry"
    if len(keep) == 1:
        return None, None

    # Sort active modes by size so e.g. 3x2x2 becomes 2x2x3
    keep.sort(key=lambda k: len(face[k]))
    shape = tuple(len(face[k]) for k in keep)

    def get_sub(idx: tuple[int, ...]) -> Any:
        c = [S[0] for S in face]
        for k, pos in zip(keep, idx):
            c[k] = face[k][pos]
        return T_dict[tuple(c)]

    if len(shape) == 2:
        if shape[0] != shape[1]:
            return None, None
        n = shape[0]
        M = [[get_sub((i, j)) for j in range(n)] for i in range(n)]
        return _det_exact(M), f"{n}x{n} minor"

    if len(shape) == 3:
        if shape == (2, 2, 2):
            a = [[[get_sub((i, j, k)) for k in range(2)] for j in range(2)] for i in range(2)]
            return cayley_hyperdeterminant(a), "Cayley 2x2x2 hyperdeterminant"
        if shape == (2, 2, 3):
            # 3 x 4 flattening: 4 signed 3x3 minors v_0..v_3; Disc = v_0*v_3 - v_1*v_2
            M3x4 = [[get_sub((i, j, k)) for i in range(2) for j in range(2)] for k in range(3)]
            v = []
            for drop in range(4):
                sub3 = [[M3x4[r][c] for c in range(4) if c != drop] for r in range(3)]
                v.append((-1) ** drop * _det_exact(sub3))
            return v[0] * v[3] - v[1] * v[2], "2x2x3 boundary discriminant"
        if shape[0] == 2 and shape[1] == shape[2]:
            n = shape[1]
            A0 = [[get_sub((0, i, j)) for j in range(n)] for i in range(n)]
            A1 = [[get_sub((1, i, j)) for j in range(n)] for i in range(n)]
            return _schlafli_2xnxn(A0, A1), f"2x{n}x{n} Schlafli discriminant"
        # Rectangular 3D faces with no nontrivial A-discriminant (boundary defect >= 2)
        if shape[2] > shape[0] * shape[1] - shape[0] - shape[1] + 2:
            return None, None

    raise NotImplementedError(f"Face discriminant not implemented for subtensor shape {shape}.")


def theorem46_factors(
    dims: tuple[int, ...],
    T_dict: dict[tuple[int, ...], Any],
    mus: list[list[Any]],
) -> dict[str, Any]:
    """Compute Theorem 4.6 predictions for content, leading coeff, and constant term.

    Returns a dict with keys:
        supported: bool (False if some face shape is unimplemented)
        content: c(T) = t_1 * prod_{F not containing 1} Disc_F(T)
        lead_pred: prod_{F containing 1, F != {1}} Disc_F(T)
        const_pred: t_1^{N-1} prod_k (mu^(k)_1)^{N_k} prod_{F not containing 1} Disc_F(T)^{e_F - 1}
        faces: list of per-face metadata dicts
    """
    corner = (1,) * len(dims)
    N = multinomial_degree(dims)
    content = T_dict[corner]
    lead_pred: Any = 1
    face_records = []

    try:
        for F in faces_of(dims):
            D, kind = face_discriminant(T_dict, F)
            if D is None:
                continue
            has_corner = all(1 in S for S in F)
            face_records.append(
                {
                    "face": [list(S) for S in F],
                    "kind": kind,
                    "disc": D,
                    "contains_corner": has_corner,
                }
            )
            if not has_corner:
                content *= D
            elif kind != "entry":
                lead_pred *= D
    except NotImplementedError as exc:
        return {"supported": False, "reason": str(exc), "faces": face_records}

    const_pred = T_dict[corner] ** (N - 1)
    for k in range(len(dims)):
        sub = list(dims)
        sub[k] -= 1
        Nk = 1 if all(s == 1 for s in sub) else multinomial_degree(tuple(sub))
        const_pred *= mus[k][0] ** Nk
    for rec in face_records:
        if not rec["contains_corner"]:
            eF = sum(1 for S in rec["face"] if 1 not in S)
            if eF > 1:
                const_pred *= rec["disc"] ** (eF - 1)

    return {
        "supported": True,
        "content": content,
        "lead_pred": lead_pred,
        "const_pred": const_pred,
        "faces": face_records,
    }
