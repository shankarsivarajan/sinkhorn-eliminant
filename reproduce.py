"""Companion script to reproduce worked examples and mathematical identities from:

    Shankar Sivarajan, "An explicit determinantal formula for Sinkhorn limits
    of matrices and tensors", HexagonMath 2611.00433 (2026).

Run from the repository root:
    python reproduce.py                 # full companion report + identity checks
    python reproduce.py --examples-only # Appendix A, RW Ex. 13, and Kruithof 1937
    python -m unittest -v reproduce.py  # standard unittest runner
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import subprocess
import sys
import unittest
from fractions import Fraction
from math import isclose, prod
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sinkhorn_eliminant import (
    eliminant_polynomial,
    find_polynomial_roots,
    matrix_diagonal_polynomial,
    multinomial_degree,
    restricted_slab_resultant,
    sinkhorn_margins,
    sinkhorn_scale,
    solve_scaling,
)
from sinkhorn_eliminant.discriminants import _det_exact
from sinkhorn_eliminant.matrix_pencil import build_matrix_pencil


def _fmt_signs(signs):
    return " ".join("(" + ",".join("+" if v > 0 else "-" for v in vec) + ")" for vec in signs)


def _print_example_summary(title: str, res, *, diag=None, full_polys: bool = False) -> None:
    e = res.eliminant
    print("=" * 78)
    print(title)
    print(
        f"Format: {'x'.join(map(str, e.dims))}, expected degree N = {e.expected_degree}, "
        f"actual degree = {e.degree}, Sylvester size = {e.sylvester_size}"
    )
    if full_polys or e.degree <= 12:
        print(f"Primitive E(x) = {e.format('primitive')}")
        if e.E_tilde_coeffs is not None:
            print(f"Theorem 4.6 E_tilde(x) = {e.format('E_tilde')}")
    else:
        max_digits = len(str(max(abs(c) for c in e.coeffs))) - 1
        print(f"Primitive E(x): degree {e.degree}, max |coeff| ~ 10^{max_digits} (pass --full-polys for all coefficients)")
    if diag is not None:
        print(f"Matrix diagonal pencil F(x) primitive == E(x): {diag.coeffs == e.coeffs}")
    print(f"Limit P[{e.entry}] = {res.limit_root:.10f}  (max |P* - Sinkhorn| = {res.sinkhorn_diff:.2e})")
    for s in res.solutions:
        if s.real:
            tag = "  <-- LIMIT" if s.positive else ""
            print(
                f"  root {s.root.real:16.10g}  feas={str(s.feasible):5}  "
                f"pos_nv={str(s.nullvec_positive):5}  signs={_fmt_signs(s.signs)}  "
                f"res={s.margin_residual:.1e}{tag}"
            )
        else:
            print(f"  root {s.root.real:14.10g} {s.root.imag:+14.10g}i")


# ---------------------------------------------------------------------------
# Worked examples from Appendix A and classical literature
# ---------------------------------------------------------------------------

def run_appendix_a1_rw1(*, verbose: bool = True, full_polys: bool = False) -> None:
    T = [[3, 9, 1], [3, 2, 9], [5, 3, 4]]
    mus = [[1, 1, 1], [1, 1, 1]]
    res = solve_scaling(T, mus, recover_all=True, check_sinkhorn=True)
    diag = matrix_diagonal_polynomial(T, mus)
    e = res.eliminant

    # Paper eq. (A.1) and Rowland-Wu Example 1, eq. (2)
    expected_e_tilde = [-997272, 1162026, 17082657, -10189476, -68393079, -17851428, 30354912]
    expected_prim = [-12312, 14346, 210897, -125796, -844359, -220388, 374752]
    assert e.E_tilde_coeffs == expected_e_tilde
    assert e.coeffs == expected_prim
    assert diag.coeffs == e.coeffs
    assert e.degree == 6
    assert isclose(res.limit_root, 0.2766771162, abs_tol=1e-9)
    assert res.sinkhorn_diff < 1e-11
    assert sum(1 for s in res.solutions if s.positive) == 1
    assert sum(1 for s in res.solutions if s.nullvec_positive) == 1
    if verbose:
        _print_example_summary("Appendix A.1: 3x3 matrix (Rowland-Wu Example 1)", res, diag=diag, full_polys=full_polys)


def run_appendix_a2_2x2x3(*, verbose: bool = True, full_polys: bool = False) -> None:
    T = [
        [[4, 3, 8], [1, 6, 4]],
        [[7, 9, 6], [9, 5, 4]],
    ]
    mus = [[9, 15], [7, 17], [7, 7, 10]]
    res = solve_scaling(T, mus, recover_all=True, check_sinkhorn=True)
    e = res.eliminant

    assert e.expected_degree == 12
    assert e.degree == 12
    assert e.E_tilde_coeffs is not None
    # Constant and leading terms match Theorem 4.6 face-discriminant formulas
    assert e.E_tilde_coeffs[0] == e.theorem46["const_pred"]
    assert e.E_tilde_coeffs[-1] == e.theorem46["lead_pred"]
    assert isclose(res.limit_root, 0.5715181134, abs_tol=1e-9)
    assert sum(1 for s in res.solutions if s.real) == 8
    assert sum(1 for s in res.solutions if s.feasible) == 5
    assert sum(1 for s in res.solutions if s.positive) == 1
    assert res.sinkhorn_diff < 1e-9
    if verbose:
        _print_example_summary("Appendix A.2: 2x2x3 tensor, general margins (N = 12)", res, full_polys=full_polys)


def run_appendix_a3_4x4(*, verbose: bool = True, full_polys: bool = False) -> None:
    T = [
        [8, 2, 9, 7],
        [9, 1, 4, 2],
        [3, 7, 4, 5],
        [2, 6, 9, 9],
    ]
    mus = [[2, 12, 4, 6], [1, 15, 2, 6]]
    res = solve_scaling(T, mus, recover_all=True, check_sinkhorn=True)
    diag = matrix_diagonal_polynomial(T, mus)
    e = res.eliminant

    assert e.expected_degree == 20
    assert e.degree == 20
    assert diag.coeffs == e.coeffs
    assert isclose(res.limit_root, 0.05702300275, abs_tol=1e-9)
    assert sum(1 for s in res.solutions if s.real) == 18
    assert sum(1 for s in res.solutions if s.feasible) == 7
    assert sum(1 for s in res.solutions if s.positive) == 1
    assert res.sinkhorn_diff < 1e-10
    if verbose:
        _print_example_summary("Appendix A.3: 4x4 matrix, general margins (N = 20)", res, diag=diag, full_polys=full_polys)


def run_appendix_a4_degenerate_2x2x2(*, verbose: bool = True, full_polys: bool = False) -> None:
    T = [
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
    ]
    mus = [[1, 3], [2, 2], [3, 1]]
    res = solve_scaling(T, mus, recover_all=True, check_sinkhorn=True)
    e = res.eliminant

    assert e.is_degenerate
    assert e.expected_degree == 6
    assert e.degree == 5
    assert e.E_tilde_coeffs == [387072, -1085952, -1686368, -780288, -142272, -8192]
    assert e.coeffs == [-12096, 33936, 52699, 24384, 4446, 256]
    assert isclose(res.limit_root, 0.2487655058, abs_tol=1e-9)
    assert sum(1 for s in res.solutions if s.positive) == 1
    assert res.sinkhorn_diff < 1e-12
    if verbose:
        _print_example_summary("Appendix A.4: Degenerate 2x2x2 tensor (degree drops 6 -> 5)", res, full_polys=full_polys)


def run_rw_example_13(*, verbose: bool = True, full_polys: bool = False) -> None:
    T13 = [
        [3, 1, 2, 2],
        [2, 2, 2, 1],
        [1, 2, 3, 2],
        [1, 4, 2, 3],
    ]
    mus13 = [[1, 1, 1, 1], [1, 1, 1, 1]]
    res = solve_scaling(T13, mus13, recover_all=True, check_sinkhorn=True)
    diag = matrix_diagonal_polynomial(T13, mus13)
    e = res.eliminant

    # Note: Rowland-Wu misprinted the x^20 coeff as 382625520076800;
    # the exact value is 488910386764800 = 3 * E_tilde[-1].
    assert 3 * e.E_tilde_coeffs[-1] == 488910386764800
    assert 3 * e.E_tilde_coeffs[0] == -246790694704250880
    assert diag.coeffs == e.coeffs
    if verbose:
        _print_example_summary("Rowland-Wu Example 13: 4x4 doubly stochastic (N = 20)", res, diag=diag, full_polys=full_polys)


def run_kruithof_1937_rw32(*, verbose: bool = True, full_polys: bool = False) -> None:
    # Kruithof (1937) 4x4 telephone traffic matrix / Rowland-Wu Example 32
    T32 = [
        [2000, 1030, 650, 320],
        [1080, 1110, 555, 255],
        [720, 580, 500, 200],
        [350, 280, 210, 160],
    ]
    mus32 = [[6000, 4000, 2500, 1000], [6225, 4000, 2340, 935]]
    res32 = solve_scaling(T32, mus32, recover_all=True, check_sinkhorn=True)
    diag = matrix_diagonal_polynomial(T32, mus32)

    assert len(res32.solutions) == 20
    assert sum(1 for s in res32.solutions if s.real) == 10
    assert sum(1 for s in res32.solutions if s.positive) == 1
    assert diag.coeffs == res32.eliminant.coeffs
    cplx32 = sorted((s.root.real, s.root.imag) for s in res32.solutions if not s.real)
    assert len(cplx32) == 10
    for i in range(0, 10, 2):
        assert isclose(cplx32[i][0], cplx32[i + 1][0], abs_tol=1e-6)
        assert isclose(cplx32[i][1], -cplx32[i + 1][1], abs_tol=1e-6)
    assert isclose(res32.limit_root, 3246.38700234, abs_tol=1e-7)
    assert res32.sinkhorn_diff < 1e-8
    if verbose:
        _print_example_summary(
            "Kruithof (1937) / Rowland-Wu Example 32: 4x4 traffic matrix (N = 20)",
            res32,
            diag=diag,
            full_polys=full_polys,
        )


# ---------------------------------------------------------------------------
# Paper theorems, propositions, remarks, and structural identities
# ---------------------------------------------------------------------------

def check_theorem_5_3_and_prop_5_4(*, verbose: bool = True) -> None:
    """Section 5: F(x) = +- t_11 det(Q) E_tilde(x) and Proposition 5.4 for det(Q)."""
    cases = [
        ([[1, 2], [3, 4]], [[1, 1], [1, 1]]),
        ([[2, 1, 3], [1, 4, 2]], [[5, 7], [3, 4, 5]]),
        ([[3, 9, 1], [3, 2, 9], [5, 3, 4]], [[1, 1, 1], [1, 1, 1]]),
        ([[8, 2, 9, 7], [9, 1, 4, 2], [3, 7, 4, 5], [2, 6, 9, 9]], [[2, 12, 4, 6], [1, 15, 2, 6]]),
    ]
    for T, mus in cases:
        e_sylv = eliminant_polynomial(T, mus)
        e_diag = matrix_diagonal_polynomial(T, mus)
        assert e_diag.coeffs == e_sylv.coeffs
        t11 = T[0][0]
        ratio_f = Fraction(e_diag.F_coeffs[-1], t11 * e_diag.det_Q * e_sylv.E_tilde_coeffs[-1])
        assert ratio_f in (1, -1)
        pencil = build_matrix_pencil(T, mus)
        assert abs(_det_exact(pencil.Q)) == e_diag.det_Q
    if verbose:
        print("[ok] Theorem 5.3 & Proposition 5.4: matrix diagonal pencil F(x) and det(Q) verified on 2x2..4x4.")


def check_remark_4_7_degenerate_minor(*, verbose: bool = True) -> None:
    """Remark 4.7: vanishing minor away from (1,1) sets c(T)=0, while diagonal pencil still succeeds."""
    T = [[3, 2, 5], [1, 2, 4], [7, 3, 6]]  # bottom-right 2x2 minor = 2*6 - 4*3 = 0
    mus = [[1, 1, 1], [1, 1, 1]]
    e_sylv = eliminant_polynomial(T, mus)
    assert e_sylv.coeffs == [0]

    e_diag = matrix_diagonal_polynomial(T, mus)
    assert e_diag.degree > 0
    P_sink, _ = sinkhorn_scale(T, mus)
    x_star = P_sink[0][0]
    val = sum(c * (x_star**j) for j, c in enumerate(e_diag.coeffs))
    scale = max(abs(c) for c in e_diag.coeffs)
    assert abs(val) / scale < 1e-12
    if verbose:
        print(f"[ok] Remark 4.7: Sylvester R(x)===0, matrix diagonal recovers degree-{e_diag.degree} eliminant.")


def check_proposition_4_5_constant_term(*, verbose: bool = True) -> None:
    """Proposition 4.5: |R(0)| = t_1^N * prod_k |Res_k| across restricted slabs."""
    T = [
        [[2, 1], [4, 1]],
        [[2, 3], [1, 5]],
    ]
    mus = [[3, 4], [2, 5], [1, 6]]
    res = eliminant_polynomial(T, mus)
    N = multinomial_degree((2, 2, 2))
    t1 = T[0][0][0]
    R0 = res.R_coeffs[0]
    slab_prod = prod(restricted_slab_resultant(T, mus, k) for k in range(3))
    assert abs(R0) == (t1**N) * abs(slab_prod)
    slab_prod_int = prod(restricted_slab_resultant(T, mus, k, integer_scaled=True) for k in range(3))
    tau = sum(mus[0])
    assert abs(res.R_int_coeffs[0]) == ((tau * t1) ** N) * abs(slab_prod_int)
    assert R0 != 0
    if verbose:
        print("[ok] Proposition 4.5: constant term R(0) matches restricted slab resultant product.")


def check_corollary_3_3_entry_permutation(*, verbose: bool = True) -> None:
    """Arbitrary target entry (i_1, ..., i_r) via mode permutation."""
    T = [
        [[4, 3, 8], [1, 6, 4]],
        [[7, 9, 6], [9, 5, 4]],
    ]
    mus = [[9, 15], [7, 17], [7, 7, 10]]
    entry = (1, 0, 2)
    res = solve_scaling(T, mus, entry=entry, recover_all=False)
    P_sink, _ = sinkhorn_scale(T, mus)
    assert isclose(res.limit_root, P_sink[1][0][2], abs_tol=1e-10)
    assert res.sinkhorn_diff < 1e-10
    if verbose:
        print(f"[ok] Arbitrary entry {entry}: P*[{entry}] = {res.limit_root:.10f} matches Sinkhorn.")


def check_multidegree_permutations(*, verbose: bool = True) -> None:
    """Lemma 3.1 / Theorem 3.2: det Phi^pi(x) = +- det Phi(x) for all mode orderings."""
    T = [
        [[2, 1], [4, 1]],
        [[2, 3], [1, 5]],
    ]
    mus = [[3, 4], [2, 5], [1, 6]]
    ref = eliminant_polynomial(T, mus, multidegree_perm=(0, 1, 2))
    for perm in itertools.permutations([0, 1, 2]):
        alt = eliminant_polynomial(T, mus, multidegree_perm=perm)
        assert alt.coeffs == ref.coeffs
        assert alt.R_coeffs[0] in (ref.R_coeffs[0], -ref.R_coeffs[0])
    if verbose:
        print("[ok] Multidegree invariance: all 6 mode permutations yield identical primitive E(x).")


def check_rational_and_rescaling(*, verbose: bool = True) -> None:
    """Exact Fraction inputs and projective rescaling invariance of T."""
    T = [
        [Fraction(2, 3), Fraction(1, 5), 3],
        [Fraction(1, 2), 4, Fraction(7, 4)],
    ]
    mus = sinkhorn_margins((2, 3))
    res = solve_scaling(T, mus, recover_all=True)
    assert res.eliminant.degree == 3
    assert res.sinkhorn_diff < 1e-12
    T_scaled = [[v * 15 for v in row] for row in T]
    e_scaled = eliminant_polynomial(T_scaled, mus)
    assert e_scaled.coeffs == res.eliminant.coeffs
    if verbose:
        print("[ok] Exact rational inputs & scaling invariance verified on 2x3 Fraction matrix.")


def check_higher_order_tensors(*, verbose: bool = True) -> None:
    """4D (2x2x2x2, N=24) and 3D (2x3x3, N=30) tensors."""
    rng4 = random.Random(42)
    T4 = [[[[rng4.randint(1, 20) for _ in range(2)] for _ in range(2)] for _ in range(2)] for _ in range(2)]
    mus4 = [[3, 5], [2, 6], [5, 3], [1, 7]]
    res4 = solve_scaling(T4, mus4, recover_all=False)
    assert res4.eliminant.degree == 24
    assert res4.sinkhorn_diff < 1e-12

    rng3 = random.Random(7)
    T3 = [[[rng3.randint(1, 15) for _ in range(3)] for _ in range(3)] for _ in range(2)]
    mus3 = [[3, 3], [2, 2, 2], [2, 2, 2]]
    e3 = eliminant_polynomial(T3, mus3)
    assert e3.degree == 30
    assert e3.theorem46["supported"]
    assert e3.E_tilde_coeffs[-1] == e3.theorem46["lead_pred"]
    assert e3.E_tilde_coeffs[0] in (e3.theorem46["const_pred"], -e3.theorem46["const_pred"])
    if verbose:
        print("[ok] Higher-order tensors: 2x2x2x2 (deg 24) and 2x3x3 (deg 30, Thm 4.6) verified.")


def check_root_finder_and_cli(*, verbose: bool = True) -> None:
    """Pure-Python root finder (zero/repeated roots) and CLI JSON output."""
    coeffs = [0, 0, 9, -6, 10, -6, 1]  # x^2 * (x - 3)^2 * (x^2 + 1)
    roots = find_polynomial_roots(coeffs)
    assert len(roots) == 6
    assert len([z for z in roots if abs(z) < 1e-8]) == 2
    assert len([z for z in roots if abs(z - 3) < 1e-4]) == 2
    imags = sorted([z for z in roots if abs(z.imag) > 0.5], key=lambda z: z.imag)
    assert len(imags) == 2
    assert isclose(imags[0].real, 0.0, abs_tol=1e-9) and isclose(imags[0].imag, -1.0, abs_tol=1e-9)
    assert isclose(imags[1].real, 0.0, abs_tol=1e-9) and isclose(imags[1].imag, 1.0, abs_tol=1e-9)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "sinkhorn_eliminant",
            "--tensor",
            "[[1,2],[3,4]]",
            "--sinkhorn",
            "--matrix-diagonal",
            "--json",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(proc.stdout)
    assert data["degree"] == 2
    assert data["matrix_diagonal_matches"]
    assert isclose(data["limit_root"], 6**0.5 - 2, abs_tol=1e-9)
    if verbose:
        print("[ok] Root finder (zero/repeated roots) and CLI --json output verified.")


# ---------------------------------------------------------------------------
# Optional unittest interface (`python -m unittest -v reproduce.py`)
# ---------------------------------------------------------------------------

class PaperCompanionSuite(unittest.TestCase):
    def test_appendix_a1_rw1(self):
        run_appendix_a1_rw1(verbose=False)

    def test_appendix_a2_2x2x3(self):
        run_appendix_a2_2x2x3(verbose=False)

    def test_appendix_a3_4x4(self):
        run_appendix_a3_4x4(verbose=False)

    def test_appendix_a4_degenerate_2x2x2(self):
        run_appendix_a4_degenerate_2x2x2(verbose=False)

    def test_rw_example_13(self):
        run_rw_example_13(verbose=False)

    def test_kruithof_1937_rw32(self):
        run_kruithof_1937_rw32(verbose=False)

    def test_theorem_5_3_and_prop_5_4(self):
        check_theorem_5_3_and_prop_5_4(verbose=False)

    def test_remark_4_7_degenerate_minor(self):
        check_remark_4_7_degenerate_minor(verbose=False)

    def test_proposition_4_5_constant_term(self):
        check_proposition_4_5_constant_term(verbose=False)

    def test_corollary_3_3_entry_permutation(self):
        check_corollary_3_3_entry_permutation(verbose=False)

    def test_multidegree_permutations(self):
        check_multidegree_permutations(verbose=False)

    def test_rational_and_rescaling(self):
        check_rational_and_rescaling(verbose=False)

    def test_higher_order_tensors(self):
        check_higher_order_tensors(verbose=False)

    def test_root_finder_and_cli(self):
        check_root_finder_and_cli(verbose=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce worked examples and mathematical identities from HexagonMath 2611.00433."
    )
    parser.add_argument(
        "--examples-only",
        action="store_true",
        help="Run only the worked examples (Appendix A.1-A.4, RW Ex. 13, Kruithof 1937).",
    )
    parser.add_argument(
        "--full-polys",
        action="store_true",
        help="Print complete E(x) and E_tilde(x) polynomial coefficients for high-degree examples (deg > 12).",
    )
    args = parser.parse_args(argv)

    examples = [
        run_appendix_a1_rw1,
        run_appendix_a2_2x2x3,
        run_appendix_a3_4x4,
        run_appendix_a4_degenerate_2x2x2,
        run_rw_example_13,
        run_kruithof_1937_rw32,
    ]
    identities = [
        check_theorem_5_3_and_prop_5_4,
        check_remark_4_7_degenerate_minor,
        check_proposition_4_5_constant_term,
        check_corollary_3_3_entry_permutation,
        check_multidegree_permutations,
        check_rational_and_rescaling,
        check_higher_order_tensors,
        check_root_finder_and_cli,
    ]

    for fn in examples:
        fn(verbose=True, full_polys=args.full_polys)

    if not args.examples_only:
        print("=" * 78)
        print("Mathematical identities & structural checks")
        for fn in identities:
            fn(verbose=True)

    total = len(examples) if args.examples_only else len(examples) + len(identities)
    print("=" * 78)
    print(f"All {total} paper companion checks completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
