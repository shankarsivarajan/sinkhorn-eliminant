"""Section 7 (tensors with zero entries) and Example A.5 of:

    Shankar Sivarajan, "An explicit determinantal formula for Sinkhorn limits
    of matrices and tensors", HexagonMath 2611.00433, v2 (2026).

The 3x3 matrix of Example A.1 with t_23 = t_32 = 0 and doubly stochastic margins. Checks, exactly over Q:
  * the eliminant R_S of the corner entry (Theorem 7.2) has degree N_S = 4 and is irreducible;
  * its real roots, with the sign vectors of the solutions, and the Sinkhorn limit;
  * the Sylvester matrix Phi(x) = [M_0 | K - x J] of size 30 has det = 0 and corank 2 = h_B(1,3);
  * the gcd of det(W Phi(x) V) over three random integer W, V equals R_S (Conjecture 7.6).

Needs sympy and numpy, which the rest of the package does not; it is therefore not imported by
sinkhorn_eliminant/__init__.py. Run via reproduce.py.
"""
import random

import numpy as np
import sympy as sp
from sympy.polys.matrices import DomainMatrix

from .sylvester import sylvester_blocks


def example_a5(verbose: bool = True) -> None:
    """Verify Example A.5; raises AssertionError on any mismatch."""
    out = print if verbose else (lambda *a, **k: None)
    T = [[3, 9, 1], [3, 2, 0], [5, 0, 4]]
    rho = kap = [1, 1, 1]
    x = sp.symbols("x")

    # eliminant of the corner entry, by a lex Groebner basis (u_1 = 1 fixes the gauge)
    u2, u3, v1, v2, v3 = sp.symbols("u2 u3 v1 v2 v3")
    u, v = [1, u2, u3], [v1, v2, v3]
    P = [[T[i][j] * u[i] * v[j] for j in range(3)] for i in range(3)]
    eqs = [sum(P[i]) - rho[i] for i in range(3)] + [sum(P[i][j] for i in range(3)) - kap[j] for j in range(2)]
    G = sp.groebner(eqs + [P[0][0] - x], u2, u3, v1, v2, v3, x, order="lex")
    R = sp.Poly([g for g in G.exprs if g.free_symbols == {x}][0], x)
    R = sp.Poly(sp.Poly(R.monic(), x).as_expr() * sp.ilcm(*[c.q for c in R.monic().all_coeffs()]), x)
    assert R.as_expr() == 2303 * x**4 + 777 * x**3 - 2724 * x**2 - 1152 * x + 288 and R.is_irreducible
    out("R_S(x) =", R.as_expr())

    # real roots and sign vectors (s^(2)_1 = +); P_11 = 3 v_1, then row 2 / column 2 give u_2, etc.
    sg = lambda a: "(" + ",".join("+" if t > 0 else "-" for t in a) + ")"
    for z in sorted((complex(r) for r in R.nroots(n=15)), key=lambda z: (abs(z.imag) > 1e-12, z.real)):
        if abs(z.imag) > 1e-12:
            out(f"root {z.real:.10f} {z.imag:+.10f}i")
            continue
        r = z.real
        best = None
        for a in np.roots([2 * r, 9 * r, -9]):  # u2 (r + 2 v2) = 1 with v2 = 1 / (9 + 2 u2)
            a = a.real
            b1 = r / 3
            b2 = 1 / (9 + 2 * a)
            c = (1 / b1 - 3 - 3 * a) / 5
            b3 = (1 / c - 5 * b1) / 4
            res = abs(3 * b1 + 9 * b2 + b3 - 1) + abs(a * (3 * b1 + 2 * b2) - 1)
            if best is None or res < best[0]:
                best = (res, [1.0, a, c], [b1, b2, b3])
        res, uu, vv = best
        assert res < 1e-8
        if vv[0] < 0:
            uu, vv = [-t for t in uu], [-t for t in vv]
        out(f"root {r:.10f}  s1={sg(uu)} s2={sg(vv)}")

    A = np.array(T, float)
    for _ in range(20000):
        A = A / A.sum(1, keepdims=True)
        A = A / A.sum(0, keepdims=True)
    out("Sinkhorn limit:\n", np.round(A, 5))

    # Sylvester matrix Phi(x) = [M_0 | K - x J] (integer scaled), corank and compressions
    cells = {(i + 1, j + 1): T[i][j] for i in range(3) for j in range(3)}
    M0, K, J = sylvester_blocks((3, 3), cells, [rho, kap])
    Phi = sp.Matrix(M0).row_join(sp.Matrix(K) - x * sp.Matrix(J))
    n = Phi.shape[0]
    rk = Phi.subs(x, sp.Rational(7, 11)).rank()
    out("size", n, "corank", n - rk)
    assert (n, n - rk) == (30, 2)
    rnd = random.Random(1)
    dets = []
    for _ in range(3):
        W = sp.Matrix(rk, n, lambda i, j: rnd.randint(-5, 5))
        V = sp.Matrix(n, rk, lambda i, j: rnd.randint(-5, 5))
        pts = list(range(9))  # det(W Phi(x) V) has degree <= 6 in x
        vals = [DomainMatrix.from_Matrix(W * Phi.subs(x, t) * V).convert_to(sp.QQ).det() for t in pts]
        dets.append(sp.Poly(sp.interpolate(list(zip(pts, vals)), x), x))
    g = dets[0]
    for d in dets[1:]:
        g = sp.gcd(g, d)
    out("degrees", [d.degree() for d in dets], " gcd == R_S:", g.monic() == R.monic())
    assert g.monic() == R.monic()
    out("[ok] Example A.5 verified.")
