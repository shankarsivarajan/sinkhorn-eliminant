"""Command-line interface for computing Sinkhorn/Kruithof eliminant polynomials."""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

from .matrix_pencil import matrix_diagonal_polynomial
from .resultant import eliminant_polynomial, format_polynomial
from .sylvester import normalize_tensor, sinkhorn_margins
from .solver import solve_scaling


def _parse_exact_nested(obj: Any) -> Any:
    if isinstance(obj, list):
        return [_parse_exact_nested(x) for x in obj]
    if isinstance(obj, str):
        return Fraction(obj)
    if isinstance(obj, bool):
        raise ValueError("Booleans are not valid tensor/margin entries.")
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return Fraction(str(obj))
    raise ValueError(f"Unsupported value {obj!r}")


def _fmt_signs(signs: list[list[int]] | None) -> str:
    if signs is None:
        return ""
    return "  ".join("(" + ",".join("+" if v > 0 else "-" for v in vec) + ")" for vec in signs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sinkhorn_eliminant",
        description="Compute the exact Sinkhorn/Kruithof eliminant polynomial and scaling limit.",
    )
    parser.add_argument("--input", "-i", type=Path, help="JSON file with {'tensor': ..., 'margins': ...}.")
    parser.add_argument("--tensor", "-t", type=str, help="JSON nested array for the input matrix/tensor T.")
    parser.add_argument(
        "--margins",
        "-m",
        type=str,
        help="JSON list of target margin vectors mu^(1), ..., mu^(r) (supports '1/3' strings).",
    )
    parser.add_argument(
        "--sinkhorn",
        action="store_true",
        help="Use Rowland-Wu / uniform Sinkhorn margins instead of --margins.",
    )
    parser.add_argument(
        "--entry",
        "-e",
        type=str,
        default=None,
        help="0-based target entry indices, e.g. '0,0' or '1,0,2' (default: corner 0,...,0).",
    )
    parser.add_argument(
        "--matrix-diagonal",
        action="store_true",
        help="For 2D matrices, also compute Fang's N x N diagonal pencil F(x) (Section 5).",
    )
    parser.add_argument(
        "--poly-only",
        action="store_true",
        help="Compute and print only the polynomial (skip root-finding and solution recovery).",
    )
    parser.add_argument(
        "--all-roots",
        action="store_true",
        help="Recover and display all complex/real roots and their sign vectors.",
    )
    parser.add_argument("--dps", type=int, default=60, help="Decimal digits for root polishing (default: 60).")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON.")

    args = parser.parse_args(argv)

    if args.input:
        payload = json.loads(args.input.read_text())
        T_raw = payload["tensor"]
        margins_raw = payload.get("margins")
        entry_raw = payload.get("entry")
    elif args.tensor:
        T_raw = json.loads(args.tensor)
        margins_raw = json.loads(args.margins) if args.margins else None
        entry_raw = None
    else:
        parser.error("Provide either --input FILE or --tensor JSON.")
        return 2

    T = _parse_exact_nested(T_raw)
    dims, _ = normalize_tensor(T)

    if args.sinkhorn:
        margins = sinkhorn_margins(dims)
    elif margins_raw is not None:
        margins = _parse_exact_nested(margins_raw)
    else:
        parser.error("Provide --margins JSON or --sinkhorn.")
        return 2

    if args.entry is not None:
        entry = tuple(int(s.strip()) for s in args.entry.split(","))
    elif entry_raw is not None:
        entry = tuple(int(s) for s in entry_raw)
    else:
        entry = None

    if args.poly_only:
        res = eliminant_polynomial(T, margins, entry=entry)
        diag_res = matrix_diagonal_polynomial(T, margins, entry=entry) if (args.matrix_diagonal and len(dims) == 2) else None
        if args.json:
            out: dict[str, Any] = {
                "dims": list(res.dims),
                "entry": list(res.entry),
                "expected_degree": res.expected_degree,
                "degree": res.degree,
                "sylvester_size": res.sylvester_size,
                "coeffs_asc": [str(c) for c in res.coeffs],
                "E_tilde_asc": [str(c) for c in res.E_tilde_coeffs] if res.E_tilde_coeffs is not None else None,
            }
            if diag_res is not None:
                out["matrix_diagonal_coeffs_asc"] = [str(c) for c in diag_res.coeffs]
            print(json.dumps(out, indent=2))
            return 0

        print(f"Format: {'x'.join(map(str, res.dims))}, entry={res.entry}, N={res.expected_degree}, degree={res.degree}")
        print(f"Sylvester matrix size: {res.sylvester_size} x {res.sylvester_size}, multidegree m={res.multidegree}")
        print(f"Primitive E(x) = {res.format('primitive')}")
        if res.E_tilde_coeffs is not None:
            print(f"Theorem 4.6 E_tilde(x) = {res.format('E_tilde')}")
        if diag_res is not None:
            print(f"Matrix diagonal matches primitive E(x): {diag_res.coeffs == res.coeffs}")
        return 0

    solve_res = solve_scaling(T, margins, entry=entry, recover_all=args.all_roots, dps=args.dps)
    elim = solve_res.eliminant
    diag_res = matrix_diagonal_polynomial(T, margins, entry=entry) if (args.matrix_diagonal and len(dims) == 2) else None

    if args.json:
        out = {
            "dims": list(elim.dims),
            "entry": list(elim.entry),
            "expected_degree": elim.expected_degree,
            "degree": elim.degree,
            "coeffs_asc": [str(c) for c in elim.coeffs],
            "E_tilde_asc": [str(c) for c in elim.E_tilde_coeffs] if elim.E_tilde_coeffs is not None else None,
            "limit_root": solve_res.limit_root,
            "limit_P": solve_res.limit_P,
            "limit_scalings": solve_res.limit_scalings,
            "sinkhorn_diff": solve_res.sinkhorn_diff,
            "solutions": [
                {
                    "re": s.root.real,
                    "im": s.root.imag,
                    "real": s.real,
                    "feasible": s.feasible,
                    "positive": s.positive,
                    "nullvec_positive": s.nullvec_positive,
                    "signs": s.signs,
                    "margin_residual": s.margin_residual,
                }
                for s in solve_res.solutions
            ],
        }
        if diag_res is not None:
            out["matrix_diagonal_matches"] = diag_res.coeffs == elim.coeffs
        print(json.dumps(out, indent=2))
        return 0

    print(f"Format: {'x'.join(map(str, elim.dims))}, entry={elim.entry}, N={elim.expected_degree}, degree={elim.degree}")
    print(f"Primitive E(x) = {elim.format('primitive')}")
    if elim.E_tilde_coeffs is not None:
        print(f"Theorem 4.6 E_tilde(x) = {elim.format('E_tilde')}")
    if diag_res is not None:
        print(f"Matrix diagonal F(x) primitive match: {diag_res.coeffs == elim.coeffs}")
    print(f"Limit P[{elim.entry}] = {solve_res.limit_root:.12g}  (max |P - Sinkhorn| = {solve_res.sinkhorn_diff:.2e})")
    print(f"Limit P* = {solve_res.limit_P}")
    if args.all_roots:
        print("\nRoots:")
        for s in solve_res.solutions:
            if s.real:
                tag = " <-- LIMIT (positive)" if s.positive else ""
                print(
                    f"  {s.root.real:18.10g}  feas={str(s.feasible):5}  pos_nv={str(s.nullvec_positive):5}  "
                    f"signs=[{_fmt_signs(s.signs)}]  res={s.margin_residual:.1e}{tag}"
                )
            else:
                print(f"  {s.root.real:12.8g} {s.root.imag:+12.8g}i  (complex)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
