"""Companion package for:
S. Sivarajan, "An explicit determinantal formula for Sinkhorn limits of
matrices and tensors" (HexagonMath 2611.00433).
"""

from .discriminants import cayley_hyperdeterminant, face_discriminant, theorem46_factors
from .matrix_pencil import MatrixDiagonalResult, MatrixPencil, build_matrix_pencil, matrix_diagonal_polynomial
from .resultant import EliminantResult, eliminant_polynomial, format_polynomial, restricted_slab_resultant
from .solver import (
    ScalingSolution,
    ScalingSolveResult,
    find_polynomial_roots,
    recover_solution,
    sinkhorn_scale,
    solve_pencil_numerical,
    solve_scaling,
)
from .sylvester import (
    monomials,
    multinomial_degree,
    sinkhorn_margins,
    sylvester_blocks,
    sylvester_matrix,
    sz_multidegree,
)

__all__ = [
    "EliminantResult",
    "MatrixDiagonalResult",
    "MatrixPencil",
    "ScalingSolution",
    "ScalingSolveResult",
    "build_matrix_pencil",
    "cayley_hyperdeterminant",
    "eliminant_polynomial",
    "face_discriminant",
    "find_polynomial_roots",
    "format_polynomial",
    "matrix_diagonal_polynomial",
    "monomials",
    "multinomial_degree",
    "recover_solution",
    "restricted_slab_resultant",
    "sinkhorn_margins",
    "sinkhorn_scale",
    "solve_pencil_numerical",
    "solve_scaling",
    "sylvester_blocks",
    "sylvester_matrix",
    "sz_multidegree",
    "theorem46_factors",
]
