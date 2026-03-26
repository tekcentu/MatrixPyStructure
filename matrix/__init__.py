"""
matrix - Object-Oriented Matrix Library for Structural Analysis
===============================================================

A pure Python matrix library designed for structural analysis applications.
Provides dense, symmetric, and skyline matrix storage schemes with
optimized solvers for linear systems.

No external numerical libraries (numpy, scipy) are used.

Classes:
    Vector          - Mathematical vector with arithmetic operations
    DenseMatrix     - General dense matrix storage
    SymmetricMatrix - Symmetric matrix using upper-triangle storage
    SkylineMatrix   - Skyline (envelope) matrix storage for FEM
    Solver          - Linear system solvers (LDL^T, Cholesky)
"""

from matrix.vector import Vector
from matrix.dense_matrix import DenseMatrix
from matrix.symmetric_matrix import SymmetricMatrix
from matrix.skyline_matrix import SkylineMatrix
from matrix.solver import Solver

__all__ = [
    "Vector",
    "DenseMatrix",
    "SymmetricMatrix",
    "SkylineMatrix",
    "Solver",
]
