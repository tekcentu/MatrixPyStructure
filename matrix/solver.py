"""
solver.py - Linear System Solvers
==================================

Purpose:
    Provides solver methods for linear systems Ax = b using different
    matrix storage schemes. Acts as a unified interface to choose the
    appropriate solution method based on the matrix type.

Supported Methods:
    - Gaussian elimination with partial pivoting (DenseMatrix)
    - LDL^T factorization via skyline method (SkylineMatrix)
    - LDL^T factorization for packed symmetric (SymmetricMatrix)

Assumptions:
    - For skyline and symmetric solvers, matrix must be SPD.
    - For dense solver, matrix must be nonsingular.
"""

from matrix.dense_matrix import DenseMatrix
from matrix.symmetric_matrix import SymmetricMatrix
from matrix.skyline_matrix import SkylineMatrix
from matrix.vector import Vector


class Solver:
    """
    Static class providing linear system solution methods.

    All methods are static; no instance is needed.
    """

    @staticmethod
    def solve(matrix, b):
        """
        Solve A * x = b by dispatching to the appropriate method.

        Inputs:
            matrix (BaseMatrix): Coefficient matrix (DenseMatrix,
                SymmetricMatrix, or SkylineMatrix).
            b (Vector): Right-hand side vector.

        Returns:
            Vector: Solution vector x.

        Note:
            The matrix is NOT modified (a copy is used for factorization).
        """
        if isinstance(matrix, SkylineMatrix):
            return Solver.solve_skyline(matrix, b)
        elif isinstance(matrix, SymmetricMatrix):
            return Solver.solve_symmetric(matrix, b)
        elif isinstance(matrix, DenseMatrix):
            return Solver.solve_dense(matrix, b)
        else:
            raise TypeError(f"Unsupported matrix type: {type(matrix)}")

    @staticmethod
    def solve_dense(matrix, b):
        """
        Solve A * x = b using Gaussian elimination with partial pivoting.

        Inputs:
            matrix (DenseMatrix): Coefficient matrix (n x n).
            b (Vector): Right-hand side vector (size n).

        Returns:
            Vector: Solution vector x.

        Algorithm:
            1. Forward elimination with row pivoting.
            2. Back substitution.

        Note:
            Works on copies; original matrix and vector are not modified.
        """
        if not isinstance(matrix, DenseMatrix):
            raise TypeError("Expected DenseMatrix")
        if not isinstance(b, Vector):
            raise TypeError("Expected Vector for RHS")

        n = matrix.size
        if b.size != n:
            raise ValueError(f"Size mismatch: matrix {n}, vector {b.size}")

        # Create augmented matrix [A|b] as list of lists
        aug = []
        for i in range(n):
            row = [matrix.get(i, j) for j in range(n)]
            row.append(b[i])
            aug.append(row)

        # Forward elimination with partial pivoting
        for col in range(n):
            # Find pivot row
            max_val = abs(aug[col][col])
            max_row = col
            for row in range(col + 1, n):
                if abs(aug[row][col]) > max_val:
                    max_val = abs(aug[row][col])
                    max_row = row

            if max_val < 1e-30:
                raise ArithmeticError(
                    f"Matrix is singular (zero pivot at column {col})"
                )

            # Swap rows
            if max_row != col:
                aug[col], aug[max_row] = aug[max_row], aug[col]

            # Eliminate below
            pivot = aug[col][col]
            for row in range(col + 1, n):
                factor = aug[row][col] / pivot
                for j in range(col, n + 1):
                    aug[row][j] -= factor * aug[col][j]

        # Back substitution
        x = Vector(n)
        for i in range(n - 1, -1, -1):
            s = aug[i][n]
            for j in range(i + 1, n):
                s -= aug[i][j] * x[j]
            x[i] = s / aug[i][i]

        return x

    @staticmethod
    def solve_skyline(matrix, b):
        """
        Solve A * x = b using LDL^T skyline factorization.

        Inputs:
            matrix (SkylineMatrix): SPD coefficient matrix.
            b (Vector): Right-hand side vector.

        Returns:
            Vector: Solution vector x.

        Note:
            Works on a copy of the matrix; original is not modified.
        """
        if not isinstance(matrix, SkylineMatrix):
            raise TypeError("Expected SkylineMatrix")
        if not isinstance(b, Vector):
            raise TypeError("Expected Vector for RHS")

        mat_copy = matrix.copy()
        mat_copy.ldlt_factorize()
        return mat_copy.ldlt_solve(b)

    @staticmethod
    def solve_symmetric(matrix, b):
        """
        Solve A * x = b for a SymmetricMatrix using LDL^T factorization.

        Inputs:
            matrix (SymmetricMatrix): SPD coefficient matrix.
            b (Vector): Right-hand side vector.

        Returns:
            Vector: Solution vector x.

        Algorithm:
            Converts to a skyline matrix (full bandwidth) and uses
            the skyline solver. For small matrices this is efficient;
            for large matrices, prefer SkylineMatrix directly.
        """
        if not isinstance(matrix, SymmetricMatrix):
            raise TypeError("Expected SymmetricMatrix")
        if not isinstance(b, Vector):
            raise TypeError("Expected Vector for RHS")

        n = matrix.size
        # Full column heights (no sparsity exploitation)
        column_heights = [j + 1 for j in range(n)]
        sky = SkylineMatrix(n, column_heights)

        for i in range(n):
            for j in range(i, n):
                val = matrix.get(i, j)
                if abs(val) > 1e-30:
                    sky.set(i, j, val)

        return Solver.solve_skyline(sky, b)
