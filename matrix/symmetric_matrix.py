"""
symmetric_matrix.py - Symmetric Matrix with Upper-Triangle Storage
==================================================================

Purpose:
    Stores only the upper triangle of a symmetric matrix, reducing
    storage from n^2 to n*(n+1)/2 elements. This is ideal for
    structural stiffness matrices which are always symmetric.

Storage Scheme:
    Upper triangle stored in a 1D array in row-major order.
    For i <= j: index = i*n - i*(i+1)/2 + j
    For i > j:  A[i][j] = A[j][i] (symmetry)

    Example for 4x4 matrix, stored elements marked with indices:
        [0  1  2  3 ]
        [   4  5  6 ]
        [      7  8 ]
        [         9 ]
    Total storage: 4*5/2 = 10 elements instead of 16.

Assumptions:
    - Matrix is symmetric: A[i][j] = A[j][i].
    - Setting A[i][j] also sets A[j][i].
"""

from matrix.base_matrix import BaseMatrix
from matrix.vector import Vector


class SymmetricMatrix(BaseMatrix):
    """
    Symmetric matrix stored as upper triangle.

    Attributes:
        _data (list[float]): Upper triangle in row-major packed format.
    """

    def __init__(self, n):
        """
        Create an n x n symmetric zero matrix.

        Inputs:
            n (int): Matrix dimension.

        Storage: n*(n+1)/2 elements.
        """
        super().__init__(n)
        self._storage_size = n * (n + 1) // 2
        self._data = [0.0] * self._storage_size

    def _index(self, i, j):
        """
        Compute packed storage index for upper triangle element (i, j)
        where i <= j.

        Inputs:
            i (int): Row index (i <= j).
            j (int): Column index.

        Returns:
            int: Index into _data array.

        Formula:
            index = i*n - i*(i+1)/2 + j
        """
        return i * self._n - i * (i + 1) // 2 + j

    def get(self, i, j):
        """
        Retrieve element at row i, column j.

        Inputs:
            i (int): Row index (0-based).
            j (int): Column index (0-based).

        Returns:
            float: A[i][j] (uses symmetry if i > j).
        """
        self._check_bounds(i, j)
        if i > j:
            i, j = j, i
        return self._data[self._index(i, j)]

    def set(self, i, j, value):
        """
        Set element at row i, column j.

        Inputs:
            i (int): Row index (0-based).
            j (int): Column index (0-based).
            value (float): Value to store.

        Note:
            Due to symmetry, set(i,j,v) and set(j,i,v) are equivalent.
        """
        self._check_bounds(i, j)
        if i > j:
            i, j = j, i
        self._data[self._index(i, j)] = float(value)

    def add(self, i, j, value):
        """
        Add value to element at row i, column j.

        Inputs:
            i (int): Row index (0-based).
            j (int): Column index (0-based).
            value (float): Value to add.

        Purpose:
            Used during FEM assembly of symmetric stiffness matrices.
        """
        self._check_bounds(i, j)
        if i > j:
            i, j = j, i
        self._data[self._index(i, j)] += float(value)

    def mat_vec(self, v):
        """
        Symmetric matrix-vector product: A * v.

        Inputs:
            v (Vector): Vector of size n.

        Returns:
            Vector: Result vector of size n.

        Note:
            Exploits symmetry: each off-diagonal element contributes
            to two rows, reducing operations by nearly half.
        """
        if not isinstance(v, Vector):
            raise TypeError("Expected Vector")
        if v.size != self._n:
            raise ValueError(f"Size mismatch: matrix {self._n}, vector {v.size}")
        result = Vector(self._n)
        n = self._n
        for i in range(n):
            # Diagonal contribution
            diag = self._data[self._index(i, i)]
            result._data[i] += diag * v._data[i]
            # Off-diagonal contributions (exploit symmetry)
            for j in range(i + 1, n):
                a_ij = self._data[self._index(i, j)]
                result._data[i] += a_ij * v._data[j]
                result._data[j] += a_ij * v._data[i]
        return result

    def copy(self):
        """
        Deep copy of the matrix.

        Returns:
            SymmetricMatrix: Independent copy.
        """
        result = SymmetricMatrix(self._n)
        result._data = self._data[:]
        return result

    @property
    def storage_size(self):
        """Return number of stored elements (upper triangle)."""
        return self._storage_size

    def get_diagonal(self):
        """
        Extract the diagonal as a Vector.

        Returns:
            Vector: Diagonal elements.
        """
        diag = Vector(self._n)
        for i in range(self._n):
            diag[i] = self._data[self._index(i, i)]
        return diag
