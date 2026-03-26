"""
dense_matrix.py - General Dense Matrix Class
=============================================

Purpose:
    Full (dense) matrix storage using a flat list in row-major order.
    Supports general matrix operations including addition, multiplication,
    transpose, and matrix-vector product.

Storage:
    Elements stored in a 1D list of size n*n in row-major order.
    Element A[i][j] is at index i*n + j.

Assumptions:
    - Square matrices only (for structural analysis stiffness matrices).
    - 0-based indexing.
"""

from matrix.base_matrix import BaseMatrix
from matrix.vector import Vector


class DenseMatrix(BaseMatrix):
    """
    General dense square matrix.

    Attributes:
        _data (list[float]): Flat storage in row-major order.
    """

    def __init__(self, n):
        """
        Create an n x n zero matrix.

        Inputs:
            n (int): Matrix dimension.
        """
        super().__init__(n)
        self._data = [0.0] * (n * n)

    @classmethod
    def from_list(cls, data):
        """
        Create a DenseMatrix from a 2D list.

        Inputs:
            data (list[list[float]]): 2D list of values, must be square.

        Returns:
            DenseMatrix: New matrix with given values.

        Raises:
            ValueError: If data is not square.
        """
        n = len(data)
        if n == 0:
            raise ValueError("Cannot create matrix from empty data")
        for row in data:
            if len(row) != n:
                raise ValueError("Input data must be square")
        mat = cls(n)
        for i in range(n):
            for j in range(n):
                mat._data[i * n + j] = float(data[i][j])
        return mat

    def get(self, i, j):
        """
        Retrieve element at row i, column j.

        Inputs:
            i (int): Row index (0-based).
            j (int): Column index (0-based).

        Returns:
            float: A[i][j].
        """
        self._check_bounds(i, j)
        return self._data[i * self._n + j]

    def set(self, i, j, value):
        """
        Set element at row i, column j.

        Inputs:
            i (int): Row index (0-based).
            j (int): Column index (0-based).
            value (float): Value to store.
        """
        self._check_bounds(i, j)
        self._data[i * self._n + j] = float(value)

    def add(self, i, j, value):
        """
        Add value to element at row i, column j.

        Inputs:
            i (int): Row index (0-based).
            j (int): Column index (0-based).
            value (float): Value to add.

        Purpose:
            Used during FEM assembly to accumulate element contributions.
        """
        self._check_bounds(i, j)
        self._data[i * self._n + j] += float(value)

    def __add__(self, other):
        """
        Matrix addition: self + other.

        Inputs:
            other (DenseMatrix): Matrix of same size.

        Returns:
            DenseMatrix: Element-wise sum.
        """
        if not isinstance(other, DenseMatrix):
            raise TypeError("Can only add DenseMatrix to DenseMatrix")
        if self._n != other._n:
            raise ValueError(f"Size mismatch: {self._n} vs {other._n}")
        result = DenseMatrix(self._n)
        for k in range(self._n * self._n):
            result._data[k] = self._data[k] + other._data[k]
        return result

    def __sub__(self, other):
        """
        Matrix subtraction: self - other.

        Inputs:
            other (DenseMatrix): Matrix of same size.

        Returns:
            DenseMatrix: Element-wise difference.
        """
        if not isinstance(other, DenseMatrix):
            raise TypeError("Can only subtract DenseMatrix from DenseMatrix")
        if self._n != other._n:
            raise ValueError(f"Size mismatch: {self._n} vs {other._n}")
        result = DenseMatrix(self._n)
        for k in range(self._n * self._n):
            result._data[k] = self._data[k] - other._data[k]
        return result

    def __mul__(self, scalar):
        """
        Scalar multiplication: self * scalar.

        Inputs:
            scalar (float): Scalar multiplier.

        Returns:
            DenseMatrix: Scaled matrix.
        """
        scalar = float(scalar)
        result = DenseMatrix(self._n)
        for k in range(self._n * self._n):
            result._data[k] = self._data[k] * scalar
        return result

    def __rmul__(self, scalar):
        """
        Allow scalar * matrix (reverse multiplication).

        Inputs:
            scalar (float): Scalar multiplier.

        Returns:
            DenseMatrix: Scaled matrix.
        """
        return self.__mul__(scalar)

    def mat_vec(self, v):
        """
        Matrix-vector product: A * v.

        Inputs:
            v (Vector): Vector of size n.

        Returns:
            Vector: Result vector of size n.
        """
        if not isinstance(v, Vector):
            raise TypeError("Expected Vector")
        if v.size != self._n:
            raise ValueError(f"Size mismatch: matrix {self._n}, vector {v.size}")
        result = Vector(self._n)
        for i in range(self._n):
            s = 0.0
            base = i * self._n
            for j in range(self._n):
                s += self._data[base + j] * v[j]
            result[i] = s
        return result

    def mat_mul(self, other):
        """
        Matrix-matrix product: self * other.

        Inputs:
            other (DenseMatrix): Matrix of same size.

        Returns:
            DenseMatrix: Product matrix.
        """
        if not isinstance(other, DenseMatrix):
            raise TypeError("Expected DenseMatrix")
        if self._n != other._n:
            raise ValueError(f"Size mismatch: {self._n} vs {other._n}")
        n = self._n
        result = DenseMatrix(n)
        for i in range(n):
            for j in range(n):
                s = 0.0
                for k in range(n):
                    s += self._data[i * n + k] * other._data[k * n + j]
                result._data[i * n + j] = s
        return result

    def transpose(self):
        """
        Compute transpose of the matrix.

        Returns:
            DenseMatrix: Transposed matrix A^T.
        """
        n = self._n
        result = DenseMatrix(n)
        for i in range(n):
            for j in range(n):
                result._data[j * n + i] = self._data[i * n + j]
        return result

    def copy(self):
        """
        Deep copy of the matrix.

        Returns:
            DenseMatrix: Independent copy.
        """
        result = DenseMatrix(self._n)
        result._data = self._data[:]
        return result

    def get_row(self, i):
        """
        Extract row i as a Vector.

        Inputs:
            i (int): Row index.

        Returns:
            Vector: Row vector.
        """
        if i < 0 or i >= self._n:
            raise IndexError(f"Row {i} out of bounds")
        base = i * self._n
        return Vector(self._data[base:base + self._n])

    def get_column(self, j):
        """
        Extract column j as a Vector.

        Inputs:
            j (int): Column index.

        Returns:
            Vector: Column vector.
        """
        if j < 0 or j >= self._n:
            raise IndexError(f"Column {j} out of bounds")
        return Vector([self._data[i * self._n + j] for i in range(self._n)])
