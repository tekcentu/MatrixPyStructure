"""
base_matrix.py - Abstract Base Class for Matrix Types
=====================================================

Purpose:
    Defines the interface (abstract base class) that all matrix storage
    schemes must implement. Ensures consistency across DenseMatrix,
    SymmetricMatrix, and SkylineMatrix.

Design Decisions:
    - Uses Python's abc module for interface enforcement
    - All matrices are square (structural stiffness matrices are square)
    - Provides default implementations for common operations where possible
"""

from abc import ABC, abstractmethod


class BaseMatrix(ABC):
    """
    Abstract base class for all matrix types.

    Attributes:
        _n (int): Matrix dimension (n x n).

    All concrete subclasses must implement get, set, and solve operations.
    """

    def __init__(self, n):
        """
        Initialize matrix dimensions.

        Inputs:
            n (int): Size of the square matrix (n x n).

        Raises:
            ValueError: If n < 1.
        """
        if n < 1:
            raise ValueError(f"Matrix size must be >= 1, got {n}")
        self._n = n

    @property
    def size(self):
        """Return the dimension n of the n x n matrix."""
        return self._n

    @abstractmethod
    def get(self, i, j):
        """
        Retrieve element at row i, column j.

        Inputs:
            i (int): Row index (0-based).
            j (int): Column index (0-based).

        Returns:
            float: The matrix element A[i][j].
        """
        pass

    @abstractmethod
    def set(self, i, j, value):
        """
        Set element at row i, column j.

        Inputs:
            i (int): Row index (0-based).
            j (int): Column index (0-based).
            value (float): Value to store.
        """
        pass

    @abstractmethod
    def add(self, i, j, value):
        """
        Add value to element at row i, column j.

        Inputs:
            i (int): Row index (0-based).
            j (int): Column index (0-based).
            value (float): Value to add.

        Purpose:
            Efficiently supports assembly operations in FEM where
            contributions are accumulated into the global stiffness matrix.
        """
        pass

    def _check_bounds(self, i, j):
        """
        Validate that indices are within matrix bounds.

        Inputs:
            i (int): Row index.
            j (int): Column index.

        Raises:
            IndexError: If indices are out of bounds.
        """
        if i < 0 or i >= self._n or j < 0 or j >= self._n:
            raise IndexError(
                f"Index ({i}, {j}) out of bounds for matrix of size {self._n}"
            )

    def __repr__(self):
        """
        Return string representation of the matrix.

        Returns:
            str: Formatted string with all elements in scientific notation,
                 one row per line.
        """
        rows = []
        for i in range(self._n):
            row = [f"{self.get(i, j):12.4e}" for j in range(self._n)]
            rows.append(" ".join(row))
        return "\n".join(rows)

    def to_dense_list(self):
        """
        Convert matrix to a 2D list (list of lists).

        Returns:
            list[list[float]]: Full dense representation.

        Purpose:
            Useful for verification and debugging.
        """
        return [[self.get(i, j) for j in range(self._n)]
                for i in range(self._n)]
