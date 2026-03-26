"""
vector.py - Vector Class for Structural Analysis
=================================================

Purpose:
    Provides a mathematical vector class supporting arithmetic operations
    needed for structural analysis: addition, subtraction, scalar
    multiplication, dot product, and norm computation.

Assumptions:
    - All elements are real-valued (float).
    - 0-based indexing is used throughout.

Units:
    No specific units assumed; the caller is responsible for unit consistency.
"""

import math


class Vector:
    """
    A mathematical vector of fixed size.

    Attributes:
        _n (int): Number of elements.
        _data (list[float]): Storage for vector elements.
    """

    def __init__(self, n_or_data):
        """
        Create a vector of given size (initialized to zero) or from a list.

        Inputs:
            n_or_data (int or list): If int, creates zero vector of that size.
                                     If list, creates vector from the data.

        Raises:
            ValueError: If size < 1 or data is empty.
        """
        if isinstance(n_or_data, (int,)):
            if n_or_data < 1:
                raise ValueError(f"Vector size must be >= 1, got {n_or_data}")
            self._n = n_or_data
            self._data = [0.0] * n_or_data
        elif isinstance(n_or_data, (list, tuple)):
            if len(n_or_data) == 0:
                raise ValueError("Cannot create vector from empty data")
            self._n = len(n_or_data)
            self._data = [float(x) for x in n_or_data]
        else:
            raise TypeError(f"Expected int or list, got {type(n_or_data)}")

    @property
    def size(self):
        """Return the number of elements in the vector."""
        return self._n

    def get(self, i):
        """
        Get element at index i.

        Inputs:
            i (int): Index (0-based).

        Returns:
            float: The element value.
        """
        if i < 0 or i >= self._n:
            raise IndexError(f"Index {i} out of bounds for vector of size {self._n}")
        return self._data[i]

    def set(self, i, value):
        """
        Set element at index i.

        Inputs:
            i (int): Index (0-based).
            value (float): Value to store.
        """
        if i < 0 or i >= self._n:
            raise IndexError(f"Index {i} out of bounds for vector of size {self._n}")
        self._data[i] = float(value)

    def add_value(self, i, value):
        """
        Add a value to element at index i (used in assembly).

        Inputs:
            i (int): Index (0-based).
            value (float): Value to add.
        """
        if i < 0 or i >= self._n:
            raise IndexError(f"Index {i} out of bounds for vector of size {self._n}")
        self._data[i] += float(value)

    def __getitem__(self, i):
        """Allow indexing with []."""
        return self.get(i)

    def __setitem__(self, i, value):
        """Allow assignment with []."""
        self.set(i, value)

    def __len__(self):
        """Return vector size."""
        return self._n

    def __add__(self, other):
        """
        Vector addition: self + other.

        Inputs:
            other (Vector): Vector of same size.

        Returns:
            Vector: Element-wise sum.
        """
        if not isinstance(other, Vector):
            raise TypeError("Can only add Vector to Vector")
        if self._n != other._n:
            raise ValueError(f"Size mismatch: {self._n} vs {other._n}")
        result = Vector(self._n)
        for i in range(self._n):
            result._data[i] = self._data[i] + other._data[i]
        return result

    def __sub__(self, other):
        """
        Vector subtraction: self - other.

        Inputs:
            other (Vector): Vector of same size.

        Returns:
            Vector: Element-wise difference.
        """
        if not isinstance(other, Vector):
            raise TypeError("Can only subtract Vector from Vector")
        if self._n != other._n:
            raise ValueError(f"Size mismatch: {self._n} vs {other._n}")
        result = Vector(self._n)
        for i in range(self._n):
            result._data[i] = self._data[i] - other._data[i]
        return result

    def __mul__(self, scalar):
        """
        Scalar multiplication: self * scalar.

        Inputs:
            scalar (float): Scalar multiplier.

        Returns:
            Vector: Scaled vector.
        """
        if isinstance(scalar, Vector):
            raise TypeError("Use dot() for vector-vector product")
        scalar = float(scalar)
        result = Vector(self._n)
        for i in range(self._n):
            result._data[i] = self._data[i] * scalar
        return result

    def __rmul__(self, scalar):
        """Allow scalar * vector."""
        return self.__mul__(scalar)

    def __neg__(self):
        """Negate the vector."""
        return self * (-1.0)

    def dot(self, other):
        """
        Dot product: self . other.

        Inputs:
            other (Vector): Vector of same size.

        Returns:
            float: Scalar dot product.
        """
        if not isinstance(other, Vector):
            raise TypeError("Dot product requires two Vectors")
        if self._n != other._n:
            raise ValueError(f"Size mismatch: {self._n} vs {other._n}")
        result = 0.0
        for i in range(self._n):
            result += self._data[i] * other._data[i]
        return result

    def norm(self):
        """
        Euclidean norm (L2 norm).

        Returns:
            float: ||self||_2
        """
        return math.sqrt(self.dot(self))

    def copy(self):
        """
        Create a deep copy of this vector.

        Returns:
            Vector: Independent copy.
        """
        result = Vector(self._n)
        result._data = self._data[:]
        return result

    def to_list(self):
        """
        Convert to a plain Python list.

        Returns:
            list[float]: Copy of internal data.
        """
        return self._data[:]

    def __repr__(self):
        """String representation."""
        items = ", ".join(f"{x:.6e}" for x in self._data)
        return f"Vector([{items}])"

    def __eq__(self, other):
        """Check equality within floating-point tolerance."""
        if not isinstance(other, Vector):
            return False
        if self._n != other._n:
            return False
        tol = 1e-12
        for i in range(self._n):
            if abs(self._data[i] - other._data[i]) > tol:
                return False
        return True
