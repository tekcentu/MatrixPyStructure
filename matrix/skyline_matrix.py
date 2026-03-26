"""
skyline_matrix.py - Skyline (Envelope) Matrix Storage and Solver
================================================================

Purpose:
    Implements the skyline (also called envelope or profile) storage scheme
    for symmetric positive-definite matrices. This is the standard storage
    scheme used in many finite element programs because:
    1. It exploits both symmetry and sparsity.
    2. The storage pattern is determined by the matrix topology (mesh
       connectivity), not by numerical values.
    3. It supports efficient LDL^T factorization without fill-in beyond
       the skyline profile.

Storage Scheme:
    For a symmetric matrix, only the upper triangle is stored. For each
    column j, we store elements from the first nonzero row in that column
    (the "skyline" row) down to the diagonal.

    The storage uses two arrays:
    - _data[]: Packed array of skyline column segments.
    - _diag_ptr[]: Array of size n, where _diag_ptr[j] gives the index
                   in _data where the diagonal element of column j is stored.
    - _column_heights[]: Height of each column (number of elements stored
                         in column j above and including the diagonal).

    Example for a 5x5 symmetric matrix with this sparsity pattern:
        [x  x  0  0  0]        Column heights: [1, 2, 2, 3, 3]
        [   x  x  0  0]        Skyline storage of upper triangle columns:
        [      x  x  x]          col 0: [a00]
        [         x  x]          col 1: [a01, a11]
        [            x]          col 2: [a12, a22]
                                  col 3: [a13, a23, a33]
                                  col 4: [a24, a34, a44]

    Total storage = sum of column heights = 1+2+2+3+3 = 11
    vs. full upper triangle = 15 elements.

Assumptions:
    - Matrix is symmetric positive-definite (SPD).
    - Column heights must be specified before filling the matrix.
    - Used for structural stiffness matrices which are SPD.

Units:
    No specific units; caller ensures consistency.
"""

import math
from matrix.base_matrix import BaseMatrix
from matrix.vector import Vector


class SkylineMatrix(BaseMatrix):
    """
    Skyline (envelope) storage for symmetric positive-definite matrices.

    Attributes:
        _column_heights (list[int]): Height of each column.
        _diag_ptr (list[int]): Index of diagonal element of each column in _data.
        _data (list[float]): Packed storage array.
    """

    def __init__(self, n, column_heights):
        """
        Create a skyline matrix with given column heights.

        Inputs:
            n (int): Matrix dimension.
            column_heights (list[int]): Height of each column j
                (number of stored elements in column j, including diagonal).
                column_heights[j] >= 1 (at least the diagonal).
                column_heights[j] <= j + 1 (cannot exceed column length).

        Raises:
            ValueError: If column heights are invalid.
        """
        super().__init__(n)

        if len(column_heights) != n:
            raise ValueError(
                f"column_heights length {len(column_heights)} != matrix size {n}"
            )

        self._column_heights = list(column_heights)

        # Validate column heights
        for j in range(n):
            if self._column_heights[j] < 1:
                raise ValueError(
                    f"Column height[{j}] = {self._column_heights[j]} must be >= 1"
                )
            if self._column_heights[j] > j + 1:
                raise ValueError(
                    f"Column height[{j}] = {self._column_heights[j]} "
                    f"exceeds maximum {j + 1}"
                )

        # Compute diagonal pointers
        # _diag_ptr[j] = cumulative sum of column heights up to column j
        self._diag_ptr = [0] * n
        self._diag_ptr[0] = 0
        for j in range(1, n):
            self._diag_ptr[j] = self._diag_ptr[j - 1] + self._column_heights[j - 1]

        # Total storage
        total = self._diag_ptr[n - 1] + self._column_heights[n - 1]
        self._data = [0.0] * total

    @classmethod
    def from_dof_connectivity(cls, n, element_dofs):
        """
        Create a SkylineMatrix by computing column heights from
        element DOF connectivity.

        Inputs:
            n (int): Total number of DOFs.
            element_dofs (list[list[int]]): For each element, a list of
                its global DOF indices.

        Returns:
            SkylineMatrix: Matrix with column heights computed from connectivity.

        Purpose:
            In FEM, the column height for column j is determined by the
            minimum DOF index that shares an element with DOF j.
            This method automates the column height computation.
        """
        # min_row[j] = minimum row index with nonzero entry in column j
        min_row = list(range(n))  # Initialize to diagonal

        for dofs in element_dofs:
            if len(dofs) == 0:
                continue
            min_dof = min(dofs)
            for dof in dofs:
                if min_dof < min_row[dof]:
                    min_row[dof] = min_dof

        # Column height = j - min_row[j] + 1
        column_heights = [j - min_row[j] + 1 for j in range(n)]

        return cls(n, column_heights)

    def _first_row(self, j):
        """
        Get the first (top) row stored in column j.

        Inputs:
            j (int): Column index.

        Returns:
            int: First stored row index for column j.
        """
        return j - self._column_heights[j] + 1

    def get(self, i, j):
        """
        Retrieve element at row i, column j.

        Inputs:
            i (int): Row index (0-based).
            j (int): Column index (0-based).

        Returns:
            float: A[i][j]. Returns 0.0 if outside the skyline profile.
        """
        self._check_bounds(i, j)
        # Exploit symmetry: ensure i <= j (upper triangle)
        if i > j:
            i, j = j, i

        # Check if (i, j) is within the skyline of column j
        first = self._first_row(j)
        if i < first:
            return 0.0  # Outside skyline profile

        # Offset within column j: diagonal is at offset (column_height - 1)
        offset = i - first
        return self._data[self._diag_ptr[j] + offset]

    def set(self, i, j, value):
        """
        Set element at row i, column j.

        Inputs:
            i (int): Row index (0-based).
            j (int): Column index (0-based).
            value (float): Value to store.

        Raises:
            IndexError: If (i, j) is outside the skyline profile.
        """
        self._check_bounds(i, j)
        if i > j:
            i, j = j, i

        first = self._first_row(j)
        if i < first:
            if abs(value) > 1e-15:
                raise IndexError(
                    f"Position ({i}, {j}) is outside skyline profile. "
                    f"Column {j} starts at row {first}."
                )
            return  # Setting zero outside profile is allowed

        offset = i - first
        self._data[self._diag_ptr[j] + offset] = float(value)

    def add(self, i, j, value):
        """
        Add value to element at row i, column j.

        Inputs:
            i (int): Row index (0-based).
            j (int): Column index (0-based).
            value (float): Value to add.

        Purpose:
            Used during FEM assembly.

        Note:
            Adding to a position outside the skyline raises an error
            unless the value is negligible.
        """
        self._check_bounds(i, j)
        if i > j:
            i, j = j, i

        first = self._first_row(j)
        if i < first:
            if abs(value) > 1e-15:
                raise IndexError(
                    f"Cannot add to ({i}, {j}): outside skyline profile."
                )
            return

        offset = i - first
        self._data[self._diag_ptr[j] + offset] += float(value)

    def mat_vec(self, v):
        """
        Skyline matrix-vector product: A * v.

        Inputs:
            v (Vector): Vector of size n.

        Returns:
            Vector: Result vector of size n.

        Note:
            Exploits symmetry and skyline structure for efficiency.
        """
        if not isinstance(v, Vector):
            raise TypeError("Expected Vector")
        if v.size != self._n:
            raise ValueError(f"Size mismatch: matrix {self._n}, vector {v.size}")

        result = Vector(self._n)
        n = self._n

        for j in range(n):
            first = self._first_row(j)
            base = self._diag_ptr[j]
            height = self._column_heights[j]

            # Diagonal contribution
            diag_val = self._data[base + height - 1]
            result._data[j] += diag_val * v._data[j]

            # Off-diagonal contributions (exploit symmetry)
            for offset in range(height - 1):
                row = first + offset
                a_val = self._data[base + offset]
                result._data[row] += a_val * v._data[j]
                result._data[j] += a_val * v._data[row]

        return result

    def ldlt_factorize(self):
        """
        In-place LDL^T factorization (Crout's skyline method).

        Purpose:
            Decomposes the skyline matrix A into L * D * L^T where:
            - L is unit lower triangular (stored as columns of upper
              triangle due to symmetry: L^T is unit upper triangular)
            - D is diagonal

        After factorization, _data stores:
            - Diagonal positions: D[j] values
            - Off-diagonal positions: L^T[i,j] = L[j,i] multipliers

        Raises:
            ArithmeticError: If a zero or negative diagonal is encountered
                (matrix is not positive definite).

        Algorithm:
            Active column method (column-by-column processing).
            For each column j:
                1. Compute L^T[i,j] for each row i in the skyline
                2. Compute D[j] = A[j,j] - sum(L^T[k,j]^2 * D[k])
        """
        n = self._n
        data = self._data
        dp = self._diag_ptr
        ch = self._column_heights

        for j in range(n):
            first_j = j - ch[j] + 1

            # Step 1: Modify off-diagonal elements in column j
            for i in range(first_j, j):
                # Compute dot product of overlapping parts of columns i and j
                first_i = i - ch[i] + 1
                row_start = max(first_i, first_j)

                dot = 0.0
                for k in range(row_start, i):
                    # L^T[k,i] * D[k] * L^T[k,j]
                    offset_ki = k - first_i
                    offset_kj = k - first_j
                    d_k = data[dp[k] + ch[k] - 1]
                    dot += data[dp[i] + offset_ki] * d_k * data[dp[j] + offset_kj]

                offset_ij = i - first_j
                data[dp[j] + offset_ij] -= dot

                # Divide by D[i] to get L^T[i,j]
                d_i = data[dp[i] + ch[i] - 1]
                if abs(d_i) < 1e-30:
                    raise ArithmeticError(
                        f"Zero diagonal D[{i}] during LDL^T factorization. "
                        f"Matrix may not be positive definite."
                    )
                data[dp[j] + offset_ij] /= d_i

            # Step 2: Compute diagonal D[j]
            diag_offset = ch[j] - 1
            d_j = data[dp[j] + diag_offset]
            for i in range(first_j, j):
                offset_ij = i - first_j
                l_ij = data[dp[j] + offset_ij]
                d_i = data[dp[i] + ch[i] - 1]
                d_j -= l_ij * l_ij * d_i

            if d_j <= 0.0:
                raise ArithmeticError(
                    f"Non-positive diagonal D[{j}] = {d_j:.6e} during "
                    f"LDL^T factorization. Matrix is not positive definite."
                )

            data[dp[j] + diag_offset] = d_j

    def ldlt_solve(self, b):
        """
        Solve A*x = b using the LDL^T factorization.

        Inputs:
            b (Vector): Right-hand side vector.

        Returns:
            Vector: Solution vector x.

        Prerequisites:
            ldlt_factorize() must have been called first.

        Algorithm:
            1. Forward substitution: L * y = b
            2. Diagonal scaling: D * z = y
            3. Back substitution: L^T * x = z
        """
        if not isinstance(b, Vector):
            raise TypeError("Expected Vector for right-hand side")
        if b.size != self._n:
            raise ValueError(f"Size mismatch: matrix {self._n}, vector {b.size}")

        n = self._n
        data = self._data
        dp = self._diag_ptr
        ch = self._column_heights

        # Copy b to solution vector
        x = b.copy()

        # Forward substitution: L * y = b
        # L is unit lower triangular; L^T columns stored in skyline
        for j in range(n):
            first_j = j - ch[j] + 1
            for i in range(first_j, j):
                offset = i - first_j
                l_ij = data[dp[j] + offset]
                x._data[j] -= l_ij * x._data[i]

        # Diagonal scaling: z = D^(-1) * y
        for j in range(n):
            d_j = data[dp[j] + ch[j] - 1]
            x._data[j] /= d_j

        # Back substitution: L^T * x = z
        for j in range(n - 1, -1, -1):
            first_j = j - ch[j] + 1
            for i in range(first_j, j):
                offset = i - first_j
                l_ij = data[dp[j] + offset]
                x._data[i] -= l_ij * x._data[j]

        return x

    def copy(self):
        """
        Deep copy of the skyline matrix.

        Returns:
            SkylineMatrix: Independent copy.
        """
        result = SkylineMatrix(self._n, self._column_heights)
        result._data = self._data[:]
        return result

    @property
    def column_heights(self):
        """
        Return list of column heights.

        Returns:
            list[int]: Copy of column heights array.
        """
        return self._column_heights[:]

    @property
    def storage_size(self):
        """
        Return total number of stored elements.

        Returns:
            int: Length of the packed storage array.
        """
        return len(self._data)

    @property
    def profile(self):
        """
        Return the profile (total storage) as a measure of efficiency.

        Returns:
            int: Number of stored elements.
        """
        return len(self._data)
