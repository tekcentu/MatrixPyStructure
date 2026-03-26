"""
banded_matrix.py - Banded Symmetric Matrix Storage and Solver
==============================================================

Purpose:
    Implements banded symmetric matrix storage for matrices where all
    nonzero elements lie within a fixed bandwidth of the diagonal.
    This is common in structural analysis when DOFs are numbered
    optimally (e.g., using Cuthill-McKee ordering).

Storage Scheme:
    For a symmetric matrix of size n with half-bandwidth hbw, only the
    upper band is stored. The storage uses a 2D array of dimensions
    n x (hbw + 1), where:
        - Column 0 stores the diagonal.
        - Column k stores the k-th super-diagonal.

    Element A[i][j] (with j >= i) is stored at:
        _data[i][j - i]   (if j - i <= hbw, else zero)

    Example for a 5x5 matrix with half-bandwidth 2:
        Original:
            [a00  a01  a02   0    0 ]
            [a01  a11  a12  a13   0 ]
            [a02  a12  a22  a23  a24]
            [ 0   a13  a23  a33  a34]
            [ 0    0   a24  a34  a44]

        Banded storage (n=5, hbw=2, stored as n x 3 array):
            Row 0: [a00, a01, a02]
            Row 1: [a11, a12, a13]
            Row 2: [a22, a23, a24]
            Row 3: [a33, a34,  0 ]   <- padded with zero
            Row 4: [a44,  0,   0 ]   <- padded with zeros

    Total storage = n * (hbw + 1) = 5 * 3 = 15
    vs. full matrix = 25 elements.
    vs. upper triangle = 15 elements.
    Advantage grows for large n with small hbw relative to n.

Comparison with Skyline:
    - Banded storage assumes a FIXED bandwidth for all columns.
    - Skyline storage adapts to the actual nonzero pattern per column.
    - Banded is simpler to implement and may be faster for uniformly
      banded matrices (e.g., regular meshes).
    - Skyline is more storage-efficient for irregular meshes.

Assumptions:
    - Matrix is symmetric positive-definite (SPD).
    - Half-bandwidth hbw is known a priori.
    - Used for structural stiffness matrices with optimal DOF numbering.

Units:
    No specific units; caller ensures consistency.
"""

from matrix.base_matrix import BaseMatrix
from matrix.vector import Vector


class BandedMatrix(BaseMatrix):
    """
    Symmetric banded matrix stored as upper band.

    Attributes:
        _hbw (int): Half-bandwidth (number of super-diagonals stored).
        _data (list[list[float]]): Storage array of size n x (hbw + 1).
            _data[i][k] stores A[i][i+k] for k = 0..hbw.
    """

    def __init__(self, n, half_bandwidth):
        """
        Create an n x n symmetric banded zero matrix.

        Inputs:
            n (int): Matrix dimension.
            half_bandwidth (int): Number of super-diagonals (hbw >= 0).
                hbw = 0 means diagonal matrix.
                hbw = n-1 means full upper triangle.

        Outputs:
            None (constructor).

        Raises:
            ValueError: If half_bandwidth < 0 or > n-1.
        """
        super().__init__(n)

        if half_bandwidth < 0:
            raise ValueError(
                f"Half-bandwidth must be >= 0, got {half_bandwidth}"
            )
        if half_bandwidth >= n:
            half_bandwidth = n - 1  # Cap at maximum possible

        self._hbw = half_bandwidth
        # Storage: n rows, each with (hbw + 1) columns
        self._data = [[0.0] * (self._hbw + 1) for _ in range(n)]

    @classmethod
    def from_dof_connectivity(cls, n, element_dofs):
        """
        Create a BandedMatrix by computing the half-bandwidth from
        element DOF connectivity.

        Inputs:
            n (int): Total number of DOFs.
            element_dofs (list[list[int]]): For each element, a list of
                its global DOF indices.

        Returns:
            BandedMatrix: Matrix with half-bandwidth determined from
                the maximum DOF index difference within any element.

        Purpose:
            The half-bandwidth equals the maximum difference between
            any two DOF indices that appear in the same element:
            hbw = max over all elements of (max(dofs) - min(dofs)).
        """
        hbw = 0
        for dofs in element_dofs:
            if len(dofs) > 0:
                diff = max(dofs) - min(dofs)
                if diff > hbw:
                    hbw = diff
        return cls(n, hbw)

    @property
    def half_bandwidth(self):
        """
        Return the half-bandwidth.

        Returns:
            int: Number of super-diagonals stored.
        """
        return self._hbw

    @property
    def bandwidth(self):
        """
        Return the full bandwidth (2 * hbw + 1).

        Returns:
            int: Total bandwidth including diagonal.
        """
        return 2 * self._hbw + 1

    @property
    def storage_size(self):
        """
        Return total number of stored elements.

        Returns:
            int: n * (hbw + 1).
        """
        return self._n * (self._hbw + 1)

    def get(self, i, j):
        """
        Retrieve element at row i, column j.

        Inputs:
            i (int): Row index (0-based).
            j (int): Column index (0-based).

        Returns:
            float: A[i][j]. Returns 0.0 if outside the band.
        """
        self._check_bounds(i, j)
        # Exploit symmetry: ensure i <= j
        if i > j:
            i, j = j, i

        diff = j - i
        if diff > self._hbw:
            return 0.0  # Outside band
        return self._data[i][diff]

    def set(self, i, j, value):
        """
        Set element at row i, column j.

        Inputs:
            i (int): Row index (0-based).
            j (int): Column index (0-based).
            value (float): Value to store.

        Raises:
            IndexError: If (i, j) is outside the band and value is nonzero.
        """
        self._check_bounds(i, j)
        if i > j:
            i, j = j, i

        diff = j - i
        if diff > self._hbw:
            if abs(value) > 1e-15:
                raise IndexError(
                    f"Position ({i}, {j}) is outside band (hbw={self._hbw}). "
                    f"Difference {diff} > {self._hbw}."
                )
            return  # Setting zero outside band is allowed
        self._data[i][diff] = float(value)

    def add(self, i, j, value):
        """
        Add value to element at row i, column j.

        Inputs:
            i (int): Row index (0-based).
            j (int): Column index (0-based).
            value (float): Value to add.

        Purpose:
            Used during FEM assembly of banded stiffness matrices.

        Raises:
            IndexError: If (i, j) is outside the band and value is nonzero.
        """
        self._check_bounds(i, j)
        if i > j:
            i, j = j, i

        diff = j - i
        if diff > self._hbw:
            if abs(value) > 1e-15:
                raise IndexError(
                    f"Cannot add to ({i}, {j}): outside band (hbw={self._hbw})."
                )
            return
        self._data[i][diff] += float(value)

    def mat_vec(self, v):
        """
        Banded symmetric matrix-vector product: A * v.

        Inputs:
            v (Vector): Vector of size n.

        Returns:
            Vector: Result vector of size n.

        Note:
            Exploits banded structure and symmetry: only processes
            elements within the band, using each off-diagonal element
            for both its row and symmetric column contribution.
        """
        if not isinstance(v, Vector):
            raise TypeError("Expected Vector")
        if v.size != self._n:
            raise ValueError(f"Size mismatch: matrix {self._n}, vector {v.size}")

        result = Vector(self._n)
        n = self._n
        hbw = self._hbw

        for i in range(n):
            # Diagonal contribution
            result._data[i] += self._data[i][0] * v._data[i]

            # Off-diagonal contributions within the band
            for k in range(1, min(hbw + 1, n - i)):
                j = i + k
                a_ij = self._data[i][k]
                result._data[i] += a_ij * v._data[j]   # Row i
                result._data[j] += a_ij * v._data[i]   # Symmetric: row j

        return result

    def ldlt_factorize(self):
        """
        In-place LDL^T factorization for banded symmetric matrix.

        Purpose:
            Decomposes A = L * D * L^T where L is unit lower triangular
            and D is diagonal. After factorization:
            - _data[i][0] stores D[i] (diagonal of D).
            - _data[i][k] for k>0 stores L^T[i, i+k] (multipliers).

        Raises:
            ArithmeticError: If a non-positive diagonal is encountered
                (matrix is not positive definite).

        Algorithm:
            Column-by-column Crout factorization restricted to the band.
            For each column j:
                1. For each row i in band above j, compute L^T[i,j].
                2. Compute D[j] from the remaining diagonal.
        """
        n = self._n
        hbw = self._hbw
        data = self._data

        for j in range(n):
            # First row in band for column j
            first_i = max(0, j - hbw)

            # Step 1: Process off-diagonal entries in column j
            for i in range(first_i, j):
                # A[i][j] is stored at data[i][j-i]
                k_ij = j - i  # offset in row i

                # Compute dot product of overlapping band portions
                dot = 0.0
                for m in range(first_i, i):
                    k_mi = i - m  # offset for A[m][i]
                    k_mj = j - m  # offset for A[m][j]
                    if k_mi <= hbw and k_mj <= hbw:
                        dot += data[m][k_mi] * data[m][0] * data[m][k_mj]

                data[i][k_ij] -= dot

                # Divide by D[i] to get L^T[i,j]
                d_i = data[i][0]
                if abs(d_i) < 1e-30:
                    raise ArithmeticError(
                        f"Zero diagonal D[{i}] during banded LDL^T. "
                        f"Matrix may not be positive definite."
                    )
                data[i][k_ij] /= d_i

            # Step 2: Compute D[j]
            d_j = data[j][0]
            for i in range(first_i, j):
                k_ij = j - i
                if k_ij <= hbw:
                    l_ij = data[i][k_ij]
                    d_j -= l_ij * l_ij * data[i][0]

            if d_j <= 0.0:
                raise ArithmeticError(
                    f"Non-positive diagonal D[{j}] = {d_j:.6e} during "
                    f"banded LDL^T. Matrix is not positive definite."
                )
            data[j][0] = d_j

    def ldlt_solve(self, b):
        """
        Solve A*x = b using the banded LDL^T factorization.

        Inputs:
            b (Vector): Right-hand side vector of size n.

        Returns:
            Vector: Solution vector x of size n.

        Prerequisites:
            ldlt_factorize() must have been called first.

        Algorithm:
            1. Forward substitution: L * y = b (within band).
            2. Diagonal scaling: D * z = y.
            3. Back substitution: L^T * x = z (within band).
        """
        if not isinstance(b, Vector):
            raise TypeError("Expected Vector for right-hand side")
        if b.size != self._n:
            raise ValueError(f"Size mismatch: matrix {self._n}, vector {b.size}")

        n = self._n
        hbw = self._hbw
        data = self._data

        # Copy b to solution vector
        x = b.copy()

        # Forward substitution: L * y = b
        for j in range(n):
            first_i = max(0, j - hbw)
            for i in range(first_i, j):
                k_ij = j - i
                if k_ij <= hbw:
                    x._data[j] -= data[i][k_ij] * x._data[i]

        # Diagonal scaling: z = D^(-1) * y
        for j in range(n):
            x._data[j] /= data[j][0]

        # Back substitution: L^T * x = z
        for j in range(n - 1, -1, -1):
            first_i = max(0, j - hbw)
            for i in range(first_i, j):
                k_ij = j - i
                if k_ij <= hbw:
                    x._data[i] -= data[i][k_ij] * x._data[j]

        return x

    def copy(self):
        """
        Deep copy of the banded matrix.

        Returns:
            BandedMatrix: Independent copy with same dimensions and data.
        """
        result = BandedMatrix(self._n, self._hbw)
        for i in range(self._n):
            result._data[i] = self._data[i][:]
        return result

    def get_diagonal(self):
        """
        Extract the diagonal as a Vector.

        Returns:
            Vector: Diagonal elements of the matrix.
        """
        diag = Vector(self._n)
        for i in range(self._n):
            diag[i] = self._data[i][0]
        return diag
