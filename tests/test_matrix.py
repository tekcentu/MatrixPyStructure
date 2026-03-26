"""
test_matrix.py - Unit Tests for Matrix Library
================================================

Purpose:
    Comprehensive unit tests for all matrix types: Vector, DenseMatrix,
    SymmetricMatrix, BandedMatrix, SkylineMatrix, and Solver.

Running:
    python -m pytest tests/test_matrix.py -v
    or
    python tests/test_matrix.py
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from matrix.vector import Vector
from matrix.dense_matrix import DenseMatrix
from matrix.symmetric_matrix import SymmetricMatrix
from matrix.banded_matrix import BandedMatrix
from matrix.skyline_matrix import SkylineMatrix
from matrix.solver import Solver


def test_vector_basic():
    """Test Vector creation and basic operations."""
    v = Vector(3)
    assert v.size == 3
    assert v[0] == 0.0

    v[0] = 1.0
    v[1] = 2.0
    v[2] = 3.0
    assert v[0] == 1.0
    assert v[2] == 3.0

    v2 = Vector([4.0, 5.0, 6.0])
    assert v2.size == 3
    assert v2[0] == 4.0


def test_vector_arithmetic():
    """Test Vector addition, subtraction, scalar multiplication."""
    v1 = Vector([1.0, 2.0, 3.0])
    v2 = Vector([4.0, 5.0, 6.0])

    v_add = v1 + v2
    assert abs(v_add[0] - 5.0) < 1e-12
    assert abs(v_add[1] - 7.0) < 1e-12
    assert abs(v_add[2] - 9.0) < 1e-12

    v_sub = v2 - v1
    assert abs(v_sub[0] - 3.0) < 1e-12
    assert abs(v_sub[1] - 3.0) < 1e-12

    v_scale = v1 * 2.0
    assert abs(v_scale[0] - 2.0) < 1e-12
    assert abs(v_scale[2] - 6.0) < 1e-12

    # Dot product
    assert abs(v1.dot(v2) - 32.0) < 1e-12  # 4+10+18

    # Norm
    assert abs(v1.norm() - math.sqrt(14.0)) < 1e-12


def test_dense_matrix_basic():
    """Test DenseMatrix creation and access."""
    m = DenseMatrix(3)
    assert m.size == 3
    assert m.get(0, 0) == 0.0

    m.set(0, 0, 1.0)
    m.set(1, 2, 5.0)
    assert m.get(0, 0) == 1.0
    assert m.get(1, 2) == 5.0

    m.add(0, 0, 2.0)
    assert abs(m.get(0, 0) - 3.0) < 1e-12


def test_dense_matrix_operations():
    """Test DenseMatrix arithmetic and mat-vec."""
    A = DenseMatrix.from_list([[1, 2], [3, 4]])
    B = DenseMatrix.from_list([[5, 6], [7, 8]])

    C = A + B
    assert abs(C.get(0, 0) - 6.0) < 1e-12
    assert abs(C.get(1, 1) - 12.0) < 1e-12

    v = Vector([1.0, 1.0])
    Av = A.mat_vec(v)
    assert abs(Av[0] - 3.0) < 1e-12  # 1+2
    assert abs(Av[1] - 7.0) < 1e-12  # 3+4

    AB = A.mat_mul(B)
    assert abs(AB.get(0, 0) - 19.0) < 1e-12  # 1*5 + 2*7
    assert abs(AB.get(0, 1) - 22.0) < 1e-12  # 1*6 + 2*8
    assert abs(AB.get(1, 0) - 43.0) < 1e-12  # 3*5 + 4*7
    assert abs(AB.get(1, 1) - 50.0) < 1e-12  # 3*6 + 4*8

    At = A.transpose()
    assert abs(At.get(0, 1) - 3.0) < 1e-12
    assert abs(At.get(1, 0) - 2.0) < 1e-12


def test_symmetric_matrix():
    """Test SymmetricMatrix storage and operations."""
    S = SymmetricMatrix(3)
    S.set(0, 0, 4.0)
    S.set(0, 1, 1.0)
    S.set(0, 2, 2.0)
    S.set(1, 1, 5.0)
    S.set(1, 2, 3.0)
    S.set(2, 2, 6.0)

    # Symmetry check
    assert abs(S.get(1, 0) - S.get(0, 1)) < 1e-12
    assert abs(S.get(2, 0) - S.get(0, 2)) < 1e-12
    assert abs(S.get(2, 1) - S.get(1, 2)) < 1e-12

    # Storage efficiency
    assert S.storage_size == 6  # 3*4/2

    # Mat-vec
    v = Vector([1.0, 1.0, 1.0])
    Sv = S.mat_vec(v)
    assert abs(Sv[0] - 7.0) < 1e-12   # 4+1+2
    assert abs(Sv[1] - 9.0) < 1e-12   # 1+5+3
    assert abs(Sv[2] - 11.0) < 1e-12  # 2+3+6


def test_banded_matrix_basic():
    """Test BandedMatrix creation, access, and symmetry."""
    # 3x3 tridiagonal: half-bandwidth = 1
    B = BandedMatrix(3, 1)

    B.set(0, 0, 4.0)
    B.set(0, 1, 1.0)
    B.set(1, 1, 4.0)
    B.set(1, 2, 1.0)
    B.set(2, 2, 4.0)

    assert abs(B.get(0, 0) - 4.0) < 1e-12
    assert abs(B.get(0, 1) - 1.0) < 1e-12
    assert abs(B.get(1, 0) - 1.0) < 1e-12  # symmetry
    assert abs(B.get(0, 2) - 0.0) < 1e-12  # outside band

    # Storage: 3 * 2 = 6
    assert B.storage_size == 6
    assert B.half_bandwidth == 1
    assert B.bandwidth == 3


def test_banded_matrix_matvec():
    """Test BandedMatrix matrix-vector product."""
    # Same tridiagonal as skyline test
    B = BandedMatrix(3, 1)
    B.set(0, 0, 4.0); B.set(0, 1, 1.0)
    B.set(1, 1, 4.0); B.set(1, 2, 1.0)
    B.set(2, 2, 4.0)

    v = Vector([1.0, 2.0, 3.0])
    Bv = B.mat_vec(v)
    # [4  1  0] [1]   [6 ]
    # [1  4  1] [2] = [12]
    # [0  1  4] [3]   [14]
    assert abs(Bv[0] - 6.0) < 1e-12
    assert abs(Bv[1] - 12.0) < 1e-12
    assert abs(Bv[2] - 14.0) < 1e-12


def test_banded_from_connectivity():
    """Test BandedMatrix creation from DOF connectivity."""
    # 4 DOFs, two elements: [0,1,2] and [1,2,3]
    B = BandedMatrix.from_dof_connectivity(4, [[0, 1, 2], [1, 2, 3]])
    # max diff = max(2-0, 3-1) = 2
    assert B.half_bandwidth == 2


def test_banded_solver():
    """Test banded LDL^T solver with known solution."""
    B = BandedMatrix(3, 1)
    B.set(0, 0, 4.0); B.set(0, 1, 1.0)
    B.set(1, 1, 4.0); B.set(1, 2, 1.0)
    B.set(2, 2, 4.0)

    x_exact = Vector([1.0, 2.0, 3.0])
    b = B.mat_vec(x_exact)

    x = Solver.solve_banded(B, b)
    for i in range(3):
        assert abs(x[i] - x_exact[i]) < 1e-8, \
            f"x[{i}] = {x[i]}, expected {x_exact[i]}"


def test_banded_solver_larger():
    """Test banded solver with 4x4 system, bandwidth 2."""
    B = BandedMatrix(4, 2)
    vals = [[10, 2, 3, 0],
            [2, 8, 1, 2],
            [3, 1, 6, 1],
            [0, 2, 1, 5]]
    for i in range(4):
        for j in range(i, 4):
            if abs(vals[i][j]) > 0:
                B.set(i, j, vals[i][j])

    x_exact = Vector([1.0, 2.0, 3.0, 4.0])
    b = B.mat_vec(x_exact)

    x = Solver.solve_banded(B, b)
    for i in range(4):
        assert abs(x[i] - x_exact[i]) < 1e-8, \
            f"x[{i}] = {x[i]}, expected {x_exact[i]}"


def test_skyline_matrix_basic():
    """Test SkylineMatrix creation and access."""
    # 3x3 tridiagonal: column heights [1, 2, 2]
    sky = SkylineMatrix(3, [1, 2, 2])

    sky.set(0, 0, 4.0)
    sky.set(0, 1, 1.0)
    sky.set(1, 1, 4.0)
    sky.set(1, 2, 1.0)
    sky.set(2, 2, 4.0)

    assert abs(sky.get(0, 0) - 4.0) < 1e-12
    assert abs(sky.get(0, 1) - 1.0) < 1e-12
    assert abs(sky.get(1, 0) - 1.0) < 1e-12  # symmetry
    assert abs(sky.get(0, 2) - 0.0) < 1e-12  # outside skyline

    # Storage: 1 + 2 + 2 = 5 elements
    assert sky.storage_size == 5


def test_skyline_from_connectivity():
    """Test SkylineMatrix creation from DOF connectivity."""
    # 4 DOFs, two elements: [0,1,2] and [1,2,3]
    sky = SkylineMatrix.from_dof_connectivity(4, [[0, 1, 2], [1, 2, 3]])

    # Column 0: min DOF = 0, height = 1
    # Column 1: min DOF = 0, height = 2
    # Column 2: min DOF = 0, height = 3
    # Column 3: min DOF = 1, height = 3
    expected_heights = [1, 2, 3, 3]
    assert sky.column_heights == expected_heights


def test_skyline_solver():
    """Test LDL^T solver with known solution."""
    sky = SkylineMatrix(3, [1, 2, 2])
    sky.set(0, 0, 4.0)
    sky.set(0, 1, 1.0)
    sky.set(1, 1, 4.0)
    sky.set(1, 2, 1.0)
    sky.set(2, 2, 4.0)

    # x = [1, 2, 3], b = A*x
    x_exact = Vector([1.0, 2.0, 3.0])
    b = sky.mat_vec(x_exact)

    x = Solver.solve_skyline(sky, b)
    for i in range(3):
        assert abs(x[i] - x_exact[i]) < 1e-8, \
            f"x[{i}] = {x[i]}, expected {x_exact[i]}"


def test_dense_solver():
    """Test dense Gaussian elimination solver."""
    A = DenseMatrix.from_list([[2, 1, 0], [1, 3, 1], [0, 1, 2]])
    x_exact = Vector([1.0, 2.0, 3.0])
    b = A.mat_vec(x_exact)

    x = Solver.solve_dense(A, b)
    for i in range(3):
        assert abs(x[i] - x_exact[i]) < 1e-8, \
            f"x[{i}] = {x[i]}, expected {x_exact[i]}"


def test_symmetric_solver():
    """Test symmetric matrix solver."""
    S = SymmetricMatrix(3)
    S.set(0, 0, 4.0)
    S.set(0, 1, 1.0)
    S.set(1, 1, 4.0)
    S.set(1, 2, 1.0)
    S.set(2, 2, 4.0)

    x_exact = Vector([1.0, 2.0, 3.0])
    b = S.mat_vec(x_exact)

    x = Solver.solve_symmetric(S, b)
    for i in range(3):
        assert abs(x[i] - x_exact[i]) < 1e-8


def run_all_tests():
    """Run all matrix tests."""
    tests = [
        test_vector_basic,
        test_vector_arithmetic,
        test_dense_matrix_basic,
        test_dense_matrix_operations,
        test_symmetric_matrix,
        test_banded_matrix_basic,
        test_banded_matrix_matvec,
        test_banded_from_connectivity,
        test_banded_solver,
        test_banded_solver_larger,
        test_skyline_matrix_basic,
        test_skyline_from_connectivity,
        test_skyline_solver,
        test_dense_solver,
        test_symmetric_solver,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  [PASS] {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed += 1

    print(f"\nMatrix Tests: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    print("Running Matrix Library Tests...")
    run_all_tests()
