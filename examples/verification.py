"""
verification.py - Matrix Library and Structural Analysis Verification
=====================================================================

Purpose:
    Comprehensive verification of the matrix library and structural
    analysis software against known analytical solutions.

Tests:
    1. Matrix operations (dense, symmetric, skyline)
    2. Linear solver verification (known solution systems)
    3. Single truss element stiffness verification
    4. 2-bar truss hand calculation comparison
    5. Simple cantilever beam verification
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
from matrix.vector import Vector
from matrix.dense_matrix import DenseMatrix
from matrix.symmetric_matrix import SymmetricMatrix
from matrix.skyline_matrix import SkylineMatrix
from matrix.solver import Solver
from structure.node import Node
from structure.material import Material
from structure.section import Section
from structure.element import Truss2D, Frame2D
from structure.load import NodalLoad, LoadCase
from structure.boundary import BoundaryCondition
from structure.model import StructuralModel
from structure.analysis import StaticAnalysis


def check(name, computed, expected, tol=1e-6):
    """
    Verify a computed value against an expected value.

    Inputs:
        name (str): Test name.
        computed (float): Computed result.
        expected (float): Expected result.
        tol (float): Tolerance for comparison.

    Returns:
        bool: True if passed.
    """
    error = abs(computed - expected)
    passed = error < tol
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: computed={computed:.8e}, "
          f"expected={expected:.8e}, error={error:.2e}")
    return passed


def verify_matrix_operations():
    """
    Verify basic matrix and vector operations.
    """
    print("\n" + "=" * 60)
    print("TEST 1: Matrix and Vector Operations")
    print("=" * 60)
    all_pass = True

    # Vector operations
    v1 = Vector([1.0, 2.0, 3.0])
    v2 = Vector([4.0, 5.0, 6.0])
    v_sum = v1 + v2
    all_pass &= check("Vector add [0]", v_sum[0], 5.0)
    all_pass &= check("Vector add [2]", v_sum[2], 9.0)
    all_pass &= check("Vector dot", v1.dot(v2), 32.0)  # 4+10+18
    all_pass &= check("Vector norm", v1.norm(), math.sqrt(14.0))

    # Dense matrix
    A = DenseMatrix.from_list([[4, 2], [2, 3]])
    v = Vector([1.0, 2.0])
    Av = A.mat_vec(v)
    all_pass &= check("Dense mat_vec [0]", Av[0], 8.0)  # 4+4
    all_pass &= check("Dense mat_vec [1]", Av[1], 8.0)  # 2+6

    # Matrix multiplication
    B = DenseMatrix.from_list([[1, 0], [0, 1]])
    AB = A.mat_mul(B)
    all_pass &= check("Dense mat_mul identity [0,0]", AB.get(0, 0), 4.0)
    all_pass &= check("Dense mat_mul identity [1,1]", AB.get(1, 1), 3.0)

    # Symmetric matrix
    S = SymmetricMatrix(3)
    S.set(0, 0, 4.0); S.set(0, 1, 1.0); S.set(0, 2, 2.0)
    S.set(1, 1, 5.0); S.set(1, 2, 3.0)
    S.set(2, 2, 6.0)
    all_pass &= check("Symmetric get(1,0) = get(0,1)", S.get(1, 0), S.get(0, 1))
    all_pass &= check("Symmetric storage", S.storage_size, 6)  # 3*4/2

    # Symmetric mat_vec
    v3 = Vector([1.0, 1.0, 1.0])
    Sv = S.mat_vec(v3)
    all_pass &= check("Sym mat_vec [0]", Sv[0], 7.0)  # 4+1+2
    all_pass &= check("Sym mat_vec [1]", Sv[1], 9.0)  # 1+5+3
    all_pass &= check("Sym mat_vec [2]", Sv[2], 11.0)  # 2+3+6

    return all_pass


def verify_skyline_solver():
    """
    Verify skyline storage and LDL^T solver with a known system.

    Test system:
        [4  1  0] [x0]   [6 ]
        [1  4  1] [x1] = [12]
        [0  1  4] [x2]   [14]

    Solution: x = [1, 2, 3] (verified: A*x = [4+2, 1+8+3, 2+12] = [6, 12, 14]).
    """
    print("\n" + "=" * 60)
    print("TEST 2: Skyline Solver (LDL^T)")
    print("=" * 60)
    all_pass = True

    # Column heights: col 0 = [4] -> h=1, col 1 = [1,4] -> h=2, col 2 = [0,1,4] -> h=2
    sky = SkylineMatrix(3, [1, 2, 2])
    sky.set(0, 0, 4.0); sky.set(0, 1, 1.0)
    sky.set(1, 1, 4.0); sky.set(1, 2, 1.0)
    sky.set(2, 2, 4.0)

    # Compute b = A * x_exact to ensure consistency
    x_exact_3 = Vector([1.0, 2.0, 3.0])
    b = sky.mat_vec(x_exact_3)
    x = Solver.solve_skyline(sky, b)

    all_pass &= check("Skyline x[0]", x[0], 1.0)
    all_pass &= check("Skyline x[1]", x[1], 2.0, tol=1e-6)
    all_pass &= check("Skyline x[2]", x[2], 3.0, tol=1e-6)

    # Verify with dense solver for comparison
    A = DenseMatrix.from_list([[4, 1, 0], [1, 4, 1], [0, 1, 4]])
    x_dense = Solver.solve_dense(A, b)
    all_pass &= check("Dense x[0]", x_dense[0], 1.0)
    all_pass &= check("Dense x[1]", x_dense[1], 2.0, tol=1e-6)
    all_pass &= check("Dense x[2]", x_dense[2], 3.0, tol=1e-6)

    # Larger system (4x4)
    print("\n  --- 4x4 system ---")
    A4 = SkylineMatrix(4, [1, 2, 3, 4])
    vals = [[10, 2, 3, 1],
            [2, 8, 1, 2],
            [3, 1, 6, 1],
            [1, 2, 1, 5]]
    for i in range(4):
        for j in range(i, 4):
            A4.set(i, j, vals[i][j])

    # b = A * [1, 2, 3, 4]
    x_exact = Vector([1.0, 2.0, 3.0, 4.0])
    b4 = A4.mat_vec(x_exact)
    x4 = Solver.solve_skyline(A4, b4)

    for i in range(4):
        all_pass &= check(f"4x4 x[{i}]", x4[i], x_exact[i], tol=1e-6)

    return all_pass


def verify_truss_element_stiffness():
    """
    Verify single truss element stiffness matrix.

    Horizontal element: Node 0 (0,0) -> Node 1 (3,0)
    E = 200, A = 0.5, L = 3
    EA/L = 200 * 0.5 / 3 = 33.333

    Expected k_global (horizontal member, angle=0):
        EA/L * [ 1  0 -1  0]
               [ 0  0  0  0]
               [-1  0  1  0]
               [ 0  0  0  0]
    """
    print("\n" + "=" * 60)
    print("TEST 3: Truss Element Stiffness Matrix")
    print("=" * 60)
    all_pass = True

    n0 = Node(0, 0.0, 0.0)
    n1 = Node(1, 3.0, 0.0)
    mat = Material(0, 200.0)
    sec = Section(0, 0.5)
    elem = Truss2D(0, n0, n1, mat, sec)

    ea_l = 200.0 * 0.5 / 3.0  # = 33.333...

    k = elem.global_stiffness()
    all_pass &= check("k[0,0]", k.get(0, 0), ea_l)
    all_pass &= check("k[0,1]", k.get(0, 1), 0.0)
    all_pass &= check("k[0,2]", k.get(0, 2), -ea_l)
    all_pass &= check("k[2,2]", k.get(2, 2), ea_l)
    all_pass &= check("k[1,1]", k.get(1, 1), 0.0)

    # Inclined element at 45 degrees: Node 0 (0,0) -> Node 2 (1,1)
    print("\n  --- 45-degree element ---")
    n2 = Node(2, 1.0, 1.0)
    elem2 = Truss2D(1, n0, n2, mat, sec)
    L2 = math.sqrt(2.0)
    ea_l2 = 200.0 * 0.5 / L2
    c = 1.0 / math.sqrt(2.0)
    s = 1.0 / math.sqrt(2.0)
    cc = c * c * ea_l2
    ss = s * s * ea_l2
    cs = c * s * ea_l2

    k2 = elem2.global_stiffness()
    all_pass &= check("k45[0,0] = cc", k2.get(0, 0), cc)
    all_pass &= check("k45[0,1] = cs", k2.get(0, 1), cs)
    all_pass &= check("k45[1,1] = ss", k2.get(1, 1), ss)

    return all_pass


def verify_two_bar_truss():
    """
    Verify a simple 2-bar truss against hand calculation.

    Problem:
        Two bars meeting at Node 2, loaded vertically.

        Node 0 (0,0) -- pinned
        Node 1 (4,0) -- pinned
        Node 2 (2,2) -- loaded, Fy = -10

        Element 0: Node 0 -> Node 2
        Element 1: Node 1 -> Node 2

    Both elements: E = 100, A = 1.0

    Hand calculation:
        L0 = L1 = sqrt(8) = 2*sqrt(2) = 2.8284
        cos0 = 2/L0 = 1/sqrt(2), sin0 = 2/L0 = 1/sqrt(2)
        cos1 = -2/L1 = -1/sqrt(2), sin1 = 2/L1 = 1/sqrt(2)

        By symmetry: both bars have same magnitude of axial force.
        Sum Fy at Node 2: 2 * F * sin(45) = 10 => F = 10/(2*sin45) = 7.071
        Both bars in compression: F = -7.071 kN
    """
    print("\n" + "=" * 60)
    print("TEST 4: Two-Bar Truss (Hand Calculation)")
    print("=" * 60)
    all_pass = True

    model = StructuralModel(analysis_type="truss")

    n0 = Node(0, 0.0, 0.0)
    n1 = Node(1, 4.0, 0.0)
    n2 = Node(2, 2.0, 2.0)
    model.add_node(n0)
    model.add_node(n1)
    model.add_node(n2)

    mat = Material(0, 100.0)
    sec = Section(0, 1.0)

    e0 = Truss2D(0, n0, n2, mat, sec)
    e1 = Truss2D(1, n1, n2, mat, sec)
    model.add_element(e0)
    model.add_element(e1)

    # Pin both supports
    model.add_boundary_condition(BoundaryCondition(0, 0))
    model.add_boundary_condition(BoundaryCondition(0, 1))
    model.add_boundary_condition(BoundaryCondition(1, 0))
    model.add_boundary_condition(BoundaryCondition(1, 1))

    lc = LoadCase("Test")
    lc.add_nodal_load(NodalLoad(2, fy=-10.0))
    model.add_load_case(lc)

    analysis = StaticAnalysis(model)
    results = analysis.solve(verbose=False)

    # Expected: by symmetry, ux at node 2 = 0, uy at node 2 = negative
    ux2 = results.displacements[n2.dof_indices[0]]
    uy2 = results.displacements[n2.dof_indices[1]]

    all_pass &= check("Node 2 ux (symmetry -> 0)", ux2, 0.0, tol=1e-6)
    print(f"  Node 2 uy = {uy2:.8e}")

    # Expected axial forces: compression (negative)
    # F = P / (2 * sin(theta)) = 10 / (2 * sin(45)) = 7.0711
    expected_force = -10.0 / (2.0 * math.sin(math.atan2(2, 2)))
    f0 = results.member_forces[0]
    f1 = results.member_forces[1]
    all_pass &= check("Element 0 axial force", f0, expected_force, tol=1e-3)
    all_pass &= check("Element 1 axial force", f1, expected_force, tol=1e-3)

    # Check reactions
    r0 = results.reactions[0]
    r1 = results.reactions[1]
    all_pass &= check("Sum Ry = 10", r0['fy'] + r1['fy'], 10.0, tol=1e-3)
    all_pass &= check("Ry0 = Ry1 (symmetry)", r0['fy'], r1['fy'], tol=1e-3)

    eq = analysis.verify_equilibrium()
    all_pass &= check("Global equilibrium Fx", eq['sum_fx'], 0.0, tol=1e-3)
    all_pass &= check("Global equilibrium Fy", eq['sum_fy'], 0.0, tol=1e-3)

    return all_pass


def verify_cantilever_beam():
    """
    Verify a simple cantilever beam against analytical solution.

    Problem:
        Fixed-free beam with point load at free end.

        Node 0 (0,0) -- fixed
        Node 1 (L,0) -- free, Fy = -P

        L = 4 m, E = 200e6, A = 0.01, I = 1e-4, P = 10 kN

    Analytical solution:
        Tip deflection: delta = P*L^3/(3*E*I) = 10*64/(3*200e6*1e-4)
                       = 640/60000 = 0.01067 m
        Tip rotation: theta = P*L^2/(2*E*I) = 10*16/(2*200e6*1e-4)
                     = 160/40000 = 0.004 rad
        Reactions: Rx=0, Ry=10, Mz = P*L = 40 kN*m
    """
    print("\n" + "=" * 60)
    print("TEST 5: Cantilever Beam (Analytical Solution)")
    print("=" * 60)
    all_pass = True

    L = 4.0
    E = 200e6
    A = 0.01
    I = 1e-4
    P = 10.0

    model = StructuralModel(analysis_type="frame")
    n0 = Node(0, 0.0, 0.0)
    n1 = Node(1, L, 0.0)
    model.add_node(n0)
    model.add_node(n1)

    mat = Material(0, E)
    sec = Section(0, A, I)
    elem = Frame2D(0, n0, n1, mat, sec)
    model.add_element(elem)

    # Fixed at Node 0
    model.add_boundary_condition(BoundaryCondition(0, 0))
    model.add_boundary_condition(BoundaryCondition(0, 1))
    model.add_boundary_condition(BoundaryCondition(0, 2))

    lc = LoadCase("Point load")
    lc.add_nodal_load(NodalLoad(1, fy=-P))
    model.add_load_case(lc)

    analysis = StaticAnalysis(model)
    results = analysis.solve(verbose=False)

    # Expected deflection
    delta_expected = -P * L**3 / (3.0 * E * I)
    theta_expected = -P * L**2 / (2.0 * E * I)

    uy1 = results.displacements[n1.dof_indices[1]]
    rz1 = results.displacements[n1.dof_indices[2]]

    all_pass &= check("Tip deflection", uy1, delta_expected, tol=1e-8)
    all_pass &= check("Tip rotation", rz1, theta_expected, tol=1e-8)

    # Expected reactions
    r = results.reactions[0]
    all_pass &= check("Reaction Ry", r['fy'], P, tol=1e-3)
    all_pass &= check("Reaction Mz", r['mz'], P * L, tol=1e-3)

    print(f"\n  Analytical: delta = PL^3/(3EI) = {delta_expected:.8e} m")
    print(f"  Computed:   delta = {uy1:.8e} m")
    print(f"  Analytical: theta = PL^2/(2EI) = {theta_expected:.8e} rad")
    print(f"  Computed:   theta = {rz1:.8e} rad")

    return all_pass


def run_all_verifications():
    """Run all verification tests and report summary."""
    print("=" * 60)
    print("COMPLETE VERIFICATION SUITE")
    print("=" * 60)

    results = {
        "Matrix Operations": verify_matrix_operations(),
        "Skyline Solver": verify_skyline_solver(),
        "Truss Element Stiffness": verify_truss_element_stiffness(),
        "Two-Bar Truss": verify_two_bar_truss(),
        "Cantilever Beam": verify_cantilever_beam(),
    }

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        all_pass &= passed

    print(f"\nOverall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    return all_pass


if __name__ == "__main__":
    run_all_verifications()
