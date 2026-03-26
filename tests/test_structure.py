"""
test_structure.py - Unit Tests for Structural Analysis
=======================================================

Purpose:
    Tests for structural analysis components: nodes, elements,
    assembly, and complete analysis.

Running:
    python -m pytest tests/test_structure.py -v
    or
    python tests/test_structure.py
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from matrix.vector import Vector
from structure.node import Node
from structure.material import Material
from structure.section import Section
from structure.element import Truss2D, Frame2D
from structure.load import NodalLoad, LoadCase
from structure.boundary import BoundaryCondition
from structure.model import StructuralModel
from structure.analysis import StaticAnalysis


def test_node_creation():
    """Test node creation and distance calculation."""
    n0 = Node(0, 0.0, 0.0)
    n1 = Node(1, 3.0, 4.0)
    assert abs(n0.distance_to(n1) - 5.0) < 1e-12


def test_truss_element_horizontal():
    """Test horizontal truss element stiffness."""
    n0 = Node(0, 0.0, 0.0)
    n1 = Node(1, 2.0, 0.0)
    mat = Material(0, 100.0)
    sec = Section(0, 0.5)
    elem = Truss2D(0, n0, n1, mat, sec)

    assert abs(elem.length - 2.0) < 1e-12
    assert abs(elem.cos_angle - 1.0) < 1e-12
    assert abs(elem.sin_angle - 0.0) < 1e-12

    k = elem.global_stiffness()
    ea_l = 100.0 * 0.5 / 2.0  # = 25

    assert abs(k.get(0, 0) - ea_l) < 1e-8
    assert abs(k.get(0, 2) - (-ea_l)) < 1e-8
    assert abs(k.get(1, 1) - 0.0) < 1e-8


def test_truss_element_inclined():
    """Test 45-degree truss element stiffness."""
    n0 = Node(0, 0.0, 0.0)
    n1 = Node(1, 1.0, 1.0)
    mat = Material(0, 200.0)
    sec = Section(0, 1.0)
    elem = Truss2D(0, n0, n1, mat, sec)

    L = math.sqrt(2.0)
    ea_l = 200.0 * 1.0 / L
    c = s = 1.0 / math.sqrt(2.0)
    cc = c * c * ea_l
    cs = c * s * ea_l

    k = elem.global_stiffness()
    assert abs(k.get(0, 0) - cc) < 1e-8
    assert abs(k.get(0, 1) - cs) < 1e-8


def test_frame_element_local_stiffness():
    """Test frame element local stiffness for a horizontal beam."""
    n0 = Node(0, 0.0, 0.0)
    n1 = Node(1, 4.0, 0.0)
    mat = Material(0, 200.0)
    sec = Section(0, 0.5, moment_of_inertia=0.01)
    elem = Frame2D(0, n0, n1, mat, sec)

    k = elem.local_stiffness()
    L = 4.0
    E = 200.0
    A = 0.5
    I = 0.01

    ea_l = E * A / L
    ei_l3 = E * I / (L ** 3)

    assert abs(k.get(0, 0) - ea_l) < 1e-8
    assert abs(k.get(1, 1) - 12 * ei_l3) < 1e-8
    assert abs(k.get(2, 2) - 4 * ei_l3 * L * L) < 1e-8


def test_model_dof_assignment():
    """Test DOF assignment for a truss model."""
    model = StructuralModel(analysis_type="truss")
    model.add_node(Node(0, 0.0, 0.0))
    model.add_node(Node(1, 1.0, 0.0))
    model.add_node(Node(2, 0.5, 1.0))
    model.assign_dofs()

    assert model.total_dofs == 6
    assert model.nodes[0].dof_indices == [0, 1]
    assert model.nodes[1].dof_indices == [2, 3]
    assert model.nodes[2].dof_indices == [4, 5]


def test_cantilever_beam_analysis():
    """
    Test cantilever beam: fixed at one end, point load at free end.
    Verify tip deflection against analytical formula: delta = PL^3/(3EI).
    """
    L = 4.0
    E = 200e6
    A = 0.01
    I = 1e-4
    P = 10.0

    model = StructuralModel(analysis_type="frame")
    model.add_node(Node(0, 0.0, 0.0))
    model.add_node(Node(1, L, 0.0))

    mat = Material(0, E)
    sec = Section(0, A, I)
    model.add_element(Frame2D(0, model.nodes[0], model.nodes[1], mat, sec))

    model.add_boundary_condition(BoundaryCondition(0, 0))
    model.add_boundary_condition(BoundaryCondition(0, 1))
    model.add_boundary_condition(BoundaryCondition(0, 2))

    lc = LoadCase("Test")
    lc.add_nodal_load(NodalLoad(1, fy=-P))
    model.add_load_case(lc)

    analysis = StaticAnalysis(model)
    results = analysis.solve(verbose=False)

    delta_expected = -P * L ** 3 / (3.0 * E * I)
    uy = results.displacements[model.nodes[1].dof_indices[1]]
    assert abs(uy - delta_expected) < 1e-10, \
        f"Tip deflection: {uy:.10e} vs expected {delta_expected:.10e}"


def test_symmetric_truss():
    """Test a symmetric 2-bar truss to verify symmetry of solution."""
    model = StructuralModel(analysis_type="truss")
    model.add_node(Node(0, 0.0, 0.0))
    model.add_node(Node(1, 4.0, 0.0))
    model.add_node(Node(2, 2.0, 2.0))

    mat = Material(0, 100.0)
    sec = Section(0, 1.0)
    model.add_element(Truss2D(0, model.nodes[0], model.nodes[2], mat, sec))
    model.add_element(Truss2D(1, model.nodes[1], model.nodes[2], mat, sec))

    for dof in [0, 1]:
        model.add_boundary_condition(BoundaryCondition(0, dof))
        model.add_boundary_condition(BoundaryCondition(1, dof))

    lc = LoadCase("Test")
    lc.add_nodal_load(NodalLoad(2, fy=-10.0))
    model.add_load_case(lc)

    analysis = StaticAnalysis(model)
    results = analysis.solve(verbose=False)

    # By symmetry, ux at Node 2 should be zero
    ux2 = results.displacements[model.nodes[2].dof_indices[0]]
    assert abs(ux2) < 1e-6, f"Symmetry check: ux2 = {ux2}"

    # Check equilibrium
    eq = analysis.verify_equilibrium()
    assert eq['equilibrium_satisfied'], "Equilibrium not satisfied"


def test_three_bar_truss_reactions():
    """Test 3-bar truss reaction forces against statics."""
    model = StructuralModel(analysis_type="truss")
    model.add_node(Node(0, 0.0, 0.0))
    model.add_node(Node(1, 4.0, 0.0))
    model.add_node(Node(2, 2.0, 3.0))

    mat = Material(0, 200e6)
    sec = Section(0, 0.001)
    model.add_element(Truss2D(0, model.nodes[0], model.nodes[1], mat, sec))
    model.add_element(Truss2D(1, model.nodes[0], model.nodes[2], mat, sec))
    model.add_element(Truss2D(2, model.nodes[1], model.nodes[2], mat, sec))

    model.add_boundary_condition(BoundaryCondition(0, 0))
    model.add_boundary_condition(BoundaryCondition(0, 1))
    model.add_boundary_condition(BoundaryCondition(1, 1))

    lc = LoadCase("Test")
    lc.add_nodal_load(NodalLoad(2, fy=-10.0))
    model.add_load_case(lc)

    analysis = StaticAnalysis(model)
    results = analysis.solve(verbose=False)

    # By statics: Ry0 = 5, Ry1 = 5, Rx0 = 0
    r0 = results.reactions[0]
    r1 = results.reactions[1]
    assert abs(r0['fx']) < 1e-3, f"Rx0 = {r0['fx']}"
    assert abs(r0['fy'] - 5.0) < 1e-3, f"Ry0 = {r0['fy']}"
    assert abs(r1['fy'] - 5.0) < 1e-3, f"Ry1 = {r1['fy']}"


def run_all_tests():
    """Run all structural analysis tests."""
    tests = [
        test_node_creation,
        test_truss_element_horizontal,
        test_truss_element_inclined,
        test_frame_element_local_stiffness,
        test_model_dof_assignment,
        test_cantilever_beam_analysis,
        test_symmetric_truss,
        test_three_bar_truss_reactions,
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

    print(f"\nStructure Tests: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    print("Running Structural Analysis Tests...")
    run_all_tests()
