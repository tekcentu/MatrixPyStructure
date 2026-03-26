"""
truss_example.py - 2D Truss Verification Example
==================================================

Purpose:
    Demonstrates the complete workflow of the structural analysis software
    using a simple 3-bar truss that can be verified by hand calculations.

Example Problem:
    A 3-bar planar truss with 4 nodes:

        Node 2 (4,3)
        /|
       / |
      /  |
     /   |
    Node 0----Node 1 (4,0)
    (0,0)     |
              |
              Node 3 (4,-3) [not connected directly to Node 0 in this example]

    Actually, let's use a classic textbook example:
    3-bar truss (Warren truss configuration):

        Node 2 (2, 3)
        / \\
       /   \\
      /     \\
    Node 0---Node 1
    (0,0)    (4,0)

    Elements:
        0: Node 0 -> Node 1 (horizontal)
        1: Node 0 -> Node 2 (inclined)
        2: Node 1 -> Node 2 (inclined)

    Supports:
        Node 0: pinned (ux=0, uy=0)
        Node 1: roller (uy=0)

    Load:
        Node 2: Fy = -10 kN (downward)

    Material: E = 200e6 kN/m^2 (steel)
    Section: A = 0.001 m^2 (all members)

Hand Calculation Verification:
    Length of element 0: L0 = 4.0 m
    Length of element 1: L1 = sqrt(4+9) = sqrt(13) = 3.6056 m
    Length of element 2: L2 = sqrt(4+9) = sqrt(13) = 3.6056 m

    By method of joints at Node 2:
    Element 1 angle from horizontal: atan(3/2) = 56.31 deg
    Element 2 angle from horizontal: atan(3/2) = 56.31 deg (from Node 1)
    -> Actually let me compute properly in the code.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from structure.node import Node
from structure.material import Material
from structure.section import Section
from structure.element import Truss2D
from structure.load import NodalLoad, LoadCase
from structure.boundary import BoundaryCondition
from structure.model import StructuralModel
from structure.analysis import StaticAnalysis
from matrix.dense_matrix import DenseMatrix
import math


def run_truss_example():
    """
    Run the 3-bar truss example with full verification output.

    Problem Setup:
        Node 0 (0, 0) - pinned support
        Node 1 (4, 0) - roller support (free in x)
        Node 2 (2, 3) - loaded node, Fy = -10 kN

        All elements: E = 200e6 kN/m^2, A = 0.001 m^2
    """
    print("=" * 70)
    print("2D TRUSS ANALYSIS - VERIFICATION EXAMPLE")
    print("3-Bar Truss with Hand Calculation Comparison")
    print("=" * 70)

    # --- Define Model ---
    model = StructuralModel(analysis_type="truss")

    # Nodes
    n0 = Node(0, 0.0, 0.0)
    n1 = Node(1, 4.0, 0.0)
    n2 = Node(2, 2.0, 3.0)
    model.add_node(n0)
    model.add_node(n1)
    model.add_node(n2)

    # Material
    steel = Material(0, elastic_modulus=200e6, name="Steel")

    # Section
    sec = Section(0, area=0.001, name="Standard")

    # Elements
    e0 = Truss2D(0, n0, n1, steel, sec)  # horizontal
    e1 = Truss2D(1, n0, n2, steel, sec)  # left inclined
    e2 = Truss2D(2, n1, n2, steel, sec)  # right inclined
    model.add_element(e0)
    model.add_element(e1)
    model.add_element(e2)

    # Boundary conditions
    # Node 0: pinned (ux=0, uy=0)
    model.add_boundary_condition(BoundaryCondition(0, 0))  # ux = 0
    model.add_boundary_condition(BoundaryCondition(0, 1))  # uy = 0
    # Node 1: roller (uy=0, free in x)
    model.add_boundary_condition(BoundaryCondition(1, 1))  # uy = 0

    # Load: Node 2, Fy = -10 kN
    lc = LoadCase("Gravity Load")
    lc.add_nodal_load(NodalLoad(2, fx=0.0, fy=-10.0))
    model.add_load_case(lc)

    # --- Print Element Properties ---
    print("\n--- Element Properties ---")
    for elem in model.elements:
        L = elem.length
        c = elem.cos_angle
        s = elem.sin_angle
        angle = math.degrees(math.atan2(s, c))
        print(f"Element {elem.elem_id}: "
              f"Nodes ({elem.node_i.node_id}->{elem.node_j.node_id}), "
              f"L={L:.4f} m, angle={angle:.2f} deg, "
              f"EA/L={steel.E * sec.A / L:.4f} kN/m")

    # --- Verify Element Stiffness Matrices ---
    print("\n--- Element Stiffness Matrices (Global) ---")
    for elem in model.elements:
        print(f"\nElement {elem.elem_id} "
              f"(Nodes {elem.node_i.node_id}->{elem.node_j.node_id}):")
        k_g = elem.global_stiffness()
        print(f"  k_global (4x4):")
        for i in range(4):
            row = [f"{k_g.get(i, j):12.2f}" for j in range(4)]
            print(f"    [{', '.join(row)}]")

    # --- Run Analysis ---
    print("\n--- Running Analysis ---")
    analysis = StaticAnalysis(model)
    results = analysis.solve(verbose=True)

    # --- Print Results ---
    results.print_displacements(model)
    results.print_reactions()
    results.print_member_forces(model)

    # --- Equilibrium Check ---
    eq = analysis.verify_equilibrium()
    print("\n--- Equilibrium Check ---")
    print(f"Sum Fx = {eq['sum_fx']:.6e}")
    print(f"Sum Fy = {eq['sum_fy']:.6e}")
    print(f"Equilibrium satisfied: {eq['equilibrium_satisfied']}")

    # --- Hand Calculation Verification ---
    print("\n" + "=" * 70)
    print("HAND CALCULATION VERIFICATION")
    print("=" * 70)

    L0 = e0.length
    L1 = e1.length
    L2 = e2.length
    print(f"\nElement lengths:")
    print(f"  L0 = {L0:.6f} m")
    print(f"  L1 = {L1:.6f} m (sqrt({2**2}+{3**2}) = sqrt({2**2+3**2}))")
    print(f"  L2 = {L2:.6f} m (sqrt({2**2}+{3**2}) = sqrt({2**2+3**2}))")

    # Method of joints verification for reactions
    # Sum Fy = 0: Ry0 + Ry1 - 10 = 0
    # Sum Mx about node 0: Ry1 * 4 - 10 * 2 = 0 => Ry1 = 5 kN
    # => Ry0 = 5 kN
    # Sum Fx = 0: Rx0 = 0
    print(f"\nExpected reactions (by statics):")
    print(f"  Rx0 = 0.0 kN (sum Fx = 0)")
    print(f"  Ry0 = 5.0 kN (moment about Node 1)")
    print(f"  Ry1 = 5.0 kN (moment about Node 0)")

    print(f"\nComputed reactions:")
    for nid in sorted(results.reactions.keys()):
        r = results.reactions[nid]
        print(f"  Node {nid}: Rx={r['fx']:.6f}, Ry={r['fy']:.6f}")

    # Check reactions match
    r0 = results.reactions[0]
    r1 = results.reactions[1]
    print(f"\nReaction verification:")
    print(f"  Rx0 error: {abs(r0['fx'] - 0.0):.2e}")
    print(f"  Ry0 error: {abs(r0['fy'] - 5.0):.2e}")
    print(f"  Ry1 error: {abs(r1['fy'] - 5.0):.2e}")

    return results


if __name__ == "__main__":
    run_truss_example()
