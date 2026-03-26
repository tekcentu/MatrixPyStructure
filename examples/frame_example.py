"""
frame_example.py - 2D Frame Verification Example
==================================================

Purpose:
    Demonstrates frame analysis with a simple portal frame that can be
    verified against hand calculations and textbook results.

Example Problem:
    Simple portal frame:

        Node 1 -------- Node 2
        |                     |
        |                     |
        |                     |
        Node 0            Node 3
        (fixed)           (fixed)

    Geometry:
        Node 0: (0, 0) - fixed support
        Node 1: (0, 4) - corner
        Node 2: (6, 4) - corner (loaded)
        Node 3: (6, 0) - fixed support

    Elements:
        0: Node 0 -> Node 1 (left column, L=4m)
        1: Node 1 -> Node 2 (beam, L=6m)
        2: Node 2 -> Node 3 (right column, L=4m)

    Material: E = 200e6 kN/m^2
    Section: A = 0.01 m^2, I = 1e-4 m^4

    Load: Fx = 10 kN at Node 1 (horizontal, lateral load)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from structure.node import Node
from structure.material import Material
from structure.section import Section
from structure.element import Frame2D
from structure.load import NodalLoad, LoadCase
from structure.boundary import BoundaryCondition
from structure.model import StructuralModel
from structure.analysis import StaticAnalysis


def run_frame_example():
    """
    Run the portal frame example with verification output.
    """
    print("=" * 70)
    print("2D FRAME ANALYSIS - PORTAL FRAME EXAMPLE")
    print("=" * 70)

    # --- Define Model ---
    model = StructuralModel(analysis_type="frame")

    # Nodes
    n0 = Node(0, 0.0, 0.0)
    n1 = Node(1, 0.0, 4.0)
    n2 = Node(2, 6.0, 4.0)
    n3 = Node(3, 6.0, 0.0)
    model.add_node(n0)
    model.add_node(n1)
    model.add_node(n2)
    model.add_node(n3)

    # Material and Section
    steel = Material(0, elastic_modulus=200e6, name="Steel")
    sec = Section(0, area=0.01, moment_of_inertia=1e-4, name="W-section")

    # Elements
    e0 = Frame2D(0, n0, n1, steel, sec)  # left column
    e1 = Frame2D(1, n1, n2, steel, sec)  # beam
    e2 = Frame2D(2, n2, n3, steel, sec)  # right column
    model.add_element(e0)
    model.add_element(e1)
    model.add_element(e2)

    # Fixed supports at Node 0 and Node 3
    for nid in [0, 3]:
        model.add_boundary_condition(BoundaryCondition(nid, 0))  # ux = 0
        model.add_boundary_condition(BoundaryCondition(nid, 1))  # uy = 0
        model.add_boundary_condition(BoundaryCondition(nid, 2))  # rz = 0

    # Lateral load at Node 1
    lc = LoadCase("Lateral Load")
    lc.add_nodal_load(NodalLoad(1, fx=10.0, fy=0.0, mz=0.0))
    model.add_load_case(lc)

    # --- Print Element Properties ---
    print("\n--- Element Properties ---")
    for elem in model.elements:
        print(f"Element {elem.elem_id}: "
              f"Nodes ({elem.node_i.node_id}->{elem.node_j.node_id}), "
              f"L={elem.length:.2f} m")

    # --- Verify element local stiffness ---
    print("\n--- Element 0 Local Stiffness Matrix (6x6) ---")
    k_local = e0.local_stiffness()
    for i in range(6):
        row = [f"{k_local.get(i, j):12.2f}" for j in range(6)]
        print(f"  [{', '.join(row)}]")

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
    print(f"Sum Mz = {eq['sum_mz']:.6e}")
    print(f"Equilibrium satisfied: {eq['equilibrium_satisfied']}")

    # --- Verification Notes ---
    print("\n--- Verification ---")
    print("For a portal frame with lateral load P=10 kN at Node 1:")
    print("  Expected: Rx0 + Rx3 = -10 kN (equilibrium in x)")
    print("  Expected: Ry0 + Ry3 = 0 kN (equilibrium in y)")

    r0 = results.reactions[0]
    r3 = results.reactions[3]
    print(f"  Computed: Rx0 + Rx3 = {r0['fx'] + r3['fx']:.6e} kN")
    print(f"  Computed: Ry0 + Ry3 = {r0['fy'] + r3['fy']:.6e} kN")

    return results


if __name__ == "__main__":
    run_frame_example()
