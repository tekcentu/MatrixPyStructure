"""
main.py - Main Entry Point for MatrixPyStructure
==================================================

Purpose:
    Entry point for the structural analysis software. Demonstrates
    the complete workflow: reading input, running analysis, and
    producing output.

Usage:
    python main.py                    # Run default examples
    python main.py input.json         # Analyze from JSON input file
    python main.py --verify           # Run verification suite

Input Format (JSON):
    {
        "analysis_type": "truss" or "frame",
        "nodes": [{"id": 0, "x": 0.0, "y": 0.0}, ...],
        "materials": [{"id": 0, "E": 200e6, "name": "Steel"}, ...],
        "sections": [{"id": 0, "A": 0.001, "I": 0.0}, ...],
        "elements": [{"id": 0, "node_i": 0, "node_j": 1,
                       "material": 0, "section": 0}, ...],
        "boundary_conditions": [{"node": 0, "dof": 0, "value": 0.0}, ...],
        "load_cases": [
            {"name": "LC1",
             "loads": [{"node": 2, "fx": 0, "fy": -10, "mz": 0}, ...]}
        ]
    }
"""

import sys
import json

from structure.node import Node
from structure.material import Material
from structure.section import Section
from structure.element import Truss2D, Frame2D
from structure.load import NodalLoad, LoadCase
from structure.boundary import BoundaryCondition
from structure.model import StructuralModel
from structure.analysis import StaticAnalysis


def load_model_from_json(filepath):
    """
    Load a structural model from a JSON input file.

    Inputs:
        filepath (str): Path to JSON input file.

    Returns:
        StructuralModel: Populated model ready for analysis.

    Input Format:
        See module docstring for JSON schema.
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    analysis_type = data.get("analysis_type", "truss")
    model = StructuralModel(analysis_type=analysis_type)

    # Nodes
    nodes_map = {}
    for nd in data["nodes"]:
        node = Node(nd["id"], nd["x"], nd["y"])
        model.add_node(node)
        nodes_map[nd["id"]] = node

    # Materials
    materials_map = {}
    for md in data["materials"]:
        mat = Material(
            md["id"], md["E"],
            poisson_ratio=md.get("nu", 0.3),
            name=md.get("name", "")
        )
        materials_map[md["id"]] = mat

    # Sections
    sections_map = {}
    for sd in data["sections"]:
        sec = Section(
            sd["id"], sd["A"],
            moment_of_inertia=sd.get("I", 0.0),
            name=sd.get("name", "")
        )
        sections_map[sd["id"]] = sec

    # Elements
    for ed in data["elements"]:
        ni = nodes_map[ed["node_i"]]
        nj = nodes_map[ed["node_j"]]
        mat = materials_map[ed["material"]]
        sec = sections_map[ed["section"]]

        if analysis_type == "truss":
            elem = Truss2D(ed["id"], ni, nj, mat, sec)
        else:
            elem = Frame2D(ed["id"], ni, nj, mat, sec)
        model.add_element(elem)

    # Boundary conditions
    for bc in data["boundary_conditions"]:
        model.add_boundary_condition(
            BoundaryCondition(bc["node"], bc["dof"], bc.get("value", 0.0))
        )

    # Load cases
    for lcd in data.get("load_cases", []):
        lc = LoadCase(lcd.get("name", "Default"))
        for ld in lcd["loads"]:
            lc.add_nodal_load(NodalLoad(
                ld["node"],
                fx=ld.get("fx", 0.0),
                fy=ld.get("fy", 0.0),
                mz=ld.get("mz", 0.0)
            ))
        model.add_load_case(lc)

    return model


def save_results_to_json(filepath, model, results):
    """
    Save analysis results to a JSON output file.

    Inputs:
        filepath (str): Output file path.
        model (StructuralModel): The analyzed model.
        results (AnalysisResults): Analysis results.

    Output Format:
        {
            "displacements": {"node_id": {"ux": ..., "uy": ..., "rz": ...}},
            "reactions": {"node_id": {"fx": ..., "fy": ..., "mz": ...}},
            "member_forces": {"elem_id": ...}
        }
    """
    output = {
        "displacements": {},
        "reactions": {},
        "member_forces": {}
    }

    for nid in sorted(model.nodes.keys()):
        node = model.nodes[nid]
        disp = {"ux": results.displacements[node.dof_indices[0]],
                "uy": results.displacements[node.dof_indices[1]]}
        if model.dofs_per_node == 3:
            disp["rz"] = results.displacements[node.dof_indices[2]]
        output["displacements"][str(nid)] = disp

    for nid, r in results.reactions.items():
        output["reactions"][str(nid)] = r

    for eid, forces in results.member_forces.items():
        if isinstance(forces, (int, float)):
            output["member_forces"][str(eid)] = forces
        else:
            output["member_forces"][str(eid)] = forces.to_list()

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {filepath}")


def run_default_example():
    """Run the default truss example and verification."""
    from examples.truss_example import run_truss_example
    from examples.verification import run_all_verifications

    print("Running default truss example...\n")
    run_truss_example()

    print("\n\nRunning verification suite...\n")
    run_all_verifications()


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--verify":
            from examples.verification import run_all_verifications
            run_all_verifications()
        elif arg == "--truss":
            from examples.truss_example import run_truss_example
            run_truss_example()
        elif arg == "--frame":
            from examples.frame_example import run_frame_example
            run_frame_example()
        elif arg.endswith(".json"):
            model = load_model_from_json(arg)
            analysis = StaticAnalysis(model)
            results = analysis.solve(verbose=True)
            results.print_displacements(model)
            results.print_reactions()
            results.print_member_forces(model)

            output_file = arg.replace(".json", "_results.json")
            save_results_to_json(output_file, model, results)
        else:
            print(f"Unknown argument: {arg}")
            print("Usage: python main.py [input.json | --verify | --truss | --frame]")
    else:
        run_default_example()


if __name__ == "__main__":
    main()
