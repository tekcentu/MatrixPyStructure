"""
analysis.py - Static Structural Analysis Solver
=================================================

Purpose:
    Performs linear static analysis of 2D truss and frame structures.
    Orchestrates the complete analysis workflow: assembly, BC application,
    solution, and post-processing.

Main Steps:
    1. Assign DOFs to nodes.
    2. Assemble global stiffness matrix (skyline storage).
    3. Assemble global load vector.
    4. Apply boundary conditions (penalty method).
    5. Solve K * u = F using LDL^T factorization.
    6. Compute support reactions.
    7. Compute member forces.

Assumptions:
    - Linear elastic static analysis.
    - Small deformation theory.
    - All loads are applied simultaneously.

Units:
    Consistent units throughout (e.g., kN and m, or lb and in).
"""

from matrix.solver import Solver
from matrix.vector import Vector


class AnalysisResults:
    """
    Container for analysis results.

    Attributes:
        displacements (Vector): Global displacement vector.
        reactions (dict[int, dict]): Reaction forces at constrained nodes.
            Key = node_id, Value = dict with 'fx', 'fy', 'mz'.
        member_forces (dict[int, object]): Member forces for each element.
            Key = elem_id, Value = axial force (truss) or force vector (frame).
        K_assembled (SkylineMatrix): Assembled stiffness matrix (before BC).
        F_assembled (Vector): Assembled load vector (before BC).
    """

    def __init__(self):
        self.displacements = None
        self.reactions = {}
        self.member_forces = {}
        self.K_assembled = None
        self.F_assembled = None

    def print_displacements(self, model):
        """
        Print nodal displacements in a formatted table.

        Inputs:
            model (StructuralModel): The analyzed model.
        """
        print("\n" + "=" * 60)
        print("NODAL DISPLACEMENTS")
        print("=" * 60)

        if model.analysis_type == "truss":
            print(f"{'Node':>6} {'ux':>14} {'uy':>14}")
            print("-" * 36)
            for nid in sorted(model.nodes.keys()):
                node = model.nodes[nid]
                ux = self.displacements[node.dof_indices[0]]
                uy = self.displacements[node.dof_indices[1]]
                print(f"{nid:>6} {ux:>14.6e} {uy:>14.6e}")
        else:
            print(f"{'Node':>6} {'ux':>14} {'uy':>14} {'rz':>14}")
            print("-" * 50)
            for nid in sorted(model.nodes.keys()):
                node = model.nodes[nid]
                ux = self.displacements[node.dof_indices[0]]
                uy = self.displacements[node.dof_indices[1]]
                rz = self.displacements[node.dof_indices[2]]
                print(f"{nid:>6} {ux:>14.6e} {uy:>14.6e} {rz:>14.6e}")

    def print_reactions(self):
        """Print support reactions in a formatted table."""
        print("\n" + "=" * 60)
        print("SUPPORT REACTIONS")
        print("=" * 60)
        print(f"{'Node':>6} {'Rx':>14} {'Ry':>14} {'Mz':>14}")
        print("-" * 50)
        for nid in sorted(self.reactions.keys()):
            r = self.reactions[nid]
            print(
                f"{nid:>6} {r['fx']:>14.6e} {r['fy']:>14.6e} "
                f"{r.get('mz', 0.0):>14.6e}"
            )

    def print_member_forces(self, model):
        """
        Print member forces in a formatted table.

        Inputs:
            model (StructuralModel): The analyzed model.
        """
        print("\n" + "=" * 60)
        print("MEMBER FORCES")
        print("=" * 60)

        if model.analysis_type == "truss":
            print(f"{'Elem':>6} {'Node_i':>8} {'Node_j':>8} {'Axial Force':>14}")
            print("-" * 38)
            for elem in model.elements:
                force = self.member_forces[elem.elem_id]
                status = "T" if force > 1e-10 else ("C" if force < -1e-10 else "-")
                print(
                    f"{elem.elem_id:>6} {elem.node_i.node_id:>8} "
                    f"{elem.node_j.node_id:>8} {force:>14.6e} ({status})"
                )
        else:
            print(
                f"{'Elem':>6} {'Node_i':>8} {'Node_j':>8} "
                f"{'N_i':>12} {'V_i':>12} {'M_i':>12} "
                f"{'N_j':>12} {'V_j':>12} {'M_j':>12}"
            )
            print("-" * 100)
            for elem in model.elements:
                f = self.member_forces[elem.elem_id]
                print(
                    f"{elem.elem_id:>6} {elem.node_i.node_id:>8} "
                    f"{elem.node_j.node_id:>8} "
                    f"{f[0]:>12.4e} {f[1]:>12.4e} {f[2]:>12.4e} "
                    f"{f[3]:>12.4e} {f[4]:>12.4e} {f[5]:>12.4e}"
                )


class StaticAnalysis:
    """
    Linear static analysis solver.

    Attributes:
        model (StructuralModel): The structural model to analyze.
        results (AnalysisResults): Analysis results after solve().
    """

    def __init__(self, model):
        """
        Create a static analysis for the given model.

        Inputs:
            model (StructuralModel): Structural model with nodes, elements,
                loads, and boundary conditions defined.
        """
        self.model = model
        self.results = AnalysisResults()

    def solve(self, verbose=True):
        """
        Perform the complete static analysis.

        Inputs:
            verbose (bool): If True, print progress messages.

        Returns:
            AnalysisResults: Object containing displacements, reactions,
                and member forces.

        Algorithm:
            1. Assign DOFs.
            2. Assemble global stiffness matrix (skyline).
            3. Assemble global load vector.
            4. Apply boundary conditions (penalty method).
            5. Solve K*u = F via LDL^T.
            6. Compute reactions: R = K_orig * u - F_orig.
            7. Compute member forces from displacements.
        """
        model = self.model

        # Step 1: Assign DOFs
        model.assign_dofs()
        if verbose:
            print(f"Model: {len(model.nodes)} nodes, "
                  f"{len(model.elements)} elements, "
                  f"{model.total_dofs} DOFs")

        # Step 2: Assemble global stiffness matrix
        if verbose:
            print("Assembling stiffness matrix (skyline storage)...")
        K = model.assemble_stiffness_matrix()
        self.results.K_assembled = K.copy()
        if verbose:
            print(f"  Skyline storage: {K.storage_size} elements "
                  f"(vs {model.total_dofs**2} full)")

        # Step 3: Assemble load vector
        if verbose:
            print("Assembling load vector...")
        F = model.assemble_load_vector(0)
        self.results.F_assembled = F.copy()

        # Step 4: Apply boundary conditions
        if verbose:
            print(f"Applying {len(model.boundary_conditions)} boundary conditions...")
        model.apply_boundary_conditions(K, F)

        # Step 5: Solve
        if verbose:
            print("Solving K*u = F (LDL^T skyline solver)...")
        u = Solver.solve_skyline(K, F)
        self.results.displacements = u

        # Step 6: Compute reactions
        if verbose:
            print("Computing reactions...")
        self._compute_reactions()

        # Step 7: Compute member forces
        if verbose:
            print("Computing member forces...")
        self._compute_member_forces()

        if verbose:
            print("Analysis complete.")

        return self.results

    def _compute_reactions(self):
        """
        Compute support reactions at constrained DOFs.

        Algorithm:
            R = K_original * u - F_original
            Only extract reactions at constrained DOFs.
        """
        model = self.model
        u = self.results.displacements
        K_orig = self.results.K_assembled
        F_orig = self.results.F_assembled

        # Compute K_orig * u
        Ku = K_orig.mat_vec(u)

        # Reactions at constrained DOFs
        for bc in model.boundary_conditions:
            node = model.nodes[bc.node_id]
            nid = bc.node_id

            if nid not in self.results.reactions:
                self.results.reactions[nid] = {"fx": 0.0, "fy": 0.0, "mz": 0.0}

            global_dof = node.dof_indices[bc.dof_local]
            reaction = Ku[global_dof] - F_orig[global_dof]

            if bc.dof_local == 0:
                self.results.reactions[nid]["fx"] = reaction
            elif bc.dof_local == 1:
                self.results.reactions[nid]["fy"] = reaction
            elif bc.dof_local == 2:
                self.results.reactions[nid]["mz"] = reaction

    def _compute_member_forces(self):
        """
        Compute member forces for all elements.

        For truss elements: axial force (scalar).
        For frame elements: end forces vector [N_i, V_i, M_i, N_j, V_j, M_j].
        """
        model = self.model
        u = self.results.displacements

        for elem in model.elements:
            dofs = elem.get_dof_indices()
            n_dofs = len(dofs)

            # Extract element displacements from global vector
            u_elem = Vector(n_dofs)
            for i in range(n_dofs):
                u_elem[i] = u[dofs[i]]

            # Compute member forces
            forces = elem.member_forces(u_elem)
            self.results.member_forces[elem.elem_id] = forces

    def verify_equilibrium(self):
        """
        Check global equilibrium: sum of applied forces + reactions = 0.

        Returns:
            dict: Equilibrium check results with residuals.

        Purpose:
            Verification step to ensure the solution is correct.
        """
        model = self.model
        results = self.results

        # Sum applied forces (including moment arms about origin)
        sum_fx = 0.0
        sum_fy = 0.0
        sum_mz = 0.0

        if len(model.load_cases) > 0:
            for load in model.load_cases[0].nodal_loads:
                node = model.nodes[load.node_id]
                sum_fx += load.fx
                sum_fy += load.fy
                # Moment about origin: Mz + x*Fy - y*Fx
                sum_mz += load.mz + node.x * load.fy - node.y * load.fx

        # Sum reactions (including moment arms about origin)
        for nid, r in results.reactions.items():
            node = model.nodes[nid]
            sum_fx += r["fx"]
            sum_fy += r["fy"]
            sum_mz += r.get("mz", 0.0) + node.x * r["fy"] - node.y * r["fx"]

        return {
            "sum_fx": sum_fx,
            "sum_fy": sum_fy,
            "sum_mz": sum_mz,
            "equilibrium_satisfied": (
                abs(sum_fx) < 1e-6 and
                abs(sum_fy) < 1e-6 and
                abs(sum_mz) < 1e-6
            )
        }
