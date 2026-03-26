"""
model.py - Structural Model Assembly
======================================

Purpose:
    Central class that assembles nodes, elements, loads, and boundary
    conditions into a complete structural model. Handles DOF numbering,
    global stiffness matrix assembly, and load vector assembly.

Main Steps:
    1. Add nodes, elements, materials, sections, loads, and BCs.
    2. Number the DOFs (assign global equation numbers).
    3. Assemble the global stiffness matrix (skyline storage).
    4. Assemble the global load vector.
    5. Apply boundary conditions.

Assumptions:
    - All elements must reference nodes already added to the model.
    - DOF numbering is sequential: node 0 gets DOFs 0,1,...; node 1 next, etc.
    - The model supports mixed truss and frame elements (though typically
      one type is used per model).
"""

from matrix.skyline_matrix import SkylineMatrix
from matrix.vector import Vector


class StructuralModel:
    """
    Container for the complete structural model.

    Attributes:
        nodes (dict[int, Node]): Nodes keyed by node_id.
        elements (list): List of elements (Truss2D or Frame2D).
        boundary_conditions (list[BoundaryCondition]): Applied BCs.
        load_cases (list[LoadCase]): Applied load cases.
        dofs_per_node (int): Number of DOFs per node (2 for truss, 3 for frame).
        total_dofs (int): Total number of DOFs in the model.
    """

    def __init__(self, analysis_type="truss"):
        """
        Create a structural model.

        Inputs:
            analysis_type (str): "truss" (2 DOFs/node) or "frame" (3 DOFs/node).
        """
        if analysis_type not in ("truss", "frame"):
            raise ValueError(f"analysis_type must be 'truss' or 'frame'")
        self.analysis_type = analysis_type
        self.dofs_per_node = 2 if analysis_type == "truss" else 3

        self.nodes = {}
        self.elements = []
        self.boundary_conditions = []
        self.load_cases = []
        self.total_dofs = 0
        self._dofs_assigned = False

    def add_node(self, node):
        """
        Add a node to the model.

        Inputs:
            node (Node): Node to add.
        """
        self.nodes[node.node_id] = node
        self._dofs_assigned = False

    def add_element(self, element):
        """
        Add an element to the model.

        Inputs:
            element: Truss2D or Frame2D element.
        """
        self.elements.append(element)

    def add_boundary_condition(self, bc):
        """
        Add a boundary condition.

        Inputs:
            bc (BoundaryCondition): Boundary condition to apply.
        """
        self.boundary_conditions.append(bc)

    def add_load_case(self, load_case):
        """
        Add a load case.

        Inputs:
            load_case (LoadCase): Load case to add.
        """
        self.load_cases.append(load_case)

    def assign_dofs(self):
        """
        Assign global DOF indices to all nodes.

        Purpose:
            Sequential DOF numbering: node with smallest ID gets DOFs
            starting from 0, etc.

        Updates:
            Sets total_dofs and each node's dof_indices.
        """
        sorted_ids = sorted(self.nodes.keys())
        dof_counter = 0
        for nid in sorted_ids:
            self.nodes[nid].assign_dofs(dof_counter, self.dofs_per_node)
            dof_counter += self.dofs_per_node
        self.total_dofs = dof_counter
        self._dofs_assigned = True

    def _get_element_dof_lists(self):
        """
        Get DOF connectivity for all elements.

        Returns:
            list[list[int]]: For each element, its global DOF indices.

        Purpose:
            Used to compute skyline column heights.
        """
        dof_lists = []
        for elem in self.elements:
            dof_lists.append(elem.get_dof_indices())
        return dof_lists

    def assemble_stiffness_matrix(self):
        """
        Assemble the global stiffness matrix using skyline storage.

        Returns:
            SkylineMatrix: Assembled global stiffness matrix.

        Algorithm:
            1. Compute column heights from element connectivity.
            2. Create skyline matrix.
            3. For each element:
               a. Compute element global stiffness matrix.
               b. Add element contributions to global matrix using
                  scatter (direct stiffness method).
        """
        if not self._dofs_assigned:
            self.assign_dofs()

        n = self.total_dofs
        element_dofs = self._get_element_dof_lists()

        # Create skyline matrix with column heights from connectivity
        K = SkylineMatrix.from_dof_connectivity(n, element_dofs)

        # Assemble element stiffness matrices
        for elem in self.elements:
            k_e = elem.global_stiffness()
            dofs = elem.get_dof_indices()
            n_dofs = len(dofs)

            for i_local in range(n_dofs):
                i_global = dofs[i_local]
                for j_local in range(n_dofs):
                    j_global = dofs[j_local]
                    # Only assemble upper triangle (i_global <= j_global)
                    # since SkylineMatrix is symmetric and add(i,j) with
                    # i>j maps to (j,i), causing double-counting otherwise
                    if i_global <= j_global:
                        val = k_e.get(i_local, j_local)
                        if abs(val) > 1e-30:
                            K.add(i_global, j_global, val)

        return K

    def assemble_load_vector(self, load_case_index=0):
        """
        Assemble the global load vector for a given load case.

        Inputs:
            load_case_index (int): Index of the load case (default 0).

        Returns:
            Vector: Global load vector of size total_dofs.
        """
        if not self._dofs_assigned:
            self.assign_dofs()

        F = Vector(self.total_dofs)

        if load_case_index >= len(self.load_cases):
            return F

        lc = self.load_cases[load_case_index]

        for load in lc.nodal_loads:
            if load.node_id not in self.nodes:
                raise ValueError(f"Load references unknown node {load.node_id}")
            node = self.nodes[load.node_id]
            dofs = node.dof_indices

            if len(dofs) >= 1:
                F.add_value(dofs[0], load.fx)
            if len(dofs) >= 2:
                F.add_value(dofs[1], load.fy)
            if len(dofs) >= 3:
                F.add_value(dofs[2], load.mz)

        return F

    def apply_boundary_conditions(self, K, F):
        """
        Apply boundary conditions using the elimination method.

        Inputs:
            K (SkylineMatrix): Global stiffness matrix (modified in place).
            F (Vector): Global load vector (modified in place).

        Purpose:
            For each constrained DOF i with prescribed value d:
            1. For all j != i within skyline: F[j] -= K[i][j] * d
            2. Zero out row i and column i within the skyline profile.
            3. Set K[i][i] = 1.0
            4. Set F[i] = d

        Note:
            The elimination method preserves positive-definiteness and
            works well with skyline storage. Zero entries outside the
            skyline are already zero and need not be modified.
        """
        if not self._dofs_assigned:
            self.assign_dofs()

        n = self.total_dofs

        for bc in self.boundary_conditions:
            if bc.node_id not in self.nodes:
                raise ValueError(f"BC references unknown node {bc.node_id}")
            node = self.nodes[bc.node_id]
            if bc.dof_local >= len(node.dof_indices):
                raise ValueError(
                    f"BC dof_local {bc.dof_local} exceeds node DOFs"
                )
            global_dof = node.dof_indices[bc.dof_local]
            d = bc.prescribed_value

            # Adjust load vector and zero out row/column
            for j in range(n):
                if j != global_dof:
                    val = K.get(global_dof, j)
                    if abs(val) > 1e-30:
                        F._data[j] -= val * d
                        # Zero entries: set K[dof, j] = 0
                        # Due to symmetry, also zeros K[j, dof]
                        try:
                            K.set(global_dof, j, 0.0)
                        except IndexError:
                            pass  # Outside skyline, already zero

            # Set diagonal to 1 and RHS to prescribed value
            K.set(global_dof, global_dof, 1.0)
            F.set(global_dof, d)

    def get_constrained_dofs(self):
        """
        Get list of constrained global DOF indices.

        Returns:
            list[int]: Global DOF indices that have boundary conditions.
        """
        constrained = []
        for bc in self.boundary_conditions:
            node = self.nodes[bc.node_id]
            global_dof = node.dof_indices[bc.dof_local]
            constrained.append(global_dof)
        return constrained

    def __repr__(self):
        return (
            f"StructuralModel(type={self.analysis_type}, "
            f"nodes={len(self.nodes)}, elements={len(self.elements)}, "
            f"dofs={self.total_dofs})"
        )
