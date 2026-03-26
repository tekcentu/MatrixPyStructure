"""
boundary.py - Boundary Conditions
===================================

Purpose:
    Handles support (boundary) conditions for the structural model.
    Supports fixed, pinned, roller, and prescribed displacement conditions.

Approach:
    Uses the penalty method or the elimination method to apply
    boundary conditions to the global stiffness matrix and load vector.

    Elimination method (used here):
        - For a fixed DOF i with prescribed value d:
          1. Set K[i][i] = 1 (or a large number)
          2. Set K[i][j] = 0 for all j != i
          3. Set K[j][i] = 0 for all j != i
          4. Adjust F[j] -= K[j][i] * d for all j before zeroing
          5. Set F[i] = d

Assumptions:
    - Boundary conditions are applied to specific DOFs.
    - Prescribed displacement defaults to zero (fixed support).
"""


class BoundaryCondition:
    """
    A single boundary condition (constraint on a DOF).

    Attributes:
        node_id (int): Node where BC is applied.
        dof_local (int): Local DOF index at the node
            (0=ux, 1=uy, 2=rz for frames; 0=ux, 1=uy for trusses).
        prescribed_value (float): Prescribed displacement/rotation.
    """

    def __init__(self, node_id, dof_local, prescribed_value=0.0):
        """
        Create a boundary condition.

        Inputs:
            node_id (int): Node identifier.
            dof_local (int): Local DOF index at the node.
            prescribed_value (float): Prescribed value (default 0.0 = fixed).
        """
        self.node_id = node_id
        self.dof_local = dof_local
        self.prescribed_value = float(prescribed_value)

    def __repr__(self):
        return (
            f"BC(node={self.node_id}, dof={self.dof_local}, "
            f"value={self.prescribed_value:.6e})"
        )
