"""
node.py - Node (Joint) Definition
==================================

Purpose:
    Represents a node (joint/point) in a 2D structural model.
    Each node has coordinates and is assigned global DOF indices.

Assumptions:
    - 2D analysis only (x, y coordinates).
    - For truss analysis: 2 DOFs per node (ux, uy).
    - For frame analysis: 3 DOFs per node (ux, uy, rotation_z).
    - 0-based indexing for node IDs and DOFs.

Units:
    Coordinates are in consistent length units (e.g., meters or inches).
"""


class Node:
    """
    A node in a 2D structural model.

    Attributes:
        node_id (int): Unique node identifier (0-based).
        x (float): X-coordinate.
        y (float): Y-coordinate.
        dof_indices (list[int]): Global DOF indices assigned to this node.
    """

    def __init__(self, node_id, x, y):
        """
        Create a node.

        Inputs:
            node_id (int): Unique node identifier.
            x (float): X-coordinate.
            y (float): Y-coordinate.
        """
        self.node_id = node_id
        self.x = float(x)
        self.y = float(y)
        self.dof_indices = []  # Assigned during model setup

    def assign_dofs(self, start_index, dofs_per_node):
        """
        Assign global DOF indices to this node.

        Inputs:
            start_index (int): First global DOF index for this node.
            dofs_per_node (int): Number of DOFs at this node (2 or 3).

        Purpose:
            Maps local node DOFs to global equation numbers.
            For truss: [ux, uy]
            For frame: [ux, uy, rz]
        """
        self.dof_indices = list(range(start_index, start_index + dofs_per_node))

    def distance_to(self, other):
        """
        Compute Euclidean distance to another node.

        Inputs:
            other (Node): Target node.

        Returns:
            float: Distance between the two nodes.
        """
        import math
        dx = other.x - self.x
        dy = other.y - self.y
        return math.sqrt(dx * dx + dy * dy)

    def __repr__(self):
        return (
            f"Node(id={self.node_id}, x={self.x:.4f}, y={self.y:.4f}, "
            f"dofs={self.dof_indices})"
        )
