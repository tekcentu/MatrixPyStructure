"""
load.py - Applied Loads
========================

Purpose:
    Represents applied loads on the structure: nodal forces and moments.

Assumptions:
    - Loads are applied at nodes (nodal loads).
    - For truss: forces in x and y directions.
    - For frame: forces in x, y and moment about z.
    - All loads are static.

Units:
    Forces in consistent force units, moments in force*length units.
"""


class NodalLoad:
    """
    A load applied at a specific node.

    Attributes:
        node_id (int): Target node identifier.
        fx (float): Force in x-direction.
        fy (float): Force in y-direction.
        mz (float): Moment about z-axis (for frames only).
    """

    def __init__(self, node_id, fx=0.0, fy=0.0, mz=0.0):
        """
        Create a nodal load.

        Inputs:
            node_id (int): Node where load is applied.
            fx (float): Force in x-direction (default 0.0).
            fy (float): Force in y-direction (default 0.0).
            mz (float): Moment about z-axis (default 0.0).
        """
        self.node_id = node_id
        self.fx = float(fx)
        self.fy = float(fy)
        self.mz = float(mz)

    def __repr__(self):
        return (
            f"NodalLoad(node={self.node_id}, "
            f"fx={self.fx:.4e}, fy={self.fy:.4e}, mz={self.mz:.4e})"
        )


class LoadCase:
    """
    A collection of loads forming a load case.

    Attributes:
        name (str): Load case name.
        nodal_loads (list[NodalLoad]): List of applied nodal loads.
    """

    def __init__(self, name="Default"):
        """
        Create a load case.

        Inputs:
            name (str): Descriptive name for the load case.
        """
        self.name = name
        self.nodal_loads = []

    def add_nodal_load(self, load):
        """
        Add a nodal load to this load case.

        Inputs:
            load (NodalLoad): Load to add.
        """
        if not isinstance(load, NodalLoad):
            raise TypeError("Expected NodalLoad")
        self.nodal_loads.append(load)

    def __repr__(self):
        return f"LoadCase(name='{self.name}', n_loads={len(self.nodal_loads)})"
