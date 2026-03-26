"""
section.py - Cross-Section Properties
======================================

Purpose:
    Stores cross-section geometric properties required for structural
    analysis: area and moment of inertia.

Assumptions:
    - Properties are constant along the element length.
    - For truss elements, only area (A) is needed.
    - For frame elements, area (A) and moment of inertia (I) are needed.

Units:
    Area in length^2, moment of inertia in length^4.
"""


class Section:
    """
    Cross-section properties.

    Attributes:
        sec_id (int): Unique section identifier.
        area (float): Cross-sectional area (A).
        moment_of_inertia (float): Second moment of area (I_z).
        name (str): Descriptive name.
    """

    def __init__(self, sec_id, area, moment_of_inertia=0.0, name=""):
        """
        Create a section.

        Inputs:
            sec_id (int): Unique identifier.
            area (float): Cross-sectional area (must be > 0).
            moment_of_inertia (float): Moment of inertia about z-axis
                (default 0.0, used for frame elements).
            name (str): Descriptive name.

        Raises:
            ValueError: If area <= 0.
        """
        if area <= 0:
            raise ValueError(f"Area must be > 0, got {area}")
        self.sec_id = sec_id
        self.area = float(area)
        self.moment_of_inertia = float(moment_of_inertia)
        self.name = name

    @property
    def A(self):
        """Shorthand for area."""
        return self.area

    @property
    def I(self):
        """Shorthand for moment of inertia."""
        return self.moment_of_inertia

    def __repr__(self):
        return (
            f"Section(id={self.sec_id}, A={self.area:.4e}, "
            f"I={self.moment_of_inertia:.4e}, name='{self.name}')"
        )
