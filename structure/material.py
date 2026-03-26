"""
material.py - Material Properties
==================================

Purpose:
    Stores material properties required for structural analysis.

Assumptions:
    - Linear elastic material behavior.
    - Isotropic material (same properties in all directions).

Units:
    Caller must ensure unit consistency.
    Typical: E in kN/m^2 or psi, density in kg/m^3 or lb/in^3.
"""


class Material:
    """
    Isotropic linear elastic material.

    Attributes:
        mat_id (int): Unique material identifier.
        elastic_modulus (float): Young's modulus (E).
        poisson_ratio (float): Poisson's ratio (nu), optional.
        density (float): Mass density, optional.
        name (str): Material name for identification.
    """

    def __init__(self, mat_id, elastic_modulus, poisson_ratio=0.3,
                 density=0.0, name=""):
        """
        Create a material.

        Inputs:
            mat_id (int): Unique identifier.
            elastic_modulus (float): Young's modulus E (must be > 0).
            poisson_ratio (float): Poisson's ratio (default 0.3).
            density (float): Mass density (default 0.0).
            name (str): Descriptive name.

        Raises:
            ValueError: If E <= 0.
        """
        if elastic_modulus <= 0:
            raise ValueError(
                f"Elastic modulus must be > 0, got {elastic_modulus}"
            )
        self.mat_id = mat_id
        self.elastic_modulus = float(elastic_modulus)
        self.poisson_ratio = float(poisson_ratio)
        self.density = float(density)
        self.name = name

    @property
    def E(self):
        """Shorthand for elastic modulus."""
        return self.elastic_modulus

    @property
    def nu(self):
        """Shorthand for Poisson's ratio."""
        return self.poisson_ratio

    def __repr__(self):
        return (
            f"Material(id={self.mat_id}, E={self.elastic_modulus:.2e}, "
            f"nu={self.poisson_ratio}, name='{self.name}')"
        )
