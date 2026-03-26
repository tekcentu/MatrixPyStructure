"""
structure - 2D Structural Analysis Package
==========================================

A modular structural analysis package for 2D truss and frame structures.
Uses the custom matrix library for all matrix operations and linear
system solution.

Modules:
    node        - Node (joint) definition
    material    - Material properties (E, etc.)
    section     - Cross-section properties (A, I)
    element     - Element types (Truss2D, Frame2D)
    load        - Applied loads and load cases
    boundary    - Boundary condition handling
    model       - Structural model assembly
    analysis    - Static analysis solver
"""

from structure.node import Node
from structure.material import Material
from structure.section import Section
from structure.element import Truss2D, Frame2D
from structure.load import NodalLoad, LoadCase
from structure.boundary import BoundaryCondition
from structure.model import StructuralModel
from structure.analysis import StaticAnalysis

__all__ = [
    "Node",
    "Material",
    "Section",
    "Truss2D",
    "Frame2D",
    "NodalLoad",
    "LoadCase",
    "BoundaryCondition",
    "StructuralModel",
    "StaticAnalysis",
]
