"""
element.py - Finite Element Types (Truss2D and Frame2D)
========================================================

Purpose:
    Implements 2D truss and frame finite elements. Each element computes
    its local stiffness matrix, transforms it to global coordinates,
    and provides methods for recovering member forces.

Element Types:
    Truss2D - 2-node bar element (axial force only)
        DOFs per node: 2 (ux, uy)
        Total DOFs: 4

    Frame2D - 2-node beam-column element (axial + bending)
        DOFs per node: 3 (ux, uy, rz)
        Total DOFs: 6

Coordinate Systems:
    Local: x-axis along element from node_i to node_j.
    Global: Standard Cartesian x-y system.

Transformation:
    T = rotation matrix from local to global coordinates.
    k_global = T^T * k_local * T

Units:
    All units must be consistent (e.g., kN and m, or lb and in).
"""

import math
from matrix.dense_matrix import DenseMatrix
from matrix.vector import Vector


class Truss2D:
    """
    2D truss (bar) element with 2 nodes and 4 DOFs.

    A truss element carries only axial force. The local stiffness
    matrix is:
        k_local = (E*A/L) * [ 1 -1]
                              [-1  1]

    Attributes:
        elem_id (int): Unique element identifier.
        node_i (Node): Start node.
        node_j (Node): End node.
        material (Material): Element material.
        section (Section): Element cross-section.
    """

    DOFS_PER_NODE = 2
    TOTAL_DOFS = 4

    def __init__(self, elem_id, node_i, node_j, material, section):
        """
        Create a 2D truss element.

        Inputs:
            elem_id (int): Unique element identifier.
            node_i (Node): Start node.
            node_j (Node): End node.
            material (Material): Material properties.
            section (Section): Cross-section properties.
        """
        self.elem_id = elem_id
        self.node_i = node_i
        self.node_j = node_j
        self.material = material
        self.section = section

    @property
    def length(self):
        """
        Compute element length.

        Returns:
            float: Distance between node_i and node_j.
        """
        return self.node_i.distance_to(self.node_j)

    @property
    def cos_angle(self):
        """
        Direction cosine (cos theta) of element axis.

        Returns:
            float: cos(theta) = (x_j - x_i) / L.
        """
        L = self.length
        return (self.node_j.x - self.node_i.x) / L

    @property
    def sin_angle(self):
        """
        Direction sine (sin theta) of element axis.

        Returns:
            float: sin(theta) = (y_j - y_i) / L.
        """
        L = self.length
        return (self.node_j.y - self.node_i.y) / L

    def get_dof_indices(self):
        """
        Get the global DOF indices for this element.

        Returns:
            list[int]: Global DOF indices [ui_x, ui_y, uj_x, uj_y].

        Note:
            Requires that assign_dofs() has been called on both nodes.
        """
        return self.node_i.dof_indices + self.node_j.dof_indices

    def local_stiffness(self):
        """
        Compute local stiffness matrix (2x2 in axial direction).

        Returns:
            DenseMatrix: 4x4 local stiffness matrix in local coordinates.

        Formula:
            k_local = (EA/L) * [ 1  0 -1  0]
                                [ 0  0  0  0]
                                [-1  0  1  0]
                                [ 0  0  0  0]
        """
        E = self.material.E
        A = self.section.A
        L = self.length
        k_coeff = E * A / L

        k = DenseMatrix(4)
        k.set(0, 0, k_coeff)
        k.set(0, 2, -k_coeff)
        k.set(2, 0, -k_coeff)
        k.set(2, 2, k_coeff)
        return k

    def transformation_matrix(self):
        """
        Compute the coordinate transformation matrix T.

        Returns:
            DenseMatrix: 4x4 transformation matrix.

        The transformation matrix T rotates from global to local:
            T = [c  s  0  0]
                [-s c  0  0]
                [0  0  c  s]
                [0  0 -s  c]

        where c = cos(theta), s = sin(theta).
        """
        c = self.cos_angle
        s = self.sin_angle

        T = DenseMatrix(4)
        T.set(0, 0, c);  T.set(0, 1, s)
        T.set(1, 0, -s); T.set(1, 1, c)
        T.set(2, 2, c);  T.set(2, 3, s)
        T.set(3, 2, -s); T.set(3, 3, c)
        return T

    def global_stiffness(self):
        """
        Compute global stiffness matrix: k_global = T^T * k_local * T.

        Returns:
            DenseMatrix: 4x4 element stiffness matrix in global coordinates.

        Note:
            For efficiency, computes k_global directly using the
            analytical expression rather than matrix multiplication.
        """
        E = self.material.E
        A = self.section.A
        L = self.length
        c = self.cos_angle
        s = self.sin_angle
        k_coeff = E * A / L

        cc = c * c * k_coeff
        ss = s * s * k_coeff
        cs = c * s * k_coeff

        k = DenseMatrix(4)
        # Row 0
        k.set(0, 0, cc);  k.set(0, 1, cs);  k.set(0, 2, -cc); k.set(0, 3, -cs)
        # Row 1
        k.set(1, 0, cs);  k.set(1, 1, ss);  k.set(1, 2, -cs); k.set(1, 3, -ss)
        # Row 2
        k.set(2, 0, -cc); k.set(2, 1, -cs); k.set(2, 2, cc);  k.set(2, 3, cs)
        # Row 3
        k.set(3, 0, -cs); k.set(3, 1, -ss); k.set(3, 2, cs);  k.set(3, 3, ss)

        return k

    def member_forces(self, global_displacements):
        """
        Compute member forces from global displacements.

        Inputs:
            global_displacements (Vector): Global displacement vector (size 4)
                for this element's DOFs: [ui_x, ui_y, uj_x, uj_y].

        Returns:
            float: Axial force in the member.
                   Positive = tension, Negative = compression.

        Algorithm:
            1. Transform global displacements to local: u_local = T * u_global
            2. Axial force = (EA/L) * (u_local_j_x - u_local_i_x)
        """
        c = self.cos_angle
        s = self.sin_angle
        E = self.material.E
        A = self.section.A
        L = self.length

        # Local displacements by direct transformation
        u1 = c * global_displacements[0] + s * global_displacements[1]
        u3 = c * global_displacements[2] + s * global_displacements[3]

        # Axial force
        axial_force = (E * A / L) * (u3 - u1)
        return axial_force

    def __repr__(self):
        return (
            f"Truss2D(id={self.elem_id}, "
            f"nodes=[{self.node_i.node_id}, {self.node_j.node_id}], "
            f"L={self.length:.4f})"
        )


class Frame2D:
    """
    2D frame (beam-column) element with 2 nodes and 6 DOFs.

    Carries axial force, shear force, and bending moment.
    DOFs per node: [ux, uy, rz] (2 translations + 1 rotation).

    Attributes:
        elem_id (int): Unique element identifier.
        node_i (Node): Start node.
        node_j (Node): End node.
        material (Material): Element material.
        section (Section): Element cross-section.
    """

    DOFS_PER_NODE = 3
    TOTAL_DOFS = 6

    def __init__(self, elem_id, node_i, node_j, material, section):
        """
        Create a 2D frame element.

        Inputs:
            elem_id (int): Unique element identifier.
            node_i (Node): Start node.
            node_j (Node): End node.
            material (Material): Material properties (needs E).
            section (Section): Cross-section properties (needs A and I).
        """
        self.elem_id = elem_id
        self.node_i = node_i
        self.node_j = node_j
        self.material = material
        self.section = section

    @property
    def length(self):
        """
        Compute element length.

        Returns:
            float: Distance between node_i and node_j.
        """
        return self.node_i.distance_to(self.node_j)

    @property
    def cos_angle(self):
        """
        Direction cosine (cos theta) of element axis.

        Returns:
            float: cos(theta) = (x_j - x_i) / L.
        """
        L = self.length
        return (self.node_j.x - self.node_i.x) / L

    @property
    def sin_angle(self):
        """
        Direction sine (sin theta) of element axis.

        Returns:
            float: sin(theta) = (y_j - y_i) / L.
        """
        L = self.length
        return (self.node_j.y - self.node_i.y) / L

    def get_dof_indices(self):
        """
        Get the global DOF indices for this element.

        Returns:
            list[int]: Global DOFs [ui_x, ui_y, ri_z, uj_x, uj_y, rj_z].

        Note:
            Requires that assign_dofs() has been called on both nodes.
        """
        return self.node_i.dof_indices + self.node_j.dof_indices

    def local_stiffness(self):
        """
        Compute 6x6 local stiffness matrix in local coordinates.

        Returns:
            DenseMatrix: 6x6 local stiffness matrix.

        Formula:
            Combines axial stiffness (EA/L) and bending stiffness (EI/L^3).

            DOF order: [u1, v1, theta1, u2, v2, theta2]
            where u = axial, v = transverse, theta = rotation.

            Axial part:
                (EA/L) at positions (0,0), (0,3), (3,0), (3,3)

            Bending part:
                12EI/L^3, 6EI/L^2, 4EI/L, 2EI/L terms for
                transverse and rotation DOFs.
        """
        E = self.material.E
        A = self.section.A
        I = self.section.I
        L = self.length
        L2 = L * L
        L3 = L2 * L

        ea_l = E * A / L
        ei_l3 = E * I / L3

        k = DenseMatrix(6)

        # Axial stiffness
        k.set(0, 0, ea_l);   k.set(0, 3, -ea_l)
        k.set(3, 0, -ea_l);  k.set(3, 3, ea_l)

        # Bending stiffness
        k.set(1, 1, 12 * ei_l3)
        k.set(1, 2, 6 * ei_l3 * L)
        k.set(1, 4, -12 * ei_l3)
        k.set(1, 5, 6 * ei_l3 * L)

        k.set(2, 1, 6 * ei_l3 * L)
        k.set(2, 2, 4 * ei_l3 * L2)
        k.set(2, 4, -6 * ei_l3 * L)
        k.set(2, 5, 2 * ei_l3 * L2)

        k.set(4, 1, -12 * ei_l3)
        k.set(4, 2, -6 * ei_l3 * L)
        k.set(4, 4, 12 * ei_l3)
        k.set(4, 5, -6 * ei_l3 * L)

        k.set(5, 1, 6 * ei_l3 * L)
        k.set(5, 2, 2 * ei_l3 * L2)
        k.set(5, 4, -6 * ei_l3 * L)
        k.set(5, 5, 4 * ei_l3 * L2)

        return k

    def transformation_matrix(self):
        """
        Compute the 6x6 coordinate transformation matrix T.

        Returns:
            DenseMatrix: 6x6 transformation matrix.

        Structure:
            T = [R  0]    where R = [ c  s  0]
                [0  R]              [-s  c  0]
                                    [ 0  0  1]
        """
        c = self.cos_angle
        s = self.sin_angle

        T = DenseMatrix(6)
        # Block 1 (node i)
        T.set(0, 0, c);  T.set(0, 1, s)
        T.set(1, 0, -s); T.set(1, 1, c)
        T.set(2, 2, 1.0)
        # Block 2 (node j)
        T.set(3, 3, c);  T.set(3, 4, s)
        T.set(4, 3, -s); T.set(4, 4, c)
        T.set(5, 5, 1.0)
        return T

    def global_stiffness(self):
        """
        Compute global stiffness matrix: k_global = T^T * k_local * T.

        Returns:
            DenseMatrix: 6x6 element stiffness matrix in global coordinates.
        """
        k_local = self.local_stiffness()
        T = self.transformation_matrix()
        T_t = T.transpose()

        # k_global = T^T * k_local * T
        temp = T_t.mat_mul(k_local)
        k_global = temp.mat_mul(T)
        return k_global

    def member_forces(self, global_displacements):
        """
        Compute member end forces from global displacements.

        Inputs:
            global_displacements (Vector): Global displacement vector (size 6)
                for this element's DOFs.

        Returns:
            Vector: Local member end forces (size 6):
                [N_i, V_i, M_i, N_j, V_j, M_j]
                N = axial force, V = shear force, M = bending moment.

        Algorithm:
            1. Transform to local: u_local = T * u_global
            2. Compute forces: f_local = k_local * u_local
        """
        T = self.transformation_matrix()
        k_local = self.local_stiffness()

        # Transform displacements to local
        u_local = T.mat_vec(global_displacements)
        # Compute local forces
        f_local = k_local.mat_vec(u_local)

        return f_local

    def __repr__(self):
        return (
            f"Frame2D(id={self.elem_id}, "
            f"nodes=[{self.node_i.node_id}, {self.node_j.node_id}], "
            f"L={self.length:.4f})"
        )
