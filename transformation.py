import math
import numpy as np

from translation_matrix_weft import rotation_matrix_weft
from translation_matrix_binder import translation_matrix_binder_parabolic, binder_fixed_angle_theta
import stiffness_matrix


# Voigt order: [11, 22, 33, 23, 13, 12]
_voigt_pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

def _build_A_tensor_from_R(R):
    """
    Build 6x6 matrix A for transforming symmetric 2nd-order tensors in
    tensor-Voigt (NO engineering shear).
    vec(S') = A vec(S), where S' = R S R^T
    """
    A = np.zeros((6, 6), dtype=float)
    for I, (i, j) in enumerate(_voigt_pairs):
        for J, (p, q) in enumerate(_voigt_pairs):
            A[I, J] = R[i, p] * R[j, q] + (R[i, q] * R[j, p] if p != q else 0.0)
    return A

def _R_y(theta):
    """Rotation about y-axis by theta (radians). Binder lies in x-z plane."""
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[ c, 0.0,  s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0,  c]], dtype=float)

def _rotate_C_engineering(C_eng, R):
    """
    Rotate stiffness matrix given in engineering-shear Voigt form.
    Steps:
      - convert to tensor-shear form using Reuter matrix
      - rotate using A (tensor rotation)
      - convert back to engineering-shear
    """
    C_eng = np.array(C_eng, dtype=float)

    # engineering shear gamma = 2 * tensor shear
    Reuter = np.diag([1, 1, 1, 2, 2, 2]).astype(float)
    Reuter_inv = np.diag([1, 1, 1, 0.5, 0.5, 0.5]).astype(float)

    A = _build_A_tensor_from_R(R)
    A_inv = np.linalg.inv(A)

    # Convert C from engineering-shear to tensor-shear on the strain side
    C_tensor = C_eng @ Reuter

    # Rotate
    C_tensor_rot = A @ C_tensor @ A_inv

    # Back to engineering-shear
    C_eng_rot = C_tensor_rot @ Reuter_inv
    return C_eng_rot

def transformation(binder_angle_scheme="orthogonal", binder_angle_deg=90.0, num_nodes=20, debug_print_angle=False):
    """
    Returns:
      C_binder (list of 6x6)
      C_weft (6x6)
      length (list)
      len(C_weft), len(C_binder)  [kept to match your older return style]
    """
    # Base stiffness (warp)
    C = stiffness_matrix.stiffness_matrix()

    # warp -> weft (90 deg)
    T_tensor_weft_inv, T_eps_weft, _ = rotation_matrix_weft(math.pi/2)
    C_weft = T_tensor_weft_inv @ C @ T_eps_weft

    # Decide binder theta/length based on scheme
    scheme = str(binder_angle_scheme).strip().lower()

    if scheme == "orthogonal":
        # True orthogonal: 90 deg from x-axis
        theta, length = binder_fixed_angle_theta(90.0, num_nodes=num_nodes)

    elif scheme == "inclined_fixed":
        # Fixed angle from Excel (measured from x-axis)
        theta, length = binder_fixed_angle_theta(binder_angle_deg, num_nodes=num_nodes)

    elif scheme == "parabolic":
        # Keep old behaviour if you want it explicitly
        theta, length, *_ = translation_matrix_binder_parabolic(10, 0, 2, num_nodes)

    else:
        # Safe default: treat unknown as inclined_fixed
        theta, length = binder_fixed_angle_theta(binder_angle_deg, num_nodes=num_nodes)

    if debug_print_angle:
        theta_deg = [math.degrees(t) for t in theta]
        print("binder theta deg (from x-axis): min/max =", min(theta_deg), max(theta_deg), "first5 =", theta_deg[:5])

    # Rotate warp stiffness into each binder segment orientation
    C_binder = []
    for th in theta:
        R = _R_y(th)
        Cb = _rotate_C_engineering(C, R)
        C_binder.append(Cb)

    return C_binder, C_weft, length, len(C_weft), len(C_binder)



