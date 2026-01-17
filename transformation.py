import math
import numpy as np
from translation_matrix_weft import rotation_matrix_weft
from translation_matrix_binder import translation_matrix_binder_parabolic
import stiffness_matrix

# =========================
# Helper for binder rotation
# Voigt order: [11, 22, 33, 23, 13, 12]
# =========================
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
    """Rotation about y-axis by theta (radians). Binder in your model lies in x-z plane."""
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
    This preserves symmetry.
    """
    C_eng = np.array(C_eng, dtype=float)

    # engineering shear gamma = 2 * tensor shear
    Reuter = np.diag([1, 1, 1, 2, 2, 2]).astype(float)
    Reuter_inv = np.diag([1, 1, 1, 0.5, 0.5, 0.5]).astype(float)

    A = _build_A_tensor_from_R(R)

    C_tensor = C_eng @ Reuter
    A_inv = np.linalg.inv(A)
    C_tensor_rot = A @ C_tensor @ A_inv
    C_eng_rot = C_tensor_rot @ Reuter_inv

    return C_eng_rot

def transformation():
    C = stiffness_matrix.stiffness_matrix()

    # ---------- warp -> weft (unchanged) ----------
    T_tensor_weft_inv, T_eps_weft, T_eps_weft_inv_transpose = rotation_matrix_weft(math.pi/2)
    C_weft = T_tensor_weft_inv @ C @ T_eps_weft

    # ---------- warp -> binder (FIXED) ----------
    theta, length, T_tensor_binder, T_eps_binder, T_tensor_binder_inv, T_eps_binder_inv_transpose = \
        translation_matrix_binder_parabolic(10, 0, 2, 20)

    # We only use theta here (do NOT use T_* binder matrices)
    C_binder = []
    for th in theta:  # theta is in radians
        R = _R_y(th)
        Cb = _rotate_C_engineering(C, R)
        C_binder.append(Cb)

    return C_binder, C_weft, length, len(C_weft), len(C_binder)



