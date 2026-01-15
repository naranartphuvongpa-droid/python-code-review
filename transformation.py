import math
import numpy as np
from translation_matrix_weft import rotation_matrix_weft
from translation_matrix_binder import translation_matrix_binder_parabolic
import stiffness_matrix
def transformation():
    C = stiffness_matrix.stiffness_matrix()  
    # From warp to weft 
    T_tensor_weft_inv, T_eps_weft, T_eps_weft_inv_transpose = rotation_matrix_weft(math.pi/2)
    C_weft = T_tensor_weft_inv @ C @ T_eps_weft
    # From warp to binder
    theta, length, T_tensor_binder, T_eps_binder, T_tensor_binder_inv, T_eps_binder_inv_transpose = translation_matrix_binder_parabolic(10, 0, 2, 20)
    C_binder = []
    for i in range(len(T_tensor_binder)):
        C_binder.append(T_tensor_binder_inv[i] @ C @ T_eps_binder[i])
    return  C_binder,C_weft,len(C_weft), len(C_binder)
