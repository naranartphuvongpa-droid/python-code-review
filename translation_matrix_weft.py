import math
import numpy as np
def rotation_matrix_weft(theta_1):
    # Let the warp be the original orientation then;
    c = math.cos(theta_1) # this is to transform from warp to weft
    s = math.sin(theta_1)
    T_tensor_weft= np.zeros((6, 6))
    T_eps_weft = np.zeros((6, 6)) 
    T_tensor_weft = ([
                [c**2, s**2, 0, 0, 0, 2*s*c],
                [s**2, c**2, 0, 0, 0, -2*s*c],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, c, -s, 0],
                [0, 0, 0, s, c, 0],
                [-s*c, s*c, 0, 0, 0, c**2 - s**2]
                ]) 
    T_eps_weft = ([
                [c**2, s**2, 0, 0, 0, c*s],
                [s**2, c**2, 0, 0, 0, -c*s],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, c, -s, 0],
                [0, 0, 0, s, c, 0],
                [-2*s*c, 2*s*c, 0, 0, 0, c**2 - s**2]
                ]) 

    T_tensor_weft_inv = np.linalg.inv(T_tensor_weft)
    T_eps_weft_inv = np.linalg.inv(T_eps_weft)
    T_eps_weft_inv_transpose = T_eps_weft_inv.T
    return T_tensor_weft_inv,T_eps_weft , T_eps_weft_inv_transpose

