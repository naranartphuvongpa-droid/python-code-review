import numpy as np
from transformation import transformation
from stiffness_matrix import stiffness_matrix
def summation_of_stiffnessmatrix(Vf_warp,Vf_weft,Vf_binder):
    C_warp = stiffness_matrix()
    C_binder, C_weft, length, _, _ = transformation()
    C_total = []
    for i in range(len(C_binder)):
        C_sum = Vf_warp*C_warp + Vf_weft*C_weft + Vf_binder*C_binder[i]
        C_total.append(C_sum)
    weights = np.array(length) / np.sum(length)
    C_total_avg = np.tensordot(weights, C_total, axes=(0,0))
    np.set_printoptions(precision=3, suppress=True)
    print(C_total_avg)
    return C_total_avg
summation_of_stiffnessmatrix(0.5, 0.3, 0.2)0.2)
