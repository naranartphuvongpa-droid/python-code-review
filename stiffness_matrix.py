import math
import numpy as np
import chamis_function as chamis
E1, V12, E2, V23, G23, G12, E3, V13, G13, al11, al12 = chamis.chamis_function(0)
def stiffness_matrix():
    S = np.zeros((6,6))
    S[0][0] = 1/E1
    S[0][1] = -V12/E1
    S[0][2] = -V13/E1
    S[1][0] = - V12/E1
    S[1][1] = 1/E2
    S[1][2] = - V23/E2
    S[2][0] = - V13/E1
    S[2][1] = - V23/E2
    S[2][2] = 1/E3  
    S[3][3] = 1/G23
    S[4][4] = 1/G13
    S[5][5] = 1/G12
    C = np.linalg.inv(S)
    return C
