import numpy as np

def resin_matrix_stiffness_isotropic(E, nu):
    lam = (nu * E) / ((1 + nu) * (1 - 2 * nu))
    mu  = E / (2 * (1 + nu))

    C = np.zeros((6, 6))
    C[0:3, 0:3] = lam
    np.fill_diagonal(C[0:3, 0:3], lam + 2*mu)

    C[3, 3] = mu
    C[4, 4] = mu
    C[5, 5] = mu
    return C


