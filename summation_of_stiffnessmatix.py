import numpy as np
from transformation import transformation
from stiffness_matrix import stiffness_matrix
from resin_stiffness import resin_matrix_stiffness_isotropic as resin_stiffness

def sym_err(M):
    """Max absolute asymmetry: max(|M - M^T|). 0 = perfectly symmetric."""
    M = np.array(M, dtype=float)
    return float(np.max(np.abs(M - M.T)))

def summation_of_stiffnessmatrix(Vf_warp, Vf_weft, Vf_binder, Vf_resin=1):

    """
    Calling for resin stiffness matrix
    """
    C_resin = resin_stiffness(100000000,1000000000)

    """
    Weighted-average stiffness summation across binder segments.
    C_sum(i) = Vf_warp*C_warp + Vf_weft*C_weft + Vf_binder*C_binder[i]
    then C_total_avg = sum_i w_i * C_sum(i), where w_i = length_i / sum(length)
    """
    C_warp = stiffness_matrix()
    C_binder, C_weft, length, _, _ = transformation()

    # Build C_total list
    C_total = []
    for i in range(len(C_binder)):
        C_sum = Vf_warp * C_warp + Vf_weft * C_weft + Vf_resin * C_resin + Vf_binder * C_binder[i]
        C_total.append(C_sum)

    # Convert to arrays
    C_total = np.array(C_total, dtype=float)          # shape: (nseg, 6, 6)
    length = np.array(length, dtype=float).reshape(-1)

    # Safety checks
    if len(length) != len(C_total):
        raise ValueError(f"length size ({len(length)}) != number of binder segments ({len(C_total)})")
    if np.sum(length) == 0:
        raise ValueError("Sum of binder segment lengths is zero.")

    # Weighted average
    weights = length / np.sum(length)                 # shape: (nseg,)
    C_total_avg = np.tensordot(weights, C_total, axes=(0, 0))  # shape: (6, 6)

    np.set_printoptions(precision=3, suppress=True)
    print("\nC_total_avg (weighted):")
    print(C_total_avg)

    return C_total_avg

def check_symmetry(Vf_warp=0.5, Vf_weft=0.3, Vf_binder=0.2):
    """Print symmetry diagnostics for each stage."""
    C_warp = stiffness_matrix()
    C_binder, C_weft, length, _, _ = transformation()

    print("\n=== Symmetry diagnostics ===")
    print("sym_err C_warp:", sym_err(C_warp))
    print("sym_err C_weft:", sym_err(C_weft))

    binder_errs = [sym_err(Cb) for Cb in C_binder]
    worst_idx = int(np.argmax(binder_errs))
    print("sym_err C_binder[0]:", binder_errs[0])
    print("sym_err C_binder worst:", max(binder_errs), "at index", worst_idx)

    i = 18
    Cb = C_binder[i]
    D = Cb - Cb.T
    print("worst asym at binder index:", i)
    print("max|Cb-Cb.T| =", np.max(np.abs(D)))

    r, c = np.unravel_index(np.argmax(np.abs(D)), D.shape)
    print("worst entry (r,c) =", (r,c), "diff =", D[r,c])
    print("Cb[r,c] =", Cb[r,c], "Cb[c,r] =", Cb[c,r])

    # Also check after summation
    C_total_avg = summation_of_stiffnessmatrix(Vf_warp, Vf_weft, Vf_binder)
    print("sym_err C_total_avg:", sym_err(C_total_avg))

    # Helpful extra info
    length = np.array(length, dtype=float).reshape(-1)
    print("\nBinder segment length min/max:", float(length.min()), float(length.max()))

if __name__ == "__main__":
    check_symmetry()


