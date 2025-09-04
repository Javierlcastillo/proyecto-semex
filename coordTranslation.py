import numpy as np

def fit_similarity_2d(P, U):
    """
    P: Nx2 array of Python points [[x,y],...]
    U: Nx2 array of Unity points  [[X,Z],...]
    returns (s, theta_rad, tx, tz)
    """
    P = np.asarray(P, float); U = np.asarray(U, float)
    Pc = P - P.mean(axis=0, keepdims=True)
    Uc = U - U.mean(axis=0, keepdims=True)

    # rotation via SVD
    H = Pc.T @ Uc
    U_svd, S, Vt = np.linalg.svd(H)
    R = U_svd @ Vt
    if np.linalg.det(R) < 0:            # prevent reflection
        U_svd[:, -1] *= -1
        R = U_svd @ Vt

    # uniform scale
    s = (np.trace((R.T @ H))) / (np.sum(Pc**2))

    # translation
    t = U.mean(axis=0) - s * (R @ P.mean(axis=0))

    theta = np.arctan2(R[1,0], R[0,0])
    return float(s), float(theta), float(t[0]), float(t[1])

# Example: fill with the 3+ matched points you measured
P_py = [[385,333],[554,497],[426,547]]
U_un = [[0,0],[-11,52],[-37,37]]
s, theta, tx, tz = fit_similarity_2d(P_py, U_un)
print("scale:", s, "theta(rad):", theta, "offset:", tx, tz)
