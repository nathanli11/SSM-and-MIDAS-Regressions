import numpy as np

from src.ssm.state_space_builder import build_mixedfreq_dynamic_factor_ar1_me
from src.ssm.steady_state_riccati import solve_periodic_steady_state

def _is_psd(A: np.ndarray, tol: float = 1e-8) -> bool:
    # Symmetrize then check eigenvalues
    A = 0.5*(A + A.T)
    w = np.linalg.eigvalsh(A)
    return bool(np.min(w) > -tol)

def test_periodic_riccati_converges_and_is_periodic():
    m = 3
    #nf = 1
    #p = 1
    #n = 3  # y + x2 + x3

    Phi_list = [np.array([[0.9]])]
    Sigma_eta = np.array([[1.0]])
    Gamma = np.array([[1.0], [0.5], [1.5]])
    d_u = np.array([0.2, 0.1, 0.3])
    sigma_eps = np.array([1.0, 1.0, 1.0])

    ssm = build_mixedfreq_dynamic_factor_ar1_me(
        m=m, Phi_list=Phi_list, Sigma_eta=Sigma_eta,
        Gamma=Gamma, d_u=d_u, sigma_eps=sigma_eps,
        include_low_freq=True
    )

    Q_tilde = ssm.R @ ssm.Q @ ssm.R.T

    ss = solve_periodic_steady_state(
        G=ssm.G, Q_tilde=Q_tilde,
        Z_list=ssm.Z_list,
        tol=1e-10, max_iter=20000, jitter=1e-12
    )

    assert len(ss.P_pred) == m
    assert len(ss.K) == m
    assert len(ss.S) == m
    assert len(ss.A) == m

    # Check symmetry + PSD-ish
    for j in range(m):
        assert np.allclose(ss.P_pred[j], ss.P_pred[j].T, atol=1e-8)
        assert _is_psd(ss.P_pred[j], tol=1e-6)

    # Check "cycle consistency": apply one full cycle starting from ss.P_pred and return close
    P = [Pj.copy() for Pj in ss.P_pred]
    for j in range(m):
        Z = ssm.Z_list[j]
        # update
        ZP = Z @ P[j]
        S = ZP @ Z.T + 1e-12*np.eye(Z.shape[0])
        K = np.linalg.solve(S, ZP).T
        I = np.eye(P[j].shape[0])
        P_upd = 0.5*((I - K@Z) @ P[j] + ((I - K@Z) @ P[j]).T)
        # predict
        P_next = 0.5*(ssm.G @ P_upd @ ssm.G.T + Q_tilde + (ssm.G @ P_upd @ ssm.G.T + Q_tilde).T)
        P[(j+1) % m] = P_next

    # Compare after one full cycle
    for j in range(m):
        assert np.max(np.abs(P[j] - ss.P_pred[j])) < 1e-7
