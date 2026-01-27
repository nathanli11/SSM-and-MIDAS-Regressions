import numpy as np

from src.ssm.state_space_builder import build_mixedfreq_dynamic_factor_ar1_me
from src.ssm.steady_state_riccati import solve_periodic_steady_state
from src.ssm.kalman_periodic import kalman_filter_periodic

def test_kalman_periodic_runs_with_steady_state():
    m = 3
    #nf = 1
    #p = 1
    #n = 2  # y + x2

    Phi_list = [np.array([[0.8]])]
    Sigma_eta = np.array([[1.0]])
    Gamma = np.array([[1.0], [1.0]])
    d_u = np.array([0.2, 0.2])
    sigma_eps = np.array([1.0, 1.0])

    ssm = build_mixedfreq_dynamic_factor_ar1_me(
        m=m, Phi_list=Phi_list, Sigma_eta=Sigma_eta,
        Gamma=Gamma, d_u=d_u, sigma_eps=sigma_eps,
        include_low_freq=True
    )
    Q_tilde = ssm.R @ ssm.Q @ ssm.R.T

    ss = solve_periodic_steady_state(
        G=ssm.G, Q_tilde=Q_tilde,
        Z_list=ssm.Z_list,
        tol=1e-10, max_iter=20000
    )

    # 2 cycles => 6 steps ; dims: j=1,2 -> 1 obs ; j=3 -> 2 obs
    y_seq = [
        np.array([0.1]),
        np.array([0.2]),
        np.array([0.05, 0.15]),
        np.array([0.0]),
        np.array([0.1]),
        np.array([0.02, 0.11]),
    ]

    out = kalman_filter_periodic(
        y_seq=y_seq,
        G=ssm.G, Q_tilde=Q_tilde,
        Z_list=ssm.Z_list,
        steady_state=ss
    )

    Tsteps = len(y_seq)
    n_state = ssm.G.shape[0]
    assert out.x_filt.shape == (Tsteps, n_state)
    assert out.P_filt.shape == (Tsteps, n_state, n_state)
    assert len(out.innov) == Tsteps
    assert len(out.S) == Tsteps

    assert np.isfinite(out.x_filt).all()
    assert np.isfinite(out.P_filt).all()
