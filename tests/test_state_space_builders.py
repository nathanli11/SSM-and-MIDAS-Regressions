import numpy as np

from src.ssm.state_space_builder import build_mixedfreq_dynamic_factor_ar1_me

def test_builder_dimensions_and_pattern():
    m = 3
    nf = 2
    p = 2
    n = 4  # y + x2 + x3 + x4

    Phi_list = [0.5*np.eye(nf), 0.2*np.eye(nf)]
    Sigma_eta = np.eye(nf)
    Gamma = np.arange(n*nf, dtype=float).reshape(n, nf) / 10.0  # deterministic
    d_u = np.array([0.1, 0.2, 0.3, 0.4])
    sigma_eps = np.ones(n)

    ssm = build_mixedfreq_dynamic_factor_ar1_me(
        m=m, Phi_list=Phi_list, Sigma_eta=Sigma_eta,
        Gamma=Gamma, d_u=d_u, sigma_eps=sigma_eps,
        include_low_freq=True,
        low_freq_name="y",
        high_freq_names=["x2", "x3", "x4"],
    )

    dim_state = nf*p + n
    assert ssm.G.shape == (dim_state, dim_state)
    assert ssm.R.shape[0] == dim_state
    assert ssm.Q.shape == (nf + n, nf + n)

    assert len(ssm.Z_list) == m
    assert len(ssm.obs_names_list) == m

    # j=1,2: only x2..x4 => obs dim = n-1
    assert ssm.Z_list[0].shape == (n-1, dim_state)
    assert ssm.Z_list[1].shape == (n-1, dim_state)
    assert ssm.obs_names_list[0] == ["x2", "x3", "x4"]

    # j=3: y + x2..x4 => obs dim = n
    assert ssm.Z_list[2].shape == (n, dim_state)
    assert ssm.obs_names_list[2] == ["y", "x2", "x3", "x4"]

    # Check that y-row loads on factor current block and on its own u component
    u_start = nf*p
    y_row = ssm.Z_list[2][0, :]          # first row is y
    assert np.allclose(y_row[0:nf], Gamma[0, :])
    assert y_row[u_start + 0] == 1.0

    # Check that x2 row (in last subperiod) corresponds to series index 1
    x2_row = ssm.Z_list[2][1, :]
    assert np.allclose(x2_row[0:nf], Gamma[1, :])
    assert x2_row[u_start + 1] == 1.0
