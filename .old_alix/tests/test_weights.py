import numpy as np

from src.ssm.weights import (
    PeriodicLinearSystem,
    build_periodic_linear_system_from_steady_state,
    periodic_impulse_weights,
    flatten_weights,
)

def test_build_periodic_linear_system_shapes():
    # simple fake periodic gains with varying observation dimension
    m = 3
    n_state = 4

    G = np.eye(n_state)

    # Z dims per phase: [1, 1, 2]
    Z_list = [
        np.random.randn(1, n_state),
        np.random.randn(1, n_state),
        np.random.randn(2, n_state),
    ]

    # K dims must match: (n_state x obs_dim_j)
    K_list = [
        np.random.randn(n_state, 1),
        np.random.randn(n_state, 1),
        np.random.randn(n_state, 2),
    ]

    system = build_periodic_linear_system_from_steady_state(G=G, K_list=K_list, Z_list=Z_list)
    assert system.m == m
    assert len(system.F_list) == m
    assert len(system.B_list) == m

    for j in range(m):
        assert system.F_list[j].shape == (n_state, n_state)
        assert system.B_list[j].shape == (n_state, Z_list[j].shape[0])


def test_periodic_impulse_weights_dimensions_vary_by_phase():
    # Construct an arbitrary periodic linear system
    m = 3
    n_state = 2

    # F_j all stable-ish
    F_list = [
        0.5 * np.eye(n_state),
        0.4 * np.eye(n_state),
        0.3 * np.eye(n_state),
    ]

    # obs dims per phase: [1, 1, 2]
    B_list = [
        np.ones((n_state, 1)),
        2.0 * np.ones((n_state, 1)),
        3.0 * np.ones((n_state, 2)),
    ]

    system = PeriodicLinearSystem(m=m, F_list=F_list, B_list=B_list)

    # target = first component of state at horizon h
    C_target = np.array([[1.0, 0.0]])  # (1 x n_state)

    W_list = periodic_impulse_weights(
        system=system,
        C_target=C_target,
        horizon_steps=2,
        K_lags=6,
        start_phase=0
    )

    assert len(W_list) == 6
    # k=0 uses phase 0 (obs dim 1), k=1 uses phase 2? Wait: phase = (start_phase - k) mod m
    # start_phase=0 => phases for k=0..5: 0,2,1,0,2,1 => dims: 1,2,1,1,2,1
    expected_dims = [1, 2, 1, 1, 2, 1]
    for k, Wk in enumerate(W_list):
        assert Wk.shape == (1, expected_dims[k])


def test_flatten_weights_places_blocks_correctly():
    m = 3
    obs_dims = [1, 1, 2]
    total_d = sum(obs_dims)

    # create fake weights for K_lags=4
    # start_phase=0 => phases for k=0..3: 0,2,1,0
    W_list = [
        np.array([[10.0]]),          # phase 0, dim1
        np.array([[1.0, 2.0]]),      # phase 2, dim2
        np.array([[20.0]]),          # phase 1, dim1
        np.array([[30.0]]),          # phase 0, dim1
    ]

    out = flatten_weights(W_list, obs_dims, system_m=m, start_phase=0)

    assert out.shape == (1, 4 * total_d)

    # offsets within one "super-vector"
    offsets = [0, 1, 2]  # phase0 starts 0, phase1 starts 1, phase2 starts 2

    # lag 0 -> phase 0 -> write at columns [0+0]
    assert out[0, 0 + offsets[0]] == 10.0
    # other cols within lag0 should be zero
    assert out[0, 0 + offsets[1]] == 0.0
    assert out[0, 0 + offsets[2] + 0] == 0.0
    assert out[0, 0 + offsets[2] + 1] == 0.0

    # lag 1 -> phase 2 -> base col = total_d
    base = total_d
    assert np.allclose(out[0, base + offsets[2]: base + offsets[2] + 2], [1.0, 2.0])
    assert out[0, base + offsets[0]] == 0.0
    assert out[0, base + offsets[1]] == 0.0

    # lag 2 -> phase 1 -> base col = 2*total_d
    base = 2 * total_d
    assert out[0, base + offsets[1]] == 20.0

    # lag 3 -> phase 0 -> base col = 3*total_d
    base = 3 * total_d
    assert out[0, base + offsets[0]] == 30.0


def test_oracle_case_h0_weights_match_direct_B_when_F_zero():
    # If all F_j = 0 and horizon_steps=0:
    # target = C * a_0, but y_{-k} affects a_0 only for k=0 via B_0 y_0 in recursion
    # Our formulation with predicted-state recursion implies weights for k>0 should be zero
    m = 2
    n_state = 3

    F_list = [np.zeros((n_state, n_state)), np.zeros((n_state, n_state))]
    B_list = [np.random.randn(n_state, 1), np.random.randn(n_state, 2)]
    system = PeriodicLinearSystem(m=m, F_list=F_list, B_list=B_list)

    C = np.random.randn(1, n_state)

    W_list = periodic_impulse_weights(
        system=system,
        C_target=C,
        horizon_steps=0,
        K_lags=5,
        start_phase=0
    )

    # k=0 uses phase 0: W0 should be C @ B0
    assert np.allclose(W_list[0], C @ B_list[0])

    # For k>=1, propagation uses only F's (all zero), so should be ~0
    for k in range(1, 5):
        assert np.max(np.abs(W_list[k])) < 1e-12
