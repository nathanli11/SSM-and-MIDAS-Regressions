import numpy as np

from src.midas.almon import exp_almon_weights, exp_almon_weights_and_grad

def test_almon_weights_basic_properties():
    w = exp_almon_weights(theta1=-0.1, theta2=-0.01, K=20, start_at_one=True)
    assert w.shape == (20,)
    assert np.all(w >= 0)
    assert abs(float(np.sum(w)) - 1.0) < 1e-12

def test_almon_weights_numerical_stability_large_params():
    # Very large magnitude parameters should not overflow due to stabilization
    w = exp_almon_weights(theta1=100.0, theta2=-10.0, K=50, start_at_one=True)
    assert np.isfinite(w).all()
    assert abs(float(np.sum(w)) - 1.0) < 1e-12

def test_almon_grad_shapes_and_sums():
    w, grad = exp_almon_weights_and_grad(theta1=-0.2, theta2=-0.01, K=30)
    assert w.shape == (30,)
    assert grad.shape == (30, 2)

    # Since sum_k w_k = 1 for all theta, sum_k dw_k/dtheta = 0
    assert abs(float(np.sum(grad[:, 0]))) < 1e-10
    assert abs(float(np.sum(grad[:, 1]))) < 1e-10
