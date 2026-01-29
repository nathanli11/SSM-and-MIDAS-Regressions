import numpy as np


def simulate_one_factor_dgp(T=40, m=3, rho=0.9, d=0.5, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    # high frequency length
    Th = T*m

    # Innovations
    eta = np.random.normal(size=Th)
    eps1 = np.random.normal(size=Th)
    eps2 = np.random.normal(size=Th)

    # Facteur Latent
    f = np.zeros(Th)
    for t in range(1, Th):
        f[t] = rho * f[t-1] + eta[t]
    
    # Measurement errors
    u1 = np.zeros(Th)
    u2 = np.zeros(Th)
    for t in range(1, Th):
        u1[t] = d * u1[t-1] + eps1[t]
        u2[t] = d * u2[t-1] + eps2[t]
    
    # Observations
    y_star = f + u1
    x = f + u2

    # Low frequency aggregation (stock variable)
    y = y_star[m-1::m]

    return y, x.reshape(-1,1), f


def simulate_two_factor_dgp(
        T=40,
        m=3,
        rho1=0.9,
        rho2=0.5,
        d=0.5,
        seed=None
):
    if seed is not None:
        np.random.seed(seed)
    Th = T * m

    # Innovations
    eta1 = np.random.normal(size=Th)
    eta2 = np.random.normal(size=Th)
    epsy = np.random.normal(size=Th)
    epsx = np.random.normal(size=Th)

    # Factors
    f1 = np.zeros(Th)
    f2 = np.zeros(Th)
    for t in range(1, Th):
        f1[t] = rho1 * f1[t - 1] + eta1[t]
        f2[t] = rho2 * f2[t - 1] + eta2[t]
    
    # Measurment errors
    uy = np.zeros(Th)
    ux = np.zeros(Th)
    for t in range(1, Th):
        uy[t] = d * uy[t - 1] + epsy[t]
        ux[t] = d * ux[t - 1] + epsx[t]
    
    # Observations
    y_star = f1 + f2 + uy
    x = f1 + ux

    # Low frequency
    y = y_star[m-1::m]

    return y, x.reshape(-1,1), f1, f2

