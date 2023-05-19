import sympy as sp


c1, c2, T, phi, r1, r2, t, b1, b2, m1, m2, h1, h2 =   \
   sp.symbols("c1, c2, T, phi, r1, r2, t, b1, b2, m1, m2, h1, h2 ", real=True)
alpha, omega, theta, S1, S2, k1, k2, k3, k4, k5, k6, k_d, x1, x1_0, x2, x2_0 =   \
    sp.symbols("alpha, omega, theta, S1, S2, k1, k2, k3, k4, k5, k6, k_d, x1, x1_0, x2, x2_0 ", real=True, positive=True, nonzero=True)
I = sp.I