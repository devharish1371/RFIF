import numpy as np
from scipy.special import gamma

def cts_char_exponent(u, params):
    # params: (alpha, C_plus, C_minus, lam_p, lam_m, mu)
    alpha, C_plus, C_minus, lam_p, lam_m, mu = params
    z1 = (lam_p - 1j*u)
    z2 = (lam_m + 1j*u)
    term_p = np.power(z1, alpha) - (lam_p**alpha)
    term_m = np.power(z2, alpha) - (lam_m**alpha)
    pref = gamma(-alpha)
    psi = 1j*u*mu + pref * (C_plus * term_p + C_minus * term_m)
    return psi

def cts_cf(u, params):
    return np.exp(cts_char_exponent(u, params))