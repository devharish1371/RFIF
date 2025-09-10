import numpy as np
from cts_char import cts_char_exponent

def weighted_cf_from_omegas(u, omegas, params_noise):
    # psi_sum(u) = sum_i psi_i(omega_i * u) ; here we assume same params per knot
    psi = np.zeros_like(u, dtype=complex)
    for w in omegas:
        if abs(w) < 1e-15:
            continue
        psi += cts_char_exponent(u * w, params_noise)
    return np.exp(psi)