import numpy as np
from scipy import interpolate

def cos_invert_cf_from_cffunc(cf_func, Ncos=2048, L=8):
    # numerical moments (small-u derivatives)
    eps = 1e-6
    phi0 = cf_func(0.0)
    phi1 = cf_func(eps); phim1 = cf_func(-eps)
    phi2 = cf_func(2*eps); phim2 = cf_func(-2*eps)
    phi_prime = (phi1 - phim1)/(2*eps)
    phi_double = (phi2 - 2*phi0 + phim2)/(eps**2)
    mean = float(np.real(-1j * phi_prime))
    var = float(np.real(-phi_double) - mean**2)
    if not np.isfinite(var) or var <= 0:
        var = 1e-6
    std = np.sqrt(var)
    a = mean - L*std; b = mean + L*std
    M = int(Ncos)
    k = np.arange(M)
    x_grid = a + (np.arange(M) + 0.5) * (b - a) / M
    omega_k = k * np.pi / (b - a)
    phi_vals = cf_func(omega_k)
    exp_term = np.exp(-1j * omega_k * a)
    Uk = 2.0/(b - a) * (phi_vals * exp_term).real
    Uk[0] *= 0.5
    # reconstruct pdf
    cos_matrix = np.cos(np.outer(omega_k, x_grid - a))
    pdf_vals = np.dot(Uk, cos_matrix)
    pdf_vals = np.maximum(pdf_vals, 0.0)
    dx = (b - a) / M
    cdf_vals = np.cumsum(pdf_vals) * dx
    cdf_vals = np.clip(cdf_vals, 0.0, 1.0)
    # enforce strict monotonicity for inverse
    for i in range(1, len(cdf_vals)):
        if cdf_vals[i] <= cdf_vals[i-1]:
            cdf_vals[i] = cdf_vals[i-1] + 1e-12
    inv_cdf = interpolate.interp1d(cdf_vals, x_grid, bounds_error=False, fill_value=(x_grid[0], x_grid[-1]))
    return x_grid, pdf_vals, cdf_vals, inv_cdf