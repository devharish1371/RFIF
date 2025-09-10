import numpy as np
from scipy import interpolate

def compute_dk_perp_heuristic(knots_t, knots_y, ell, r, sensitivity=0.5, dmin=0.05, dmax=0.99):
    t_dom = knots_t[ell:r+1]
    y_dom = knots_y[ell:r+1]
    if len(t_dom) < 2:
        return 0.5
    A = np.vstack([t_dom, np.ones_like(t_dom)]).T
    sol, *_ = np.linalg.lstsq(A, y_dom, rcond=None)
    alpha, beta = sol[0], sol[1]
    y_fit = alpha * t_dom + beta
    residuals = np.abs(y_dom - y_fit)
    max_res = np.max(residuals)
    y_range = np.max(y_dom) - np.min(y_dom)
    res_norm = max_res / y_range if y_range > 0 else 0.0
    d_k = 1.0 - sensitivity * res_norm
    return float(max(dmin, min(dmax, d_k)))

def build_recurrent_fif_with_dvec(knots_t, knots_y, d_vec, m=2, iters=300, grid_len=1500, verbose=False):
    K = len(knots_t)
    t0, tN = knots_t[0], knots_t[-1]
    maps = []
    for k in range(1, K):
        ell = max(0, (k-1) - m)
        r = min(K-1, (k-1) + m + 1)
        t_l, t_r = knots_t[ell], knots_t[r]
        a_k = (knots_t[k] - knots_t[k-1]) / (t_r - t_l + 1e-16)
        b_k = knots_t[k-1] - a_k * t_l
        d_k = float(d_vec[k-1])  # map k uses d_vec[k-1]
        y_l, y_r = knots_y[ell], knots_y[r]
        A = np.array([[t_l, 1.0],[t_r, 1.0]])
        rhs = np.array([knots_y[k-1] - d_k * y_l, knots_y[k] - d_k * y_r])
        try:
            sol = np.linalg.solve(A, rhs)
            c_k, e_k = float(sol[0]), float(sol[1])
        except np.linalg.LinAlgError:
            c_k, e_k = 0.0, float(knots_y[k-1])
        maps.append({'k_idx': k,'ell': ell,'r': r,
                     'a': float(a_k),'b': float(b_k),
                     'd': float(d_k),'c': float(c_k),'e': float(e_k),
                     't_l': float(t_l),'t_r': float(t_r)})
    # grid & initial f
    t_grid = np.linspace(t0, tN, grid_len)
    f = np.interp(t_grid, knots_t, knots_y)
    domain_t_arrays, mapped_x_arrays = [], []
    for mp in maps:
        mask = (t_grid >= mp['t_l']) & (t_grid <= mp['t_r'])
        domain_t = t_grid[mask]
        domain_t_arrays.append((mask, domain_t))
        mapped_x_arrays.append(mp['a'] * domain_t + mp['b'])
    # iteration
    for it in range(iters):
        f_new = np.zeros_like(f); count = np.zeros_like(f)
        for idx_map, mp in enumerate(maps):
            mask, domain_t = domain_t_arrays[idx_map]
            x = mapped_x_arrays[idx_map]
            d_k, c_k, e_k = mp['d'], mp['c'], mp['e']
            f_s = np.interp(domain_t, t_grid, f)
            v = c_k * domain_t + d_k * f_s + e_k
            deposited = np.interp(t_grid, x, v, left=0.0, right=0.0)
            contributed_mask = deposited != 0.0
            f_new += deposited
            count += contributed_mask.astype(float)
        nonzero = count > 0
        f_new[nonzero] = f_new[nonzero] / count[nonzero]
        f_new[~nonzero] = f[~nonzero]
        maxdiff = np.max(np.abs(f_new - f))
        f = f_new
        if verbose and (it % 50 == 0 or it == iters-1):
            print(f"Iter {it+1}/{iters}, maxdiff={maxdiff:.6g}")
        if maxdiff < 1e-9:
            break
    f_fn = interpolate.interp1d(t_grid, f, bounds_error=False, fill_value="extrapolate")
    return t_grid, f, f_fn, maps