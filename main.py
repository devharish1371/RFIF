import time
import warnings
warnings.filterwarnings("ignore")
from utils import fetch_weekly_nifty, extract_extrema
from RecFIF import build_recurrent_fif_with_dvec
from coordinate_tuner import simple_tuner
from cts_char import cts_char_exponent
from wCF import weighted_cf_from_omegas
from COS_inversion_helper import cos_invert_cf_from_cffunc
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate



def main_propagation(full_run=True, sample_every=1, Ncos_per_week=1024, basis_iters=250, basis_grid=1400, m=2):
    """
    full_run: if False, will run a sampled quick run (sample_every > 1)
    sample_every: process every k-th week (useful to check quickly)
    Ncos_per_week: COS terms per week (increase to 2048/4096 for final)
    basis_iters / basis_grid: resolution to precompute basis functions
    """
    print("Loading weekly NIFTY data...")
    df = fetch_weekly_nifty()
    open_prices = df['Open'].values
    idx = extract_extrema(open_prices, order=1)
    knots_t = df['t'].values[idx]; knots_y = open_prices[idx]
    K = len(knots_t)
    print(f"Knots selected: {K}")

    # try to find tuned d_vec_opt in current globals
    global_vars = globals()
    if 'd_vec_opt' in global_vars:
        d_vec = d_vec_opt
        print("Using existing d_vec_opt from environment.")
    else:
        print("No d_vec_opt found. Running fallback tuner (this will take a while)...")
        d_vec = simple_tuner(knots_t, knots_y, df, m=m, passes=1, candidate_factors=[0.8,0.9,1.0,1.1], fast_iters=120, fast_grid=800)
        print("Tuning completed (fallback).")

    # Precompute basis recurrent-FIFs (one per knot)
    print(f"Precomputing {K} basis recurrent-FIFs (iters={basis_iters}, grid={basis_grid}) ...")
    t0_time = time.time()
    basis_fns = []
    for i in range(K):
        y_basis = np.zeros(K); y_basis[i] = 1.0
        _, _, f_basis_fn, _ = build_recurrent_fif_with_dvec(knots_t, y_basis, d_vec, m=m, iters=basis_iters, grid_len=basis_grid, verbose=False)
        basis_fns.append(f_basis_fn)
        if (i+1) % 50 == 0 or (i+1) == K:
            print(f"  basis {i+1}/{K} done")
    elapsed = time.time() - t0_time
    print(f"Basis precompute finished in {elapsed:.1f}s")

    # compute high-res interpolant for open prices using tuned d_vec (for band transformation)
    _, _, f_fn_full, _ = build_recurrent_fif_with_dvec(knots_t, knots_y, d_vec, m=m, iters=400, grid_len=2500, verbose=False)
    interp_open_vals = f_fn_full(df['t'].values)

    # need params_hat in environment (CTS params). If missing, fit coarse CTS quickly here:
    if 'params_hat' in global_vars:
        params_noise = params_hat
    else:
        print("Warning: params_hat (CTS fit) not found in environment. Fitting CTS (coarse) to returns now...")
        returns = (df['Close'].values - df['Open'].values) / df['Open'].values
        # quick heuristic params: alpha ~1.4, C small, lambda ~ 9.5, mu mean
        params_noise = (1.4, 1e-3, 1e-3, 9.5, 9.5, float(np.mean(returns)))
        print("Using heuristic CTS params (consider replacing with fitted params_hat for accuracy).")

    # per-week propagation (optionally sampled)
    n = len(df)
    weeks_idx = np.arange(0, n, sample_every)
    q_low = np.zeros(n); q_high = np.zeros(n)
    t_start = time.time()
    for counter, i in enumerate(weeks_idx):
        tval = df['t'].values[i]
        # compute omegas from basis functions
        omegas = np.array([float(fn(tval)) for fn in basis_fns])
        # define CF function for this week
        cf_func = lambda u: weighted_cf_from_omegas(u, omegas, params_noise)
        # COS inversion (moderate resolution)
        _, pdf_w, cdf_w, inv_cdf_w = cos_invert_cf_from_cffunc(cf_func, Ncos=Ncos_per_week, L=8)
        q_low[i] = float(inv_cdf_w(0.025)); q_high[i] = float(inv_cdf_w(0.975))
        if (counter+1) % 50 == 0:
            print(f"Propagated {counter+1}/{len(weeks_idx)} weeks (index {i})")
    total_time = time.time() - t_start
    print(f"Propagation finished in {total_time:.1f}s (sample_every={sample_every}, Ncos={Ncos_per_week})")

    # Fill in any weeks not computed (if sampled) via linear interpolation on quantiles
    computed_idx = weeks_idx
    # For simplicity, fill zeros for uncomputed weeks then interpolate
    if sample_every > 1:
        # simple linear interpolation over indices
        valid = computed_idx
        qlow_vals = q_low[valid]; qhigh_vals = q_high[valid]
        qlow_interp = interpolate.interp1d(valid, qlow_vals, bounds_error=False, fill_value="extrapolate")
        qhigh_interp = interpolate.interp1d(valid, qhigh_vals, bounds_error=False, fill_value="extrapolate")
        for j in range(n):
            if j not in valid:
                q_low[j] = float(qlow_interp(j)); q_high[j] = float(qhigh_interp(j))

    # convert to close bands
    close_low = interp_open_vals * (1 + q_low)
    close_high = interp_open_vals * (1 + q_high)
    closes = df['Close'].values
    coverage = np.mean((closes >= close_low) & (closes <= close_high))*100
    print(f"Coverage of actual closes inside 95% bands: {coverage:.2f}%")

    # Plots
    plt.figure(figsize=(14,6))
    plt.plot(df['Date'], df['Open'].values, label='Open (actual)', alpha=0.6)
    plt.plot(df['Date'], interp_open_vals, label='Tuned recurrent FIF interpolant', lw=1.0)
    plt.fill_between(df['Date'], close_low, close_high, color='gray', alpha=0.25, label='95% per-week bands')
    plt.plot(df['Date'], df['Close'].values, label='Close (actual)', alpha=0.9)
    plt.legend(); plt.title(f'Per-week CTS propagation — Coverage: {coverage:.2f}%'); plt.show()

    return {
        'df': df,
        'knots_idx': idx,
        'd_vec': d_vec,
        'interp_open': interp_open_vals,
        'close_low': close_low,
        'close_high': close_high,
        'coverage': coverage
    }

# ----------------------------
# 8) Run main propagation
# ----------------------------
if __name__ == "__main__":
    # Parameters you may wish to change for a faster check:
    # - full_run=False and sample_every=4 (check every 4th week)
    # - Ncos_per_week=1024 for quick run (increase to 2048/4096 for better tails)
    d_vec_opt = np.load("/Users/dev1371/Downloads/RFIF/d_vec_opt.npy")  # load pre-tuned d_vec if available
    out = main_propagation(full_run=True, sample_every=1, Ncos_per_week=1024, basis_iters=250, basis_grid=1400, m=2)
    print("Done. Coverage:", out['coverage'])
