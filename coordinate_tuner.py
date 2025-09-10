import numpy as np
from RecFIF import compute_dk_perp_heuristic, build_recurrent_fif_with_dvec

def simple_tuner(knots_t, knots_y, df, m=2, init_sensitivity=0.5, dmin=0.05, dmax=0.99,
                 passes=1, candidate_factors=[0.75, 0.9, 1.0, 1.1], fast_iters=100, fast_grid=800):
    K = len(knots_t)
    d_vec = np.array([compute_dk_perp_heuristic(knots_t, knots_y, max(0,(k-1)-m), min(K-1,(k-1)+m+1),
                                               sensitivity=init_sensitivity, dmin=dmin, dmax=dmax)
                      for k in range(1, K)])
    # quick initial eval
    _, _, f_fn0, _ = build_recurrent_fif_with_dvec(knots_t, knots_y, d_vec, m=m, iters=fast_iters, grid_len=fast_grid)
    base_mape = np.mean(np.abs(f_fn0(df['t'].values) - df['Open'].values)/df['Open'].values)*100
    print("Tuner: initial MAPE (fast):", base_mape)
    for p in range(passes):
        for k in range(len(d_vec)):
            best = d_vec[k]; best_mape = base_mape
            for fac in candidate_factors:
                cand = float(np.clip(d_vec[k] * fac, dmin, dmax))
                d_try = d_vec.copy(); d_try[k] = cand
                _, _, f_fn_try, _ = build_recurrent_fif_with_dvec(knots_t, knots_y, d_try, m=m, iters=fast_iters, grid_len=fast_grid)
                mape_try = np.mean(np.abs(f_fn_try(df['t'].values) - df['Open'].values)/df['Open'].values)*100
                if mape_try < best_mape:
                    best_mape = mape_try; best = cand
            d_vec[k] = best
            base_mape = best_mape
    return d_vec
