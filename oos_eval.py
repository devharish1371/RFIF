import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from utils import fetch_weekly_nifty, extract_extrema
from RecFIF import build_recurrent_fif_with_dvec
from coordinate_tuner import simple_tuner
from wCF import weighted_cf_from_omegas
from COS_inversion_helper import cos_invert_cf_from_cffunc


def evaluate_oos(
    start: str = "2007-09-17",
    end: str = "2022-01-24",
    m: int = 2,
    basis_iters: int = 150,
    basis_grid: int = 800,
    Ncos: int = 1024,
    L: int = 8,
    max_basis_knots: int | None = None,
):
    """
    Train on all but the last available week; predict the last week (held-out) and
    report absolute/relative error of Close and band coverage.

    If max_basis_knots is provided, only the nearest that many knots around the
    test time are used for basis precomputation to speed up evaluation.
    """
    # 1) Load full weekly data
    df_full = fetch_weekly_nifty(start=start, end=end)
    if len(df_full) < 10:
        raise RuntimeError("Not enough data points fetched for OOS evaluation.")

    # 2) Define train/test split
    train_df = df_full.iloc[:-1].copy()
    test_row = df_full.iloc[-1].copy()
    t_test = float(test_row['t'])

    # 3) Build knots from training Open
    open_train = train_df['Open'].values
    idx_train = extract_extrema(open_train, order=1)
    knots_t_full = train_df['t'].values[idx_train]
    knots_y_full = open_train[idx_train]

    # Optionally pick nearest subset around test time
    if max_basis_knots is not None and max_basis_knots < len(knots_t_full):
        order_idx = np.argsort(np.abs(knots_t_full - t_test))
        keep_idx = np.sort(order_idx[:max_basis_knots])
        knots_t = knots_t_full[keep_idx]
        knots_y = knots_y_full[keep_idx]
    else:
        knots_t = knots_t_full
        knots_y = knots_y_full
    K_train = len(knots_t)

    # 4) Obtain d_vec for training knots: use pre-tuned if available, else tune quickly
    d_vec = None
    try:
        d_vec_loaded = np.load('d_vec_opt.npy')
        if d_vec_loaded.shape[0] == K_train - 1:
            d_vec = d_vec_loaded.astype(float)
            print(f"Using pre-tuned d_vec_opt.npy with shape {d_vec.shape}")
        else:
            print(f"Pre-tuned d_vec_opt.npy shape {d_vec_loaded.shape} does not match K_train-1={K_train-1}; tuning...")
    except Exception:
        pass
    if d_vec is None:
        d_vec = simple_tuner(
            knots_t,
            knots_y,
            train_df,
            m=m,
            passes=1,
            candidate_factors=[0.9, 1.0, 1.1],
            fast_iters=80,
            fast_grid=500,
        )

    # 5) Precompute basis recurrent-FIFs on training knots (or subset)
    basis_fns = []
    for i in range(K_train):
        y_basis = np.zeros(K_train)
        y_basis[i] = 1.0
        _, _, f_basis_fn, _ = build_recurrent_fif_with_dvec(
            knots_t,
            y_basis,
            d_vec,
            m=m,
            iters=basis_iters,
            grid_len=basis_grid,
            verbose=False,
        )
        basis_fns.append(f_basis_fn)

    # 6) Interpolant for Open on training and extrapolate to t_test
    _, _, f_open_fn, _ = build_recurrent_fif_with_dvec(
        knots_t,
        knots_y,
        d_vec,
        m=m,
        iters=max(300, basis_iters),
        grid_len=max(1200, basis_grid),
        verbose=False,
    )
    open_pred_test = float(f_open_fn(t_test))

    # 7) Set CTS noise params (heuristic, replace with fitted if available)
    returns_train = (train_df['Close'].values - train_df['Open'].values) / train_df['Open'].values
    params_noise = (
        1.4,
        1e-3,
        1e-3,
        9.5,
        9.5,
        float(np.mean(returns_train)),
    )

    # 8) Build week-specific CF via omegas at t_test and invert with COS
    omegas = np.array([float(fn(t_test)) for fn in basis_fns])
    cf_func = lambda u: weighted_cf_from_omegas(u, omegas, params_noise)
    _, _, _, inv_cdf = cos_invert_cf_from_cffunc(cf_func, Ncos=Ncos, L=L)

    q025 = float(inv_cdf(0.025))
    q500 = float(inv_cdf(0.50))
    q975 = float(inv_cdf(0.975))

    # 9) Convert return quantiles to Close level bands using predicted Open
    close_pred_med = open_pred_test * (1.0 + q500)
    close_low = open_pred_test * (1.0 + q025)
    close_high = open_pred_test * (1.0 + q975)

    # 10) Compare to actual test Close
    close_actual = float(test_row['Close'])
    abs_err = abs(close_pred_med - close_actual)
    rel_err_pct = abs_err / close_actual * 100.0 if close_actual != 0 else np.nan
    covered = (close_actual >= close_low) and (close_actual <= close_high)

    results = {
        'train_rows': int(len(train_df)),
        'test_date': pd.to_datetime(test_row['Date']).strftime('%Y-%m-%d'),
        'open_pred_test': float(open_pred_test),
        'close_pred_med': float(close_pred_med),
        'close_band_low': float(close_low),
        'close_band_high': float(close_high),
        'close_actual': float(close_actual),
        'abs_error': float(abs_err),
        'rel_error_pct': float(rel_err_pct),
        'covered_in_95_band': bool(covered),
        'K_train': int(K_train),
        'Ncos': int(Ncos),
        'L': int(L),
        'm': int(m),
        'basis_iters': int(basis_iters),
        'basis_grid': int(basis_grid),
        'max_basis_knots': int(max_basis_knots) if max_basis_knots is not None else None,
    }
    return results


if __name__ == "__main__":
    out = evaluate_oos()
    print("Out-of-sample evaluation (one-week ahead):")
    for k, v in out.items():
        print(f"{k}: {v}")
