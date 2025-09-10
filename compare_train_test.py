import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import fetch_weekly_nifty, extract_extrema
from RecFIF import build_recurrent_fif_with_dvec
from coordinate_tuner import simple_tuner
from wCF import weighted_cf_from_omegas
from COS_inversion_helper import cos_invert_cf_from_cffunc


def compute_train_fit_errors(train_df: pd.DataFrame, f_open_fn) -> dict:
    t_vals = train_df['t'].values
    open_actual = train_df['Open'].values
    open_fit = f_open_fn(t_vals)
    residuals = open_fit - open_actual
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mape = float(np.mean(np.abs(residuals / open_actual)) * 100.0)
    return {
        'train_open_mae': mae,
        'train_open_rmse': rmse,
        'train_open_mape_pct': mape
    }


def predict_week_close(t_val: float, basis_fns, params_noise, open_level: float, Ncos: int = 512, L: int = 6):
    omegas = np.array([float(fn(t_val)) for fn in basis_fns])
    cf_func = lambda u: weighted_cf_from_omegas(u, omegas, params_noise)
    _, _, _, inv_cdf = cos_invert_cf_from_cffunc(cf_func, Ncos=Ncos, L=L)
    q025 = float(inv_cdf(0.025))
    q500 = float(inv_cdf(0.50))
    q975 = float(inv_cdf(0.975))
    close_med = open_level * (1.0 + q500)
    close_low = open_level * (1.0 + q025)
    close_high = open_level * (1.0 + q975)
    return close_med, close_low, close_high


def main_compare(
    train_end: str = "2022-01-21",
    test_end: str = "2022-01-28",
    start: str = "2007-09-17",
    m: int = 2,
    basis_iters: int = 80,
    basis_grid: int = 600,
    Ncos: int = 512,
    L: int = 6,
    output_csv: str = "compare_metrics.csv",
    output_png: str = "compare_plot.png",
):
    # 1) Load train and test data
    df_train = fetch_weekly_nifty(start=start, end=train_end)
    df_all = fetch_weekly_nifty(start=start, end=test_end)
    if len(df_train) < 10 or len(df_all) <= len(df_train):
        raise RuntimeError("Insufficient data for train/test split.")

    test_row = df_all.iloc[-1].copy()

    # 2) Knots from training Open
    open_train = df_train['Open'].values
    idx = extract_extrema(open_train, order=1)
    knots_t = df_train['t'].values[idx]
    knots_y = open_train[idx]
    K = len(knots_t)

    # 3) Get d_vec: try pre-tuned else tune fast
    d_vec = None
    try:
        d_loaded = np.load('d_vec_opt.npy')
        if d_loaded.shape[0] == K - 1:
            d_vec = d_loaded.astype(float)
            print(f"Using pre-tuned d_vec_opt.npy with shape {d_vec.shape}")
        else:
            print(f"d_vec_opt.npy shape {d_loaded.shape} != K-1={K-1}; tuning quickly...")
    except Exception:
        pass
    if d_vec is None:
        d_vec = simple_tuner(
            knots_t, knots_y, df_train, m=m, passes=1,
            candidate_factors=[0.9, 1.0, 1.1], fast_iters=60, fast_grid=400
        )

    # 4) Precompute basis functions
    basis_fns = []
    for i in range(K):
        y_basis = np.zeros(K); y_basis[i] = 1.0
        _, _, f_basis_fn, _ = build_recurrent_fif_with_dvec(
            knots_t, y_basis, d_vec, m=m, iters=basis_iters, grid_len=basis_grid, verbose=False
        )
        basis_fns.append(f_basis_fn)

    # 5) High-res interpolant for Open and compute train fit errors
    _, _, f_open_fn, _ = build_recurrent_fif_with_dvec(
        knots_t, knots_y, d_vec, m=m, iters=max(200, basis_iters), grid_len=max(1200, basis_grid), verbose=False
    )
    train_errs = compute_train_fit_errors(df_train, f_open_fn)

    # 6) CTS params from training returns
    returns_train = (df_train['Close'].values - df_train['Open'].values) / df_train['Open'].values
    params_noise = (1.4, 1e-3, 1e-3, 9.5, 9.5, float(np.mean(returns_train)))

    # 7) Test week predictions
    t_test = float(test_row['t'])
    open_pred_test = float(f_open_fn(t_test))
    close_med, close_low, close_high = predict_week_close(t_test, basis_fns, params_noise, open_pred_test, Ncos=Ncos, L=L)
    close_actual = float(test_row['Close'])
    abs_err = abs(close_med - close_actual)
    rel_err_pct = abs_err / close_actual * 100.0 if close_actual != 0 else np.nan
    covered = (close_actual >= close_low) and (close_actual <= close_high)

    # NEW: Test Open error for comparison
    open_actual_test = float(test_row['Open'])
    test_open_abs_error = abs(open_pred_test - open_actual_test)
    test_open_rel_error_pct = test_open_abs_error / open_actual_test * 100.0 if open_actual_test != 0 else np.nan

    # 8) Save metrics CSV
    metrics = {
        'train_rows': len(df_train),
        'train_open_mae': train_errs['train_open_mae'],
        'train_open_rmse': train_errs['train_open_rmse'],
        'train_open_mape_pct': train_errs['train_open_mape_pct'],
        'test_date': pd.to_datetime(test_row['Date']).strftime('%Y-%m-%d'),
        'test_open_actual': open_actual_test,
        'test_open_pred': open_pred_test,
        'test_open_abs_error': test_open_abs_error,
        'test_open_rel_error_pct': test_open_rel_error_pct,
        'close_pred_med': close_med,
        'close_band_low': close_low,
        'close_band_high': close_high,
        'close_actual': close_actual,
        'test_abs_error': abs_err,
        'test_rel_error_pct': rel_err_pct,
        'test_covered_in_95_band': covered,
        'K': K,
        'Ncos': Ncos,
        'L': L,
        'm': m,
        'basis_iters': basis_iters,
        'basis_grid': basis_grid,
    }
    pd.DataFrame([metrics]).to_csv(output_csv, index=False)
    print(f"Saved metrics to {output_csv}")

    # 9) Plot: train Open vs RFIF fit; test week with band
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    # Train fit
    axes[0].plot(df_train['Date'], df_train['Open'].values, label='Open (actual)', alpha=0.6)
    axes[0].plot(df_train['Date'], f_open_fn(df_train['t'].values), label='RFIF interpolant (Open)', lw=1.5)
    axes[0].set_title('Training Fit: Open (actual) vs RFIF interpolant')
    axes[0].legend()

    # Test week band
    # Show a small window of last 10 train weeks + test week
    tail_n = 10
    df_tail = df_all.iloc[-(tail_n+1):].copy()  # includes test
    axes[1].plot(df_tail['Date'], df_tail['Close'].values, label='Close (actual)', marker='o')
    # Band only for last point (test)
    axes[1].fill_between([df_tail['Date'].iloc[-1], df_tail['Date'].iloc[-1]],
                         [close_low], [close_high], color='gray', alpha=0.3, label='95% band')
    axes[1].scatter([df_tail['Date'].iloc[-1]], [close_med], color='red', label='Predicted Close (median)')
    axes[1].axhline(close_actual, color='green', linestyle='--', alpha=0.7, label='Actual Close (test)')
    axes[1].set_title(f"Test Week {pd.to_datetime(test_row['Date']).strftime('%Y-%m-%d')} — Open abs={test_open_abs_error:.1f} ({test_open_rel_error_pct:.2f}%), Close abs={abs_err:.1f} ({rel_err_pct:.2f}%)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches='tight')
    print(f"Saved plot to {output_png}")

    return metrics


if __name__ == "__main__":
    out = main_compare()
    for k, v in out.items():
        print(f"{k}: {v}")
