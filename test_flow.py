#!/usr/bin/env python3
"""
Test script to verify the RFIF (Recurrent Fractal Interpolation Function) flow works correctly.
This script tests all major components of the system.
"""

import sys
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def test_imports():
    """Test that all required modules can be imported."""
    print("=" * 60)
    print("TESTING IMPORTS")
    print("=" * 60)
    
    try:
        from utils import fetch_weekly_nifty, extract_extrema
        print("✓ utils module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import utils: {e}")
        return False
    
    try:
        from RecFIF import build_recurrent_fif_with_dvec, compute_dk_perp_heuristic
        print("✓ RecFIF module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import RecFIF: {e}")
        return False
    
    try:
        from coordinate_tuner import simple_tuner
        print("✓ coordinate_tuner module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import coordinate_tuner: {e}")
        return False
    
    try:
        from cts_char import cts_char_exponent, cts_cf
        print("✓ cts_char module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import cts_char: {e}")
        return False
    
    try:
        from wCF import weighted_cf_from_omegas
        print("✓ wCF module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import wCF: {e}")
        return False
    
    try:
        from COS_inversion_helper import cos_invert_cf_from_cffunc
        print("✓ COS_inversion_helper module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import COS_inversion_helper: {e}")
        return False
    
    print("✓ All imports successful!")
    return True

def test_data_fetching():
    """Test data fetching functionality."""
    print("\n" + "=" * 60)
    print("TESTING DATA FETCHING")
    print("=" * 60)
    
    try:
        from utils import fetch_weekly_nifty, extract_extrema
        
        print("Fetching NIFTY data...")
        df = fetch_weekly_nifty()
        print(f"✓ Data fetched successfully: {len(df)} rows")
        print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"  Columns: {list(df.columns)}")
        
        # Test extrema extraction
        open_prices = df['Open'].values
        idx = extract_extrema(open_prices, order=1)
        print(f"✓ Extrema extraction successful: {len(idx)} knots found")
        
        return df, idx
        
    except Exception as e:
        print(f"✗ Data fetching failed: {e}")
        return None, None

def test_recurrent_fif():
    """Test the recurrent FIF construction."""
    print("\n" + "=" * 60)
    print("TESTING RECURRENT FIF CONSTRUCTION")
    print("=" * 60)
    
    try:
        from RecFIF import build_recurrent_fif_with_dvec, compute_dk_perp_heuristic
        
        # Create simple test data
        knots_t = np.array([0.0, 0.3, 0.7, 1.0])
        knots_y = np.array([1.0, 1.5, 0.8, 1.2])
        
        # Test d_k computation
        d_k = compute_dk_perp_heuristic(knots_t, knots_y, 0, 3)
        print(f"✓ d_k computation successful: {d_k}")
        
        # Create d_vec
        d_vec = np.array([0.5, 0.6, 0.7])
        
        # Test FIF construction
        print("Building recurrent FIF...")
        t_grid, f, f_fn, maps = build_recurrent_fif_with_dvec(
            knots_t, knots_y, d_vec, m=2, iters=100, grid_len=500, verbose=True
        )
        
        print(f"✓ FIF construction successful:")
        print(f"  Grid size: {len(t_grid)}")
        print(f"  Number of maps: {len(maps)}")
        print(f"  Function range: [{np.min(f):.3f}, {np.max(f):.3f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Recurrent FIF test failed: {e}")
        return False

def test_coordinate_tuning():
    """Test coordinate tuning functionality."""
    print("\n" + "=" * 60)
    print("TESTING COORDINATE TUNING")
    print("=" * 60)
    
    try:
        from coordinate_tuner import simple_tuner
        
        # Create test data
        knots_t = np.linspace(0, 1, 10)
        knots_y = np.sin(2 * np.pi * knots_t) + 0.1 * np.random.randn(10)
        
        # Create mock dataframe
        import pandas as pd
        df = pd.DataFrame({
            't': knots_t,
            'Open': knots_y
        })
        
        print("Running coordinate tuner...")
        d_vec = simple_tuner(
            knots_t, knots_y, df, m=2, passes=1, 
            candidate_factors=[0.8, 0.9, 1.0, 1.1], 
            fast_iters=50, fast_grid=200
        )
        
        print(f"✓ Coordinate tuning successful:")
        print(f"  d_vec shape: {d_vec.shape}")
        print(f"  d_vec range: [{np.min(d_vec):.3f}, {np.max(d_vec):.3f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Coordinate tuning test failed: {e}")
        return False

def test_characteristic_functions():
    """Test characteristic function computations."""
    print("\n" + "=" * 60)
    print("TESTING CHARACTERISTIC FUNCTIONS")
    print("=" * 60)
    
    try:
        from cts_char import cts_char_exponent, cts_cf
        from wCF import weighted_cf_from_omegas
        
        # Test CTS parameters
        params = (1.4, 1e-3, 1e-3, 9.5, 9.5, 0.0)
        
        # Test characteristic exponent
        u_vals = np.linspace(-5, 5, 100)
        psi_vals = cts_char_exponent(u_vals, params)
        print(f"✓ CTS characteristic exponent computed successfully")
        print(f"  Input range: [{np.min(u_vals):.1f}, {np.max(u_vals):.1f}]")
        print(f"  Output range: [{np.min(np.real(psi_vals)):.3f}, {np.max(np.real(psi_vals)):.3f}]")
        
        # Test characteristic function
        cf_vals = cts_cf(u_vals, params)
        print(f"✓ CTS characteristic function computed successfully")
        print(f"  |CF| range: [{np.min(np.abs(cf_vals)):.3f}, {np.max(np.abs(cf_vals)):.3f}]")
        
        # Test weighted CF
        omegas = np.array([0.1, 0.2, 0.3, 0.4])
        wcf_vals = weighted_cf_from_omegas(u_vals, omegas, params)
        print(f"✓ Weighted characteristic function computed successfully")
        print(f"  |WCF| range: [{np.min(np.abs(wcf_vals)):.3f}, {np.max(np.abs(wcf_vals)):.3f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Characteristic function test failed: {e}")
        return False

def test_cos_inversion():
    """Test COS inversion functionality."""
    print("\n" + "=" * 60)
    print("TESTING COS INVERSION")
    print("=" * 60)
    
    try:
        from COS_inversion_helper import cos_invert_cf_from_cffunc
        from cts_char import cts_cf
        
        # Test with simple CTS characteristic function
        params = (1.4, 1e-3, 1e-3, 9.5, 9.5, 0.0)
        cf_func = lambda u: cts_cf(u, params)
        
        print("Running COS inversion...")
        x_grid, pdf_vals, cdf_vals, inv_cdf = cos_invert_cf_from_cffunc(
            cf_func, Ncos=512, L=6
        )
        
        print(f"✓ COS inversion successful:")
        print(f"  Grid size: {len(x_grid)}")
        print(f"  PDF range: [{np.min(pdf_vals):.6f}, {np.max(pdf_vals):.6f}]")
        print(f"  CDF range: [{np.min(cdf_vals):.6f}, {np.max(cdf_vals):.6f}]")
        
        # Test inverse CDF
        test_probs = [0.025, 0.5, 0.975]
        quantiles = [inv_cdf(p) for p in test_probs]
        print(f"  Quantiles at {test_probs}: {[f'{q:.3f}' for q in quantiles]}")
        
        return True
        
    except Exception as e:
        print(f"✗ COS inversion test failed: {e}")
        return False

def test_mini_propagation():
    """Test a mini version of the main propagation."""
    print("\n" + "=" * 60)
    print("TESTING MINI PROPAGATION")
    print("=" * 60)
    
    try:
        from utils import fetch_weekly_nifty, extract_extrema
        from RecFIF import build_recurrent_fif_with_dvec
        from coordinate_tuner import simple_tuner
        from cts_char import cts_char_exponent
        from wCF import weighted_cf_from_omegas
        from COS_inversion_helper import cos_invert_cf_from_cffunc
        
        # Use a smaller dataset for testing
        print("Fetching test data...")
        df = fetch_weekly_nifty(start="2020-01-01", end="2021-01-01")
        open_prices = df['Open'].values
        idx = extract_extrema(open_prices, order=1)
        knots_t = df['t'].values[idx]
        knots_y = open_prices[idx]
        
        print(f"Using {len(knots_t)} knots for mini test")
        
        # Quick tuning
        print("Running quick coordinate tuning...")
        d_vec = simple_tuner(
            knots_t, knots_y, df, m=2, passes=1,
            candidate_factors=[0.9, 1.0, 1.1],
            fast_iters=50, fast_grid=200
        )
        
        # Build basis functions (just a few for testing)
        print("Building basis functions...")
        K = min(5, len(knots_t))  # Use only first 5 knots for speed
        basis_fns = []
        for i in range(K):
            y_basis = np.zeros(len(knots_t))
            y_basis[i] = 1.0
            _, _, f_basis_fn, _ = build_recurrent_fif_with_dvec(
                knots_t, y_basis, d_vec, m=2, iters=100, grid_len=300, verbose=False
            )
            basis_fns.append(f_basis_fn)
        
        # Test propagation for a few points
        print("Testing propagation...")
        params_noise = (1.4, 1e-3, 1e-3, 9.5, 9.5, 0.0)
        test_indices = [0, len(df)//2, len(df)-1]
        
        for i in test_indices:
            tval = df['t'].values[i]
            omegas = np.array([float(fn(tval)) for fn in basis_fns])
            cf_func = lambda u: weighted_cf_from_omegas(u, omegas, params_noise)
            _, pdf_w, cdf_w, inv_cdf_w = cos_invert_cf_from_cffunc(cf_func, Ncos=256, L=6)
            q_low = float(inv_cdf_w(0.025))
            q_high = float(inv_cdf_w(0.975))
            print(f"  Week {i}: quantiles = [{q_low:.4f}, {q_high:.4f}]")
        
        print("✓ Mini propagation test successful!")
        return True
        
    except Exception as e:
        print(f"✗ Mini propagation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("RFIF FLOW TEST SUITE")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    
    start_time = time.time()
    
    # Run all tests
    tests = [
        ("Imports", test_imports),
        ("Data Fetching", test_data_fetching),
        ("Recurrent FIF", test_recurrent_fif),
        ("Coordinate Tuning", test_coordinate_tuning),
        ("Characteristic Functions", test_characteristic_functions),
        ("COS Inversion", test_cos_inversion),
        ("Mini Propagation", test_mini_propagation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:25} : {status}")
        if result:
            passed += 1
    
    total_time = time.time() - start_time
    print(f"\nPassed: {passed}/{len(tests)} tests")
    print(f"Total time: {total_time:.2f} seconds")
    
    if passed == len(tests):
        print("\n🎉 ALL TESTS PASSED! The RFIF flow is working correctly.")
        return True
    else:
        print(f"\n⚠️  {len(tests) - passed} tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

