#!/usr/bin/env python3
"""
Example usage of the RFIF (Recurrent Fractal Interpolation Function) system.
This script demonstrates how to use the main components of the system.
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def example_basic_usage():
    """Demonstrate basic usage of the RFIF system."""
    print("=" * 60)
    print("RFIF SYSTEM - BASIC USAGE EXAMPLE")
    print("=" * 60)
    
    # Import required modules
    from utils import fetch_weekly_nifty, extract_extrema
    from RecFIF import build_recurrent_fif_with_dvec
    from coordinate_tuner import simple_tuner
    from cts_char import cts_char_exponent
    from wCF import weighted_cf_from_omegas
    from COS_inversion_helper import cos_invert_cf_from_cffunc
    
    # 1. Load and preprocess data
    print("1. Loading NIFTY data...")
    df = fetch_weekly_nifty(start="2020-01-01", end="2021-01-01")  # Smaller dataset for demo
    open_prices = df['Open'].values
    idx = extract_extrema(open_prices, order=1)
    knots_t = df['t'].values[idx]
    knots_y = open_prices[idx]
    
    print(f"   Data loaded: {len(df)} weeks, {len(idx)} knots")
    
    # 2. Quick parameter tuning
    print("2. Tuning fractal parameters...")
    d_vec = simple_tuner(
        knots_t, knots_y, df, m=2, passes=1,
        candidate_factors=[0.9, 1.0, 1.1],
        fast_iters=50, fast_grid=200
    )
    print(f"   Tuned {len(d_vec)} parameters")
    
    # 3. Build fractal interpolant
    print("3. Building recurrent FIF...")
    t_grid, f, f_fn, maps = build_recurrent_fif_with_dvec(
        knots_t, knots_y, d_vec, m=2, iters=100, grid_len=500, verbose=True
    )
    
    # 4. Test characteristic functions
    print("4. Testing characteristic functions...")
    params = (1.4, 1e-3, 1e-3, 9.5, 9.5, 0.0)  # CTS parameters
    u_vals = np.linspace(-2, 2, 50)
    psi_vals = cts_char_exponent(u_vals, params)
    print(f"   Characteristic exponent computed for {len(u_vals)} points")
    
    # 5. Test COS inversion
    print("5. Testing COS inversion...")
    cf_func = lambda u: np.exp(cts_char_exponent(u, params))
    x_grid, pdf_vals, cdf_vals, inv_cdf = cos_invert_cf_from_cffunc(cf_func, Ncos=256, L=6)
    print(f"   PDF reconstructed with {len(x_grid)} grid points")
    
    # 6. Simple visualization
    print("6. Creating visualization...")
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Data and interpolant
    plt.subplot(2, 2, 1)
    plt.plot(df['Date'], df['Open'].values, 'b-', alpha=0.7, label='Actual Open')
    plt.plot(df['Date'], f_fn(df['t'].values), 'r-', linewidth=2, label='RFIF Interpolant')
    plt.scatter(df['Date'].iloc[idx], knots_y, color='red', s=50, zorder=5, label='Knots')
    plt.title('NIFTY Data and RFIF Interpolant')
    plt.legend()
    plt.xticks(rotation=45)
    
    # Plot 2: Fractal parameters
    plt.subplot(2, 2, 2)
    plt.plot(d_vec, 'o-', markersize=4)
    plt.title('Tuned Fractal Parameters (d_vec)')
    plt.xlabel('Parameter Index')
    plt.ylabel('Parameter Value')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Characteristic function
    plt.subplot(2, 2, 3)
    plt.plot(u_vals, np.real(psi_vals), 'b-', label='Real part')
    plt.plot(u_vals, np.imag(psi_vals), 'r-', label='Imaginary part')
    plt.title('CTS Characteristic Exponent')
    plt.xlabel('u')
    plt.ylabel('ψ(u)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Probability density
    plt.subplot(2, 2, 4)
    plt.plot(x_grid, pdf_vals, 'g-', linewidth=2)
    plt.title('Reconstructed PDF (COS Method)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rfif_example_output.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("   Visualization saved as 'rfif_example_output.png'")
    print("\n✓ Basic usage example completed successfully!")

def example_advanced_usage():
    """Demonstrate advanced usage with confidence bands."""
    print("\n" + "=" * 60)
    print("RFIF SYSTEM - ADVANCED USAGE EXAMPLE")
    print("=" * 60)
    
    from main import main_propagation
    
    print("Running advanced propagation with confidence bands...")
    
    # Run with reduced parameters for demonstration
    result = main_propagation(
        full_run=False,
        sample_every=5,      # Every 5th week
        Ncos_per_week=512,   # Reduced COS terms
        basis_iters=100,     # Reduced iterations
        basis_grid=500,      # Reduced grid
        m=2
    )
    
    print(f"✓ Advanced example completed!")
    print(f"  Coverage: {result['coverage']:.2f}%")
    print(f"  Knots used: {len(result['knots_idx'])}")
    print(f"  Data points: {len(result['df'])}")

def example_custom_parameters():
    """Demonstrate custom parameter configuration."""
    print("\n" + "=" * 60)
    print("RFIF SYSTEM - CUSTOM PARAMETERS EXAMPLE")
    print("=" * 60)
    
    # Custom CTS parameters for different market conditions
    conservative_params = (1.2, 5e-4, 5e-4, 12.0, 12.0, 0.0)  # Lower volatility
    aggressive_params = (1.6, 2e-3, 2e-3, 7.0, 7.0, 0.0)      # Higher volatility
    
    from cts_char import cts_char_exponent
    from COS_inversion_helper import cos_invert_cf_from_cffunc
    
    u_vals = np.linspace(-3, 3, 100)
    
    # Compare different parameter sets
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for params, label in [(conservative_params, 'Conservative'), (aggressive_params, 'Aggressive')]:
        psi_vals = cts_char_exponent(u_vals, params)
        plt.plot(u_vals, np.real(psi_vals), label=f'{label} (Real)')
    plt.title('Characteristic Exponents Comparison')
    plt.xlabel('u')
    plt.ylabel('Re(ψ(u))')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for params, label in [(conservative_params, 'Conservative'), (aggressive_params, 'Aggressive')]:
        cf_func = lambda u: np.exp(cts_char_exponent(u, params))
        x_grid, pdf_vals, _, _ = cos_invert_cf_from_cffunc(cf_func, Ncos=256, L=6)
        plt.plot(x_grid, pdf_vals, label=label)
    plt.title('PDF Comparison')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rfif_parameter_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Custom parameters example completed!")
    print("  Comparison plot saved as 'rfif_parameter_comparison.png'")

if __name__ == "__main__":
    print("RFIF SYSTEM - USAGE EXAMPLES")
    print("This script demonstrates various ways to use the RFIF system.")
    print("Make sure you have activated the virtual environment: source myenv/bin/activate")
    print()
    
    try:
        # Run examples
        example_basic_usage()
        example_advanced_usage()
        example_custom_parameters()
        
        print("\n" + "=" * 60)
        print("🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Generated files:")
        print("  - rfif_example_output.png")
        print("  - rfif_parameter_comparison.png")
        print("\nFor more information, see README.md")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
