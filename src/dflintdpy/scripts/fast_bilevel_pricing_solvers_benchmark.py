
from dflintdpy.scripts.read_synthetic_data import read_synthetic_data
from dflintdpy.solvers.fast_solvers.fast_pricing_solver import FastBilevelPricingSolver
import numpy as np

def benchmark_all_methods(c, Sigma, gamma, budget, p_max=None):
    """
    Run all methods and compare performance.
    
    Returns:
        DataFrame with comparison
    """
    import time
    
    solver = FastBilevelPricingSolver(c, Sigma, gamma, budget, p_max)
    
    methods = [
        ('SQP', lambda: solver.solve_sequential_quadratic()),
        ('Projected Gradient', lambda: solver.solve_projected_gradient()),
        ('Alternating', lambda: solver.solve_alternating_optimization()),
        ('Multi-Start (3)', lambda: solver.solve_multistart(n_starts=3)),
        ('Trust Region', lambda: solver.solve_trust_region()),
    ]
    
    results = []
    
    print("=" * 80)
    print("BENCHMARKING ALL METHODS")
    print("=" * 80)
    
    for name, method_func in methods:
        print(f"\n{name}:")
        print("-" * 40)
        
        start_time = time.time()
        try:
            result = method_func()
            elapsed = time.time() - start_time
            
            if result['success']:
                results.append({
                    'Method': name,
                    'Revenue': result['revenue'],
                    'Time (s)': elapsed,
                    'Success': '✓'
                })
                print(f"✓ Revenue: {result['revenue']:.4f}, Time: {elapsed:.2f}s")
            else:
                results.append({
                    'Method': name,
                    'Revenue': 0,
                    'Time (s)': elapsed,
                    'Success': '✗'
                })
                print(f"✗ Failed")
        except Exception as e:
            elapsed = time.time() - start_time
            results.append({
                'Method': name,
                'Revenue': 0,
                'Time (s)': elapsed,
                'Success': '✗'
            })
            print(f"✗ Error: {e}")
    
    return results


def main():
    """Example usage"""
    
    print("=" * 80)
    print("FAST BILEVEL PRICING SOLVER - WITH BUDGET CONSTRAINT")
    print("=" * 80)
    
    # Problem parameters
    N = 1000
    noise = 1
    deg = 16
    n_test_samples = 200

    # Read synthetic data
    x_train, y_train, x_valid, y_valid, x_test, y_test, cov, gamma = \
        read_synthetic_data(N, noise, deg)
    
    # Extract problem parameters
    n = y_train.shape[1]
    c = y_test[0, :]
    Sigma = cov
    budget = c.sum() * 0.3
    
    print("\nProblem:")
    print(f"  n = {n} items")
    print(f"  Price budget: {budget}")
    print(f"  Risk tolerance γ: {gamma}")
    
    # Benchmark all methods on mutiple runs
    results = []
    for c in y_test[:n_test_samples, :]:
        result = benchmark_all_methods(c, Sigma, gamma, budget)
        results.append(result) # results is a list of lists
    
    # Process results
    aggregated_results = {}
    for run_results in results:
        for r in run_results:
            method = r['Method']
            if method not in aggregated_results:
                aggregated_results[method] = {
                    'Revenue': [],
                    'Time (s)': [],
                    'Success': []
                }
            aggregated_results[method]['Revenue'].append(r['Revenue'])
            aggregated_results[method]['Time (s)'].append(r['Time (s)'])
            aggregated_results[method]['Success'].append(r['Success'])

    # Print comparison
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"\n{'Method':<25} {'Revenue':>12} {'Time (s)':>12} {'Success rate':>8}")
    print("-" * 65)
    for method, r in aggregated_results.items():
        print(f"{method:<25} {np.mean(r['Revenue']):>12.4f} {np.mean(r['Time (s)']):>12.2f} {r['Success'].count('✓') / len(r['Success']):>8}")

    # Find best
    successful = [(m, r) for m, r in aggregated_results.items() if r['Success'].count('✓') > 0]
    if successful:
        best = max(successful, key=lambda x: np.mean(x[1]['Revenue']))
        fastest = min(successful, key=lambda x: np.mean(x[1]['Time (s)']))

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"🏆 {'Best Revenue:':<15} {best[0]:<15} ({np.mean(best[1]['Time (s)']):.2f}s) ({np.mean(best[1]['Revenue']):.4f})")
        print(f"⚡ {'Fastest:':<15} {fastest[0]:<15} ({np.mean(fastest[1]['Time (s)']):.2f}s) ({np.mean(fastest[1]['Revenue']):.4f})")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()