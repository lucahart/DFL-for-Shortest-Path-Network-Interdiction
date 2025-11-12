import numpy as np
import torch
from typing import Dict, List, Tuple
from dflintdpy.solvers.bilevel_pricing import BilevelPricingProblem
from dflintdpy.solvers.fast_solvers.fast_pricing_solver import FastBilevelPricingSolver
import time

def get_predictions(x: np.ndarray, 
                   pred_model_dfl, 
                   pred_model_pfl, 
                   pred_model_adfl) -> Dict[str, np.ndarray]:
    """
    Get predictions from all three models for a single sample.
    
    Args:
        x: Feature vector for a single sample
        pred_model_dfl: DFL prediction model
        pred_model_pfl: PFL prediction model
        pred_model_adfl: A-DFL prediction model
    
    Returns:
        Dictionary with predictions from each model
    """
    x_tensor = torch.tensor(x)
    
    predictions = {
        'dfl': pred_model_dfl(x_tensor).detach().numpy(),
        'pfl': pred_model_pfl(x_tensor).detach().numpy(),
        'adfl': pred_model_adfl(x_tensor).detach().numpy()
    }
    
    return predictions

def solve_buyer_problem(opt_model, 
                       predictions: Dict[str, np.ndarray], 
                       c: np.ndarray) -> Dict[str, Dict]:
    """
    Solve the buyer's optimization problem for each predictor.
    
    Args:
        opt_model: Optimization model object
        predictions: Dictionary of predictions from each model
        c: True cost vector
    
    Returns:
        Dictionary containing solutions and objective values for each predictor
    """
    results = {}
    
    for model_name, pred in predictions.items():
        opt_model.setObj(pred)
        sol, _ = opt_model.solve()
        obj_value = sol @ c
        
        results[model_name] = {
            'solution': sol,
            'objective': obj_value
        }
    
    return results

def solve_bilevel_pricing(c: np.ndarray, 
                         cov: np.ndarray, 
                         gamma: float, 
                         budget: float,
                         use_fast_solver: bool = True,
                         gurobi_time_limit: int = 60,
                         n_starts: int = 5,
                         verbose: bool = False,
                         fast_only: bool = False) -> Dict:
    """
    Solve the bilevel pricing problem using both Gurobi and fast solver.
    
    Args:
        c: Cost vector
        cov: Covariance matrix
        gamma: Risk tolerance parameter
        budget: Budget constraint
        use_fast_solver: Whether to use the fast solver
        gurobi_time_limit: Time limit for Gurobi solver
        n_starts: Number of random starts for fast solver
    
    Returns:
        Dictionary with pricing results from both solvers
    """
    if not fast_only:
        problem = BilevelPricingProblem(c, cov, gamma, budget=budget)
        result_gurobi = problem.solve_with_gurobi_miqcp(M=100, time_limit=gurobi_time_limit)
        
        results = {
            'gurobi': result_gurobi
        }
    else:
        results = {}

    if use_fast_solver:
        fast_solver = FastBilevelPricingSolver(c, cov, gamma, budget=budget)
        result_fast = fast_solver.solve(n_starts=n_starts, verbose=verbose)
        results['fast'] = result_fast
    
    return results

def solve_buyer_with_pricing(opt_model,
                             predictions: Dict[str, np.ndarray],
                             c: np.ndarray,
                             p: np.ndarray) -> Dict[str, Dict]:
    """
    Solve buyer's problem after seller sets prices.
    
    Args:
        opt_model: Optimization model object
        predictions: Dictionary of predictions from each model
        c: True cost vector
        p: Price vector set by seller
    
    Returns:
        Dictionary with solutions and objectives for each predictor
    """
    results = {}
    
    # Compute buyer's response for each predictor
    for model_name, pred in predictions.items():
        opt_model.setObj(pred - p)
        sol, _ = opt_model.solve()
        obj_value = sol @ (c - p)
        
        results[model_name] = {
            'solution': sol,
            'objective': obj_value
        }
    
    # Append ground truth solution
    opt_model.setObj(c - p)
    sol_gt, obj_value_gt = opt_model.solve()
    results['ground_truth'] = {
        'solution': sol_gt,
        'objective': obj_value_gt
    }
    
    return results

def evaluate_single_sample(idx: int,
                          x_test: np.ndarray,
                          y_test: np.ndarray,
                          pred_model_dfl,
                          pred_model_pfl,
                          pred_model_adfl,
                          opt_model,
                          cov: np.ndarray,
                          gamma: float,
                          budget_factor: float = 0.3,
                          verbose: bool = True,
                          gurobi_time_limit: int = 60,
                          n_starts: int = 5,
                          fast_only: bool = False) -> Dict:
    """
    Evaluate all predictors on a single test sample.
    
    Args:
        idx: Index of test sample
        x_test: Test features
        y_test: Test targets (cost vectors)
        pred_model_dfl: DFL prediction model
        pred_model_pfl: PFL prediction model
        pred_model_adfl: A-DFL prediction model
        opt_model: Optimization model
        cov: Covariance matrix
        gamma: Risk tolerance
        budget_factor: Budget as fraction of total cost
        verbose: Whether to print detailed results
        gurobi_time_limit: Time limit for Gurobi solver
        n_starts: Number of starts for fast solver
    
    Returns:
        Dictionary with all results for this sample
    """
    # Extract sample data
    c = y_test[idx, :]
    x = x_test[idx, :]
    budget = c.sum() * budget_factor
    
    # Get predictions
    predictions = get_predictions(x, pred_model_dfl, pred_model_pfl, pred_model_adfl)
    
    # Solve buyer's problem without pricing
    buyer_results_no_pricing = solve_buyer_problem(opt_model, predictions, c)
    
    # Solve bilevel pricing problem
    pricing_results = solve_bilevel_pricing(c, cov, gamma, budget, 
                                           gurobi_time_limit=gurobi_time_limit,
                                           n_starts=n_starts,
                                           verbose=verbose,
                                           fast_only=fast_only)
    
    # Solve buyer's problem with pricing (both solvers)
    buyer_results_gurobi = None
    buyer_results_fast = None
    
    if 'gurobi' in pricing_results and pricing_results['gurobi']['success']:
        p_gurobi = pricing_results['gurobi']['p_opt']
        buyer_results_gurobi = solve_buyer_with_pricing(opt_model, predictions, c, p_gurobi)
    
    if 'fast' in pricing_results:
        p_fast = pricing_results['fast']['p_opt']
        buyer_results_fast = solve_buyer_with_pricing(opt_model, predictions, c, p_fast)
    
    # Compile results
    results = {
        'idx': idx,
        'c': c,
        'x': x,
        'budget': budget,
        'predictions': predictions,
        'buyer_no_pricing': buyer_results_no_pricing,
        'pricing': pricing_results,
        'buyer_gurobi_pricing': buyer_results_gurobi,
        'buyer_fast_pricing': buyer_results_fast,
        'prediction_stats': {
            'true': {'mean': c.mean(), 'std': c.std()},
            'dfl': {'mean': predictions['dfl'].mean(), 'std': predictions['dfl'].std()},
            'pfl': {'mean': predictions['pfl'].mean(), 'std': predictions['pfl'].std()},
            'adfl': {'mean': predictions['adfl'].mean(), 'std': predictions['adfl'].std()}
        }
    }
    
    if verbose:
        print_sample_results(results)
    
    return results

def print_sample_results(results: Dict):
    """Print formatted results for a single sample."""
    print("\n" + "=" * 80)
    print(f"BILEVEL PRICING OPTIMIZATION ON TEST SAMPLE {results['idx']}")
    print("=" * 80)
    
    print("\nProblem Parameters:")
    print(f"Cost vector c: {results['c']}")
    print(f"Problem size: n = {results['c'].shape[0]}")
    print(f"Budget: {results['budget']:.4f}")
    print(f"Risk tolerance γ: (stored in results)")
    
    print("\n" + "-" * 80)
    print("BUYER OBJECTIVES WITHOUT PRICING")
    print("-" * 80)
    for model_name, model_results in results['buyer_no_pricing'].items():
        print(f"{model_name.upper():6s}: Buyer's objective value: {model_results['objective']:.4f}")
    
    if results['pricing']['gurobi'] is not None and results['pricing']['gurobi']['success']:
        gurobi_result = results['pricing']['gurobi']
        print("\n" + "=" * 80)
        print("PLAYER PAYOFFS AFTER PRICING (GUROBI)")
        print("=" * 80)
        print(f"\nOptimal prices p: {gurobi_result['p_opt']}")
        print(f"Buyer response y: {gurobi_result['y_opt']}")
        print(f"Seller's revenue: {gurobi_result['revenue']:.4f}")
        
        if results['buyer_gurobi_pricing'] is not None:
            for model_name, model_results in results['buyer_gurobi_pricing'].items():
                print(f"{model_name.upper():6s}: Buyer's objective value: {model_results['objective']:.4f}")
        
        print(f"\nVerification gap: {gurobi_result['verification_gap']:.2e}")
        print(f"MIP gap: {gurobi_result['mip_gap']:.2e}")
        print(f"Solve time: {gurobi_result['solve_time']:.2f}s")
    
    if 'fast' in results['pricing']:
        fast_result = results['pricing']['fast']
        print("\n" + "-" * 80)
        print("PLAYER PAYOFFS AFTER PRICING (FAST SOLVER)")
        print("-" * 80)
        print(f"Seller's revenue: {fast_result['revenue']:.4f}")
        
        if results['buyer_fast_pricing'] is not None:
            for model_name, model_results in results['buyer_fast_pricing'].items():
                print(f"{model_name.upper():6s}: Buyer's objective value: {model_results['objective']:.4f}")
    
    print("\n" + "-" * 80)
    print("PREDICTION STATISTICS")
    print("-" * 80)
    stats = results['prediction_stats']
    print(f"True values:        {stats['true']['mean']:.4f}, {stats['true']['std']:.4f}")
    print(f"DFL: Predictions:   {stats['dfl']['mean']:.4f}, {stats['dfl']['std']:.4f}")
    print(f"PFL: Predictions:   {stats['pfl']['mean']:.4f}, {stats['pfl']['std']:.4f}")
    print(f"A-DFL: Predictions: {stats['adfl']['mean']:.4f}, {stats['adfl']['std']:.4f}")

def evaluate_multiple_samples(num_samples: int,
                              x_test: np.ndarray,
                              y_test: np.ndarray,
                              pred_model_dfl,
                              pred_model_pfl,
                              pred_model_adfl,
                              opt_model,
                              cov: np.ndarray,
                              gamma: float,
                              budget_factor: float = 0.3,
                              verbose: bool = False,
                              gurobi_time_limit: int = 60,
                              n_starts: int = 5,
                              fast_only: bool = False) -> List[Dict]:
    """
    Evaluate all predictors on multiple test samples.
    
    Args:
        num_samples: Number of test samples to evaluate
        x_test: Test features
        y_test: Test targets
        pred_model_dfl: DFL prediction model
        pred_model_pfl: PFL prediction model
        pred_model_adfl: A-DFL prediction model
        opt_model: Optimization model
        cov: Covariance matrix
        gamma: Risk tolerance
        budget_factor: Budget as fraction of total cost
        verbose: Whether to print detailed results for each sample
        gurobi_time_limit: Time limit for Gurobi solver
        n_starts: Number of starts for fast solver
    
    Returns:
        List of result dictionaries, one per sample
    """
    all_results = []
    
    print(f"\nEvaluating {num_samples} test samples...")
    start_time = time.time()
    
    for idx in range(num_samples):
        if verbose:
            print(f"Processing sample {idx+1}/{num_samples}...", end='\r')
        
        try:
            results = evaluate_single_sample(
                idx=idx,
                x_test=x_test,
                y_test=y_test,
                pred_model_dfl=pred_model_dfl,
                pred_model_pfl=pred_model_pfl,
                pred_model_adfl=pred_model_adfl,
                opt_model=opt_model,
                cov=cov,
                gamma=gamma,
                budget_factor=budget_factor,
                verbose=verbose,
                gurobi_time_limit=gurobi_time_limit,
                n_starts=n_starts,
                fast_only=fast_only,
            )
            all_results.append(results)
        except Exception as e:
            print(f"\nError processing sample {idx}: {str(e)}")
            continue
    
    elapsed_time = time.time() - start_time
    print(f"\nCompleted {len(all_results)}/{num_samples} samples in {elapsed_time:.2f}s")
    
    return all_results

def summarize_results(all_results: List[Dict]) -> Dict:
    """
    Compute summary statistics across all test samples.
    
    Args:
        all_results: List of result dictionaries from evaluate_multiple_samples
    
    Returns:
        Dictionary with summary statistics
    """
    num_samples = len(all_results)
    
    # Initialize aggregators
    objectives_no_pricing = {'dfl': [], 'pfl': [], 'adfl': []}
    objectives_gurobi = {'dfl': [], 'pfl': [], 'adfl': []}
    objectives_fast = {'dfl': [], 'pfl': [], 'adfl': []}
    revenues_gurobi = []
    revenues_fast = []
    solve_times = []
    
    # Aggregate results
    for result in all_results:
        # No pricing objectives
        for model in ['dfl', 'pfl', 'adfl']:
            objectives_no_pricing[model].append(
                result['buyer_no_pricing'][model]['objective']
            )
        
        # Gurobi pricing objectives
        if result['buyer_gurobi_pricing'] is not None:
            for model in ['dfl', 'pfl', 'adfl']:
                objectives_gurobi[model].append(
                    result['buyer_gurobi_pricing'][model]['objective']
                )
            revenues_gurobi.append(result['pricing']['gurobi']['revenue'])
            solve_times.append(result['pricing']['gurobi']['solve_time'])
        
        # Fast solver objectives
        if result['buyer_fast_pricing'] is not None:
            for model in ['dfl', 'pfl', 'adfl']:
                objectives_fast[model].append(
                    result['buyer_fast_pricing'][model]['objective']
                )
            revenues_fast.append(result['pricing']['fast']['revenue'])
    
    # Compute statistics
    summary = {
        'num_samples': num_samples,
        'no_pricing': {
            model: {
                'mean': np.mean(objs),
                'std': np.std(objs),
                'min': np.min(objs),
                'max': np.max(objs)
            } for model, objs in objectives_no_pricing.items()
        },
        'gurobi_pricing': {
            'buyer_objectives': {
                model: {
                    'mean': np.mean(objs),
                    'std': np.std(objs),
                    'min': np.min(objs),
                    'max': np.max(objs)
                } for model, objs in objectives_gurobi.items() if objs
            },
            'seller_revenue': {
                'mean': np.mean(revenues_gurobi) if revenues_gurobi else None,
                'std': np.std(revenues_gurobi) if revenues_gurobi else None,
                'min': np.min(revenues_gurobi) if revenues_gurobi else None,
                'max': np.max(revenues_gurobi) if revenues_gurobi else None
            },
            'solve_time': {
                'mean': np.mean(solve_times) if solve_times else None,
                'std': np.std(solve_times) if solve_times else None,
                'min': np.min(solve_times) if solve_times else None,
                'max': np.max(solve_times) if solve_times else None
            }
        },
        'fast_pricing': {
            'buyer_objectives': {
                model: {
                    'mean': np.mean(objs),
                    'std': np.std(objs),
                    'min': np.min(objs),
                    'max': np.max(objs)
                } for model, objs in objectives_fast.items() if objs
            },
            'seller_revenue': {
                'mean': np.mean(revenues_fast) if revenues_fast else None,
                'std': np.std(revenues_fast) if revenues_fast else None,
                'min': np.min(revenues_fast) if revenues_fast else None,
                'max': np.max(revenues_fast) if revenues_fast else None
            }
        }
    }
    
    return summary

def print_summary(summary: Dict):
    """Print formatted summary statistics."""
    print("\n" + "=" * 80)
    print(f"SUMMARY STATISTICS OVER {summary['num_samples']} SAMPLES")
    print("=" * 80)
    
    print("\n" + "-" * 80)
    print("BUYER OBJECTIVES WITHOUT PRICING")
    print("-" * 80)
    print(f"{'Model':<7s} {'Mean':<8s} {'Std':<6s} [{'Min,':<7s} {'Max':<6s}]")
    print(f"{'-'*7} {'-'*8} {'-'*6} {'-'*7} {'-'*6}")
    for model in ['dfl', 'pfl', 'adfl']:
        stats = summary['no_pricing'][model]
        print(f"{model.upper():6s}: {stats['mean']:.4f} ± {stats['std']:.4f} "
              f"[{stats['min']:.4f}, {stats['max']:.4f}]")
    
    if summary['gurobi_pricing']['buyer_objectives']:
        print("\n" + "-" * 80)
        print("BUYER OBJECTIVES WITH GUROBI PRICING")
        print("-" * 80)
        print(f"{'Model':<7s} {'Mean':<8s} {'Std':<6s} [{'Min,':<7s} {'Max':<6s}]")
        print(f"{'-'*7} {'-'*8} {'-'*6} {'-'*7} {'-'*6}")
        for model in ['dfl', 'pfl', 'adfl']:
            if model in summary['gurobi_pricing']['buyer_objectives']:
                stats = summary['gurobi_pricing']['buyer_objectives'][model]
                print(f"{model.upper():6s}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                      f"[{stats['min']:.4f}, {stats['max']:.4f}]")
        
        print("\nSeller Revenue (Gurobi):")
        rev_stats = summary['gurobi_pricing']['seller_revenue']
        if rev_stats['mean'] is not None:
            print(f"  {rev_stats['mean']:.4f} ± {rev_stats['std']:.4f} "
                  f"[{rev_stats['min']:.4f}, {rev_stats['max']:.4f}]")
        
        print("\nSolve Time (Gurobi):")
        time_stats = summary['gurobi_pricing']['solve_time']
        if time_stats['mean'] is not None:
            print(f"  {time_stats['mean']:.2f}s ± {time_stats['std']:.2f}s "
                  f"[{time_stats['min']:.2f}s, {time_stats['max']:.2f}s]")
    
    if summary['fast_pricing']['buyer_objectives']:
        print("\n" + "-" * 80)
        print("BUYER OBJECTIVES WITH FAST SOLVER PRICING")
        print("-" * 80)
        print(f"{'Model':<7s} {'Mean':<8s} {'Std':<6s} [{'Min,':<7s} {'Max':<6s}]")
        print(f"{'-'*7} {'-'*8} {'-'*6} {'-'*7} {'-'*6}")
        for model in ['dfl', 'pfl', 'adfl']:
            if model in summary['fast_pricing']['buyer_objectives']:
                stats = summary['fast_pricing']['buyer_objectives'][model]
                print(f"{model.upper():6s}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                      f"[{stats['min']:.4f}, {stats['max']:.4f}]")
        
        print("\nSeller Revenue (Fast Solver):")
        rev_stats = summary['fast_pricing']['seller_revenue']
        if rev_stats['mean'] is not None:
            print(f"  {rev_stats['mean']:.4f} ± {rev_stats['std']:.4f} "
                  f"[{rev_stats['min']:.4f}, {rev_stats['max']:.4f}]")


# Example usage:
if __name__ == "__main__":
    # For single sample (your original code):
    # results = evaluate_single_sample(
    #     idx=0,
    #     x_test=x_test,
    #     y_test=y_test,
    #     pred_model_dfl=pred_model_dfl,
    #     pred_model_pfl=pred_model_pfl,
    #     pred_model_adfl=pred_model_adfl,
    #     opt_model=opt_model,
    #     cov=cov,
    #     gamma=gamma,
    #     verbose=True
    # )
    
    # For multiple samples:
    # all_results = evaluate_multiple_samples(
    #     num_samples=10,
    #     x_test=x_test,
    #     y_test=y_test,
    #     pred_model_dfl=pred_model_dfl,
    #     pred_model_pfl=pred_model_pfl,
    #     pred_model_adfl=pred_model_adfl,
    #     opt_model=opt_model,
    #     cov=cov,
    #     gamma=gamma,
    #     verbose=False
    # )
    
    # summary = summarize_results(all_results)
    # print_summary(summary)
    pass