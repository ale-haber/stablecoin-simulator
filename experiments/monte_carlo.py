import numpy as np
from typing import List
from dataclasses import dataclass
from model.parameters import SystemParameters
from .collateral_shock import run_collateral_shock_experiment
import itertools

@dataclass
class MonteCarloResults:
    """Results from Monte Carlo simulation."""
    collapse_probabilities: np.ndarray
    parameter_grid: dict
    mean_time_to_collapse: np.ndarray
    mean_max_drawdown: np.ndarray

def run_monte_carlo_stress_test(
    param_ranges: dict,
    n_trials: int,
    base_params: SystemParameters,
    n_steps: int = 1000
) -> MonteCarloResults:
    """
    Run Monte Carlo stress testing over parameter space.
    
    Args:
        param_ranges: Dict of parameter names to (min, max) ranges
        n_trials: Number of random trials per parameter combo
        base_params: Base system parameters
        n_steps: Simulation steps per trial
        
    Returns:
        MonteCarloResults with collapse statistics
    """
    # Create parameter grid
    grid_size = 10
    param_values = {
        name: np.linspace(min_val, max_val, grid_size)
        for name, (min_val, max_val) in param_ranges.items()
    }
    
    # Initialize result arrays
    n_params = len(param_ranges)
    collapse_prob = np.zeros([grid_size] * n_params)
    mean_ttc = np.zeros([grid_size] * n_params)
    mean_drawdown = np.zeros([grid_size] * n_params)
    
    # Grid search
    param_names = list(param_ranges.keys())
    
    for indices in itertools.product(*[range(grid_size)] * n_params):
        # Set parameters
        test_params = SystemParameters()
        # Copy base params first might be safer if we added a copy method, 
        # but here we just instantiate new and overwrite
        
        # Actually better to copy base_params or set all fields. 
        # For now assuming defaults in SystemParameters are 'base' or close enough, 
        # but let's copy fields from base_params if we can.
        # Since SystemParameters is a dataclass...
        from dataclasses import asdict
        test_params = SystemParameters(**asdict(base_params))
        
        for i, param_name in enumerate(param_names):
            setattr(test_params, param_name, param_values[param_name][indices[i]])
        
        # Run trials
        collapses = 0
        ttc_sum = 0
        drawdown_sum = 0
        
        for trial in range(n_trials):
            # Randomize shock
            shock_mag = np.random.uniform(-0.5, -0.2)
            shock_time = np.random.randint(50, 150)
            
            test_params.random_seed = base_params.random_seed + trial
            
            result = run_collateral_shock_experiment(
                shock_magnitude=shock_mag,
                shock_time=shock_time,
                params=test_params,
                n_steps=n_steps
            )
            
            if result.time_to_collapse < np.inf:
                collapses += 1
                ttc_sum += result.time_to_collapse
            
            drawdown_sum += result.max_drawdown
        
        # Store results
        collapse_prob[indices] = collapses / n_trials
        mean_ttc[indices] = ttc_sum / max(collapses, 1)
        mean_drawdown[indices] = drawdown_sum / n_trials
    
    return MonteCarloResults(
        collapse_probabilities=collapse_prob,
        parameter_grid=param_values,
        mean_time_to_collapse=mean_ttc,
        mean_max_drawdown=mean_drawdown
    )


def plot_monte_carlo_heatmap(results: MonteCarloResults, save_path: str = None):
    """
    Plot 2D heatmap of collapse probability.
    """
    import matplotlib.pyplot as plt
    
    # Extract parameter names and values
    param_names = list(results.parameter_grid.keys())
    if len(param_names) != 2:
        print("Heatmap requires exactly 2 parameters.")
        return
        
    p1_name, p2_name = param_names
    p1_vals = results.parameter_grid[p1_name]
    p2_vals = results.parameter_grid[p2_name]
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(
        results.collapse_probabilities.T, 
        extent=[p1_vals.min(), p1_vals.max(), p2_vals.min(), p2_vals.max()],
        origin='lower',
        aspect='auto',
        cmap='RdYlGn_r', # Red = High Collapse Prob (Bad), Green = Low (Good)
        alpha=0.8
    )
    plt.colorbar(label='Collapse Probability')
    plt.xlabel(p1_name)
    plt.ylabel(p2_name)
    plt.title(f'Stability Heatmap: {p1_name} vs {p2_name}')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")

if __name__ == "__main__":
    base_params = SystemParameters()
    
    # 2D Sweep: Demand Elasticity vs Mint Coefficient
    # "Which combination is safe?"
    results = run_monte_carlo_stress_test(
        param_ranges={
            'demand_elasticity': (0.1, 5.0),
            'mint_coefficient': (0.01, 1.0)
        },
        n_trials=10, 
        base_params=base_params,
        n_steps=500
    )
    print("Monte Carlo test completed.")
    
    import os
    os.makedirs("results/plots", exist_ok=True)
    plot_monte_carlo_heatmap(results, save_path="results/plots/monte_carlo_heatmap.png")
