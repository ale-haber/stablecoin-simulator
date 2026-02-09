import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
from model.dynamics import simulate_step
from model.parameters import SystemParameters
import os

@dataclass
class ExperimentResults:
    """Results from a collateral shock experiment."""
    time: np.ndarray
    supply: np.ndarray
    price: np.ndarray
    collateral: np.ndarray
    liquidity: np.ndarray
    demand: np.ndarray
    
    # Metrics
    peg_deviation_integral: float
    time_to_collapse: float
    max_drawdown: float
    recovered: bool
    
def run_collateral_shock_experiment(
    shock_magnitude: float,
    shock_time: int,
    params: SystemParameters,
    n_steps: int = 1000
) -> ExperimentResults:
    """
    Simulate sudden collateral reduction.
    
    Args:
        shock_magnitude: Percentage shock (e.g., -0.3 for 30% drop)
        shock_time: Time step when shock occurs
        params: System parameters
        n_steps: Total simulation steps
        
    Returns:
        ExperimentResults with time series and metrics
    """
    # Initialize state
    S = params.initial_supply
    P = params.initial_price
    C = params.initial_collateral
    L = params.initial_liquidity
    D = params.initial_demand
    
    # Storage
    time = np.arange(n_steps) * params.dt
    supply = np.zeros(n_steps)
    price = np.zeros(n_steps)
    collateral = np.zeros(n_steps)
    liquidity = np.zeros(n_steps)
    demand = np.zeros(n_steps)
    
    # Set random seed
    np.random.seed(params.random_seed)
    
    # Simulate
    for t in range(n_steps):
        # Apply shock at specified time
        c_shock = shock_magnitude if t == shock_time else 0.0
        
        # Store state
        supply[t] = S
        price[t] = P
        collateral[t] = C
        liquidity[t] = L
        demand[t] = D
        
        # Step forward
        S, P, C, L, D = simulate_step(
            (S, P, C, L, D),
            params,
            collateral_shock=c_shock
        )
    
    # Compute metrics
    peg_deviation = np.trapezoid(np.abs(price - 1.0), time)
    
    collapse_mask = price < params.collapse_price_threshold
    if np.any(collapse_mask):
        time_to_collapse = time[np.argmax(collapse_mask)]
    else:
        time_to_collapse = np.inf
    
    max_drawdown = np.max(1.0 - price)
    
    # Check recovery
    if time_to_collapse < np.inf:
        post_collapse = price[collapse_mask]
        recovered = np.any(post_collapse > params.recovery_price_threshold)
    else:
        recovered = True
    
    return ExperimentResults(
        time=time,
        supply=supply,
        price=price,
        collateral=collateral,
        liquidity=liquidity,
        demand=demand,
        peg_deviation_integral=peg_deviation,
        time_to_collapse=time_to_collapse,
        max_drawdown=max_drawdown,
        recovered=recovered
    )

def plot_results(results: ExperimentResults, save_path: str = None):
    """
    Plot experiment results.
    
    Args:
        results: ExperimentResults object
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Price
    axes[0, 0].plot(results.time, results.price, 'b-', linewidth=2)
    axes[0, 0].axhline(y=1.0, color='k', linestyle='--', label='Peg')
    axes[0, 0].axhline(y=0.5, color='r', linestyle='--', label='Collapse')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Price')
    axes[0, 0].set_title('Stablecoin Price')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Supply
    axes[0, 1].plot(results.time, results.supply, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Supply')
    axes[0, 1].set_title('Stablecoin Supply')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Collateral & Liquidity
    ax1 = axes[1, 0]
    ax2 = ax1.twinx()
    ax1.plot(results.time, results.collateral, 'r-', linewidth=2, label='Collateral')
    ax2.plot(results.time, results.liquidity, 'b-', linewidth=2, label='Liquidity')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Collateral', color='r')
    ax2.set_ylabel('Liquidity', color='b')
    ax1.tick_params(axis='y', labelcolor='r')
    ax2.tick_params(axis='y', labelcolor='b')
    axes[1, 0].set_title('Collateral & Liquidity')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Demand
    axes[1, 1].plot(results.time, results.demand, 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Demand')
    axes[1, 1].set_title('Aggregate Demand')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    # plt.show()  # specific for non-interactive environments

if __name__ == "__main__":
    # Run example experiment
    params = SystemParameters()
    
    results = run_collateral_shock_experiment(
        shock_magnitude=-0.4,  # 40% collateral drop (Testing Breakpoint)
        shock_time=100,
        params=params,
        n_steps=1000
    )
    
    print(f"Peg Deviation Integral: {results.peg_deviation_integral:.4f}")
    print(f"Time to Collapse: {results.time_to_collapse:.2f}")
    print(f"Max Drawdown: {results.max_drawdown:.4f}")
    print(f"Recovered: {results.recovered}")
    
    # Ensure results directory exists
    os.makedirs("results/plots", exist_ok=True)
    plot_results(results, save_path="results/plots/collateral_shock.png")
