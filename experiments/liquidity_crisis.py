import numpy as np
import matplotlib.pyplot as plt
from .collateral_shock import ExperimentResults, plot_results
from model.dynamics import simulate_step
from model.parameters import SystemParameters
import os

def run_liquidity_crisis_experiment(
    liquidity_shock_magnitude: float,
    shock_time: int,
    params: SystemParameters,
    n_steps: int = 1000
) -> ExperimentResults:
    """
    Simulate sudden liquidity reduction.
    
    Args:
        liquidity_shock_magnitude: Percentage shock (e.g., -0.5 for 50% drop)
        shock_time: Time step when shock occurs
        params: System parameters
        n_steps: Total simulation steps
        
    Returns:
        ExperimentResults with time series and metrics
    """
    S = params.initial_supply
    P = params.initial_price
    C = params.initial_collateral
    L = params.initial_liquidity
    D = params.initial_demand
    
    time = np.arange(n_steps) * params.dt
    supply = np.zeros(n_steps)
    price = np.zeros(n_steps)
    collateral = np.zeros(n_steps)
    liquidity = np.zeros(n_steps)
    demand = np.zeros(n_steps)
    
    np.random.seed(params.random_seed)
    
    for t in range(n_steps):
        l_shock = liquidity_shock_magnitude if t == shock_time else 0.0
        
        supply[t] = S
        price[t] = P
        collateral[t] = C
        liquidity[t] = L
        demand[t] = D
        
        S, P, C, L, D = simulate_step(
            (S, P, C, L, D),
            params,
            liquidity_shock=l_shock
        )
    
    # Compute metrics
    peg_deviation = np.trapezoid(np.abs(price - 1.0), time)
    
    collapse_mask = price < params.collapse_price_threshold
    if np.any(collapse_mask):
        time_to_collapse = time[np.argmax(collapse_mask)]
    else:
        time_to_collapse = np.inf
    
    max_drawdown = np.max(1.0 - price)
    
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

if __name__ == "__main__":
    params = SystemParameters()
    results = run_liquidity_crisis_experiment(
        liquidity_shock_magnitude=-0.9, # 90% liquidity dry-up (Rug Pull)
        shock_time=100,
        params=params,
        n_steps=1000
    )
    
    print(f"Peg Deviation: {results.peg_deviation_integral:.4f}")
    
    os.makedirs("results/plots", exist_ok=True)
    plot_results(results, save_path="results/plots/liquidity_crisis.png")
