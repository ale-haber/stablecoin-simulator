"""
Formal Economic Simulator for Algorithmic Stablecoins - Experiments.

This module provides experiment implementations for stress testing
algorithmic stablecoin systems under various shock scenarios.
"""

from .collateral_shock import (
    ExperimentResults,
    run_collateral_shock_experiment,
    plot_results,
)
from .liquidity_crisis import run_liquidity_crisis_experiment
from .monte_carlo import (
    MonteCarloResults,
    run_monte_carlo_stress_test,
)

__all__ = [
    # Collateral shock
    "ExperimentResults",
    "run_collateral_shock_experiment",
    "plot_results",
    # Liquidity crisis
    "run_liquidity_crisis_experiment",
    # Monte Carlo
    "MonteCarloResults",
    "run_monte_carlo_stress_test",
]
