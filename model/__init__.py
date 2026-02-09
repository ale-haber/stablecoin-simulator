"""
Formal Economic Simulator for Algorithmic Stablecoins - Core Model.

This module provides the mathematical foundation for simulating
algorithmic stablecoin dynamics, including state transition functions
and parameter definitions.
"""

from .parameters import SystemParameters
from .dynamics import (
    update_supply,
    update_price,
    update_collateral,
    update_liquidity,
    update_demand,
    simulate_step,
)
from .market import (
    arbitrage_opportunity,
    reflexivity_coefficient,
)

__all__ = [
    # Parameters
    "SystemParameters",
    # Dynamics
    "update_supply",
    "update_price",
    "update_collateral",
    "update_liquidity",
    "update_demand",
    "simulate_step",
    # Market
    "arbitrage_opportunity",
    "reflexivity_coefficient",
]
