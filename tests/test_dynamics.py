import pytest
import numpy as np
from model.dynamics import (
    update_supply, update_price, update_collateral,
    update_liquidity, update_demand, simulate_step
)
from model.parameters import SystemParameters

def test_supply_increases_when_price_above_peg():
    """Supply should increase when P > 1."""
    params = SystemParameters()
    S_initial = 1e6
    P = 1.1  # Above peg
    L = 1e6
    
    S_new = update_supply(S_initial, P, L, params)
    assert S_new > S_initial

def test_supply_decreases_when_price_below_peg():
    """Supply should decrease when P < 1."""
    params = SystemParameters()
    S_initial = 1e6
    P = 0.9  # Below peg
    L = 1e6
    
    S_new = update_supply(S_initial, P, L, params)
    assert S_new < S_initial

def test_price_reflects_supply_demand():
    """Price should reflect supply-demand ratio."""
    params = SystemParameters()
    
    # High demand, low supply → high price
    P1 = update_price(S=1e6, D=2e6, L=1e6, params=params)
    
    # Low demand, high supply → low price
    P2 = update_price(S=2e6, D=1e6, L=1e6, params=params)
    
    assert P1 > P2

def test_collateral_shock_applies_correctly():
    """Collateral should change by shock percentage."""
    C_initial = 1e6
    shock = -0.3  # 30% drop
    
    C_new = update_collateral(C_initial, shock)
    assert np.isclose(C_new, C_initial * 0.7)

def test_simulate_step_maintains_positive_values():
    """All state variables should remain positive."""
    params = SystemParameters()
    state = (1e6, 1.0, 1.5e6, 1e6, 1e6)
    
    new_state = simulate_step(state, params)
    
    assert all(x > 0 for x in new_state)

def test_deterministic_simulation():
    """Same seed should produce same results."""
    params = SystemParameters(random_seed=42)
    state = (1e6, 1.0, 1.5e6, 1e6, 1e6)
    
    result1 = simulate_step(state, params)
    result2 = simulate_step(state, params)
    
    assert result1 == result2
