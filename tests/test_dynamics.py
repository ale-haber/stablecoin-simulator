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
    P = 1.1
    L = 1e6
    C = 1.5e6  # Fully collateralized

    S_new = update_supply(S_initial, P, L, C, params)
    assert S_new > S_initial

def test_supply_decreases_when_price_below_peg():
    """Supply should decrease when P < 1 (if mechanism works)."""
    params = SystemParameters()
    S_initial = 1e6
    P = 0.95  # Mild depeg (mechanism still works)
    L = 1e6
    C = 1.5e6  # Fully collateralized

    S_new = update_supply(S_initial, P, L, C, params)
    assert S_new < S_initial

def test_burn_mechanism_fails_under_stress():
    """Burn mechanism should be ineffective when CR is low."""
    params = SystemParameters()
    S_initial = 1e6
    P = 0.7  # Significant depeg
    L = 1e6
    C = 0.5e6  # Severely undercollateralized (CR = 0.5)

    S_new = update_supply(S_initial, P, L, C, params)
    # Burn should be much smaller than normal due to low effectiveness
    normal_burn = params.burn_coefficient * 0.3 * S_initial
    actual_burn = S_initial - S_new
    assert actual_burn < normal_burn * 0.5  # Less than 50% of normal

def test_price_reflects_supply_demand():
    """Price should reflect supply-demand ratio."""
    params = SystemParameters()

    P1 = update_price(S=1e6, D=2e6, L=1e6, C=1.5e6, params=params)
    P2 = update_price(S=2e6, D=1e6, L=1e6, C=3e6, params=params)

    assert P1 > P2

def test_price_capped_by_collateral_ratio():
    """Price should be pulled toward CR when undercollateralized."""
    params = SystemParameters()
    
    # Severely undercollateralized: CR = 0.5
    P = update_price(S=1e6, D=1e6, L=1e6, C=0.5e6, params=params)
    
    # Price should be significantly below 1.0, pulled toward 0.5
    assert P < 0.85  # Price pulled toward CR=0.5

def test_collateral_shock_applies_correctly():
    """Collateral should change by shock percentage."""
    C_initial = 1e6
    shock = -0.3
    P = 1.0  # Stable price, no reflexive crash
    S = 1e6

    C_new = update_collateral(C_initial, P, S, shock)
    assert np.isclose(C_new, C_initial * 0.7)

def test_collateral_reflexive_crash():
    """Collateral should crash when price depegs (reflexivity)."""
    C_initial = 1e6
    P = 0.7  # Significant depeg
    S = 1e6
    shock = 0.0  # No external shock

    C_new = update_collateral(C_initial, P, S, shock)
    # Should crash due to reflexivity
    assert C_new < C_initial

def test_liquidity_flees_instability():
    """Liquidity should decrease when price is unstable."""
    params = SystemParameters()
    L_initial = params.initial_liquidity
    P = 0.8  # Significant depeg
    S = 1e6
    C = 0.8e6  # Low CR

    L_new = update_liquidity(L_initial, P, S, C, params)
    assert L_new < L_initial

def test_demand_panics_at_low_cr():
    """Demand should crash when collateral ratio drops."""
    params = SystemParameters()
    D_initial = params.initial_demand
    P = 0.9
    S = 1e6
    C = 0.6e6  # CR = 0.6, below panic threshold
    L = params.initial_liquidity

    D_new = update_demand(D_initial, P, S, C, L, params)
    # Should see significant panic-driven demand destruction
    assert D_new < D_initial * 0.8

def test_simulate_step_maintains_positive_values():
    """All state variables should remain positive."""
    params = SystemParameters()
    state = (1e6, 1.0, 1.5e6, 1e6, 1e6)

    new_state = simulate_step(state, params)
    assert all(x > 0 for x in new_state)

def test_death_spiral_occurs():
    """System should collapse under severe shock."""
    params = SystemParameters()
    state = (1e6, 1.0, 1.5e6, 1e6, 1e6)  # S, P, C, L, D
    
    # Apply severe collateral shock
    state = simulate_step(state, params, collateral_shock=-0.5)
    
    # Run for many steps
    for _ in range(100):
        state = simulate_step(state, params)
    
    S, P, C, L, D = state
    # Price should have collapsed significantly
    assert P < 0.5, f"Expected collapse, but P={P:.3f}"
