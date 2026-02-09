import numpy as np
from typing import Tuple
from .parameters import SystemParameters

def update_supply(
    S: float, 
    P: float, 
    L: float, 
    params: SystemParameters
) -> float:
    """
    Update stablecoin supply based on price deviation.
    
    Minting when P > 1, burning when P < 1.
    Constrained by liquidity.
    
    Args:
        S: Current supply
        P: Current price
        L: Current liquidity
        params: System parameters
        
    Returns:
        Updated supply S[t+1]
    """
    price_deviation = P - 1.0
    
    if price_deviation > 0:
        # Mint new coins (limited by liquidity)
        delta_mint = params.mint_coefficient * price_deviation * S
        delta_mint = min(delta_mint, L * 0.1)  # Max 10% of liquidity
        return S + delta_mint
    else:
        # Burn coins
        delta_burn = params.burn_coefficient * abs(price_deviation) * S
        delta_burn = min(delta_burn, S * 0.5)  # Max 50% of supply
        return S - delta_burn

def update_price(
    S: float,
    D: float,
    L: float,
    params: SystemParameters
) -> float:
    """
    Market clearing price based on supply, demand, liquidity.
    
    Uses simple supply-demand model with liquidity friction.
    
    Args:
        S: Current supply
        D: Current demand
        L: Current liquidity
        params: System parameters
        
    Returns:
        Market price P[t+1]
    """
    # Base price from supply-demand
    base_price = (D / S) if S > 0 else 1.0
    
    # Liquidity friction - low liquidity increases volatility
    liquidity_factor = 1.0 / (1.0 + params.liquidity_depth / max(L, 1e-6))
    
    # Price with friction
    price = base_price * (1.0 + liquidity_factor * 0.1)
    
    return max(price, 0.0)  # Price cannot be negative

def update_collateral(
    C: float,
    shock: float = 0.0
) -> float:
    """
    Update collateral value (exogenous shocks).
    
    Args:
        C: Current collateral value
        shock: Percentage shock (e.g., -0.3 for 30% drop)
        
    Returns:
        Updated collateral C[t+1]
    """
    return C * (1.0 + shock)

def update_liquidity(
    L: float,
    P: float,
    S: float,
    params: SystemParameters
) -> float:
    """
    Update market liquidity based on price stability.
    
    Liquidity decreases when price deviates from peg.
    
    Args:
        L: Current liquidity
        P: Current price
        S: Current supply
        params: System parameters
        
    Returns:
        Updated liquidity L[t+1]
    """
    # Liquidity flees when price is unstable
    stability = 1.0 - abs(P - 1.0)
    liquidity_change = 0.01 * stability * L
    
    return max(L + liquidity_change, params.liquidity_depth * 0.1)

def update_demand(
    D: float,
    P: float,
    params: SystemParameters
) -> float:
    """
    Update aggregate demand based on price.
    
    Demand decreases when price is above peg (elastic).
    
    Args:
        D: Current demand
        P: Current price
        params: System parameters
        
    Returns:
        Updated demand D[t+1]
    """
    # Elastic demand - decreases when price rises
    price_effect = -params.demand_elasticity * (P - 1.0)
    demand_change = price_effect * D
    
    return max(D + demand_change, D * 0.1)  # Minimum 10% of initial demand

def simulate_step(
    state: Tuple[float, float, float, float, float],
    params: SystemParameters,
    collateral_shock: float = 0.0,
    liquidity_shock: float = 0.0
) -> Tuple[float, float, float, float, float]:
    """
    Simulate one time step of the system.
    
    Args:
        state: Tuple of (S, P, C, L, D)
        params: System parameters
        collateral_shock: Exogenous collateral shock
        liquidity_shock: Exogenous liquidity shock
        
    Returns:
        Updated state tuple (S', P', C', L', D')
    """
    S, P, C, L, D = state
    
    # Update each variable
    C_new = update_collateral(C, collateral_shock)
    L_new = update_liquidity(L, P, S, params)
    L_new = L_new * (1.0 + liquidity_shock)  # Apply shock
    D_new = update_demand(D, P, params)
    S_new = update_supply(S, P, L_new, params)
    P_new = update_price(S_new, D_new, L_new, params)
    
    return (S_new, P_new, C_new, L_new, D_new)
