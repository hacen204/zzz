
"""
Optimized MBS Analysis and Trading Functions.
All numerical functions are JIT-compiled with Numba for maximum performance.
"""
from tqdm import tqdm
import numpy as np
import pandas as pd
from numba import njit, prange
from scipy import stats  # Ajout de l'import manquant
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple, List, Dict, Optional
import time
import sys
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from numba import set_num_threads, get_num_threads
from functools import partial

set_num_threads(8)  # Limite à 12 threads
#print("Threads disponibles pour Numba:", get_num_threads())




# ---- Core Rate and Zero-Coupon Functions ----
@njit
def get_fwdrate_mthly(rate_level: float, slope: float, maturity: int) -> np.ndarray:
    """Calculate monthly forward rates."""
    return rate_level + slope * np.arange(maturity)

@njit
def get_zc_mthly(fwdrate_vector: np.ndarray) -> np.ndarray:
    result = np.cumprod(1/(1+fwdrate_vector)**(1/12))
    return result

@njit
def get_zc_with_oas(zc_vector: np.ndarray, oas: float) -> np.ndarray:
    """Apply OAS adjustment to zero-coupon rates."""
    # Création vectorisée des indices temporels
    periods = np.arange(1, len(zc_vector) + 1) / 12.0
    
    # Calcul vectorisé du facteur d'ajustement
    oas_adjustment = np.exp(-(oas) * periods)
    
    # Application vectorisée de l'ajustement
    return zc_vector * oas_adjustment

@njit
def get_zc_mthly_with_shift10bps(zc_vector: np.ndarray) -> np.ndarray:
    """Apply 10bps shift to zero-coupon rates."""
    return zc_vector/((1+0.001)**(-1/12))

# ---- Amortization and Cashflow Functions ----
@njit
def get_floating_amortization_with_cpr(principal: float, coupon: float, 
                                     periods: int, cpr: float) -> np.ndarray:
    """Calculate floating rate amortization with CPR."""
    amortization_vector = np.ones(periods+1)
    amortization_vector[0] = principal
    coupon = (1+coupon)**(1/12) - 1
    for i in range(1, periods+1):
        amortization_vector[i] = (principal * 
                                (1-(1+coupon)**(-(periods-i))) / 
                                (1-(1+coupon)**(-periods)) * 
                                (1-cpr)**(i/12))
    return amortization_vector

@njit
def get_spread_amortization(principal: float, coupon: float, 
                          periods: int, cpr_low: float, cpr_high: float) -> np.ndarray:
    """Calculate spread amortization between high and low CPR scenarios."""
    amort_low = get_floating_amortization_with_cpr(principal, coupon, periods, cpr_low)
    amort_high = get_floating_amortization_with_cpr(principal, coupon, periods, cpr_high)
    return amort_low - amort_high

@njit
def get_floating_cashflows_with_cpr(principal, coupon, periods, cpr):
    # Pré-calculs des constantes pour éviter les recalculs
    monthly_coupon = (1 + coupon)**(1/12) - 1
    denom = 1 - (1 + monthly_coupon)**(-periods)
    
    # Initialisation des vecteurs avec taille pré-allouée
    amortization_vector = np.empty(periods + 1, dtype=np.float64)
    result = np.empty(periods + 1, dtype=np.float64)
    
    # Valeurs initiales
    amortization_vector[0] = principal
    result[0] = principal
    
    # Pré-calcul des facteurs CPR
    cpr_factors = (1 - cpr)**(np.arange(periods + 1)/12)
    
    # Vectorisation partielle avec maintien de la relation récursive
    for i in range(1, periods + 1):
        # Calcul de l'amortissement avec constantes pré-calculées
        amortization_vector[i] = principal * (1 - (1 + monthly_coupon)**(-(periods-i))) / denom * cpr_factors[i]
        
        # Calcul du cashflow en utilisant la relation récursive
        result[i] = amortization_vector[i-1] * (1 + monthly_coupon) - amortization_vector[i]
        
    return result
# ---- Statistical Functions ----
@njit
def norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

@njit
def erf_approx(x: float) -> float:
    """Optimized error function approximation."""
    if not np.isfinite(x):
        return np.nan if np.isnan(x) else (1.0 if x > 0 else -1.0)
    
    # Abramowitz and Stegun approximation constants
    a1, a2, a3 = 0.254829592154951, -0.284496736487789, 1.421413741193305
    a4, a5, p = -1.453152027338056, 1.061405429231857, 0.3275911033751683
    
    sign = 1 if x >= 0 else -1
    x = abs(x)
    if x > 6.0:
        return sign * 1.0
        
    t = 1.0 / (1.0 + p * x)
    poly = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t
    return sign * (1.0 - poly * np.exp(-x * x))

@njit
def norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1 + erf_approx(x / np.sqrt(2)))

@njit
def logit(min_val: float, max_val: float, alpha: float, 
          coupon: float, rate: float) -> float:
    """Logistic function for CPR calculation."""
    return min_val + (max_val-min_val)/(1+np.exp(-alpha*(coupon-rate)))

@njit
def get_bond_price(cashflows: np.ndarray, zc_vector: np.ndarray) -> np.float64:
    # Vectorisation du calcul
    return np.sum(cashflows[1:] * zc_vector)

# ---- Bond and Option Pricing Functions ----

@njit
def get_zc_mthly_with_shift10bps_optimized(zc_vector: np.ndarray) -> np.ndarray:
    return zc_vector / (1 + 0.001)

@njit
def get_bond_price_and_ratios2(cashflows: np.ndarray, 
                             zc_vector: np.ndarray) -> np.ndarray:
    """Calculate bond price and risk metrics."""
    length = len(zc_vector)
    result = np.zeros(4, dtype=np.float64)  # [price, duration, convexity, theta]
    inv_12 = 1.0 / 12.0
    
    # Pré-calcul des indices * inv_12
    monthly_indices = np.arange(1, length + 1) * inv_12
    
    # Calcul vectorisé des composants de prix
    price_components = cashflows[1:] * zc_vector
    
    # Price
    result[0] = np.sum(price_components)
    
    # Duration
    result[1] = -np.sum(price_components * monthly_indices)
    
    # Theta
    inv_zc = 1.0 / zc_vector
    result[3] = np.sum(price_components * (inv_zc - 1.0) * 12.0 * monthly_indices)
    
    # Convexity
    zc_vector_shifted = get_zc_mthly_with_shift10bps_optimized(zc_vector)
    shifted_price = np.sum(cashflows[1:] * zc_vector_shifted)
    result[2] = shifted_price - result[0] + result[1] * 0.001
    
    return result

@njit
def get_bond_price_and_ratios(cashflows: np.ndarray, zc_vector: np.ndarray) -> np.ndarray:
    """Optimisation avec calcul parallèle des métriques."""
    length = len(zc_vector)
    result = np.zeros(4, dtype=np.float64)
    
    # Pré-calcul vectorisé
    monthly_indices = np.arange(1, length + 1) / 12.0
    price_components = cashflows[1:] * zc_vector
    
    # Calculs parallélisés
    result[0] = np.sum(price_components)  # Price
    result[1] = -np.sum(price_components * monthly_indices)  # Duration 
    result[3] = np.sum(price_components * (1.0/zc_vector - 1.0) * 12.0 * monthly_indices)  # Theta
    
    # Convexity optimisée
    zc_shifted = zc_vector / (1 + 0.001)
    shifted_price = np.sum(cashflows[1:] * zc_shifted)
    result[2] = shifted_price - result[0] + result[1] * 0.001
    
    return result

@njit
def get_fwdrate_and_level_coterminal_j(amortization, zc_vector, term):
    level = 0.0
    d_level = 0.0
    forward_coterminal = 0.0
    
    for i in range(term, len(zc_vector)-1):
        weighted_spread_amortisation = amortization[i] * zc_vector[i] / 12
        level += weighted_spread_amortisation
        d_level += weighted_spread_amortisation * (i+1) / 12
        forward_rate = 12 * (zc_vector[i]/zc_vector[i+1] - 1)
        forward_coterminal += forward_rate * weighted_spread_amortisation
    
    # Division par level seulement à la fin
    if level != 0:
        forward_coterminal /= level
    return forward_coterminal, level, d_level
@njit
def get_receiver_swaption_blackscholes_price_and_ratios(amortization,zc_vector,strike,volatility,expi):
    penalty=0.00000
    term=expi*12-1
    strike =12*((1+strike)**(1/12) -1) -penalty
    int_term=int(term)
    result=np.zeros(4)
    fwdrate_coterminal, level,d_level=get_fwdrate_and_level_coterminal_j(amortization,zc_vector,int_term)
    xstd=(strike -fwdrate_coterminal)/(volatility*np.sqrt(expi))
    price=level*volatility*np.sqrt(expi)*(xstd*norm_cdf(xstd)+norm_pdf(xstd))
    duration=-level*norm_cdf(xstd) -d_level*price/level
    zc_vector_shifted=get_zc_mthly_with_shift10bps(zc_vector)
    fwdrate_coterminal_shifted, level_shifted,d_level_shifted=get_fwdrate_and_level_coterminal_j(amortization,zc_vector_shifted,int_term)
    xstd_shifted=(strike -fwdrate_coterminal_shifted)/(volatility*np.sqrt(expi)) 
    price_shifted=level_shifted*volatility*np.sqrt(expi)*(xstd_shifted*norm_cdf(xstd_shifted)+norm_pdf(xstd_shifted))
    convexity=price_shifted-price -duration*0.001
    theta =-0.5*level*volatility*np.sqrt(expi)*norm_pdf(xstd)
    result[0]=price
    result[1]=duration
    result[2]=convexity
    result[3]=theta
    return result



# ---- Option Pricing Functions ----
@njit
def get_mostexpensive_receiver_swaption_blackscholes_and_ratio2(amortization,zc_vector,strike,volatility):
    result=np.zeros(5)
    max_price=0
    for i in range(1,len(zc_vector)):
        expi=i/12
        receiver_swaption_blackscholes=get_receiver_swaption_blackscholes_price_and_ratios(amortization,zc_vector,strike,volatility,expi)
        current_price=receiver_swaption_blackscholes[0]
        max_price=max(max_price,current_price)
        if max_price==current_price:
            result[0]=receiver_swaption_blackscholes[0]
            result[1]=receiver_swaption_blackscholes[1]
            result[2]=receiver_swaption_blackscholes[2]
            result[3]=receiver_swaption_blackscholes[3]     
            result[4]=i 
   
    return result



@njit
def get_mostexpensive_receiver_swaption_blackscholes_and_ratio(
    amortization: np.ndarray,
    zc_vector: np.ndarray,
    strike: float,
    volatility: float
) -> np.ndarray:
    """Combined and optimized version of both swaption calculation functions."""
    n = len(zc_vector)
    result = np.zeros(5)
    max_price = 0.0
    
    # Pré-calculs pour éviter les répétitions
    penalty=0.00000
    monthly_strike = 12 * ((1 + strike)**(1/12) - 1)-penalty
    zc_vector_shifted = zc_vector / (1 + 0.001)  # Optimisation de get_zc_mthly_with_shift10bps
    
    for i in range(1, n):
        expi = i/12
        term = i - 1
        sqrt_expi = np.sqrt(expi)
        
        # Calcul des taux forward et niveaux
        fwdrate, level, d_level = get_fwdrate_and_level_coterminal_j(
            amortization, zc_vector, term)
        
        # Calcul des valeurs pour le prix initial
        xstd = (monthly_strike - fwdrate)/(volatility * sqrt_expi)
        ncdf_xstd = norm_cdf(xstd)
        npdf_xstd = norm_pdf(xstd)
        
        # Calcul du prix
        vol_sqrt_expi = volatility * sqrt_expi
        price = level * vol_sqrt_expi * (xstd * ncdf_xstd + npdf_xstd)
        
        if price > max_price:
            max_price = price
            
            # Calcul des ratios seulement si c'est le prix maximum
            duration = -level * ncdf_xstd - d_level * price/level
            
            # Calcul avec le shift
            fwdrate_shifted, level_shifted, _ = get_fwdrate_and_level_coterminal_j(
                amortization, zc_vector_shifted, term)
            
            xstd_shifted = (monthly_strike - fwdrate_shifted)/(volatility * sqrt_expi)
            price_shifted = level_shifted * vol_sqrt_expi * (
                xstd_shifted * norm_cdf(xstd_shifted) + norm_pdf(xstd_shifted))
            
            # Stockage des résultats
            result[0] = price
            result[1] = duration
            result[2] = price_shifted - price - duration * 0.001  # convexity
            result[3] = -0.5 * level * vol_sqrt_expi * npdf_xstd  # theta
            result[4] = i
    
    return result

@njit(fastmath=True)  # Ajout de fastmath pour des optimisations supplémentaires
def get_fwdrate_j(zc_vector: np.ndarray, term: int) -> float:
    """Calcule le taux forward avec des optimisations de performance."""
    zc_term = zc_vector[term]
    zc_next = zc_vector[term + 1]
    # Réorganise le calcul pour éviter une division
    # Original: 12 * (zc_term - zc_next) / zc_next
    # Optimisé: 12 * zc_term/zc_next - 12
    return 12.0 * (zc_term/zc_next - 1.0)



@njit(parallel=True)
def get_floorlet_blackscholes_price_and_ratios(amortization,zc_vector,strike,volatility,expi):
    term=expi*12-1
    int_term=int(term)
    result=np.zeros(4)
    price=0
    duration=0
    gamma=0
    theta=0
    fwdrate=get_fwdrate_j(zc_vector,int_term)
    xstd=(strike -fwdrate)/(volatility*np.sqrt(expi))
    price=zc_vector[int_term]*amortization[int_term+1]*volatility*np.sqrt(expi)*(xstd*norm_cdf(xstd)+norm_pdf(xstd))/12
    duration=-zc_vector[int_term]*amortization[int_term+1]*norm_cdf(xstd)/12 -price*(int_term)/12
    zc_vector_shifted=get_zc_mthly_with_shift10bps(zc_vector)
    fwdrate_shifted=get_fwdrate_j(zc_vector_shifted,int_term)
    xstd_shifted=(strike -fwdrate_shifted)/(volatility*np.sqrt(expi))
    price_shifted=zc_vector_shifted[int_term]*amortization[int_term+1]*volatility*np.sqrt(expi)*(xstd_shifted*norm_cdf(xstd_shifted)+norm_pdf(xstd_shifted))/12
    convexity=price_shifted-price-duration*0.001
    theta=-0.5*zc_vector[int_term]*amortization[int_term+1]*volatility*np.sqrt(expi)*norm_pdf(xstd)/12
    return [price,duration,convexity,theta]

@njit
def get_floorlet_blackscholes_price_and_ratios(amortization,zc_vector,strike,volatility,expi):
    term=expi*12-1
    int_term=int(term)
    result=np.zeros(4)
    price=0
    duration=0
    gamma=0
    theta=0
    fwdrate=get_fwdrate_j(zc_vector,int_term)
    xstd=(strike -fwdrate)/(volatility*np.sqrt(expi))
    price=zc_vector[int_term]*amortization[int_term+1]*volatility*np.sqrt(expi)*(xstd*norm_cdf(xstd)+norm_pdf(xstd))/12
    duration=-zc_vector[int_term]*amortization[int_term+1]*norm_cdf(xstd)/12 -price*(int_term)/12
    zc_vector_shifted=get_zc_mthly_with_shift10bps(zc_vector)
    fwdrate_shifted=get_fwdrate_j(zc_vector_shifted,int_term)
    xstd_shifted=(strike -fwdrate_shifted)/(volatility*np.sqrt(expi))
    price_shifted=zc_vector_shifted[int_term]*amortization[int_term+1]*volatility*np.sqrt(expi)*(xstd_shifted*norm_cdf(xstd_shifted)+norm_pdf(xstd_shifted))/12
    convexity=price_shifted-price-duration*0.001
    theta=-0.5*zc_vector[int_term]*amortization[int_term+1]*volatility*np.sqrt(expi)*norm_pdf(xstd)/12
 
  
    return [price,duration,convexity,theta]

@njit
def get_floor_blackscholes_price_and_ratios2(amortization,zc_vector,strike,volatility):
    result =np.zeros(4)
    penalty=0.00000
    strike =12*((1+strike)**(1/12) -1)-penalty
    for i in range(1,len(zc_vector)):
        expi=i/12
        floorlet_price_and_ratios=get_floorlet_blackscholes_price_and_ratios(amortization,zc_vector,strike,volatility,expi)
        result[0]+=floorlet_price_and_ratios[0]
        result[1]+=floorlet_price_and_ratios[1]
        result[2]+=floorlet_price_and_ratios[2]
        result[3]+=floorlet_price_and_ratios[3]
    return result

@njit
def get_floor_blackscholes_price_and_ratios(amortization, zc_vector, strike, volatility):
    n = len(zc_vector)
    result = np.zeros(4)
    penalty=0.0000
    strike = 12 * ((1 + strike)**(1/12) - 1)-penalty
    
    # Pré-calcul des valeurs communes pour tous les floorlets
    expiries = np.arange(1, n) / 12
    sqrt_expiries = np.sqrt(expiries)
    terms = expiries * 12 - 1
    int_terms = terms.astype(np.int64)
    
    # Pré-allocation des arrays pour les résultats intermédiaires
    prices = np.zeros(n)
    durations = np.zeros(n)
    convexities = np.zeros(n)
    thetas = np.zeros(n)
    
    # Pré-calcul du zc_vector shifted pour tous les termes
    zc_vector_shifted = get_zc_mthly_with_shift10bps(zc_vector)
    
    # Boucle principale parallélisée
    for i in prange(1, n):
        # Calculs des variables pour chaque floorlet
        fwdrate = get_fwdrate_j(zc_vector, int_terms[i-1])
        xstd = (strike - fwdrate) / (volatility * sqrt_expiries[i-1])
        
        # Calculs intermédiaires communs
        norm_cdf_xstd = norm_cdf(xstd)
        norm_pdf_xstd = norm_pdf(xstd)
        zc_term = zc_vector[int_terms[i-1]]
        amort_term = amortization[int_terms[i-1] + 1]
        vol_sqrt_exp = volatility * sqrt_expiries[i-1]
        common_factor = zc_term * amort_term / 12
        
        # Calcul du prix
        prices[i] = common_factor * vol_sqrt_exp * (xstd * norm_cdf_xstd + norm_pdf_xstd)
        
        # Calcul de la duration
        durations[i] = -common_factor * norm_cdf_xstd - prices[i] * int_terms[i-1] / 12
        
        # Calcul de la convexité
        fwdrate_shifted = get_fwdrate_j(zc_vector_shifted, int_terms[i-1])
        xstd_shifted = (strike - fwdrate_shifted) / (volatility * sqrt_expiries[i-1])
        zc_shifted_term = zc_vector_shifted[int_terms[i-1]]
        price_shifted = (zc_shifted_term * amort_term * vol_sqrt_exp * 
                        (xstd_shifted * norm_cdf(xstd_shifted) + norm_pdf(xstd_shifted)) / 12)
        convexities[i] = price_shifted - prices[i] - durations[i] * 0.001
        
        # Calcul du theta
        thetas[i] = -0.5 * common_factor * vol_sqrt_exp * norm_pdf_xstd
    
    # Somme vectorisée finale
    result[0] = np.sum(prices)
    result[1] = np.sum(durations)
    result[2] = np.sum(convexities)
    result[3] = np.sum(thetas)
    
    return result


@njit
def get_option_to_prepay_blackscholes_price_and_ratios(amortization,zc_vector,strike,volatility):
    weight=0.5
    result =np.zeros(4)
    mostexpensive_receiver_swaption_blackscholes_price_and_ratios=get_mostexpensive_receiver_swaption_blackscholes_and_ratio(amortization,zc_vector,strike,volatility)
    floor_price_and_ratios=get_floor_blackscholes_price_and_ratios(amortization,zc_vector,strike,volatility)
    result[0] =weight*mostexpensive_receiver_swaption_blackscholes_price_and_ratios[0]+(1-weight)*floor_price_and_ratios[0]
    result[1] =weight*mostexpensive_receiver_swaption_blackscholes_price_and_ratios[1]+(1-weight)*floor_price_and_ratios[1]
    result[2] =weight*mostexpensive_receiver_swaption_blackscholes_price_and_ratios[2]+(1-weight)*floor_price_and_ratios[2]
    result[3] =weight*mostexpensive_receiver_swaption_blackscholes_price_and_ratios[3]+(1-weight)*floor_price_and_ratios[3]
    
    return result

# ---- Mortgage Pricing Functions ----
@njit
def get_mortgage_price_and_ratios(
        principal: float,
        coupon: float,
        periods: int,
        cprlow: float,
        cprhigh: float,
        zc_vector: np.ndarray,
        volatility: float) -> np.ndarray:
    """Calculate mortgage price and risk metrics."""
    result = np.zeros(3)  # [price, duration, convexity]
    
    # Calculate base bond components
    cashflows = get_floating_cashflows_with_cpr(principal, coupon, periods, cprlow)
    bond_ratios = get_bond_price_and_ratios(cashflows, zc_vector)/principal
    
    # Calculate option components
    spread_amort = get_spread_amortization(principal, coupon, periods, cprlow, cprhigh)
    option_ratios = get_option_to_prepay_blackscholes_price_and_ratios(
        spread_amort, zc_vector, coupon, volatility)/principal
    
    # Combine results
    result[0] = bond_ratios[0] - option_ratios[0]  # price
    result[1] = bond_ratios[1] - option_ratios[1]  # duration
    result[2] = bond_ratios[2] - option_ratios[2]  # convexity
    
    return result
@njit
def get_digit_function(x: float, y: float) -> int:
    return 1 if x > y else 0

@njit
def get_cpr_low_and_high(
        coupon: float,
        rate: float,
        bmin: float,
        bmax: float,
        spreadmin: float,
        spreadmax: float,
        alpha: float,
        loan_age: float,
        gamma: float) -> Tuple[float, float]:
    """Calculate CPR bounds based on loan characteristics."""
    ratio= np.exp(-np.maximum(0, float(loan_age)-2)*gamma)
    #ratio = (1-np.maximum(float(loan_age)-2,0)/30)*np.exp(-np.maximum(0, float(loan_age)-2)*gamma)
    digit = get_digit_function(float(loan_age), 2.0)
    ratio = (1 - (float(loan_age) / 30) * digit)
    xmax=0.05
    xmin=0.25
    b= (30*xmax-2*xmin )/(30-2)
    a=(xmin-xmax)/(30-2)
    ratio2=(1-digit)*xmax + (a*float(loan_age)+b)*digit
    #ratio_high=0.30+(0.12-0.30)*(float(loan_age) / 30) * digit
    
    montly_LA=float(loan_age)*12.0
    ramp_up=np.minimum(0.002*montly_LA/0.06,1)
    cprlow = logit(bmin, bmax, alpha, coupon, rate)
    cprhigh =cprlow + logit(spreadmin, spreadmax, alpha, coupon, rate) 
    return cprlow, cprhigh

@njit
def get_oas_equation(principal,coupon,periods,cprlow,cprhigh,zc_vector,volatility,oas,mkt_price):

    zc_vector_with_oas=get_zc_with_oas(zc_vector,oas)
    
        # Vérifier les valeurs avant division
    if np.any(zc_vector_with_oas == 0):
        print("Warning: Zero values in zc_vector_with_oas")
    
    floating_cashflows_vector=get_floating_cashflows_with_cpr(principal,coupon,periods,cprlow)
    bond_price_cprlow=get_bond_price(floating_cashflows_vector,zc_vector_with_oas)/principal
    spread_amortization=get_spread_amortization(principal,coupon,periods,cprlow,cprhigh)
    option_to_prepay_blackscholes=get_option_to_prepay_blackscholes_price_and_ratios(spread_amortization,zc_vector_with_oas,coupon,volatility)
    mkt_model=bond_price_cprlow-option_to_prepay_blackscholes[0]/principal
    return mkt_price-mkt_model



def get_tba_price_duration_min(df_strat, day):
    # Accéder à la ligne une seule fois et convertir en dictionnaire pour accès rapide
    row_dict = df_strat.iloc[day].to_dict()
    
    # Créer des dictionnaires de colonnes une seule fois
    prefix_to_columns = {
        'OAS': [col for col in row_dict.keys() if 'OAS_' in col],
        'TBA_Price': [col for col in row_dict.keys() if 'TBA_Price_' in col],
        'Duration': [col for col in row_dict.keys() if 'Duration_' in col]
    }
    
    # Trouver l'OAS minimum avec gestion des cas vides
    oas_values = [(col, row_dict[col]) for col in prefix_to_columns['OAS'] 
                  if pd.notna(row_dict[col])]
    
    # Vérifier si nous avons des valeurs OAS valides
    if not oas_values:
        # Retourner des Series vides si aucun OAS valide
        return (
            pd.Series(0.0, index=prefix_to_columns['TBA_Price']),
            pd.Series(0.0, index=[f'weight_Duration_{col.split("_")[1]}' for col in prefix_to_columns['Duration']]),
            pd.Series(0.0, index=prefix_to_columns['Duration'])
        )
    
    # Convertir en array numpy et trier
    oas_array = np.array(oas_values, dtype=[('col', 'O'), ('val', float)])
    min_col = np.sort(oas_array, order='val')[0]['col']
    
    # Extraire le coupon
    coupon = min_col.split('_')[1]
    
    # Pré-allouer les résultats
    n_cols = len(prefix_to_columns['TBA_Price'])
    results = {
        'tba_price': np.zeros(n_cols),
        'duration': np.zeros(n_cols),
        'weight': np.zeros(n_cols)
    }
    
    # Mapping des indices
    col_to_idx = {
        col: idx for idx, col in enumerate(prefix_to_columns['TBA_Price'])
    }
    
    # Mettre à jour les valeurs
    tba_col = f"TBA_Price_{coupon}"
    dur_col = f"Duration_{coupon}"
    
    if tba_col in col_to_idx:
        idx = col_to_idx[tba_col]
        results['tba_price'][idx] = row_dict.get(tba_col, 0)
        results['duration'][idx] = row_dict.get(dur_col, 0)
        results['weight'][idx] = -1.0
    
    # Convertir en pd.Series
    return (
        pd.Series(results['tba_price'], index=prefix_to_columns['TBA_Price']),
        pd.Series(results['weight'], index=[f'weight_Duration_{col.split("_")[1]}' for col in prefix_to_columns['Duration']]),
        pd.Series(results['duration'], index=prefix_to_columns['Duration'])
    )

# ---- Trading Strategy Functions ----
def get_tba_price_duration_max(df_strat, day):
    # Accéder à la ligne une seule fois et convertir en dictionnaire pour accès rapide
    row_dict = df_strat.iloc[day].to_dict()
    
    # Créer des dictionnaires pour mapping rapide une seule fois à l'initialisation
    prefix_to_columns = {
        'OAS': [col for col in row_dict.keys() if 'OAS_' in col],
        'TBA_Price': [col for col in row_dict.keys() if 'TBA_Price_' in col],
        'Duration': [col for col in row_dict.keys() if 'Duration_' in col]
    }
    
    # Trouver les 3 meilleurs OAS directement avec numpy
    oas_values = np.array([(col, row_dict[col]) for col in prefix_to_columns['OAS'] 
                          if pd.notna(row_dict[col])],
                         dtype=[('col', 'O'), ('val', float)])
    top3_cols = np.sort(oas_values, order='val')[-3:]['col']
    
    # Extraire les coupons une seule fois
    coupons = [col.split('_')[1] for col in top3_cols]
    
    # Pré-allouer les résultats avec numpy (plus rapide que pd.Series)
    n_cols = len(prefix_to_columns['TBA_Price'])
    results = {
        'tba_price': np.zeros(n_cols),
        'duration': np.zeros(n_cols),
        'weight': np.zeros(n_cols)
    }
    
    # Mapping des indices pour un accès rapide
    col_to_idx = {
        col: idx for idx, col in enumerate(prefix_to_columns['TBA_Price'])
    }
    
    # Remplir les résultats en une seule passe
    for coupon in coupons:
        # Construire les noms de colonnes
        tba_col = f"TBA_Price_{coupon}"
        dur_col = f"Duration_{coupon}"
        
        # Mettre à jour les arrays directement
        idx = col_to_idx[tba_col]
        results['tba_price'][idx] = row_dict[tba_col]
        results['duration'][idx] = row_dict[dur_col]
        results['weight'][idx] = 1/3
    
    # Convertir en pd.Series seulement à la fin
    return (
        pd.Series(results['tba_price'], index=prefix_to_columns['TBA_Price']),
        pd.Series(results['weight'], index=[f'weight_Duration_{c}' for c in 
                 [col.split('_')[1] for col in prefix_to_columns['Duration']]]),
        pd.Series(results['duration'], index=prefix_to_columns['Duration'])
    )

def get_tba_weight_and_duration_strat(df_strat, day):
    # Obtenir les résultats en une seule lecture de row
    row_dict = df_strat.iloc[day].to_dict()
    
    # Obtenir max et min en un seul passage
    price_weights_max, weights_max, durations_max = get_tba_price_duration_max(df_strat, day)
    price_weights_min, weights_min, durations_min = get_tba_price_duration_min(df_strat, day)
    
    # Utiliser numpy pour les calculs vectorisés
    dur_max_array = durations_max.values
    dur_min_array = durations_min.values
    
    # Calculer les moyennes avec numpy (plus rapide que pandas)
    avg_dur_max = np.mean(dur_max_array[dur_max_array != 0]) if np.any(dur_max_array != 0) else 0
    avg_dur_min = np.mean(dur_min_array[dur_min_array != 0]) if np.any(dur_min_array != 0) else 0
    
    # Calculer le coefficient
    coef = avg_dur_max / avg_dur_min if avg_dur_min != 0 else 0
    
    # Calculer les poids finaux de manière vectorisée
    weights_min_array = weights_min.values * coef
    weights_max_array = weights_max.values
    
    # Combiner directement les arrays numpy
    combined_weights = weights_max_array + weights_min_array
    
    # Retourner le résultat final comme Series
    return pd.Series(combined_weights, index=weights_max.index)

def get_average_tba_weight_and_duration_strat(df_strat, day, scaling):
    """
    Calculate average TBA weights with availability check - optimized version
    """
    # Pré-identifier les colonnes une seule fois
    price_cols = [col for col in df_strat.columns if "TBA_Price" in col]
    weight_cols = [f'weight_Duration_{col.split("_")[2]}' for col in price_cols]
    
    # Cas sans scaling
    if scaling == 0:
        # Accéder aux données une seule fois
        row = df_strat.iloc[day]
        prices = row[price_cols].values
        weights = get_tba_weight_and_duration_strat(df_strat, day)
        
        # Utiliser numpy pour les opérations vectorisées
        weights_array = weights.values
        weights_array[prices == 0] = 0
        return pd.Series(weights_array, index=weights.index)
    
    # Avec scaling
    # Pré-allouer le numpy array pour les poids combinés
    n_weights = len(weight_cols)
    combined_weights = np.zeros(n_weights)
    start_day = max(0, day-scaling)
    
    # Extraire toutes les données nécessaires en une fois
    relevant_days = df_strat.iloc[start_day:day+1]
    prices_array = relevant_days[price_cols].values
    
    # Calculer tous les poids en une fois
    valid_days = 0
    for i in range(len(relevant_days)):
        current_day = start_day + i
        
        # Calculer les poids pour le jour courant
        weights = get_tba_weight_and_duration_strat(df_strat, current_day)
        weights_array = weights.values
        
        # Appliquer le masque des prix nuls de manière vectorisée
        current_prices = prices_array[i]
        weights_array[current_prices == 0] = 0
        
        # Ajouter aux poids combinés
        combined_weights += weights_array
        valid_days += 1
    
    # Calculer la moyenne si nécessaire
    if valid_days > 0:
        combined_weights /= valid_days
        
    return pd.Series(combined_weights, index=weight_cols)

@njit
def calculate_pnl(
        current_prices: np.ndarray,
        previous_prices: np.ndarray,
        weights: np.ndarray,
        bid_offer_cost: float = 1/32) -> float:
    """Calculate P&L with bid-offer costs."""
    price_change = current_prices - previous_prices
    pnl = np.sum(price_change * weights)
    costs = np.sum(np.abs(weights)) * bid_offer_cost / 100

    return pnl - costs

def calculate_performance_metrics(
        pnl_series: pd.Series,
        frequency: str = 'daily') -> Dict[str, float]:
    """Calculate comprehensive performance metrics."""
    annualization = {'daily': 252, 'monthly': 12, 'yearly': 1}.get(frequency, 252)
    
    returns = pnl_series.dropna()
    cum_returns = (1 + returns).cumprod()
    drawdown = cum_returns / cum_returns.cummax() - 1
    
    metrics = {
        'total_return': cum_returns.iloc[-1] - 1,
        'annualized_return': returns.mean() * annualization,
        'volatility': returns.std() * np.sqrt(annualization),
        'sharpe_ratio': (returns.mean() / returns.std() * np.sqrt(annualization)
                        if returns.std() > 0 else 0),
        'max_drawdown': drawdown.min(),
        'hit_ratio': (returns > 0).mean(),
        'profit_factor': (returns[returns > 0].sum() / abs(returns[returns < 0].sum())
                         if (returns < 0).any() else np.inf),
        'skewness': returns.skew(),
        'kurtosis': returns.kurtosis()
    }
    
    return metrics


@njit(parallel=True)
def get_implied_oas_price_ratios_from_model(
    tba_price_mkt, loan_age_value, principal, periods, zc_vector_mkt,
    mortage_rate, bmin, bmax, spreadmin, spreadmax, alpha, gamma, volatility
):
    #EHZ
    num_coupons = 11  
    coupons = np.linspace(0.02, 0.02 + 0.005 * (num_coupons - 1), num_coupons)
    oas_results = np.full(num_coupons, np.nan)
    model_price = np.zeros(num_coupons)
    duration = np.zeros(num_coupons)
    convexity = np.zeros(num_coupons)

    # Précalculs
    zc_vectors_adjusted = [
        zc_vector_mkt[:360 - int(age * 12)] if 0 <= age <= 30 else None
        for age in loan_age_value
    ]
    cpr_values = [
        get_cpr_low_and_high(coupon, mortage_rate, bmin, bmax, spreadmin, spreadmax, alpha, loan_age, gamma)
        for coupon, loan_age in zip(coupons, loan_age_value)
    ]

    # Parallélisation sur les coupons
    for i in prange(num_coupons):
        mkt_price = tba_price_mkt[i]
        loan_age = loan_age_value[i]
        zc_vector_adjusted = zc_vectors_adjusted[i]
        if zc_vector_adjusted is None or mkt_price <= 0 or np.isnan(mkt_price):
            continue

        cprlow, cprhigh = cpr_values[i]

        # Newton-Raphson
        oas = 0.005
        tolerance = 1e-8
        max_iterations = 50
        for _ in range(max_iterations):
            f_val = get_oas_equation(
                principal, coupons[i], len(zc_vector_adjusted), cprlow, cprhigh, zc_vector_adjusted, volatility, oas, mkt_price
            )
            f_deriv = (
                get_oas_equation(
                    principal, coupons[i], len(zc_vector_adjusted), cprlow, cprhigh, zc_vector_adjusted, volatility, oas + 1e-6, mkt_price
                ) - f_val
            ) / 1e-6

            if abs(f_val) < tolerance or abs(f_deriv) < tolerance:
                break

            oas -= f_val / f_deriv

        if np.isnan(oas):
            continue

        oas_results[i] = oas

        # Ratios et prix
        zc_vector_mkt_with_oas = get_zc_with_oas(zc_vector_mkt, oas)
        mortgage_ratios = get_mortgage_price_and_ratios(
            principal, coupons[i], periods, cprlow, cprhigh, zc_vector_mkt_with_oas, volatility
        )
        model_price[i] = mortgage_ratios[0]
        duration[i] = mortgage_ratios[1]
        convexity[i] = mortgage_ratios[2]

    return coupons, oas_results, model_price, duration, convexity


def get_tba_prices_loan_age_and_zc(df, date):
    """Extraire les prix TBA, âges des prêts et taux zéro-coupon de manière optimisée."""
    # Pré-compilation des patterns pour les colonnes
    price_pattern = 'FNCL' and 'N Price'
    loan_age_pattern = 'FNCL' and 'Loan Age'
    
    # Créer des masques pour les colonnes une seule fois
    price_mask = df.columns.str.contains('FNCL') & df.columns.str.contains('N Price')
    loan_age_mask = df.columns.str.contains('FNCL') & df.columns.str.contains('Loan Age')
    zc_mask = df.columns.str.contains('zc_')
    
    # Obtenir les indices des colonnes plutôt que de faire des sélections multiples
    price_cols = df.columns[price_mask].values
    loan_age_cols = df.columns[loan_age_mask].values
    zc_cols = df.columns[zc_mask].values[:360]  # Limite directe à 360
    
    # Utiliser loc pour une seule sélection des données
    daily_data = df.loc[df['Dates'] == date, :]
    
    # Extraire et traiter les données en une seule opération
    tba_prices = daily_data[price_cols].values.flatten().astype(np.float32) / 100
    loan_ages = np.round(np.nan_to_num(daily_data[loan_age_cols].values.flatten(), 0), 3).astype(np.float32)
    zc_vector = np.round(daily_data[zc_cols].values.flatten().astype(np.float32), 6)
    
    return tba_prices, loan_ages, zc_vector

def build_date_cache(df_gross_data):
    cache = {}
    unique_dates = pd.to_datetime(df_gross_data['Dates']).unique()
    
    for date in unique_dates:
        date_str = date.strftime("%Y-%m-%d")
        try:
            tba_data = get_tba_prices_loan_age_and_zc(df_gross_data, date_str)
            mortgage_rate = df_gross_data[df_gross_data['Dates'] == date_str]['Mortgage Rate'].iloc[0] / 100
            volatility = df_gross_data[df_gross_data['Dates'] == date_str]['Volatility'].iloc[0] / 100
         
            cache[date_str] = {
                'tba_data': tba_data,
                'mortgage_rate': mortgage_rate,
                'volatility': volatility
            }
        except Exception as e:
            print(f"Error caching date {date_str}: {str(e)}")
            continue
    return cache

def calculate_oas_pivoted_table(df_gross_data, output_csv_path, params=None):
    """Calcule les OAS pour chaque date et coupon."""
    print("Début du calcul OAS...")
    
    # Default parameters if none provided
    if params is None:
        params = {
            'bmin': 0.10,
            'bmax': 0.10,
            'spreadmin': 0.30,
            'spreadmax': 0.30,
            'alpha': 150,
            'gamma': 0.05,
            'vol_multiplier': 1.4
        }
    
    try:
        # Conversion des dates
        df_gross_data['Dates'] = pd.to_datetime(df_gross_data['Dates'])
        df_gross_data = df_gross_data.dropna(subset=['Dates'])

        # Préparation
         # Utilisation du cache
        date_cache = build_date_cache(df_gross_data)
        coupons = [f"{i/10:.1f}" for i in range(20, 75, 5)]  # [2.0, 2.5, ..., 7.0]
        results = []
        unique_dates = df_gross_data['Dates'].sort_values().unique()
        
        for idx, date in enumerate(unique_dates):
            if idx % 50 == 0:
                z=0
                #print(f"Traitement de la date {idx+1}/{len(unique_dates)}: {date}")
                
            date_str = date.strftime("%Y-%m-%d")
            try:
                # Extraction des données
                cached_data = date_cache[date_str]
                tba_prices_loan_age_and_zc = get_tba_prices_loan_age_and_zc(df_gross_data, date_str)
                tba_price_mkt = tba_prices_loan_age_and_zc[0]
                loan_age_value = tba_prices_loan_age_and_zc[1]
                zc_vector_mkt = tba_prices_loan_age_and_zc[2]
                
                mortgage_rate = df_gross_data[df_gross_data['Dates'] == date_str]['Mortgage Rate'].iloc[0] / 100
                volatility = df_gross_data[df_gross_data['Dates'] == date_str]['Volatility'].iloc[0] / 100
                
                # Calcul OAS avec les paramètres fournis
                [coupons, oas_results, model_price, duration, convexity] = get_implied_oas_price_ratios_from_model(
                    tba_price_mkt=tba_price_mkt,
                    loan_age_value=loan_age_value,
                    principal=10000,
                    periods=360,
                    zc_vector_mkt=zc_vector_mkt,
                    mortage_rate=mortgage_rate,
                    bmin=params['bmin'],
                    bmax=params['bmax'],
                    spreadmin=params['spreadmin'],
                    spreadmax=params['spreadmax'],
                    alpha=params['alpha'],
                    gamma=params['gamma'],
                    volatility=volatility * params['vol_multiplier']
                )
                
                duration = -duration
                
                # Construction du dictionnaire résultat
                result_dict = {'Dates': date_str}
                
                # Ajout des TBA Prices
                for i, coupon in enumerate(coupons):
                    if not np.isnan(tba_price_mkt[i]):
                        result_dict[f"TBA_Price_{coupon:.3f}"] = tba_price_mkt[i] * 100
                
                # Ajout des OAS
                for i, coupon in enumerate(coupons):
                    if not np.isnan(oas_results[i]):
                        result_dict[f"OAS_{coupon:.3f}"] = oas_results[i] * 10000
                
                # Ajout des Durations
                for i, coupon in enumerate(coupons):
                    if not np.isnan(duration[i]):
                        result_dict[f"Duration_{coupon:.3f}"] = duration[i]
                
                results.append(result_dict)
                
            except Exception as e:
                print(f"Erreur lors du traitement de la date {date_str}: {str(e)}")
                continue
        
        # Création du DataFrame
        results_df = pd.DataFrame(results)
        
        # Réorganisation des colonnes
        cols = ['Dates']
        
        # TBA Price columns
        tba_cols = [col for col in results_df.columns if 'TBA_Price' in col]
        tba_cols.sort(key=lambda x: float(x.split('_')[2]))
        cols.extend(tba_cols)
        
        # OAS columns
        oas_cols = [col for col in results_df.columns if 'OAS_' in col]
        oas_cols.sort(key=lambda x: float(x.split('_')[1]))
        cols.extend(oas_cols)
        
        # Duration columns
        duration_cols = [col for col in results_df.columns if 'Duration_' in col]
        duration_cols.sort(key=lambda x: float(x.split('_')[1]))
        cols.extend(duration_cols)
        
        # Réorganiser le DataFrame
        results_df = results_df[cols]
        
        # Sauvegarder
        if output_csv_path:
            results_df.to_csv(output_csv_path, index=False)
            print(f"Tableau pivoté enregistré dans {output_csv_path}")
        
        return results_df

        
    except Exception as e:
        print("Erreur dans calculate_oas_pivoted_table:")
        print(str(e))
        print(traceback.format_exc())
        raise




def get_daily_pnl(df_strat, day, scaling):
    """
    Calcule le P&L quotidien avec alignement explicite des colonnes.
    """
    # 1. Identifier explicitement les colonnes de prix TBA disponibles
    tba_price_cols = [col for col in df_strat.columns if 'TBA_Price' in col]
    coupon_values = [col.split('_')[-1] for col in tba_price_cols]
    
    # 2. Créer les noms de colonnes pour les poids correspondants
    weight_cols = [f'weight_Duration_{coupon}' for coupon in coupon_values]
    
    # 3. Extraire les prix
    tba_prices_current_day = df_strat.iloc[day][tba_price_cols].astype(float).fillna(0)
    tba_prices_previous_day = df_strat.iloc[day-1][tba_price_cols].astype(float).fillna(0)
    
    # 4. Obtenir les poids avec les mêmes coupons que les prix
    weights = get_average_tba_weight_and_duration_strat(df_strat, day-1, scaling)
    weights = weights[weight_cols]  # Sélectionner uniquement les poids correspondant aux prix
    
    # 5. Vérifier l'alignement
    assert len(tba_prices_current_day) == len(weights), \
        f"Mismatch: prices={len(tba_prices_current_day)}, weights={len(weights)}"
    
    # 6. Gestion des prix manquants
    for col in tba_price_cols:
        if tba_prices_current_day[col] == 0 and tba_prices_previous_day[col] != 0:
            tba_prices_current_day[col] = tba_prices_previous_day[col]
    
    # 7. Calculer les poids courants pour le coût bid-offer
    weights_current = get_average_tba_weight_and_duration_strat(df_strat, day, scaling)
    weights_current = weights_current[weight_cols]
    
    # 8. Calculer le coût bid-offer
    bid_offer_cost = np.abs(weights_current - weights).sum() * (1/32) 
    
    # 9. Calculer le P&L
    daily_pnl = np.dot(
        (tba_prices_current_day - tba_prices_previous_day).values,
        weights.values
    ) - bid_offer_cost
    
    return daily_pnl



def debug_df(df, name="DataFrame"):
    """Fonction utilitaire pour déboguer un DataFrame"""
    print(f"\nDébogage de {name}:")
    print(f"Shape: {df.shape}")
    print("\nPremières lignes:")
    print(df.head())
    print("\nTypes des colonnes:")
    print(df.dtypes)
    print("\nValeurs manquantes:")
    print(df.isnull().sum())

def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """Charge et prépare les données initiales."""
    print(f"\nChargement des données depuis {file_path}...")
    
    # Chargement des données
    df = pd.read_csv(file_path)
    required_columns = ['Dates', 'Mortgage Rate', 'Volatility']
    
    # Vérification des colonnes obligatoires
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Colonnes manquantes: {', '.join(missing_cols)}")
    
    # Conversion des dates avec gestion des erreurs
    df['Dates'] = pd.to_datetime(df['Dates'], errors='coerce')
    
    # Vérification des dates invalides
    invalid_dates = df['Dates'].isna().sum()
    if invalid_dates > 0:
        print(f"Attention : {invalid_dates} dates invalides détectées et supprimées.")
        df = df.dropna(subset=['Dates'])  # Suppression des lignes avec des dates invalides
    
    # Affichage de la période couverte
    print(f"Période couverte: {df['Dates'].min()} à {df['Dates'].max()}")
    return df

def process_day(df_strat, i, scaling, coupon_list):
    """Effectue le calcul des poids et du P&L pour un jour donné."""
    weights = get_average_tba_weight_and_duration_strat(df_strat, i, scaling)
    daily_pnl = float(get_daily_pnl(df_strat, i, scaling))
    weight_row = {f'weight_{coupon}': weights.get(f'weight_Duration_{coupon}', 0) for coupon in coupon_list}
    return i, daily_pnl, weight_row

def process_day_wrapper(args):
    """Wrapper pour appeler process_day avec plusieurs arguments."""
    df_strat, i, scaling, coupon_list = args
    return process_day(df_strat, i, scaling, coupon_list)

def calculate_daily_weights_and_pnl(df_strat: pd.DataFrame, scaling: int = 10) -> pd.DataFrame:
    """Calcule les poids et P&L quotidiens en parallèle."""
    result_df = df_strat.copy()
    n_days = len(df_strat)
    
    # Initialisation des colonnes
    coupon_list = ['0.020', '0.025', '0.030', '0.035', '0.040', '0.045', 
                   '0.050', '0.055', '0.060', '0.065', '0.070']
    for coupon in coupon_list:
        result_df[f'weight_{coupon}'] = 0.0

    # Préparez les arguments pour chaque jour
    args_list = [(df_strat, i, scaling, coupon_list) for i in range(11, n_days)]

    # Parallélisation des calculs
    with ProcessPoolExecutor(max_workers=12) as executor:
        results = list(executor.map(process_day_wrapper, args_list))

    # Mise à jour des résultats
    for i, daily_pnl, weight_row in results:
        result_df.at[i, 'Daily_PnL'] = daily_pnl
        for col, val in weight_row.items():
            result_df.at[i, col] = val

    # Calcul du P&L cumulatif
    result_df['Cumulative_PnL'] = result_df['Daily_PnL'].cumsum()
    
    return result_df
    
def analyze_performance_complete(p_and_l: np.ndarray, dates: np.ndarray) -> Tuple[Dict, Dict]:
    df = pd.DataFrame({
        'P&L': p_and_l,
        'Date': pd.to_datetime(dates)
    })
    df.set_index('Date', inplace=True)
    df['Year'] = df.index.year
    df['Cumulative P&L'] = df['P&L'].cumsum()
    
    def calculate_metrics(data: pd.DataFrame) -> Dict:
        pnl_data = data['P&L']
        cum_pnl = pnl_data.cumsum()
        drawdown = cum_pnl - cum_pnl.cummax()
        
        gains = pnl_data[pnl_data > 0]
        losses = pnl_data[pnl_data < 0]
        downside_risk = losses.std() if not losses.empty else 0
        
        metrics = {
            "Total Return": cum_pnl.iloc[-1],
            "Annualized Return": pnl_data.mean() * 252,
            "Annualized Volatility": pnl_data.std() * np.sqrt(252),
            "Sharpe Ratio": (pnl_data.mean() / pnl_data.std() * np.sqrt(252)) if pnl_data.std() != 0 else 0,
            "Sortino Ratio": (pnl_data.mean() / downside_risk * np.sqrt(252)) if downside_risk != 0 else 0,
            "Maximum Drawdown": drawdown.min(),
            "Hit Ratio": len(gains) / len(pnl_data),
            "Average Daily P&L": pnl_data.mean(),
            "Max Daily Gain": pnl_data.max(),
            "Max Daily Loss": pnl_data.min(),
            "Trading Days": len(pnl_data),
            "Gain/Loss Ratio": (gains.mean() / abs(losses.mean())) if not losses.empty and losses.mean() != 0 else np.inf,
            "Skewness": stats.skew(pnl_data),
            "Kurtosis": stats.kurtosis(pnl_data),
            "Calmar Ratio": (pnl_data.mean() * 252 / abs(drawdown.min())) if drawdown.min() != 0 else np.inf
        }
        return metrics

    # Calculate metrics
    global_metrics = calculate_metrics(df)
    yearly_metrics = {year: calculate_metrics(group) for year, group in df.groupby('Year')}
    
    # Format display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', lambda x: f"{x:.4f}")
    
    # Print global metrics
    print("\n=== GLOBAL METRICS ===")
    print("-" * 100)
    for metric, value in global_metrics.items():
        print(f"{metric:25}: {value:,.4f}")
    
    # Print yearly metrics as a styled table
    print("\n=== YEARLY METRICS ===")
    print("-" * 100)
    metrics_df = pd.DataFrame(yearly_metrics).T
    
    print("\nDetailed metrics by year:")
    metrics_columns = ["Total Return", "Annualized Return", "Annualized Volatility", 
                      "Sharpe Ratio", "Sortino Ratio", "Maximum Drawdown", "Calmar Ratio", 
                      "Hit Ratio", "Average Daily P&L", "Max Daily Gain", "Max Daily Loss",
                      "Gain/Loss Ratio", "Trading Days", "Skewness", "Kurtosis"]
    
    metrics_df = metrics_df[metrics_columns]
    print(metrics_df.to_string(float_format=lambda x: '{:,.4f}'.format(x)))
    
    # Print average metrics
    print("\nMetric averages:")
    for col in metrics_df.columns:
        print(f"- {col:25}: {metrics_df[col].mean():,.4f}")
    
    return global_metrics, yearly_metrics

# Example usage:
#global_metrics, yearly_metrics = analyze_performance_complete(p_and_l, dates)


def calculate_metrics_for_params(df_gross_data, params):
    bmin, bmax, spreadmin, spreadmax, alpha, gamma, vol_multiplier = params
    
    try:
        unique_dates = pd.to_datetime(df_gross_data['Dates']).unique()
        results = []
        
        for date in unique_dates:
            date_str = date.strftime("%Y-%m-%d")
            data = get_tba_prices_loan_age_and_zc(df_gross_data, date_str)
            
            mortgage_rate = df_gross_data[df_gross_data['Dates'] == date_str]['Mortgage Rate'].iloc[0] / 100
            volatility = df_gross_data[df_gross_data['Dates'] == date_str]['Volatility'].iloc[0] / 100
            
            [coupons, oas_results, model_price, duration, convexity] = get_implied_oas_price_ratios_from_model(
                tba_price_mkt=data[0],
                loan_age_value=data[1],
                principal=10000,
                periods=360,
                zc_vector_mkt=data[2],
                mortage_rate=mortgage_rate,
                bmin=bmin,
                bmax=bmax,
                spreadmin=spreadmin,
                spreadmax=spreadmax,
                alpha=alpha,
                gamma=gamma,
                volatility=volatility*vol_multiplier
            )
            
            result_dict = {'Dates': date_str}
            for i, coupon in enumerate(coupons):
                if not np.isnan(data[0][i]):
                    result_dict[f"TBA_Price_{coupon:.3f}"] = data[0][i] * 100
                if not np.isnan(oas_results[i]):
                    result_dict[f"OAS_{coupon:.3f}"] = oas_results[i] * 10000
                if not np.isnan(duration[i]):
                    result_dict[f"Duration_{coupon:.3f}"] = -duration[i]
            
            results.append(result_dict)
        
        df_results = pd.DataFrame(results)
        df_results['Dates'] = pd.to_datetime(df_results['Dates'])
        df_results_with_pnl = calculate_daily_weights_and_pnl(df_results, scaling=10)
        
        daily_pnl = df_results_with_pnl['Daily_PnL'].dropna()
        annualized_return = daily_pnl.mean() * 252
        annualized_vol = daily_pnl.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else 0
        
        return {
            'bmin': bmin,
            'bmax': bmax,
            'spreadmin': spreadmin,
            'spreadmax': spreadmax,
            'alpha': alpha,
            'gamma': gamma,
            'vol_multiplier': vol_multiplier,
            'sharpe_ratio': sharpe_ratio,
            'annualized_return': annualized_return,
            'annualized_vol': annualized_vol
        }
    
    except Exception as e:
        print(f"Error with parameters {params}: {str(e)}")
        return None

def process_param_combination(params, date_data_cache):
    """Traite une combinaison de paramètres et retourne les résultats."""
    bmin, bmax, spreadmin, spreadmax, alpha, gamma, vol_multiplier = params
    results_list = []
    
    try:
        # Process each date
        for date_str, cached_data in date_data_cache.items():
            [coupons, oas_results, model_price, duration, convexity] = get_implied_oas_price_ratios_from_model(
                tba_price_mkt=cached_data['tba_data'][0],
                loan_age_value=cached_data['tba_data'][1],
                principal=10000,
                periods=360,
                zc_vector_mkt=cached_data['tba_data'][2],
                mortage_rate=cached_data['mortgage_rate'],
                bmin=bmin, bmax=bmax,
                spreadmin=spreadmin, spreadmax=spreadmax,
                alpha=alpha, gamma=gamma,
                volatility=cached_data['volatility'] * vol_multiplier
            )
            
            result_dict = {'Dates': date_str}
            for j, coupon in enumerate(coupons):
                if not np.isnan(cached_data['tba_data'][0][j]):
                    result_dict[f"TBA_Price_{coupon:.3f}"] = cached_data['tba_data'][0][j] * 100
                if not np.isnan(oas_results[j]):
                    result_dict[f"OAS_{coupon:.3f}"] = oas_results[j] * 10000
                if not np.isnan(duration[j]):
                    result_dict[f"Duration_{coupon:.3f}"] = -duration[j]
            
            results_list.append(result_dict)
        
        # Calculate metrics
        if results_list:
            df_results = pd.DataFrame(results_list)
            df_results['Dates'] = pd.to_datetime(df_results['Dates'])
            df_results_with_pnl = calculate_daily_weights_and_pnl(df_results, scaling=10)
            
            daily_pnl = df_results_with_pnl['Daily_PnL'].dropna()
            annualized_return = daily_pnl.mean() * 252
            annualized_vol = daily_pnl.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else 0
            
            return {
                'bmin': bmin, 'bmax': bmax,
                'spreadmin': spreadmin, 'spreadmax': spreadmax,
                'alpha': alpha, 'gamma': gamma,
                'vol_multiplier': vol_multiplier,
                'sharpe_ratio': sharpe_ratio,
                'annualized_return': annualized_return,
                'annualized_vol': annualized_vol
            }
    except Exception as e:
        print(f"Error with params {params}: {str(e)}")
        return None
    

def cache_date_data(date, df_gross_data):
    """Cache les données pour une date donnée."""
    if pd.isna(date):
        print(f"Date invalide détectée : {date}")
        return None, None
    
    date_str = date.strftime("%Y-%m-%d")
    try:
        tba_data = get_tba_prices_loan_age_and_zc(df_gross_data, date_str)
        mortgage_rate = df_gross_data[df_gross_data['Dates'] == date_str]['Mortgage Rate'].iloc[0] / 100
        volatility = df_gross_data[df_gross_data['Dates'] == date_str]['Volatility'].iloc[0] / 100
        return date_str, {
            'tba_data': tba_data,
            'mortgage_rate': mortgage_rate,
            'volatility': volatility
        }
    except Exception as e:
        print(f"Error caching date {date_str}: {str(e)}")
        return date_str, None

# Déplacez cette fonction en dehors de optimize_parameters_efficient
def process_param_combination_with_progress_wrapper(args):
    """Wrapper pour process_param_combination_with_progress."""
    return process_param_combination_with_progress(*args)

def process_param_combination_with_progress(params, idx, total_combinations, start_time, date_data_cache):
    """Traite une combinaison de paramètres avec affichage de l'état d'avancement."""
    bmin, bmax, spreadmin, spreadmax, alpha, gamma, vol_multiplier = params
    elapsed = time.time() - start_time
    remaining = (elapsed / (idx + 1)) * (total_combinations - (idx + 1))
    
    print(f"\nTesting {idx + 1}/{total_combinations}")
    print(f"Elapsed: {elapsed / 60:.1f}m, Remaining: {remaining / 60:.1f}m")
    
    try:
        results_list = []
        for date_str, cached_data in date_data_cache.items():
            if cached_data is None or 'tba_data' not in cached_data:
                print(f"Skipping invalid cached data for date: {date_str}")
                continue
            
            try:
                [coupons, oas_results, model_price, duration, convexity] = get_implied_oas_price_ratios_from_model(
                    tba_price_mkt=cached_data['tba_data'][0],
                    loan_age_value=cached_data['tba_data'][1],
                    principal=10000,
                    periods=360,
                    zc_vector_mkt=cached_data['tba_data'][2],
                    mortage_rate=cached_data['mortgage_rate'],
                    bmin=bmin, bmax=bmax,
                    spreadmin=spreadmin, spreadmax=spreadmax,
                    alpha=alpha, gamma=gamma,
                    volatility=cached_data['volatility'] * vol_multiplier
                )
            except Exception as model_error:
                print(f"Error processing date {date_str} with parameters {params}: {model_error}")
                continue
            
            result_dict = {'Dates': date_str}
            for j, coupon in enumerate(coupons):
                if not np.isnan(cached_data['tba_data'][0][j]):
                    result_dict[f"TBA_Price_{coupon:.3f}"] = cached_data['tba_data'][0][j] * 100
                if not np.isnan(oas_results[j]):
                    result_dict[f"OAS_{coupon:.3f}"] = oas_results[j] * 10000
                if not np.isnan(duration[j]):
                    result_dict[f"Duration_{coupon:.3f}"] = -duration[j]
            results_list.append(result_dict)
        
        if results_list:
            df_results = pd.DataFrame(results_list)
            df_results['Dates'] = pd.to_datetime(df_results['Dates'])
            df_results_with_pnl = calculate_daily_weights_and_pnl(df_results, scaling=10)
            daily_pnl = df_results_with_pnl['Daily_PnL'].dropna()
            annualized_return = daily_pnl.mean() * 252
            annualized_vol = daily_pnl.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else 0
            return {
                'bmin': bmin, 'bmax': bmax,
                'spreadmin': spreadmin, 'spreadmax': spreadmax,
                'alpha': alpha, 'gamma': gamma,
                'vol_multiplier': vol_multiplier,
                'sharpe_ratio': sharpe_ratio,
                'annualized_return': annualized_return,
                'annualized_vol': annualized_vol
            }
    except Exception as e:
        print(f"Error with params {params}: {str(e)}")
        return None
    
def optimize_parameters_efficient(df_gross_data, sample_size=None):
    param_ranges = {
        'bmin': np.array([0.05,0.07]),
        'bmax': np.array([0.05,0.07]),
        'spreadmin': np.array([0.50,0.45,0.40]),
        'spreadmax': np.array([0.30,0.25,0.35]),
        'alpha': np.array([100]),
        'gamma': np.array([0.06]),
        'vol_multiplier': np.array([1,1.25])
    }
    param_combinations = list(product(*param_ranges.values()))
    if sample_size:
        param_combinations = param_combinations[:sample_size]
    total_combinations = len(param_combinations)
    start_time = time.time()
    print(f"\nTesting {total_combinations} parameter combinations")
    
    unique_dates = pd.to_datetime(df_gross_data['Dates']).unique()
    with ProcessPoolExecutor(max_workers=12) as executor:
        cached_data = list(executor.map(partial(cache_date_data, df_gross_data=df_gross_data), unique_dates))
    date_data_cache = {k: v for k, v in cached_data if v is not None}
    
    # Préparer les arguments pour process_param_combination_with_progress_wrapper
    param_args = [
        (params, idx, total_combinations, start_time, date_data_cache)
        for idx, params in enumerate(param_combinations)
    ]
    
    # Exécution parallèle avec ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=12) as executor:
        results = list(executor.map(process_param_combination_with_progress_wrapper, param_args))

    results = [res for res in results if res is not None]
    if not results:
        return None
    return pd.DataFrame(results).sort_values('sharpe_ratio', ascending=False)

def main(folder = 'DATA_SAMY', params=None):
    start_time = time.time()
    
    try:
        # Configure paths
        data_dir = Path(f"Data/{folder}")
        input_file = data_dir / "mbs_global_data_cleaned_final_2010_2024.csv"
        oas_file = data_dir / "oas_result_global_2010_2024.csv"
        output_file_param = data_dir / "maximise_param_2010_2024_new.csv"
        output_file = data_dir / "oas_result_global_2010_2024_with_pnl.csv"
        
        print("\n=== Phase 1: Loading and Processing Data ===")
        df_gross_data = load_and_prepare_data(input_file)
        df_results = calculate_oas_pivoted_table(df_gross_data, oas_file, params=params)
        print("OAS calculation completed")
        
        print("\n=== Phase 2: Initial PnL Calculation ===")
        df_strat = pd.read_csv(oas_file)
        df_strat['Dates'] = pd.to_datetime(df_strat['Dates'])
        df_results_with_pnl = calculate_daily_weights_and_pnl(df_strat)
        df_results_with_pnl.to_csv(output_file, index=False)
        
        print("\n=== Phase 3: Performance Analysis ===")
        dates = df_results_with_pnl['Dates'].values
        p_and_l = df_results_with_pnl['Daily_PnL'].values
        global_metrics, yearly_metrics = analyze_performance_complete(p_and_l, dates)
        
        '''print("\n=== Phase 4: Parameter Optimization ===")
        print("Testing initial parameters...")
        test_params = [0.03, 0.03, 0.37, 0.37, 125, 0.05, 1.0]
        test_result = calculate_metrics_for_params(df_gross_data, test_params)
        if test_result:
            print("Initial test successful, proceeding with optimization...")
            
            try:
                print("\nStarting full grid search...")
                results = optimize_parameters_efficient(df_gross_data, sample_size=None)
                
                if results is not None:
                    print("\nOptimization completed successfully")
                    print("\nTop 10 parameter combinations:")
                    print(results.head(10))
                    results.to_csv(output_file_param, index=False)
                    print(f"Results saved to: {output_file_param}")
                else:
                    print("Optimization failed to produce valid results")
            except Exception as opt_error:
                print(f"Error during optimization: {str(opt_error)}")
                traceback.print_exc()
        else:
            print("Initial parameter test failed, stopping optimization")'''
            
    except Exception as e:
        print(f"\nCritical error in main process:")
        print(str(e))
        traceback.print_exc()
        
    finally:
        duration = time.time() - start_time
        print(f"\nTotal execution time: {duration:.2f} seconds ({duration/60:.2f} minutes)")

if __name__ == "__main__":
    main()