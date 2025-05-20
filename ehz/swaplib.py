from math import isnan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
from scipy.interpolate import CubicSpline
from tqdm import tqdm
import utils

def add_bd(date, n):
    """
    Adds n business days to a given date.
    
    Parameters:
    - date (str or pd.Timestamp): The initial date.
    - n (int): Number of business days to add.
    
    Returns:
    - pd.Timestamp: The adjusted date.
    """
    return pd.Timestamp(date) + pd.offsets.BDay(n)

def to_bd(date):
    """
    gives date if date is a business days otherwise gives next business day.
    
    Parameters:
    - date (str or pd.Timestamp): The initial date.
    
    Returns:
    - pd.Timestamp: The adjusted date.
    """
    return pd.Timestamp(date) - pd.offsets.BDay() + pd.offsets.BDay()


def compute_payment_schedule_fixed(start_date, maturity_date, payment_lag):
    """Compute start dates, end dates, and payment dates for all cash flows."""
    # Ensure dates are converted to pandas timestamps
    start_date = pd.Timestamp(start_date)
    maturity_date = pd.Timestamp(maturity_date)

    # If maturity_date is within 1 year of start_date, handle it as a single period
    if maturity_date <= start_date + pd.DateOffset(years=1):
        end_dates = pd.Index([maturity_date])
    else:
        # Generate yearly end dates starting from start_date + 1 year
        end_dates = pd.date_range(start=start_date + pd.DateOffset(years=1), end=maturity_date, freq=pd.DateOffset(years=1))

        # Ensure we include the maturity_date explicitly
        if len(end_dates) == 0 or end_dates[-1] != maturity_date:
            end_dates = end_dates.append(pd.Index([maturity_date]))


    # Compute start dates: First date is start_date, then shift from end_dates
    start_dates = pd.Index([start_date]).append(end_dates[:-1])

    # Compute payment dates: End dates + payment lag (adjusted for business days)
    payment_dates = end_dates.map(lambda date: add_bd(date, payment_lag))

    return pd.DataFrame({"Start Date": start_dates, "End Date": end_dates, "Payment Date": payment_dates})


def interpolate_zc(date, zc) :

    dt = (zc.index - date).days/360
    known_rates = -np.log(zc)/dt
    interpolator = CubicSpline(dt, known_rates, bc_type='natural')

    dates_target = pd.date_range(date, date+pd.DateOffset(years=40))
    dt_target = (dates_target - date).days/360

    rates_interpolated = interpolator(dt_target)
    rates_interpolated = np.clip(rates_interpolated, -1, 1)
    log_zc = -rates_interpolated*dt_target
    zc_interpolated = pd.Series(index = dates_target, data = np.exp(log_zc))
    return zc_interpolated


def swap_rates_to_zc(date, swap_rates_date):
    date = pd.Timestamp(date)
    r_t = swap_rates_date.loc["1D"] / 100
    
    zc_prices = {}
    T0 = add_bd(date, Swap.SLAG)
    zc_prices[T0] = np.exp(-r_t * (T0 - date).days / 360)
    
    for tenor, swap_rate in swap_rates_date.loc['1M':'1Y'].items():
        if np.isnan(swap_rate):
            continue
        k_star = swap_rate / 100
        months = int(tenor[:-1]) if tenor.endswith("M") else int(tenor[:-1]) * 12
        T1 = to_bd(T0 + pd.DateOffset(months=months))

        delta = (T1 - T0).days / 360
        P_T1 = zc_prices[T0] / (1 + k_star * delta)
        zc_prices[T1] = P_T1
    
    zc_prices = pd.Series(zc_prices).sort_index()



    def objective(zc, swap, T, P_T):
        zc_temp = zc.copy()
        zc_temp.loc[T] = P_T
        zc_interpolated = interpolate_zc(date, zc_temp)
        return swap.price(date, zc_interpolated)[0]
    
    for tenor, swap_rate in swap_rates_date.loc['18M':].items():
        if np.isnan(swap_rate):
            continue
        swap = Swap.standard_swap(date, tenor, swap_rate)
        months = int(tenor[:-1]) if tenor.endswith("M") else int(tenor[:-1]) * 12
        T = to_bd(T0 + pd.DateOffset(months=months))

        
        P_T_guess = zc_prices.iloc[-1]
        P_T_solution = newton(lambda P_T: objective(zc_prices, swap, T, P_T), P_T_guess)
        zc_prices[T] = P_T_solution
    
    return zc_prices.sort_index()


# Final implementation with standard_swap and show_dates

class Swap:
    SLAG = 2
    PLAG = 2
    RLAG = 2
    PFREQ = 1
    def __init__(self, notional, fixed_rate, trade_date, maturity_date, settlement_lag=SLAG, payment_lag=PLAG, reset_lag=RLAG, payment_freq=PFREQ):
        """
        Initializes a SOFR Swap with annual payments for both legs.

        Parameters:
        - notional (float): Notional amount of the swap.
        - fixed_rate (float): Fixed rate for the fixed leg (as a pctge, e.g., 3 for 3%).
        - trade_date (str or pd.Timestamp): Trade date of the swap (format: 'YYYY-MM-DD' or Timestamp).
        - maturity_date (str or pd.Timestamp): Maturity date of the swap (format: 'YYYY-MM-DD' or Timestamp).
        - settlement_lag (int, optional): Settlement lag in business days (default: 2).
        - payment_lag (int, optional): Payment lag in business days (default: 2).
        - reset_lag (int, optional): Reset lag in business days (default: 2).
        """
        self.notional = notional
        self.fixed_rate = fixed_rate
        self.trade_date = pd.Timestamp(trade_date)
        self.settlement_lag = settlement_lag
        self.payment_lag = payment_lag
        self.reset_lag = reset_lag
        self.start_date = add_bd(self.trade_date, self.settlement_lag)  # Compute start date
        self.maturity_date = pd.Timestamp(maturity_date)
        self.payment_frequency =   payment_freq # Annual payments for both legs
        self.day_count_fixed = 360  # 30/360 convention
        self.day_count_floating = 360  # Actual/360 convention

        # Ensure that the maturity date is after the start date
        if self.maturity_date <= self.start_date:
            raise ValueError("Maturity date must be after the start date.")

        # Compute all payment dates
        self._compute_payment_schedule()

    def _compute_payment_schedule(self):
        """Compute start dates, end dates, and payment dates for all cash flows."""
        payment_schedule = compute_payment_schedule_fixed(self.start_date, self.maturity_date, self.payment_lag)

        self._start_dates = payment_schedule["Start Date"]
        self._end_dates = payment_schedule["End Date"]
        self._payment_dates = payment_schedule["Payment Date"]

    def show_dates(self):
        """Display the swap's start, end, and payment dates as a DataFrame."""
        dates_df = pd.DataFrame({
            "Start Date": self._start_dates,
            "End Date": self._end_dates,
            "Payment Date": self._payment_dates
        })
        return dates_df

    @classmethod
    def standard_swap(cls, trade_date, maturity, fixed_rate):
        """
        Creates a standard SOFR swap using standard conventions.

        Parameters:
        - trade_date (str or pd.Timestamp): Trade date (format: 'YYYY-MM-DD' or Timestamp).
        - maturity (str): Maturity in 'XM' (months) or 'XY' (years), e.g., '1M', '5Y'.
        - fixed_rate (float): Fixed rate for the swap.

        Returns:
        - Swap instance.
        """
        # Convert trade_date to pandas Timestamp
        trade_date = pd.Timestamp(trade_date)
        start_date = add_bd(trade_date, cls.SLAG)
        # Parse maturity (e.g., "5Y" → 5 years, "3M" → 3 months)
        if maturity.endswith("Y"):
            years = int(maturity[:-1])
            maturity_date = start_date + pd.DateOffset(years=years)
        elif maturity.endswith("M"):
            months = int(maturity[:-1])
            maturity_date = start_date + pd.DateOffset(months=months)
        else:
            raise ValueError("Invalid maturity format. Use 'XM' for months or 'XY' for years.")

        maturity_date = to_bd(maturity_date)
        # Define standard swap parameters
        notional = 1  # Standardized to 1 unit (e.g., percentage of a larger notional)

        # Return a new Swap instance
        return cls(notional, fixed_rate, trade_date, maturity_date)


    def price(self, valuation_date, zc_prices, sofr_rates=0):
        """
        Computes the price of the swap given a valuation date, zero-coupon prices, and SOFR rates.

        Parameters:
        - valuation_date (str or pd.Timestamp): The date for which we compute the swap price.
        - zc_prices (pd.Series): Zero-coupon bond prices indexed by dates.
        - sofr_rates (pd.Series): Historical SOFR rates indexed by dates (in % form, so divide by 100).

        Returns:
        - tuple: (swap price, fixed leg price, floating leg price, accrued floating interest, cash)
        """
        # Convert SOFR rates to decimal
        sofr_rates = sofr_rates / 100
        k = self.fixed_rate/100

        # Find the current period index (i) based on valuation_date
        i = np.searchsorted(self._start_dates, valuation_date, side="right") - 1

        # Ensure i is within valid bounds
        if i < 0:
            i = 0

        # Extract key dates
        T_i_1 = self._start_dates[i]  # T_{i-1}
        T_i = self._end_dates[i]  # T_i
        T_p_i = self._payment_dates[i]  # Payment date

        zc_payment_dates = zc_prices.loc[self._payment_dates.loc[i:]].values
        #zc_start_dates = zc_prices.loc[[valuation_date] + self._start_dates.loc[i+1:].to_list()]

        if T_i_1 < valuation_date : 
            zc_start_dates = zc_prices.loc[[valuation_date] + self._start_dates.loc[i+1:].to_list()].values
        else :
            zc_start_dates = zc_prices.loc[self._start_dates.loc[i:]].values

        zc_end_dates = zc_prices.loc[self._end_dates.loc[i:]]
        dt = ((self._end_dates - self._start_dates).loc[i:].apply(lambda x: x.days)/360).values

        fixed_leg_price = k * self.notional * ((zc_payment_dates)*dt).sum()
        floating_leg_price = self.notional*(zc_payment_dates*(zc_start_dates-zc_end_dates)/zc_end_dates).sum()

        # Compute accrued floating interest
        accrued_floating_leg = 0
        if T_i_1 <= valuation_date < T_i:
            sofr_dates = sofr_rates.loc[T_i_1:valuation_date].index
            deltas = sofr_dates.diff().dropna().days/360
            # Compute compounded SOFR accrued rate
            sofr_accrued = np.prod(1 + sofr_rates.loc[sofr_dates[:-1]].to_numpy() * deltas) - 1
            accrued_floating_leg = self.notional * sofr_accrued

        # If we're between end of accrual and payment, discount the cash flow
        T0 = self._start_dates.iloc[i-1]
        T1 = T_i_1
        TP = self._payment_dates.iloc[i-1]
        cash = 0
        if T1 <= valuation_date <= TP:

            cashflow_fixed_leg = k*self.notional*(T1-T0).days/360  

            sofr_dates = sofr_rates.loc[T0:T1].index
            deltas = sofr_dates.diff().dropna().days/360
            # Compute compounded SOFR accrued rate
            sofr_accrued = np.prod(1 + sofr_rates.loc[sofr_dates[:-1]].to_numpy() * deltas) - 1
            cashflow_floating_leg = self.notional * sofr_accrued

            if valuation_date < TP : 
                fixed_leg_price += cashflow_fixed_leg*zc_prices.loc[TP]
                floating_leg_price += cashflow_floating_leg*zc_prices.loc[TP]

            else : 
                cash = (cashflow_fixed_leg-cashflow_floating_leg)

        swap_price = fixed_leg_price - floating_leg_price - accrued_floating_leg

        return swap_price, cash, fixed_leg_price, floating_leg_price, accrued_floating_leg


    def break_even_rate(self, valuation_date, zc_prices, sofr_rates=0):
        """
        Computes the price of the swap given a valuation date, zero-coupon prices, and SOFR rates.

        Parameters:
        - valuation_date (str or pd.Timestamp): The date for which we compute the swap price.
        - zc_prices (pd.Series): Zero-coupon bond prices indexed by dates.
        - sofr_rates (pd.Series): Historical SOFR rates indexed by dates (in % form, so divide by 100).

        Returns:
        - tuple: (swap price, fixed leg price, floating leg price, accrued floating interest, cash)
        """
        # Convert SOFR rates to decimal
        sofr_rates = sofr_rates / 100

        # Find the current period index (i) based on valuation_date
        i = np.searchsorted(self._start_dates, valuation_date, side="right") - 1

        # Ensure i is within valid bounds
        if i < 0:
            i = 0


        # Extract key dates
        T_i_1 = self._start_dates[i]  # T_{i-1}
        T_i = self._end_dates[i]  # T_i
        T_p_i = self._payment_dates[i]  # Payment date

        zc_payment_dates = zc_prices.loc[self._payment_dates.loc[i:]]
        zc_start_dates = zc_prices.loc[[valuation_date] + self._start_dates.loc[i+1:].to_list()]
        zc_end_dates = zc_prices.loc[self._end_dates.loc[i:]]
        dt = ((self._end_dates - self._start_dates).loc[i:].apply(lambda x: x.days)/360).values

        fixed_leg_price_factor = self.notional * ((zc_payment_dates)*dt).sum()
        floating_leg_price = self.notional*(zc_payment_dates*(zc_start_dates-zc_end_dates)/zc_end_dates).sum()

        # Compute accrued floating interest
        accrued_floating_leg = 0
        if T_i_1 <= valuation_date < T_i:
            sofr_dates = sofr_rates.loc[T_i_1:valuation_date].index
            deltas = sofr_dates.diff().dropna().days/360
            # Compute compounded SOFR accrued rate
            sofr_accrued = np.prod(1 + sofr_rates.loc[sofr_dates[:-1]].to_numpy() * deltas) - 1
            accrued_floating_leg = self.notional * sofr_accrued


        # If we're between end of accrual and payment, discount the cash flow
        T0 = self._start_dates.iloc[i-1]
        T1 = T_i_1
        TP = self._payment_dates.iloc[i-1]
        cash = 0
        if T1 <= valuation_date <= TP:

            cashflow_fixed_leg_factor = self.notional*(T1-T0).days/360  

            sofr_dates = sofr_rates.loc[T0:T1].index
            deltas = sofr_dates.diff().dropna().days/360
            # Compute compounded SOFR accrued rate
            sofr_accrued = np.prod(1 + sofr_rates.loc[sofr_dates[:-1]].to_numpy() * deltas) - 1
            cashflow_floating_leg = self.notional * sofr_accrued

            if valuation_date < TP : 
                fixed_leg_price_factor += cashflow_fixed_leg_factor*zc_prices.loc[TP]
                floating_leg_price += cashflow_floating_leg*zc_prices.loc[TP]

        fixed_rate = (floating_leg_price+accrued_floating_leg)/fixed_leg_price_factor

        return fixed_rate*100, fixed_leg_price_factor
