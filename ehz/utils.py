# pylint: disable-all
import pandas as pd 
import numpy as np 
from scipy import optimize 
import os 
try :
    from xbbg import blp
except :
    pass

parent = os.path.dirname(__file__)



fut_dic = {
    '2Y'  : 'tu1',
    '3Y'  : '3y1',
    '5Y'  : 'fv1',
    '10Y' : 'ty1',
    '10Y Ultra': 'uxy1',
    '20Y': 'us1',
    '30Y': 'wn1'
}
    
tenors_dic = {
    '2Y': 2,
    '3Y': 3,
    '5Y': 5,
    '10Y':7,
    '10Y Ultra': 10,
    '20Y': 20,
    '30Y':30
}

freq_dic = {
    '2Y': 1,
    '3Y': 1,
    '5Y': 1,
    '10Y': 1,
    '10Y Ultra': 3,
    '20Y': 3,
    '30Y': 3
}


# index of fut_codes for when the contract table started being complete
start_dic = {
    '2Y': 'Mar 2010',
    '3Y': 'Mar 2024',
    '5Y': 'Mar 2010',
    '10Y': 'Mar 2010',
    '10Y Ultra': 'Mar 2016',
    '20Y': 'Mar 2010',
    '30Y': 'Mar 2010'
}


months = ['H', 'M', 'U', 'Z']
months_names = ['Mar', 'Jun', 'Sep', 'Dec']
end_year = pd.Timestamp.today().year +2
end_date = pd. Timestamp.today () + pd.DateOffset(months=4)
tickers = []
contracts = []
for y in np.arange(2004, end_year):
    for m, month in zip(months, months_names):
        if pd.to_datetime (f'{month} {y}')< end_date:
            tickers.append(f'{m}{str(y)[-2:]}')
            contracts.append (f'{month} {y}')


fut_codes = pd.Series(index = contracts, data=tickers, dtype=str)




def format_bond (bond_data):
    label = bond_data
    return f'{(label.coupon)} {label.maturity.day}/{label.maturity.month}/{label.maturity.year}'

def parse_price(line):
    line = line.split('-')
    p = float (line [0])
    if len(line) == 1:
        return p

    line = line[1]
    line = line.split('+')
    if len(line) == 1:
        p = p + float (line [0]) /32
    else :
        p = p+ float(line [0])/32 + 1/64
    return p


def bond_cusip_to_ot_description (x) :
    c = str(x.coupon)
    while len(c)<5:
        c = c + '0'
    m = x.maturity.strftime ('%y%m')
    return f'USTB{m}_{c}'


# Data readers

def get_bonds (bond_type= 'us'):
    if bond_type=='us':
        bonds = pd.read_csv (f'{parent}/data/us_bonds/bonds_us_1980.csv', index_col=0,
                                parse_dates= ['maturity', 'issue_dt','int_acc_dt'] )
        
    bonds ['length'] = (bonds ['maturity'] - bonds ['issue_dt']).apply(lambda x : x.days/360).round()
    return bonds.sort_values(by="issue_dt")


def get_bond_prices (bond_type='us', ptype="mid"):

    if bond_type=='us':
        prices = pd.read_csv(f'{parent}/data/us_bonds/bonds_{ptype}_price.csv',
        index_col=0, parse_dates=True)
    return prices

def get_bond_yields (bond_type = 'us', ptype='mid'):
    if bond_type == 'us':
        yields = pd.read_csv(f'{parent}/data/us_bonds/bonds_{ptype}_yield.csv',
        index_col=0, parse_dates=True)
    return yields


def get_bond_duration (bond_type='us', ptype='mid'):
    if bond_type =='us' :
        D = pd.read_csv (f'{parent}/data/us_bonds/durations_{ptype}.csv',
        index_col=0, parse_dates=True)
    return D

def get_bond_perfs (bond_type='us'):
    if bond_type== 'us':
        prices = pd.read_csv(f'{parent}/data/us_bonds/bonds_perf.csv',
        index_col=0, parse_dates=True)
    return prices


def get_fut_perfs (bond_type='us'):
    if bond_type == 'us':
        prices = pd.read_csv(f'{parent}/swap_pv/data/Futs_on_the_run_perf.csv',
        index_col=0, parse_dates=True)
    return prices

def get_fld_bonds (fld = 'PV_sofr'):
    tab = pd.read_csv(f'{parent}/data/us_bonds/bonds_{fld}.csv',
    index_col=0, parse_dates=True)
    return tab

def get_swap_rates (source='bloom'):
    rates = pd.read_csv(f'data/us_rates/{source}/rates.csv', index_col=0, parse_dates = True)
    return rates


def get_zc_rates_and_maturity (source='bloom'):
    zero_coupon_rates = pd.read_csv(f'data/us_rates/{source}/stripped_zc.csv', index_col=0, parse_dates= True).sort_index()
    zero_coupon_maturity = pd.read_csv(f'data/us_rates/{source}/zc_maturity.csv',index_col=0,parse_dates=True).sort_index()
    zero_coupon_maturity = zero_coupon_maturity.applymap(lambda x: pd.to_datetime(x, format = '%Y-%m-%d')) # type: ignore 
    return zero_coupon_rates, zero_coupon_maturity

def get_zc_rates(source='bloom'):
    zero_coupon_rates = pd.read_csv(f'data/us_rates/{source}/stripped_zc.csv', index_col=0, parse_dates= True).sort_index()
    zero_coupon_rates.columns = zero_coupon_rates.columns.map(int)
    return zero_coupon_rates

def get_asset_swap_indices():
    ASW = pd.read_csv('swap_pv/asw.csv', index_col=0, parse_dates=True)
    ASW. columns = [int(c) for c in ASW.columns]
    return ASW

def get_bonds_zspread() :
    Z = pd.read_csv ('data/us_bonds/bonds_z_spread_calc.csv', index_col= 0, parse_dates=True)
    return Z

def get_bonds_pyd() :
    pyd = pd.read_csv('data/us_bonds/estimated_payment_dates.csv', index_col=0, 
        parse_dates= ['payment_date'])
    return pyd.iloc[:,0]


calendar = get_bond_prices().index.to_series()

roll_dates = pd.read_csv ('data/us_bonds/roll_dates.csv', index_col=0, parse_dates=['0']).iloc[:,0]
roll_in_dates = roll_dates.shift(1).dropna()
roll_out_dates = roll_dates.loc[roll_in_dates.index]
futs_calendar = calendar.loc[roll_in_dates.iloc[0]:].apply(
    lambda x: roll_in_dates[roll_in_dates<=x].index[-1])


def update_calendar () :

    global calendar 
    global roll_dates 
    global roll_in_dates 
    global roll_out_dates 
    global futs_calendar

    calendar = get_bond_prices().index.to_series()
    calendar.loc[pd.to_datetime ('11/12/2024')] = pd.to_datetime ('11/12/2024')
    roll_dates = pd.read_csv ('data/us_bonds/roll_dates.csv', index_col=0, parse_dates=['0']).iloc[:,0]
    roll_in_dates = roll_dates.shift(1).dropna()
    roll_out_dates = roll_dates.loc[roll_in_dates.index]
    futs_calendar = calendar.loc[roll_in_dates.iloc[0]:].apply(
        lambda x: roll_in_dates[roll_in_dates<=x].index[-1])



def add_business_day(date, n=1):
    """
    Shift a given date by n business days within a sorted calendar, handling missing holidays.
    
    Parameters:
    - calendar (np.array of datetime64[D]): Sorted array of business days (holidays skipped)
    - date (datetime64[D]): The input date
    - n (int): Number of business days to add/subtract

    Returns:
    - datetime64[D]: Adjusted business day
    
    """
    
    if date < calendar.iloc[0] or date > calendar.iloc[-1]   :
        return date + pd.offsets.BDay(n)        

    if n<0 and date < calendar.iloc[-n] :
        return date + pd.offsets.BDay(n)
    
    if n>0 and date > calendar.iloc[-n-1] :
        return date + pd.offsets.BDay(n)
    
    
    idx = np.searchsorted(calendar, date)

    # If date is in the calendar, apply direct shift
    if idx < len(calendar) and calendar.iloc[idx] == date:
        if idx+n >= len(calendar) or idx+n < 0 :
            return date + pd.offsets.BDay(n)
        return calendar.iloc[idx + n]

    # If date is a holiday, find surrounding business days
    prev_idx = max(idx - 1, 0)  # d1 (previous business day)
    next_idx = min(idx, len(calendar) - 1)  # d2 (next business day)

    # Recursively apply shift from closest valid business day
    if n > 0:
        return add_business_day(calendar.iloc[next_idx], n - 1)
    else:
        return add_business_day(calendar.iloc[prev_idx], n + 1)

def next_business_day (date):
    bday = pd.offsets.BDay()
    if (date > calendar.index[-1] or  date < calendar. index[0]):
        return date - bday + bday
    if date in calendar.index:
        return date 
    else :
        return calendar.loc[date:].iloc[0]



coupon_dict = {}
def get_coupon (line: str):
    """
    get a float coupon value from lines like: "T 3 % 03/31/30"
    """
    
    if len (line) <14:
        return float (line [2])
    
    coupon = float(line[2]) + coupon_dict[line[4]]
    return coupon


def get_maturity(line: str):
    """ 
    get a datetime maturity value from lines like: "T 3 % 03/31/30"
    """  
    maturity = pd.to_datetime(line [-8:], format='%m/%d/%y')
    return maturity


def get_coupon_payment_dates (maturity, settlement_date):
    if maturity .month == 2 and maturity. day>28:
        maturity = maturity.replace(day= 28)

    date = maturity.replace (year=settlement_date.year)

    if date.month <=6:
        date1, date2 = date, date + pd.DateOffset(months=6)

    else :
        date1, date2 = date - pd.DateOffset (months=6), date

    if settlement_date < date1:
        previous_payment = date1 - pd.DateOffset(months=6)
        next_payment= date1

    elif date1<=settlement_date:
        if settlement_date < date2:
            previous_payment = date1
            next_payment= date2
        else :
            previous_payment = date2
            next_payment = date2 + pd.DateOffset (months=6)

    t1 = pd.offsets.BusinessDay (1)
    if (previous_payment. day>27 or next_payment. day>27):
        previous_payment = previous_payment - t1 + pd.offsets.MonthEnd()
        next_payment = next_payment - t1 + pd.offsets.MonthEnd()

    if previous_payment> settlement_date:
        return next_payment - pd.DateOffset (years=1), previous_payment 
    
    return previous_payment, next_payment

def calculate_conversion_factor (first_delivery, coupon, maturity, long=True, ref_coup=6):
    coupon = coupon/100
    n_months = len(pd.date_range(first_delivery, maturity, freq=pd.DateOffset(months=1))) - 1
    n = n_months//12
    z = n_months%12
    if long:
        z = 3* (z//3)

    v=z 
    if z>=7:
        v = z-6
        if long :
            v = 3

    a = 1/(1.03**(v/6))
    b = coupon/2 * (6-v) /6
    if z<7:
        c = 1/(1.03** (2*n))
    else :
        c = 1/(1.03** (2*n+1))

    d = (1-c) * coupon/0.06
    conversion_factor = a* (coupon/2 + c + d) - b
    return np. round(conversion_factor,4)


def cf_from_cusip(first_delivery, cusip, bonds=None):
    if bonds is None:
        bonds = get_bonds ()
    coupon = bonds.loc[ cusip, 'coupon ']
    maturity = bonds.loc[cusip, 'maturity']
    long = (maturity - first_delivery).days/365 > 6
    return calculate_conversion_factor (first_delivery, coupon, maturity, long)

def get_CF(cusips, basket, first_delivery, long):
    CF = basket.loc[cusips].apply(lambda x : 
                calculate_conversion_factor(first_delivery, x.coupon, x.maturity, long),
                axis= 1)
    return CF

##########################
#multiple functions used for carry calculation
##########################

# get key values for carry and accrued interest calculation
def get_coupon_key_values_from_single_date (maturity, settlement_date, delivery_day):
    """ 
    Notations:

    cas: coupon ante (avans) settlement 
    cps: coupon post (aprés) settlement
    d: delivery day

    Three possible scenarios

    1 cas=cad   s      d    cps=cpd         |DC<0 n_c=-1
    
    2   cas     s   cps=cad    d    cpd     |DC=0 n_c=0

    3   cas     s     cps     cad    d  cpd |DC>0 n_c=1

    D1 = cps - s : days between settlement and next payment
    DC1= cps - cas : days between coupons surronding settlement day
    D2 = d - cad: days between delivery and previous payment
    DC2 = cpd - cad : days between coupons surronding delivery day
    DC = cad-cps:
    days between the two following coupons:
    the one before delivery day and the one after settlement
    """
 
    p1, n1 = get_coupon_payment_dates (maturity, settlement_date)
    D1 = (n1 - settlement_date).days
    DC1 =(n1 - p1).days

    p2, n2 = get_coupon_payment_dates (maturity, delivery_day)
    D2 = (delivery_day - p2).days
    DC2 = (n2 - p2).days

    DC = (p2 - n1).days
    nc = len(pd.date_range(n1, p2, freq=pd. DateOffset(months=6)))
    return D1, D2, DC1, DC2, DC, nc

# addapt previous function for arrays of dates
def get_coupon_key_values (maturity, settlement_date, delivery_day) :

    if hasattr (settlement_date,'__iter__'):
        coupon_info = [
            np.array (
                get_coupon_key_values_from_single_date (
                    maturity, date, delivery_day
                )
            ) 
            for date in settlement_date
        ] 
        
        D1, D2, DC1, DC2, DC, nc = np.array(coupon_info).T
        return D1, D2, DC1, DC2, DC, nc 
    
    else:
        return get_coupon_key_values_from_single_date (maturity, settlement_date, delivery_day)

#################
# The following two functions are used for efficient
# calculation of carry on a large number of dates

def get_mask(matrix, start, end, axis=1):
    """function used in calculation of carry for multiple dates
    Args:
    matrix (ndarray): is a repeated vector of repo series
                    matrix = (r, r, r, r). T

    start (ndarray): sating values of interest for each period
        for example interest for the coupon CPS start at (settlement_date + D1)
    
    end (type): ending values of interest for each period

    Returns:
    ndarray: an array mask where mask[i,jj = 1 only if start[i]<jend[i]
    """

    indices = np.arange(matrix.shape [axis])

    mask = (indices >= start[:, None]) & (indices < end[: ,None])
    return mask


def calculate_carry_accross_dates (full_price, dates, D1, D2, DC1, DC2, DC, nc, coupon, rp=None, term_rp=None) :
   
    """
    Important note on interest periods for repo gains from coupons and payments on bonds full price
    
    interest period for the full price:
        carry_period => rp[DO: D0+D1+D2+DC]

    interest period for the coupon CPS 
        cps => rP[D0+D1: D0+carry_period]
        if there is no coupon CPS then : DO+D1 > D0+carry_period 
        and the previous slicing gives an empty array
    interest period for the coupon CAD
    cad => rp[DO+D1+DC: DO + carry_period]
    this gain is only counted if DC>0: CAS I= CPS
    This function only works if there is less than two coupons between settlement and delivery
    """

    coupon_income = coupon/2 * (D1/DC1 + D2/DC2 + nc-1)

    carry_period = D1+DC+D2
    # t0: starting date of the series rp
    # DO: settlement date - tO
    if rp is not None:
        rp = rp.shift(1)
        D0 = pd.TimedeltaIndex(dates - rp.index[0]).days.to_numpy()

        rp = rp.to_numpy()
        rp = rp/(360*100)
        rp = np.tile(rp, (full_price.shape[0],1))

        interest_in_price = full_price*np.sum(get_mask(rp, D0, D0+carry_period, axis=1)*rp, axis=1)
        interest_in_cps = 0.5*coupon*np.sum(get_mask(rp, D0+D1, D0+carry_period, axis=1)*rp,axis=1)
        interest_in_cad = 0.5* coupon*np. sum(get_mask (rp, D0+D1+DC, D0+carry_period, axis=1)+rp, axis=1) *np.where(DC>0, 1, 8)
    
    if term_rp is not None:

        if type(term_rp) is pd.Series:
            #D0 = pd.TimedeltaIndex(dates - term_rp.index[0]).days.to_numpy()
            rp = term_rp.shift(1). loc[dates]
            rp = rp.to_numpy()
            rp = rp/(360*100)
        else :
            rp = term_rp/ (360*100)

        interest_in_price = full_price*rp*carry_period
        interest_in_cps = 0.5* coupon* rp* (carry_period-D1) *np.where(DC>=0, 1, 0)
        interest_in_cad = 0.5* coupon* rp* (carry_period-D1-DC) *np.where (DC>0, 1, 0)
    
    financing_cost = interest_in_price - interest_in_cps - interest_in_cad
    carry = coupon_income - financing_cost
    return np.round (32*carry, 2)

# The following function is used for efficient calculation of carry
# on one fixed settlement date, accross a large number of trajectory simulation.
# Note that here the settlement date can't be an array

def bond_key_values(coupon, maturity, settlement_date, delivery_day, rp=None, term_rp=None):
    bond_dict = {}
    D1, D2, DC1, DC2, DC, nc = get_coupon_key_values (maturity, settlement_date, delivery_day)
    
    # accrued interest
    AI = 0.5* coupon* (1 - D1/DC1)
    # coupon income
    coupon_income = coupon/2 * (D1/DC1 + D2/DC2 + nc-1)

    # add repo interest gain from coupon
    carry_period = D1+DC+D2
    rp = rp.loc[settlement_date:].to_numpy() if term_rp is None else term_rp*np.ones(carry_period)
    rp = rp/(368*100)
    interest_gain_cps = 0 if DC<0 else 0.5* coupon *np.sum(rp[D1: carry_period])
    interest_gain_cad = 0 if DC<=0 else 0.5* coupon*np.sum(rp[:D1+DC: carry_period])
    
    coupon_income += interest_gain_cad + interest_gain_cps

    # repo rate paid on bond_price. To be multiplied by bonds full price
    interest_rate_in_price = np.sum(rp[: carry_period])
    bond_dict['accrued_interest'] = AI
    bond_dict['coupon_income'] = coupon_income
    bond_dict['interest_rate_in_price'] = interest_rate_in_price

    return bond_dict

def calculate_carry (
    bond_price, coupon, maturity, settlement_date,
    delivery_day, rp=None, term_rp=None, bond_dict=None):
    
    """The following function is an adaptated function to calulate carry 
    efficiently in the two following cases:

    Case 1:

    bond_price is a large array resulting from a monte carlo simulation settlement_date is a fixed date
    Case 2:
    settlement_date is a large array If dates
    bond_price is an array of the same shape
    Returns:
    array: calculated carry from settlement date to delivery
    """
    if hasattr (settlement_date,'__iter__'):
        D1, D2, DC1, DC2, DC, nc = get_coupon_key_values (maturity, settlement_date, delivery_day)
        full_price = bond_price + 0.5*coupon* (1- D1/DC1)
        return calculate_carry_accross_dates(full_price, settlement_date, D1, D2, DC1, DC2, DC, nc, 
                                             coupon, rp, term_rp)

    else :
        if delivery_day<settlement_date:
            return 0
        bond_dict = (
            bond_dict or
            bond_key_values (coupon, maturity, settlement_date, delivery_day, rp, term_rp)
        )

        full_price = np.array(bond_price) + bond_dict['accrued_interest']
        financing_cost = full_price*bond_dict['interest_rate_in_price']
        carry = bond_dict['coupon_income'] - financing_cost
        return np.round (carry *32, 2)


def P_to_PF(P0, cusips, basket, set_day, delivery_day, term_rp):

    PF = np.nan*P0.copy()
    for cusip in cusips : 
        set_day_cusip = max(set_day, basket.loc[cusip, 'issue_dt'])
        bond_price = P0.loc[cusip]
        coupon = basket.loc[cusip, 'coupon']
        maturity = basket.loc[cusip, 'maturity']
        carry = calculate_carry(
                    bond_price, coupon, maturity, set_day_cusip,
                    delivery_day, term_rp=term_rp)
        PF.loc[cusip] = bond_price - carry/32

    return PF

# end of carry related functions
##########
##########
    
def calculate_basis (bond_price, future_price, c_factor):
    basis = np.array (32* (bond_price - c_factor*future_price))
    basis = np.round (basis, 2)
    return basis

def calculate_bnoc(bond_price=None, coupon=None, maturity=None, c_factor=None, settlement_date=None, 
                   last_delivery=None,future_price=None, rp=None, term_rp=None, bond_dict=None, 
                   basis=None, carry=None):
    
    basis = basis if basis is not None else calculate_basis (bond_price, future_price, c_factor)
    carry = carry if carry is not None else calculate_carry (bond_price ,coupon, maturity, settlement_date, 
                                                             last_delivery, rp, term_rp, bond_dict)
    return basis - carry

def calculate_accrued_interest (coupon, maturity, settlement_date):
    """ 
    Args:
        settlement_date:
            settlement_date, for the calculation of accrued interest on a bond purchase 
            
        delivery date: 
            for the calculation of accrued interest on a future delivery

    """
    if not hasattr (settlement_date,'__iter__'):
        previous_payment, next_payment = get_coupon_payment_dates(maturity, settlement_date)
        interest_period = (settlement_date - previous_payment).days
        days_between_coupons = (next_payment - previous_payment).days
    else :
        D1, _, DC1, _,_,_ = get_coupon_key_values(maturity, settlement_date, settlement_date[0])
        interest_period, days_between_coupons = DC1 - D1, DC1
    
    accrued_interest = (coupon/2)* (interest_period)/days_between_coupons
    return accrued_interest


def calculate_implied_repo_rate (bond_price, coupon, maturity, conversion_factor, settlement_date,
                                delivery_date, future_price, days_to_delivery=None, rounding=5):

    if days_to_delivery is None:
        days_to_delivery = (delivery_date - settlement_date)/np. timedelta64(1, 'D')
        days_to_delivery = max(days_to_delivery, 1)

    inv_price = conversion_factor*future_price + calculate_accrued_interest(coupon, maturity, delivery_date)
    purchase_price = bond_price + calculate_accrued_interest(coupon, maturity, settlement_date)
    _, next_payment = get_coupon_payment_dates (maturity, settlement_date)
    # check if a coupon payment was made before delivery
    if next_payment <= delivery_date:
        n2 = (delivery_date-next_payment)/np.timedelta64(1, 'D')
        numerator = (inv_price+coupon/2 - purchase_price) *360
        denominator = days_to_delivery*purchase_price - (n2* coupon/2)
        irr = 100*numerator/denominator
    else:
        irr = 100*360* (inv_price/purchase_price -1)/days_to_delivery
        irr = np.where(np.isnan (bond_price), -np.inf, irr)
        irr = np.round(irr, rounding)
    return irr


def calculate_bond_price(y, coupon, maturity, settlement_date):
    """ 
    y: float or np.array 
    """
    if not hasattr(settlement_date,'__iter__') :

        # normal case : no arrays
        # or MC simulation: y is an array and coupon might be an array
        previous_payment, next_payment = get_coupon_payment_dates (maturity, settlement_date)
        payment_dates = pd.date_range (next_payment, maturity, freq=pd.DateOffset(months=6))
        m = len (payment_dates)
        T = (next_payment - settlement_date). days / (next_payment - previous_payment) . days
        N=100

        # y of shape n: number of mc simulation
        v = np.array (1/(1+y/200)).reshape(-1, 1) # shape (n, 1)
        powers = np.arange(0, m) .reshape(1,-1) + T # shape (1,m) m: number of fut payments
        discount_factor = v**powers # shape (n, m)

        # coupon should be either of shape (n,m) or (1, m)
        coupon = np.array (coupon)
        coupon = np. tile(coupon. reshape (-1,1) , m)
        full_price = np.sum(0.5* coupon*discount_factor, axis=1) + N*discount_factor[:, -1]
        return full_price[0] if len(full_price)==1 else full_price
    
    else :
        # get necessary values for carry calculation
        D1, D2, DC1, DC2, DC, nc = get_coupon_key_values (maturity, settlement_date, maturity)
        # get necessary values for price calculation
        n_flux = nc
        max_n_flux = int (np.max (n_flux))
        y = np.array (y)
        y = np.tile(y.reshape(-1,1), max_n_flux)
        powers = np.arange(0, max_n_flux). reshape(-1, 1)
        powers = np.tile(powers, len(settlement_date)).T
        n_flux = np.tile(n_flux.reshape(-1,1), max_n_flux)
        powers = np.where(powers<n_flux, powers, np.inf)
        v = 1/ (1+y/200)
        T = D1/DC1
        N=100
        full_price = (v[:,0]**T) * (np.sum((0.5*coupon*(v**powers)), axis=1) + N*v[:,0]**(n_flux[:, 0]-1))
        return full_price
    
def yield_to_price (y, full_price, coupon, maturity, settlement_date):
    return calculate_bond_price(y, coupon, maturity, settlement_date) - full_price

def calculate_bond_yield(bond_price, coupon, maturity, settlement_date, is_dirty = False):
    bond_price_filled = np.where(np.isnan (bond_price), 100, bond_price)
    if not is_dirty:
        full_price = bond_price_filled + calculate_accrued_interest(coupon, maturity, settlement_date)
    else :
        full_price = bond_price_filled

    y0 = 3*np.ones_like(full_price)
    root = optimize.newton(yield_to_price, x0=y0, args=(full_price, coupon, maturity, settlement_date))
    root = np.where(np.isnan(bond_price), np.nan, root)
    if hasattr (root, '__itter__'):
        return root if len(root)>1 else root[0]
    
    else:
        return root
    
def calculate_forward_yield (y, coupon, maturity, settlement_date, delivery_date, rp=None, term_rp=None):

    # the expected yield of the bond at the delivery date
    # the bond is baught in repo by paying at del_date p-carry
    full_price = calculate_bond_price(y, coupon, maturity, settlement_date)
    AI = calculate_accrued_interest(coupon, maturity, settlement_date)
    bond_price = full_price - AI
    carry = calculate_carry(bond_price, coupon, maturity, settlement_date, delivery_date, term_rp=term_rp)
    forward_price = bond_price - carry /32
    forward_yield = calculate_bond_yield(forward_price, coupon, maturity, delivery_date)
    return forward_yield


def calculate_forward_price (y, coupon, maturity, settlement_date, delivery_date, rp=None, term_rp=None):

    # the expected yield of the bond at the delivery date
    # the bond is baught in repo by paying at del_date p-carry
    full_price = calculate_bond_price(y, coupon, maturity, settlement_date)
    AI = calculate_accrued_interest (coupon, maturity, settlement_date)
    bond_price = full_price - AI
    carry = calculate_carry (bond_price, coupon, maturity, settlement_date, delivery_date, term_rp=term_rp)

    forward_price = bond_price - carry/32
    return forward_price

## invoice functions
def get_fut_fwd_yld(cusip_data, fut_price, conversion_factor, fut_expi_date):

    fwd_ctd_price = fut_price* conversion_factor
    y = calculate_bond_yield(
        bond_price = fwd_ctd_price,
        coupon = cusip_data.loc[ 'coupon'], # type: ignore
        maturity = cusip_data. loc['maturity'], # type: ignore
        settlement_date = fut_expi_date
    )
    return y


def get_fut_fwd_risk(cusip_data, fut_fwd_yld, conversion_factor, fut_expi_date):

    d = calculate_duration(

        bond_price=100,
        coupon = cusip_data.loc[ 'coupon'], # type: ignore
        maturity = cusip_data.loc['maturity'], # type: ignore
        settlement_date = fut_expi_date,
        y = fut_fwd_yld
    )
    return d/conversion_factor


def calculate_duration (bond_price, coupon, maturity, settlement_date, y, mod=False):
    """ 
    y: float or np.array 
    """
    if not mod:
        bond_price=100

    if hasattr (bond_price,'__iter__') :
        # flatten array bond price so the d/bond_price don't give an error
        bond_price = np.array (bond_price).flatten()

    if not hasattr (settlement_date,'__iter__'):
        # normal case : no arrays
        # or MC simulation: y is an array and coupon might be an array
        previous_payment, next_payment = get_coupon_payment_dates (maturity, settlement_date)
        payment_dates = pd. date_range (next_payment, maturity, freq=pd.DateOffset (months=6))
        m = len (payment_dates)
        T = (next_payment - settlement_date) .days / (next_payment - previous_payment). days
        N=100
        # y of shape n : number of mc simulation
        v = np.array(1/(1+y/200)).reshape (-1,1) # shape (n, 1)
        powers = np.arange(0, m). reshape (1,-1)+ T # shape (1,m) m: number of fut payments
        discount_factor = v**powers # shape (n, m)
        # coupon should be either of shape (n,m) or (1, m)
        coupon = np.array (coupon)
        coupon = np.tile(coupon.reshape (-1, 1) , m)
        d = (np. sum(0.5* coupon *powers*discount_factor, axis=1) + N*powers[:, -1]*discount_factor[:, -1] )
        d = d*v.flatten()
        d = d/ (2*bond_price)
        return d[0] if len(d)==1 else d
        
    else:
        # get necessary values for carry calculation
        D1, D2, DC1, DC2, DC, nc = get_coupon_key_values (maturity, settlement_date, maturity)

        # get necessary values for price calculation
        n_flux = nc
        max_n_flux = int(np.max(n_flux))
        y = np.array (y)
        y = np. tile(y.reshape(-1,1), max_n_flux)
        powers = np.arange(0, max_n_flux). reshape(-1,1)
        powers = np. tile (powers, len(settlement_date)).T
        n_flux = np.tile(n_flux.reshape (-1,1), max_n_flux)
        powers = np.where(powers<n_flux, powers, np.inf)

        v = 1/ (1+y/200)
        T = D1/DC1
        N=100
        powers_with_inf = (powers+T.reshape(-1,1))
        powers_with_zeros = np.where(powers_with_inf==np.inf, 0, powers_with_inf)

        p1 = powers_with_inf
        p2 = powers_with_zeros

        d = np.sum((0.5* coupon*p2* (v**p1)), axis=1) + N*p2.max (axis=1)*v[:, 0]**p2.max (axis=1)
        d = d*v[:,0]
        d = d/(2*bond_price)
        return d


def calculate_gamma (coupon, maturity, settlement_date, y) :
    """
    y: float or np.array 
    """

    if not hasattr (settlement_date,'__iter__'):
        # normal case: no arrays
        # or MC simulation: y is an array and coupon might be an array
        previous_payment, next_payment = get_coupon_payment_dates (maturity, settlement_date)
        payment_dates = pd.date_range (next_payment, maturity, freq=pd.DateOffset (months=6))
        m = len (payment_dates)
        T = (next_payment - settlement_date).days / (next_payment - previous_payment). days
        N=100

        # y of shape n : number of mc simulation
        v = np. array (1/(1+y/200)) . reshape(-1,1) # shape (n, 1)
        powers = np.arange(0, m). reshape(1, -1)+ T # shape (1,m) m: numpber of fut payments
        discount_factor = v**powers # shape (n, m)

        # coupon should be either of shape (n,m) or (1, m)
        coupon = np.array (coupon)
        coupon = np.tile (coupon. reshape (-1, 1) , m)
        d = (np.sum(0.5* coupon*powers*(powers+1)*discount_factor, axis=1) + N*powers [:, -1]* (powers[:, -1]+1)*discount_factor[:, -1])
        d = 0.25* d*v.flatten()**2/N
        return d[0] if len(d)==1 else d
    
    else :
        # get necessary values for carry calculation
        D1, D2, DC1, DC2, DC, nc = get_coupon_key_values (maturity, settlement_date, maturity)

        # get necessary values for price calculation
        n_flux = nc
        max_n_flux = int (np.max(n_flux))
        y = np.array(y)
        y = np.tile(y.reshape (-1,1), max_n_flux)
        powers = np.arange(0, max_n_flux) .reshape (-1, 1)
        powers = np.tile(powers, len(settlement_date)).T
        n_flux = np.tile(n_flux.reshape(-1,1), max_n_flux)
        powers = np.where(powers<n_flux, powers, np.inf)
        v = 1/(1+y/200)
        T = D1/DC1
        N=100
        powers_with_inf = (powers+T.reshape(-1,1))
        powers_with_zeros = np.where (powers_with_inf==np.inf, 0, powers_with_inf)

        p1 = powers_with_inf
        p2 = powers_with_zeros

        d = np.sum( (0.5* coupon*p2* (p2+1)* (v*p1)), axis=1) + N*(p2.max(axis=1)+1)*p2.max(axis=1)*v[:,0]**p2.max(axis=1)
        d = 0.25*d*v[:, 0]**2/N
        return d
    

def calculate_theta(y, coupon, maturity, settlement_date, bond_price=None, full_price=None):

    previous_payment, next_payment = get_coupon_payment_dates (maturity, settlement_date)
    dc = (next_payment - previous_payment).days
    if full_price is None :
        if bond_price is None:
            full_price = calculate_bond_price (y, coupon, maturity, settlement_date)
        else :
            full_price = bond_price + calculate_accrued_interest (coupon ,maturity, settlement_date)

    clean_theta = -np. log(1/(1+y/200))+full_price/dc - coupon/2/dc
    return clean_theta/100


def calculate_daccrued(coupon, maturity, settlement_date):

    previous_payment, next_payment = get_coupon_payment_dates(maturity, settlement_date)
    dc = (next_payment - previous_payment).days
    return coupon/2/dc/100


def calculate_dfinancing(coupon, maturity, settlement_date, term_rp, bond_price=None, full_price=None):

    assert ( not(full_price is None and bond_price is None)), "no prices given. specify bond_price or full_price" 
    
    if full_price is None:
        full_price = bond_price + calculate_accrued_interest(coupon , maturity, settlement_date)
    
    return -full_price*term_rp/3600000


def get_sensis (cusip, settlement_date, y, term_rp, bonds) :

    coupon = bonds.loc [cusip, 'coupon']
    maturity = bonds.loc[cusip, 'maturity']

    full_price = calculate_bond_price(y, coupon, maturity, settlement_date)
    bond_price = full_price - calculate_accrued_interest(coupon, maturity, settlement_date)

    duration = calculate_duration (bond_price, coupon, maturity, settlement_date, y)
    gamma = calculate_gamma (coupon, maturity, settlement_date, y)
    theta = calculate_theta(y, coupon, maturity, settlement_date, bond_price=bond_price)
    daccrued = calculate_daccrued (coupon, maturity, settlement_date)
    repo = calculate_dfinancing(coupon, maturity, settlement_date, term_rp, bond_price)

    delta = -duration

    result = {
    'full_price': full_price/100,
    'bond_price': bond_price/100,
    'delta': delta,
    'gamma': gamma,
    'theta': theta + daccrued + repo,
    'clean_theta': theta,
    'accrued': daccrued,
    'repo': repo
    }
    return result


def snap_repo (fut_type=None, contract=None, ticker=None, date=None):

    if date is None:
        date = pd.Timestamp.today()

    if ticker is None:
        fut_file = fut_dic[fut_type]
        fut_ticker = ''.join([s.capitalize() for s in fut_file[:-1]])
        ticker = fut_ticker + fut_codes.loc[contract]

    if date >= pd.Timestamp.today() - pd.offsets. BDay(0, normalize=True):
        term_rp = blp.bds(ticker+ 'Comdty', flds= ['FUT_ACTUAL_REPO_RT'])

    else:
        term_rp = blp.bdh(ticker+ ' Comdty', flds=['FUT_ACTUAL_REPO_RT'], start_date=date, end_date=date)
    
    assert(len(term_rp>0)), "No repo data in bloom"
    return term_rp.iloc[0,0]


def snap_bond(cusips, fld, date=None) :

    if date is None:
        date=pd.Timestamp.today()

    if hasattr(cusips,'__iter__') and type (cusips) != str:
        ticker = [c + ' Govt' for c in cusips]

    else : 
        ticker = cusips + ' Govt'
    
    if date >= pd.Timestamp.today() - pd.offsets.BDay(0, normalize=True):
        res = blp.bdp(ticker, fld)
        res = res.T
        assert(len(res>0)), "No bond data in bloom"

    else:
        res = blp.bdh(ticker, fld, start_date=date, end_date=date)
        res.columns = res.columns. get_level_values(0)
        assert(len(res>0)), "No bond data in bloom"

    res.columns = [c[:-5] for c in res.columns]
    res = res.iloc[0]
    return res


def snap_fut (fut_type=None, contract=None, ticker=None, fld='px_mid', date=None):

    if date is None:
        date=pd.Timestamp.today()

    if ticker is None:
        fut_file= fut_dic[fut_type]
        fut_ticker =''.join([s.capitalize() for s in fut_file[:-1]])
        ticker = fut_ticker + fut_codes.loc[contract]

    if date >= pd.Timestamp.today() - pd.offsets.BDay(0, normalize=True):
        res = blp.bdp(ticker+ ' Comdty', fIds=fld)
    else:
        res = blp.bdh(ticker+ ' Comdty', flds=fld, start_date=date, end_date=date)

    assert(len (res>0)), "No fut data in bloom" 
    
    return res.iloc[0,0]



deliverable_rules = {
    '2Y'       : ( 1, 9, 2, 1, 5, 3),
    '3Y'       : ( 2, 9, 3, 0, 5, 3),
    '5Y'       : ( 4, 2, 5, 3, 5, 3),
    '10YN'     : ( 6, 6, 7, 9,10, 5),
    '10Y'      : ( 6, 6,10, 0,10, 5),
    '10Y Ultra': ( 9, 5,10, 0,10, 5),
    '20Y'      : (15, 0,25, 0,40, 0),
    '30Y'      : (25, 0,50, 0,50, 0)
}


def get_accurate_basket (bonds, contract, fut_type) :
        
    fut_file = fut_dic[fut_type]
    baskets = pd.read_csv(f'data/{fut_file}/baskets_{fut_file}.csv', index_col=0)
    cusips = baskets[contract].dropna()
    basket = bonds.loc[cusips]
    return basket


def get_basket(bonds, first_delivery, y_lower, m_lower, y_upper, m_upper, y_i, m_i):

    def filter (start, maturity):
        cond = (
        start + pd.DateOffset (years=y_lower, months=m_lower) <= maturity <
        start + pd.DateOffset(years=y_upper, months=m_upper)#+3)
        )
        return cond
    
    # condition on remaining maturity
    cond1 = bonds['maturity'].apply(lambda maturity: filter (first_delivery, maturity))
    # condition on original maturity
    cond2 = bonds['issue_dt'] + pd.DateOffset(months=m_i, years=y_i) >= bonds['maturity']
    # condition on existence of the bond
    cond3 = bonds ['issue_dt'] < first_delivery

    basket = bonds [cond1*cond2*cond3].copy()
    
    return basket



def get_contract_table(fut_type, contract):

    fut_file = fut_dic[fut_type]
    fut_ticker = ''.join([s.capitalize() for s in fut_file[:-1]])
    fut_code = fut_codes.loc[contract]
    fut_table = pd.read_csv(f'data/{fut_file}/fut_tables/{fut_ticker}{fut_code}.csv', index_col=0, parse_dates=True)
    set_dates = fut_table.index.to_series ().shift(-1)
    set_dates.iloc[-1] = set_dates.index[-1] + pd.offsets.BDay()
    fut_table['settlement_date'] = set_dates
    return fut_table

def get_fut_table(fut_type):

    fut_file = fut_dic[fut_type]
    fut_table = pd.read_csv(f'data/{fut_file}/{fut_file}_histo.csv', index_col=0, parse_dates=True)
    fut_table = add_contract_key(fut_table, fut_type)
    return fut_table

shifts = {
    '30Y': 7, # trading expires 7 b days before end of contract month
    '20Y': 7, # trading expires 7 b days before end of contract month
    '10Y': 7, # trading expires 7 b days before end of contract month
    '5Y' : 0, # trading expires on the lastrb day of contract month
    '2Y' : 0 # trading expires on the last b day of contract month
}

def add_contract_key (futures_table, fut_type) :

    fut_file = fut_dic[fut_type]
    last_trade = pd.read_csv(f'data/{fut_file}/{fut_file}_last_trade_dt.csv', index_col=0, parse_dates=[1])
    futures_table['contract'] = futures_table.apply(lambda x: last_trade[last_trade.values >= x.name].index[0], axis=1)
    set_dates = futures_table.index.to_series().shift(-1)
    set_dates.iloc[-1] = set_dates.index[-1] + pd.offsets.BDay ()
    futures_table['settlement_date'] = set_dates

    return futures_table

def get_repo (repo, bp_spread=0):
    """ 
    repo : str in [tger, ffund]
    """
 
    if repo == 'ffund':
        r = pd.read_csv (f'data/repo/repo_{repo}.csv', index_col=0, parse_dates=True).iloc[:,0]
    
    elif repo == 'sofr':
        source = 'bloom'
        rates = pd.read_csv (f'data/us_rates/{source}/rates.csv', index_col=0, parse_dates = True)
        r = rates ['1D'].shift(-1)
    else : 
        raise ValueError(f"repo {repo} is not saved in data/repo folder.")
    return r + bp_spread/100 
    
    
def interpol_repo(repo, date, delivery_date):

    exp_date = date + pd.DateOffset(months=3)
    deltat1 = (delivery_date - date).days
    deltat2 = (exp_date - date).days
    ins, trois_mois = repo.loc[date].values
    rt = ins + (deltat1)*(trois_mois-ins)/deltat2
    return rt


def get_vol (vol1mois, vol3mois, date, first_delivery):

    exp_date1 = date+pd.DateOffset(months=1)
    exp_date3 = date+pd.DateOffset (months=3)
    if type(vol1mois) is pd.Series: 
        un_mois, trois_mois = vol1mois.loc[date], vol3mois.loc[date]
    else :
        un_mois, trois_mois = vol1mois, vol3mois

    if first_delivery<exp_date1:
        return un_mois
    
    if first_delivery < date:
        return un_mois
    
    deltat1 = (first_delivery - exp_date1).days
    deltat2 = (exp_date3 - exp_date1).days
    vol_first_delivery = un_mois + (deltat1) * (trois_mois-un_mois)/deltat2
    return vol_first_delivery


# from ot code to my codes
ot_to_fut_type = {
    'TNOTE2Y_CBT'       : '2Y',
    'TNOTE5Y_CBT'       : '5Y',
    'TNOTE10Y_CBT'      : '10Y',
    'ULTRA_TNOTE10Y_CBT': '10Y_Ultra',
    'TNOTE30Y_CBT'      : '20Y',
    'TNOTE25Y_CBT'      : '30Y'

}

fut_type_to_ot = {

    '2Y'       :'TNOTE2Y_CBT'       , 
    '5Y'       :'TNOTE5Y_CBT'       ,
    '10Y'      :'TNOTE10Y_CBT'      ,
    '10Y_Ultra':'ULTRA_TNOTE10Y_CBT',
    '20Y'      :'TNOTE30Y_CBT'      ,
    '30Y'      :'TNOTE25Y_CBT'      ,

}

def get_ot_fut_id(fut_type, contract):

    ot_fut_id = fut_type_to_ot [fut_type]
    ot_fut_id += ':' + contract[:3] + contract [-2:]
    return ot_fut_id


def fut_type_to_ticker (fut_type, contract):
    fut_file= fut_dic[fut_type]
    fut_ticker = ''.join([s.capitalize() for s in fut_file[: -1]])
    ticker = fut_ticker + fut_codes. loc[contract]
    return ticker


# on vs off utils

def get_bonds_list(tenor, bonds = None, tab=None):

    if bonds is None:
        bonds = get_bonds ()

    if tab is None:
        tab = get_bond_perfs()

    sub = bonds[bonds.length == float (tenor)].sort_values (by='issue_dt') # type: ignore
    bond_list = list(sub[sub.issue_dt > tab.index[0]].sort_values(by='issue_dt').index)
    return bond_list


def last_auctioned_bond(x, i):
    return x.dropna().iloc[-i-1]

def last_issued_bond(x, i):
    date = x.name
    issue_dates = bonds.loc[x.index, 'issue_dt'] # type: ignore

    # drop non issued
    x = x[issue_dates <= date]

    x = x.loc[issue_dates.sort_values().index] # type: ignore
    return x.dropna().iloc[-i-1]


def last_auctioned_bond_cusip(x,i) :
    return x.dropna().index[-i-1]

def last_issued_bond_cusip(x,i, bonds) :
    date = x.name
    issue_dates = bonds.loc[x.index, 'issue_dt'] # type: ignore
    x = x.loc[issue_dates.sort_values().index] # type: ignore
    # drop non issued
    x = x[issue_dates <= date]
    return x.dropna().index[-i-1]

def get_tab_otr(tab, tenor, i, bonds=None):

    bond_list = get_bonds_list(tenor, bonds = bonds)
    # bonds could be absent from the tab
    #bond_list = [c for c in bond_list if c in tab. columns]
    start = tab[bond_list[i]].dropna().index[0]
    return tab.loc[start: , bond_list].apply(lambda x: last_auctioned_bond(x,i), axis=1)


def get_tab_otr_cusip(tab, tenor, i, bonds=None):
    bond_list = get_bonds_list(tenor, bonds = bonds)
    # bonds could be absent from the tab
    #bond_list = [c for c in bond list if c in tab.columns]
    start = tab[bond_list[i]].dropna().index[0]
    return tab.loc[start:, bond_list].apply(lambda x: last_auctioned_bond_cusip(x,i), axis = 1)


''' 
## z spread utils
from scipy import optimize
from swap_pv import swap_lib as swlb


def get_ois_z_spread(cusip, date, rt, dirty, bonds, pyd):

    set_date = max(add_business_day(date) , bonds. loc[cusip, 'issue_dt'])
    payment_dates = pyd.loc[cusip]
    payment_dates = payment_dates [payment_dates> set_date]
    cash_flows = np.zeros_like(payment_dates).astype(float)
    cash_flows = cash_flows + bonds.loc[cusip, 'coupon']/2
    cash_flows [-1] += 100

    zc = swlb.strip_2c_from_swap(rt, date)
    zc = swlb.interpolate_zc(zc, date)
    if np.isnan (dirty):
        return np.nan 
    
    def price(shift):
        h = 1
        dt = (zc.index - date).days/360

        spot_rates = (zc**(-h/dt) - 1)/h +shift/10000
        zc_shift = (1+spot_rates*h)**(-dt/h)
        pv = ((zc_shift).loc[payment_dates]@cash_flows)
        return pv-dirty*zc.loc[set_date]
    
    root : float = optimize.newton (price, x0=0) # type: ignore
    return np.round(root, 2)


def bond_pv(cusip, date, rt, z_spread, bonds, pyd) :

    coupon = bonds.loc[cusip, 'coupon']
    set_date = max(add_business_day(date) , bonds. loc[cusip, 'issue_dt'])
    payment_dates = pyd.loc[cusip]
    payment_dates = payment_dates [payment_dates> set_date]
    if len (payment_dates)>0:
        cash_flows = np.zeros_like (payment_dates).astype(float)
        cash_flows = cash_flows + coupon/2
        cash_flows[-1] += 100

    C = pd.Series (index=payment_dates, data=cash_flows)
    zc = swlb.strip_zc_from_swap(rt, date)
    zc = swlb.interpolate_zc(zc, date)
    h = 1
    dt = (zc.index - date).days/360
    spot_rates = (zc** (-h/dt) - 1) /h + z_spread/10000
    zc_shift = (1+spot_rates*h)** (-dt/h)

    return (C*zc_shift).dropna().sum()/zc_shift.loc[set_date]


def bond_pv_from_zc(cusip, date, zc, z_spread, bonds, pyd) :

    act = 360
    coupon = bonds. loc [cusip, 'coupon']
    set_date = max(add_business_day (date) , bonds. loc[cusip, 'issue_dt'])
    payment_dates = pyd.loc[cusip]
    payment_dates = payment_dates [payment_dates> set_date]
    if len (payment_dates)>0:
        cash_flows = np.zeros_like(payment_dates).astype(float)
        cash_flows = cash_flows + coupon/2
        cash_flows[-1] += 100 
        
    else:
        return 100
    
    C = pd.Series (index=payment_dates, data=cash_flows)
    h = 1
    dt = (zc.index - date).days/act
    spot_rates = (zc** (-h/dt) - 1)/h +z_spread/10000
    zc_shift = (1+spot_rates*h)**(-dt/h)

    return (C*zc_shift).dropna().sum()/zc_shift.loc[set_date]


def bond_delta_curve(cusip, date, bonds, pyd, rt=None, z_spread=None, rates=None, Z = None) :

    if rt is None :
        rt = rates.loc[date] # type: ignore

    if z_spread is None:
        z_spread = Z.loc[date, cusip] # type: ignore

    pv = bond_pv(cusip, date, rt, z_spread, bonds, pyd)

    dv01 = rt.copy().apply(lambda x:0)

    rt_temp = rt.copy()
    for tenor in dv01.index:
        rt_temp.loc[:] = rt
        rt_temp.loc [tenor] += 0.01
            
        pv1bp = bond_pv(cusip, date, rt_temp, z_spread, bonds, pyd)
        dv01.loc[tenor] = pv1bp - pv

    dv01 = (dv01*100).round(2)

    return dv01


def bond_delta_curve_spot(cusip, date, zero_coupon_rates, zero_coupon_maturity, bonds, pyd, Z) :
    ACT = 360
    zc_rates_t = zero_coupon_rates.loc[date]
    zc_rates_t.index = zero_coupon_maturity.loc[date]
    zc_rates_t_interpol = swlb.interpolate_zc(zc_rates_t, date)

    z_spread = Z.loc[date, cusip]
    pv = bond_pv_from_zc(cusip, date, zc_rates_t_interpol, z_spread, bonds, pyd)
    t = (zc_rates_t.index - date) .days /ACT
    spot_rates = (zc_rates_t**(-1/t) -1)
    spot_rates_temp = spot_rates.copy()
    dv01 = zero_coupon_maturity.loc[date].apply (lambda x: 0)

    for tenor in dv01.index:
        spot_rates_temp.loc[:] = spot_rates
        spot_rates_temp.loc[zero_coupon_maturity.loc[date, tenor]] += 1e-4
        zc_rates_t_temp = (1+spot_rates_temp)**(-t)

        c_rates_t_temp_interpol = swlb.interpolate_zc(zc_rates_t_temp, date)
        pv1bp = bond_pv_from_zc(cusip, date, c_rates_t_temp_interpol, z_spread, bonds, pyd)
        dv01.loc [tenor] = (pv1bp-pv)/100

    return dv01


def bond_dv01(cusip, date, bonds, pyd, rt=None, _spread=None, rates=None, Z = None):

    if rt is None :
        rt = rates.loc[date] # type: ignore

    if z_spread is None :
        z_spread = Z.loc[date, cusip] # type: ignore

    pv = bond_pv(cusip, date, rt, z_spread, bonds, pyd)
    pv1bp = bond_pv(cusip, date, rt +0.01, z_spread, bonds, pyd)
    return pv1bp - pv


def bond_pv_vect(cusips, date, rt, bonds, pyd, Z_spread = None, value_at_settlement=False):

    zc = swlb.strip_zc_from_swap(rt, date)
    zc = swlb.interpolate_zc(zc, date)
    C = pd.DataFrame (index = zc.index, columns = cusips, dtype = float)
    ZC= pd.DataFrame (index = zc.index, columns = cusips, dtype = float)
    for cusip in cusips:
        coupon = bonds.loc[cusip, 'coupon' ]
        set_date = max(add_business_day(date) , bonds. loc[cusip, 'issue_dt' ])
        payment_dates = pyd.loc[cusip]
        payment_dates = payment_dates[payment_dates> set_date]
        if len (payment_dates) >0:
            cash_flows = np.zeros_like (payment_dates).astype(float)
            cash_flows = cash_flows + coupon/2
            cash_flows [-1] += 100
            C.loc[payment_dates, cusip] = cash_flows

        ZC[cusip] = zc.copy ()
    
    C = C.dropna (how='all', axis=0)
    ZC = ZC.loc[C.index].copy()

    if Z_spread is None :
        Z_spread = np.zeros (len (cusips))
    dt = (ZC.index - date).days/360
    h=1 ## assuming zc = (1+r*h)** (-dt/h)
    spot_rates = (np.exp(np.log(ZC).multiply(-h/dt,axis=0)) -1)/h
    ZC = np.exp(np. log (1+ (spot_rates+Z_spread/10000)*h).multiply(-dt/h, axis=0))

    PV = (ZC*C).sum()

    if value_at_settlement :
        for cusip in cusips:
            set_date = max(add_business_day (date) , bonds. loc[cusip, 'issue_dt'])
            df = zc. loc [set_date]
            PV.loc[cusip] = PV.loc[cusip]/df
    return PV


def bond_dv01_vect (cusips, date, rt, bonds, pyd) :

    return bond_pv_vect(cusips, date, rt, bonds, pyd) - bond_pv_vect(cusips, date, rt+0.01, bonds, pyd)

def get_ois_z_spread_vect (cusips, date, rates, Dirty, bonds, pyd):

    rt = rates.loc [date]
    mids = Dirty.loc[date, cusips]
    zc = swlb.strip_zc_from_swap(rt, date)
    zc = swlb.interpolate_zc(zc, date)
    C = pd. DataFrame (index = zc.index, columns = cusips, dtype = float)
    ZC = pd.DataFrame (index = zc.index, columns = cusips, type = float)
    discounted_mids = pd.Series (index=cusips, dtype=float)
 
    for cusip in cusips:
        coupon = bonds. loc [cusip, 'coupon']
        set_date = max(add_business_day (date) , bonds. loc[cusip, 'issue_dt'])
        discounted_mids.loc[cusip] = zc. loc [set_date]*mids. loc [cusip]
        payment_dates = pyd. loc [cusip]
        payment_dates = payment_dates [payment_dates> set_date]

        if len (payment_dates)>0:
            cash_flows = np.zeros_like (payment_dates).astype(float)
            cash_flows = cash_flows + coupon/2
            cash_flows [-1] += 100
            C. loc[payment_dates, cusip] = cash_flows

        ZC [cusip] = zc


    C = C. dropna (how= 'all', axis=0)
    ZC = ZC.loc[C.index].copy()
    dt = (ZC. index - date). days/360
    h=1
    ## assuming zc = (1+r*h)**(-dt/h)
    spot_rates = (np.exp(np.log(ZC).multiply(-h/dt,axis=0))-1)/h

    def price(shift):
        zc_shift = np.exp(np.log(1+ (spot_rates+shift/10000)*h).multiply(-dt/h, axis=0))
        return ((zc_shift*C).sum() - discounted_mids).values
    
    shift = np.zeros_like(mids)
    root = optimize.newton (price, x0=shift)
    z_spread = pd.Series (index = cusips, data = root)
    return z_spread


'''

## filter utils 
def get_ctd_tab (fut_type):
    fut_file = fut_dic[fut_type]
    ctd_tab = pd.read_csv(f'data/filters/{fut_file}_ctd.csv', index_col=0)
    ctd_tab.columns = ctd_tab.columns.astype(int)
    return ctd_tab


def get_ctd_tabs ():
    ctd_tabs = pd.DataFrame()
    for fut_type in fut_dic.keys () :
        fut_file = fut_dic[fut_type]
        ctd_tab = get_ctd_tab(fut_type)
        ctd_tab.columns = [f'{fut_file[:-1]}_{c}' for c in ctd_tab.columns]
        ctd_tabs = pd.concat((ctd_tabs, ctd_tab), axis=1)
    return ctd_tabs 

def get_otr_tab(tenor):
    otr_tab = pd.read_csv(f'data/filters/{tenor}_otr.csv', index_col=0, parse_dates=True)
    return otr_tab 

def get_otr_tabs ():
    otr_tabs = pd.DataFrame()
    tenors = [2,3,5,7, 10, 20, 30]
    for tenor in tenors:
        otr_tab = get_otr_tab(tenor)
        #otr_tab.columns = [f'CT{tenor}_{c}' for c in otr_tab.columns]
        otr_tabs = pd. concat ((otr_tabs, otr_tab), axis=1)
    return otr_tabs


def get_repo_tab(tenor, ptype) :
    repo_tab = pd.read_csv (f'data/repo/repo_{tenor}Y_{ptype}.csv', index_col=0, parse_dates=True)
    return repo_tab


def get_repo_tabs () :
    repo_tabs = pd.DataFrame()
    repo_tabs_bid_ask = pd.Dataframe()
    tenors = [2,3,5, 7, 10, 20, 30]
    for tenor in tenors:
        repo_tab = get_repo_tab(tenor, ptype='ask')
        bidask = repo_tab - get_repo_tab(tenor, ptype='bid')

        repo_tabs = pd.concat( (repo_tabs, repo_tab), axis=1)
        repo_tabs_bid_ask = pd.concat((repo_tabs_bid_ask, bidask), axis=1)
    return repo_tabs, repo_tabs_bid_ask