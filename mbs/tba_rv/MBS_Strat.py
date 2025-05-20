
# MBS_Strat.py
############################################
from re import I
import streamlit as st
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
import plotly.express as px
from scipy.optimize import minimize_scalar, minimize, LinearConstraint
import dataiku
import pickle

from functions import *

st.set_page_config(layout="wide")

### GLOBAL VARIABLES ###
MTG_MAT = 30
MONTHLY = True
factor = 1
if MONTHLY:
    factor *= 12

TRIANG = np.tri(MTG_MAT*factor+1, MTG_MAT*factor+1).T
ALL_CPNS = np.arange(2.5, 7.0, 0.5)
ALL_CPNS_TBAS = np.arange(1.5, 8.0, 0.5)
mode = st.sidebar.selectbox("Mode", ["Steps", "General Study"])

#### DEFINING USEFUL FUNCTIONS FOR THE CODE ####
from numba import jit
@jit
def GENERATE_PRICER(
    SOFR_DATA, COUPON=2.5/100, OAS=102/100, VOL=102/100, 
    triangular_matrix=np.tri(30*12+1, 30*12+1).T, floor_quantity=0.5, 
    most_expensive_quantity=0.5, LOW_CPR=0.055, HIGH_CPR=0.42, MTG_MATURITY=30, 
    IS_MONTHLY=True, my_pickle=None):

    factor = 1
    if IS_MONTHLY:
        factor = 12
    LENGTH = MTG_MATURITY * factor + 1
    LENGTH_IDX = np.arange(LENGTH)
    WIDTH = 15
    my_arr = np.empty(shape=(LENGTH, WIDTH))
    # annuity
    my_arr[:, 0] = 10000*(1-(1+COUPON)**-((MTG_MATURITY*factor-LENGTH_IDX)/factor))/(1-(1+COUPON)**(-MTG_MATURITY))
    # low cpr
    my_arr[:, 1] = my_arr[:, 0]*np.exp(-LOW_CPR/factor*LENGTH_IDX)
    # high cpr
    my_arr[:, 2] = my_arr[:, 0]*np.exp(-HIGH_CPR/factor*LENGTH_IDX)
    # CF bond
    x = np.ascontiguousarray(my_arr[:,1])
    my_arr[1:, 3] = -np.diff(x)+ COUPON*np.roll(my_arr[:,1],1)[1:]/factor
    # Notio (low - High)
    my_arr[:,4] = my_arr[:,1] - my_arr[:,2]
    # Rates + OAS
    my_arr[:, 5] = (SOFR_DATA + OAS) / 100
    # ZC
    my_arr[:, 6] = 1
    for i in range(1, len(my_arr[:, 5])):
        my_arr[i, 6] = my_arr[i-1, 6]/(1 + my_arr[i,5]/factor)

    # DURATION
    my_arr[:, 7] = triangular_matrix @ (my_arr[:, 6] * my_arr[:, 4]) / 10000 / factor

    # Fwd Sofr
    my_arr[:,8] = triangular_matrix @ (my_arr[:, 5]*my_arr[:, 6] * my_arr[:, 4])/my_arr[:, 7]/10000/factor

    # xSwaption
    my_arr[:,9] = (COUPON - my_arr[:,8])/(VOL*np.sqrt(LENGTH_IDX/factor))
    
    # TODO comprendre  
    ERF_swpt = np.array([retrieve_erf(x, my_pickle) for x in my_arr[:,9]])
    density_swpt = (1/np.sqrt(2*np.pi))*np.exp((-my_arr[:,9]**2)/2)

    # PSwaptions
    my_arr[:,10] = my_arr[:,7]*VOL*np.sqrt(LENGTH_IDX/factor)*(my_arr[:,9]*ERF_swpt + density_swpt)*10000

    # Delta Swaptions  
    my_arr[:,11] = my_arr[:,7]*ERF_swpt + triangular_matrix@(LENGTH_IDX*my_arr[:,6]/factor)/factor/10000*(my_arr[:,7]*VOL*np.sqrt(LENGTH_IDX/factor)*(my_arr[:,9]*ERF_swpt + density_swpt)*10000)/my_arr[:,7]/factor

    #xCap
    my_arr[:,12] = (COUPON - my_arr[:,5])/(VOL*np.sqrt(LENGTH_IDX/factor))
    ERF_cap = np.array([retrieve_erf(x, my_pickle) for x in my_arr[:,12]])
    density_cap = (1/np.sqrt(2*np.pi))*np.exp((-my_arr[:,12]**2)/2)
   
    #Pcap
    my_arr[:,13] = my_arr[:,6]*VOL*np.sqrt(LENGTH_IDX/factor)*(my_arr[:,12]*ERF_cap+density_cap)*my_arr[:,4]/factor

    #Delta Cap
    #commented

    x = my_arr[:,10]
    ret_max_spt = np.nanmax(x)

    ret = (np.nansum((my_arr[1:, 3]*my_arr[1:,6])) -(most_expensive_quantity*ret_max_spt+ floor_quantity*np.nansum(my_arr[:, 13])))/100

    return ret


def solve_oas(coupon):
    def f(x):
        th_price = GENERATE_PRICER(ref_data, coupon/100, x/100, 10000, TRIANG, HIGH_CPR=0.5, my_pickle=pkl)
        tba = TBAS_PRICES.loc[ref_date, coupon]
        ret = abs(th_price-tba)
        return ret
    res = minimize_scalar(f, bounds=(-1000, 1000), method="bounded", tol=0.01)
    return res

def dataiku_retrieve():
    tod = datetime.datetime.today()
    st.session_state["today"] = tod
    with st.spinner("My data is being retrieved from Dataiku"):
        st.session_state["today"] = tod
        dataiku.set_remote_dss("https://automation-dataiku.ca.cib", 
                               "NaHD9EXjcKlncEfpRfZ30on1EBAMKnzzV", 
                               no_check_certificate=True)
        client = dataiku.api_client()
        project = client.get_project("FICONVEXITYHEDGINGNEEDS")
        df = project.get_dataset("forwardFull - ATM").get_dataframe()
        df.set_index("Date", inplace=True)
        swaptions_histo = df.filter(regex="ATM")
        swaptions_histo.columns = [col[:-4] for col in swaptions_histo.columns]
        swaptions_histo = swaptions_histo.sort_index(ascending=False)
        swaptions_histo.index = [t.to_pydatetime().strftime("%Y-%m-%d") for t in swaptions_histo.index]
        st.session_state["implied_vol"] = swaptions_histo


def multisolving(ref_date, x0, pkl, IS_MONTHLY = False, vs_mortgage = False):

    ref_data = DSCT_DATA.loc[ref_date]
    ref_data.index = ref_data.index.astype(int)
    ref_data.loc[0] = ref_data.loc[1]
    ref_data.index = ref_data.index/30
    factor = 1
    if IS_MONTHLY:
        factor *= 12
    ref_data = ref_data.loc[np.arange(30*factor+1)].to_numpy()
    VOL = IMPLIED_VOL_DATA.loc[ref_date]
    MTG_value = MTG.loc[ref_date].values
    mtg = 0
    if vs_mortgage:
        mtg = 1
    def f(x):
        tbas = TBAS_PRICES.loc[ref_data, ALL_CPNS].dropna()
        bmin = x[0]
        bmax_cst = x[1]
        beta = x[2]
        squared_sum = 0
        for i, tba in enumerate(tbas):
            cpn = list(tbas.index)[i]
            bmax = bmax_cst + ((cpn-mtg*MTG_value)/100)*beta + ((cpn-mtg*MTG_value)/100)**2*gamma
            OAS = OAS_DATA.loc[ref_data, cpn]
            if np.isnan(OAS):
                pass
            else:
                th_price = GENERATE_PRICER(ref_data, cpn/100, OAS/100, VOL/10000, TRIANG, LOW_CPR, HIGH_CPR, IS_MONTHLY=IS_MONTHLY, my_pickle=pkl)
                squared_sum += (th_price - tba)**2
        return squared_sum
    linear_const1 = LinearConstraint([-1, 1, 2/100], lb=[0], ub=[np.inf])
    linear_const2 = LinearConstraint([-1, -1, 1], lb=[-np.inf], ub=[0])

    res = minimize(f, x0= x0, bounds = ([0.0001, np.inf], [0.0001, np.inf]), 
                   constraints=[linear_const1, linear_const2], method='SLSQP')
    
    return res.x[0], res.x[1], res.x[2]


def get_oas_from_price(ref_date, TO_FILL, idx_x, idx_pk1, low_cpr=0.5, high_cpr=0.5, variable_to_fit="OAS", IS_MONTHLY=False, high_barrier=None):
    ref_data = DSCT_DATA.loc[ref_date]
    ref_data.index = ref_data.index.astype(int)
    ref_data.loc[0] = ref_data.loc[1]
    ref_data.index = ref_data.index / 30
    factor = 1
    if IS_MONTHLY:
        factor *= 12

    ref_data = ref_data.loc[np.arange(30 * factor + 1)].to_numpy()
    VOL = IMPLIED_VOL_DATA.loc[ref_date]

    match variable_to_fit:
        case "OAS":
            low_bound = -1000
            high_bound = 1000
        case "High barrier":
            low_bound = low_cpr
            high_bound = high_cpr

    for idx_col, c in enumerate(ALL_CPNS):
        # lp = LineProfiler()
        # lp_wrapper = lp(GENERATE_PRICER)
        # lp_wrapper(ref_data, my_pickle=pkl)
        # lp.print_stats()

        def f(x):
            match variable_to_fit:
                case "OAS":
                    high_cpr = high_barrier.loc[ref_date, c]
                    th_price = GENERATE_PRICER(
                        ref_data, c/100, x/100, VOL/10000, TRIANG,
                        LOW_CPR=low_cpr, HIGH_CPR=high_cpr,
                        IS_MONTHLY=IS_MONTHLY, my_pickle=pkl
                    )
                case "High barrier":
                    OAS = OAS_DATA.loc[ref_date, c]
                    th_price = GENERATE_PRICER(
                        ref_data, c / 100, OAS / 100, VOL / 100, TRIANG,
                        LOW_CPR=low_cpr, HIGH_CPR=high_cpr,
                        IS_MONTHLY=IS_MONTHLY, my_pickle=pkl
                    )
            tba = TBAS_PRICES.loc[ref_date, c]
            ret = abs(th_price - tba)
            return ret

        if np.isnan(TBAS_PRICES.loc[ref_date, c]):
            TO_FILL[idx_x, idx_col] = np.nan

        else:
            res = minimize_scalar(f, bounds=(low_bound, high_bound), method="bounded", tol=0.001)
            TO_FILL[idx_x, idx_col] = res.x

            # if fill_second_array:
            #     variable_to_fit = "OAS"
            #     high_cpr = TO_FILL[idx_idx, idx_col]
            #     res = minimize_scalar(f, bounds=(-1000, 1000), method="bounded", tol=0.05)
            #     TO_FILL2[idx_idx, idx_col] = res.x

    return TO_FILL  # , TO_FILL2





#################################### CODE START ####################################

match mode:
    case "General Study":
        start = datetime.datetime.now()
        TBAS_PRICES, IMPLIED_VOL_DATA = load_data()
        MTG = pd.read_csv(r"Data_extraction/Data/TBA/MORTGAGE_RATE.csv", index_c
