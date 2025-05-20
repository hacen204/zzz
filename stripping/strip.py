# pylint: disable-all
main_path = 'e://ehz'
import sys
import os
sys.path.append(main_path)
sys.path.append('e://stripping/KR_example-main/src')
os.chdir(main_path)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(rc = { 'figure.figsize': (15, 5) })


import utils
#import fut_pricer

import kr_model
import kr_utils

import plotly.express as px




def get_bond_position(x, basket):
    return basket[basket['length'] == x.length].sort_values(by = 'issue_dt', ascending = False).index.get_loc(x.name)

def roundPartial (value, resolution):
    return round (value / resolution) * resolution

def custom_bond_position(x, basket, R):
    return basket[
                basket['ttm'].apply(lambda ttm: roundPartial(ttm, R)) == 
                roundPartial(x.ttm, R)
                ].sort_values(by = 'issue_dt', ascending = False).index.get_loc(x.name)


def construct_C (ttm,  pyd,  bonds,  date):     
    # Matrix defining al1 future flows:     
    # [i,j] = flow at time tj for bond i
    max_ttm = ttm.max()    
    n_bonds = len(ttm)
    C = np.zeros((n_bonds, int(max_ttm)))
    for i  in range(n_bonds):        
        cusip = ttm.index[i]
        coupon = bonds.loc[cusip, 'coupon']
        xij = pyd.loc[[cusip]].apply(lambda x:(x-date).days).values         
        xij = xij[xij>0]
        C[i, xij-1] = coupon/2
        T = xij[-1]
        C[i, T-1] +=100     
    return C

def prep_data(P, C ,density, bonds_density):
    P_array = P.values
    n_bonds = len(P)
    ytm, dur = np.zeros (n_bonds), np.zeros(n_bonds) # YTM and duration
    ttm = np.zeros(n_bonds) # time to maturity in days
    for i in range(n_bonds):
        time_to_cashflow_inday = np.where(C[i]!=0)[0] + 1
        ytm [i] , dur[i] = kr_utils.get_ytm_and_duration(C[i][time_to_cashflow_inday-1], time_to_cashflow_inday, P_array[i])
        ttm [ i ] = max(time_to_cashflow_inday)

    n_tenors = len(density)
    h = (bonds_density*n_tenors).values
    inv_w = (dur*P_array) **2*h
    data = { 'P':P_array, 'C':C, 'ytm':ytm, 'dur':dur, 'ttm':ttm, 'inv_w':inv_w }
    return data
    

def strip():

    P = utils.get_bond_prices(bond_type='us', ptype='mid')
    bonds = utils.get_bonds()
    bonds['length'] = bonds['length'].apply(lambda x: x if x!=31 else 30)
    pyd = utils.get_bonds_pyd()
    Y = utils.get_bond_yields(bond_type='us', ptype='mid')

    act = 365
    alpha = 0.05
    delta = 0
    ridge = 1
    ffund = utils.get_repo('ffund')

    pyd = utils.get_bonds_pyd()
    from tqdm import tqdm

    G =pd.read_csv('data/US_TREAS_CURVE/OTR/discount_factors.csv',index_col=0, parse_dates=True)
    RATES= pd.read_csv('data/US_TREAS_CURVE/OTR/rates.csv', index_col=0, parse_dates=True)
    E=pd.read_csv('data/US_TREAS_CURVE/OTR/errors.csv', index_col=0, parse_dates=True)
    G.columns=[int(c) for c in G.columns]
    RATES.columns=[int(c) for c in RATES.columns]

    failed=[]
    dates=list(reversed(P.loc[G.index[-1]:].index))


    i=1
    for date in tqdm(dates):

        print(date)
        s_date= utils.add_business_day(date, n=1)
        P0=P.loc[date]
        

        used_date= s_date

        filter1 = bonds['issue_dt']<=date
        filter2 = bonds['maturity']>date
        filter3 = bonds.index.isin(P.columns)

        basket= bonds[filter1 & filter2 & filter3].copy()

        t1=pd.offsets.BusinessDay(1)
        basket['ttm']= (basket['maturity']-t1+t1 - used_date).apply(lambda x:x.days/act)
        basket['age']= (used_date - basket['issue_dt']).apply(lambda x:x.days/act)
        basket['ttm_p'] =100*basket['ttm']/basket['length']
        basket['length_text']=basket['length'].astype(int).astype(str)
        basket['otr_position']=basket.apply(lambda x: get_bond_position(x, basket), axis=1)
        R=0.5
        basket['custom_position'] = basket.apply(lambda x: custom_bond_position(x, basket,R), axis=1)

        filter1 = basket['ttm']>1
        filter2 = basket['ttm_p']>50
        filter3 = basket['otr_position']<1
        filter4 = basket['custom_position']==0

        filters=pd.concat([
            #filter1,
            #filter2,
            filter3,
            #filter4        
            ], axis=1)

        intersec_filters=filters.prod(axis=1).astype (bool)


        basket=basket[intersec_filters].copy()

        
        basket['p_mid']=P0[basket.index]
        basket['AI']=basket.apply(lambda x: utils.calculate_accrued_interest (x.coupon, x.maturity, s_date), axis=1)
        basket['dirty']=basket['AI']+basket['p_mid']


        r1d = ffund.loc[date]
        basket.loc['1D']=pd.Series({
                                    'id_isin': '1D',
                                    'maturity': s_date+pd.offsets.Day(),
                                    'int_acc_dt': None,
                                    'issue_dt': s_date,
                                    'coupon': 0,
                                    'length': 1/act,
                                    'ttm': 1/act,
                                    'length _text': 'ID',
                                    'y_mid': r1d,
                                    'p_mid': 100*np.exp(-r1d*1/act/100),
                                    'AI': 0,
                                    'dirty': 100*np.exp(-r1d*1/act/100)
                                })


        density=basket.groupby(lambda x: round(basket. loc[x, 'ttm'])) .apply (lambda x: len(x))
        basket_density=basket.apply(lambda x: density.loc[round(x.ttm)], axis=1)

        # P array
        P0=basket['dirty']
        P_array= P0.values
        n_bonds= len(P)
        # term to maturity
        ttm=(basket['ttm']*act)
        ttm_array=ttm.values

        # payment dates
        pyd.loc['1D'] = basket.loc['1D', 'maturity']
        # see docs of 'construct_C'
        C=construct_C(ttm, pyd, basket, s_date)

        data=prep_data(P0, C, density, basket_density)

        # max time to maturity in days
        N=int(ttm.max())
        # generate kerne] matrix
        K=kr_model.generate_kernel_matrix(alpha, delta, N, N)

        #** fit KR model
        # KR ridge penalty term

        dict_fit=kr_model.KR(    C=data['C'], # cashflow matrix
                                B=data['P'], # price vector
                                ridge=ridge, # ridge hyper-parameter
                                inv_w=data['inv_w'], #p.ones like(data['inv w']I. # inverse of the weichting vector
                                K=K, # kernel matrix.
                                start_curve=None
        )

                            
        r=dict_fit['y_solved']
        g=dict_fit['g_solved']
        c=data['C']
        M=C.shape[0]
        B_fitted=C@g[:C.shape[1]]
        e=- (data['P']-B_fitted)*100
        e=e/(data['dur']*data['P']/100)
        np.abs(e).mean(), np.abs(e).std()
        e=pd.Series(index=basket.index, data=e)


        G.loc[date, np.arange(1 , len(g)+1)] = g
        RATES.loc[date, np.arange(1 ,len(g)+1)] = r
        for cusip in e.index:
            E.loc[date, cusip]= e.loc[cusip]

        if i%30==0:
            G=G.sort_index()
            G.to_csv('data/US_TREAS_CURVE/OTR/discount_factors.csv')
            RATES=RATES.sort_index()
            RATES.to_csv(f'data/US_TREAS_CURVE/OTR/rates.csv')
            E=E.sort_index()
            E.to_csv('data/US_TREAS_CURVE/OTR/errors.csv')

        i=i+1

       
    G=G.sort_index()
    G.to_csv('data/US_TREAS_CURVE/OTR/discount_factors.csv')
    RATES=RATES.sort_index()
    RATES.to_csv(f'data/US_TREAS_CURVE/OTR/rates.csv')
    E=E.sort_index()
    E.to_csv('data/US_TREAS_CURVE/OTR/errors.csv')


    import plotly.express as px
    sub_folder='OTR'
    E=pd.read_csv(f'data/US_TREAS_CURVE/{sub_folder}/errors.csv', index_col=0, parse_dates=True)
    AGG_E= E.T.groupby(lambda x: bonds.loc[x, 'length']if x !='1D' else 0).sum().T
    AGG_E['MAE']=AGG_E.abs().mean(axis=1)
    AGG_E['MAX']=AGG_E.abs().max(axis=1)
    fig = px.line(AGG_E[['MAE', 'MAX']])
    return fig