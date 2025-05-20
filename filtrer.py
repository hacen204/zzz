## filter.py

# pylint: disable=all
import os
import sys

main_path = 'e:\\ehz'
sys.path.append(main_path)
os.chdir(main_path)

import pandas as pd
import numpy as np
import plotly.express as px
import utils

from tqdm import tqdm
import swaplib as swlb

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(10, 5)})

bonds = utils.get_bonds()

P = utils.get_bond_prices(ptype='mid')
Dirty = utils.get_bond_prices(ptype='dirty')
Y = utils.get_bond_yields()
D = utils.get_bond_duration(ptype = 'mid')

roll_out_dates = utils.fut_codes.index.to_series().apply(
    lambda x: utils.add_business_day(pd.to_datetime(x), -4))
roll_out_dates.to_csv('data/us_bonds/roll_dates.csv')
roll_in_dates = roll_out_dates.shift(1)

zero_coupon_rates = utils.get_zc_rates(source='bloom')


for fut_type in utils.fut_dic:
    fut_file = utils.fut_dic[fut_type]

    first_contract = utils.start_dic[fut_type]
    contracts = utils.fut_codes.loc[first_contract:].index[:-1]
    dates = roll_in_dates.loc[contracts]
    contracts = list(contracts) + ['Jun 2025', 'Sep 2025', 'Dec 2025']

    CTDs = pd.DataFrame(index=contracts[:-3], columns=np.arange(4))
    for contract in tqdm(contracts[:-3]):

        date = roll_in_dates.loc[contract]
        set_day = utils.add_business_day(date)

        sub_contracts = contracts[contracts.index(contract):][:4]

        for contract_position, sub_contract in enumerate(sub_contracts):
            first_delivery = pd.to_datetime(sub_contract)
            try : 
                basket = utils.get_accurate_basket (bonds, sub_contract, fut_type)
            except : 
                y_lower, m_lower, y_upper, m_upper, y_i, m_i = utils.deliverable_rules[fut_type]
                basket = utils.get_basket(bonds, first_delivery, y_lower, m_lower, y_upper, m_upper, y_i, m_i)

            P_date = P.loc[date, basket.index].dropna()
            basket = basket.loc[P_date.index]

            if len(basket) == 0 : 
                CTDs.loc[contract, contract_position] = np.nan
                continue

            long = utils.tenors_dic[fut_type] > 6
            CF = utils.get_CF(basket.index, basket, first_delivery, long)
            dt = (first_delivery - date).days
            term_rp = ((zero_coupon_rates.loc[date, dt])**(-365/dt) - 1)*100
            PF_date = utils.P_to_PF(P_date, P_date.index, bonds, set_day, first_delivery, term_rp).dropna()
            PF_conv = (PF_date/CF).dropna()
            ctd = PF_conv.idxmin()
            CTDs.loc[contract, contract_position] = ctd

    CTDs.to_csv(f'data/filters/{fut_file}_ctd.csv')


n_otrs = 3
for tenor in [2,3,5,7,10,20,30]:
    basket = bonds.loc[P.columns]
    basket = basket[basket.length == tenor]
    basket = basket[basket.issue_dt > P.index[0]]
    #oldest = basket['issue_dt'].idxmin()
    #auction_dates = P[basket.index].apply(lambda x: x.dropna().index[0]) 
    #start = P[oldest].dropna().index[0]
    OTRs = pd.DataFrame(index=utils.calendar, columns=np.arange(n_otrs))

    basket.sort_values(by='issue_dt', inplace=True)
    for date in tqdm(OTRs.index):
        live_bonds = P.loc[date, basket.index].dropna()
        not_old = basket.loc[live_bonds.index, 'issue_dt'].apply(lambda x: (date-x).days <365)
        live_bonds = live_bonds[not_old]
        live_bonds = live_bonds.iloc[-n_otrs:]
        live_bonds = live_bonds.index.to_series()
        n = len(live_bonds)
        live_bonds.index = np.arange(n-1, -1, -1)
        OTRs.loc[date] = live_bonds

    OTRs.columns = [f'CT_{tenor}_{c}' for c in OTRs.columns]
    OTRs.to_csv(f'data/filters/{tenor}_otr.csv')