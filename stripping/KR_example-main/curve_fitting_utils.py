# pylint: disable-all
main_path = 'e:\\ehz'
import sys
import os
sys.path.append(main_path)
sys.path.append('e:\\stripping\KR_example-main\src')
os.chdir(main_path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xbbg import blp
from utils import *
from fut_pricer import *

import kr_model
import kr_utils

import plotly.express as px

def p_plot (tab,  x,  y, c) :    
    hover_data =  {}    
    for col in tab. columns:
        hover_data[col] = True
    px.scatter(tab,  y=y,  x = x,  color=c,  hover_data=hover_data)    
    plt.show()


def construct_C (ttm,  pyd,  bonds,  date):     
    # Matrix defining al1 future flows:     
    # [i,j] = flow at time tj for bond i
    max_ttm = ttm.max()    
    n_bonds = len(ttm)
    C = np.zeros((n_bonds, int(max_ttm)))
    for i  in range(n_bonds):        
        cusip = ttm.index[i]
        coupon = bonds.loc[cusip, 'coupon']
        xij = pyd.loc[[cusip]].apply(lambda x:(pd.to_datetime(x)-date).days).values         
        xij = xij[xij>0]
        C[i, xij-1] = coupon/2
        T = xij[-1]
        C[i, T-1] +=100     
        return C
    