# pylint: disable-all
main_path = 'e:\\ehz'
import sys
import os
sys.path.append(main_path)
sys.path.append('e:\\stripping\KR_example-main\src')
os.chdir(main_path)

from xbbg import blp
import utils
import pandas as pd 
import numpy as np
from tqdm import tqdm


def update_tab(path,tickers,flds, tickers_mapping,debut=None, last_date=None, cdr='US'):

    # t]function updates data up to yesterday closing


    # updating some series stored in path
    print(f"updating{path}")

    today= pd.Timestamp.today()+pd.offsets.BDay(0,normalize=True)
    df=pd.read_csv(path,index_col=0,parse_dates=True)

    # last correct date in the data
    # if none sets to day_before_yesterday
    if last_date is not None:
        #last_date=min(last_date,today-pd.offsets.BDay())
        df=df.loc[:last_date]
    else:
        last_date=df.index[-2]
        df=df.loc[:last_date]


    debut=df.index[-1]+pd.offsets.Day()
    end= today - pd.offsets.Day()

    if debut>end:
        print ('nothing to update')
        return None
    
    update=blp.bdh(tickers=tickers,flds=flds,
                    start_date=debut,end_date=end,Per='D',calendarCodeOverride=cdr) # type: ignore

    update=update.droplevel(level=1,axis=1)
    update.columns=tickers_mapping.loc[update.columns]
    df= pd.concat((df,update),axis=0)
    df.index=df.index.to_series().apply(pd.to_datetime) # type: ignore
    df.to_csv(path)


def update_bond_payment_dates():

    try : 
        pyd=pd.read_csv('data/us_bonds/estimated_payment_dates.csv' , index_col=0,
                        parse_dates=['payment_date'])
    except FileNotFoundError: 
        pyd = pd.DataFrame(columns = ['payment_date'], dtype=pd.Timestamp)

    def correct_coupon_date(x):
        t1=pd.offsets.BusinessDay(1)
        if x.day>20:
            # get × to month end
            x = x -t1 + pd.offsets.MonthEnd()
            return utils.next_business_day(x)

        else :
            # get × to day 15
            x = x.replace(day=15)
            return utils.next_business_day(x)
        
    bonds=utils.get_bonds()
    cusips=[c for c in bonds.index if c not in pyd.index]
    for cusip in tqdm(cusips) :
        payments=pd.date_range(bonds.loc[cusip,'int_acc_dt'], bonds.loc[cusip,'maturity']+pd.offsets.BDay(29), freq=pd.DateOffset (months=6), closed='right')
        payments=pd.DataFrame(index=[cusip for _ in range(len(payments))], data=payments,columns=['payment_date'])
        payments['payment_date']=payments['payment_date'].apply(correct_coupon_date)

        pyd=pd.concat( (pyd, payments)) 

    pyd.to_csv('data/us_bonds/estimated_payment_dates.csv')