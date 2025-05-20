# pylint: disable-all
import os
import sys
main_path = 'e:\\ehz'
sys.path.append(main_path)
os.chdir(main_path)

from xbbg import blp
import utils
import pandas as pd 
import numpy as np
import time
from tqdm import tqdm

def update_fut_baskets(fut_type):

    fut_dic = utils.fut_dic
    fut_codes = utils.fut_codes
    fut_file= fut_dic[fut_type]

    fut_ticker =''.join([s.capitalize() for s in fut_file[:-1]])
    tickers_per_contract = fut_codes.apply(lambda x: fut_ticker + x + ' comdty')
    contracts = tickers_per_contract.index
    baskets = pd.DataFrame(columns =contracts, index=np.arange (70), dtype=str)
    baskets_filled = pd.read_csv(f'data/{fut_file}/baskets_{fut_file}.csv', index_col=0)

    for contract in baskets_filled.columns :
        baskets.loc[baskets_filled.index, contract] = baskets_filled.loc[:, contract].values 

    for contract, ticker in tickers_per_contract.iloc[-3:].items () :
        cusips = blp.bds (ticker,'FUT_DLVRBLE_BNDS_CUSIPS')
        if len(cusips)> 0:
            cusips = cusips['deliverable_bond_cusip_and_yellow_key']
            cusips_cut = [c[:-5] for c in cusips]
            baskets.loc[:len(cusips_cut)-1, contract] = np.array(cusips_cut)

    baskets = baskets. dropna(how='all')
    baskets = baskets.dropna(how='all', axis=1)

    output_dir = f'data/{fut_file}'
    if not os.path.isdir(output_dir):
        os.mkdir (output_dir)
    baskets.to_csv(f'data/{fut_file}/baskets_{fut_file}.csv')


def update_bonds():

    bonds= utils.get_bonds()

    fut_dic = utils.fut_dic

    all_cusips = bonds.index
    all_missing = []
    for fut_type in utils.fut_dic.keys():
        fut_file= fut_dic[fut_type]

        baskets = pd.read_csv(f'data/{fut_file}/baskets_{fut_file}.csv', index_col=0)
        cusips = np.unique(baskets.fillna('912810QC').values.flatten())
        missing = [c for c in cusips if c not in all_cusips] # type: ignore
        print(fut_type, len(missing))
        all_missing = all_missing + missing

    print('total: ' , len(all_missing))

    if len(all_missing) > 0:
        bonds_update = blp.bdp(
            [f'{cusip} Govt' for cusip in all_missing],
            ['id_isin', 'maturity','Int_acc_dt', "ISSUE_DT", "COUPON", "AMT_ISSUED" ]
        )

        bonds_update.index = bonds_update.index.to_series().apply(lambda x:x[:-5]) # type: ignore

        bonds_update.sort_values(by='issue_dt', inplace=True)
        bonds = pd.concat((bonds, bonds_update), axis=0)

        bonds['maturity'] = bonds['maturity'].apply(pd.to_datetime)
        bonds['int_acc_dt'] = bonds['int_acc_dt'].apply(pd.to_datetime)
        bonds['issue_dt'] = bonds['issue_dt'].apply(pd.to_datetime)
        
        output_dir = f'data/us_bonds'
        if not os.path.isdir(output_dir):
            os.mkdir (output_dir)
        bonds.to_csv('data/us_bonds/bonds_us_1980.csv')


def add_columns_to_tab (path, tickers, flds, tickers_mapping) :

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    tickers = [c for c in tickers if tickers_mapping.loc[c] not in df.columns]
    if len(tickers) == 0:
        return 0
    debut, end = df.index[0], df.index[-1]
    update = blp.bdh(tickers, flds=flds ,start_date=debut, end_date=end, Per='D', calendarCodeOverride='US')
    update.index = pd.to_datetime(update.index)
    update = update.droplevel(level=1, axis=1)
    update.columns = tickers_mapping.loc[update.columns]
    df = pd.concat((df , update), axis=1)
    df.to_csv(path)
    return 1


def update_bond_fld_tab (path, flds, bonds) : 

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    cusips = list(df.columns)
    tickers = [c + ' Govt' for c in cusips]
    tickers_mapping = pd.Series(index=tickers, data=cusips)

    debut = df.index[-1]
    end = pd.Timestamp.today()
    data = blp.bdh(tickers, flds=flds ,start_date=debut, end_date=end, Per='D', calendarCodeOverride='US')

    update = data
    update.index = pd.to_datetime(update.index)
    update = update.droplevel(level=1, axis=1)
    update.columns = tickers_mapping.loc[update.columns]
    df = pd.concat((df.iloc[:-1] , update), axis=0)
    df.to_csv(path)

    ## add missing bonds

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    cusips = [c for c in bonds.index if c not in df.columns]

    if len(cusips) > 0 :
        tickers = [c + ' Govt' for c in cusips]
        tickers_mapping = pd.Series(index=tickers, data=cusips)

        debut = bonds.loc[cusips]['issue_dt'].min() - pd.DateOffset(months=1)
        end = df.index[-1]
        update = blp.bdh(tickers, flds=flds ,start_date=debut, end_date=end, Per='D', calendarCodeOverride='US')
        update.index = pd.to_datetime(update.index)
        update = update.droplevel(level=1, axis=1)
        update.columns = tickers_mapping.loc[update.columns]

        df = pd.concat((df , update), axis=1)
        df.to_csv(path)


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


def update_mbs(tickers, flds, path, snap=False):

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    debut = df.index[-1] 
    end = pd.Timestamp.today() + pd.offsets.BDay(0, normalize=True)

    data = []
    for i, ticker in enumerate(tickers): 
        try : 
            col = blp.bdh(tickers=ticker,
                            flds=flds,
                            start_date=debut,
                            end_date=end,
                            Per='D',
                            calendarCodeOverride= 'US') 
            data.append(col)

        except ValueError :
            print (ticker, ' errored')

    update = pd.concat(data, axis=1)
    update = update.droplevel(level=1,axis=1)
    update.index=update.index.to_series().apply(pd.to_datetime)
    if snap : 
        # only set to true when loading price data 
        tba_symbol = tickers[0].split(' ')[0]
        update.columns = update.columns.map(lambda x: x.split(' ')[1]).map(lambda x: f'{tba_symbol} {x} N Mtge')
    df = pd.concat((df.iloc[:-1], update), axis=0)
    df.to_csv(path)

#

def main():

    print('updating cusips')

    for fut_type in utils.fut_dic.keys():
        update_fut_baskets(fut_type)
    update_bonds()

    bonds = utils.get_bonds()
    pyd = pd.DataFrame(columns = ['payment_date'])

    cusips=[c for c in bonds.index if c not in pyd.index]
    for cusip in tqdm(cusips) :
        payments=pd.date_range(bonds.loc[cusip,'int_acc_dt'], bonds.loc[cusip,'maturity']+pd.offsets.BDay(29), freq=pd.DateOffset (months=6), inclusive='right')
        payments=pd.DataFrame(index=[cusip for _ in range(len(payments))], data=payments,columns=['payment_date'])
        payments['payment_date']=payments['payment_date'].apply(correct_coupon_date)

        pyd=pd.concat( (pyd, payments)) 

    pyd.to_csv('data/us_bonds/estimated_payment_dates.csv')



    print('updating prices')

    ptype = 'mid'
    path = f'data/us_bonds/bonds_{ptype}_price.csv'
    flds = f'px_{ptype}'

    update_bond_fld_tab (path, flds, bonds) 

    ptype = 'bid'
    path = f'data/us_bonds/bonds_{ptype}_price.csv'
    flds = f'px_{ptype}'

    update_bond_fld_tab (path, flds, bonds) 


    print('updating yields duration gamma, ...')

    bonds = utils.get_bonds()
    P = utils.get_bond_prices()
    pyd = utils.get_bonds_pyd()

    Dirty = utils.get_bond_prices(ptype='dirty')
    Y = utils.get_bond_yields()
    D = utils.get_bond_duration()
    G = utils.get_fld_bonds('gamma')

    calculated_dates = Dirty.index

    missing = [c for c in P.columns if c not in Dirty.columns] 

    for c in missing:
        for tab in [Dirty, Y, D, G] :
            tab[c] = np.nan


    missing_dates = [date for date in P.index if date not in Dirty.index] 

    for date in missing_dates:
        for tab in [Dirty, Y, D, G] :
            tab.loc[date] = np.nan


    for date in tqdm(missing_dates):

        set_date = utils.add_business_day(date)

        basket = bonds.loc[P.loc[date].dropna().index]
        basket = basket[(basket['maturity'] > set_date) ]

        cusips = basket.index
        clean_prices = P.loc[date, cusips]


        AI = basket.apply(lambda x : 
                        utils.calculate_accrued_interest(x.coupon, x.maturity, max(set_date, x.issue_dt)),
                        axis=1
        )

        dirty_prices = clean_prices + AI
        Dirty.loc[date, cusips] = dirty_prices


        for cusip in cusips:
            maturity = bonds.loc[cusip, 'maturity']
            coupon = bonds.loc[cusip, 'coupon']
            set_date_cusip = max(set_date , bonds. loc[cusip, 'issue_dt'])
            dirty = dirty_prices.loc[cusip]
            bond_yield = utils.calculate_bond_yield(dirty, coupon, maturity, set_date_cusip, is_dirty=True)
            Y.loc[date, cusip] = bond_yield
            if np.isnan(bond_yield) : 
                print(cusip, dirty, coupon, maturity, set_date_cusip)
            bond_duration = utils.calculate_duration(dirty, coupon, maturity, set_date_cusip, bond_yield, mod=False)
            D.loc[date, cusip] = bond_duration

            bond_gamma = utils.calculate_gamma(coupon, maturity, set_date_cusip, bond_yield)
            G.loc[date, cusip] = bond_gamma

    Y.to_csv(f'data/us_bonds/bonds_mid_yield.csv')
    D.to_csv(f'data/us_bonds/durations_mid.csv')
    Dirty.to_csv(f'data/us_bonds/bonds_dirty_price.csv')
    G.to_csv(f'data/us_bonds/bonds_gamma.csv')

    print('update repo')


    ticker = 'FEDL01 Index'
    flds = 'px_last'
    path = f'data/repo/repo_ffund.csv'

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    debut = df.index[-1]
    end = pd.Timestamp.today()

    update = blp.bdh(ticker, flds=flds ,start_date=debut, end_date=end, Per='D', calendarCodeOverride='US')
    update.index = pd.to_datetime(update.index)
    update = update.droplevel(level=1, axis=1)
    update.columns = ['ffund']

    df = pd.concat((df.iloc[:-1], update))
    df.to_csv(path)

    print('updating mbs')

    print('update mtg rate')

    for tba_symbol in ['FNCL', 'G2SF'] :

        folder = f'data/mbs/{tba_symbol}'

        ticker = 'NMCMFR30 Index'
        flds = 'px_last'
        path = f'{folder}/mtg_rate.csv'

        df = pd.read_csv(path, index_col=0, parse_dates=True)
        debut = df.index[-1]
        end = pd.Timestamp.today()

        update = blp.bdh(ticker, flds=flds ,start_date=debut, end_date=end, Per='D', calendarCodeOverride='US')
        update.index = pd.to_datetime(update.index)
        update = update.droplevel(level=1, axis=1)
        update.columns = ['mtg_rate']

        df = pd.concat((df.iloc[:-1], update))
        df.to_csv(path)

        print('updating mbs prices')

        tickers = []
        for i in np.arange(1, 12, 0.25):
            tickers.append(f'{tba_symbol} {i} N Mtge')
        tickers_names = blp.bdp(tickers, ['name', 'issue_dt', 'px_mid']).dropna()
        tickers_names['coupon'] = tickers_names['name'].apply(lambda x : float(x.split(' ')[1]))
        tickers_names.sort_values(by = 'coupon', inplace=True)
        tickers = tickers_names.index
        tickers = list(tickers)

        path = f'{folder}/mbs.csv'
        flds = 'px_mid'
        update_mbs(tickers, flds, path, snap=True)

        print('updating mbs oas')
        tickers_names['oas ticker'] = tickers_names['coupon'].apply(lambda x : str(x).replace('.', ''))
        tickers_names['oas ticker'] = tickers_names['oas ticker'].apply(lambda x : f'{tba_symbol}0{x}S Index')
        oas_tickers = list(tickers_names['oas ticker'])

        flds = 'px_last'
        path = f'{folder}/mbs_oas.csv'
        update_mbs(oas_tickers, flds, path)