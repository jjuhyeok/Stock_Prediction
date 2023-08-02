import pandas as pd
from pykrx import stock, bond
import time
import numpy as np

# 공통 처리
def process_dataframe(fdf, rename_dict):
    fdf = fdf.reset_index()
    fdf = fdf.rename(columns=rename_dict)
    fdf['date'] = pd.to_datetime(fdf['date'])
    # fdf = fdf.set_index('date')
    return fdf

# OHLCV 데이터
def get_ohlcv(ticker, start_date, end_date):
    time.sleep(1)
    fdf = stock.get_market_ohlcv(start_date, end_date, ticker, adjusted=True)
    fdf = process_dataframe(fdf, {'날짜': 'date', '시가': 'Open', '고가': 'High', '저가': 'Low', '종가': 'Close', '거래량': 'Volume'})
    return fdf

# Fundamental 데이터
def get_fdm(ticker, start_date, end_date):
    # time.sleep(0.125)
    fdf = stock.get_market_fundamental_by_date(start_date, end_date, ticker)
    fdf = process_dataframe(fdf, {'날짜': 'date', 'BPS': 'BPS', 'PER': 'PER', 'PBR': 'PBR',  'EPS': 'EPS', 'DIV': 'DIV', 'DPS': 'DPS'})
    return fdf

# Trading Value 데이터
def get_value(ticker, start_date, end_date):
    # time.sleep(0.125)
    fdf = stock.get_market_trading_value_by_date(start_date, end_date, ticker, etf=True, etn=True, elw=True, detail=True)
    rename_dict = {
        '날짜': 'date',
        '금융투자': 'financial_investment',
        '보험': 'insurance',
        '투신': 'investment_trust',
        '사모': 'private_equity',
        '은행': 'bank',
        '기타금융': 'other_finance',
        '연기금': 'pension_fund',
        '기타법인': 'other_corporations',
        '개인': 'individuals',
        '외국인': 'foreigners',
        '기타외국인': 'other_foreigners',
        '전체': 'total'
    }
    fdf = process_dataframe(fdf, rename_dict)
    return fdf

# 시가총액, 상장주식수 데이터
def get_market_cap_info(ticker, start_date, end_date):
    # time.sleep(0.125)
    fdf = stock.get_market_cap_by_date(start_date, end_date, ticker)
    fdf = process_dataframe(fdf, {
        '날짜': 'date',
        '거래량': 'trading_volume',
        '거래대금': 'trading_value',
        '시가총액': 'market_cap',
        '상장주식수': 'shares_outstanding',
    })
    return fdf

# 외국인 보유량 및 한도 소진률
def get_exh_foreign_investor(ticker, start_date, end_date):
    # time.sleep(0.125)
    fdf = stock.get_exhaustion_rates_of_foreign_investment(start_date, end_date, ticker)
    fdf = process_dataframe(fdf, {
        '날짜': 'date',
        '상장주식수': 'shares_outstanding',
        '보유수량': 'holding_volume',
        '지분율': 'exhaustion_rate',
        '한도수량': 'limit_exhaustion_volume',
        '한도소진률': 'limit_exhaustion_rate',
        })
    return fdf


n=0
batch_size = 2000
start = '20230531'
end = '20230728'
df = pd.read_csv('train.csv')
df.columns = ['date', 'ticker', 'name', 'volume', 'open', 'high', 'low', 'close']
df.sort_values(by=['ticker', 'date'], ascending=True, inplace=True)
df.reset_index(drop=True, inplace=True)
df['ticker'] = df['ticker'].apply(lambda x: x[1:])
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df_ticker = df['ticker'].unique()
from tqdm import tqdm
for i, ticker in tqdm(enumerate(df_ticker)):
      try :
        ohlcv = get_ohlcv(ticker, start, end)
        fdm = get_fdm(ticker, start, end)
        value = get_value(ticker, start, end)
        mc_info = get_market_cap_info(ticker, start, end)
        exh_info = get_exh_foreign_investor(ticker, start, end)

        merged = pd.concat([ohlcv, fdm, value, mc_info, exh_info], axis=1)
        # merged = ohlcv
        merged['ticker'] = ticker


        if n == 0:
            total = merged
        else :
            total = pd.concat([total, merged], axis=0)

        n+=1
      except :
        n+=1
        print(n, 'error')
        continue

      # Print progress
      # print(f'Processed {i+1} out of {len(df_ticker)} tickers')

      if n % batch_size == 0 or i == len(df_ticker)-1:
          total.to_csv(f'ticker_data_{i+1-batch_size+1}_{i+1}.csv')


