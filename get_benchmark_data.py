# %%

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

import random
import time
import os

SEED = 123
random.seed(SEED)

# %%

##### get benchmark returns #####

# SPY is an S&P 500 ETF that is weighted by market cap
# RSP is an equal weight S&P 500 ETF

# from 1990-01-02 to 2019-12-31
STOCK_SHAPE = (7559,1)
DATE_RANGES = ["1990-01-02","2020-01-01"]

spy_df = yf.download('SPY',start=DATE_RANGES[0], end=DATE_RANGES[1])

# %%

spy_df['Close'].to_csv('spy_df_19930129_20191231')

# %%

spy_df.index

# %%

stocks_df = pd.read_csv('stocks_df_19900102_20191230', index_col='Date')
stocks_df.index


# %%

##### get benchmark returns #####

# SPY is an S&P 500 ETF that is weighted by market cap
# RSP is an equal weight S&P 500 ETF

# from 1990-01-02 to 2019-12-31
STOCK_SHAPE = (7559,1)
DATE_RANGES = ["1990-01-01","2020-01-01"]

rsp_df = yf.download('RSP',start=DATE_RANGES[0], end=DATE_RANGES[1])

# %%

rsp_df.head()


# %%

rsp_df.tail()

# %%

rsp_df['Close'].to_csv('rsp_df_20030501_20191231')

# %%

rsp_df.index

# %%

spy_df = pd.read_csv('spy_df_19930129_20191231', index_col='Date')
spy_df.index




# %%

fig, ax = plt.subplots(1,1)

plt.plot(spy_df)
plt.xlabel(np.arange('1993','2019', dtype='datetime64[Y]'))

# %%

np.arange('1993','2019', dtype='datetime64[Y]')
# %%

spy_df.groupby(pd.Grouper(freq='M'))
# %%

spy_df.plot()
# %%

spy_df_datetime = pd.to_datetime(spy_df.index, format="%Y-%m-%d")

# %%

spy_df_datetime


# %%

new_spy_df = spy_df

new_spy_df.index = spy_df_datetime

# %%

new_spy_df.tail()
# %%

new_spy_df.index
# %%

testing_df = new_spy_df.groupby(pd.Grouper(freq='2W')).mean()


# %%

testing_df.head(n=30)
# %%

testing_df.shape
# %%

testing_df.plot()
# %%

testing_df_max = new_spy_df.groupby(pd.Grouper(freq='Q')).max()
testing_df_min = new_spy_df.groupby(pd.Grouper(freq='Q')).min()


# %%

fig, ax = plt.subplots(1,1)

plt.plot(testing_df_max)
plt.plot(testing_df_min)

plt.show()

# %%

##### get benchmark returns #####

# SPY is an S&P 500 ETF that is weighted by market cap
# RSP is an equal weight S&P 500 ETF

# from 1990-01-02 to 2019-12-31
STOCK_SHAPE = (7559,1)
DATE_RANGES = ["1990-01-01","2020-01-01"]

gspc_df = yf.download('^GSPC',start=DATE_RANGES[0], end=DATE_RANGES[1])


# %%

gspc_df.index
type(gspc_df)
gspc_df.head()

# %%

gspc_df['Close'].to_csv('gspc_19900102_20191231.csv')


# %%

gspc_df = gspc_df['Close']

# %%

testing_df = gspc_df.groupby(pd.Grouper(freq='Y')).mean()

testing_df.plot()

# %%

(testing_df[-1] - testing_df[0]) / testing_df[0]
# %%
