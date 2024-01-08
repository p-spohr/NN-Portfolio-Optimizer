# %%

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras, data

import os

SEED = 123

# %%

work_path = "C:\\Users\\pat_h\\OneDrive\\Desktop\\Portfolio Optimizer\\data\\stocks"

# using list_files() from tf just adds extra complexity when I could use os.walk()
# work_files = data.Dataset.list_files(work_path + '\\*.csv', shuffle=None, seed=SEED)


# %%

stock_selection = ['AAPL', 'WMT', 'MFC']
stock_selection_paths = []
saved_tickers = []
invalid_tickers = []

for dirpath, dirnames, filenames in os.walk(work_path):
    # save tickers
    for file in filenames:
        ticker = file.split('.')[0]
        saved_tickers.append(ticker)

    # check ticker is available
    for stock in stock_selection:
        if stock not in saved_tickers:
            invalid_tickers.append(stock)

    if invalid_tickers:        
        print(f'{invalid_tickers} are invalid tickers or not in directory.')

    # save path to string for downloading
    for file in filenames:
        split_file = file.split('.')[0]
        if split_file in stock_selection:
            stock_selection_paths.append((split_file, os.path.join(dirpath, file)))



# %%

print(stock_selection_paths)

# %%

stock_dict = {}
date_ranges = ["2001-01-01","2001-12-31"]
stock_list = []

for ticker, path in stock_selection_paths:
    stock_dict[ticker] = pd.read_csv(path, usecols=['Date', 'Close'], index_col='Date') # index_col='Date'
    stock_dict[ticker].columns = [ticker]
    stock_dict[ticker] = stock_dict[ticker].query('`Date` > @date_ranges[0] and `Date` < @date_ranges[1] ')
    stock_list.append(stock_dict[ticker])


# %%

print(stock_dict['AAPL'].head())

print(stock_dict['MFC'].head())

print(stock_dict['WMT'].head())


# %%

# make sure to assert that all shapes for every df is the same or else that means a stock price wasn't recorded in that time range
# assert wmt_dated.shape == mfc_dated.shape == aapl_dated.shape


# %%

# passing a dictionary will create a df with multilevel keys set as the dictionary key
# new_df = pd.concat(stock_dict, axis=1)
# new_df.head()

# transforming the dict to a list removes the multilevel keys when concatenating the dfs
processed_df = pd.concat(stock_list, axis=1)
processed_df.head()


# %%
