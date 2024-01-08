# %%

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras, data

import time

import os

SEED = 123

# %%

# get the path of the csv stock data
work_path = os.path.dirname(os.path.abspath(__file__)) + '\stock_data\stocks'
print(work_path)

# %%

# input the desired stocks by listing the tickers
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
shape_list = []

for ticker, path in stock_selection_paths:
    stock_dict[ticker] = pd.read_csv(path, usecols=['Date', 'Close'], index_col='Date') # set the index column to the date
    stock_dict[ticker].columns = [ticker]
    stock_dict[ticker] = stock_dict[ticker].query('`Date` > @date_ranges[0] and `Date` < @date_ranges[1] ')
    stock_list.append(stock_dict[ticker])
    shape_list.append(stock_dict[ticker].shape)

# %%
    
print(shape_list)
print(set(shape_list))

# %%
    
assert len(set(shape_list)) == 1, "The stocks have incongruent dates. Pick different stocks or change the date range."

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

##### find list of stocks with suitable dates (shapes) #####

start_time = time.time()

STOCK_SHAPE = (247,1)
DATE_RANGES = ["2001-01-01","2001-12-31"]

stock_dict = {}
stock_selection_paths = []
saved_tickers = []
invalid_tickers = []
suitable_stocks = 0

for dirpath, dirnames, filenames in os.walk(work_path):

    for file in filenames:
        path = os.path.join(dirpath, file)

        ticker = file.split('.')[0]
        
        stock_dict[ticker] = pd.read_csv(path, usecols=['Date', 'Close'], index_col='Date') # set the index column to the date
        stock_dict[ticker].columns = [ticker]
        stock_dict[ticker] = stock_dict[ticker].query('`Date` > @DATE_RANGES[0] and `Date` < @DATE_RANGES[1] ')

        if STOCK_SHAPE == stock_dict[ticker].shape:
            stock_count += 1
            # saved_tickers.append(ticker)
            # stock_list.append(stock_dict[ticker])
        else:
            stock_dict.pop(ticker)

        # saved_tickers.append(ticker)

    # save path to string for downloading
    # for file in filenames:
    #     split_file = file.split('.')[0]
    #     if split_file in stock_selection:
    #         stock_selection_paths.append((split_file, os.path.join(dirpath, file)))


end_time = time.time()

print(f'Total Runtime: {round(end_time - start_time, 3)} seconds') # Total Runtime: 379.671 seconds
print(suitable_stocks) # Suitable Stocks: 2137



# %%

##### find list of stocks with suitable dates (shapes) #####

start_time = time.time()

DATE_RANGES = ["1990-01-01","2020-01-01"]

stock_dict = {}
checker = 0

for dirpath, dirnames, filenames in os.walk(work_path):

    for file in filenames:
        path = os.path.join(dirpath, file)

        ticker = file.split('.')[0]
        
        stock_dict[ticker] = pd.read_csv(path, usecols=['Date', 'Close'], index_col='Date') # set the index column to the date
        stock_dict[ticker].columns = [ticker]
        stock_dict[ticker] = stock_dict[ticker].query('`Date` > @DATE_RANGES[0] and `Date` < @DATE_RANGES[1] ')

        if stock_dict[ticker].shape[0] < 1000:
            stock_dict.pop(ticker)

        checker +=1
        if checker % 100 == 0:
            check_time = time.time()
            print(f'100 Stocks Finished ... {round(check_time - start_time, 3)} seconds')

end_time = time.time()

print(f'Total Runtime: {round(end_time - start_time, 3)} seconds') # Total Runtime: 316.18 seconds

# %%

print(len(stock_dict.keys()))

# %%

stock_shapes = []

for key, value in stock_dict.items():
    stock_shapes.append(value.shape)

# %%

stock_shapes_df = pd.DataFrame(stock_shapes)
stock_shapes_df.head()

# %%

stock_shapes_df[0].value_counts() 
# Total Observations: 7559
# Total Stocks: 813

# %%

##### find list of stocks with suitable dates (shapes) #####

start_time = time.time()

work_path = os.path.dirname(os.path.abspath(__file__)) + '\stock_data\stocks'

# from 1990-01-02 to 2019-12-31
STOCK_SHAPE = (7559,1)
DATE_RANGES = ["1990-01-02","2019-12-31"]

stock_dict = {}
suitable_stocks = 0
deleted_stocks = 0

for dirpath, dirnames, filenames in os.walk(work_path):

    for file in filenames:
        path = os.path.join(dirpath, file)

        ticker = file.split('.')[0]
        
        stock_dict[ticker] = pd.read_csv(path, usecols=['Date', 'Close'], index_col='Date') # set the index column to the date
        stock_dict[ticker].columns = [ticker]
        stock_dict[ticker] = stock_dict[ticker].query('`Date` > @DATE_RANGES[0] and `Date` < @DATE_RANGES[1] ')

        if STOCK_SHAPE == stock_dict[ticker].shape:
            suitable_stocks += 1
            check_time = time.time()
            if suitable_stocks % 100 == 0:
                print(f'Suitable: {suitable_stocks} | Deleted: {deleted_stocks} ... {round(check_time - start_time, 3)} seconds')
            # print('Found one! ', stock_dict[ticker].shape)
        else:
            stock_dict.pop(ticker)
            deleted_stocks += 1
            # print(f'{ticker} deleted!')

end_time = time.time()

print(f'Total Runtime: {round(end_time - start_time, 3)} seconds') # Total Runtime: 349.277 seconds
print(f'Total Stocks: {suitable_stocks}') # Suitable Stocks: 813


# %%

col_names = []

for key, value in stock_dict.items():
    col_names.append(key)

print(col_names)

# %%



stocks_df = pd.DataFrame.from_dict(stock_dict, orient='index', columns=col_names)

# %%


# %%

stocks_df = pd.DataFrame()

for key, value in stock_dict.items():
    stocks_df[key] = value

# %%

stocks_df.head()

# %%

stocks_df.shape

# %%

stocks_df.tail()
# %%

stocks_df.to_csv('stocks_df_19900102_20191230')
# %%
