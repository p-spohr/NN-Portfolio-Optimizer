# %%
# https://github.com/shilewenuw/deep-learning-portfolio-optimization/blob/main/Model.py
# https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset/data

import numpy as np
import pandas as pd
import random

import tensorflow as tf
from tensorflow import keras
from keras.layers import LSTM, Flatten, Dense
from keras.models import Sequential
import keras.backend as K

import os
import time


# setting the seed allows for reproducible results
SEED = 12345
RNG = np.random.default_rng(SEED)
EPOCHS = 100
VERBOSE = 0

tf.random.set_seed(SEED)


# %%

##### put allocations into dict by year and month #####

work_path = "C:\\Users\\pat_h\\OneDrive\\Desktop\\public-repos\\NN-Portfolio-Optimizer\\allocations_rfr\\1M"
allocations_years_dict = {}

for dirpath, dirnames, filenames in os.walk(work_path):
    
    allocations_df = 0
    print(dirpath)

    for file in filenames:
        allocations_df = pd.read_csv(os.path.join(dirpath, file), index_col=0)
        dict_key = int(file.split('_')[0])
        allocations_years_dict[dict_key] = allocations_df
    
# %%
        
##### get Nasdaq stocks #####

stocks_df = pd.read_csv('stocks_19900102_20191231.csv', index_col='Date')

random_stock_selection = RNG.choice(stocks_df.keys(), 10)

random_stocks_df = stocks_df[random_stock_selection]

random_stocks_df.index = pd.to_datetime(stocks_df.index, format="%Y-%m-%d")

random_stocks_df.head()

# %%

##### create dictionary with keys as years and values as monthly prices #####

start_time = time.time()

stocks_years_dict = {}
stock_month_list = []

year_range = range(1990,2020)
month_range = range(1,13)

for year in year_range:
    
    stock_month_list = []
    stocks_year_df = random_stocks_df.iloc[random_stocks_df.index.year==year]

    for month in month_range:

        stocks_quarter_df = stocks_year_df.iloc[(stocks_year_df.index.month==month)]

        stock_month_list.append(stocks_quarter_df)

    stocks_years_dict[year] = stock_month_list

   
end_time = time.time()

print(f'Total Runtime: {round(end_time - start_time, 3)} seconds')
print(stocks_years_dict.keys())
print(stocks_years_dict[2000][0].shape)

# %%

stocks_years_dict[2000][1]

# %%

allocations_years_dict[2000]

# %%

##### combine prices and allocations for easier unpacking later #####

set_portfolio_prices_list = []
set_portfolio_allocations_list = []
all_prices_allocations = []

for year_dict_1, month_prices in stocks_years_dict.items():

    set_portfolio_prices_list = []
    set_portfolio_allocations_list = []

    # set monthly prices in a list
    for prices in month_prices:

        set_portfolio_prices_list.append(prices)

    # set monthly allocations in a list
    for year_dict_2, month_allocations in allocations_years_dict.items():
        
        if year_dict_1 == year_dict_2:
            
            for month in month_allocations.keys():

                set_portfolio_allocations_list.append(month_allocations[month])

    # all prices and allocations are tuple pairs for the whole 120 quarters
    all_prices_allocations.extend(list(zip(set_portfolio_prices_list, set_portfolio_allocations_list)))

print(len(all_prices_allocations))


# %%

##### offset the list of prices and allocations, so the first quarter allocations will be used for second quarter prices #####

all_prices_offset = [prices for prices, allocations in all_prices_allocations[1:]]

all_allocations_offset = [allocations for prices, allocations in all_prices_allocations[0:-1]]

all_prices_allocations_offset = list(zip(all_prices_offset, all_allocations_offset))

print(len(all_prices_allocations_offset))

# %%

all_prices_allocations_offset_copy = all_prices_allocations_offset


# %%

##### calculate returns for each month using optimal weights #####

opt_portfolio_month_returns_list = []
stocks_month_returns_list = []
stock_month_returns_df = pd.DataFrame()

for month_prices_df, month_allocations_df in all_prices_allocations_offset_copy:
    
    # reset data frame
    stock_month_returns_df = pd.DataFrame()

    # rename series to shorter name 'weight'
    renamed_month_allocations_df = month_allocations_df.rename('weight')

    # multiply the stock price by the optimal weight
    for company in month_prices_df.keys():
        
        # if df is empty then assign new df to empty df
        if stock_month_returns_df.empty == True:
            
            returns_sr = month_prices_df.loc[:, company] * renamed_month_allocations_df.loc[company]

            returns_df = pd.DataFrame(returns_sr)

            stock_month_returns_df = returns_df
        
        # if df is not empty then merge the df for more efficent results
        else:
            
            returns_sr = month_prices_df.loc[:, company] * renamed_month_allocations_df.loc[company]

            returns_df = pd.DataFrame(returns_sr)
           
            stock_month_returns_df = stock_month_returns_df.merge(returns_df, on='Date')

    save_year = stock_month_returns_df.index.year[0]
    save_month = stock_month_returns_df.index.month[0]


    stock_month_returns_df.to_csv(os.path.join('portfolio_rfr_value', '1M', f'{save_year}_{save_month}.csv'))
  
   