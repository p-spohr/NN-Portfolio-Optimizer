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

work_path = "C:\\Users\\pat_h\\OneDrive\\Desktop\\public-repos\\NN-Portfolio-Optimizer\\allocations\\1M"
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

allocations_years_dict[2000]


# %%

##### offset the list of prices and allocations, so the first quarter allocations will be used for second quarter prices #####

all_prices_offset = [prices for prices, allocations in all_prices_allocations[1:]]

all_allocations_offset = [allocations for prices, allocations in all_prices_allocations[0:-1]]

all_prices_allocations_offset = list(zip(all_prices_offset, all_allocations_offset))

print(len(all_prices_allocations_offset))

# %%

all_prices_allocations_offset[0][0]

# %%

all_prices_allocations_offset[0][1] #.rename('weight')

renamed = all_prices_allocations_offset[0][1].rename('weight')
print(renamed)
print(pd.DataFrame(renamed))

# %%

all_prices_allocations_offset[-1][0].loc[:, 'PEP'] * all_prices_allocations_offset[-1][1].loc['PEP']

# %%

all_prices_allocations_offset[0][0].loc[:,'PEP'] * renamed.loc['PEP']

# %%

all_prices_allocations_offset[0][0].keys()


# %%

all_prices_allocations_offset_copy = all_prices_allocations_offset


# %%

##### calculate returns for each month using optimal weights #####

opt_portfolio_month_returns_list = []
stocks_month_returns_list = []
stock_month_returns_df = pd.DataFrame()

for month_prices_df, month_allocations_df in all_prices_allocations_offset_copy:
    
    stock_month_returns_df = pd.DataFrame()
    renamed_month_allocations_df = month_allocations_df.rename('weight')

    # multiply the stock price by the optimal weight
    for company in month_prices_df.keys():
        
        if stock_month_returns_df.empty == True:

            # print(month_prices_df.loc[:, company] * month_allocations_df.loc[company])
            
            returns_sr = month_prices_df.loc[:, company] * renamed_month_allocations_df.loc[company]

            returns_df = pd.DataFrame(returns_sr)

            stock_month_returns_df = returns_df
            
        else:
            
            returns_sr = month_prices_df.loc[:, company] * renamed_month_allocations_df.loc[company]

            returns_df = pd.DataFrame(returns_sr)
           
            stock_month_returns_df = stock_month_returns_df.merge(returns_df, on='Date')

        # print(stock_month_returns_df)

    save_year = stock_month_returns_df.index.year[0]
    save_month = stock_month_returns_df.index.month[0]

    # print(save_year)
    # print(save_month)

    stock_month_returns_df.to_csv(os.path.join('portfolio_value', '1M', f'{save_year}_{save_month}.csv'))
    # print(stock_month_returns_df)
        
    # # sum values for value of portfolio at time
    # month_prices_df = month_prices_df.sum(axis=1)

    # # add returns for each month to list
    # opt_portfolio_month_returns_list.append((month_prices_df.iloc[-1] - month_prices_df.iloc[0]) / month_prices_df.iloc[0])

# print(len(opt_portfolio_month_returns_list))
   
# %%

opt_portfolio_month_returns_df = pd.DataFrame(opt_portfolio_month_returns_list, columns=['returns'])
opt_portfolio_month_returns_df.head()

# %%

opt_portfolio_month_returns_df.plot()

# %%

opt_portfolio_month_returns_df.sum(axis=0)

# %%

##### calculate returns for each month using equal weights #####

equal_portfolio_month_returns_list = []

for month_prices_df, month_allocations_df in all_prices_allocations_offset:
    
    # multiply the stock price by equal weight
    for column in month_prices_df.keys():
        
        month_prices_df.loc[:, column] = month_prices_df.loc[:, column] * (1 / len(month_prices_df.keys()))
        print(month_prices_df.loc[:, column])
        break
    break
    # sum values for value of portfolio at time
    month_prices_df = month_prices_df.sum(axis=1)
    
    # add returns for each month to list
    equal_portfolio_month_returns_list.append((month_prices_df.iloc[-1] - month_prices_df.iloc[0]) / month_prices_df.iloc[0])

print(len(equal_portfolio_month_returns_list))

# %%

equal_portfolio_month_returns_df = pd.DataFrame(equal_portfolio_month_returns_list, columns=['returns'])
equal_portfolio_month_returns_df.head()

# %%

equal_portfolio_month_returns_df.plot()

# %%

equal_portfolio_month_returns_df.sum(axis=0)

# %%

for month_prices_df, month_allocations_df in all_prices_allocations_offset:
    
    # multiply the stock price by equal weight
    for column in month_prices_df.keys():
        
        month_prices_df.loc[:, column] = month_prices_df.loc[:, column] * (1 / len(month_prices_df.keys()))
        print(month_prices_df.loc[:,'GOLD'])
        break
    break
# %%

for month_prices_df, month_allocations_df in all_prices_allocations_offset:
    
    # multiply the stock price by equal weight
    for column in month_prices_df.keys():
        
        month_prices_df.loc[:, column] = month_prices_df.loc[:, column] * month_allocations_df.loc[column].loc['Allocation Weight']
        print(month_prices_df.loc[:,column])
        print(month_allocations_df.loc[column].loc['Allocation Weight'])
        month_prices_df.loc[:, column] = 0
        print(month_prices_df.loc[:,column])
        break
    break
# %%

stocks_years_dict[1990][1].loc[:,'PEP']
# %%
