# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random

import os
import time


# setting the seed allows for reproducible results
SEED = 12345
RNG = np.random.default_rng(SEED)
EPOCHS = 100
VERBOSE = 0

random.seed(SEED)
# %%

##### download all csv and combine into one #####

work_path = "C:\\Users\\pat_h\\OneDrive\\Desktop\\public-repos\\NN-Portfolio-Optimizer\\portfolio_rfr_value\\1M"

portfolio_values_years_list = []

for dirpath, dirnames, filenames in os.walk(work_path):
    print(dirpath)

    for file in filenames:
        # print(file)
        portfolio_month_df = pd.read_csv(os.path.join(dirpath,file), index_col=[0])
        portfolio_values_years_list.append(portfolio_month_df)


all_portfolio_values_years_df = pd.concat(portfolio_values_years_list)

all_portfolio_values_years_df.index = pd.to_datetime(all_portfolio_values_years_df.index, format='%Y-%m-%d')

all_portfolio_values_years_df = all_portfolio_values_years_df.sort_values(by='Date')

all_portfolio_values_years_df.to_csv('portfolio_rfr_value_1M.csv')


# %%

# all_portfolio_values_years_df = pd.read_csv('portfolio_value_1M.csv', index_col=0)
# all_portfolio_values_years_df.index = pd.to_datetime(all_portfolio_values_years_df.index, format='%Y-%m-%d')
# all_portfolio_values_years_df = all_portfolio_values_years_df.sort_values(by='Date')


# %%

all_portfolio_values_years_df.head()

# %%

def normalize_prices(prices_df : pd.DataFrame):

    max_price = prices_df.max()
    min_price = prices_df.min()

    for price in range(len(prices_df)):

        prices_df.iloc[price] = (prices_df.iloc[price] - min_price) / (max_price - min_price)

    return prices_df


# %%

monthly_returns_list = []
grouped_month_df = pd.DataFrame()

for year_key in set(all_portfolio_values_years_df.index.year):
    
    grouped_month_df = pd.DataFrame()
   
    for month_key in set(all_portfolio_values_years_df.index.month):

        grouped_month_df = pd.DataFrame()

        grouped_month_df = all_portfolio_values_years_df.iloc[(all_portfolio_values_years_df.index.year==int(year_key)) & (all_portfolio_values_years_df.index.month==int(month_key))]
        
        if grouped_month_df.empty == False:

            portfolio_return = (grouped_month_df.iloc[-1] - grouped_month_df.iloc[0]) / grouped_month_df.iloc[0]

            monthly_returns_list.append(portfolio_return)



pd.DataFrame(monthly_returns_list[0]).transpose()
# %%

all_monthly_returns_df = pd.DataFrame()

for port_return in monthly_returns_list:
    
    if all_monthly_returns_df.empty == True:

        all_monthly_returns_df = pd.DataFrame(port_return).transpose()

    else:

        transposed_returns = pd.DataFrame(port_return).transpose()

        all_monthly_returns_df = pd.concat([all_monthly_returns_df, transposed_returns])


# %%
        
all_monthly_returns_df.head()

# %%

all_monthly_pct_sr = all_monthly_returns_df.sum(axis=1)

# %%
all_monthly_pct_sr.sum(axis=0)

# %%

initial_investment = 10000
monthly_investment = 100
total_monthly_investment = 0
portfolio_value = 0

for pct_return in all_monthly_pct_sr:
    
    portflio_value = initial_investment

    monthly_gainloss = (portflio_value + monthly_investment) * pct_return

    portfolio_value += monthly_gainloss

    total_monthly_investment += monthly_investment
   
   
print(portfolio_value)
print(total_monthly_investment)

# %%
