# %%
# https://github.com/shilewenuw/deep-learning-portfolio-optimization/blob/main/Model.py
# https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset/data

import numpy as np
import pandas as pd
import random

# setting the seed allows for reproducible results
SEED = 12345
RNG = np.random.default_rng(SEED)

random.seed(SEED)


# %%
        
##### get Nasdaq stocks #####

stocks_df = pd.read_csv('..\\Daten\\stocks_19900102_20191231.csv', index_col='Date')

random_stock_selection = RNG.choice(stocks_df.keys(), 10)

random_stocks_df = stocks_df[random_stock_selection]

random_stocks_df.index = pd.to_datetime(stocks_df.index, format="%Y-%m-%d")

random_stocks_df.head()

# %%

##### assign equal weights to all stock prices #####

equal_weight_stock_price_df = random_stocks_df.iloc[:,:] * (1 / len(random_stocks_df.keys()))

# save file
# equal_weight_stock_price_df.to_csv('equal_portfolio_1M.csv')

# %%

equal_weight_stock_price_df = equal_weight_stock_price_df.iloc[equal_weight_stock_price_df.index >= '1990-02-01'] 

# %%

monthly_returns_list = []
grouped_month_df = pd.DataFrame()

for year_key in set(equal_weight_stock_price_df.index.year):
    
    grouped_month_df = pd.DataFrame()
   
    for month_key in set(equal_weight_stock_price_df.index.month):

        grouped_month_df = pd.DataFrame()

        grouped_month_df = equal_weight_stock_price_df.iloc[(equal_weight_stock_price_df.index.year==int(year_key)) & (equal_weight_stock_price_df.index.month==int(month_key))]
        
        if grouped_month_df.empty == False:

            portfolio_return = (grouped_month_df.iloc[-1] - grouped_month_df.iloc[0]) / grouped_month_df.iloc[0]

            monthly_returns_list.append(portfolio_return)


# %%

all_monthly_returns_df = pd.DataFrame()

for port_return in monthly_returns_list:
    
    if all_monthly_returns_df.empty == True:

        all_monthly_returns_df = pd.DataFrame(port_return).transpose()

    else:

        transposed_returns = pd.DataFrame(port_return).transpose()

        all_monthly_returns_df = pd.concat([all_monthly_returns_df, transposed_returns])


