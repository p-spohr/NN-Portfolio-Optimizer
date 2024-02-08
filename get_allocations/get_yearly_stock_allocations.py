# %%

import numpy as np
import pandas as pd


import os
import random

SEED = 12345
random.seed(SEED)



# %%

##### create df for every stock that contains all years and all months #####

work_path = "..\\allocations_rfr\\1M"

for dirpath, dirnames, filenames in os.walk(work_path):
   
    print(dirpath)
    
    # store each stock available
    stock_tickers = []

    # store total years avaialbe
    year_index = []

    # store month columns
    month_cols = []

    # get ticker of each stock
    for file in filenames[0:1]:
        
        current_stock_df = pd.read_csv(os.path.join(dirpath, file), index_col=0)

        stock_tickers = list(current_stock_df.index)

        month_cols = list(current_stock_df.columns)

    for file in filenames:

        year_index.append(file.split('_')[0])

    # create df of yearly allocations for each stock
    for ticker in stock_tickers:

        stock_yearly_allocations_df = pd.DataFrame(np.zeros([len(year_index), len(month_cols)]), columns = month_cols, index = year_index)

        for file in filenames:
            
            current_stock_df = pd.read_csv(os.path.join(dirpath, file), index_col=0)

            current_year = file.split('_')[0]

            stock_yearly_allocations_df.loc[current_year] = current_stock_df.loc[ticker]


        stock_yearly_allocations_df.to_csv(f"..\\allocations_rfr\\stock_yearly\\{ticker}_all_allocations.csv")

            
# %%

