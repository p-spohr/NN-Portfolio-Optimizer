# %%

import pandas as pd
import numpy as np
import random

import time

import os

SEED = 123
random.seed(SEED)

# %%

ust_rfr = pd.read_csv('UST_10_rfr.csv', index_col=0, names=['Date', 'rfr'], header=0)
ust_rfr.index = pd.to_datetime(ust_rfr.index, format='%Y-%m-%d')

ust_rfr.head()


# %%

gspc_df = pd.read_csv('gspc_19900102_20191231.csv', index_col=0, names=['Date', 'gspc'], header=0)
gspc_df.index = pd.to_datetime(gspc_df.index, format='%Y-%m-%d')

ust_rfr_new = ust_rfr.loc[gspc_df.index]

# %%

for i in range(len(ust_rfr_new['rfr'])):

    index_num = i
    index_num_bef = i-1
    index_num_aft = i+1

    # print(index_num)

    if ust_rfr_new['rfr'].iloc[index_num] == '.':

        print(f'Change at index {index_num}')

        rate_bef = float(ust_rfr_new['rfr'].iloc[index_num_bef])
        rate_aft = float(ust_rfr_new['rfr'].iloc[index_num_aft])

        # print(rate_bef, type(rate_bef))
        # print(rate_aft, type(rate_aft))
        
        new_rate = (rate_bef + rate_aft) / 2

        ust_rfr_new['rfr'].iloc[index_num] = new_rate

        # print(new_rate)

    

# %%

ust_rfr_new['rfr'] = ust_rfr_new['rfr'].astype('float64')

# %%

ust_rfr_new['rfr'] = ust_rfr_new['rfr'] / 100

# %%

ust_rfr_new.to_csv('UST_10_rfr_update.csv')


# %%

