# %%

import pandas as pd
import numpy as np

import random

SEED = 12345
random.seed(SEED)

RNG = np.random.default_rng(SEED)

# %%

stocks_df = pd.read_csv('stocks_df_19900102_20191231', index_col='Date')

# %%

random_stock_selection = RNG.choice(stocks_df.keys(), 10)

# %%

random_stocks_df = stocks_df[random_stock_selection]

# %%

random_stocks_df.index = pd.to_datetime(stocks_df.index, format="%Y-%m-%d")

# %%

random_stocks_portfolio_df = random_stocks_df.sum(axis=1)


random_stocks_portfolio_year_df = random_stocks_portfolio_df.iloc[random_stocks_portfolio_df.index.year==2000]

#
