# %%

import pandas as pd
import numpy as np

import random

SEED = 12345
random.seed(SEED)

RNG = np.random.default_rng(SEED)

# %%

stocks_df = pd.read_csv('stocks_df_19900102_20191231', index_col='Date')
stocks_df.shape

# %%

stocks_df.head()

# %%

stocks_df.tail()

# %%

random_stock_selection = RNG.choice(stocks_df.keys(), 10)

# %%

print(random_stock_selection)

# %%

random_stocks_df = stocks_df[random_stock_selection]

# %%

random_stocks_df.index = pd.to_datetime(stocks_df.index, format="%Y-%m-%d")
print(type(random_stocks_df.index))
# %%

random_stocks_portfolio_df = random_stocks_df.sum(axis=1)

# %%

random_stocks_df.groupby(pd.Grouper(freq='Q')).mean().plot()

# %%

random_stocks_portfolio_df.groupby(pd.Grouper(freq='Q')).mean().plot()
# %%

print(f'{100 *(random_stocks_portfolio_df.iloc[-1] - random_stocks_portfolio_df.iloc[0]) / random_stocks_portfolio_df.iloc[0]} %')

# %%

print(random_stocks_portfolio_df.iloc[0])
print(random_stocks_portfolio_df.iloc[-1])
# %%

random_stocks_portfolio_df.index

random_stocks_portfolio_year_df = random_stocks_portfolio_df.iloc[random_stocks_portfolio_df.index.year==2000]

# %%

random_stocks_portfolio_year_df.shape

# %%

random_stocks_portfolio_year_df.plot()
# %%
