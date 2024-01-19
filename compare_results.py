# %%

import pandas as pd
import matplotlib.pyplot as plt

import random

SEED=12345
random.seed(SEED)

# %%


equal_df = pd.read_csv('equal_portfolio_1M.csv', index_col=0)
equal_df = equal_df.iloc[equal_df.index >= '1990-02-01']
equal_df.index = pd.to_datetime(equal_df.index, format='%Y-%m-%d')
equal_df.head()

# %%

opt_df = pd.read_csv('portfolio_value_1M.csv', index_col=0)
opt_df.index = pd.to_datetime(opt_df.index, format='%Y-%m-%d')
opt_df.head()

# %%

gspc_df = pd.read_csv('gspc_19900102_20191231.csv', index_col=0)
gspc_df.index = pd.to_datetime(gspc_df.index, format='%Y-%m-%d')
gspc_df.head()

# %%

equal_df.pct_change().sum(axis=0)

# %%

opt_df.pct_change().sum(axis=0)
# %%

equal_df.groupby(pd.Grouper(freq='MS')).mean().pct_change().sum(axis=0).sum()

# %%

opt_df.groupby(pd.Grouper(freq='MS')).mean().pct_change().sum(axis=0).sum()

# %%

# portfolio value of equal weights
equal_df.groupby(pd.Grouper(freq='M')).mean().sum(axis=1).pct_change().sum()


# %%

# portfolio value of optimal weights
opt_df.groupby(pd.Grouper(freq='M')).mean().sum(axis=1).pct_change().sum()

# %%

opt_df.groupby(pd.Grouper(freq='M')).mean().sum(axis=1).plot()

# %%

equal_df.groupby(pd.Grouper(freq='M')).mean().sum(axis=1).plot()
# %%

gspc_df.groupby(pd.Grouper(freq='M')).mean().sum(axis=1).plot()

# %%

gspc_df.groupby(pd.Grouper(freq='M')).mean().pct_change().sum()

# %%

(gspc_df.iloc[-1] - gspc_df.iloc[0]) / gspc_df.iloc[0]
# %%
(equal_df.iloc[-1] - equal_df.iloc[0]) / equal_df.iloc[0]
# %%
(opt_df.iloc[-1] - opt_df.iloc[0]) / opt_df.iloc[0]

# %%

def normalize_prices(prices_df : pd.DataFrame):

    max_price = prices_df.max()
    min_price = prices_df.min()

    for price in range(len(prices_df)):

        prices_df.iloc[price] = (prices_df.iloc[price] - min_price) / (max_price - min_price)

    return prices_df
# %%

opt_df.sum(axis=1)
# %%

##### normalize portfolio prices for plotting #####

normalized_gspc_portfolio_value = normalize_prices(gspc_df.sum(axis=1))
normalized_equal_portfolio_value = normalize_prices(equal_df.sum(axis=1))
normalized_opt_portfolio_value = normalize_prices(opt_df.sum(axis=1))

normalized_gspc_portfolio_value = normalized_gspc_portfolio_value.groupby(pd.Grouper(freq='M')).mean()
normalized_equal_portfolio_value = normalized_equal_portfolio_value.groupby(pd.Grouper(freq='M')).mean()
normalized_opt_portfolio_value = normalized_opt_portfolio_value.groupby(pd.Grouper(freq='M')).mean()

# %%

fig, ax = plt.subplots(1,1)

plt.plot(normalized_gspc_portfolio_value, label='gspc')
plt.plot(normalized_equal_portfolio_value, label='equal')
plt.plot(normalized_opt_portfolio_value, label='opt')
plt.legend()


# %%

print(normalized_gspc_portfolio_value.std())
print(normalized_equal_portfolio_value.std())
print(normalized_opt_portfolio_value.std())

# %%

##### Sharpe Ratio of gspc #####
# standard deviation
gspc_df.groupby(pd.Grouper(freq='M')).mean().pct_change().std()

# %%

# expected return
gspc_df.groupby(pd.Grouper(freq='M')).mean().pct_change().sum()


# %%

gspc_df.groupby(pd.Grouper(freq='M')).mean().pct_change().sum() / gspc_df.groupby(pd.Grouper(freq='M')).mean().pct_change().std()



# %%

##### Sharpe Ratio of equal portfolio #####
# standard deviation
equal_df.groupby(pd.Grouper(freq='M')).mean().sum(axis=1).pct_change().std()

# %%

# expected return
equal_df.groupby(pd.Grouper(freq='M')).mean().sum(axis=1).pct_change().sum()

# %%

equal_df.groupby(pd.Grouper(freq='M')).mean().sum(axis=1).pct_change().sum() / equal_df.groupby(pd.Grouper(freq='M')).mean().sum(axis=1).pct_change().std()

# %%

##### Sharpe ratio of optimized portfolio #####
# standard deviation
opt_df.groupby(pd.Grouper(freq='M')).mean().sum(axis=1).pct_change().std()

# %%

# expected return
opt_df.groupby(pd.Grouper(freq='M')).mean().sum(axis=1).pct_change().sum()

# %%

opt_df.groupby(pd.Grouper(freq='M')).mean().sum(axis=1).pct_change().sum() / opt_df.groupby(pd.Grouper(freq='M')).mean().sum(axis=1).pct_change().std()

# %%

print('SHARPE RATIOS')
print(f'GSPC: {gspc_df.groupby(pd.Grouper(freq="M")).mean().pct_change().sum() / gspc_df.groupby(pd.Grouper(freq="M")).mean().pct_change().std()}')
print(f'EQUAL: {equal_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change().sum() / equal_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change().std()}')
print(f'OPT: {opt_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change().sum() / opt_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change().std()}')

# %%

# monthly mean returns
gspc_df.groupby(pd.Grouper(freq='M')).mean().pct_change()

# %%

# monthly return standard deviations
gspc_df.pct_change().groupby(pd.Grouper(freq='M')).std()

# %%

# monthly Sharpe ratios
gspc_df.groupby(pd.Grouper(freq='M')).mean().pct_change() / gspc_df.pct_change().groupby(pd.Grouper(freq='M')).std()

# %%

# mean of all monthly Sharpe ratios
(gspc_df.groupby(pd.Grouper(freq='M')).mean().pct_change() / gspc_df.pct_change().groupby(pd.Grouper(freq='M')).std()).mean()

# %%

# monthly mean returns
equal_df.groupby(pd.Grouper(freq='M')).mean().sum(axis=1).pct_change()

# %%

# monthly return standard deviations
equal_df.pct_change().sum(axis=1).groupby(pd.Grouper(freq='M')).std()

# %%

# monthly Sharpe ratios
equal_df.groupby(pd.Grouper(freq='M')).mean().sum(axis=1).pct_change() / equal_df.pct_change().sum(axis=1).groupby(pd.Grouper(freq='M')).std()
# %%

# mean of all monthly Sharpe ratios
(equal_df.groupby(pd.Grouper(freq='M')).mean().sum(axis=1).pct_change() / equal_df.pct_change().sum(axis=1).groupby(pd.Grouper(freq='M')).std()).mean()

# %%

# monthly mean returns
opt_df.groupby(pd.Grouper(freq='M')).mean().sum(axis=1).pct_change()

# %%

# monthly return standard deviations
opt_df.pct_change().sum(axis=1).groupby(pd.Grouper(freq='M')).std()

# %%

# monthly Sharpe ratios
opt_df.groupby(pd.Grouper(freq='M')).mean().sum(axis=1).pct_change() / opt_df.pct_change().sum(axis=1).groupby(pd.Grouper(freq='M')).std()

# %%

# mean of all monthly Sharpe ratios
(opt_df.groupby(pd.Grouper(freq='M')).mean().sum(axis=1).pct_change() / opt_df.pct_change().sum(axis=1).groupby(pd.Grouper(freq='M')).std()).mean()
# %%

# standard deviation of all monthly Sharpe ratios
(gspc_df.groupby(pd.Grouper(freq='M')).mean().pct_change() / gspc_df.pct_change().groupby(pd.Grouper(freq='M')).std()).std()

#%%

(equal_df.groupby(pd.Grouper(freq='M')).mean().sum(axis=1).pct_change() / equal_df.pct_change().sum(axis=1).groupby(pd.Grouper(freq='M')).std()).std()
# %%

(opt_df.groupby(pd.Grouper(freq='M')).mean().sum(axis=1).pct_change() / opt_df.pct_change().sum(axis=1).groupby(pd.Grouper(freq='M')).std()).std()

# %%

(gspc_df.groupby(pd.Grouper(freq='M')).mean().pct_change() / gspc_df.pct_change().groupby(pd.Grouper(freq='M')).std()).max()

# %%

(equal_df.groupby(pd.Grouper(freq='M')).mean().sum(axis=1).pct_change() / equal_df.pct_change().sum(axis=1).groupby(pd.Grouper(freq='M')).std()).max()

# %%

(opt_df.groupby(pd.Grouper(freq='M')).mean().sum(axis=1).pct_change() / opt_df.pct_change().sum(axis=1).groupby(pd.Grouper(freq='M')).std()).max()

# %%
