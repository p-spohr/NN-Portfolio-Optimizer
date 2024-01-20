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

rfr_df = pd.read_csv('portfolio_rfr_value_1M.csv', index_col=0)
rfr_df.index = pd.to_datetime(rfr_df.index, format='%Y-%m-%d')
rfr_df.head()

# %%

gspc_df = pd.read_csv('gspc_19900102_20191231.csv', index_col=0)
gspc_df.index = pd.to_datetime(gspc_df.index, format='%Y-%m-%d')
gspc_df = gspc_df.loc[gspc_df.index >= '1990-02-01']
gspc_df.head()

# %%

ust_df = pd.read_csv('UST_10_rfr_update.csv', index_col=0)
ust_df.index = pd.to_datetime(ust_df.index, format='%Y-%m-%d')
ust_df = ust_df.loc[ust_df.index >= '1990-02-01']
ust_df.head()


# %%

def normalize_prices(prices_df : pd.DataFrame):

    max_price = prices_df.max()
    min_price = prices_df.min()

    for price in range(len(prices_df)):

        prices_df.iloc[price] = (prices_df.iloc[price] - min_price) / (max_price - min_price)

    return prices_df

# %%

##### normalize portfolio prices for plotting #####

normalized_gspc_portfolio_value = normalize_prices(gspc_df.sum(axis=1))
normalized_equal_portfolio_value = normalize_prices(equal_df.sum(axis=1))
normalized_opt_portfolio_value = normalize_prices(opt_df.sum(axis=1))
normalized_rfr_portfolio_value = normalize_prices(rfr_df.sum(axis=1))


normalized_gspc_portfolio_value = normalized_gspc_portfolio_value.groupby(pd.Grouper(freq='M')).mean()
normalized_equal_portfolio_value = normalized_equal_portfolio_value.groupby(pd.Grouper(freq='M')).mean()
normalized_opt_portfolio_value = normalized_opt_portfolio_value.groupby(pd.Grouper(freq='M')).mean()
normalized_rfr_portfolio_value = normalized_rfr_portfolio_value.groupby(pd.Grouper(freq='M')).mean()

# %%

fig, ax = plt.subplots(1,1)

plt.plot(normalized_gspc_portfolio_value, label='gspc')
plt.plot(normalized_equal_portfolio_value, label='equal')
plt.plot(normalized_opt_portfolio_value, label='opt')
plt.plot(normalized_rfr_portfolio_value, label='rfr')

plt.legend()


# %%

print(normalized_gspc_portfolio_value.std())
print(normalized_equal_portfolio_value.std())
print(normalized_opt_portfolio_value.std())
print(normalized_rfr_portfolio_value.std())

# %%

rfr_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change() - ust_df.groupby(pd.Grouper(freq="M")).mean()['rfr']

# %%

(rfr_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change() - ust_df.groupby(pd.Grouper(freq="M")).mean()['rfr']).sum() / rfr_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change().std()

# %%


print(f'RFR return: {rfr_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change().sum()}')
print(f'OPT rate return: {opt_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change().sum()}')
print(f'EQU return: {equal_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change().sum()}')
print(f'GSPC return: {gspc_df.groupby(pd.Grouper(freq="M")).mean().pct_change().sum()}')

print('-' * 10)

print('SHARPE RATIOS TOTAL')
print(f'RFR: {rfr_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change().sum() / rfr_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change().std()}')
print(f'OPT: {opt_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change().sum() / opt_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change().std()}')
print(f'EQUAL: {equal_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change().sum() / equal_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change().std()}')
print(f'GSPC: {gspc_df.groupby(pd.Grouper(freq="M")).mean().pct_change().sum() / gspc_df.groupby(pd.Grouper(freq="M")).mean().pct_change().std()}')

print('-' * 10)

print('SHARPE RATIOS MONTHLY MEAN')

print(f'RFR: {(rfr_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change() / rfr_df.pct_change().sum(axis=1).groupby(pd.Grouper(freq="M")).std()).mean()}')
print(f'OPT: {(opt_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change() / opt_df.pct_change().sum(axis=1).groupby(pd.Grouper(freq="M")).std()).mean()}')
print(f'EQU: {(equal_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change() / equal_df.pct_change().sum(axis=1).groupby(pd.Grouper(freq="M")).std()).mean()}')
print(f'GSPC: {(gspc_df.groupby(pd.Grouper(freq="M")).mean().pct_change() / gspc_df.pct_change().groupby(pd.Grouper(freq="M")).std()).mean()}')

print('-' * 10)


# %%

print('SHARPE RATIOS TOTAL WITH RISK-FREE RATE')

rfr_monthly = rfr_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change()
ust_monthly = ust_df.groupby(pd.Grouper(freq="M")).mean()["rfr"]
rfr_std = rfr_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change().std()

print(f'RFR: {(rfr_monthly - ust_monthly).sum() / rfr_std}')

opt_monthly = opt_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change()
ust_monthly = ust_df.groupby(pd.Grouper(freq="M")).mean()["rfr"]
opt_std = opt_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change().std()

print(f'OPT: {(opt_monthly - ust_monthly).sum() / opt_std}')

equal_monthly = equal_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change()
ust_monthly = ust_df.groupby(pd.Grouper(freq="M")).mean()["rfr"]
equal_std = equal_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change().std()

print(f'EQU: {(equal_monthly - ust_monthly).sum() / equal_std}')

gspc_monthly = gspc_df.groupby(pd.Grouper(freq="M")).mean().pct_change()["Close"]
ust_monthly = ust_df.groupby(pd.Grouper(freq="M")).mean()["rfr"]
gspc_std = gspc_df.groupby(pd.Grouper(freq="M")).mean().pct_change().std().iloc[0]

print(f'GSPC: {(gspc_monthly - ust_monthly).sum() / gspc_std}')


# %%

print('SHARPE RATIOS MONTHLY MEAN WITH RISK-FREE RATE')

rfr_monthly = rfr_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change()
ust_monthly = ust_df.groupby(pd.Grouper(freq="M")).mean()["rfr"]
rfr_monthly_std = rfr_df.sum(axis=1).pct_change().groupby(pd.Grouper(freq="M")).std()

print(f'RFR: {((rfr_monthly - ust_monthly) / rfr_std).mean()}')

opt_monthly = opt_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change()
ust_monthly = ust_df.groupby(pd.Grouper(freq="M")).mean()["rfr"]
opt_monthly_std = opt_df.sum(axis=1).pct_change().groupby(pd.Grouper(freq="M")).std()

print(f'OPT: {((opt_monthly - ust_monthly) / opt_std).mean()}')

equal_monthly = equal_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change()
ust_monthly = ust_df.groupby(pd.Grouper(freq="M")).mean()["rfr"]
equal_monthly_std = equal_df.sum(axis=1).pct_change().groupby(pd.Grouper(freq="M")).std()

print(f'EQU: {((equal_monthly - ust_monthly) / equal_std).mean()}')

gspc_monthly = gspc_df.groupby(pd.Grouper(freq="M")).mean().pct_change()["Close"]
ust_monthly = ust_df.groupby(pd.Grouper(freq="M")).mean()["rfr"]
gspc_monthly_std = gspc_df.pct_change().groupby(pd.Grouper(freq="M")).std()["Close"]

print(f'GSPC: {((gspc_monthly - ust_monthly) / gspc_std).mean()}')


# %%

