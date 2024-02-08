# %%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import random

# setting the seed allows for reproducible results
SEED=12345
random.seed(SEED)
RNG = np.random.default_rng(SEED)

# %%

equal_df = pd.read_csv('..\\portfolio_values\\portfolio_equ_value_1M.csv', index_col=0) # equal weights
equal_df = equal_df.iloc[equal_df.index >= '1990-02-01']
equal_df.index = pd.to_datetime(equal_df.index, format='%Y-%m-%d')
equal_df.head()

# %%

opt_df = pd.read_csv('..\\portfolio_values\\portfolio_opt_value_1M.csv', index_col=0) # no risk-free rate
opt_df.index = pd.to_datetime(opt_df.index, format='%Y-%m-%d')
opt_df.head()

# %%

rfr_df = pd.read_csv('..\\portfolio_values\\portfolio_opt_rfr_value_1M.csv', index_col=0) # with risk-free rate
rfr_df.index = pd.to_datetime(rfr_df.index, format='%Y-%m-%d')
rfr_df.head()

# %%

gspc_df = pd.read_csv('..\\Daten\\gspc_19900102_20191231.csv', index_col=0) # benchmark S&P 500
gspc_df.index = pd.to_datetime(gspc_df.index, format='%Y-%m-%d')
gspc_df = gspc_df.loc[gspc_df.index >= '1990-02-01']
gspc_df.head()

# %%

ust_df = pd.read_csv('portfolio_value/ust_10_rfr_19900102_20191231.csv', index_col=0)
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

##### line chart of normalized portfolio values #####

fig, ax = plt.subplots(4,1)

plt.figure(figsize=[10,7])

fig.suptitle("Normalized Portfolio Values from 1990 to 2020")

ax[0].plot(normalized_gspc_portfolio_value, label='gspc', color='purple', linewidth=1)
ax[0].legend()
ax[0].set_xticks([])

ax[1].plot(normalized_equal_portfolio_value, label='equal', color='blue', linewidth=1)
ax[1].legend()
ax[1].set_xticks([])
ax[1].set_yticks([])

ax[2].plot(normalized_opt_portfolio_value, label='opt', color='orange', linewidth=1)
ax[2].legend()
ax[2].set_xticks([])
ax[2].set_yticks([])

ax[3].plot(normalized_rfr_portfolio_value, label='opt_rfr', color='green', linewidth=1)
ax[3].legend()
ax[3].set_yticks([])
ax[3].set_xlabel('Years')

# save file
# fig.savefig("results_imgs/Normalized Portfolio Values from 1990 to 2019")

# %%



# %%


print('SUM OF PORTFOLIO MONTHLY RETURNS')
print(f'RFR: {round(rfr_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change().sum(), 4)}')
print(f'OPT: {round(opt_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change().sum(), 4)}')
print(f'EQU: {round(equal_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change().sum(), 4)}')
print(f'GSPC: {round(gspc_df.groupby(pd.Grouper(freq="M")).mean().pct_change().sum().loc["Close"], 4)}')

print('-' * 40)

print('SHARPE RATIOS PORTFOLIO TOTAL WITHOUT RFR')
print(f'RFR: {round(rfr_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change().sum() / rfr_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change().std(), 4)}')
print(f'OPT: {round(opt_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change().sum() / opt_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change().std(), 4)}')
print(f'EQUAL: {round(equal_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change().sum() / equal_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change().std(), 4)}')
print(f'GSPC: {round((gspc_df.groupby(pd.Grouper(freq="M")).mean().pct_change().sum() / gspc_df.groupby(pd.Grouper(freq="M")).mean().pct_change().std()).loc["Close"], 4)}')

print('-' * 40)

print('SHARPE RATIOS PORTFOLIO MONTHLY MEAN WITHOUT RFR')

print(f'RFR: {round((rfr_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change() / rfr_df.pct_change().sum(axis=1).groupby(pd.Grouper(freq="M")).std()).mean(), 4)}')
print(f'OPT: {round((opt_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change() / opt_df.pct_change().sum(axis=1).groupby(pd.Grouper(freq="M")).std()).mean(), 4)}')
print(f'EQU: {round((equal_df.groupby(pd.Grouper(freq="M")).mean().sum(axis=1).pct_change() / equal_df.pct_change().sum(axis=1).groupby(pd.Grouper(freq="M")).std()).mean(), 4)}')
print(f'GSPC: {round((gspc_df.groupby(pd.Grouper(freq="M")).mean().pct_change() / gspc_df.pct_change().groupby(pd.Grouper(freq="M")).std()).mean().loc["Close"], 4)}')

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

##### get selected Nasdaq stock tickers and read stock meta data #####

stocks_df = pd.read_csv('..//Daten//stocks_19900102_20191231.csv', index_col='Date')

random_stock_selection = RNG.choice(stocks_df.keys(), 10)

random_stocks_df = stocks_df[random_stock_selection]

random_stocks_df.index = pd.to_datetime(stocks_df.index, format="%Y-%m-%d")

random_stocks_tickers = list(random_stocks_df.keys())

meta_data = pd.read_csv('stock_data/symbols_valid_meta.csv')

##### download meta data from selected stocks #####

# save file
# meta_data.loc[:,['Symbol', 'Security Name', 'ETF']].set_index(meta_data['Symbol']).loc[random_stocks_tickers].to_csv('selected_stocks_meta.csv')


# %%

def plot_monthly_price_stock(stock_prices : pd.DataFrame):

    for stock in stock_prices.keys():
        
        plt.subplots(1,1)

        plt.plot(random_stocks_df[stock].groupby(pd.Grouper(freq="M")).mean())
        
        plt.xlabel('Years')
        plt.ylabel('Price USD')
        plt.title(f'{stock} Price from 1990 to 2019')

        plt.savefig(f'results_imgs/{stock}_monthly_price')

        plt.close()

# %%

# save plots of individual stock prices
# plot_monthly_price_stock(random_stocks_df)


# %%

##### create stock metrics for individual stocks #####

ust_df = pd.read_csv('..\\Daten\\ust_10_rfr_19900102_20191231.csv', index_col=0)
ust_df.index = pd.to_datetime(ust_df.index, format='%Y-%m-%d')
ust_df.head() 

def get_stock_analysis(stock_prices : pd.DataFrame, rfr_df : pd.DataFrame):

    for stock in stock_prices.keys():
        
        metrics_df = pd.DataFrame(columns=['metrics'])

        price_df = pd.DataFrame()

        print(f'----------{stock}----------')

        price_df = stock_prices[stock]

        highest_price = price_df.max()
        highest_price_date = price_df.loc[price_df == price_df.max()].index[0]

        lowest_price = price_df.min()
        lowest_price_date = price_df.loc[price_df == price_df.min()].index[0]

        total_monthly_returns = price_df.groupby(pd.Grouper(freq='M')).mean().pct_change().sum()

        mean_monthly_returns = price_df.groupby(pd.Grouper(freq='M')).mean().pct_change().mean()

        monthly_return = price_df.groupby(pd.Grouper(freq='M')).mean().pct_change()
        monthly_rfr = rfr_df['rfr'].groupby(pd.Grouper(freq='M')).mean().pct_change()
        monthly_std = price_df.pct_change().groupby(pd.Grouper(freq="M")).std()

        mean_monthly_sharpe = ( (monthly_return - monthly_rfr) / monthly_std )

        # for when std is zero get mean of earlier and later month
        for i in range(len(mean_monthly_sharpe)):
        
            if (mean_monthly_sharpe.iloc[i] == -np.inf) or (mean_monthly_sharpe.iloc[i] == np.inf):
                
                new_sharpe_mean = (mean_monthly_sharpe.iloc[i-2] + mean_monthly_sharpe.iloc[i+2]) / 4
                
                mean_monthly_sharpe.iloc[i] = new_sharpe_mean


        new_mean_monthly_sharpe = mean_monthly_sharpe.mean()

        hp_ser = pd.Series(highest_price, index=['highest_price'])
        hpd_ser = pd.Series(highest_price_date, index=['highest_price_date'])
        lp_ser = pd.Series(lowest_price, index=['lowest_price'])
        lpd_ser = pd.Series(lowest_price_date, index=['lowest_price_date'])
        tmr_ser = pd.Series(total_monthly_returns, index=['total_monthly_returns'])
        mmr_ser = pd.Series(mean_monthly_returns, index=['mean_monthly_returns'])
        nmms_ser = pd.Series(new_mean_monthly_sharpe, index=['mean_monthly_sharpe'])

        metrics_df['metrics'] = pd.concat([hp_ser, hpd_ser, lp_ser, lpd_ser, tmr_ser, mmr_ser, nmms_ser])

        metrics_df.to_csv(f'stock_metrics/{stock}_metrics.csv')


# %%

# save stock metrics as csv    
# get_stock_analysis(random_stocks_df, ust_df)

