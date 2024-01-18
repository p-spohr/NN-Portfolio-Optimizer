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


# %%
    
##### updated model to include more results #####

class Model:
    def __init__(self):
        self.data = None
        self.model = None
        
    def __build_model(self, input_shape, outputs):
        '''
        Builds and returns the Deep Neural Network that will compute the allocation ratios
        that optimize the Sharpe Ratio of the portfolio
        
        inputs: input_shape - tuple of the input shape, outputs - the number of assets
        returns: a Deep Neural Network model
        '''
        model = Sequential([
            LSTM(64, input_shape=input_shape),
            Flatten(),
            Dense(outputs, activation='softmax')
        ])

        def sharpe_loss(_, y_pred):
            
            # make all time-series start at 1
            data = tf.divide(self.data, self.data[0])  
            
            # value of the portfolio after allocations applied
            portfolio_values = tf.reduce_sum(tf.multiply(data, y_pred), axis=1) 

            # [:-1] stops iteration just before last index
            portfolio_returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1]  # % change formula

            # (sharpe = return - risk free rate) / standard deviation
            # did not take risk free rate into account
            sharpe = tf.math.reduce_mean(portfolio_returns) / tf.math.reduce_std(portfolio_returns)
            
            # since we want to maximize Sharpe, while gradient descent minimizes the loss, 
            # we can negate Sharpe (the min of a negated function is its max)
            return -sharpe
        
        model.compile(loss=sharpe_loss, optimizer='adam')
        return model
    
    def get_allocations(self, data: pd.DataFrame, num_epochs: int, show_epochs):
        '''
        Computes and returns the allocation ratios that optimize the Sharpe over the given data
        
        input: data - DataFrame of historical closing prices of various assets
        
        return: the allocations ratios for each of the given assets
        '''
        
        # data with returns
        # this step doesn't make sense since the data_w_ret isn't used anywhere else except to use it's shape to build the model
        data_w_ret = np.concatenate([ data.values[1:], data.pct_change().values[1:] ], axis=1)
        
        data = data.iloc[1:]
        self.data = tf.cast(tf.constant(data), float)
        
        if self.model is None:
            self.model = self.__build_model(data_w_ret.shape, len(data.columns))
        
        # change shape of data to (1, row, col) since TensorFlow requires a batch size
        fit_predict_data = data_w_ret[np.newaxis,:]  # include returns in input 
        self.model.fit(fit_predict_data, np.zeros((1, len(data.columns))), epochs=num_epochs, shuffle=False, verbose=show_epochs)

        # calculated weights and check for 100% weight total
        allocations = self.model.predict(fit_predict_data)[0]
        assert allocations.sum().round() == 1, allocations.sum()

        # create DataFrame with tickers for easy viewing
        results_df = pd.DataFrame(allocations, index= data.columns, columns=['Allocation Weight'])

        return results_df
    

# %%

# Nasdaq stocks
stocks_df = pd.read_csv('stocks_19900102_20191231.csv', index_col='Date')
stocks_df.head()


# %%

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

stocks_years_dict[1990][1]

# %%

##### get optimal weights for every month #####

start_time = time.time()

allocations_dict = {}
allocations_list = []

for year, month_list in stocks_years_dict.items():
    
    allocations_list = []
    portfolio_optimizer = Model()

    for month_prices in month_list:

        allocations_df = portfolio_optimizer.get_allocations(month_prices, num_epochs=50, show_epochs=0)

        allocations_list.append(allocations_df)

    allocations_dict[year] = allocations_list

print(f'Total Runtime: {round(end_time - start_time, 3)} seconds') # 307.847 seconds

# %%

stocks_years_dict[1990][1]

# %%

##### save allocations for every year and its months #####

allocations_year_df = pd.DataFrame()
month_columns = ['January', 'February', 'March', 'April','May', 'June', 'July', 'August','September', 'October', 'November', 'December']

for year, allo_list in allocations_dict.items():
    
    allocations_year_df = pd.DataFrame()

    allocations_month_list = list(zip(month_columns, allo_list))

    for month_column, month_allocation in allocations_month_list:

        allocations_year_df[month_column] = month_allocation

    allocations_year_df.to_csv(os.path.join('allocations','1M', f'{year}_monthly_allocations'))




# %%

##### combine prices and allocations for easier unpacking later #####

set_portfolio_prices_list = []
set_portfolio_allocations_list = []
all_prices_allocations = []

for year_dict_1, month_prices in stocks_years_dict.items():

    set_portfolio_prices_list = []
    set_portfolio_allocations_list = []

    # set quarterly prices in a list
    for prices in month_prices:

        set_portfolio_prices_list.append(prices)

    # set quarterly allocations in a list
    for year_dict_2, month_allocations in allocations_dict.items():
        
        if year_dict_1 == year_dict_2:

            for allocations in month_allocations:

                set_portfolio_allocations_list.append(allocations)

    # all prices and allocations are tuple pairs for the whole 120 quarters
    all_prices_allocations.extend(list(zip(set_portfolio_prices_list, set_portfolio_allocations_list)))



# %%

##### offset the list of prices and allocations, so the first quarter allocations will be used for second quarter prices #####

all_prices_offset = [prices for prices, allocations in all_prices_allocations[1:]]

all_allocations_offset = [allocations for prices, allocations in all_prices_allocations[0:-1]]

all_prices_allocations_offset = list(zip(all_prices_offset, all_allocations_offset))

print(len(all_prices_allocations_offset))

# %%

all_prices_allocations_offset[0][0]

# %%

stocks_years_dict[1990][1]

# %%

all_prices_allocations_offset[0][1]

# %%

##### calculate returns for each month using optimal weights #####

opt_portfolio_month_returns_list = []

for month_prices_df, month_allocations_df in all_prices_allocations_offset:
    
    # multiply the stock price by the optimal weight
    for column in month_prices_df.keys():
        
        month_prices_df.loc[:, column] = month_prices_df.loc[:, column] * month_allocations_df.loc[column].loc['Allocation Weight']
    
    # sum values for value of portfolio at time
    month_prices_df = month_prices_df.sum(axis=1)

    # add returns for each month to list
    opt_portfolio_month_returns_list.append((month_prices_df.iloc[-1] - month_prices_df.iloc[0]) / month_prices_df.iloc[0])

print(len(opt_portfolio_month_returns_list))
   
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
