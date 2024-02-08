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

tf.random.set_seed(SEED)

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
        # does the added percent returns influence the outcome? Possible change.
        # it is included but I'm not sure how much it impacts the result
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

##### get stock data and select 10 at random #####
    
stocks_df = pd.read_csv('..\\Daten\\stocks_19900102_20191231.csv', index_col='Date')

random_stock_selection = RNG.choice(stocks_df.keys(), 10)

random_stocks_df = stocks_df[random_stock_selection]

random_stocks_df.index = pd.to_datetime(stocks_df.index, format="%Y-%m-%d")

random_stocks_df.head()

# %%

##### create dictionary with keys as years and values as monthly prices #####
# iterate over each month of every year

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
print(stocks_years_dict[2000][0].shape) # January 2000


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

##### save allocations for every year and its months #####

allocations_year_df = pd.DataFrame()
month_columns = ['January', 'February', 'March', 'April','May', 'June', 'July', 'August','September', 'October', 'November', 'December']

for year, allo_list in allocations_dict.items():
    
    allocations_year_df = pd.DataFrame()

    allocations_month_list = list(zip(month_columns, allo_list))

    for month_column, month_allocation in allocations_month_list:

        allocations_year_df[month_column] = month_allocation

    # save file
    # allocations_year_df.to_csv(os.path.join('allocations','1M', f'{year}_monthly_allocations.csv'))


