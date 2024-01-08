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

import time

# setting the seed allows for reproducible results
random.seed(123)

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
    
    def get_allocations(self, data: pd.DataFrame, num_epochs: int):
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
        self.model.fit(fit_predict_data, np.zeros((1, len(data.columns))), epochs=num_epochs, shuffle=False)

        # calculated weights and check for 100% weight total
        allocations = self.model.predict(fit_predict_data)[0]
        assert allocations.sum().round() == 1, allocations.sum()

        # create DataFrame with tickers for easy viewing
        results_df = pd.DataFrame(allocations, index= data.columns, columns=['Allocation Weight'])

        return results_df
    

# %%

# Nasdaq stocks
stocks_df = pd.read_csv('stocks_df_19900102_20191230', index_col='Date')
stocks_df.head()

# %%

# from 1990-01-02 to 2019-12-31
# DATE_RANGES = ["1990-01-02","2019-12-31"]

DATE_RANGES = ["2001-01-02","2010-12-31"]
STOCK_TICKERS = []
EPOCHS = 100

target_stocks_df = stocks_df.query('`Date` > @DATE_RANGES[0] and `Date` < @DATE_RANGES[1] ')
target_stocks_df = target_stocks_df.iloc[:,0:12]
target_stocks_df.head()

# %%
 
start_time = time.time()

portfolio_optimizer = Model()

allocations = portfolio_optimizer.get_allocations(target_stocks_df, EPOCHS)

end_time = time.time()
print(f'Total Runtime: {round(end_time - start_time, 3)} seconds')


# %%

print(allocations)

# %%

