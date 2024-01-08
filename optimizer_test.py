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

# setting the seed allows for reproducible results
random.seed(123)

# %%
# updated model to only include prices in the input of the model
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

            #[:-1] stops iteration just before last index
            portfolio_returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1]  # % change formula

            sharpe = K.mean(portfolio_returns) / K.std(portfolio_returns)
            
            # since we want to maximize Sharpe, while gradient descent minimizes the loss, 
            #   we can negate Sharpe (the min of a negated function is its max)
            return -sharpe
        
        model.compile(loss=sharpe_loss, optimizer='adam')
        return model
    
    def get_allocations(self, data: pd.DataFrame):
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
            self.model = self.__build_model(data.shape, len(data.columns)) # data_w_ret.shape
        
        # change shape of data to (1, row, col) since TensorFlow requires a batch size
        # I got it to work by only using the prices instead of with returns, the weights seemed to have changed a little bit
        fit_predict_data = self.data[np.newaxis,:]  # data_w_ret[np.newaxis,:]  
        self.model.fit(fit_predict_data, np.zeros((1, len(data.columns))), epochs=20, shuffle=False)
        return self.model.predict(fit_predict_data)[0]


# %%

# unchanged model
# there seems to be no change in the weights if I don't use the returns in the model input
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

            #[:-1] stops iteration just before last index
            portfolio_returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1]  # % change formula

            sharpe = K.mean(portfolio_returns) / K.std(portfolio_returns)
            
            # since we want to maximize Sharpe, while gradient descent minimizes the loss, 
            #   we can negate Sharpe (the min of a negated function is its max)
            return -sharpe
        
        model.compile(loss=sharpe_loss, optimizer='adam')
        return model
    
    def get_allocations(self, data: pd.DataFrame):
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
            self.model = self.__build_model(data_w_ret.shape, len(data.columns)) # adjust input shape to include daily returns
        
        # change shape of data to (1, row, col) since TensorFlow requires a batch size
        # I got it to work by only using the prices instead of with returns, the weights seemed to have changed a little bit
        fit_predict_data = data_w_ret[np.newaxis,:]  # include returns in input 
        self.model.fit(fit_predict_data, np.zeros((1, len(data.columns))), epochs=20, shuffle=False)
        return self.model.predict(fit_predict_data)[0]



#%%
    
from get_data import processed_df

processed_df.head()

portfolio_optimizer = Model()

type(portfolio_optimizer)

# %%

allocations = portfolio_optimizer.get_allocations(processed_df)

# %%

# should equal 1
assert allocations.sum().round() == 1, allocations.sum()

allocations_pd = pd.DataFrame(allocations, index=processed_df.columns, columns=['allocation weight'])
print(allocations_pd.head())

# %%

print(processed_df.values[1:].shape)
print(processed_df.values[np.newaxis,:].shape)


# %%

print(processed_df.pct_change().values[1:])

# %%

# pct_change() needs to start at index 1 since it takes the value of index - 1

print((processed_df.values[1][0] - processed_df.values[0][0]) / processed_df.values[0][0])
print((processed_df.values[2][0] - processed_df.values[1][0]) / processed_df.values[1][0])


# %%

print(processed_df.pct_change().values[1:])


# %%

data_w_ret = np.concatenate([ processed_df.values[1:], processed_df.pct_change().values[1:] ], axis=1)

# %%

print(data_w_ret.shape)
print(data_w_ret[0].round(2))

# %%

new_array = processed_df.iloc[1:]
print(new_array[0:4])

# %%

tf_data = tf.cast(tf.constant(new_array), float)

# %%

print(tf_data.get_shape())
print(type(tf_data))
print(tf.get_static_value(tf_data[0:4]))

# %%

# this turns the data into a shape of (1,246,6) which means it is one batch, but why do this?
fit_predict_data = data_w_ret[np.newaxis,:]   
print(type(fit_predict_data))
print(fit_predict_data.shape)
print(tf.get_static_value(fit_predict_data[0][0]))

# %%

zero_matrix = np.zeros((1, len(processed_df.columns)))
print(type(zero_matrix))
print(zero_matrix.shape)


# %%

# %%

check_divide = tf.divide(tf_data[0:4], tf_data[0])
print(check_divide)


# %%

print(tf_data)


# %%

portfolio_values = tf.multiply(check_divide[0:4], np.ones((1, len(processed_df.columns)))*(1/3))
print(tf.get_static_value(portfolio_values[0:4]))
print(portfolio_values.shape)


# %%

# price of the stock (has been normalized) times the weights (here it 33%) and then the value is summed on the columns to get the total portfolio value (assumed only one share)
portfolio_values = tf.reduce_sum(tf.multiply(check_divide, np.ones((1, len(processed_df.columns)))*(1/3)), axis=1) 
print(portfolio_values)


# %%

print(portfolio_values[1:])
print(portfolio_values[:-1])
print(portfolio_values[1:] - portfolio_values[:-1])

# %%

# okay this is a bit strange at first but now I understand. I was confused since I thought [:-1] was iterating through the array backwards
# but [:-1] means that it iterates through the array from the beginning without the last index
portfolio_returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1]  # % change formula
print(portfolio_returns)

# %%

x = [1,2,3,4,5]
print(x[1:])
# this is confusing but ultimately it goes through the whole array and then stops one before the end
print(x[:-1])
# using :: changes the step size through the list, -1 steps through the array backwards
print(x[1::2])
print(x[::-1])
print(x[:len(x)-1])
# this stops 4 indexes before the end
print(x[:-4])


# %%

sharpe = K.mean(portfolio_returns) / K.std(portfolio_returns)

# %%

print(sharpe)
# %%
