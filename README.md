# Neural Network Portfolio Optimizer

## Motive

Use neural network to determine new weights of a portfolio by using Sharpe Ratio as the loss function. 

## Goals

- get_stock_data
    1. retrieves individual stock prices from csv files
    2. checks that all the stocks exist and have same time frame
    3. prepares data for NN model
- portfolio_optimizer
    1. imports stock_portfolio from get_stock_data and feeds it into model
    2. model returns weights of stocks in portfolio


## Links to Sources

**Neural Network Model**
https://github.com/shilewenuw/deep-learning-portfolio-optimization/blob/main/Model.py

**Stock Data**
https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset/data
