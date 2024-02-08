# https://gurobi-optimization-gurobi-optimods.readthedocs-hosted.com/en/stable/mods/sharpe-ratio.html
# https://gurobi-optimization-gurobi-optimods.readthedocs-hosted.com/en/stable/api.html#gurobi_optimods.sharpe_ratio.max_sharpe_ratio

# %% 

# Solve the following MIP:
#  maximize
#        x +   y + 2 z
#  subject to
#        x + 2 y + 3 z <= 4
#        x +   y       >= 1
#        x, y, z binary

import gurobipy as gp
from gurobi_optimods.sharpe_ratio import max_sharpe_ratio

# Create a new model
m = gp.Model()
m.addMVar()

# Create variables
x = m.addVar(vtype="B", name="x")
y = m.addVar(vtype="B", name="y")
z = m.addVar(vtype="B", name="z")

# Set objective function
m.setObjective(x + y + 2 * z, gp.GRB.MAXIMIZE)

# Add constraints
m.addConstr(x + 2 * y + 3 * z <= 4)
m.addConstr(x + y >= 1)

# Solve it!
m.optimize()

print(f"Optimal objective value: {m.objVal}")
print(f"Solution values: x={x.X}, y={y.X}, z={z.X}")

# %%

import gurobipy as gp
from gurobi_optimods.sharpe_ratio import max_sharpe_ratio
from get_data import processed_df

processed_df.head()


# %%

import numpy as np
import pandas as pd

stock_cov = np.cov(processed_df, rowvar=False)
print(pd.DataFrame(stock_cov, index=processed_df.columns.values, columns=processed_df.columns.values))
print(f'AAPL var: {processed_df["AAPL"].var()}')
print(f'MFC var: {processed_df["MFC"].var()}')
print(f'WMT var: {processed_df["WMT"].var()}')


print(f'AAPL expected returns: {processed_df["AAPL"][1:].pct_change()}')
print(f'MFC expected returns: {processed_df["MFC"][1:].pct_change()}')
print(f'WMT expected returns: {processed_df["WMT"][1:].pct_change()}')


# %%

# first value of returned df is Nan so start at [1:]
aapl_ret = processed_df["AAPL"].pct_change()[1:].sum()
mfc_ret = processed_df["MFC"].pct_change()[1:].sum()
wmt_ret = processed_df["WMT"].pct_change()[1:].sum()

stock_mu = np.array([aapl_ret, mfc_ret, wmt_ret])
print(stock_mu)



# %%
# https://github.com/Gurobi/gurobi-optimods

stock_port = max_sharpe_ratio(cov_matrix=stock_cov, mu=stock_mu)

# %%

print(f'Allocations:\n{pd.DataFrame(stock_port.x.round(3), index=processed_df.columns.values, columns=["Weight"])}')
print(f'Sharpe Ratio: {stock_port.sharpe_ratio}')
print(f'Return: {stock_port.ret}')
print(f'Risk: {stock_port.risk}')

# %%

# %%

#test np.cov()

rand_dict = {
    'x':[1,2,3,4,5,6,7,8],
    'y':[1,2,3,4,5,0,0,0],
    'z':[2,4,6,8,10,0,0,0]
}

dict_pd = pd.DataFrame.from_dict(rand_dict)
print(dict_pd.head())
# cov(x,x) is the variance of x
print(f'x var: {dict_pd["x"].var()}')
print(f'y var: {dict_pd["y"].var()}')
print(f'z var: {dict_pd["z"].var()}')

# %%

cov_mat = np.cov(dict_pd, rowvar=False)
print(pd.DataFrame(cov_mat, index=dict_pd.columns.values, columns=dict_pd.columns.values))


# %%

x = [1.23, 2.12, 3.34, 4.5]
 
y = [2.56, 2.89, 3.76, 3.95]
 
# find out covariance with respect  rows
cov_mat = np.stack((x, y), axis = 1) 

print(cov_mat, cov_mat.shape)
# %%

x = [1.23, 2.12, 3.34, 4.5]
 
y = [2.56, 2.89, 3.76, 3.95]
 
# find out covariance with respect  rows
cov_mat = np.stack((x, y), axis = 0) 

print(cov_mat, cov_mat.shape)
# %%
