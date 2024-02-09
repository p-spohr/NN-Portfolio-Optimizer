# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random

import os

SEED = 12345
random.seed(SEED)

# %%

##### put all allocation csvs into dictionary ######

work_path = "..\\allocations_rfr\\stock_yearly"

all_allocations_dict = {}

for dirpath, dirnames, filenames in os.walk(work_path):

    print(dirpath)

    for file in filenames:
        
        ticker = file.split('_')[0]

        yearly_allocations = pd.read_csv(os.path.join(dirpath, file), index_col=0)

        all_allocations_dict[ticker] = yearly_allocations



# %%

##### plot all ten stocks on one figure #####

line_colors = ['blue', 'darkturquoise', 'olivedrab', 'red', 'indigo', 'fuchsia', 'green', 'orange', 'dimgray', 'steelblue']

plot_dims = []
allocations_array = []
ticker_array = []

# create the axs tuple
for i in range(0,5):
    for j in range(0,2):
        plot_dims.append((i,j))

for ticker, allocation in all_allocations_dict.items():
    
    ticker_array.append(ticker)
    allocations_array.append(allocation)
    
ticker_allocation_dims = list(zip(ticker_array, allocations_array, plot_dims, line_colors))

print(ticker_allocation_dims[0][2])
print(plot_dims[-1][0], plot_dims[-1][1])

# %%

fig, axs = plt.subplots(plot_dims[-1][0] + 1, plot_dims[-1][1] + 1, figsize=(8,4))

plt.figure(figsize=(15,10))

fig.suptitle('Durchschnittliche Gewichte über 30 Jahre')
fig.supylabel("Aktie Gewicht")
fig.supxlabel("Jahr")

for ticker, allocation, dims, color in ticker_allocation_dims:

    allo_mean = allocation.mean(axis=1)

    axs[dims].plot(allo_mean, label = ticker, color = color)

    if dims[0] == 0:
        axs[dims].set_yticks([0,1])
    else:
        axs[dims].set_yticks([])

    if dims[0] < 4:

        axs[dims].set_xticks([])

    axs[dims].legend(fontsize = 8)


fig.set_size_inches(10,6)
fig.savefig('plot_allocations_30_years.png')



# %%
    

fig, axs = plt.subplots(plot_dims[-1][0] + 1, plot_dims[-1][1] + 1)

axs[(0,0)].plot(np.arange(0,20))
axs[(4,1)].plot(np.arange(0,20))

# %%


