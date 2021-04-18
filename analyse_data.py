# %% imports
%matplotlib widget

import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from IPython.display import display
# %% define values and get data file

# Change matplotlib backend
num_agent_values = [5,10,15]

connectedness_values = [0.2,0.5,0.8]
ranges = [[1,2],[1,5],[1,9],[6,7],[6,10]]
str_ranges = ['[1,2]','[1,5]','[1,9]','[6,7]','[6,10]']

df = pd.read_pickle("param_data.pkl")
#param_results = np.load('results/params 2.npz',allow_pickle = True)['arr_0']

# %% function to access the real parameters from the simulation

def get_real_precisions(n,c,l,u,r1, r2, r3):
    n_i = num_agent_values.index(n)
    c_i = connectedness_values.index(c)
    all_agent_params = param_results[n_i,c_i,r1,r2,r3]
    return all_agent_params
# %% function to access the cluster ratio from the inputted params

def get_ratio_from_parameters(n, c, r1, r2, r3):
    #first need to rount to the closest valuable option 
    indices = (n, c, r1[0], r1[1], r2[0], r2[1], r3[0], r3[1])
    return df[indices]

# %% make widgets

n_d = widgets.Dropdown(value = 5, options = num_agent_values, description = "number of agents")
c_d = widgets.Dropdown(value = 0.5, options = connectedness_values, description = 'graph connectedness')

r1_d = widgets.Dropdown(value = '[1,5]', options = str_ranges, description = 'precision ranges')
r2_d = widgets.Dropdown(value = '[1,5]', options = str_ranges, description = 'env precision ranges')
r3_d = widgets.Dropdown(value = '[1,5]', options = str_ranges, description = 'belief precision ranges') 

# %% update function
#here i want to make something where you can click a button and see what the parameters really were 

# %% initialize plot

fig, ax = plt.subplots(figsize=(6, 4))
values = []
for r in ranges:
    values.append(get_ratio_from_parameters(n_d.value, c_d.value, r, ranges[0], ranges[0]))
plt.plot(str_ranges, values)
ax.set_ylim([0,1])
ax.grid(True)
plt.show()

# %% update function
output = widgets.Output()
@output.capture
def n_update(change):
    print("Fuck u")
    with output:
        replot(change.new, c_d.value, r2_d.value, r3_d.value)
@output.capture
def c_update(change):
    replot(n_d.value, change.new, r2_d.value, r3_d.value)
@output.capture
def r2_update(change):
    replot(n_d.value, c_d.value, change.new, r3_d.value)
@output.capture
def r3_update(change):
    print("Hello?")
    plt.plot([1,2],[3,4])
    plt.show()
    replot(n_d.value, c_d.value, r2_d.value, change.new)

def replot(n, c, env_r, b_r):
    #output.clear_output()
    r2 = ranges[str_ranges.index(env_r)]
    r3 = ranges[str_ranges.index(b_r)]
    new_data = []
    for r in ranges:
        new_data.append(get_ratio_from_parameters(n, c, r, env_r, b_r))
    with output:
        plt.plot(str_ranges, new_data)
        plt.show()
        display(output)

# %% define the observations
n_d.observe(n_update, names='value')
c_d.observe(c_update, names='value')
r2_d.observe(r2_update, names='value')
r3_d.observe(r3_update, names='value')

display(n_d)
display(c_d)
display(r1_d)
display(r2_d)
display(r3_d)
# %%
