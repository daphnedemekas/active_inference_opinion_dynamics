# %% imports
%matplotlib widget

import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from IPython.display import display
import seaborn as sns
from Analysis.plots import plot_beliefs_over_time
# %% define values and get data file
num_agent_values = [5,10,15]

connectedness_values = [0.2,0.5,0.8]
ranges = [[1,2],[1,5],[1,9],[6,7],[6,10]]
str_ranges = ['[1,2]','[1,5]','[1,9]','[6,7]','[6,10]']

df = pd.read_pickle("param_data.pkl")
param_results = np.load('results/params 2.npz',allow_pickle = True)['arr_0']
sim_results = np.load('results/all_results 2.npz',allow_pickle=True)['arr_0']
# %% function to access the real parameters from the simulation

def get_real_precisions(n,c,l,u,r1, r2, r3):
    n_i = num_agent_values.index(n)
    c_i = connectedness_values.index(c)
    r1_i = str_ranges.index(r1)
    r2_i = str_ranges.index(r2)
    r3_i = str_ranges.index(r3)
    all_agent_params = param_results[n_i,c_i,r1_i,r2_i,r3_i]
    return all_agent_params
# %% function to access the cluster ratio from the inputted params

def get_ratio_from_parameters(n, c, r1, r2, r3):
    #first need to rount to the closest valuable option 
    indices = (n, c, r1[0], r1[1], r2[0], r2[1], r3[0], r3[1])
    return df[indices]

def get_sim_results_from_parameters(n, c, r1, r2, r3):
    n_i = num_agent_values.index(n)
    c_i = connectedness_values.index(c)
    r1_i = str_ranges.index(r1)
    r2_i = str_ranges.index(r2)
    r3_i = str_ranges.index(r3)
    adj_mat = sim_results[n_i,c_i,r1_i,r2_i,r3_i][0]
    all_qs = sim_results[n_i,c_i,r1_i,r2_i,r3_i][1]
    all_tweets = sim_results[n_i,c_i,r1_i,r2_i,r3_i][2]
    all_neighbour_samplings = sim_results[n_i,c_i,r1_i,r2_i,r3_i][3]
    result = {'adj_mat' : adj_mat, 'all_qs':all_qs, 'all_tweets':all_tweets, 'all_neighbour_sampling':all_neighbour_samplings}
    return result

# %% make widgets
n_d = widgets.Dropdown(value = 5, options = num_agent_values, description = "number of agents")
c_d = widgets.Dropdown(value = 0.5, options = connectedness_values, description = 'graph connectedness')

r1_d = widgets.Dropdown(value = '[1,5]', options = str_ranges, description = 'precision ranges')
r2_d = widgets.Dropdown(value = '[1,5]', options = str_ranges, description = 'env precision ranges')
r3_d = widgets.Dropdown(value = '[1,5]', options = str_ranges, description = 'belief precision ranges') 

def display_dropdown():
    display(n_d)
    display(c_d)
    display(r1_d)
    display(r2_d)
    display(r3_d)
# %% update function
#here i want to make something where you can click a button and see what the parameters really were 

# %% initialize plot

def heatmap_of_tweets():
    fig, ax = plt.subplots(figsize=(6,4))
    all_tweets = get_sim_results_from_parameters(n_d.value, c_d.value, r1_d.value, r2_d.value, r3_d.value)['all_tweets']
    ax = sns.heatmap(np.transpose(all_tweets))
    plt.show()
    display_dropdown()

def beliefs_over_time():
    fig, ax = plt.subplots(figsize=(6,4))
    belief_hist = get_sim_results_from_parameters(n_d.value, c_d.value, r1_d.value, r2_d.value, r3_d.value)['all_qs']
    plot_beliefs_over_time(belief_hist)
    plt.show()
    display_dropdown()

def get_cluster_ratio(all_qs):
    cluster_ratios = np.zeros((all_qs.shape[0],1))
    believers = np.where(all_qs[-1,1,:] > 0.5)[0]
    nonbelievers = np.where(all_qs[-1,1,:] < 0.5)[0]
    for t in range(all_qs.shape[0]):
        if np.sum(all_qs[t,0,believers]) == 0 or np.sum(all_qs[t,1,nonbelievers]) == 0:
            cluster_ratio = 0
        else:
            cluster_ratio = np.sum(all_qs[t,0,believers]) / np.sum(all_qs[t,1,nonbelievers])
            cluster_ratio = cluster_ratio if cluster_ratio < 1 else 1/cluster_ratio
        cluster_ratios[t] = cluster_ratio
    return cluster_ratios

def cluster_ratio_over_time():
    fig, ax = plt.subplots(figsize=(6,2))
    belief_hist = get_sim_results_from_parameters(n_d.value, c_d.value, r1_d.value, r2_d.value, r3_d.value)['all_qs']
    cluster_ratios = get_cluster_ratio(belief_hist)
    ax = sns.heatmap(np.transpose(cluster_ratios))
    plt.show()
    display_dropdown()

# fig, ax = plt.subplots(figsize=(6, 4))
# values = []
# for r in ranges:
#     values.append(get_ratio_from_parameters(n_d.value, c_d.value, r, ranges[0], ranges[0]))
# plt.plot(str_ranges, values)
# plt.title("Clustering Ratio at last iteration")
# ax.set_ylim([0,1])
# ax.grid(True)
# plt.show()












# # %% update function
# output = widgets.Output()
# @output.capture
# def n_update(change):
#     with output:
#         replot(change.new, c_d.value, r2_d.value, r3_d.value)
# @output.capture
# def c_update(change):
#     replot(n_d.value, change.new, r2_d.value, r3_d.value)
# @output.capture
# def r2_update(change):
#     replot(n_d.value, c_d.value, change.new, r3_d.value)
# @output.capture
# def r3_update(change):
#     plt.plot([1,2],[3,4])
#     plt.show()
#     replot(n_d.value, c_d.value, r2_d.value, change.new)

# def replot(n, c, env_r, b_r):
#     #output.clear_output()
#     r2 = ranges[str_ranges.index(env_r)]
#     r3 = ranges[str_ranges.index(b_r)]
#     new_data = []
#     for r in ranges:
#         new_data.append(get_ratio_from_parameters(n, c, r, env_r, b_r))
#     with output:
#         plt.plot(str_ranges, new_data)
#         plt.show()
#         display(output)
# # %% replot function to call in vs code
   
# def replot_vs():
#     n = n_d.value
#     c = c_d.value
#     #output.clear_output()
#     r2 = ranges[str_ranges.index(r2_d.value)]
#     r3 = ranges[str_ranges.index(r3_d.value)]
#     new_data = []
#     for r in ranges:
#         new_data.append(get_ratio_from_parameters(n, c, r, r2, r3))
#     fig, ax = plt.subplots(figsize=(3, 2))

#     plt.plot(str_ranges, new_data)
#     plt.ylabel("Cluster Homogeneity")
#     plt.xlabel("ECB Precision Range")
#     plt.show()
#     display(n_d)
#     display(c_d)
#     display(r1_d)
#     display(r2_d)
#     display(r3_d)

# # %% define the observations
# n_d.observe(n_update, names='value')
# c_d.observe(c_update, names='value')
# r2_d.observe(r2_update, names='value')
# r3_d.observe(r3_update, names='value')

# display(n_d)
# display(c_d)
# display(r1_d)
# display(r2_d)
# display(r3_d)
# # %%



