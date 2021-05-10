# %% imports
import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from IPython.display import display
import seaborn as sns
from plots import plot_beliefs_over_time, plot_conclusion_thresholds
# %% define values and get data file

#fake_results = np.load('results/sbm_test0003.npz')['arr_1']
# %% function to access the real parameters from the simulation

def get_real_precisions(t,n,c,ecb, env, bel):
    n_i = num_agent_values.index(n)
    c_i = connectedness_values.index(c)
    ecb_i = ecb_precision_gammas.index(ecb)
    env_i = env_precision_gammas.index(env)
    bel_i = b_precision_gammas.index(bel)
    all_agent_params = param_results[t,n_i,c_i,ecb_i,env_i,bel_i]
    return all_agent_params

def display_dropdown(params):
    display(params.n_d)
    display(params.c_d)
    display(params.ecb_d)
    display(params.env_d)
    display(params.b_d)
# %% update function
#here i want to make something where you can click a button and see what the parameters really were 

# %% initialize plot
def heatmap_of_tweets(params):
    fig, ax = plt.subplots(figsize=(6,4))
    all_tweets = get_sim_results_from_parameters(params)['all_tweets']
    ax = sns.heatmap(np.transpose(all_tweets))
    plt.show()
    display_dropdown(params)

def beliefs_over_time(params, p):
    fig, ax = plt.subplots(figsize=(6,4))
    belief_hist = get_sim_results_from_parameters(params)['all_qs']
    plot_beliefs_over_time(belief_hist)
    plt.title(p)
    plt.show()
    #display_dropdown(n_d, c_d, ecb_d, env_d, b_d)

# %% initialize plot

def evaluate_clustering(params, p):
    results = {}
    belief_hist = get_sim_results_from_parameters(params)['all_qs']
    results["davies bouldin index"] = davies_bouldin(belief_hist) 
    results["cluster kl div"] = cluster_kl(belief_hist)[-1] 
    results["cluster meanvar"] = cluster_mean_over_variance(belief_hist)

    #results["agent convergence times"] = [conclusion_thresholds(belief_hist)
    #results["cluster ratio"] = get_cluster_ratio(belief_hist)[-1]
    results["ECB"] = params.ecb_d.value
    results["Graph Connnectedness"] = params.c_d.value
    results["num agents"] = params.n_d.value
    #display_dropdown(n_d, c_d, ecb_d, env_d, b_d)
    return results 

def cluster_ratio_over_time(params):
    fig, ax = plt.subplots(figsize=(6,2))
    belief_hist = get_sim_results_from_parameters(params)['all_qs']
    cluster_ratios = get_cluster_ratio(belief_hist)
    ax = sns.heatmap(np.transpose(cluster_ratios), vmin = 0, vmax = 1)
    plt.show()
    display_dropdown(params)

def cluster_kl_over_time(params):
    fig, ax = plt.subplots(figsize=(6,2))
    belief_hist = get_sim_results_from_parameters(paramss)['all_qs']
    cluster_metric = get_cluster_metric(belief_hist)
    
    ax = sns.heatmap(np.transpose(cluster_metric), vmin = 0.2, vmax = 0.8)
    plt.show()
    display_dropdown(n_d, c_d, ecb_d, env_d, b_d)


#metric for sampling behaviour? larger means more variance - just make a list of who they sampled from. 
def sampling_variance(params):
    actions = get_sim_results_from_parameters(params)['all_neighbour_sampling']
    sampling_variances = np.zeros(params.n_d.value)
    for a in range(params.n_d.value):
        sampling_variances[a] = np.var(actions[:,a])
    return sampling_variances













# fig, ax = plt.subplots(figsize=(6, 4))
# values = []
# for r in ranges:
#     values.append(get_ratio_from_parameters(n_d.value, c_d.value, r, ranges[0], ranges[0]))
# plt.plot(str_ranges, values)
# plt.title("Clustering Ratio at last iteration")
# ax.set_ylim([0,1])
# ax.grid(True)
# plt.show()









# def get_ratio_from_parameters(n, c, ecb, env, r3):
#     #first need to rount to the closest valuable option 
#     indices = (n, c, r1[0], r1[1], env[0], env[1], r3[0], r3[1])
#     return df[indices]


# # %% update function
# output = widgets.Output()
# @output.capture
# def n_update(change):
#     with output:
#         replot(change.new, c_d.value, env_d.value, r3_d.value)
# @output.capture
# def c_update(change):
#     replot(n_d.value, change.new, env_d.value, r3_d.value)
# @output.capture
# def env_update(change):
#     replot(n_d.value, c_d.value, change.new, r3_d.value)
# @output.capture
# def r3_update(change):
#     plt.plot([1,2],[3,4])
#     plt.show()
#     replot(n_d.value, c_d.value, env_d.value, change.new)

# def replot(n, c, env_r, b_r):
#     #output.clear_output()
#     env = ranges[str_ranges.index(env_r)]
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
#     env = ranges[str_ranges.index(env_d.value)]
#     r3 = ranges[str_ranges.index(r3_d.value)]
#     new_data = []
#     for r in ranges:
#         new_data.append(get_ratio_from_parameters(n, c, r, env, r3))
#     fig, ax = plt.subplots(figsize=(3, 2))

#     plt.plot(str_ranges, new_data)
#     plt.ylabel("Cluster Homogeneity")
#     plt.xlabel("ECB Precision Range")
#     plt.show()
#     display(n_d)
#     display(c_d)
#     display(r1_d)
#     display(env_d)
#     display(r3_d)

# # %% define the observations
# n_d.observe(n_update, names='value')
# c_d.observe(c_update, names='value')
# env_d.observe(env_update, names='value')
# r3_d.observe(r3_update, names='value')

# display(n_d)
# display(c_d)
# display(r1_d)
# display(env_d)
# display(r3_d)
# # %%




# %%
# %% function to access the cluster ratio from the inputted params
""" def get_all_sim_results(params):
    n_i = num_agent_values.index(params.n_d.value)
    c_i = connectedness_values.index(params.c_d.value)
    ecb_i = ecb_precision_gammas.index(ecb)
    env_i = env_precision_gammas.index(env)
    bel_i = b_precision_gammas.index(bel)
    adj_mat = sim_results[t,n_i,c_i,ecb_i,env_i,bel_i][0]
    all_qs = sim_results[t,n_i,c_i,ecb_i,env_i,bel_i][1] 
    all_tweets = sim_results[t,n_i,c_i,ecb_i,env_i,bel_i][2] 
    all_neighbour_samplings = sim_results[:,n_i,c_i,ecb_i,env_i,bel_i][3]
    result = {'adj_mat' : adj_mat, 'all_qs':all_qs, 'all_tweets':all_tweets, 'all_neighbour_sampling':all_neighbour_samplings}
    return result """
