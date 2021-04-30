# %% imports
#%matplotlib widget

import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from IPython.display import display
import seaborn as sns
import itertools
#from Analysis.plots import plot_beliefs_over_time, plot_conclusion_thresholds
from analysis_functions import *

class ParameterAnalysis(object):

    def __init__(
        self,
        num_agent_values,
        connectedness_values,
        ecb_precision_gammas,
        env_precision_gammas,
        b_precision_gammas,   
        n_trials
        ):       

        self.num_agent_values = num_agent_values
        self.connectedness_values = connectedness_values
        self.ecb_precision_gammas = ecb_precision_gammas
        self.env_precision_gammas = env_precision_gammas
        self.b_precision_gammas = b_precision_gammas
        self.n_trials = n_trials
        self.n = len(num_agent_values)
        self.c = len(connectedness_values)

        self.param_results = np.load('results/params.npz',allow_pickle = True)['arr_0']
        self.sim_results = np.load('results/all_results.npz',allow_pickle=True)['arr_0']
        
        
        self.n_d = widgets.Dropdown(value = 5, options = num_agent_values, description = "number of agents")
        self.c_d = widgets.Dropdown(value = 0.5, options = connectedness_values, description = 'graph connectedness')

        self.ecb_d = widgets.Dropdown(value = 4, options = ecb_precision_gammas, description = 'ecb precision gammas')
        self.env_d = widgets.Dropdown(value = 8, options = env_precision_gammas, description = 'env precision gammas')
        self.b_d = widgets.Dropdown(value = 8, options = b_precision_gammas, description = 'belief precision gammas') 


        #self.parameter_args = {"num_agent_values" : self.num_agent_values, "connectedness_values": self.connectedness_values, 
        #                    "ecb_precision_gammas":self.ecb_precision_gammas, "env_precision_gammas": self.env_precision_gammas, 
        #                   "b_precision_gammas" : self.b_precision_gammas, "n_d" : self.n_d, "c_d" : self.c_d, "ecb_d": self.ecb_d, "env_d" : self.env_d, "b_d":self.b_d }


    def display_both():
        cluster_metric_over_time(self)
        beliefs_over_time(self)

    def evaluate_clusters(self):
        cluster_summary = evaluate_clustering(self)
        beliefs_over_time(self)
        heatmap_of_tweets(self)
        return cluster_summary
    

    def get_variance_of(metric):

        all_qs = sim_results[:,:,:,:,:,:][1]
        param_combos = itertools.product(num_agent_values, connectedness_values, ecb_precision_gammas,env_precision_gammas,b_precision_gammas)
        num_combos = len(list(param_combos))
        cluster_ratios = np.zeros((num_combos, num_trials, 50))
        cluster_metrics = np.zeros((num_combos, num_trials, 50))
        param_combos = itertools.product(num_agent_values, connectedness_values, ecb_precision_gammas,env_precision_gammas,b_precision_gammas)

        for i, combo in enumerate(param_combos):
            for trial in range(num_trials):
                beliefs = get_all_sim_results(trial,combo[0],combo[1],combo[2],combo[3],combo[4], num_agent_values, connectedness_values, ecb_precision_gammas,b_precision_gammas,env_precision_gammas)['all_qs']
                cluster_ratios[i, trial, :] = get_cluster_ratio(beliefs).squeeze()
                cluster_metrics[i, trial, :] = get_cluster_metric(beliefs).squeeze()
    
        cluster_ratio_variances_per_timestep = np.zeros((num_combos, 50))
        cluster_metric_variances_per_timestep = np.zeros((num_combos, 50))

        param_combos = itertools.product(num_agent_values, connectedness_values, ecb_precision_gammas,env_precision_gammas,b_precision_gammas)

        for i, combo in enumerate(param_combos):
            cluster_ratio_variances_per_timestep[i,:] = np.var(cluster_ratios[i,:], axis = 0)
            cluster_metric_variances_per_timestep[i, :] = np.var(cluster_metrics[i,:],axis=0)

        average_overall_ratio_variance = np.average(cluster_ratio_variances_per_timestep)
        average_overall_metric_variance = np.average(cluster_metric_variances_per_timestep)


num_agent_values = [5,10,15]
connectedness_values = [0.2,0.5,0.8]
#precision_ranges = [[1,2],[1,5],[1,9],[6,7],[6,10]]
ecb_precision_gammas = [1,4,6,8]
env_precision_gammas = [5,8,10]
b_precision_gammas = [5,8,10]
n_trials = 5 

params = ParameterAnalysis(num_agent_values, connectedness_values, ecb_precision_gammas, env_precision_gammas, b_precision_gammas, n_trials)

#what is the relationship between graph structure and ECB in terms of clustering behaviour? 

params.n_d.value = 10
params.env_d.value = 10
params.b_d.value = 5
params.c_d.value = 0.
params.ecb_d.value = 6
cluster_summary = params.evaluate_clusters()
print(cluster_summary)
raise


connectedness_values = [0.2,0.5,0.8]
ecb_precision_gammas = [1,4,6,8]
params.n_d.value = 10
params.env_d.value = 10
params.b_d.value = 5

for c in connectedness_values:
    for e in ecb_precision_gammas:
        params.c_d.value = c
        params.ecb_d.value = e
        cluster_summary = params.evaluate_clusters()
        print("PARAMETERS")
        print(c)
        print(e)
        print()
        print()
        print("SUMMARY")
        print(cluster_summary)

# %%
