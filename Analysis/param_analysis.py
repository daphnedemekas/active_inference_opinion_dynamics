import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from IPython.display import display
import seaborn as sns
import itertools
#from Analysis.plots import plot_beliefs_over_time, plot_conclusion_thresholds
#from analysis_functions import *
import metrics as cm 
import os 

class ParameterAnalysis(object):

    def __init__(
        self,
        results_file,
        params_file,
        num_agent_values,
        connectedness_values,
        ecb_precision_gammas,
        env_precision_gammas,
        b_precision_gammas,   
        lrs,
        variances,
        n_trials
        ):       
        
        self.results_file = results_file
        self.params_file = params_file
        self.num_agent_values = num_agent_values
        self.connectedness_values = connectedness_values
        self.ecb_precision_gammas = ecb_precision_gammas
        self.env_precision_gammas = env_precision_gammas
        self.b_precision_gammas = b_precision_gammas
        self.lrs = lrs
        self.variances = variances 
        self.n_trials = n_trials
        self.n = len(num_agent_values)
        self.c = len(connectedness_values)
        
        self.n_d = widgets.Dropdown(value = self.num_agent_values[0], options = num_agent_values, description = "number of agents")
        self.c_d = widgets.Dropdown(value = connectedness_values[0], options = connectedness_values, description = 'graph connectedness')

        self.ecb_d = widgets.Dropdown(value = ecb_precision_gammas[0], options = ecb_precision_gammas, description = 'ecb precision gammas')
        self.env_d = widgets.Dropdown(value = env_precision_gammas[0], options = env_precision_gammas, description = 'env precision gammas')
        self.b_d = widgets.Dropdown(value = b_precision_gammas[0], options = b_precision_gammas, description = 'belief precision gammas') 
        self.lr_d = widgets.Dropdown(value = lrs[0], options = lrs, description = "learning rates")
        self.v_d = widgets.Dropdown(value = variances[0], options = variances, description = "variances")

        self.param_combinations = itertools.product(self.num_agent_values, self.connectedness_values, self.ecb_precision_gammas, self.b_precision_gammas, self.env_precision_gammas)

    def load_results(self, data_dir):
        all_data = np.zeros((len(list(self.get_param_combinations())), self.n_trials, 4), dtype=object)
        
        for i in range(len(list(self.get_param_combinations()))):
            for trial in range(self.n_trials):
                all_data[i, trial] = np.load(str(data_dir) + "/" + str(i) +"/" + str(trial) + ".npz", allow_pickle = True)['arr_0']
        self.all_data = all_data
        return all_data
    
    def get_sim_results_from_files(self):
        adj_mat = np.array([self.all_data[self.param_idx, i][0] for i in range(self.n_trials)])
        all_qs = np.array([self.all_data[self.param_idx, i][1] for i in range(self.n_trials)])
        all_tweets = np.array([self.all_data[self.param_idx, i][2] for i in range(self.n_trials)])
        all_neighbour_samplings = np.array([self.all_data[self.param_idx, i][3] for i in range(self.n_trials)])
        result = {'adj_mat' : adj_mat, 'all_qs':all_qs, 'all_tweets':all_tweets, 'all_neighbour_sampling':all_neighbour_samplings}
        self.all_qs = all_qs
        self.all_tweets = all_tweets
        self.adj_mat = adj_mat
        self.all_neighbour_samplings = all_neighbour_samplings
        return result

    def get_param_combinations(self):
        self.param_combinations = itertools.product(self.num_agent_values, self.connectedness_values, self.ecb_precision_gammas, self.b_precision_gammas, self.env_precision_gammas, self.variances, self.lrs)
        return self.param_combinations

    def update_params(self, params):
        self.n_d.value = params[0]
        self.c_d.value = params[1]
        self.ecb_d.value = params[2]
        self.b_d.value = params[3]
        self.env_d.value = params[4]
        self.v_d.value = params[5]
        self.lr_d.value = params[6]

        self.param_idx = np.where(np.all(np.array(list(self.get_param_combinations())) == params, axis=1) == True)[0].tolist()[0]

    def get_overall_metrics(self, from_files = False):
        n_parameters = len(np.array(list(self.get_param_combinations())))
        self.avg_belief_extremity = np.zeros(n_parameters)
        self.avg_belief_diff = np.zeros(n_parameters)
        self.times_to_cluster = np.zeros(n_parameters)

        for i, combo in enumerate(list(self.get_param_combinations())[:-1]):
            print(i)
            self.update_params(combo)
            if from_files:
                self.get_sim_results_from_files()
            else:
                self.get_all_sim_results_from_parameters()
            self.avg_belief_extremity[i] = np.mean(np.array([cm.average_belief_extremity(self.all_qs[j,:,:,:]) for j in range(self.n_trials)]))
            self.avg_belief_diff[i] = np.mean(np.array([cm.average_belief_difference(self.all_qs[j,:,:,:]) for j in range(self.n_trials)]))
            self.times_to_cluster[i] = np.mean(np.array([cm.time_to_cluster(self.all_qs[j,:,:,:]) for j in range(self.n_trials)]))
            
