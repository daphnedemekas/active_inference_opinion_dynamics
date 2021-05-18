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
import cluster_metrics as cm 


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
        n_trials
        ):       
        
        self.results_file = results_file
        self.params_file = params_file
        self.num_agent_values = num_agent_values
        self.connectedness_values = connectedness_values
        self.ecb_precision_gammas = ecb_precision_gammas
        self.env_precision_gammas = env_precision_gammas
        self.b_precision_gammas = b_precision_gammas
        self.n_trials = n_trials
        self.n = len(num_agent_values)
        self.c = len(connectedness_values)

        self.param_results = np.load(self.params_file,allow_pickle = True)['arr_0']
        self.sim_results = np.load(self.results_file,allow_pickle=True)['arr_0']
        
        
        self.n_d = widgets.Dropdown(value = self.num_agent_values[0], options = num_agent_values, description = "number of agents")
        self.c_d = widgets.Dropdown(value = connectedness_values[0], options = connectedness_values, description = 'graph connectedness')

        self.ecb_d = widgets.Dropdown(value = ecb_precision_gammas[0], options = ecb_precision_gammas, description = 'ecb precision gammas')
        self.env_d = widgets.Dropdown(value = env_precision_gammas[0], options = env_precision_gammas, description = 'env precision gammas')
        self.b_d = widgets.Dropdown(value = b_precision_gammas[0], options = b_precision_gammas, description = 'belief precision gammas') 


        self.param_combinations = itertools.product(self.num_agent_values, self.connectedness_values, self.ecb_precision_gammas, self.b_precision_gammas, self.env_precision_gammas)
   
    def get_sim_results_from_parameters(self):
        n_i = self.num_agent_values.index(self.n_d.value)
        c_i = self.connectedness_values.index(self.c_d.value)
        ecb_i = self.ecb_precision_gammas.index(self.ecb_d.value)
        env_i = self.env_precision_gammas.index(self.env_d.value)
        bel_i = self.b_precision_gammas.index(self.b_d.value)
        adj_mat = np.mean(np.array([self.sim_results[i,n_i,c_i,ecb_i,env_i,bel_i][0] for i in range(params.n_trials)]),axis=0)#take the average 
        avg_all_qs = np.mean(np.array([self.sim_results[i,n_i,c_i,ecb_i,env_i,bel_i][1] for i in range(params.n_trials)]),axis=0)
        avg_all_tweets = np.mean(np.array([self.sim_results[i,n_i,c_i,ecb_i,env_i,bel_i][2] for i in range(params.n_trials)]),axis=0)
        all_neighbour_samplings = self.sim_results[0,n_i,c_i,ecb_i,env_i,bel_i][3]
        result = {'adj_mat' : adj_mat, 'all_qs':avg_all_qs, 'all_tweets':avg_all_tweets, 'all_neighbour_sampling':all_neighbour_samplings}
        #self.all_qs = avg_all_qs
        self.all_tweets = avg_all_tweets
        self.adj_mat = adj_mat
        self.all_neighbour_samplings = all_neighbour_samplings
        return result
    
    def get_all_sim_results_from_parameters(self):
        n_i = self.num_agent_values.index(self.n_d.value)
        c_i = self.connectedness_values.index(self.c_d.value)
        ecb_i = self.ecb_precision_gammas.index(self.ecb_d.value)
        env_i = self.env_precision_gammas.index(self.env_d.value)
        bel_i = self.b_precision_gammas.index(self.b_d.value)
        adj_mat = np.array([self.sim_results[i,n_i,c_i,ecb_i,env_i,bel_i][0] for i in range(self.n_trials)])
        avg_all_qs = np.array([self.sim_results[i,n_i,c_i,ecb_i,env_i,bel_i][1] for i in range(self.n_trials)])
        avg_all_tweets = np.array([self.sim_results[i,n_i,c_i,ecb_i,env_i,bel_i][2] for i in range(self.n_trials)])
        all_neighbour_samplings = np.array([self.sim_results[i,n_i,c_i,ecb_i,env_i,bel_i][3] for i in range(self.n_trials)])
        result = {'adj_mat' : adj_mat, 'all_qs':avg_all_qs, 'all_tweets':avg_all_tweets, 'all_neighbour_sampling':all_neighbour_samplings}
        self.all_qs = avg_all_qs
        self.all_tweets = avg_all_tweets
        self.adj_mat = adj_mat
        self.all_neighbour_samplings = all_neighbour_samplings
        return result
    
    def get_real_precisions(t,n,c,ecb, env, bel):
        n_i = num_agent_values.index(n)
        c_i = connectedness_values.index(c)
        ecb_i = ecb_precision_gammas.index(ecb)
        env_i = env_precision_gammas.index(env)
        bel_i = b_precision_gammas.index(bel)
        all_agent_params = param_results[t,n_i,c_i,ecb_i,env_i,bel_i]
        return all_agent_params

    def get_param_combinations(self):
        self.param_combinations = itertools.product(self.num_agent_values, self.connectedness_values, self.ecb_precision_gammas, self.b_precision_gammas, self.env_precision_gammas)
        return self.param_combinations

    def update_params(self, params):
        self.n_d.value = params[0]
        self.c_d.value = params[1]
        self.ecb_d.value = params[2]
        self.b_d.value = params[3]
        self.env_d.value = params[4]

    def get_overall_metrics(self):
        self.db_indices = np.zeros(len(list(self.get_param_combinations())))
        self.cluster_kls = np.zeros(len(list(self.get_param_combinations())))
        self.sampling_ratios = np.zeros((len(list(self.get_param_combinations())),2,9))
        self.sampling_ratios_test = []
        for i, combo in enumerate(list(self.get_param_combinations())):
            self.update_params(combo)
            self.get_all_sim_results_from_parameters()
            self.db_indices[i] = np.mean(np.array([cm.davies_bouldin(self.all_qs[j,:,:,:]) for j in range(self.n_trials)]))
            self.cluster_kls[i] = np.mean(np.array([cm.cluster_kl(self.all_qs[j,:,:,:])[-1] for j in range(self.n_trials)]))
            self.sampling_ratios[i,:,:] = np.mean([cm.sampling_ratio(self.all_qs[i,:,:,:], self.all_neighbour_samplings[i,:,:]) for i in range(self.n_trials)],axis=0)
            self.sampling_ratios_test.append([cm.sampling_ratio(self.all_qs[i,:,:,:], self.all_neighbour_samplings[i,:,:]) for i in range(self.n_trials)])


    def display_dropdown(self):
        display(self.n_d)
        display(self.c_d)
        display(self.ecb_d)
        display(self.env_d)
        display(self.b_d)
    
    def display_both():
        cluster_metric_over_time(self)
        beliefs_over_time(self)

  #  def evaluate_clusters(self, parameters):
  #      cluster_summary = evaluate_clustering(self, parameters)
  #      return cluster_summary
""" 

connectedness_values = [0.2,0.3,0.4,0.6,0.8]
ecb_precision_gammas = [1,2,3,4,5,6,7,8,9]

num_agent_values = [4,6,8,10]


env_precision_gammas = [10]
b_precision_gammas = [5]

n_trials = 30

params = ParameterAnalysis(num_agent_values, connectedness_values, ecb_precision_gammas, env_precision_gammas, b_precision_gammas, n_trials)

#what is the relationship between graph structure and ECB in terms of clustering behaviour? 

params.n_d.value = num_agent_values[0]
params.env_d.value = env_precision_gammas[0]
params.b_d.value = b_precision_gammas[0]
params.c_d.value = connectedness_values[0]
params.ecb_d.value = ecb_precision_gammas[0]




# %% 
params.get_overall_metrics()

db_per_c = []
for c in num_agent_values:
    c_params = np.where(np.isin(np.array(list(params.get_param_combinations()))[:,0],c))[0]
    params_to_test = np.array(list(params.get_param_combinations()))[c_params]
    db_indices = params.db_indices[c_params]
    avg_db = np.average(db_indices)
    db_per_c.append(avg_db)

plt.plot(num_agent_values, db_per_c)
plt.show()
# %% 


lowest_db_index = np.where(params.db_indices == min(params.db_indices))[0][0]
# %% 

print("Lowest DB index")
db_params = list(params.get_param_combinations())[lowest_db_index]
print(db_params)
#check the plots
params.update_params(db_params)
#params.get_sim_results_from_parameters()

#HERE PLOT EVERYTHING 
plot_beliefs_over_time(params.all_qs[-1,:,:,:])
plt.show()
# %%  """



#How does clustering change with graph structure? 

""" 

#now we can begin to ask some questions ... 
#what if we restrict the number of agents to 12?
# %% 
ten_agents = np.where(np.isin(np.array(list(params.get_param_combinations()))[:,0],10))[0]
ten_agent_dbs = params.db_indices[ten_agents]
lowest_db_ten = np.where(ten_agent_dbs == min(ten_agent_dbs))[0][0]
#its way higher !! 
ten_params = np.array(list(params.get_param_combinations()))[ten_agents][lowest_db_ten]
print(ten_params)
#check the plots
params.update_params(ten_params)
params.get_sim_results_from_parameters()
plot_beliefs_over_time(params.all_qs)

#we can already conclude that adding more agents makes clustering more difficult, but still possible, with lower connectedness and a higher ECB 


# %% 
#What if ECB is 1?
no_ecb = np.where(np.isin(np.array(list(params.get_param_combinations()))[:,2],2))[0]
no_ecb_dbs = params.dunn_indices[no_ecb]
lowest_db_no_ecb = np.where(no_ecb_dbs == max(no_ecb_dbs))[0][0]
#its way higher !! 
no_ecb_params = np.array(list(params.get_param_combinations()))[no_ecb][lowest_db_no_ecb]
print(no_ecb_params)
#check the plots
params.update_params(no_ecb_params)
params.get_sim_results_from_parameters()
plot_beliefs_over_time(params.all_qs)

# %% 

#investigate the relationship between ECB and clustering
#Low ecb - what do we need for clustering? 


#parameters with low ECB 
low_ecb2 = np.where(np.isin(np.array(list(params.get_param_combinations()))[:,2],2))[0]
low_ecb1 = np.where(np.isin(np.array(list(params.get_param_combinations()))[:,2],1))[0]
low_ecb = np.hstack([low_ecb1, low_ecb2])
low_ecb_db = params.db_indices[low_ecb]
#can we get clustering with low ECB?
lowest_db_no_ecb = np.where(low_ecb_db < 0.2)[0]
low_ecb_params = np.array(list(params.get_param_combinations()))[low_ecb][lowest_db_no_ecb]

# %%  """


"""     def get_variance_of(metric):

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
 """