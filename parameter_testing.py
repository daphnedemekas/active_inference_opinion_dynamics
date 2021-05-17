# %% Imports

import numpy as np
import networkx as nx
from Model.agent import Agent
from Simulation.simtools import initialize_agent_params, initialize_network, run_simulation, connect_edgeless_nodes, clip_edges
from Analysis.analysis_tools import collect_idea_beliefs, collect_sampling_history, collect_tweets
from Model.pymdp import maths
from Model.pymdp import utils
import copy
from matplotlib import pyplot as plt
import seaborn as sns
from Analysis.plots import *
import csv
import pandas as pd 
import itertools

h_idea_mapping = utils.softmax(np.eye(2) * 1.0)

connectedness_values = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
ecb_precision_gammas = [9]

num_agent_values = [2,3,4,5,6,7,8,9,10]

n = len(num_agent_values)
c = len(connectedness_values)
#precision_ranges = [[1,2],[1,5],[1,9],[6,7],[6,10]]
env_precision_gammas = [10]
b_precision_gammas = [5]

r_len = len(ecb_precision_gammas)
n_trials = 30

param_combos = itertools.product(num_agent_values, connectedness_values, ecb_precision_gammas,env_precision_gammas,b_precision_gammas)
# %% construct network
iter = 0

all_parameters_to_store = utils.obj_array((n_trials, n,c,r_len,r_len,r_len))
all_results_to_store = utils.obj_array((n_trials, n,c,r_len,r_len,r_len))

for param_config in param_combos:
    num_agents_i, connectedness_i, ecb_p_i, env_precision_i, b_precision_i = param_config
    N, p, T = num_agents_i, connectedness_i, 50

    all_trials_ecbs = np.random.gamma(shape=ecb_p_i,size=(n_trials,)) # one random gamma distributed precision for each trial
    all_trials_envs = np.random.gamma(shape=env_precision_i,size=(n_trials,))
    all_trials_bs = np.random.gamma(shape=b_precision_i,size=(n_trials,))
    G = nx.fast_gnp_random_graph(N,p)

        # make sure graph is connected and all agents have at least one edge
    if not nx.is_connected(G):
        G = connect_edgeless_nodes(G) # make sure graph is connected
    while np.array(list(G.degree()))[:,1].min() < 2: # make sure no agents with only 1 edge
        #G = nx.stochastic_block_model(sizes, probs, seed=0) # create the graph for this trial & condition
        G = nx.fast_gnp_random_graph(N,p)

        if not nx.is_connected(G):
            G = connect_edgeless_nodes(G) # make sure graph is 

    for trial_i in range(n_trials):
        indices = (trial_i, num_agent_values.index(num_agents_i), connectedness_values.index(connectedness_i), ecb_precision_gammas.index(ecb_p_i), env_precision_gammas.index(env_precision_i), b_precision_gammas.index(b_precision_i))

        ecb_precisions = all_trials_ecbs[trial_i]
        env_precisions = all_trials_envs[trial_i]
        b_precisions = all_trials_bs[trial_i]

        agent_constructor_params, store_params = initialize_agent_params(G, h_idea_mappings = h_idea_mapping, \
                                    ecb_precisions = ecb_precisions, B_idea_precisions = env_precisions, \
                                        B_neighbour_precisions = b_precisions, reduce_A=True)

        all_parameters_to_store[indices] = store_params
        G = initialize_network(G, agent_constructor_params, T = T)
        G = run_simulation(G, T = T)

        all_qs = collect_idea_beliefs(G)
        all_neighbour_samplings = collect_sampling_history(G)
    
        adj_mat = nx.to_numpy_array(G)
        all_tweets = collect_tweets(G)

        believers = np.where(all_qs[-1,0,:] > 0.5)
        nonbelievers = np.where(all_qs[-1,0,:] < 0.5)
        
        

        all_results_to_store[indices] = (adj_mat, all_qs, all_tweets, all_neighbour_samplings)

        if iter % 10 ==0:
            print(iter)
        iter +=1

np.savez('results/params_best', all_parameters_to_store)
np.savez('results/all_results_best', all_results_to_store)






# def get_cluster_ratio(all_qs):
#     if np.sum(all_qs[-1,0,believers]) == 0 or np.sum(all_qs[-1,1,nonbelievers]) == 0:
#         cluster_ratio = 0
#     else:
#         cluster_ratio = np.sum(all_qs[-1,0,believers]) / np.sum(all_qs[-1,1,nonbelievers])
#         cluster_ratio = cluster_ratio if cluster_ratio < 1 else 1/cluster_ratio
#     return cluster_ratio

#s = pd.Series(data,index=indices)

#s.to_pickle("param_data.pkl")

# combinations = []
# for i in num_agent_values:
#     for j in connectedness_values:
#         for k in precision_ranges:
#             for l in precision_ranges:
#                 for r in precision_ranges:
#                     combinations.append((i,j,k[0],k[1],l[0],l[1],r[0],r[1]))

#tuples = list(pd.MultiIndex.from_tuples(combinations, names = ["num_agents", "connectedness", "ecb_lower", "ecb_upper", "B_idea_lower", "B_idea_upper", "B_n_lower", "B_n_upper"]))

#indices = pd.MultiIndex.from_tuples(tuples)
#data = np.zeros((num_trials, len(tuples)))
# %%
