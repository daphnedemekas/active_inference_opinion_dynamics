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

h_idea_mapping = utils.softmax(np.eye(2) * 1.0)

num_agent_values = [5,10,15]
n = len(num_agent_values)
connectedness_values = [0.2,0.5,0.8]
c = len(connectedness_values)
precision_ranges = [[1,2],[1,5],[1,9],[6,7],[6,10]]
r_len = len(precision_ranges)
num_trials = 5

# %% construct network
iter = 0

all_parameters_to_store = utils.obj_array((n,c,r_len,r_len,r_len))
all_results_to_store = utils.obj_array((n,c,r_len,r_len,r_len))

for trial in range(num_trials):
    for i_n, n in enumerate(num_agent_values):
        for i_p, p in enumerate(connectedness_values):
            N, p, T = n, p, 50
            G = nx.fast_gnp_random_graph(N,p)

            # make sure graph is connected and all agents have at least one edge
            if not nx.is_connected(G):
                G = connect_edgeless_nodes(G) # make sure graph is connected
            while np.array(list(G.degree()))[:,1].min() < 2: # make sure no agents with only 1 edge
                #G = nx.stochastic_block_model(sizes, probs, seed=0) # create the graph for this trial & condition
                G = nx.fast_gnp_random_graph(N,p)

                if not nx.is_connected(G):
                    G = connect_edgeless_nodes(G) # make sure graph is 
            print("graph created")
            for i_e, e_range in enumerate(precision_ranges):
                # range of the uniform distributions
                    ecb_precision_range = e_range

                    for i_env, env_range in enumerate(precision_ranges):
                    
                        env_determinism_range = env_range

                        for i_b, b_range in enumerate(precision_ranges):
                            belief_determinism_range = b_range
        
                            agent_constructor_params, store_params = initialize_agent_params(G, h_idea_mappings = h_idea_mapping, \
                                                        ecb_precisions = ecb_precision_range, B_idea_precisions = env_determinism_range, \
                                                            B_neighbour_precisions = belief_determinism_range, reduce_A=True)
                            all_parameters_to_store[i_n,i_p,i_e,i_env,i_b] = store_params
                            G = initialize_network(G, agent_constructor_params, T = T)
                            G = run_simulation(G, T = T)

                            all_qs = collect_idea_beliefs(G)
                            all_neighbour_samplings = collect_sampling_history(G)
                            adj_mat = nx.to_numpy_array(G)
                            all_tweets = collect_tweets(G)

                            believers = np.where(all_qs[-1,0,:] > 0.5)
                            nonbelievers = np.where(all_qs[-1,0,:] < 0.5)

                            all_results_to_store[i_n,i_p,i_e,i_env,i_b] = (adj_mat, all_qs, all_tweets, all_neighbour_samplings)

                            #choose a metric to store (or more than one)
                            #metric = get_cluster_ratio(all_qs)
                            #data[trial, iter] = metric

                            if iter % 10 ==0:
                                print(iter)
                            iter +=1

np.savez('results/params', all_parameters_to_store)
np.savez('results/all_results', all_results_to_store)






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