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

num_agent_values = [5,8,13]
n = len(num_agent_values)
connectedness_values = [0.2,0.5,0.8]
c = len(connectedness_values)
lower_bounds = [1,4,7]
l = len(lower_bounds)
upper_bounds = [i+j for i in lower_bounds for j in [1,5]]
print(upper_bounds)
u = len(upper_bounds)
arrays = [num_agent_values, connectedness_values, lower_bounds, upper_bounds, lower_bounds, upper_bounds, lower_bounds, upper_bounds]

tuples = list(pd.MultiIndex.from_product(arrays, names = ["num_agents", "connectedness", "ecb_lower", "ecb_upper", "B_idea_lower", "B_idea_upper", "B_n_lower", "B_n_upper"]))

for t in tuples:
    if t[2]<t[3] or t[4] < t[5] or t[6] < t[7]:
        tuples.remove(t)

indices = pd.MultiIndex.from_tuples(tuples)
print(indices)
raise
#dataframe = np.zeros((n,c,l,u,l,u,l,u))
data = []
# %% construct network
iter = 0

all_parameters_to_store = utils.obj_array((n,c,l,u,l,u,l,u))
all_results_to_store = utils.obj_array((n,c,l,u,l,u,l,u))
for i_n, n in enumerate(num_agent_values):
    print("number of agents" + str(n))
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
        for i_e, e_lower in enumerate(lower_bounds):
            for e_r, e_upper in enumerate(upper_bounds):
                if e_lower > e_upper:
                    continue
            # range of the uniform distributions
                ecb_precision_range = [e_lower, e_upper]

                for i_env, env_lower in enumerate(lower_bounds):
                    for env_r, env_upper in enumerate(upper_bounds):
                        if env_lower > env_upper:
                            continue
                        env_determinism_range = [env_lower, env_upper]

                        for i_b, b_lower in enumerate(lower_bounds):
                            for b_r, b_upper in enumerate(upper_bounds):
                                if b_lower > b_upper:
                                    continue
                                belief_determinism_range = [b_lower, b_upper]
            
                                agent_constructor_params, store_params = initialize_agent_params(G, h_idea_mappings = h_idea_mapping, \
                                                            ecb_precisions = ecb_precision_range, B_idea_precisions = env_determinism_range, \
                                                                B_neighbour_precisions = belief_determinism_range, reduce_A=True)
                                all_parameters_to_store[i_n,i_p,i_e,e_r, i_env, env_r, i_b, b_r] = store_params
                                G = initialize_network(G, agent_constructor_params, T = T)
                                G = run_simulation(G, T = T)

                                all_qs = collect_idea_beliefs(G)
                                all_neighbour_samplings = collect_sampling_history(G)
                                adj_mat = nx.to_numpy_array(G)
                                all_tweets = collect_tweets(G)

                                believers = np.where(all_qs[-1,0,:] > 0.5)
                                nonbelievers = np.where(all_qs[-1,0,:] < 0.5)

                                all_results_to_store[i_n,i_p,i_e,e_r, i_env, env_r, i_b, b_r] = (adj_mat, all_qs, all_tweets, all_neighbour_samplings)

                                if np.sum(all_qs[-1,0,believers]) == 0 or np.sum(all_qs[-1,1,nonbelievers]) == 0:
                                    cluster_ratio = 0
                                else:
                                    cluster_ratio = np.sum(all_qs[-1,0,believers]) / np.sum(all_qs[-1,1,nonbelievers])
                                    cluster_ratio = cluster_ratio if cluster_ratio < 1 else 1/cluster_ratio
                                data.append(cluster_ratio)
                                if iter % 10:
                                    print(iter)
                                iter +=1

np.savez('results/params', all_parameters_to_store)
np.savez('results/all_results', all_results_to_store)
s = pd.Series(data,index=indices)

s.to_pickle("param_data.pkl")

