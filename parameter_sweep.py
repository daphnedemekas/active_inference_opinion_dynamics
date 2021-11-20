# %% Imports

import numpy as np
import networkx as nx
from Model.agent import Agent
from Simulation.simtools import generate_network, initialize_agent_params, initialize_network, run_simulation, connect_edgeless_nodes, clip_edges
from Analysis.analysis_tools import collect_idea_beliefs, collect_sampling_history, collect_tweets
from Model.pymdp import maths
from Model.pymdp import utils
from matplotlib import pyplot as plt
from Analysis.plots import *
#import pandas as pd 
import itertools
import os
#import multiprocessing

def run_sweep(param_combos):
    iter = 0
    for p_idx, param_config in enumerate(param_combos):
        if p_idx < 33:
            continue
        print(p_idx)
        print(param_config)

        num_agents_i, connectedness_i, ecb_p_i, env_precision_i, b_precision_i, v_i, lr_i = param_config
        N, p, T = num_agents_i, connectedness_i, 50
        G = generate_network(N,p)

        if not os.path.isdir('Analysis/ecb_lr/' + str(p_idx)  +"/"):
            os.mkdir('Analysis/ecb_lr/' + str(p_idx)+"/")

        for trial_i in range(n_trials):
            #if trial_i < 30:
            #    continue

            agent_constructor_params, _ = initialize_agent_params(G, h_idea_mappings = h_idea_mapping, \
                                        ecb_precisions = ecb_p_i, B_idea_precisions = env_precision_i, \
                                            B_neighbour_precisions = b_precision_i, variance = v_i, E_noise = lr_i)
            G = initialize_network(G, agent_constructor_params, T = T)
            G, _, _ = run_simulation(G, T = T)
            
            all_qs = collect_idea_beliefs(G)
            all_neighbour_samplings = collect_sampling_history(G)
        
            adj_mat = nx.to_numpy_array(G)
            all_tweets = collect_tweets(G)

            trial_results = np.array([adj_mat, all_qs, all_tweets, all_neighbour_samplings], dtype=object)


            np.savez('Analysis/ecb_lr/' + str(p_idx) + "/" + str(trial_i) , trial_results)

            iter +=1

if __name__ == '__main__':

    h_idea_mapping = utils.softmax(np.array([[1,0],[0,1]])* np.random.uniform(0.3,2.1))

    connectedness_values = [0.3]
    #connectedness_values = np.linspace(0.2,0.8,15)
    ecb_precision_gammas = np.linspace(3.5,9,15)
    #ecb_precision_gammas = np.append(ecb_precision_gammas, False)
    num_agent_values = [12]

    n = len(num_agent_values)
    c = len(connectedness_values)
    env_precision_gammas = [9]
    b_precision_gammas = [6]
    #b_precision_gammas = np.linspace(0.05,0.8,15)
    lr = np.linspace(0.05,0.9,15)
    variances = [0.1]
    r_len = len(ecb_precision_gammas)
    e_len = len(env_precision_gammas)
    b_len = len(b_precision_gammas)
    v_len = len(variances)
    lr_len = len(lr)
    n_trials = 50

    param_combos = itertools.product(num_agent_values, connectedness_values, ecb_precision_gammas,env_precision_gammas,b_precision_gammas, variances, lr)
    # %% construct network
    length = len(list(param_combos))
    print("number of combinations: " +str(length))
    param_combos = itertools.product(num_agent_values, connectedness_values, ecb_precision_gammas,env_precision_gammas,b_precision_gammas, variances, lr)

    run_sweep(param_combos)
    
    
    