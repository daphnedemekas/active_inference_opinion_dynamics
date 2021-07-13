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
        #start = time.time()

        print(param_config)

        num_agents_i, connectedness_i, ecb_p_i, env_precision_i, b_precision_i, v_i, lr_i = param_config
        N, p, T = num_agents_i, connectedness_i, 100
        G = generate_network(N,p)

        if not os.path.isdir('Analysis/results/lr_br_results_structure/' + str(p_idx)  +"/"):
            os.mkdir('Analysis/results/lr_br_results_structure/' + str(p_idx)+"/")

        for trial_i in range(n_trials):
            print(trial_i)
            indices = (trial_i, num_agent_values.index(num_agents_i), connectedness_values.index(connectedness_i), ecb_precision_gammas.index(ecb_p_i), env_precision_gammas.index(env_precision_i), b_precision_gammas.index(b_precision_i), variances.index(v_i), lr.index(lr_i))

            agent_constructor_params, _ = initialize_agent_params(G, h_idea_mappings = h_idea_mapping, \
                                        ecb_precisions = ecb_p_i, B_idea_precisions = env_precision_i, \
                                            B_neighbour_precisions = b_precision_i, variance = v_i, E_noise = lr_i)
            G = initialize_network(G, agent_constructor_params, T = T)
            G, _, _ = run_simulation(G, T = T)

            #print("Inference time: " + str(inference_time))
            

            all_qs = collect_idea_beliefs(G)
            all_neighbour_samplings = collect_sampling_history(G)
        
            adj_mat = nx.to_numpy_array(G)
            all_tweets = collect_tweets(G)

            trial_results = np.array([adj_mat, all_qs, all_tweets, all_neighbour_samplings], dtype=object)


            np.savez('Analysis/results/lr_br_results_structure/' + str(p_idx) + "/" + str(trial_i) , trial_results)

            iter +=1
        #print("time taken for config sweep: " + str(time.time() - start))

    #np.savez('Analysis/results/reduced_vectorized_params', all_parameters_to_store)


if __name__ == '__main__':

    h_idea_mapping = utils.softmax(np.array([[1,0],[0,1]])* np.random.uniform(0.3,2.1))


    connectedness_values = [0.3,0.5,0.7]
    #ecb_precision_gammas = [2.2,2.4,2.6,2.8,3]
    ecb_precision_gammas = [4,5,6,7,8]
    ecb_precision_gammas = [7]
    connectedness_values = [0.4]

    #num_agent_values = [3,5,8]
    #num_agent_values = [5]
    num_agent_values = [15]

    n = len(num_agent_values)
    c = len(connectedness_values)
    env_precision_gammas = [9]
    #b_precision_gammas = [7]
    b_precision_gammas = [3,4,5,6,7,8]
    lr = [0.3]
    lr = [0.001,0.1,0.3,0.5,0.7,0.9,1.2]

    #variances = [0.01, 0.1, 0.5, 0.8, 1.1, 1.5, 2]
    variances = [0.1]
    r_len = len(ecb_precision_gammas)
    e_len = len(env_precision_gammas)
    b_len = len(b_precision_gammas)
    v_len = len(variances)
    lr_len = len(lr)
    n_trials = 40

    param_combos = itertools.product(num_agent_values, connectedness_values, ecb_precision_gammas,env_precision_gammas,b_precision_gammas, variances, lr)
    # %% construct network
    length = len(list(param_combos))
    print("number of combinations: " +str(length))
    param_combos = itertools.product(num_agent_values, connectedness_values, ecb_precision_gammas,env_precision_gammas,b_precision_gammas, variances, lr)

    #all_parameters_to_store = utils.obj_array((n_trials, n,c,r_len,e_len,b_len, v_len, lr_len))
    #all_results_to_store = utils.obj_array((n_trials, n,c,r_len,e_len,b_len, v_len, lr_len))

    run_sweep(param_combos)
    
    
    