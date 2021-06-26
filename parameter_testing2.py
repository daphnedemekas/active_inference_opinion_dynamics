# %% Imports

import numpy as np
import networkx as nx
from Model.agent import Agent
from Simulation.simtools import initialize_agent_params, initialize_network, run_simulation, connect_edgeless_nodes, clip_edges
from Analysis.analysis_tools import collect_idea_beliefs, collect_sampling_history, collect_tweets
from Model.pymdp import maths
from Model.pymdp import utils
from matplotlib import pyplot as plt
from Analysis.plots import *
#import pandas as pd 
import itertools
import time 
#import multiprocessing

def run_sweep(param_combos,all_parameters_to_store,all_results_to_store):
    iter = 0
    for param_config in param_combos:
        print(iter)
        print(param_config)

        num_agents_i, connectedness_i, ecb_p_i, env_precision_i, b_precision_i, v_i, lr_i = param_config
        N, p, T = num_agents_i, connectedness_i, 100

        all_trials_ecbs = np.random.normal(ecb_p_i,size=(n_trials,)) # one random gamma distributed precision for each trial
        all_trials_envs = np.random.normal(env_precision_i,size=(n_trials,))
        all_trials_bs = np.random.normal(b_precision_i,size=(n_trials,))
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
            indices = (trial_i, num_agent_values.index(num_agents_i), connectedness_values.index(connectedness_i), ecb_precision_gammas.index(ecb_p_i), env_precision_gammas.index(env_precision_i), b_precision_gammas.index(b_precision_i), variances.index(v_i), lr.index(lr_i))

            ecb_precisions = np.random.gamma(shape = np.absolute(all_trials_ecbs[trial_i]))
            env_precisions = np.random.gamma(shape= np.absolute(all_trials_envs[trial_i]))
            b_precisions = np.random.gamma(shape = np.absolute(all_trials_bs[trial_i]))

            agent_constructor_params, store_params = initialize_agent_params(G, h_idea_mappings = h_idea_mapping, \
                                        ecb_precisions = ecb_p_i, B_idea_precisions = env_precision_i, \
                                            B_neighbour_precisions = b_precision_i, variance = v_i, E_noise = lr_i)
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
            plot_beliefs_over_time(all_qs)
            plt.show()
            #if iter % 10 ==0:
                 
            #    print(str(iter) + "/" + str(length*30))              
            #    np.savez('Analysis/results/E_p2', all_parameters_to_store)
            #    np.savez('Analysis/results/E_r2', all_results_to_store)

            iter +=1

    np.savez('Analysis/results/HK_model_uniform', all_parameters_to_store)
    np.savez('Analysis/results/HK_model_uniform', all_results_to_store)
    return all_parameters_to_store,all_results_to_store


if __name__ == '__main__':

    h_idea_mapping = utils.softmax(np.array([[1,0],[0,1]])* np.random.uniform(0.3,2.1))


    connectedness_values = [0.4]
    ecb_precision_gammas = [7]

    #num_agent_values = [3,5,8]
    num_agent_values = [6]

    n = len(num_agent_values)
    c = len(connectedness_values)
    env_precision_gammas = [9]
    b_precision_gammas = [4,7]
    lr = [0.1,1]

    variances = [0.1,1]

    r_len = len(ecb_precision_gammas)
    e_len = len(env_precision_gammas)
    b_len = len(b_precision_gammas)
    v_len = len(variances)
    lr_len = len(lr)
    n_trials = 5

    param_combos = itertools.product(num_agent_values, connectedness_values, ecb_precision_gammas,env_precision_gammas,b_precision_gammas, variances, lr)
    # %% construct network
    length = len(list(param_combos))
    print("number of combinations: " +str(length))
    param_combos = itertools.product(num_agent_values, connectedness_values, ecb_precision_gammas,env_precision_gammas,b_precision_gammas, variances, lr)

    all_parameters_to_store = utils.obj_array((n_trials, n,c,r_len,e_len,b_len, v_len, lr_len))
    all_results_to_store = utils.obj_array((n_trials, n,c,r_len,e_len,b_len, v_len, lr_len))

    indices = np.linspace(0,length,10)
    processes = []
    run_sweep(param_combos, all_parameters_to_store, all_results_to_store)
    
    
    