# %% Imports

import numpy as np
import networkx as nx
from model.agent import Agent
from simulation.simtools import generate_network, initialize_agent_params, initialize_network, run_simulation, connect_edgeless_nodes, clip_edges
from analysis.analysis_tools import collect_idea_beliefs, collect_sampling_history, collect_tweets
from model.pymdp import maths
from model.pymdp import utils
from matplotlib import pyplot as plt
from analysis.plots import *
#import pandas as pd
import itertools
import os
#import multiprocessing

def run_sweep(param_combos):
    iter = 0
    for p_idx, param_config in enumerate(param_combos):
        if p_idx < 163 or p_idx > 175:
            continue

        num_agents_i, connectedness_i, ecb_p_i, env_precision_i, b_precision_i, lr_i = param_config
        N, p, T = num_agents_i, connectedness_i, 100
        G = generate_network(N,p)

 
        if not os.path.isdir('analysis/hyp1_results/' + str(p_idx)  +"/"):
            os.mkdir('analysis/hyp1_results/' + str(p_idx)+"/")
        for trial_i in range(n_trials):
            if p_idx == 163:
                if trial_i < 41:
                    continue

            agent_constructor_params = initialize_agent_params(G, h_idea_mapping = h_idea_mapping, \
                                        ecb_precision = ecb_p_i, B_idea_precisions = env_precision_i, \
                                            B_neighbour_precisions = b_precision_i, E_noise = lr_i)
            G = initialize_network(G, agent_constructor_params, T = T, model = None)
            G  = run_simulation(G, T = T)
            all_qs = collect_idea_beliefs(G)
            all_neighbour_samplings = collect_sampling_history(G)
            adj_mat = nx.to_numpy_array(G)
            all_tweets = collect_tweets(G)

            trial_results = np.array([adj_mat, all_qs, all_tweets, all_neighbour_samplings], dtype=object)


            np.savez('analysis/hyp1_results/' + str(p_idx) + "/" + str(trial_i) , trial_results)

            iter +=1

if __name__ == '__main__':

    h_idea_mapping = utils.softmax(np.array([[1,0],[0,1]])* np.random.uniform(0.3,2.1))

    connectedness_values = np.linspace(0.2,0.8,15)
    ecb_precision_gammas = np.linspace(3,9,15)
    num_agent_values = [15]

    n = len(num_agent_values)
    c = len(connectedness_values)
    env_precision_gammas = [9]
    b_precision_gammas = [0.6]
    lr = [0]
    r_len = len(ecb_precision_gammas)
    e_len = len(env_precision_gammas)
    b_len = len(b_precision_gammas)
    lr_len = len(lr)
    n_trials = 100

    param_combos = itertools.product(num_agent_values, connectedness_values, ecb_precision_gammas,env_precision_gammas,b_precision_gammas, lr)
    # %% construct network
    length = len(list(param_combos))
    print("number of combinations: " +str(length))
    param_combos = itertools.product(num_agent_values, connectedness_values, ecb_precision_gammas,env_precision_gammas,b_precision_gammas, lr)

    run_sweep(param_combos)
