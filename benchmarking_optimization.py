# %% Imports

import numpy as np
import networkx as nx
import time

from model.agent import Agent
from simulation.simtools import initialize_agent_params, initialize_network, run_simulation
from model.pymdp import utils

# %% Run simulation for different network sizes

network_sizes = [5,6,7,8]

optimization_options = [[False, False, False], [True, True, False], [True, False, True], [True, True, True]]
# optimization_options = [[True, True, False], [True, False, True], [True, True, True]]

n_trials = 3

h_idea_mapping = utils.softmax(np.eye(2) * 1.0)
ecb_precis = 4.5
env_precision = 8.5
belief_precision = 5.5

T = 5

times_taken = np.zeros( (len(network_sizes), len(optimization_options), n_trials) )
inference_times_taken = np.zeros( (len(network_sizes), len(optimization_options), n_trials) )
control_times_taken = np.zeros( (len(network_sizes), len(optimization_options), n_trials) )

for ii, N in enumerate(network_sizes):

    print(f'Now running network size {N}\n')

    for jj, option_list in enumerate(optimization_options):

        print(f'Now running optimization option set {jj}\n')

        for trial_i in range(n_trials):
            
            G = nx.complete_graph(N)

            optim_options = {'reduce_A': option_list[0], 'reduce_A_inference': option_list[1], 'reduce_A_policies': option_list[2]}

            # construct agent-specific generative model parameters
            agent_constructor_params, store_params = initialize_agent_params(G, h_idea_mappings = h_idea_mapping, \
                                                ecb_precisions = ecb_precis, B_idea_precisions = env_precision, \
                                                    B_neighbour_precisions = belief_precision, variance = 0.2, E_noise = 0.1, optim_options = optim_options)

            # fill node attributes of graph object `G` with agent-specific properties (generative model, Agent class, history of important variables)
            G = initialize_network(G, agent_constructor_params, T = T)

            # start_time = time.time()
            G, inference_time_cost, control_time_cost = run_simulation(G, T = T)
            # end_time = time.time()
            # times_taken[ii,jj,trial_i] = end_time - start_time
            inference_times_taken[ii,jj,trial_i] = inference_time_cost
            control_times_taken[ii,jj,trial_i] = control_time_cost


# %% Run simulation

times_taken.mean(axis=2)




