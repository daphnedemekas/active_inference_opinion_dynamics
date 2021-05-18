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

# %% construct network
N, p, T = 10, 0.6, 35
G = nx.fast_gnp_random_graph(N,p) # create the graph for this trial & condition

# stochastic block model
# sizes = [7, 7] # two communities
# probs = [[0.8, 0.1], [0.1, 0.9]] # connection probabilities within and between communities
# G = nx.stochastic_block_model(sizes, probs, seed=0) # generate the model

# make sure graph is connected and all agents have at least one edge
if not nx.is_connected(G):
    G = connect_edgeless_nodes(G) # make sure graph is connected
while np.array(list(G.degree()))[:,1].min() < 2: # make sure no agents with only 1 edge
    # G = nx.fast_gnp_random_graph(N,p) # create the graph for this trial & condition
    G = nx.stochastic_block_model(sizes, probs, seed=0) # create the graph for this trial & condition
    if not nx.is_connected(G):
        G = connect_edgeless_nodes(G) # make sure graph is 

h_idea_mapping = utils.softmax(np.eye(2) * 1.0)
ecb_precis = 4.5
env_precision = 8.5
belief_precision = 5.5

# construct agent-specific generative model parameters
agent_constructor_params, store_params = initialize_agent_params(G, h_idea_mappings = h_idea_mapping, \
                                    ecb_precisions = ecb_precis, B_idea_precisions = env_precision, \
                                        B_neighbour_precisions = belief_precision, reduce_A=True)

# fill node attributes of graph object `G` with agent-specific properties (generative model, Agent class, history of important variables)
G = initialize_network(G, agent_constructor_params, T = T)

# %% Run simulation
G = run_simulation(G, T = T)

all_qs = collect_idea_beliefs(G)

plt.plot(all_qs[:,0,:])

all_neighbour_samplings = collect_sampling_history(G)
all_tweets = collect_tweets(G)

believers = np.where(all_qs[-1,0,:] > 0.5)
nonbelievers = np.where(all_qs[-1,0,:] < 0.5)

adj_mat = nx.to_numpy_array(G)

# %%
# from tempfile import TemporaryFile
# outfile = TemporaryFile()

fname = 'results/sbm_test' 
np.savez(fname, adj_mat, all_qs, all_tweets)


