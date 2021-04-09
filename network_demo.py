# %% Imports

import numpy as np
import networkx as nx
from Model.agent import Agent
from Simulation.simtools import initialize_agent_params, initialize_network, run_simulation, connect_edgeless_nodes, clip_edges, get_initial_observations
from Analysis.analysis_tools import collect_idea_beliefs
from Model.pymdp import maths
from Model.pymdp import utils
import copy
from matplotlib import pyplot as plt
import seaborn as sns

# %% construct network
N, p, T = 5, 0.5, 10
G = nx.fast_gnp_random_graph(N,p) # create the graph for this trial & condition

# make sure graph is connected and all agents have at least one edge
if not nx.is_connected(G):
    G = connect_edgeless_nodes(G) # make sure graph is connected
while np.array(list(G.degree()))[:,1].min() < 2: # make sure no agents with only 1 edge
    G = nx.fast_gnp_random_graph(N,p) # create the graph for this trial & condition
    if not nx.is_connected(G):
        G = connect_edgeless_nodes(G) # make sure graph is 


h_idea_mapping = utils.softmax(np.eye(2) * 1.0)
ecb_precision_range = [4.0, 5.0]
env_determinism_range = [8.0, 9.0]
belief_determinism_range = [5.0, 6.0]

# construct agent-specific generative model parameters
agent_constructor_params = initialize_agent_params(G, h_idea_mappings = h_idea_mapping, ecb_precisions = ecb_precision_range, B_idea_precisions = env_determinism_range, B_neighbour_precisions = belief_determinism_range, reduce_A=True)

# fill node attributes of graph object `G` with agent-specific properties (generative model, Agent class, history of important variables)
G = initialize_network(G, agent_constructor_params, T = T)

# %% Run simulation
G = run_simulation(G, T = T)

# %%

# all_tweets = collect_tweets(G, 'self')
# all_others = collect_tweets(G, 'other')

all_qs = collect_idea_beliefs(G)
# %%
