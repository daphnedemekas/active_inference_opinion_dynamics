#%%
import numpy as np

def calculate_esteem(all_qs, t, threshold_high = 3, threshold_low = 1):
    """ all_qs is a numpy array of shape (timesteps, idea_levels, num_agents) 
        t: the current timestep
    
    The esteem for each agent will be a functionn of how much their belief differs from the average beliefs of the group"""
    num_agents = all_qs.shape[-1]

    #average of everybody's qs_0 and calculate the difference between the agent's qs_0

    #return the norm of a tuple (float)
    
    #observation = [reward, neutral, rejection]

    #TODO: make the values into global parameters
    agent_esteems = np.zeros(num_agents)

    for agent in num_agents:
        agent_belief = all_qs[t,:,agent] #array of len(idea_levels)
        neighbour_belief_avg = np.average(all_qs[t,:,[np.arange(num_agents)!=agent]], axis = 0) #array of len(idea_levels) 
        agent_std = np.std([agent_belief, neighbour_belief_avg]) #standard deviation between the agent and the norm 

        if agent_std == 0:
            #division by zero 
            raise 

        difference = np.linalg.norm(agent_belief - neighbour_belief_avg)
        num_stds = (difference / agent_std)[0]

        if num_stds < threshold_low: #agent is sufficiently close to the norm 
            esteem = 0  #reward 
        elif num_stds > threshold_high: #agent is sufficiently fara way from the norm 
            esteem = 2  #rejection
        else:
            esteem = 1
        agent_esteems[agent] = esteem 
    return agent_esteems 




# %% Imports

import numpy as np
import networkx as nx
from model.agent import Agent
from Simulation.simtools import initialize_agent_params, initialize_network, run_simulation, connect_edgeless_nodes, clip_edges
from analysis.analysis_tools import collect_idea_beliefs, collect_sampling_history, collect_tweets
from model.pymdp import maths
from model.pymdp import utils
import copy
from matplotlib import pyplot as plt
import seaborn as sns

# %% construct network
N, p, T = 10, 0.6, 35
G = nx.fast_gnp_random_graph(N,p) # create the graph for this trial & condition

# make sure graph is connected and all agents have at least one edge
if not nx.is_connected(G):
    G = connect_edgeless_nodes(G) # make sure graph is connected
while np.array(list(G.degree()))[:,1].min() < 2: # make sure no agents with only 1 edge
    G = nx.fast_gnp_random_graph(N,p) # create the graph for this trial & condition
    #G = nx.stochastic_block_model(sizes, probs, seed=0) # create the graph for this trial & condition
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
G = run_simulation(G, T = T, esteem_model = True)

all_qs = collect_idea_beliefs(G)