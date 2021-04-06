#### Script for running opinion dynamics simulations for different settings of the p parameter of the Erdos-Renyi random graph

# %% Imports

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import networkx as nx
from Model.agent import Agent
from Model.network_tools import create_multiagents, clip_edges, connect_edgeless_nodes
from Model.pymdp import maths
from Model.pymdp import utils
import copy
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import pyplot as plt
# %% Helper functions

def initialize_graph_and_agents(G, num_H, idea_levels, h_idea_mapping, belief2tweet_mapping, reduce_A = False):

    agents_dict = {}
    agents = utils.obj_array(G.number_of_nodes()) # an empty object array of size num_agents
    for i in G.nodes():

        neighbors_i = list(nx.neighbors(G, i)) #each agent has a different number of neighbours 
        num_neighbours = len(neighbors_i)

        #confirmation_bias_param = np.random.uniform(low=2.5, high=5.0) #confirmation bias params? one param per idea level but should be per neighbour
        per_neighbour_cb_params = np.random.uniform(low=4, high=7)*np.ones((num_neighbours, idea_levels))
        env_det_param =  6 #how determinstic the environmennt is in general
        belief_det_params = np.random.uniform(low=3.0, high=9.0, size=(num_neighbours,)) #how deterministic the nieghbours are specifically
        initial_tweet, initial_neighbour_to_sample = np.random.randint(num_H), np.random.randint(num_neighbours) 
        agent_i_params = {

            "neighbour_params" : {
                "precisions" :  per_neighbour_cb_params,
                "num_neighbours" : num_neighbours,
                "env_determinism": env_det_param,
                "belief_determinism": belief_det_params
                },

            "idea_mapping_params" : {
                "num_H" : num_H,
                "idea_levels": idea_levels,
                "h_idea_mapping": h_idea_mapping
                },

            "policy_params" : {
                "initial_action" : [initial_tweet, initial_neighbour_to_sample],
                "belief2tweet_mapping" : belief2tweet_mapping
                },

            "C_params" : {
                "preference_shape" : None,
                "cohesion_exp" : None,
                "cohesion_temp" : None
                }
        }

        agents_dict[i] = agent_i_params
        agents[i] = Agent(**agent_i_params, reduce_A=reduce_A)


    nx.set_node_attributes(G, agents_dict, 'agent')
    return G, agents

def initialize_observation_buffer(G, agents):

    N = G.number_of_nodes()

    observation_buffer = utils.obj_array(N)
    agent_neighbours_global = {}

    for agent_id, agent in enumerate(agents):

        agent_neighbours_global[agent_id] = np.array(list(nx.neighbors(G, agent_id)))

        initial_obs_agent_i = np.zeros(agent.genmodel.num_modalities,dtype=int)
        initial_obs_agent_i[0] = int(agent.initial_action[0]) # what I'm tweeting
        initial_obs_agent_i[-1] = int(agent.initial_action[-1]) # my last observation is who I'm sampling

        which_neighbour_local = agent.initial_action[1]
        which_neighbour_global = agent_neighbours_global[agent_id][which_neighbour_local] # convert from 'local' (focal-agent-relative) neighbour index to global (network-relative) index

        initial_obs_agent_i[which_neighbour_local+1] = int(agents[which_neighbour_global].initial_action[0]+1) # my neighbour's tweet is my observation in the (neighbour+1)-th modality

        observation_buffer[agent_id] = np.copy(initial_obs_agent_i)
    
    return observation_buffer, agent_neighbours_global

def multi_agent_loop(G, agents, T, observation_buffer, agent_neighbours_global):

    N = G.number_of_nodes()

    all_actions = utils.obj_array( (T, N) )
    all_beliefs = utils.obj_array( (T, N) )
    all_observations = utils.obj_array( (T, N) )

    for t in range(T):

        all_observations[t,:] = copy.deepcopy(observation_buffer)

        # First loop over agents: Do belief-updating (inference) and action selection
        if t == 0:

            for agent_id, agent in enumerate(agents):
                qs = agent.infer_states(True, tuple(observation_buffer[agent_id]))
                agent.infer_policies(qs)
                action = agent.sample_action()
                all_actions[t,agent_id] = np.copy(action[-2:]) # we only store the last two control factor actions (what I'm tweeting and who I'm looking at)
                all_beliefs[t,agent_id] = copy.deepcopy(qs) # deepcopy perhaps not needed here

        else:

            for agent_id, agent in enumerate(agents):
                qs = agent.infer_states(False, tuple(observation_buffer[agent_id]))
                agent.infer_policies(qs)
                action = agent.sample_action()
                all_actions[t,agent_id] = np.copy(action[-2:]) # we only store the last two control factor actions (what I'm tweeting and who I'm looking at)
                all_beliefs[t,agent_id] = copy.deepcopy(qs) # deepcopy perhaps not needed here
        
        # Second loop over agents: based on what actions everyone selected, now get actions

        observation_buffer = utils.obj_array(N) # reset the buffer

        for agent_id, agent in enumerate(agents):

            obs_agent_i = np.zeros(agent.genmodel.num_modalities,dtype=int)
            obs_agent_i[0] = int(agent.action[-2]) # what I'm tweeting
            obs_agent_i[-1] = int(agent.action[-1]) # who I'm sampling

            which_neighbour_local = int(agent.action[-1])
            which_neighbour_global = agent_neighbours_global[agent_id][which_neighbour_local] # convert from 'local' (focal-agent-relative) neighbour index to global (network-relative) index

            obs_agent_i[which_neighbour_local+1] = int(agents[which_neighbour_global].action[-2]+1) # my neighbour's tweet is my observation in the (neighbour+1)-th modality

            observation_buffer[agent_id] = np.copy(obs_agent_i)
    
    return G, agents, all_observations, all_beliefs, all_actions
       

# %% Run the parameter sweep

T = 10           # number of timesteps in the simulation
N = 10           # total number of agents in network
idea_levels = 2  # the levels of beliefs that agents can have about the idea (e.g. 'True' vs. 'False', in case `idea_levels` ==2)
num_H = 2        # the number of hashtags, or observations/tweet contents that each agent can tweet & observe from other agents

h_idea_mapping_base = maths.softmax(np.eye(num_H) * 1.0)

belief2tweet_mapping = np.eye(num_H)

p_vec = np.linspace(0.1,1,10) # different levels of random connection parameter in Erdos-Renyi random graphs
num_trials = 10 # number of trials per level of the ER parameter

for param_idx, p in enumerate(p_vec):

    for trial_i in range(num_trials):

        G = nx.fast_gnp_random_graph(N,p) # create the graph for this trial & condition

        if not nx.is_connected(G):
            G = connect_edgeless_nodes(G) # make sure graph is connected

        while np.array(list(G.degree()))[:,1].min() < 2: # make sure no agents with only 1 edge

            G = nx.fast_gnp_random_graph(N,p) # create the graph for this trial & condition

            if not nx.is_connected(G):
                G = connect_edgeless_nodes(G) # make sure graph is 

        G, agents = initialize_graph_and_agents(G, num_H, idea_levels, h_idea_mapping_base, belief2tweet_mapping, reduce_A = True)

        observation_buffer, agent_neighbours_global = initialize_observation_buffer(G, agents)

        G, agents, observation_hist, belief_hist, action_hist = multi_agent_loop(G, agents, T, observation_buffer, agent_neighbours_global)
# %%

T = 25          # number of timesteps in the simulation
N = 10      # total number of agents in network
idea_levels = 2 # the levels of beliefs that agents can have about the idea (e.g. 'True' vs. 'False', in case `idea_levels` ==2)
num_H = 2       # the number of hashtags, or observations/tweet contents that each agent can tweet & observe from other agents

h_idea_mapping_base = maths.softmax(np.eye(num_H) * 1.0)

belief2tweet_mapping = np.eye(num_H)

G = nx.complete_graph(N) # create the graph for this trial & condition

# G = nx.fast_gnp_random_graph(N,0.2) # create the graph for this trial & condition

# if not nx.is_connected(G):
#     G = connect_edgeless_nodes(G) # make sure graph is 

# while np.array(list(G.degree()))[:,1].min() < 2:

#     G = nx.fast_gnp_random_graph(N,0.2) # create the graph for this trial & condition

#     if not nx.is_connected(G):
#         G = connect_edgeless_nodes(G) # make sure graph is 

G, agents = initialize_graph_and_agents(G, num_H, idea_levels, h_idea_mapping_base, belief2tweet_mapping, reduce_A = True)

observation_buffer, agent_neighbours_global = initialize_observation_buffer(G, agents)

G, agents, observation_hist, belief_hist, action_hist = multi_agent_loop(G, agents, T, observation_buffer, agent_neighbours_global)

beliefs_idea_1 = np.zeros((T, N))
for t_idx in range(T):
    for n in range(N):
        beliefs_idea_1[t_idx,n] = belief_hist[t_idx,n][0][0]

plt.plot(beliefs_idea_1)

# %%
