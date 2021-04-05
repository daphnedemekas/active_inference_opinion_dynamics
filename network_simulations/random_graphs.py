#### Script for running opinion dynamics simulations for different settings of the p parameter of the Erdos-Renyi random graph

# %% Imports
import numpy as np
import networkx as nx
from Model.agent import Agent
from Model.network_tools import connect_edgeless_nodes
from Model.pymdp import maths
from Model.pymdp import utils

from matplotlib import pyplot as plt

# %% Helper functions

def initialize_graph_and_agents(G, num_H, idea_levels, h_idea_mapping, belief2tweet_mapping):

    agents_dict = {}
    agents = utils.obj_array(G.number_of_nodes())

    for i in G.nodes():

        neighbors_i = list(nx.neighbors(G, i))
        num_neighbours = len(neighbors_i)

        confirmation_bias_param = np.random.uniform(low=2.5, high=5.0)
        env_det_param = np.random.uniform(low = 1.0, high = 5.0)
        belief_det_params = np.random.uniform(low=5.0, high=10.0, size=(num_neighbours,))
        initial_tweet, initial_neighbour_to_sample = np.random.randint(num_H), np.random.randint(num_neighbours)

        agent_i_params = {

            "neighbour_params" : {
                "precisions" :  confirmation_bias_param * np.ones(idea_levels),
                "num_neighbours" : num_neighbours,
                "env_determinism": env_det_param,
                "belief_determinism": belief_det_params
                },

            "idea_mapping_params" : {
                "num_H" : num_H,
                "idea_levels": idea_levels,
                "h_idea_mapping": h_idea_mapping_base
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
        agents[i] = Agent(**agent_i_params)

    nx.set_node_attributes(G, agents_dict, 'agent')

    return G, agents

def initialize_observation_buffer(G, agents):

    N = G.number_of_nodes()

    observation_buffer = utils.obj_array(N)
    agent_neighbours = [None] * N

    for agent_id, agent in enumerate(agents):

        agent_neighbours_global[agent_id] = list(nx.neighbors(G, agent_id))

        initial_obs_agent_i = [0] * agent.genmodel.num_modalities
        initial_obs_agent_i[0] = agent.initial_action[0] # what I'm tweeting

        which_neighbour_local = agent.initial_action[1]
        which_neighbour_global = agent_neighbours_global[agent_id][which_neighbour_local] # convert from 'local' (focal-agent-relative) neighbour index to global (network-relative) index

        initial_obs_agent_i[which_neighbour_local+1] = agents[which_neighbour_global].initial_actions[0] # my neighbour's tweet is my observation in the (neighbour+1)-th modality

        observation_buffer[agent_id] = initial_obs_agent_i
    
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

            for agent_idx, agent in enumerate(agents):
                agent.infer_states(True, tuple(observation_buffer[agent_idx]))
                agent.infer_policies()
                action = agent.sample_action()
                all_actions[t,agent_idx] = action[-2:] # we only store the last two control factor actions (what I'm tweeting and who I'm looking at)
                all_beliefs[t,agent_idx] = copy.deepcopy(agent.qs) # deepcopy perhaps not needed here

        else:

            for agent_idx, agent in enumerate(agents):

                qs = agent.infer_states(False, tuple(observation_buffer[agent_idx]))
                agent.infer_policies()
                action = agent.sample_action()
                all_actions[t,agent_idx] = action[-2:] # we only store the last two control factor actions (what I'm tweeting and who I'm looking at)
                all_beliefs[t,agent_idx] = copy.deepcopy(agent.qs) # deepcopy perhaps not needed here
        
        # Second loop over agents: based on what actions everyone selected, now get actions

        for agent_id, agent in enumerate(agents):

            obs_agent_i = [0] * agent.genmodel.num_modalities
            obs_agent_i[0] = agent.action[-2] # what I'm tweeting

            which_neighbour_local = agent.action[-1]
            which_neighbour_global = agent_neighbours_global[agent_id][which_neighbour_local] # convert from 'local' (focal-agent-relative) neighbour index to global (network-relative) index

            obs_agent_i[which_neighbour_local+1] = agents[which_neighbour_global].action[-2] # my neighbour's tweet is my observation in the (neighbour+1)-th modality

            observation_buffer[agent_id] = obs_agent_i
    
        return G, agents, all_observations, all_beliefs, all_actions
       

# %% Run the parameter sweep

T = 25          # number of timesteps in the simulation
N = 8           # total number of agents in network
idea_levels = 2 # the levels of beliefs that agents can have about the idea (e.g. 'True' vs. 'False', in case `idea_levels` ==2)
num_H = 2       # the number of hashtags, or observations/tweet contents that each agent can tweet & observe from other agents

h_idea_mapping_base = maths.softmax(np.eye(num_H) * 1.0)
belief2tweet_mapping = np.eye(num_H)

p_vec = np.linspace(0.1,1.0,5) # different levels of random connection parameter in Erdos-Renyi random graphs
num_trials = 1 # number of trials per level of the ER parameter

for param_idx, p in enumerate(p_vec):

    for trial_i in range(num_trials):

        G = nx.fast_gnp_random_graph(N,p) # create the graph for this trial & condition

        if not nx.is_connected(G):
            G = connect_edgeless_nodes(G) # make sure graph is connected
        
        G, agents = initialize_graph_and_agents(G, num_H, idea_levels, h_idea_mapping_base, belief2tweet_mapping)

        observation_buffer, agent_neighbours_global = initialize_observation_buffer(G, agents)

        G, agents, observation_hist, belief_hist, action_hist = multi_agent_loop(G, agents, T, observation_buffer, agent_neighbours_global)



        