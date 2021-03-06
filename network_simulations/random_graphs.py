#### Script for running opinion dynamics simulations for different settings of the p parameter of the Erdos-Renyi random graph

# %% Imports
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

def initialize_graph_and_agents(G, num_H, idea_levels, h_idea_mapping, belief2tweet_mapping):

    agents_dict = {}
    agents = utils.obj_array(G.number_of_nodes()) # an empty object array of size num_agents
    for i in G.nodes():

        neighbors_i = list(nx.neighbors(G, i)) #each agent has a different number of neighbours 
        num_neighbours = len(neighbors_i)

        #confirmation_bias_param = np.random.uniform(low=2.5, high=5.0) #confirmation bias params? one param per idea level but should be per neighbour
        per_neighbour_cb_params = np.random.uniform(low=4, high=7)*np.ones((num_neighbours, idea_levels))
        env_det_param =  6#how determinstic the environmennt is in general
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
        agents[i] = Agent(**agent_i_params)


    nx.set_node_attributes(G, agents_dict, 'agent')
    return G, agents

def initialize_observation_buffer(G, agents):

    N = G.number_of_nodes()

    observation_buffer = utils.obj_array(N)
    agent_neighbours_global = {}

    for agent_id, agent in enumerate(agents):
        
        agent_neighbours_global[agent_id] = list(nx.neighbors(G, agent_id))
        agent.observations[0] = int(agent.action[-2]) # what I'm tweeting
        agent.observations[-1] = int(agent.action[-1]) # my last observation is who I'm sampling
        which_neighbour_local = agent.action[-1]
        which_neighbour_global = agent_neighbours_global[agent_id][which_neighbour_local] # convert from 'local' (focal-agent-relative) neighbour index to global (network-relative) index

        agent.observations[which_neighbour_local+1] = int(agents[which_neighbour_global].action[-2]) # my neighbour's tweet is my observation in the (neighbour+1)-th modality

        observation_buffer[agent_id] = agent.observations

    return observation_buffer, agent_neighbours_global

def multi_agent_loop(G, agents, T, observation_buffer, agent_neighbours_global):

    N = G.number_of_nodes()

    all_actions = utils.obj_array( (T, N) )
    all_beliefs = utils.obj_array( (T, N) )
    all_observations = utils.obj_array( (T, N) )
    for t in range(T):
        print(str(t) + '/' + str(T))

        all_observations[t,:] = copy.deepcopy(observation_buffer)

        # First loop over agents: Do belief-updating (inference) and action selection
        if t == 0:

            for agent_idx, agent in enumerate(agents):
                agent.infer_states(True, tuple(agent.observations))
                agent.infer_policies()
                action = agent.sample_action()

                all_actions[t,agent_idx] = action[-2:] # we only store the last two control factor actions (what I'm tweeting and who I'm looking at)
                all_beliefs[t,agent_idx] = copy.deepcopy(agent.qs) # deepcopy perhaps not needed here
        else:

            for agent_idx, agent in enumerate(agents):
                qs = agent.infer_states(False, tuple(agent.observations))
                print("qs[0]")
                print(qs[0])
                print()
                agent.infer_policies()
                action = agent.sample_action()

                all_actions[t,agent_idx] = action[-2:] # we only store the last two control factor actions (what I'm tweeting and who I'm looking at)
                all_beliefs[t,agent_idx] = copy.deepcopy(agent.qs) # deepcopy perhaps not needed here
        
        # Second loop over agents: based on what actions everyone selected, now get actions

        for agent_id, agent in enumerate(agents):
            agent.observations = [0]*agent.genmodel.num_modalities
            agent.observations[0] = int(agent.action[-2]) # what I'm tweeting

            which_neighbour_local = int(agent.action[-1])
            which_neighbour_global = agent_neighbours_global[agent_id][which_neighbour_local] # convert from 'local' (focal-agent-relative) neighbour index to global (network-relative) index

            agent.observations[which_neighbour_local+1] = int(agents[which_neighbour_global].action[-2]) # my neighbour's tweet is my observation in the (neighbour+1)-th modality
            agent.observations[-1] = which_neighbour_local
            observation_buffer[agent_id] = agent.observations
        
    
    return G, agents, all_observations, all_beliefs, all_actions
       

# %% Run the parameter sweep

T = 50          # number of timesteps in the simulation
N = 6   # total number of agents in network
idea_levels = 2 # the levels of beliefs that agents can have about the idea (e.g. 'True' vs. 'False', in case `idea_levels` ==2)
num_H = 2       # the number of hashtags, or observations/tweet contents that each agent can tweet & observe from other agents

h_idea_mapping = maths.softmax(np.eye(num_H) * 0.1)
#h_idea_mapping = np.eye(num_H)
#h_idea_mapping[:,0] = maths.softmax(h_idea_mapping[:,0]*0.1)
#h_idea_mapping[:,1] = maths.softmax(h_idea_mapping[:,1]*0.1)


belief2tweet_mapping = np.eye(num_H)

p_vec = np.linspace(0.6,1,1) # different levels of random connection parameter in Erdos-Renyi random graphs
num_trials = 1 # number of trials per level of the ER parameter

for param_idx, p in enumerate(p_vec):

    for trial_i in range(num_trials):

        G = nx.fast_gnp_random_graph(N,p) # create the graph for this trial & condition

        if not nx.is_connected(G):
            G = connect_edgeless_nodes(G) # make sure graph is connected
        
        G, agents = initialize_graph_and_agents(G, num_H, idea_levels, h_idea_mapping, belief2tweet_mapping)
        #G, agents_dict, agents, agent_neighbours_global = create_multiagents(G, N)

        observation_buffer, agent_neighbours_global = initialize_observation_buffer(G, agents)

        G, agents, observation_hist, belief_hist, action_hist = multi_agent_loop(G, agents, T, observation_buffer, agent_neighbours_global)

#how can we quantify the behaviour working other than just making plots? 
#the idea would be that 


    action_h1 = []
    action_h2 = []
    for t in range(T):
        timestep = []
        timestep_b2 = []

        for a in range(N):
            timestep.append(np.sum(action_hist[t,a][-2]))
        action_h1.append(timestep)
        #observation_h2.append(timestep_b2)

    sns.heatmap(np.transpose(np.array(action_h1)), cmap='gray', vmax=1., vmin=0., cbar=True)
    #for a in range(N):
    #    plt.plot(np.array(observation_h2)[:,a], color = "orange")
    plt.title("p = " + str(p))
    plt.show()




    # observation_h1 = []
    # observation_h2 = []
    # for t in range(T):
    #     timestep = []
    #     timestep_b2 = []

    #     for a in range(N):
    #         print(observation_hist[t,a])
    #         timestep.append(np.sum(observation_hist[t,a][:-2]))
    #         timestep_b2.append(len(observation_hist[t,a][:-2]) - np.sum(observation_hist[t,a][:-2]))
    #     observation_h1.append(timestep)
    #     observation_h2.append(timestep_b2)

    # for a in range(N):
    #     plt.plot(np.array(observation_h1)[:,a], color = "red")
    # for a in range(N):
    #     plt.plot(np.array(observation_h2)[:,a], color = "orange")
    # plt.title("p = " + str(p))
    # plt.show()




    belief_in_idea_1 = []
    belief_in_idea_2 = []

    for t in range(T):
        timestep = []
        timestep_b2 = []

        for a in range(N):
            print(belief_hist[t,a])
            timestep.append(belief_hist[t,a][1][0])
            timestep_b2.append(belief_hist[t,a][1][1])
        belief_in_idea_1.append(timestep)
        belief_in_idea_2.append(timestep_b2)

    print(belief_in_idea_1)
    for a in range(N):
        print(a)
        plt.plot(np.array(belief_in_idea_1)[:,a], color = "blue")
    for a in range(N):
        print(a)
        plt.plot(np.array(belief_in_idea_2)[:,a], color = "green")
    plt.title("p = " + str(p))
    plt.show()
for a in belief_in_idea_1:
    plt.plot(T, a)
plt.show()
sns.heatmap(belief_hist, cmap='gray', vmax=1., vmin=0., cbar=True)
plt.savefig('belief_hist.png',dpi=325)


sns.heatmap(action_hist, cmap='gray', vmax=1., vmin=0., cbar=True)
plt.savefig('action_hist.png',dpi=325)