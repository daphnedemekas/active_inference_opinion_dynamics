import numpy as np
from Model.genmodel import GenerativeModel
from Model.agent import Agent
import networkx as nx
from Model.pymdp import utils
from Model.pymdp.utils import obj_array, index_list_to_onehots, sample, reduce_a_matrix
from Model.pymdp.maths import softmax, spm_dot, spm_log, get_joint_likelihood, calc_free_energy
from Model.pymdp.inference import update_posterior_states
from Analysis.analysis_tools import collect_idea_beliefs, collect_sampling_history, collect_tweets

import seaborn as sns
from matplotlib import pyplot as plt

def create_multiagents(G, N , idea_levels = 2, num_H = 2, precision_params = None, env_determinism = None, belief_determinism = None):
    """
    Populates a networkx graph object G with N active inference agents
    """

    agents_dict = {}

    if env_determinism is None:
        env_determinism = 10  # min and max values of uniform distribution over environmental determinism parameters
    
    if belief_determinism is None:
        belief_determinism = 5 # min and max values of uniform distribution over neighbour-belief determinism parameters
    
    for i in G.nodes():
        num_neighbours = G.degree(i)
        initial_tweet, initial_neighbour_to_sample = np.random.randint(num_H), np.random.randint(num_neighbours) 
        agent_i_params = {

            "neighbour_params" : {
                "ecb_precisions" :  5*np.ones((num_neighbours, idea_levels)),
                "num_neighbours" : num_neighbours,
                "env_determinism":  10,
                "belief_determinism": 5*np.ones(num_neighbours,)
                },

            "idea_mapping_params" : {
                "num_H" : num_H,
                "idea_levels": idea_levels,
                "h_idea_mapping": None
                },

            "policy_params" : {
                "initial_action" : [initial_tweet, initial_neighbour_to_sample],
                "belief2tweet_mapping" : np.eye(num_H)
                },

            "C_params" :  {
                "preference_shape" : None,
                "cohesion_exp" : None,
                "cohesion_temp" : None
                },
            "reduce_A": True
        }

        agents_dict[i] = agent_i_params
    nx.set_node_attributes(G, agents_dict, 'agent')
    agents = []

    agent_neighbours = {}

    for agent_i in G.nodes():
        agent_neighbours[agent_i] = list(nx.neighbors(G, agent_i))
        agent = Agent(**agents_dict[agent_i])
        agents.append(agent)
        
    return G, agents_dict, agents, agent_neighbours

def connect_edgeless_nodes(G):
    """
    This function ensures there are no nodes with only 1 edge in the graph
    """

    for node_i in G.nodes():
        connected = [target for (source, target) in G.edges(node_i)]
        while len(connected) <= 1:
            candidate_additions = [n_i for (n_i) in G.nodes()]
            candidate_additions.pop(node_i)
            addition = random.choice(candidate_additions)
            G.add_edge(node_i,addition)
            print('\tEdge added:\t %d -- %d'%(node_i, addition))
            connected = [target for (source, target) in G.edges(node_i)]
    
    return G


def agent_loop(agent, observations = None, initial = False, initial_action = None):  
    if initial == True:
        t = 0
    else:
        t = 1
    qs = agent.infer_states(t, tuple(observations))
    policy = agent.infer_policies()
    action = agent.sample_action()[-2:]
    who_i_looked_at = int(action[-1]+1)
    what_they_tweeted = observations[int(action[-1])+1]

    if initial == True:
        action = agent.action
    return action, qs

def multi_agent_loop(T, agents, agent_neighbours_local):
    observation_t = [[None] * agent.genmodel.num_modalities for agent in agents] 

    initial = True
    all_actions = obj_array((T,N))
    all_beliefs = obj_array((T,N))
    all_observations = obj_array((T,N))

    for t in range(T):
        print(str(t) + "/" + str(T))
        actions = []
        for i in range(len(agents)):
            if i == N-2:
                all_actions[t,i] = [1,None]
                actions.append([1,None])
            elif i == N-1:
                all_actions[t,i] = [0,None]
                actions.append([0,None])
            else:
                action, belief = agent_loop(agents[i], observation_t[i], initial)
                all_actions[t,i] = action
                all_beliefs[t,i] = belief
                actions.append(action)

        initial = False

        for idx, agent in enumerate(agents):
            if idx == N-2:
                my_tweet = 1
                observation_t[idx][0] = my_tweet
            elif idx == N-1:
                my_tweet = 0
                observation_t[idx][0] = my_tweet
            else:
                for n in range(agent.genmodel.num_neighbours):
                    observation_t[idx][n+1] = 0
                    my_tweet = int(actions[idx][-2])
                    observation_t[idx][0] = my_tweet
                    observed_neighbour = int(actions[idx][-1])
                    observed_agent = agent_neighbours_local[idx][observed_neighbour]
                    observation_t[idx][observed_neighbour+1] = int(actions[observed_agent][-2]) + 1
                    observation_t[idx][-1] = int(actions[idx][-1]) 
        all_observations[t,:] = observation_t

    return all_actions, all_beliefs, all_observations


def inference_loop(G,N): #just goes until you get a graph that has the right connectedness
    try:
        G = nx.fast_gnp_random_graph(N,p)
        G, agents_dict, agents, agent_neighbours = create_multiagents(G, N)


        G = connect_edgeless_nodes(G)
        all_actions, all_beliefs, all_observations = multi_agent_loop(T, agents, agent_neighbours)
    except ValueError:
        all_actions, all_beliefs, all_observations, agents, agent_neighbours = inference_loop(G,N)
    
    return all_actions, all_beliefs, all_observations, agents, agent_neighbours


if __name__ == '__main__':

    N = 8 # total number of agents
    idea_levels = 2 
    num_H = 2

    ecb_precision = 5

    N = 10

    env_precision = 10

    b_precision = 5


    p = 1 # different levels of random connection parameter in Erdos-Renyi random graphs
    num_trials = 5 # number of trials per level of the ER parameter
    T = 300
    #fig, axs = plt.subplots(len(p_vec)/2, len(p_vec)/2)
            
    G = nx.fast_gnp_random_graph(N,p) # create the graph for this trial & condition
    #this performs the multiagent inference
    all_actions, all_beliefs, all_observations, agents, agent_neighbours = inference_loop(G,N)
    np.savez('results/medianodedata', all_actions, all_beliefs, all_observations, agent_neighbours)

