import numpy as np
import networkx as nx
import random
import copy

from Model.agent import Agent
from Model.pymdp import utils

def initialize_agent_params(G, 
                            num_H = 2, 
                            idea_levels = 2, 
                            h_idea_mappings = None, 
                            belief2tweet_mappings = None, 
                            ecb_precisions = None, 
                            B_idea_precisions = None,
                            B_neighbour_precisions = None, 
                            c_params = None,
                            reduce_A = False):
    """
    Initialize dictionaries of agent-specific generative model parameters
    """

    """
    1. Set defaults if not specified (e.g. draw `low` 
    parameter of uniform distribution from Gamma distributions, then add +1 to make maximum)
    """

    num_agents = G.number_of_nodes()

    ecb_spread = 1.0# these could be fed in as hyperparams in theory
    b_idea_spread = 1.0 # these could be fed in as hyperparams in theory
    b_neighbour_spread = 1.0 # these could be fed in as hyperparams in theory

    # set every single agent's h_idea_mapping
    if h_idea_mappings is None:
        h_idea_mappings_all = {}
        for i in G.nodes():
            h_idea_mappings_all[i] = utils.softmax(np.eye(num_H) * 1.0)
    elif isinstance(h_idea_mappings,np.ndarray):
        h_idea_mappings_all = {i: h_idea_mappings for i in G.nodes()}

    # set every single agent's belief2tweet_mapping
    if belief2tweet_mappings is None:
        belief2tweet_mappings_all = {}
        for i in G.nodes():
            belief2tweet_mappings_all[i] = np.eye(num_H)
    elif isinstance(belief2tweet_mappings,np.ndarray):
        belief2tweet_mappings_all = {i: belief2tweet_mappings for i in G.nodes()}
    
    # set every single agent's ecb precision parameters, if not provided (min and max parameters of a uniform distribution)
    if ecb_precisions is None:
        mins = np.random.gamma(2.0, 2.0, size = (num_agents, ))
        ranges = np.hstack( (mins.reshape(-1,1), mins.reshape(-1,1) + ecb_spread )) # min and maxes of random uniform distributions
        ecb_precisions_all = {}
        for i in G.nodes():
            ecb_precisions_all[i] = ranges[i,:]
    elif isinstance(ecb_precisions,list):
        ecb_precisions_all = {i: np.array(ecb_precisions) for i in G.nodes()}
    
    if B_idea_precisions is None:
        mins = np.random.gamma(4.0,2, size = (num_agents, ))
        ranges = np.hstack( (mins.reshape(-1,1), mins.reshape(-1,1) + b_idea_spread )) # min and maxes of random uniform distributions
        B_idea_precisions_all = {}
        for i in G.nodes():
            B_idea_precisions_all[i] = ranges[i,:]
    elif isinstance(B_idea_precisions,list):
        B_idea_precisions_all = {i: np.array(B_idea_precisions) for i in G.nodes()}

    if B_neighbour_precisions is None:
        mins = np.random.gamma(4.0,2, size = (num_agents, ))
        ranges = np.hstack( (mins.reshape(-1,1), mins.reshape(-1,1) + b_neighbour_spread )) # min and maxes of random uniform distributions
        B_neighbour_precisions_all = {}
        for i in G.nodes():
            B_neighbour_precisions_all[i] = ranges[i,:]
    elif isinstance(B_neighbour_precisions,list):
        B_neighbour_precisions_all = {i: np.array(B_neighbour_precisions) for i in G.nodes()}
    
    # set every single agent's prior preference parameters
    # @ TODO: Make the cohesion_exp and cohension_temps, if not provided, drawn from some distribution (e.g. a Gamma distribution or something)
    if c_params is None:
        c_params_all = {}
        for i in G.nodes():
            c_params_all[i] = {
                "preference_shape" : None,
                "cohesion_exp" : None,
                "cohesion_temp" : None
                }
    elif all (k in c_params for k in ("preference_shape","cohesion_exp", "cohesion_temp")):
        c_params_all = { i : c_params for i in G.nodes() }
    
    agent_constructor_params = {}

    for i in G.nodes():

        num_neighbours = G.degree(i)

        initial_tweet, initial_neighbour_to_sample = np.random.randint(num_H), np.random.randint(num_neighbours) 

        agent_constructor_params[i] = {

            "neighbour_params" : {
                "ecb_precisions" :  [np.random.uniform(low = ecb_precisions_all[i][0], high = ecb_precisions_all[i][1], size = (idea_levels,) ) for n in range(num_neighbours)],
                "num_neighbours" : num_neighbours,
                "env_determinism":  np.random.uniform(low = B_idea_precisions_all[i][0], high = B_idea_precisions_all[i][1]),
                "belief_determinism": np.random.uniform(low = B_neighbour_precisions_all[i][0], high = B_neighbour_precisions_all[i][1], size = (num_neighbours,))
                },

            "idea_mapping_params" : {
                "num_H" : num_H,
                "idea_levels": idea_levels,
                "h_idea_mapping": h_idea_mappings_all[i]
                },

            "policy_params" : {
                "initial_action" : [initial_tweet, initial_neighbour_to_sample],
                "belief2tweet_mapping" : belief2tweet_mappings_all[i]
                },

            "C_params" : c_params_all[i],
            "reduce_A": reduce_A
        }


    return agent_constructor_params


def initialize_network(G, agent_constructor_params, T = 10):
    """
    Initializes a network object G that stores agent-level information (e.g. parameters of individual
    generative models, global node-indices, ...) and information about the generative process.
    """
            
    single_node_attrs = {
            'agent': {},
            'self_global_label_mapping': {},
            'qs': {},
            'q_pi': {},
            'o': {},
            'selected_actions': {},
            'my_tweet': {},
            'other_tweet': {},
            'sampled_neighbors': {}
            }

    single_node_attrs['stored_data'] = {i:list(single_node_attrs.keys()) for i in G.nodes()}
    
    for agent_i in G.nodes():

        agent = Agent(**agent_constructor_params[agent_i])
        self_global_label_mapping = dict(zip(range(G.degree(agent_i)), list(nx.neighbors(G, agent_i))))

        single_node_attrs['agent'][agent_i] = agent

        single_node_attrs['self_global_label_mapping'][agent_i] = self_global_label_mapping
      
        single_node_attrs['qs'][agent_i] = np.empty((T, agent.genmodel.num_factors), dtype=object) # history of the posterior beliefs  about hidden states of `agent_i` 

        single_node_attrs['q_pi'][agent_i] = np.empty((T, len(agent.genmodel.policies)), dtype=object) # history of the posterior beliefs about policies of `agent_i` 

        single_node_attrs['o'][agent_i] = np.zeros((T+1, agent.genmodel.num_modalities), dtype=int) # history of the indices of the observations made by `agent_i`. One extra time index for the last timestep, which has no subsequent active inference loop 
      
        single_node_attrs['selected_actions'][agent_i] = np.zeros((T, agent.genmodel.num_factors),  dtype=int) # history indices of the actions selected by `agent_i`

        single_node_attrs['my_tweet'][agent_i] = np.zeros(T+1) # history of indices of `my_tweet` (same as G.nodes()[agent_i][`o`][:,0])

        single_node_attrs['other_tweet'][agent_i] = np.zeros(T+1)  # history of indices of `other_tweet` (same as G.nodes()[agent_i][`o`][t,n+1]) where `n` is the index of the selected neighbour at time t

        single_node_attrs['sampled_neighbors'][agent_i] = np.zeros(T+1) 

    for attr, attr_dict in single_node_attrs.items():

        nx.set_node_attributes(G, attr_dict, attr)
    
    return G

def run_simulation(G, T):

    # run first timestep

    G = get_observations_time_t(G,0)

    # run active inference loop over time

    for t in range(T):

        G = run_single_timestep(G, t)
    
    return G

def run_single_timestep(G, t):

    # Two loops over agents, first to update beliefs given most recent observations and select actions, second loop to get new set of observations
    # First loop over agents: Do belief-updating (inference) and action selection

    for i in G.nodes():

        node_attrs = G.nodes()[i]

        agent_i = node_attrs['agent']

        qs = agent_i.infer_states(t==0, tuple(node_attrs['o'][t,:]))
        node_attrs['qs'][t,:] = copy.deepcopy(qs) 
        q_pi = agent_i.infer_policies(qs)
        node_attrs['q_pi'][t,:] = np.copy(q_pi)
        action = agent_i.sample_action()
        node_attrs['selected_actions'][t,:] = action
    
    for i in G.nodes(): # get observations for next timestep

        G = get_observations_time_t(G,t+1)

    return G

def get_observations_time_t(G, t):

    for i in G.nodes():

        node_attrs = G.nodes()[i] # get attributes for the i-th node
        agent_i = node_attrs['agent']      # get agent class for the i-th node
        
        node_attrs['o'][t,0] = int(agent_i.action[-2])      # my first observation is what I'm tweeting
        node_attrs['my_tweet'][t] = int(agent_i.action[-2]) # what I'm tweeting

        node_attrs['o'][t,-1] = int(agent_i.action[-1]) # my last observation is who I'm sampling
        which_neighbour = int(agent_i.action[-1]) # who I'm sampling

        global_neighbour_idx = node_attrs['self_global_label_mapping'][which_neighbour] # convert from 'local' (focal-agent-relative) neighbour index to global (network-relative) index
        
        node_attrs['sampled_neighbors'][t] = global_neighbour_idx  # index of the neighbour I'm sampling, in terms of that neighbour's global index

        sampled_node_attrs = G.nodes()[global_neighbour_idx]

        node_attrs['other_tweet'][t] = int(sampled_node_attrs['agent'].action[-2]+1) # what they're tweeting, offset by +1 to account for the 0-th observation in my-reading-them modality (the null observation)

        node_attrs['o'][t,which_neighbour+1] = node_attrs['other_tweet'][t] # my observation in the (neighbour+1)-th modality is my neighbour's tweet (their 0-th control factor action) 
    
    return G

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

def clip_edges(G, max_degree = 10):
    """
    This function iteratively removes edges from nodes that have more than max_degree edges
    """
    single_edge_node = False 

    for node_i in G.nodes():
        connected = [target for (source, target) in G.edges(node_i)]
        while G.degree(node_i) > max_degree:
            
            # list of the numbers of edges of each node connected to our considered node
            deg_of_neighbors = np.zeros(len(connected))

            for idx,neighbor_i in enumerate(connected):
                # this gets the neighbors of the neighbor of node_i we're currently considering (namely, neighbor_i)
                deg_of_neighbors[idx] = G.degree(neighbor_i)
            
            if np.any(deg_of_neighbors > 2):
                idx = np.where(deg_of_neighbors > 2)[0]
                remove = connected[random.choice(idx)]
            else:
                # you'll have to remove a random edge anyway and run 'connect_edgeless_nodes' afterwards
                remove = random.choice(connected)
                single_edge_node = True
            G.remove_edge(node_i,remove)
            connected.remove(remove)
  
            print('\tEdge removed:\t %d -- %d'%(node_i, remove))

    return G, single_edge_node