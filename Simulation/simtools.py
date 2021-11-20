import numpy as np
import networkx as nx
import random
import copy
import time

from Model.agent import Agent
from Model.pymdp import utils
from Model.pymdp.utils import softmax

def generate_network(N,p):
    G = nx.fast_gnp_random_graph(N,p) # create the graph for this trial & condition

        # make sure graph is connected and all agents have at least one edge
    if not nx.is_connected(G):
        G = connect_edgeless_nodes(G) # make sure graph is connected
    while np.array(list(G.degree()))[:,1].min() < 2: # make sure no agents with only 1 edge
        #G = nx.stochastic_block_model(sizes, probs, seed=0) # create the graph for this trial & condition
        G = nx.fast_gnp_random_graph(N,p)

        if not nx.is_connected(G):
            G = connect_edgeless_nodes(G) # make sure graph is 
    return G


def generate_quick_agent_observation(reduce_A = True, num_neighbours = 2, reduce_A_policies = True, reduce_A_inference = True ):

    idea_levels = 2 # the levels of beliefs that agents can have about the idea (e.g. 'True' vs. 'False', in case `idea_levels` ==2)
    num_H = 2 #the number of hashtags, or observations that can shed light on the idea
    h_idea_mapping = np.eye(num_H)
    h_idea_mapping[:,0] = utils.softmax(h_idea_mapping[:,0]*1.0)
    h_idea_mapping[:,1] = utils.softmax(h_idea_mapping[:,1]*1.0)
    agent_params = {

        "neighbour_params" : {
            "ecb_precisions" : np.array([[8.0,8.0], [8.0, 8.0]]),
            "num_neighbours" : num_neighbours,
            "env_determinism": 9.0,
            "belief_determinism": np.array([7.0, 7.0])
            },

        "idea_mapping_params" : {
            "num_H" : num_H,
            "idea_levels": idea_levels,
            "h_idea_mapping": h_idea_mapping
            },

        "policy_params" : {
            "initial_action" : [np.random.randint(num_H), 0],
            "belief2tweet_mapping" : np.eye(num_H),
            "E_lr" : 0.7
            },

        "C_params" : {
            "preference_shape" : None,
            "cohesion_exp" : None,
            "cohesion_temp" : None
            }
    }
    observation = np.zeros(num_neighbours + 3)
    observation[2] = 1
    agent = Agent(**agent_params,reduce_A=reduce_A, reduce_A_policies = reduce_A_policies, reduce_A_inferennce=reduce_A_inference)
    
    
    return agent, observation







def initialize_agent_params(G, 
                            num_H = 2, 
                            idea_levels = 2, 
                            h_idea_mappings = None, 
                            belief2tweet_mappings = None, 
                            ecb_precisions = None, 
                            B_idea_precisions = None,
                            B_neighbour_precisions = None, 
                            variance = None,
                            E_noise = None,
                            c_params = None,
                            optim_options = None):
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
        mins = np.random.gamma(4.0,size = (num_agents, ))
        ranges = np.hstack( (mins.reshape(-1,1), mins.reshape(-1,1) + ecb_spread )) # min and maxes of random uniform distributions
        ecb_precisions_all = {}
        for i in G.nodes():
            ecb_precisions_all[i] = ranges[i,:]
    elif ecb_precisions == False:
        pass
    else:
        ecb_precisions_all = {i: np.array(ecb_precisions) for i in G.nodes()}
    
    if B_idea_precisions is None:
        mins = np.random.gamma(5.0,size = (num_agents, ))
        ranges = np.hstack( (mins.reshape(-1,1), mins.reshape(-1,1) + b_idea_spread )) # min and maxes of random uniform distributions
        B_idea_precisions_all = {}
        for i in G.nodes():
            B_idea_precisions_all[i] = ranges[i,:]
    else:
        B_idea_precisions_all = {i: np.array(B_idea_precisions) for i in G.nodes()}

    if B_neighbour_precisions is None:
        mins = np.random.gamma(8.0,size = (num_agents, ))
        ranges = np.hstack( (mins.reshape(-1,1), mins.reshape(-1,1) + b_neighbour_spread )) # min and maxes of random uniform distributions
        B_neighbour_precisions_all = {}
        for i in G.nodes():
            B_neighbour_precisions_all[i] = ranges[i,:]
    else:
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
    
    if optim_options is None:
        optim_options = {'reduce_A': True, 'reduce_A_inference': True, 'reduce_A_policies': True}
    
    agent_constructor_params = {}

    store_parameters = utils.obj_array(len(G.nodes))
    for i in G.nodes():

        num_neighbours = G.degree(i)

        initial_tweet, initial_neighbour_to_sample = np.random.randint(num_H), np.random.randint(num_neighbours) 

        # ecb_precisions = [np.random.uniform(low = ecb_precisions_all[i][0], high = ecb_precisions_all[i][1], size = (idea_levels,) ) for n in range(num_neighbours)]
        if ecb_precisions != False:
            ecb_precisions_i = np.absolute(np.random.normal(ecb_precisions_all[i], variance, size=(num_neighbours, idea_levels)))
        else:
            ecb_precisions_i = False
        #ecb_precisions = np.ones((num_neighbours, idea_levels)) * ecb_precisions_all[i]
        env_determinism = B_idea_precisions_all[i]

        #belief_determinism = np.absolute(np.random.normal(B_neighbour_precisions_all[i], variance, size=(num_neighbours,)) )
        belief_determinism = np.random.normal(B_neighbour_precisions_all[i]) * np.ones((num_neighbours,))

        #h_idea_mapping = utils.softmax(np.array([[1,0],[0,1]])* np.random.uniform(0.3,3))
        h_idea_mapping = np.eye(num_H)
        h_idea_mapping[:,0] = utils.softmax(h_idea_mapping[:,0]*1.0)
        h_idea_mapping[:,1] = utils.softmax(h_idea_mapping[:,1]*1.0)
        params = [ecb_precisions,env_determinism,belief_determinism,h_idea_mapping]
        store_parameters[i] = params

        agent_constructor_params[i] = {

            "neighbour_params" : {
                "ecb_precisions" :  ecb_precisions_i,
                "num_neighbours" : num_neighbours,
                "env_determinism":  env_determinism,
                "belief_determinism": belief_determinism
                },

            "idea_mapping_params" : {
                "num_H" : num_H,
                "idea_levels": idea_levels,
                "h_idea_mapping": h_idea_mapping
                },

            "policy_params" : {
                "initial_action" : [initial_tweet, initial_neighbour_to_sample],
                "belief2tweet_mapping" : belief2tweet_mappings_all[i],
                "E_lr" : E_noise
                },

            "C_params" : c_params_all[i],
            "reduce_A": optim_options['reduce_A'],
            'reduce_A_inference': optim_options['reduce_A_inference'],
            "reduce_A_policies": optim_options['reduce_A_policies']
        }


    return agent_constructor_params, store_parameters


def initialize_network(G, agent_constructor_params, T):
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
      
        single_node_attrs['selected_actions'][agent_i] = np.zeros((T, 2),  dtype=int) # history indices of the actions selected by `agent_i`

        single_node_attrs['my_tweet'][agent_i] = np.zeros(T+1) # history of indices of `my_tweet` (same as G.nodes()[agent_i][`o`][:,0])

        single_node_attrs['other_tweet'][agent_i] = np.zeros(T+1)  # history of indices of `other_tweet` (same as G.nodes()[agent_i][`o`][t,n+1]) where `n` is the index of the selected neighbour at time t

        single_node_attrs['sampled_neighbors'][agent_i] = np.zeros(T+1) 

    for attr, attr_dict in single_node_attrs.items():

        nx.set_node_attributes(G, attr_dict, attr)
    
    return G

def run_simulation(G, T):

    # run first timestep
    priors_over_policies = []

    G = get_observations_time_t(G,0)

    # run active inference loop over time
    inference_time_cost = 0 
    control_time_cost = 0

    for t in range(T):
        #print(str(t) + "/" + str(T))
        G, infer_time_cost_t, control_time_cost_t = run_single_timestep(G, t)
    
    return G, infer_time_cost_t, control_time_cost_t

def run_single_timestep(G, t):

    # Two loops over agents, first to update beliefs given most recent observations and select actions, second loop to get new set of observations
    # First loop over agents: Do belief-updating (inference) and action selection

    inference_time_cost = 0
    control_time_cost = 0

    for i in G.nodes():

        node_attrs = G.nodes()[i]

        agent_i = node_attrs['agent']

        infer_start_time = time.time()
        qs = agent_i.infer_states(t, tuple(node_attrs['o'][t,:]))
        infer_end_time = time.time()

        inference_time_cost += (infer_end_time - infer_start_time)

        node_attrs['qs'][t,:] = copy.deepcopy(qs) 

        policy_start_time = time.time()
        q_pi = agent_i.infer_policies()
        policy_end_time = time.time()

        control_time_cost += (policy_end_time - policy_start_time)

        node_attrs['q_pi'][t,:] = np.copy(q_pi)
        if t == 0:
            action = agent_i.action[-2:]
        else:
            action = agent_i.sample_action()[-2:]
        node_attrs['selected_actions'][t,:] = action
    
    for i in G.nodes(): # get observations for next timestep

        G = get_observations_time_t(G,t+1)

    return G, inference_time_cost, control_time_cost

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





