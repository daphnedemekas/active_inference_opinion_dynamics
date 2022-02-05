import numpy as np
import networkx as nx
import random
import copy
import time

from model.agent import Agent
from model.pymdp import utils
from model.pymdp.utils import softmax

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

def initialize_agent_params(G, 
                            num_H = 2, 
                            idea_levels = 2, 
                            h_idea_mapping = None, 
                            belief2tweet_mappings = None, 
                            ecb_precisions = None, 
                            B_idea_precisions = None,
                            B_neighbour_precisions = None, 
                            E_noise = None,
                            ecb_spread = 0.1,
                            volatility_spread = 0.1,
                            optim_options = None,
                            model = None,
                            model_parameters = {}):
    """
    Initialize dictionaries of agent-specific generative model parameters
    """

    """
    1. Set defaults if not specified (e.g. draw `low` 
    parameter of uniform distribution from Gamma distributions, then add +1 to make maximum)
    """

    # set every single agent's belief2tweet_mapping
    if belief2tweet_mappings is None:
        belief2tweet_mappings_all = {}
        for i in G.nodes():
            belief2tweet_mappings_all[i] = np.eye(num_H)
    elif isinstance(belief2tweet_mappings,np.ndarray):
        belief2tweet_mappings_all = {i: belief2tweet_mappings for i in G.nodes()}
    
    # set every single agent's ecb precision parameters, if not provided (min and max parameters of a uniform distribution)
    ecb_precisions_all = {i: np.array(ecb_precisions) for i in G.nodes()}
    
    B_idea_precisions_all = {i: B_idea_precisions for i in G.nodes()}

    B_neighbour_precisions_all = {i: np.array(B_neighbour_precisions) for i in G.nodes()}

    if optim_options is None:
        optim_options = {'reduce_A': True, 'reduce_A_inference': True, 'reduce_A_policies': True}
    
    agent_constructor_params = {}

    for i in G.nodes():

        num_neighbours = G.degree(i)

        initial_tweet, initial_neighbour_to_sample = np.random.randint(num_H), np.random.randint(num_neighbours) 

        ecb_precisions_i = np.absolute(np.random.normal(ecb_precisions_all[i], ecb_spread, size=(num_neighbours, idea_levels)))

        env_determinism = B_idea_precisions_all[i]

        belief_determinism = np.absolute(np.random.normal(B_neighbour_precisions_all[i], volatility_spread, size=(num_neighbours,)) )

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
            
            "model_params": model_parameters

        }

    return agent_constructor_params


def initialize_network(G, agent_constructor_params, T, model):
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


        agent = Agent(**agent_constructor_params[agent_i], model = model)
        self_global_label_mapping = dict(zip(range(G.degree(agent_i)), list(nx.neighbors(G, agent_i))))

        single_node_attrs['agent'][agent_i] = agent

        single_node_attrs['self_global_label_mapping'][agent_i] = self_global_label_mapping
      
        single_node_attrs['qs'][agent_i] = np.empty((T, agent.genmodel.num_factors), dtype=object) # history of the posterior beliefs  about hidden states of `agent_i` 

        single_node_attrs['q_pi'][agent_i] = np.empty((T, len(agent.genmodel.policies)), dtype=object) # history of the posterior beliefs about policies of `agent_i` 

        single_node_attrs['o'][agent_i] = np.zeros((T+1, agent.genmodel.num_modalities), dtype=int) # history of the indices of the observations made by `agent_i`. One extra time index for the last timestep, which has no subsequent active inference loop 

        #todo: generalize the 2 to be the number of control states
        single_node_attrs['selected_actions'][agent_i] = np.zeros((T, 2),  dtype=int) # history indices of the actions selected by `agent_i`

        single_node_attrs['my_tweet'][agent_i] = np.zeros(T+1) # history of indices of `my_tweet` (same as G.nodes()[agent_i][`o`][:,0])

        single_node_attrs['other_tweet'][agent_i] = np.zeros(T+1)  # history of indices of `other_tweet` (same as G.nodes()[agent_i][`o`][t,n+1]) where `n` is the index of the selected neighbour at time t

        single_node_attrs['sampled_neighbors'][agent_i] = np.zeros(T+1) 

    for attr, attr_dict in single_node_attrs.items():

        nx.set_node_attributes(G, attr_dict, attr)
    
    return G


def get_observations_time_t(G, t, model):
    
    if model == "self_esteem" and t > 0:
        belief_mu =  np.mean([G.nodes[i]['qs'][t-1,0] for i in range(len(G.nodes()))],axis=0)
        belief_std =  np.std([G.nodes[i]['qs'][t-1,0] for i in range(len(G.nodes()))],axis=0)
       # print("average beliefs: " + str(belief_mu))
    
    else: 
        belief_mu = belief_std = None

    for i in G.nodes():

        node_attrs = G.nodes()[i] # get attributes for the i-th node
        agent_i = node_attrs['agent']      # get agent class for the i-th node

        node_attrs['o'][t,agent_i.genmodel.focal_h_idx] = int(agent_i.action[agent_i.genmodel.h_control_idx])      # my first observation is what I'm tweeting
        
        #redundant
        node_attrs['my_tweet'][t] = int(agent_i.action[agent_i.genmodel.h_control_idx]) # what I'm tweeting

        node_attrs['o'][t,agent_i.genmodel.who_obs_idx] = int(agent_i.action[agent_i.genmodel.who_idx]) # my last observation is who I'm sampling

        which_neighbour = int(agent_i.action[agent_i.genmodel.who_idx]) # who I'm sampling

        global_neighbour_idx = node_attrs['self_global_label_mapping'][which_neighbour] # convert from 'local' (focal-agent-relative) neighbour index to global (network-relative) index

        node_attrs['sampled_neighbors'][t] = global_neighbour_idx  # index of the neighbour I'm sampling, in terms of that neighbour's global index

        sampled_node_attrs = G.nodes()[global_neighbour_idx]

        node_attrs['other_tweet'][t] = int(sampled_node_attrs['agent'].action[sampled_node_attrs['agent'].genmodel.h_control_idx]+1) # what they're tweeting, offset by +1 to account for the 0-th observation in my-reading-them modality (the null observation)
        
        node_attrs['o'][t,which_neighbour+1] = node_attrs['other_tweet'][t] # my observation in the (neighbour+1)-th modality is my neighbour's tweet (their 0-th control factor action) 


        if agent_i.model == "self_esteem":
            if t <= 1:
                node_attrs['o'][t,agent_i.genmodel.focal_esteem_idx] = 1 #calculated using a threshold on the number of standard deviations between the focal agent's belief and the average belief
                for idx in agent_i.genmodel.neighbour_esteem_idx:
                    node_attrs['o'][t,idx] = 1
            else:

                focal_esteem = calculate_esteem(G.nodes()[0]['qs'][t-1,0], belief_mu, belief_std)
                node_attrs['o'][t,agent_i.genmodel.focal_esteem_idx] = focal_esteem #calculated using a threshold on the number of standard deviations between the focal agent's belief and the average belief
                for i, idx in enumerate(agent_i.genmodel.neighbour_esteem_idx):
                    global_neighbour_idx = node_attrs['self_global_label_mapping'][i] # convert from 'local' (focal-agent-relative) neighbour index to global (network-relative) index
                    
                    neighbour_esteem = calculate_esteem(G.nodes()[global_neighbour_idx]['qs'][t-1,0], belief_mu, belief_std) #calculated using a threshold on the number of standard deviations between the neighbours' belief and the average belief
                    node_attrs['o'][t,idx] = neighbour_esteem

                    #here i would need to get the expected state for each agent 

                # node_attrs['o'][t,idx] = calculate_esteem(agent_i.qs[i+1], belief_mu, belief_std) #calculated using a threshold on the number of standard deviations between the focal agent's belief about the neighbours' belief and the average belief
                    #TODO: should we be using the focal agent's belief about the neighbours' belief or the neighbours actual belief to generate the focal agent's observation of the neighbours' esteem?
    return G

def calculate_esteem(qs, belief_mu, belief_std):
    """ This function measures the number of standard deviations away the qs is from the average belief"""
    difference = np.absolute(qs - belief_mu)
    num_stds = np.mean(difference / belief_std)
    if num_stds < 0.8: 
        return 0 
    if num_stds > 1.3: 
        return 2
    return 1

def calculate_esteem_as_KL(expected_state, real_state):
    """ sum of action of "likes" which depends on KL divergence between focal belief and belief about neighbours' belief""


def calculate_esteem_as_KL(expected_state, real_state):
This function should generate the esteem observation for all agents at any time step

Should be a KL divergence between the expected state and the real state

Input: a vector of the expected states for each agent as well as the real state at that time step
Output: The KL divergence (scalar) for each agent between those two vectors 
Need to convert this into three distinct thresholds for generating the esteem observation """ 
    
    pass 



def run_simulation(G, T, model = None):

    G = get_observations_time_t(G,0, model)

    # run active inference loop over time
    for t in range(T):
        print("Time: " + str(t))
        G = run_single_timestep(G, t, model)
    
    return G

def run_single_timestep(G, t, model = None):

    # Two loops over agents, first to update beliefs given most recent observations and select actions, second loop to get new set of observations
    # First loop over agents: Do belief-updating (inference) and action selection
    for i in G.nodes():

        node_attrs = G.nodes()[i]

        agent_i = node_attrs['agent']
        #print("observation: " + str(tuple(node_attrs['o'][t,:])))
        qs = agent_i.infer_states(t, tuple(node_attrs['o'][t,:]))

        node_attrs['qs'][t,:] = copy.deepcopy(qs) 

        q_pi = agent_i.infer_policies()
 
        node_attrs['q_pi'][t,:] = np.copy(q_pi)
        if t == 0:
            action = agent_i.action[[agent_i.genmodel.h_control_idx, agent_i.genmodel.who_idx] ]
        else:

            action = agent_i.sample_action()
           # print("Action: " + str(action))
            action = action[tuple([[agent_i.genmodel.h_control_idx, agent_i.genmodel.who_idx]])]

        node_attrs['selected_actions'][t,:] = action
    
    for i in G.nodes(): # get observations for next timestep

        G = get_observations_time_t(G,t+1, model)

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
