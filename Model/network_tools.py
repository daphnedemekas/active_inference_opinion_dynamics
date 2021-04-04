import numpy as np
import networkx as nx
import random

from .agent import Agent


def create_multiagents(G, N , idea_levels = 2, num_H = 2, precision_params = None, env_determinism = None, belief_determinism = None):
    """
    Populates a networkx graph object G with N active inference agents
    """

    agents_dict = {}

    if precision_params is None:
        precision_params = [[0.3, 3.0] for i in range(N)] # min and max values of uniform distribution over precision parameters
    
    if env_determinism is None:
        env_determinism = [[5, 10] for i in range(N)]  # min and max values of uniform distribution over environmental determinism parameters
    
    if belief_determinism is None:
        belief_determinism = [[5, 10] for i in range(N)] # min and max values of uniform distribution over neighbour-belief determinism parameters
    
    for i in G.nodes():

        neighbors_i = list(nx.neighbors(G, i))
        num_neighbours = len(neighbors_i)

        agent_i_params = {

            "neighbour_params" : {
                "precisions" : np.random.uniform(low=precision_params[i][0], high=precision_params[i][1], size=(num_neighbours,)),
                "num_neighbours" : num_neighbours,
                "env_determinism": np.random.uniform(low = env_determinism[i][0], high = env_determinism[i][1]),
                "belief_determinism": np.random.uniform(low=belief_determinism[i][0], high=belief_determinism[i][1], size=(num_neighbours,))
                },

            "idea_mapping_params" : {
                "num_H" : num_H,
                "idea_levels": idea_levels,
                "h_idea_mapping": None
                },

            "policy_params" : {
                "initial_action" : [np.random.randint(num_H), np.random.randint(num_neighbours)],
                "belief2tweet_mapping" : None
                },

            "C_params" : {
                "preference_shape" : None,
                "cohesion_exp" : None,
                "cohesion_temp" : None
                }
        }

        agents_dict[i] = agent_i_params
    
    nx.set_node_attributes(G, agents_dict, 'agent')

    return G, agents_dict

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
