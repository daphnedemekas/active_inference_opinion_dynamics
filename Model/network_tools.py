import numpy as np
import networkx as nx

from .agent import Agent


def create_multiagents(G, N = N, idea_levels = 2, num_H = 2, precision_params = None, env_volatility = None, belief_volatility = None):
    """
    Populates a networkx graph object G with N active inference agents
    """

    agents_dict = {}

    if precision_params is None:
        precision_params = [[3.0, 10.0] for i in range(N)] # min and max values of uniform distribution over precision parameters
    
    if env_volatility is None:
        env_volatility = [[0.5, 3.0] for i in range(N)]  # min and max values of uniform distribution over environmental volatility parameters
    
    if belief_volatility is None:
        belief_volatility = [[0.5, 3.0] for i in range(N)] # min and max values of uniform distribution over neighbour-belief volatility parameters
    
    for agent_i in G.nodes():

        neighbors_i = list(nx.neighbors(G, agent_i))
        num_neighbours = len(neighbors_i)

        agent_i_params = {

            "neighbour_params" : {
                "precisions" : np.random.uniform(low=precision_params[i][0], high=precision_params[i][1], size=(num_neighbours,)),
                "num_neighbours" : num_neighbours,
                "env_volatility": np.random.uniform(low = env_volatility[i][0], high = env_volatility[i][1])
                "belief_volatility": np.random.uniform(low=belief_volatility[i][0], high=belief_volatility[i][1], size=(num_neighbours,))
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

        agents_dict[agent_i] = agent_i_params
    
    nx.set_node_attributes(G, agents_dict, 'agent')

    return G, agents_dict

