import numpy as np
import networkx as nx

def collect_idea_beliefs(G, start_t = 0, end_t = None):

    if end_t is None:
        end_t = G.nodes()[0]['qs'].shape[0]

    N = G.number_of_nodes()

    idea_levels = G.nodes()[0]['agent'].genmodel.idea_levels

    belief_matrix = np.zeros((end_t-start_t, idea_levels, N))

    for i in G.nodes():
        belief_matrix[:,:,i] = np.stack(G.nodes(data=True)[i]['qs'][start_t:end_t,0])

    return belief_matrix
