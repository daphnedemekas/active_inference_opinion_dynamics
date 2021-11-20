import numpy as np
import networkx as nx

def collect_idea_beliefs(G, start_t = 0, end_t = None):
    """
    Collect the history of agents' beliefs about the idea (the first factor of their posterior beliefs)
    over time and return as a (T, num_idea_levels, N) matrix of posterior distributions, where num_idea_levels
    is the dimensionality/support of the posterior marginal of the first hidden state factor
    """

    if end_t is None:
        end_t = G.nodes()[0]['qs'].shape[0]

    N = G.number_of_nodes()

    idea_levels = G.nodes()[0]['agent'].genmodel.idea_levels

    belief_matrix = np.zeros((end_t-start_t, idea_levels, N))

    for i in G.nodes():
        belief_matrix[:,:,i] = np.stack(G.nodes()[i]['qs'][start_t:end_t,0])

    return belief_matrix

def collect_tweets(G, start_t = 0, end_t = None):

    if end_t is None:
        end_t = len(G.nodes()[0]['my_tweet'])

    N = G.number_of_nodes()

    tweet_matrix = np.zeros((end_t-start_t, N))

    for i in G.nodes():
        tweet_matrix[:,i] = G.nodes()[i]['my_tweet'][start_t:end_t]

    return tweet_matrix

def collect_reading_history(G, start_t = 0, end_t = None):

    if end_t is None:
        end_t = len(G.nodes()[0]['other_tweet'])

    N = G.number_of_nodes()

    other_tweet_matrix = np.zeros((end_t-start_t, N))

    for i in G.nodes():
        other_tweet_matrix[:,i] = G.nodes()[i]['other_tweet'][start_t:end_t]

    return other_tweet_matrix

def collect_sampling_history(G, start_t = 0, end_t = None):

    if end_t is None:
        end_t = len(G.nodes()[0]['sampled_neighbors'])

    N = G.number_of_nodes()

    sampling_matrix = np.zeros((end_t-start_t, N))

    for i in G.nodes():
        sampling_matrix[:,i] = G.nodes()[i]['sampled_neighbors'][start_t:end_t]

    return sampling_matrix



