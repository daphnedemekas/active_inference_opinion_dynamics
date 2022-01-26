#%%
import numpy as np
from model.agent import Agent
import networkx as nx
from model.pymdp import utils
from model.pymdp.utils import obj_array, index_list_to_onehots, sample, reduce_a_matrix
from model.pymdp.maths import softmax, spm_dot, spm_log, get_joint_likelihood, calc_free_energy
from model.pymdp.inference import update_posterior_states
from simulation.simtools import initialize_agent_params, generate_network, initialize_network, run_simulation
from analysis.analysis_tools import collect_idea_beliefs, collect_sampling_history, collect_tweets

import seaborn as sns
from matplotlib import pyplot as plt



""" Set up the generative model """

idea_levels = 3 # the levels of beliefs that agents can have about the idea (e.g. 'True' vs. 'False', in case `idea_levels` ==2)
num_H = 3 #the number of hashtags, or observations that can shed light on the idea
num_neighbors = 2
h_idea_mapping = np.eye(num_H)
h_idea_mapping[:,0] = softmax(h_idea_mapping[:,0]*1.0)
h_idea_mapping[:,1] = softmax(h_idea_mapping[:,1]*1.0)
h_idea_mapping[:,2] = softmax(h_idea_mapping[:,2]*1.0)


""" Set parameters and generate agents"""
env_d = 8
c = 0
ecb = 4
belief_d = 7
T = 10 #the number of timesteps 
N = 3
p = 1

G = generate_network(N,p)

agent_constructor_params = initialize_agent_params(G, num_H = num_H, idea_levels = idea_levels, h_idea_mapping = h_idea_mapping, \
                                    ecb_precisions = ecb, B_idea_precisions = env_d, \
                                        B_neighbour_precisions = belief_d)

G = initialize_network(G, agent_constructor_params, T = T, model = "sequencing")


G = run_simulation(G, T = T, model = "sequencing")


