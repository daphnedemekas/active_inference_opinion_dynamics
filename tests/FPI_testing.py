#%%
import numpy as np
from model.genmodel import GenerativeModel
from model.agent import Agent
from model.pymdp.maths import softmax
from model.pymdp.inference import update_posterior_states
from model.pymdp.fpi import run_fpi, run_fpi_factorized
from model.pymdp.utils import process_observation

from matplotlib import pyplot as plt

# %% Set up generative model
idea_levels = 2 # the levels of beliefs that agents can have about the idea (e.g. 'True' vs. 'False', in case `idea_levels` ==2)
num_H = 2 #the number of hashtags, or observations that can shed light on the idea
num_neighbours = 2
h_idea_mapping = np.eye(num_H)
h_idea_mapping[:,0] = softmax(h_idea_mapping[:,0]*1.0)
h_idea_mapping[:,1] = softmax(h_idea_mapping[:,1]*1.0)
h_idea_mapping = softmax(np.array([[1,0],[0,1]])* np.random.uniform(0.3,3))
agent_params = {

            "neighbour_params" : {
                "ecb_precisions" : np.array([[8.0,8.0], [8.0, 5.0]]),
                "num_neighbours" : 2,
                "env_determinism": 10.0,
                "belief_determinism": np.array([2.0, 9.0])
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

agent = Agent(**agent_params,reduce_A=True)

#%%

A_reduced = agent.genmodel.A_reduced
informative_dims = agent.genmodel.informative_dims
n_states = agent.genmodel.num_states
n_obs = agent.genmodel.num_obs

obs = process_observation((0, 1, 0, 0), agent.genmodel.num_modalities, n_obs)

qs = run_fpi_factorized(obs, A_reduced, informative_dims, n_states)

qs = run_fpi(agent.genmodel.A, obs, n_obs, n_states)

