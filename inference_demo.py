#%%
import numpy as np
from Model.genmodel import GenerativeModel
from Model.agent import Agent
import networkx as nx
from Model.pymdp.utils import obj_array, index_list_to_onehots, sample
from Model.pymdp.maths import softmax, spm_log
from Model.pymdp.inference import update_posterior_states

import seaborn as sns
from matplotlib import pyplot as plt

#%%
idea_levels = 2 # the levels of beliefs that agents can have about the idea (e.g. 'True' vs. 'False', in case `idea_levels` ==2)
num_H = 2 #the number of hashtags, or observations that can shed light on the idea

h_idea_mapping = np.eye(num_H)
h_idea_mapping[:,0] = softmax(h_idea_mapping[:,0]*1.0)
h_idea_mapping[:,1] = softmax(h_idea_mapping[:,1]*1.0)

agent_params = {

            "neighbour_params" : {
                # "precisions" : np.random.uniform(low=0.3, high=3.0, size=(2,)),
                "precisions" : np.array([5.0,5.0]),
                "num_neighbours" : 2,
                "env_volatility": np.random.uniform(low = 5.0, high = 10.0),
                "belief_volatility": np.random.uniform(low=5.0, high = 10.0, size=(2,))
                },

            "idea_mapping_params" : {
                "num_H" : num_H,
                "idea_levels": idea_levels,
                "h_idea_mapping": h_idea_mapping
                },

            "policy_params" : {
                "initial_action" : [np.random.randint(num_H), np.random.randint(2)],
                "belief2tweet_mapping" : None
                },

            "C_params" : {
                "preference_shape" : None,
                "cohesion_exp" : None,
                "cohesion_temp" : None
                }
        }

agent = Agent(**agent_params)

observation = (0, 1, 0, 0)


T = 100

history_of_idea_beliefs = np.zeros((T,idea_levels))
history_of_beliefs_about_other = np.zeros((T,2))

qs = agent.infer_states(True, observation)

history_of_idea_beliefs[0,:] = qs[0]
history_of_beliefs_about_other[0,:] = qs[1]

constant_action = [0, 0, 0, 0, 0] # always looking at neighbour 0 and tweeting hasthag 0

for t in range(1,T):

    if t % 50 == 0:
        observation = (0, 2, 0, 0)

    empirical_prior = obj_array(agent.genmodel.num_factors)
    for f in range(agent.genmodel.num_factors):   
        empirical_prior[f] = spm_log(agent.genmodel.B[f][:,:, constant_action[f]].dot(qs[f]))

    qs = update_posterior_states(observation, agent.genmodel.A, prior=empirical_prior, **agent.inference_params)

    history_of_idea_beliefs[t,:] = qs[0]
    history_of_beliefs_about_other[t,:] = qs[1]

plt.figure(figsize=(12,8))
plt.plot(history_of_idea_beliefs[:,0],label='My beliefs about the idea')
plt.plot(history_of_beliefs_about_other[:,0],label='My beliefs about your beliefs about the idea')
plt.legend(fontsize=18)
plt.savefig('self_vs_other_beliefs_with_change.png')
# %%
