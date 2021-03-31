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
import time
#%%
def make_agent_params(number_of_neighbours):
    idea_levels = 2 # the levels of beliefs that agents can have about the idea (e.g. 'True' vs. 'False', in case `idea_levels` ==2)
    num_H = 2 #the number of hashtags, or observations that can shed light on the idea
    num_neighbours = number_of_neighbours
    h_idea_mapping = np.eye(num_H)
    h_idea_mapping[:,0] = softmax(h_idea_mapping[:,0]*0.1)
    h_idea_mapping[:,1] = softmax(h_idea_mapping[:,1]*0.1)
    agent_params = {

                "neighbour_params" : {
                    # "precisions" : np.random.uniform(low=0.3, high=3.0, size=(2,)),
                    "precisions" : np.array([1.0,10.0]),
                    "num_neighbours" : num_neighbours,
                    "env_volatility": np.random.uniform(low = 5.0, high = 6.0),
                    #"belief_volatility": np.random.uniform(low=9.0, high = 10.0, size=(2,))
                    "belief_volatility": np.random.uniform(low=9.0, high = 10.0, size=(num_neighbours,))

                    },

                "idea_mapping_params" : {
                    "num_H" : num_H,
                    "idea_levels": idea_levels,
                    "h_idea_mapping": h_idea_mapping
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
    return agent_params


times = []
neighbours = [1,2,3,4,5,6,7,8,9,10,11,12]

for n in neighbours:
    agent_params = make_agent_params(n)
    start = time.time()
    agent = Agent(**agent_params)
    end = time.time()
    times.append(end-start)
    print("num neighbours" + str(n))
    print("time taken" + str(end-start))

plt.plot(neighbours, times)
plt.xlabel("Number of Neighbours")
plt.ylabel("Time taken")
plt.show()