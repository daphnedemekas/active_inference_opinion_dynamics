#%%
import numpy as np
from Model.pymdp.maths import spm_MDP_G, spm_MDP_G_old
import seaborn as sns
from matplotlib import pyplot as plt
import time
from Model.pymdp.utils import obj_array, reduce_A_matrix, to_numpy, obj_array_uniform
from Model.pymdp.maths import spm_dot, dot_likelihood, softmax
from Model.pymdp.inference import average_states_over_policies
from Model.genmodel import GenerativeModel
from Model.agent import Agent
from Model.pymdp.control import get_expected_states, get_expected_obs

#%%
idea_levels = 2 # the levels of beliefs that agents can have about the idea (e.g. 'True' vs. 'False', in case `idea_levels` ==2)
num_H = 2 #the number of hashtags, or observations that can shed light on the idea
h_idea_mapping = np.eye(num_H)
h_idea_mapping[:,0] = softmax(h_idea_mapping[:,0]*0.1)
h_idea_mapping[:,1] = softmax(h_idea_mapping[:,1]*0.1)

def get_params(n):
    agent_params = {

                "neighbour_params" : {
                    # "precisions" : np.random.uniform(low=0.3, high=3.0, size=(2,)),
                    "precisions" : np.array([1.0,10.0]),
                    "num_neighbours" : n,
                    "env_volatility": np.random.uniform(low = 5.0, high = 6.0),
                    #"belief_volatility": np.random.uniform(low=9.0, high = 10.0, size=(2,))
                    "belief_volatility": np.random.uniform(low=9.0, high = 10.0, size=(n,))

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
    return agent_params

times = []
times_old = []
neighbours = [2,5,10,11,12,13,14,15,20,30]

for n in neighbours:
    print("neighbours "+ str(n))
    params = get_params(n)
    genmodel = GenerativeModel(**params['neighbour_params'], **params['idea_mapping_params'], **params['policy_params'], **params['C_params'])
    #A = genmodel.A[1]
    reduced_A = genmodel.reduced_A[1]
    fake_qs = obj_array_uniform(reduced_A.shape[1:])
    start = time.time()
    G = spm_MDP_G(reduced_A, fake_qs)
    end = time.time()
    print(end-start)
    times.append(end-start)
    
    start = time.time()
    G = spm_MDP_G_old(reduced_A, fake_qs)
    end = time.time()
    print(end-start)
    times_old.append(end-start)
    
plt.plot(neighbours, times, label = "new spm_MDP_G")
plt.plot(neighbours, times_old, label = "old_spm_MDP_G")
plt.title("Comparing SPM_MDP_G with reduced A modality 1")
plt.xlabel("Number of Neighbours")
plt.ylabel("Time taken")
plt.savefig("time_reduced_A_modality_1")
plt.show()