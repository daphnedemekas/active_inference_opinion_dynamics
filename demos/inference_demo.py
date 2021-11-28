#%%
# import numpy as np
# from Model.genmodel import GenerativeModel
# from Model.agent import Agent
# import networkx as nx
# from Model.pymdp.utils import obj_array, index_list_to_onehots, sample
# from Model.pymdp.maths import softmax, spm_log
# from Model.pymdp.inference import update_posterior_states

# import seaborn as sns
# from matplotlib import pyplot as plt

# #%%
# idea_levels = 2 # the levels of beliefs that agents can have about the idea (e.g. 'True' vs. 'False', in case `idea_levels` ==2)
# num_H = 2 #the number of hashtags, or observations that can shed light on the idea

# h_idea_mapping = np.eye(num_H)
# h_idea_mapping[:,0] = softmax(h_idea_mapping[:,0]*1.0)
# h_idea_mapping[:,1] = softmax(h_idea_mapping[:,1]*1.0)

# def agent_p(belief_d, env_d):
#     agent_params = {

#                 "neighbour_params" : {
#                     "ecb_precisions" : np.array([[8.0,8.0], [8.0, 8.0]]),
#                     "num_neighbours" : 2,
#                     "env_determinism": env_d,
#                     "belief_determinism": np.array([belief_d, belief_d])
#                     },

#                 "idea_mapping_params" : {
#                     "num_H" : num_H,
#                     "idea_levels": idea_levels,
#                     "h_idea_mapping": h_idea_mapping
#                     },

#                 "policy_params" : {
#                     "initial_action" : [np.random.randint(num_H), 0],
#                     "belief2tweet_mapping" : np.eye(num_H),
#                     "E_lr" : 0.7
#                     },

#                 "C_params" : {
#                     "preference_shape" : None,
#                     "cohesion_exp" : None,
#                     "cohesion_temp" : None
#                     }
#             }
#     return agent_params
# plt.figure(figsize=(12,8))

# for env_d in [3,9]:
#     for belief_d in [3,9]:
#         agent_params = agent_p(belief_d, env_d)

#         agent = Agent(**agent_params)

#         observation = (0, 1, 0, 0)


#         T = 100

#         history_of_idea_beliefs = np.zeros((T,idea_levels))
#         history_of_beliefs_about_other = np.zeros((T,2))
#         history_of_idea_beliefs[0,:] = 0.5
#         history_of_beliefs_about_other[0,:] = 0.5
#         qs = agent.infer_states(False, observation)
#         #print(qs)
#         history_of_idea_beliefs[1,:] = qs[0]
#         history_of_beliefs_about_other[1,:] = qs[1]

#         constant_action = [0, 0, 0, 0, 0] # always looking at neighbour 0 and tweeting hasthag 0

#         for t in range(2,T):

#             if t % 50 == 0:
#                 observation = (0, 1, 0, 0)

#             empirical_prior = obj_array(agent.genmodel.num_factors)
#             for f in range(agent.genmodel.num_factors):   
#                 empirical_prior[f] = spm_log(agent.genmodel.B[f][:,:, constant_action[f]].dot(qs[f]))

#             qs = agent.infer_states(True, observation)

#             history_of_idea_beliefs[t,:] = qs[0]
#             history_of_beliefs_about_other[t,:] = qs[1]

#         plt.plot(history_of_idea_beliefs[:,0],label="my beliefs")
#         plt.plot(history_of_beliefs_about_other[:,0],label="my beliefs about your beliefs")
#         plt.legend(fontsize=9)
#         plt.xlabel("Time")
#         plt.ylabel("Posterior beleifs about the idea")
#         plt.title("belief_d:  " +str(belief_d) + ", env_d: " + str(env_d))
#         plt.show()
#plt.savefig('self_vs_other_beliefs_with_change.png')
# %%
#%%
import numpy as np
from model.genmodel import GenerativeModel
from model.agent import Agent
import networkx as nx
from model.pymdp.utils import obj_array, index_list_to_onehots, sample
from model.pymdp.maths import softmax, spm_log
from model.pymdp.inference import update_posterior_states

import seaborn as sns
from matplotlib import pyplot as plt

#%%
idea_levels = 2 # the levels of beliefs that agents can have about the idea (e.g. 'True' vs. 'False', in case `idea_levels` ==2)
num_H = 2 #the number of hashtags, or observations that can shed light on the idea

h_idea_mapping = np.eye(num_H)
h_idea_mapping[:,0] = softmax(h_idea_mapping[:,0]*1.0)
h_idea_mapping[:,1] = softmax(h_idea_mapping[:,1]*1.0)

def agent_p(belief_d, env_d=None, ecb=None):
    agent_params = {

                "neighbour_params" : {
                    "ecb_precisions" : np.array([[ecb,ecb], [ecb, ecb]]),
                    "num_neighbours" : 2,
                    "env_determinism": env_d,
                    "belief_determinism": np.array([belief_d, belief_d])
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

            }
    return agent_params
plt.figure(figsize=(12,8))


idea_beliefs = np.zeros((10,10))
neighbour_beliefs = np.zeros((10,10))
T = 20


for i, ecb in enumerate(np.linspace(3,10,2)):
    for j, env_d in enumerate(np.linspace(3,10,10)):
        average = 0
        agent_params = agent_p(belief_d = 7, env_d = env_d, ecb = ecb)

        agent = Agent(**agent_params)

        observation = (0, 1, 0, 0)
        history_of_idea_beliefs = np.zeros((T,idea_levels))
        history_of_beliefs_about_other = np.zeros((T,2))
        
        qs = agent.infer_states(0, observation)
        print(qs)
        history_of_idea_beliefs[1,:] = qs[0]
        history_of_beliefs_about_other[1,:] = qs[1]

        constant_action = [0, 0, 0, 0, 0] # always looking at neighbour 0 and tweeting hasthag 0

        for t in range(1,T):

            if t % 10 == 0:
                observation = (0, 2, 0, 0)

            empirical_prior = obj_array(agent.genmodel.num_factors)
            for f in range(agent.genmodel.num_factors):   
                empirical_prior[f] = spm_log(agent.genmodel.B[f][:,:, constant_action[f]].dot(qs[f]))

            qs = agent.infer_states(1, observation)
            print(qs[0])
            
            history_of_idea_beliefs[t,:] = qs[0]
            history_of_beliefs_about_other[t,:] = qs[1]
        idea_beliefs[i, j] = average
        #neighbour_beliefs[i, j] = average
        plt.plot(history_of_idea_beliefs[:,0],label="env_d:  " +str(env_d) + ", ecb: " + str(ecb))
        #plt.plot(history_of_beliefs_about_other[:,0],label="my beliefs about your beliefs")
#plt.legend(fontsize=9)
plt.xlabel("Time")
plt.ylabel("Posterior beleifs about the idea")
#         plt.title("belief_d:  " +str(belief_d) + ", env_d: " + str(env_d))
plt.show()

plt.imshow(idea_beliefs)
plt.ylabel("Epistemic Confirmation Bias")
plt.xlabel("Environmental Determinism")
plt.title("Rate of switching of posterior over beliefs about the idea")
plt.colorbar()
plt.show()
#plt.savefig('self_vs_other_beliefs_with_change.png')
# %%
