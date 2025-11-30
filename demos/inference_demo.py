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

                "C_params" : {
                    "preference_shape" : None,
                    "cohesion_exp" : None,
                    "cohesion_temp" : None
                    }
            }
    return agent_params
plt.figure(figsize=(12,8))


ecb_vals = np.linspace(3,10,10)
env_d_vals = np.linspace(3,10,10)

idea_beliefs = np.zeros((len(ecb_vals), len(env_d_vals)))
neighbour_beliefs = np.zeros((len(ecb_vals), len(env_d_vals)))
T = 20


for i, ecb in enumerate(ecb_vals):
    for j, env_d in enumerate(env_d_vals):
        average = 0
        agent_params = agent_p(belief_d = 7, env_d = env_d, ecb = ecb)

        agent = Agent(**agent_params)

        observation = (0, 1, 0, 0)
        history_of_idea_beliefs = np.zeros((T,idea_levels))
        history_of_beliefs_about_other = np.zeros((T,2))
        
        qs = agent.infer_states(False, observation)
        #print(qs)
        history_of_idea_beliefs[1,:] = qs[0]
        history_of_beliefs_about_other[1,:] = qs[1]

        constant_action = [0, 0, 0, 0, 0] # always looking at neighbour 0 and tweeting hasthag 0

        for t in range(1,T):

            #if t % 10 == 0:
            #    observation = (0, 2, 0, 0)

            empirical_prior = obj_array(agent.genmodel.num_factors)
            for f in range(agent.genmodel.num_factors):   
                empirical_prior[f] = spm_log(agent.genmodel.B[f][:,:, constant_action[f]].dot(qs[f]))

            qs2 = agent.infer_states(True, observation)
            if t < 13 and t > 8:
                average += qs2[0][0] - qs[0][0]
            qs = qs2
            
            history_of_idea_beliefs[t,:] = qs[0]
            history_of_beliefs_about_other[t,:] = qs[1]
        print(average)
        idea_beliefs[i, j] = average
        #neighbour_beliefs[i, j] = average
        plt.plot(history_of_idea_beliefs[:,0],label="env_d:  " +str(env_d) + ", ecb: " + str(ecb))
        #plt.plot(history_of_beliefs_about_other[:,0],label="my beliefs about your beliefs")
#plt.legend(fontsize=9)
plt.xlabel("Time")
plt.ylabel("Posterior beleifs about the idea")
#         plt.title("belief_d:  " +str(belief_d) + ", env_d: " + str(env_d))
plt.savefig(f"results/inference_demo_beliefs_i{i}.png")
plt.close()

plt.imshow(idea_beliefs)
plt.ylabel("Epistemic Confirmation Bias")
plt.xlabel("Environmental Determinism")
plt.title("Rate of switching of posterior over beliefs about the idea")
plt.colorbar()
plt.savefig("results/inference_demo_switching_rate.png")
plt.close()
#plt.savefig('self_vs_other_beliefs_with_change.png')
# %%
