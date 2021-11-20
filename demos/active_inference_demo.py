#%%
import numpy as np
from Model.genmodel import GenerativeModel
from Model.agent import Agent
import networkx as nx
from Model.pymdp import utils
from Model.pymdp.utils import obj_array, index_list_to_onehots, sample, reduce_a_matrix
from Model.pymdp.maths import softmax, spm_dot, spm_log, get_joint_likelihood, calc_free_energy
from Model.pymdp.inference import update_posterior_states

import seaborn as sns
from matplotlib import pyplot as plt

# %% Set up generative model
idea_levels = 2 # the levels of beliefs that agents can have about the idea (e.g. 'True' vs. 'False', in case `idea_levels` ==2)
num_H = 2 #the number of hashtags, or observations that can shed light on the idea
num_neighbours = 2 
h_idea_mapping = np.eye(num_H)
h_idea_mapping[:,0] = softmax(h_idea_mapping[:,0]*1.0)
h_idea_mapping[:,1] = softmax(h_idea_mapping[:,1]*1.0)

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
                    "initial_action" : [1, 0],
                    "belief2tweet_mapping" : np.eye(num_H),
                    "E_lr" : 0.3
                    },

                "C_params" : {
                    "preference_shape" : None,
                    "cohesion_exp" : None,
                    "cohesion_temp" : None
                    }
            }
    return agent_params
        # %%
fig, axs = plt.subplots(2, 2)        #plt.figure(figsize=(12,8))
fig.set_figheight(20)
fig.set_figwidth(20)
env_d = 8
c = 0
for i, ecb in enumerate(np.linspace(3,9,2)):
    print("ECB")
    print(ecb)
    for j, belief_d in enumerate(np.linspace(3,9,2)):
        print("BELIEF D")
        print(belief_d)
        agent_params = agent_p(belief_d = belief_d, env_d = env_d, ecb = ecb)
        
        agent = Agent(**agent_params,reduce_A=True)
        T = 100

        neighbour_0_tweets = 1*np.ones(T) # neighbour 1 tweets a bunch of Hashtag 1's
        neighbour_1_tweets = 2*np.ones(T) # neighbour 2 tweets a bunch of Hashtag 2's

        my_first_neighbour = 0
        my_first_tweet = 0

        if my_first_neighbour == 0:
            observation = (my_first_tweet, int(neighbour_0_tweets[0]), 0, my_first_neighbour)
        elif my_first_neighbour == 1:
            observation = (my_first_tweet, 0, int(neighbour_1_tweets[0]), my_first_neighbour)

        history_of_idea_beliefs = np.zeros((T,idea_levels)) # history of my own posterior over the truth/falsity of the idea
        history_of_beliefs_about_other = np.zeros((T,agent.genmodel.num_states[1],num_neighbours)) # histoyr of my posterior beliefs about the beliefs of my two neighbours about the truth/falsity of the idea

        qs = agent.infer_states(0, observation)

        history_of_idea_beliefs[0,:] = 0.5
        print(qs[1])
        history_of_beliefs_about_other[0,:,0] = 0.5
        history_of_beliefs_about_other[0,:,1] = 0.5

        history_of_idea_beliefs[1,:] = qs[0]
        print(qs[1])
        history_of_beliefs_about_other[1,:,0] = qs[1]
        history_of_beliefs_about_other[1,:,1] = qs[2]

        agent.action = np.array([0., 0., 0., my_first_tweet, my_first_neighbour]) # start by looking at neighbour 0 and tweeting hasthag 0

        history_of_who_im_looking_at = np.zeros(T)
        history_of_who_im_looking_at[0] = my_first_neighbour

        neighbour_sampling_probs = np.zeros((T, 2))
        neighbour_sampling_probs[0] = [0.5,0.5]
        neighbour_sampling_probs[1] = [0.5,0.5]

        for t in range(2,T):

            agent.infer_states(t-1,observation)

            q_pi = agent.infer_policies()
            neighbour_sampling_probs[t,0] = q_pi[0] + q_pi[2] # add the probabilities of policies corresponding to sampling neighbour 0
            neighbour_sampling_probs[t,1] = q_pi[1] + q_pi[3] # add the probabilities of policies corresponding to sampling neighbour 1

            action = agent.sample_action()

            if action[-1] == 0.:

                what_im_reading = int(neighbour_0_tweets[t])
                who_im_inspecting = 0
                what_im_tweeting = int(action[-2])

                observation = (what_im_tweeting, what_im_reading, 0, who_im_inspecting)

            elif action[-1] == 1.:

                what_im_reading = int(neighbour_1_tweets[t])
                who_im_inspecting = 1
                what_im_tweeting = int(action[-2])

                observation = (what_im_tweeting, 0, what_im_reading, who_im_inspecting)

            history_of_who_im_looking_at[t] = who_im_inspecting
            history_of_idea_beliefs[t,:] = agent.qs[0]
            history_of_beliefs_about_other[t,:,0] = agent.qs[1]
            history_of_beliefs_about_other[t,:,1] = agent.qs[2]
        print(c)
        axs[i, j].plot(history_of_idea_beliefs[:,0],label="beliefs about idea", color = 'tomato', linewidth = 4 )
        axs[i, j].plot(history_of_beliefs_about_other[:,0,0],label='beliefs about neighbour 1', color = 'cornflowerblue')
        axs[i, j].plot(history_of_beliefs_about_other[:,0,1],label='beliefs about neighbour 2', color = 'navy')
        axs[i, j].set_title( r'$\gamma$' + ": " + str(ecb) + ", " r'$\omega_{s}$' + ": " + str(belief_d), fontsize=30)
        axs[i, j].tick_params(labelsize = 25)
        axs[i, j].tick_params(labelsize = 25)

        #axs[i, j].legend(fontsize=15)

        #plt.show()
        plt.figure(figsize=(12,8))
        plt.title("Sampling probabilities over time, ecb: " + str(ecb) + ", " r'$\omega_{s}$' + ": " + str(belief_d), fontsize=15)
        fig = plt.figure(figsize=(10, 3))
        plt.imshow(neighbour_sampling_probs.T[:,3:33], cmap = 'gray')
        plt.title("Probability of sampling")
        plt.ylabel("Neighbour")
        plt.xlabel("Time")
        #plt.savefig("test2")


        #axs[i, j].plot(neighbour_sampling_probs[:,0],label="prob of sampling neighbour 1",  color = 'darkgreen')
        #axs[i, j].plot(neighbour_sampling_probs[:,1],label="prob of sampling neighbour 2", color = 'springgreen')

        #axs[i, j].legend(fontsize=15)

#plt.legend(fontsize = 20)
#fig.suptitle("Three agent demo with opposing neighbours", fontsize = 30)
#plt.savefig("Figure 1")
#plt.show()

        # plt.scatter(np.arange(T)[history_of_who_im_looking_at == 0], 0.75*np.ones(T)[history_of_who_im_looking_at==0], c = 'red')
        # plt.scatter(np.arange(T)[history_of_who_im_looking_at == 1], 0.25*np.ones(T)[history_of_who_im_looking_at==1], c = 'green')

        #plt.legend(fontsize=18)


        # %%
