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
num_neighbours = 2 
h_idea_mapping = np.eye(num_H)
h_idea_mapping[:,0] = softmax(h_idea_mapping[:,0]*0.1)
h_idea_mapping[:,1] = softmax(h_idea_mapping[:,1]*0.1)

agent_params = {

            "neighbour_params" : {
                # "precisions" : np.random.uniform(low=0.3, high=3.0, size=(2,)),
                "precisions" : np.array([1.0,10.0]),
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

T = 100

neighbour_0_tweets = 1*np.ones(T) # neighbour 1 tweets a bunch of Hashtag 1's
neighbour_1_tweets = 2*np.ones(T) # neighbour 2 tweets a bunch of Hashtag 2's

my_first_neighbour = 1
my_first_tweet = 1

# observation = (my_first_tweet, int(neighbour_0_tweets[0]), 0, my_first_tweet, my_first_neighbour)
observation = (my_first_tweet, 0, int(neighbour_1_tweets[0]), my_first_tweet, my_first_neighbour)

history_of_idea_beliefs = np.zeros((T,idea_levels)) # history of my own posterior over the truth/falsity of the idea
history_of_beliefs_about_other = np.zeros((T,agent.genmodel.num_states[1],num_neighbours)) # histoyr of my posterior beliefs about the beliefs of my two neighbours about the truth/falsity of the idea

qs = agent.infer_states(True, observation)

history_of_idea_beliefs[0,:] = qs[0]
history_of_beliefs_about_other[0,:,0] = qs[1]
history_of_beliefs_about_other[0,:,1] = qs[2]

agent.action = np.array([0., 0., 0., my_first_tweet, my_first_neighbour]) # start by looking at neighbour 0 and tweeting hasthag 0

history_of_who_im_looking_at = np.zeros(T)
history_of_who_im_looking_at[0] = my_first_neighbour

neighbour_sampling_probs = np.zeros((T, 2))
# %%
for t in range(1,T):

    # empirical_prior = obj_array(agent.genmodel.num_factors)
    # for f in range(agent.genmodel.num_factors):   
    #     empirical_prior[f] = spm_log(agent.genmodel.B[f][:,:, action[f]].dot(qs[f]))

    # qs = update_posterior_states(observation, agent.genmodel.A, prior=empirical_prior, **agent.inference_params)

    # agent.qs = qs

    agent.infer_states(False,observation)

    q_pi = agent.infer_policies(qs)
    neighbour_sampling_probs[t,0] = q_pi[0] + q_pi[2]
    neighbour_sampling_probs[t,1] = q_pi[1] + q_pi[3]

    action = agent.sample_action()

    if action[-1] == 0.:

        what_im_reading = int(neighbour_0_tweets[t])
        who_im_inspecting = 0
        what_im_tweeting = int(action[-2])

        observation = (what_im_tweeting, what_im_reading, 0, what_im_tweeting, who_im_inspecting)

    elif action[-1] == 1.:

        what_im_reading = int(neighbour_1_tweets[t])
        who_im_inspecting = 1
        what_im_tweeting = int(action[-2])

        observation = (what_im_tweeting, 0, what_im_reading, what_im_tweeting, who_im_inspecting)

    history_of_who_im_looking_at[t] = who_im_inspecting
    history_of_idea_beliefs[t,:] = agent.qs[0]
    history_of_beliefs_about_other[t,:,0] = agent.qs[1]
    history_of_beliefs_about_other[t,:,1] = agent.qs[2]

# %%


plt.figure(figsize=(12,8))
plt.plot(history_of_idea_beliefs[:,0],label='My beliefs about the idea')
plt.plot(history_of_beliefs_about_other[:,0,0],label='My beliefs about Neighbour 1''s beliefs about the idea')
plt.plot(history_of_beliefs_about_other[:,0,1],label='My beliefs about Neighbour 2''s beliefs about the idea')
# plt.plot(neighbour_sampling_probs[:,0],label='Probabiliy of sampling neighbour 1')
# plt.plot(neighbour_sampling_probs[:,1],label='Probabiliy of sampling neighbour 2')

plt.scatter(np.arange(T)[history_of_who_im_looking_at == 0], 0.75*np.ones(T)[history_of_who_im_looking_at==0], c = 'red')
plt.scatter(np.arange(T)[history_of_who_im_looking_at == 1], 0.25*np.ones(T)[history_of_who_im_looking_at==1], c = 'green')

plt.legend(fontsize=18)
plt.savefig('self_vs_other_beliefs_with_actions.png')
# %%
