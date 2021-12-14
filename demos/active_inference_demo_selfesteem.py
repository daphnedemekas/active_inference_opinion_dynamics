#%%
import numpy as np
from model.agent import Agent
import networkx as nx
from model.pymdp import utils
from model.pymdp.utils import obj_array, index_list_to_onehots, sample, reduce_a_matrix
from model.pymdp.maths import softmax, spm_dot, spm_log, get_joint_likelihood, calc_free_energy
from model.pymdp.inference import update_posterior_states

import seaborn as sns
from matplotlib import pyplot as plt

""" this is a demo that iterates over different values of epistemic confirmation bias and belief determinism and creates an agent initialized with each combination 
then the agent receives observations at each time step from two artificial neighbours that tweet opposing hashtags 
and the plots created at the end represent which belief the agent converges to (if any)"""
# %% Set up generative model
idea_levels = 2 # the levels of beliefs that agents can have about the idea (e.g. 'True' vs. 'False', in case `idea_levels` ==2)
num_H = 2 #the number of hashtags, or observations that can shed light on the idea
num_neighbours = 3
h_idea_mapping = np.eye(num_H)
h_idea_mapping[:,0] = softmax(h_idea_mapping[:,0]*1.0)
h_idea_mapping[:,1] = softmax(h_idea_mapping[:,1]*1.0)

def agent_p(belief_d, env_d=None, ecb=None, learning_rate = None):
    agent_params = {

                "neighbour_params" : {
                    "ecb_precisions" : np.array([[ecb,ecb], [ecb, ecb],[ecb, ecb]]),
                    "num_neighbours" : num_neighbours,
                    "env_determinism": env_d,
                    "belief_determinism": np.array([belief_d, belief_d, belief_d])
                    },

                "idea_mapping_params" : {
                    "num_H" : num_H,
                    "idea_levels": idea_levels,
                    "h_idea_mapping": h_idea_mapping
                    },

                "policy_params" : {
                    "initial_action" : [1, 0],
                    "belief2tweet_mapping" : np.eye(num_H),
                    "E_lr" : learning_rate
                    },

            }
    return agent_params
        # %%
fig, axs = plt.subplots(2, 2)        #plt.figure(figsize=(12,8))
fig.set_figheight(20)
fig.set_figwidth(20)
env_d = 8
c = 0

ecb = 8
belief_d = 6

agent_params = agent_p(belief_d = belief_d, env_d = env_d, ecb = ecb, learning_rate = 0.3)

agent = Agent(**agent_params,reduce_A=True)
T = 100 #the number of timesteps 

neighbour_0_tweets = 1*np.ones(T) # neighbour 1 tweets a bunch of Hashtag 1's
neighbour_1_tweets = 2*np.ones(T) # neighbour 2 tweets a bunch of Hashtag 2's

neighbour_0_rewards = 1*np.ones(T) # neighbour 1 tweets a bunch of Hashtag 1's
neighbour_1_rewards = 0*np.ones(T) # neighbour 2 tweets a bunch of Hashtag 2's

my_first_neighbour = 0 #initial action 
my_first_tweet = 0 #initial action 

my_first_reward = 0


if my_first_neighbour == 0:
    observation = (my_first_tweet, int(neighbour_0_tweets[0]), 0,0, my_first_neighbour, my_first_reward, int(neighbour_0_rewards[0]),  int(neighbour_1_rewards[0]),int(neighbour_1_rewards[0]))
elif my_first_neighbour == 1:
    observation = (my_first_tweet, 0, 0,int(neighbour_1_tweets[0]), my_first_neighbour, my_first_reward, int(neighbour_0_rewards[0]),int(neighbour_1_rewards[0]),int(neighbour_1_rewards[0]))

#just initializing arrays for the plots 
history_of_idea_beliefs = np.zeros((T,idea_levels)) # history of my own posterior over the truth/falsity of the idea
history_of_beliefs_about_other = np.zeros((T,agent.genmodel.num_states[1],num_neighbours)) # histoyr of my posterior beliefs about the beliefs of my two neighbours about the truth/falsity of the idea

#the first posterior distribution over states for timestep 0
print(observation)
qs = agent.infer_states(0, observation)

#initializing the beliefs to start at 0.5
history_of_idea_beliefs[0,:] = 0.5
print(qs[1])
history_of_beliefs_about_other[0,:,0] = 0.5
history_of_beliefs_about_other[0,:,1] = 0.5

#then updating the history of the beliefs to be what the agent just inferred on line 87
history_of_idea_beliefs[1,:] = qs[0]
print(qs[1])
history_of_beliefs_about_other[1,:,0] = qs[1]
history_of_beliefs_about_other[1,:,1] = qs[2]

#here we hardcode the action based on who the agent looked at / tweeted initially 
agent.action = np.array([0., 0., 0., my_first_tweet, my_first_neighbour]) # start by looking at neighbour 0 and tweeting hasthag 0

#initializing arrays for plots 
history_of_who_im_looking_at = np.zeros(T)
history_of_who_im_looking_at[0] = my_first_neighbour

neighbour_sampling_probs = np.zeros((T, 2))
neighbour_sampling_probs[0] = [0.5,0.5]
neighbour_sampling_probs[1] = [0.5,0.5]

#this is the loop over time 
for t in range(2,T):

    agent.infer_states(t-1,observation) #infer the states based on the previous observation 

    q_pi = agent.infer_policies() #inferring the policies 

    neighbour_sampling_probs[t,0] = q_pi[0] + q_pi[2] # add the probabilities of policies corresponding to sampling neighbour 0
    neighbour_sampling_probs[t,1] = q_pi[1] + q_pi[3] # add the probabilities of policies corresponding to sampling neighbour 1

    action = agent.sample_action() #sampling the action from the policies 

    if action[-1] == 0.: #if you read neighbour 0 then update your observation based on the fact that neighbour 0 only tweets tweet 0

        what_im_reading = int(neighbour_0_tweets[t])
        who_im_inspecting = 0
        what_im_tweeting = int(action[-2])

        observation = (what_im_tweeting, what_im_reading, 0, who_im_inspecting)

    elif action[-1] == 1.: #if you read neighbour 1 then update your observation based on the fact that neighbour 0 only tweets tweet 1


        what_im_reading = int(neighbour_1_tweets[t])
        who_im_inspecting = 1
        what_im_tweeting = int(action[-2])

        observation = (what_im_tweeting, 0, what_im_reading, who_im_inspecting)

    history_of_who_im_looking_at[t] = who_im_inspecting
    history_of_idea_beliefs[t,:] = agent.qs[0]
    history_of_beliefs_about_other[t,:,0] = agent.qs[1]
    history_of_beliefs_about_other[t,:,1] = agent.qs[2]

#plotting
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
