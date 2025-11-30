#%%
import numpy as np
from model.genmodel import GenerativeModel
from model.agent import Agent
from model.pymdp import utils
from model.pymdp.utils import obj_array, index_list_to_onehots, sample, reduce_a_matrix
from model.pymdp.maths import softmax, spm_dot, spm_log, get_joint_likelihood, calc_free_energy
from model.pymdp.inference import update_posterior_states
import matplotlib.pyplot as plt 

from time import time

idea_levels = 2 # the levels of beliefs that agents can have about the idea (e.g. 'True' vs. 'False', in case `idea_levels` ==2)
num_H = 2 #the number of hashtags, or observations that can shed light on the idea
num_neighbours = 2
h_idea_mapping = np.eye(num_H)
h_idea_mapping[:,0] = softmax(h_idea_mapping[:,0]*1.0)
h_idea_mapping[:,1] = softmax(h_idea_mapping[:,1]*1.0)
h_idea_mapping = utils.softmax(np.array([[1,0],[0,1]])* np.random.uniform(0.3,3))
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


# %%
T = 100
agent = Agent(**agent_params,reduce_A=True, reduce_A_policies= True)
print(agent.genmodel.E)
neighbour_0_tweets = 1*np.ones(T) # neighbour 1 tweets a bunch of Hashtag 1's
neighbour_1_tweets = 2*np.ones(T) # neighbour 2 tweets a bunch of Hashtag 2's

my_first_neighbour = np.where(agent.genmodel.D[-1])[0][0]
my_first_tweet = sample(agent.genmodel.belief2tweet_mapping.dot(agent.genmodel.D[-2]))

if my_first_neighbour == 0:
    observation = (my_first_tweet, int(neighbour_0_tweets[0]), 0, my_first_neighbour)
elif my_first_neighbour == 1:
    observation = (my_first_tweet, 0, int(neighbour_1_tweets[0]), my_first_neighbour)

history_of_idea_beliefs = np.zeros((T,idea_levels)) # history of my own posterior over the truth/falsity of the idea
history_of_beliefs_about_other = np.zeros((T,agent.genmodel.num_states[1],num_neighbours)) # histoyr of my posterior beliefs about the beliefs of my two neighbours about the truth/falsity of the idea

qs = agent.infer_states(0, observation)


history_of_idea_beliefs[0,:] = qs[0]
history_of_beliefs_about_other[0,:,0] = qs[1]
history_of_beliefs_about_other[0,:,1] = qs[2]

agent.action = np.array([0., 0., 0., my_first_tweet, my_first_neighbour]) # start by looking at neighbour 0 and tweeting hasthag 0

history_of_who_im_looking_at = np.zeros(T)
history_of_who_im_looking_at[0] = my_first_neighbour

neighbour_sampling_probs = np.zeros((T, 2))


start = time()
q_pi_1 = agent.infer_policies()
total = time() - start
print(f"Time taken for vectorized version: {total}")


#%%
agent = Agent(**agent_params,reduce_A=True, reduce_A_policies= False)

my_first_neighbour = np.where(agent.genmodel.D[-1])[0][0]
my_first_tweet = sample(agent.genmodel.belief2tweet_mapping.dot(agent.genmodel.D[-2]))

if my_first_neighbour == 0:
    observation = (my_first_tweet, int(neighbour_0_tweets[0]), 0, my_first_neighbour)
elif my_first_neighbour == 1:
    observation = (my_first_tweet, 0, int(neighbour_1_tweets[0]), my_first_neighbour)

qs = agent.infer_states(0, observation)
agent.action = np.array([0., 0., 0., my_first_tweet, my_first_neighbour]) # start by looking at neighbour 0 and tweeting hasthag 0

start = time()
q_pi_2 = agent.infer_policies()
total = time() - start
print(f"Time taken for old version: {total}")


#%%
for t in range(1,T):


    agent.infer_states(False,observation)
    print()
    print(agent.qs[0])

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


plt.figure(figsize=(12,8))
plt.plot(history_of_idea_beliefs[:,0],label='My beliefs about the idea')
plt.plot(history_of_beliefs_about_other[:,0,0],label='My beliefs about Neighbour 1''s beliefs about the idea')
plt.plot(history_of_beliefs_about_other[:,0,1],label='My beliefs about Neighbour 2''s beliefs about the idea')
#plt.plot(neighbour_sampling_probs[:,0],label='Probabiliy of sampling neighbour 1')
#plt.plot(neighbour_sampling_probs[:,1],label='Probabiliy of sampling neighbour 2')

plt.scatter(np.arange(T)[history_of_who_im_looking_at == 0], 0.75*np.ones(T)[history_of_who_im_looking_at==0], c = 'red')
plt.scatter(np.arange(T)[history_of_who_im_looking_at == 1], 0.25*np.ones(T)[history_of_who_im_looking_at==1], c = 'green')
plt.xlabel("Time")
plt.ylabel("Beliefs")

plt.legend(fontsize=10, loc = "upper right")
plt.savefig('self_vs_other_beliefs_with_actions.png')
# %%
