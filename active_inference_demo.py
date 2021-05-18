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

agent_params = {

            "neighbour_params" : {
                "ecb_precisions" : np.random.uniform(low=0.3, high=3.0, size=(num_neighbours, idea_levels)),
                "num_neighbours" : 2,
                "env_determinism": 5.0,
                "belief_determinism": np.array([6.0, 3.0])
                },

            "idea_mapping_params" : {
                "num_H" : num_H,
                "idea_levels": idea_levels,
                "h_idea_mapping": h_idea_mapping
                },

            "policy_params" : {
                "initial_action" : [np.random.randint(num_H), np.random.randint(2)],
                "belief2tweet_mapping" : np.eye(num_H),
                },

            "C_params" : {
                "preference_shape" : None,
                "cohesion_exp" : None,
                "cohesion_temp" : None
                }
        }

agent = Agent(**agent_params,reduce_A=True)

# %%
T = 100

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

# %%
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

# %%


plt.figure(figsize=(12,8))
plt.plot(history_of_idea_beliefs[:,0],label='My beliefs about the idea')
plt.plot(history_of_beliefs_about_other[:,0,0],label='My beliefs about Neighbour 1''s beliefs about the idea')
plt.plot(history_of_beliefs_about_other[:,0,1],label='My beliefs about Neighbour 2''s beliefs about the idea')
plt.plot(neighbour_sampling_probs[:,0],label='Probabiliy of sampling neighbour 1')
plt.plot(neighbour_sampling_probs[:,1],label='Probabiliy of sampling neighbour 2')

# plt.scatter(np.arange(T)[history_of_who_im_looking_at == 0], 0.75*np.ones(T)[history_of_who_im_looking_at==0], c = 'red')
# plt.scatter(np.arange(T)[history_of_who_im_looking_at == 1], 0.25*np.ones(T)[history_of_who_im_looking_at==1], c = 'green')

plt.legend(fontsize=18)
# plt.savefig('self_vs_other_beliefs_with_actions.png')

# %% Debugging the belief updating

# obs = observation

# initial = True

# empirical_prior = utils.obj_array(agent.genmodel.num_factors)

# if initial == True:
#     for f in range(agent.genmodel.num_factors):
#         empirical_prior[f] = spm_log(agent.genmodel.D[f])
# else:
#     for f, ns in enumerate(self.genmodel.num_states):
#         empirical_prior[f] = spm_log(agent.genmodel.B[f][:,:, int(agent.action[f])].dot(agent.qs[f]))

# # safe convert to numpy
# A = utils.to_numpy(agent.genmodel.A)

# # collect model dimensions
# if utils.is_arr_of_arr(A):
#     n_factors = A[0].ndim - 1
#     n_states = list(A[0].shape[1:])
#     n_modalities = len(A)
#     n_observations = []
#     for m in range(n_modalities):
#         n_observations.append(A[m].shape[0])
# else:
#     n_factors = A.ndim - 1
#     n_states = list(A.shape[1:])
#     n_modalities = 1
#     n_observations = [A.shape[0]]

# obs = utils.process_observation(obs, n_modalities, n_observations)
# if prior is not None:
#     prior = utils.process_prior(prior, n_factors)

# # get model dimensions
# n_modalities = len(n_observations)
# n_factors = len(n_states)

# """
# =========== Step 1 ===========
#     Loop over the observation modalities and use assumption of independence 
#     among observation modalitiesto multiply each modality-specific likelihood 
#     onto a single joint likelihood over hidden factors [size n_states]
# """

# likelihood = get_joint_likelihood(A, obs, n_states)

# likelihood = np.log(likelihood + 1e-16)

# """
# =========== Step 2 ===========
#     Create a flat posterior (and prior if necessary)
# """

# qs = np.empty(n_factors, dtype=object)
# for factor in range(n_factors):
#     qs[factor] = np.ones(n_states[factor]) / n_states[factor]

# """
# If prior is not provided, initialise prior to be identical to posterior 
# (namely, a flat categorical distribution). Take the logarithm of it (required for 
# FPI algorithm below).
# """
# if prior is None:
#     prior = np.empty(n_factors, dtype=object)
#     for factor in range(n_factors):
#         prior[factor] = np.log(np.ones(n_states[factor]) / n_states[factor] + 1e-16)

# """
# =========== Step 3 ===========
#     Initialize initial free energy
# """
# prev_vfe = calc_free_energy(qs, prior, n_factors)

# """
# =========== Step 4 ===========
#     If we have a single factor, we can just add prior and likelihood,
#     otherwise we run FPI
# """

# dF_tol = 0.001
# num_iter = 5

# if n_factors == 1:
#     qL = spm_dot(A, qs, [0])
#     return softmax(qL + prior[0])

# else:
#     """
#     =========== Step 5 ===========
#     Run the FPI scheme
#     """

#     curr_iter = 0
#     while curr_iter < num_iter and dF >= dF_tol:
#         # Initialise variational free energy
#         vfe = 0

#         # List of orders in which marginal posteriors are sequentially multiplied into the joint likelihood:
#         # First order loops over factors starting at index = 0, second order goes in reverse
#         factor_orders = [range(n_factors), range((n_factors - 1), -1, -1)]

#         # iteratively marginalize out each posterior marginal from the joint log-likelihood
#         # except for the one associated with a given factor
#         for factor_order in factor_orders:
#             for factor in factor_order:
#                 qL = spm_dot(likelihood, qs, [factor])
#                 qs[factor] = softmax(qL + prior[factor])

#         # calculate new free energy
#         vfe = calc_free_energy(qs, prior, n_factors, likelihood)

#         # stopping condition - time derivative of free energy
#         dF = np.abs(prev_vfe - vfe)
#         prev_vfe = vfe

#         curr_iter += 1