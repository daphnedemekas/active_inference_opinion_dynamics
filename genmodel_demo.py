

import numpy as np 
import os 
print(os.getcwd())
print(os.path.exists("model"))
from model.genmodel_self_esteem_temp import GenerativeModel
from simulation.simtools import initialize_agent_params
from model.pymdp.maths import softmax
from simulation.model_params import self_esteem_model_params

""" Parameter space """
esteem_parameters = [1.5,0.2,-1.5]
C_params = [2,0,-1]
idea_levels = 3 # the levels of beliefs that agents can have about the idea (e.g. 'True' vs. 'False', in case `idea_levels` ==2)
num_H = 3 #the number of hashtags, or observations that can shed light on the idea
h_idea_mapping = np.eye(num_H)
h_idea_mapping[:,0] = softmax(h_idea_mapping[:,0]*0.7)
h_idea_mapping[:,1] = softmax(h_idea_mapping[:,1]*0.7)
h_idea_mapping[:,2] = softmax(h_idea_mapping[:,1]*0.7)

""" Set parameters and generate agents"""
env_d = 8
c = 0
#in the rewritten generate_likelihood() function you just h_idea_mapping directly

belief_d = 5

num_neighbours = 2

env_determinism = env_d

belief_determinism = np.absolute(np.random.normal(belief_d, 0.1, size=(num_neighbours,)) )


model_parameters = { "esteem_parameters": esteem_parameters, "C_params":C_params}
agent_constructor_params = self_esteem_model_params(num_neighbours, env_d, belief_determinism, num_H, idea_levels, h_idea_mapping, initial_tweet = 0, initial_neighbour_to_sample=1,
                    belief2tweetmapping=np.eye(num_H), E_noise=0, model_parameters=model_parameters)
neighbour_params = agent_constructor_params["neighbour_params"]
idea_mapping_params = agent_constructor_params["idea_mapping_params"]
policy_params = agent_constructor_params["policy_params"]


genmodel = GenerativeModel(
                reduce_A=True,
                **neighbour_params, **idea_mapping_params, **policy_params,
                **agent_constructor_params["model_params"])


print(genmodel.A.shape)
print(genmodel.A[0].shape)
print("(my_h_obs, my_belief, n1_belief, n2_belief,my_tweet, who ")
print(genmodel.A[1].shape)
print("(n1_h_obs, my_belief, n1_belief, n2_belief,my_tweet, who ")
print("Me observing n1 tweet, how does that affect what I believe?")
print(genmodel.A[1][:,:,2,0,0,1].shape)
print("n1 tweets, my beliefs ")
print(genmodel.A[1][:,:,1,0,0,0])
print()
print("Me observing you tweeting hashtag 0, how does that affect when my other neighbour believes?")
print(genmodel.A[1][:,0,:,0,0,0])
