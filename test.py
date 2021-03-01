import numpy as np
from Model.pymdp.utils import obj_array, index_list_to_onehots, sample
from Model.pymdp.maths import spm_dot, dot_likelihood, softmax
from Model.pymdp.inference import average_states_over_policies
from Model.genmodel import GenerativeModel
from Model.agent import Agent

import itertools

num_neighbours = 2
num_H = 2
idea_levels = 2

#h_idea_mapping = np.array([[0.9, 0.1], [0.1, 0.9]])

true_false_precisions = np.random.uniform(low=3.0, high=10.0, size=(num_neighbours,))
h_control_mapping = np.eye(num_H)

volatility_levels = np.random.uniform(low=0.5, high=3.0, size=(num_neighbours+1,)) # in theory, the first hidden state factor (my beliefs) should be parameterised based on the focal agent's beliefs _about the inherent stochasticity_ of the world,

# MY PARAMETERS 
neighbour_params = {
    "precisions" : true_false_precisions,
    "num_neighbours" : num_neighbours,
    "volatility_levels": volatility_levels
    }

idea_mapping_params = {
    "num_H" : num_H,
    "idea_levels": idea_levels,
    "h_idea_mapping": None
    }

policy_params = {
    "starting_state" : None,
    "belief2tweet_mapping" : None
    }

C_params = {
    "preference_shape" : None,
    "cohesion_exp" : None,
    "cohesion_temp" : None
    }
    

#need to define a starting state 
# starting_state = (num_neighbours + 1) * [1] + [0] + [2]
# starting_state = (num_neighbours + 1) * [1] + [0] + [1]
starting_state = (num_neighbours+1) * [np.random.randint(idea_levels)] + [0] + [1]
policy_params['starting_state'] = starting_state

johnny = Agent(neighbour_params, idea_mapping_params, policy_params, C_params)

state_vector = index_list_to_onehots(starting_state, johnny.genmodel.num_states)
observations = [sample(spm_dot(johnny.genmodel.A[m],state_vector)) for m in range(johnny.genmodel.num_modalities)]

# print(johnny.genmodel.num_states)
# print(johnny.genmodel.num_factors)

# print(johnny.genmodel.num_obs)
# print(johnny.genmodel.num_modalities)


print('H idea mapping:\n')
print(johnny.genmodel.h_idea_mapping)
qs = johnny.infer_states(0, tuple(observations))
print('Beliefs about the idea:\n')
print(qs[0])
q_pi = johnny.infer_policies(qs)
print('Probability over policies:\n')
print(q_pi)
action = johnny.sample_action()

hashtag_names = ['Hashtag 1', 'Hashtag 2']
neighbour_names = ['Mia', 'Vincent']
print(f'What Johnny will tweet: {hashtag_names[int(action[johnny.genmodel.h_control_idx])]} \n')
print(f'Who Johnny will look at: {neighbour_names[int(action[johnny.genmodel.who_idx])]} \n')
# print(qs)

