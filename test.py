import numpy as np
from Model.pymdp.utils import obj_array
from Model.pymdp.maths import spm_dot, dot_likelihood, softmax
from Model.pymdp.inference import average_states_over_policies
from Model.genmodel import GenerativeModel
from Model.agent import Agent

import itertools


num_neighbours = 5
num_H = 3
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
    

johnny = Agent(neighbour_params, idea_mapping_params, policy_params, C_params)

#need to define a starting state 
starting_state = (num_neighbours + 1) * [1] + [0] + [2]
johnny.genmodel.starting_state = starting_state
johnny.genmodel.D = johnny.genmodel.generate_prior_states()

timestep = 0
qs = johnny.infer_states(timestep)
print(qs)