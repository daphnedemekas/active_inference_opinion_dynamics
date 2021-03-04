import numpy as np

""" 
params: 
-- idea levels 
-- num_H
-- num_neighbours: how many neighbours the agent has - a subset of the total number of agents  

-- h_idea_mapping: a predefined mapping between hidden states (beliefs of idea_levels) to hashtags / outcomes. if none, this will be random (initialzied inside agent). 
    h_idea_mapping is a matrix of size (num_H x idea_levels). 
    for eg h_idea_mapping[0][0] represents how much the hidden state 1 (idea level 1) gives evidence for a tweet with hashtag 1. 
    this mapping is being used to construct a probability distribution over outcomes in the A matrix - (observation|hidden state)
    -  to construct the columns of the A matrix which define the probability of seeing some hashtag given the possible combinations of beliefs. 
    the COLUMNS need to sum to 1 because, this matrix maps to the slice of the A matrix, for a given neighbour, where the columns are: 
    p(observe hashtag 1 | believe true) + p(observe hashtag 1 | believe false) = 1

-- true_false_precisions: one precision PER NEIGHBOUR. if None, will be random. 
    this is a vector of size num_neighbours representing the precisions with which the agent will update their beliefs in accordance with a given neighbours's tweets. this will scale h_idea_mapping 
    when constructing the A matrix to determine how susceptible the agent's beliefs are is to the neighbours' tweets. 
    
-- volalitility levels: a vector of size 1 + num_neighbours used to paramterize the transition matrix -- a larger volatilty level means the neighbour will be more likely to change beliefs in time. 

-- belief2tweetmapping: parameterises the policy mapping. belief 2 tweet are the weights linking each belief state to the probability of tweeting one of hashtags. this will be used to construct the mapping
between the current beliefs (qs) and the policies -- which are all combinations of actions (tweet H, look at N). but this mapping is only initialized once, but is called every time we have a new q . if none, random in genmodel.

-- starting state : need an initial state for each of the hidden states 

"""

#GLOBAL PARAMS
total_number_of_agents = 4
num_H = 2
idea_levels = 2
h_control_mapping = np.eye(num_H)


#AGENT 1 Paramaters

"""
 QUESTIONS TO BE ADDRESSED  -- HOW TO INITIALIZE AGENTS 
 1.  What does the range of precisions signify? higher precisions versus lower precisions? bigger range versus smaller range?
 2.  What about volatility levels? higher volatility -- expecting those neighbours to be less stubborn 
 3.  What is the effect of the starting state?
 4. For the C params -- higher exponent should make agents more prone to agreeing -- positive exponent should make them indifferent to the idea level they are agreeing on (symmetric)
 5. belief2tweetmapping should probably have some relation to h_idea_mapping? it maps what beliefs correspond to a higher / lower probability of tweeting a certain hashtag. need some function that creates both, perhaps?

"""

num_neighbours = 2
mia_params = {

    "neighbour_params" : {
        "precisions" : np.random.uniform(low=3.0, high=10.0, size=(num_neighbours,)),
        "num_neighbours" : num_neighbours,
        "volatility_levels": np.random.uniform(low=0.5, high=3.0, size=(num_neighbours+1,))
        },

    "idea_mapping_params" : {
        "num_H" : num_H,
        "idea_levels": idea_levels,
        "h_idea_mapping": None
        },

    "policy_params" : {
        "starting_state" : (num_neighbours+1) * [np.random.randint(idea_levels)] + [np.random.randint(num_H)] + [np.random.randint(num_neighbours)],
        "belief2tweet_mapping" : None
        },

    "C_params" : {
        "preference_shape" : None,
        "cohesion_exp" : None,
        "cohesion_temp" : None
        }
}


num_neighbours = 2
vincent_params = {
    "neighbour_params"  : {
        "precisions" : np.random.uniform(low=3.0, high=10.0, size=(num_neighbours,)),
        "num_neighbours" : num_neighbours ,
        "volatility_levels": np.random.uniform(low=0.5, high=3.0, size=(num_neighbours+1,))
        },

    "idea_mapping_params" : {
        "num_H" : num_H,
        "idea_levels": idea_levels,
        "h_idea_mapping": None
        },

    "policy_params" : {
        "starting_state" : (num_neighbours+1) * [np.random.randint(idea_levels)] + [np.random.randint(num_H)] + [np.random.randint(num_neighbours)],
        "belief2tweet_mapping" : None
        },

    "C_params" : {
        "preference_shape" : None,
        "cohesion_exp" : None,
        "cohesion_temp" : None
        }
    }

num_neighbours = 2
jules_params = {
    "neighbour_params" : {
        "precisions" : np.random.uniform(low=3.0, high=10.0, size=(num_neighbours,)),
        "num_neighbours" : num_neighbours,
        "volatility_levels": np.random.uniform(low=0.5, high=3.0, size=(num_neighbours+1,))
        },

    "idea_mapping_params" : {
        "num_H" : num_H,
        "idea_levels": idea_levels,
        "h_idea_mapping": None
        },

    "policy_params" : {
        "starting_state" : (num_neighbours+1) * [np.random.randint(idea_levels)] + [np.random.randint(num_H)] + [np.random.randint(num_neighbours)],
        "belief2tweet_mapping" : None
        },

    "C_params" : {
        "preference_shape" : None,
        "cohesion_exp" : None,
        "cohesion_temp" : None
        }
    }
