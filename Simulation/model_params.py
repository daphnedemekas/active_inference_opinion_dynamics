
def epistemic_community_params(ecb_precisions, num_neighbours, env_determinism, belief_determinism, num_H, num_idea_levels, h_idea_mapping, initial_tweet, initial_neighbour_to_sample,
                                belief2tweetmapping, E_noise):
    agent_constructor_params = {

    "neighbour_params" : {
        "ecb_precisions" :  ecb_precisions,
        "num_neighbours" : num_neighbours,
        "env_determinism":  env_determinism,
        "belief_determinism": belief_determinism
        },

    "idea_mapping_params" : {
        "num_H" : num_H,
        "num_idea_levels": num_idea_levels,
        "h_idea_mapping": h_idea_mapping
        },

    "policy_params" : {
        "initial_action" : [initial_tweet, initial_neighbour_to_sample],
        "belief2tweet_mapping" : belief2tweetmapping,
        "E_lr" : E_noise
        },
    }
    return agent_constructor_params



def sequencing_model_params(num_neighbours, env_determinism, belief_determinism, num_H, num_idea_levels, h_idea_mapping, initial_tweet, initial_neighbour_to_sample,
                                belief2tweetmapping, E_noise, model_parameters):
    
    agent_constructor_params = {

    "neighbour_params" : {
        "num_neighbours" : num_neighbours,
        "env_determinism":  env_determinism,
        "belief_determinism": belief_determinism
        },

    "idea_mapping_params" : {
        "num_H" : num_H,
        "num_idea_levels": num_idea_levels,
        "h_idea_mapping": h_idea_mapping
        },

    "policy_params" : {
        "initial_action" : [initial_tweet, initial_neighbour_to_sample],
        "belief2tweet_mapping" : belief2tweetmapping,
        "E_lr" : E_noise
        },
    
    "model_params": model_parameters

    }
    return agent_constructor_params

def self_esteem_model_params(num_neighbours, env_determinism, belief_determinism, num_H, num_idea_levels, h_idea_mapping, initial_tweet, initial_neighbour_to_sample,
                                belief2tweetmapping, E_noise, model_parameters):
    
    agent_constructor_params = {

    "neighbour_params" : {
        "num_neighbours" : num_neighbours,
        "env_determinism":  env_determinism,
        "belief_determinism": belief_determinism
        },

    "idea_mapping_params" : {
        "num_H" : num_H,
        "num_idea_levels": num_idea_levels,
        "h_idea_mapping": h_idea_mapping
        },

    "policy_params" : {
        "initial_action" : [initial_tweet, initial_neighbour_to_sample],
        "belief2tweet_mapping" : belief2tweetmapping,
        "E_lr" : E_noise
        },
    
    "model_params": model_parameters

    }
    return agent_constructor_params