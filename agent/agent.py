import numpy as np
from genmodel import GenerativeModel
from inference import update_posterior_hidden_states

class Agent(object):

    #imagine we have different types of agents such as agent_types = ['influencer','shy',etc..]
    #this would then feed into different generator functions instead of having random distributions as below

    def __init__(
        self,
        neighbour_params,
        idea_mapping_params,
        policy_params,
        C_params
        ):            

        self.genmodel = GenerativeModel(**neighbour_params, **idea_mapping_params, **policy_params, **C_params)

        def infer_states(self, observation):

            self.qs = update_posterior_hidden_states(observation, self.genmodel.A, self.genmodel.B, self.genmodel.D, **inference_hyperparams)

            return qs

        def infer_policies(self):

            self.genmodel.E = self.genmodel.get_policy_prior() 

            self.q_pi = update_posterior_policies(self.qs, self.genmodel.A, self.genmodel.B, self.genmodel.C, self.genmodel.D, self.genmodel.E, **policy_hyperparams)

            return q_pi

        def sample_action(self)

            action = q_pi.sample()

            return action

        def set_starting_state_and_priors(starting_state)
            self.genmodel.D = self.genmodel.generate_prior_states(starting_state)

        #here i just used random arrays for everything, but obviously these need to be specified by agent "roles"
        def set_hidden_states(self):
            focal_agent_beliefs = np.random.dirichlet(np.ones(2))
            neighbour_beliefs = np.ndarray([self.n_neighbors, 2])
            for n_idx in range(self.n_neighbors):
                neighbour_beliefs[n_idx] = np.random.dirichlet(np.ones(2))
            hashtag_control_state = np.random.dirichlet(np.ones(self.n_hashtags))
            attending_neighbor_control_state = np.random.dirichlet(np.ones(self.n_neighbours))

            return None

        def set_priors(self):
            focal_prior = np.random.dirichlet(np.ones(2))
            neighbour_priors = np.ndarray([self.n_neighbors, 2])
            for n_idx in range(self.n_neighbours):
                neighbour_priors[n_idx] = np.random.dirichlet(np.ones(2))
            
            return None

        def set_observations(self):
            return None
        
        def set_A_matrix(self):
            n_modalities = n_hidden_states
            return None

        def set_B_matrix(self):
            return None

        def set_C_matrix(self):
            return None 