import numpy as np

class Agent(object):


    #imagine we have different types of agents such as agent_types = ['influencer','shy',etc..]
    #this would then feed into different generator functions instead of having random distributions as below

    def __init__(
        self,
        A=None,
        B=None,
        C=None,
        D=None,
        n_neighbours=None,
        n_hashtags=None,
        n_hidden_states = None
    ):

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