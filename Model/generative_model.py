import numpy as np 
import itertools
import time
from pymdp.utils import obj_array, obj_array_uniform,softmax, onehot, reduce_a_matrix
from pymdp.maths import spm_log
from pymdp.learning import *
import warnings 

class GenerativeModelSuper(object):

    """
    parameters:

    ecb_precisions : an array of shape (num_neighbours, num_idea_levels)         
    num_neighbours : int

    num_H: int, number of hashtags 
    idea_levels: int

    h_idea_mapping: an array of shape (num_H, num_idea_levels) that maps how the agent believes the hashtags correspond to the idea levels
                    if h idea mapping is the identity matrix, then the agent will always believe that observing hasthag 0 represents idea 0 and observing hashtag 1 represents idea 1

    belief2tweet_mapping: an array of shape (num_H, num_idea_levels)
    E_lr: float 


    env_determinism: float
    belief_determinism: an array of length number of neighbours, representing the inverse volatility with respect to each neighbour

    """
    def __init__(
        self,
        ecb_precisions, 
        num_neighbours, 

        num_H,
        idea_levels,
    
        initial_action = None,

        h_idea_mapping = None,

        belief2tweet_mapping = None,
        E_lr = None,

        env_determinism = 9,
        belief_determinism = None,

        reduce_A = True

    ):

        
        self.num_H = num_H
        self.idea_levels = idea_levels

        if h_idea_mapping is None:
            self.h_idea_mapping = self.create_idea_mapping()
        else:
            assert h_idea_mapping.shape == (self.num_H, self.idea_levels), "Your h_idea_mapping has the wrong shape. It should be (num_H, idea_levels)"
            self.h_idea_mapping = h_idea_mapping

        self.ecb_precisions = ecb_precisions
        self.num_neighbours = num_neighbours

        
        assert np.isscalar(env_determinism), "Your env_determinism has the wrong shape. It should be a scalar"
        self.env_determinism = float(env_determinism)

        if belief_determinism is None:
            belief_determinism = np.ones((num_neighbours,))*6
        self.belief_determinism = belief_determinism   
        assert self.belief_determinism.shape == (num_neighbours,), "Your belief_determinism has the wrong shape. It should be (num_neighbours,)"

        self.belief2tweet_mapping = belief2tweet_mapping 
        self.h_control_mapping = np.eye(self.num_H)
        
        self.num_obs = [self.num_H] + (self.num_neighbours) * [self.num_H+1] + [self.num_neighbours]  # list that contains the dimensionalities of each observation modality 

        self.num_modalities = len(self.num_obs) # total number of observation modalities
        self.num_states = (1+ self.num_neighbours) * [self.idea_levels] + [self.num_H] + [self.num_neighbours] #list that contains the dimensionality of each state factor 
        self.num_factors = len(self.num_states) # total number of hidden state factors

        self.focal_h_idx = 0 # index of the observation modality corresponding to my observing my own hashtags
        self.neighbour_h_idx = [(self.focal_h_idx + n + 1) for n in range(self.num_neighbours)] # indices of the observation modalities corresponding to observation of my neighbours' hashtags
        self.who_obs_idx = self.neighbour_h_idx[-1] + 1 # index of the observation modality corresponding to the observation of one's own sampling action

        # create variables to name the hidden state factor indices
        self.focal_belief_idx = 0
        self.neighbour_belief_idx = [(self.focal_belief_idx + n + 1) for n in range(self.num_neighbours)]
        self.h_control_idx = self.neighbour_belief_idx[-1] + 1
        self.who_idx = self.h_control_idx + 1

        self.control_factor_idx = [self.h_control_idx, self.who_idx]
        num_controls = [1] * self.num_factors
        num_controls[self.h_control_idx] = self.num_H
        num_controls[self.who_idx] = self.num_neighbours
        self.num_controls = num_controls
        self.policies = self.generate_policies()
        self.E_lr = E_lr
        self.initial_action = initial_action


    def insert_multiple(self, s, indices, items):
        for idx in range(len(items)):
            s.insert(indices[idx], items[idx])
        return s
        

    def initialize_A(self):
        """initialize the A matrix and fill the modalities with their correct shape"""
        A = obj_array(self.num_modalities)
        for o_idx, o_dim in enumerate(self.num_obs): 
            modality_shape = [o_dim] + self.num_states # num_obs[m] rows and as many lagging dimensions as there are hidden states, with each lagging dimension == num_states[i]
            A[o_idx] = np.zeros(modality_shape)
        return A

    def get_null_matrix(self, o_dim, o_idx):
        null_matrix = np.zeros((o_dim,self.num_states[o_idx-1]))
        null_matrix[0,:] = np.ones(self.num_states[o_idx-1]) # create a matrix that corresponds to NOT sampling the current neighbour -- every observation is the 'null' observation because we're sampling someone else
        return null_matrix


    def fill_slice(self, A_o, A_slice, irrelevant_dimensions, fill_indices, slice_indices):
        """ 
        A_o: the slice of A you want to fill indexed by the the observation modality index (i.e. A[0] for first observation modality)
        A_slice: the numpy array with which you want to fill this slice 
        irrelevant_dimesnions: the dimensions of A that are not informative for this particular slicing 
        fill_indices: the indices of A to be filled 
        slice indices: a list of indices which you will insert into fill_indices in the function insert_multiple() """

        for item in itertools.product(*[list(range(d)) for d in irrelevant_dimensions]):
            slice_ = list(item)
            A_indices = self.insert_multiple(slice_, fill_indices, slice_indices) #here we insert the correct values for the fill indices for this slice                    
            A_o[tuple(A_indices)] = A_slice
        return A_o   

    def po_h_given_s(self, A_0):
        """ fill the 0th sensory modality which corresponds to the agent's observation of which hashtag it has tweeted
            which should map via h_control_mapping (default identity matrix) to which hashtag state the agent is in """
        num_observable_hashtags = self.h_control_mapping.shape[0]
        dimensions = [num_observable_hashtags] + self.num_states #this is the shape of the modality-specific A matrix A[o_idx] the first index is the dimension of the observation modality and then the rest is the full dimensionality of the states
        informative_dimensions = [0,self.h_control_idx+1] # these are indices of the dimensions we need to fill for this modality. we have to add 1 to h_control_idx because the first dimensions corresponds to the observation
        
        uninformative_dimensions = np.delete(dimensions, informative_dimensions) #these are the lagging dimensions that don't matter for this mapping
        
        A_0 = self.fill_slice(A_0, self.h_control_mapping,uninformative_dimensions, informative_dimensions, [slice(0,num_observable_hashtags), slice(0,num_observable_hashtags)])

        return A_0

    def po_who_given_s(self, A_o, who_obs_idx):
        """ fill the observation modality corresponding to which neighbour the agent is sampling """
        who_obs = self.num_obs[who_obs_idx]
        sampling_A = np.eye(who_obs)

        dimensions = [who_obs] + self.num_states #this is the shape of the modality-specific A matrix A[o_idx]
        informative_dimensions = [0,self.who_idx+1] # these are the indices of the dimensions we need to fill for this modality
        uninformative_dimensions = np.delete(dimensions, informative_dimensions) 
        A_o = self.fill_slice(A_o, sampling_A, uninformative_dimensions, informative_dimensions, [slice(0,who_obs), slice(0,self.num_states[self.who_idx])])
        return A_o

    def scale_idea_mapping(self,neighbour_idx, o_dim, truth_level):
        """ Scales the hashtag to idea mapping based on corresponding beliefs (truth_level) """
        ecb = self.ecb_precisions[neighbour_idx-1][truth_level]
        h_idea_mapping_scaled = np.copy(self.h_idea_mapping)
        h_idea_mapping_scaled[:,truth_level] = softmax(ecb * self.h_idea_mapping[:,truth_level])
        if h_idea_mapping_scaled[truth_level,truth_level] < self.h_idea_mapping[truth_level,truth_level]:
            warnings.warn('ECB precision scaling is not high enough!')

        # augment the h->idea mapping matrix with a row of 0s on top to account for the the null observation (this is for the case when you are sampling the agent whose modality we're considering)
        h_idea_scaled_with_null = np.zeros((o_dim,self.num_states[neighbour_idx-1]))
        h_idea_scaled_with_null[1:,:] = np.copy(h_idea_mapping_scaled)

        h_idea_with_null = np.zeros((o_dim,self.num_states[neighbour_idx-1]))

        h_idea_with_null[1:,:] = np.copy(self.h_idea_mapping)
        
        return h_idea_scaled_with_null, h_idea_with_null

    def get_broadcast_dims(self, broadcast_dims, neighbour_i): 
        """ gets the broadcast dimensions specifically for a given neighbour index """              
        broadcast_dims[self.focal_belief_idx+1] = 1
        broadcast_dims[self.who_idx+1] = 1
        broadcast_dims[neighbour_i+2] = 1

        return broadcast_dims

    def scale_A_by_ecb(self, A_o, neighbour_i, h_idea_scaled_with_null, h_idea_with_null, broadcast_dims_specific, idx_vec_o, reshape_vector, truth_level):
        """ This scales the slice of the A matrix mapping the belief state of a neighbour to the observation of this neighbours tweet 
        in the case that the agent and neighbour shar ebelifs, by the epistemic confirmation bias """
        state_dim = self.num_states[neighbour_i+1]
        for belief_level in range(state_dim):
            if truth_level == belief_level:
                idx_vec_o[neighbour_i+2] = slice(belief_level,belief_level+1,None)
                belief_level_specific_column = np.reshape(h_idea_scaled_with_null[:,truth_level],reshape_vector)
                A_o[tuple(idx_vec_o)] = np.tile(belief_level_specific_column, tuple(broadcast_dims_specific)) 
                idx_vec_o[neighbour_i+2] = slice(state_dim)
            else:
                idx_vec_o[neighbour_i+2] = slice(belief_level,belief_level+1,None)
                belief_level_specific_column = np.reshape(h_idea_with_null[:,belief_level],reshape_vector)
                A_o[tuple(idx_vec_o)] = np.tile(belief_level_specific_column, tuple(broadcast_dims_specific)) 
                idx_vec_o[neighbour_i+2] = slice(state_dim)
        return A_o

    def fill_B_states(self, matrix, precision):
        """ Fills non-control slices of the B matrix using the volatility value (precision) with the given matrix """
        B_s = np.expand_dims(softmax(matrix * precision),axis = 2)
        return B_s

    def fill_B_control_states(self, modality_shape, num_actions):
        """ Fills the control slices of the B matrix such that actions correspond to the control states """
        B_s = np.zeros(modality_shape)
        for action in range(num_actions):
            B_s[action,:,action] = np.ones(num_actions)
        return B_s

    def generate_policies(self):
        """Generate a set of policies

        Each policy is encoded as a numpy.ndarray of shape (n_steps, n_factors), where each 
        value corresponds to the index of an action for a given time step and control factor. The variable 
        `policies` that is returned is a list of each policy-specific numpy nd.array.

        If `self.control_fac_idx` does not exist, then the control factors are set to all the hidden state factors

        Returns:
        -------
        - `policies`: list of np.ndarrays, where each array within the list is a 
                        numpy.ndarray of shape (n_steps, n_factors).
                    Each value in a policy array corresponds to the index of an action for 
                    a given timestep and control factor.
        """

        if self.control_factor_idx is None:
            self.control_factor_idx = list(range(self.num_factors))
        
        n_control = []
        for f_idx, f_dim in enumerate(self.num_states):
            if f_idx in self.control_factor_idx:
                n_control.append(f_dim)
            else:
                n_control.append(1)

        policies = list(itertools.product(*[list(range(i)) for i in n_control]))

        for pol_i in range(len(policies)):
            policies[pol_i] = np.array(policies[pol_i]).reshape(1, self.num_factors)

        return policies

    def generate_policy_mapping(self):
        """
        Creates a 'link' function that maps current beliefs about the Idea (in practice, qx[0], the first marginal
        factor of the posterior beliefs about hidden states) to the prior over policies - the E matrix. 
        First, a belief2tweet_mapping is created that parameterises the weights linking the belief state
        to the probability to tweet one of the `num_H` hashtags. Then the full policy mapping (over all policies, 
        which also include other control factors like the which_neighbour factor) is generated from this
        policy mapping over just the `hashtag` control factor.
        """
        num_policies = len(self.policies)
        policy_mapping = np.zeros((num_policies, self.idea_levels))
        
        if self.belief2tweet_mapping is None:
            self.belief2tweet_mapping = np.eye(self.num_H, self.idea_levels)
        else:
            assert self.belief2tweet_mapping.shape == (self.num_H , self.idea_levels), "Your belief2tweet_mapping has the wrong shape. It should be (self.num_H , self.idea_levels)"
        self.belief2tweet_mapping = self.belief2tweet_mapping / self.belief2tweet_mapping.sum(axis=0)
        array_policies = np.array(self.policies).squeeze()
        for policy_idx, policy in enumerate(array_policies):
            for action_idx in range(self.num_H):
                normalising_constant = (array_policies[:,self.h_control_idx] == action_idx).sum()
                if policy[self.h_control_idx] == action_idx:
                    policy_mapping[policy_idx,:] = self.belief2tweet_mapping[action_idx,:] / normalising_constant
        return policy_mapping
    
    
    def create_idea_mapping(self):
        h_idea_mapping = softmax(np.array([[0,1],[1,0]])* np.random.uniform(0.3,3))

        return h_idea_mapping
    
    def get_policy_prior(self, qs_f):

        E = spm_log(self.policy_mapping.dot(qs_f))

        return E
    
    def generate_indices_for_policy_updating(self):
        
        reshape_dims_base = np.ones(len(self.num_controls),dtype=int)
        reshape_dims_per_modality = []
        tile_dims_per_modality = []

        for g in range(self.num_modalities):
            tmp = reshape_dims_base.copy()
            control_idx = np.intersect1d(self.informative_dims[g], self.control_factor_idx)
            tmp[control_idx] = np.array(self.num_controls)[list(control_idx)]

            reshape_dims_per_modality.append(tuple(tmp))

            tmp = 1 + np.array(self.num_controls) - tmp
            tile_dims_per_modality.append(tuple(tmp))
        
        return reshape_dims_per_modality, tile_dims_per_modality

    def generate_likelihood(self):
        pass 


    def generate_transition(self):
        pass

    def generate_prior_preferences(self):
        pass
    
    def generate_prior_states(self, initial_action = None):
        pass
