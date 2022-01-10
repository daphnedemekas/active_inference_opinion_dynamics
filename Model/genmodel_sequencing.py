import numpy as np 
import itertools
import time
from pymdp.utils import obj_array, obj_array_uniform, onehot, reduce_a_matrix
from pymdp.maths import *
from pymdp.learning import *
import warnings 
from .generative_model import GenerativeModelSuper

def spm_log(arr):
    """
    Adds small epsilon value to an array before natural logging it
    """
    return np.log(arr + EPS_VAL)


"""

SEQUENCING (mirroring)

hashtags - ABCD

sequence - ABC

hidden state: [sequencing]

observations: mirroring


p(observe A, : | sequence started) = 1
p(observe not A or C | in sequence) = 1
p(mirroring | in sequence) = 1
p(not mirroring | not in sequence) = 1
p(observe C | sequence finished) = 1


p(sampling not X | sampled X, in sequence) = 0 -- this needs to be expanded in pymdp

preference over mirroring -- observation modality of mirroring 

--------------


"""

class GenerativeModel(GenerativeModelSuper):

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
        super().__init__(ecb_precisions, num_neighbours, num_H,idea_levels,initial_action, h_idea_mapping,belief2tweet_mapping ,E_lr ,env_determinism,belief_determinism,reduce_A)

        self.num_obs = [self.num_H] + (self.num_neighbours) * [self.num_H+1] + [self.num_neighbours] + [2] # list that contains the dimensionalities of each observation modality 

        self.num_modalities = len(self.num_obs) # total number of observation modalities

        self.B = self.generate_transition()
        self.C = self.generate_prior_preferences()

        self.E = np.ones(len(self.policies))

        self.policy_mapping = self.generate_policy_mapping()

        self.reduce_A = reduce_A

        self.initialize_A()
        self.generate_likelihood()



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
        print(truth_level)
        print(self.h_idea_mapping)
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
    
    
    def get_policy_prior(self, qs_f):

        E = spm_log(self.policy_mapping.dot(qs_f))

        return E
    
    def generate_indices_for_policy_updating(self, informative_dims):
        
        reshape_dims_base = np.ones(len(self.num_controls),dtype=int)
        reshape_dims_per_modality = []
        tile_dims_per_modality = []

        for g in range(self.num_modalities):
            tmp = reshape_dims_base.copy()
            control_idx = np.array(np.intersect1d(informative_dims[g], self.control_factor_idx),dtype=int)
            tmp[control_idx] = np.array(self.num_controls)[list(control_idx)]

            reshape_dims_per_modality.append(tuple(tmp))

            tmp = 1 + np.array(self.num_controls) - tmp
            tile_dims_per_modality.append(tuple(tmp))
        
        return reshape_dims_per_modality, tile_dims_per_modality

    

    def generate_likelihood(self):

        """ This function generates the A matrix mapping states to observations 
        The logic of this function is as followsL 
        
        For the first observation modality focal_h_idx, the only informative slice of the A matrix is that which maps to the state num_H, which corresponds 
        to the state which sets which hashtag the agent is tweeting 
        This slice should be self.h_control_idx, which is defaulted as the identity matrix, so that the agent has complete knowledge over which tweet it is tweeting 
        
        For observation modalities that correspond to the neighbours' tweets, we need to use the epistemic confirmation bias to scale the slices 
        that map the observation of their tweet to the belief about their idea, in the case that the agent and neighbour share the same beliefs and 
        the agent is in the state (who_idx) that correspodns to the observed neighbour (meaning the agent is actually observing this neighbour)

        if  the agent is in any other who_idx state ,then the observation modality corresponding to the observed neighbour should be null 
        if the agent is in the given who_idx state but the agent and neighbour do not share the same beliefs, then the slice should just be the unscaled h_idea_mapping 

        For the observation modality who_idx, which corresponds to which neighbour the agent is observing, this should be an 
        """

        #initialize the A matrix 
        A = self.initialize_A()
        
        idx_vec_s = [slice(self.num_states[f]) for f in range(self.num_factors)] #a vector of slices of the A matrix for each state factor 
        broadcast_dims = [1] + self.num_states # this is template broadcast dimension list

        for o_idx, o_dim in enumerate(self.num_obs):  #iterate over the observation modalities to fill in A for each modality 

            if o_idx == self.focal_h_idx: #this happens for o_idx == 0 -- we are in observation modality corresponding to the agent's observation of its own tweet 
                A[o_idx] = self.po_h_given_s(A[o_idx])

            if o_idx in self.neighbour_h_idx: # now we're considering one of the observation modalities corresponding to seeing my neighbour's tweets
                null_matrix = self.get_null_matrix(o_dim, o_idx) # create the null matrix to tile throughout the appropriate dimensions (this matrix is for the case when you're _not_ sampling the neighbour whose modality we're considering)

                for truth_level in range(self.num_states[self.focal_belief_idx]): # the precision of the mapping is dependent on the truth value of the hidden state 
                    h_idea_scaled_with_null, h_idea_with_null = self.scale_idea_mapping(o_idx, o_dim, truth_level)
                    idx_vec_o = [slice(0, o_dim)] + idx_vec_s.copy()
                    idx_vec_o[self.focal_belief_idx+1] = slice(truth_level,truth_level+1,None)
                    
                    for neighbour_i in range(self.num_states[self.who_idx]): #iterate over the possible observed neighbours from state factor who_idx

                        # create a list of which dimensions you need to reshape along for the broadcasted tiling
                        broadcast_dims_specific = self.get_broadcast_dims(broadcast_dims, neighbour_i)
                        idx_vec_o[self.who_idx+1] = slice(neighbour_i,neighbour_i+1,None)
                        reshape_vector = [o_dim] + [1] * self.num_factors

                        if (o_idx - 1) == neighbour_i: # this is the case when the observation modality in question `o_idx` corresponds to the modality of the neighbour we're sampling `who_i`               
                            A[o_idx] = self.scale_A_by_ecb(A[o_idx] , neighbour_i, h_idea_scaled_with_null, h_idea_with_null, broadcast_dims_specific, idx_vec_o, reshape_vector, truth_level)

                        else: # this is the case when the observation modality in question `o_idx` corresponds to a modality _other than_ the neighbour we're sampling `who_i` 
                            reshape_vector[neighbour_i+2] = self.num_states[neighbour_i+1]
                            null_matrix_reshaped = np.reshape(null_matrix,reshape_vector)
                            A[o_idx][tuple(idx_vec_o)] = np.tile(null_matrix_reshaped, tuple(broadcast_dims_specific))

            if o_idx == self.who_obs_idx:   #this is the observation modality corresponding to which neighbour the agent is samplign 
                A[o_idx] = self.po_who_given_s(A[o_idx], o_idx)

        if self.reduce_A:
            self.A_reduced = obj_array(self.num_modalities)
            informative_dims = []
            for g in range(self.num_modalities):
                self.A_reduced[g], factor_idx = reduce_a_matrix(A[g])
                informative_dims.append(factor_idx)
            self.informative_dims = informative_dims
            
            self.reshape_dims_per_modality, self.tile_dims_per_modality = self.generate_indices_for_policy_updating(informative_dims)
        self.A = A
        return A


    def generate_transition(self):
        """ This generates the transition B matrix mapping the transition between states given actions 
        
        The logic of the B matrix is as follows:
        for the first state factor focal_belief_idx, the B matrix should be the identity matrix scaled by the environmental precision 
        
        for the state factors corresponding to the nieghbours' beliefs, the B matrix should be the idenitty matrix scaled by 
        the relevant index of belief_determinism correspondign to the given neighbour 
        
        for the state factor corresponding to which hashtag the agent is tweeting, the informative slice is that 
        conditioned on the action of which tweet the agent is tweeting. The B matrix for this informative slice should be a vector of ones 
        such that the agent's control state gets updated identically to its actions 
        
        Similarly for the state factor corresponding to which neighbour the agent is sampling, this should be identically mapped
        for the slice conditioned on the action of which neighbour the agent is sampling """

        transition_identity = np.eye(self.idea_levels, self.num_H)
        B = obj_array(self.num_factors)

        for f_idx, f_dim in enumerate(self.num_states): #iterate over the state factors
            print
            print(f_idx)
            print(f_dim)
            if f_idx == self.focal_belief_idx: #the state factor corresponding to what the agent is tweeting 
                B[f_idx] = self.fill_B_states(matrix = np.eye(f_dim, f_dim), precision = self.env_determinism)
                
            if f_idx in self.neighbour_belief_idx: #the first N+1 hidden state factors are variations of the identity matrix based on belief volatiliy    
                B[f_idx] = self.fill_B_states(matrix = np.eye(f_dim, f_dim), precision = self.belief_determinism[f_idx-1])

            if f_idx == self.h_control_idx: #for the hashtag control state we have rows of ones corresponding to the next state
                B[f_idx] = self.fill_B_control_states(modality_shape = self.num_H*[f_dim] + [self.num_H], num_actions = self.num_H)
            
            if f_idx == self.who_idx: #same as above for the who control state
                B[f_idx] = self.fill_B_control_states(modality_shape = 2*[self.num_neighbours] + [self.num_neighbours], num_actions = self.num_neighbours)
        return B


    def generate_prior_preferences(self):
        # Currently there are no prior preferences
        C = obj_array(self.num_modalities)

        for o_idx, o_dim in enumerate(self.num_obs): 
            
            C[o_idx] = np.zeros(o_dim)
        #C[-1] = -100
                
        return C
    
    def generate_prior_states(self, initial_action = None):
        # Currently prior states are completely unbiased 

        D = obj_array(self.num_factors)

        # if initial_action is not None:
        #     for f_idx, f_dim in enumerate(self.num_states):

        #         if f_idx == self.focal_belief_idx or f_idx in self.neighbour_belief_idx: #the first N+1 hidden state factors are variations of the identity matrix based on stubborness
                    
        #             D[f_idx] = np.ones(f_dim)/f_dim
        #             #D[f_idx] = np.random.uniform(0,1,f_dim) #this is if we want the prior preferences to be uniform 
                
        #         elif f_idx == self.h_control_idx:
                    
        #             D[f_idx] = onehot(initial_action[f_idx],f_dim)

        #         elif f_idx == self.who_idx: 
        #             D[f_idx] = onehot(initial_action[f_idx],f_dim)
        # else:
        D = obj_array_uniform(self.num_states)
        self.D = D
        return D

