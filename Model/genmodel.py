import numpy as np 
import itertools
import time
from .pymdp.utils import obj_array, obj_array_uniform, insert_multiple, softmax, onehot, reduce_a_matrix
from .pymdp.maths import spm_log
from .pymdp.learning import *
import warnings 
class GenerativeModel(object):

    #imagine we have different types of agents such as agent_types = ['influencer','shy',etc..]
    #this would then feed into different generator functions instead of having random distributions as below

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

        preference_shape = None,
        cohesion_exp = None,
        cohesion_temp = None,

        env_determinism = None,
        belief_determinism = None,

        reduce_A = False


    ):

        
        self.num_H = num_H
        self.idea_levels = idea_levels

        self.h_idea_mapping = h_idea_mapping

        if self.h_idea_mapping is None:
            self.h_idea_mapping = self.create_idea_mapping()
        else:
            assert self.h_idea_mapping.shape == (self.num_H, self.idea_levels), "Your h_idea_mapping has the wrong shape. It should be (num_H, idea_levels)"

        if preference_shape is None:
            self.preference_shape = "linear"
        if cohesion_exp is None:
            self.cohesion_exp = 2.0
        if cohesion_temp is None:
            self.cohesion_temp = 5.0

        self.ecb_precisions = ecb_precisions
        self.num_neighbours = num_neighbours
        self.num_cohesion_levels = 2 * (self.num_neighbours+1)

        self.env_determinism = float(env_determinism)

        if self.env_determinism is None:
            self.env_determinism = np.random.uniform(low=0.5, high=3.0)
        else:
            assert np.isscalar(self.env_determinism), "Your env_determinism has the wrong shape. It should be a scalar"


        self.belief_determinism = belief_determinism   
        if self.belief_determinism is None:
            self.belief_determinism = np.random.uniform(low=0.5, high=3.0, size=(num_neighbours,))
        else:
            assert self.belief_determinism.shape == (num_neighbours,), "Your belief_determinism has the wrong shape. It should be (num_neighbours,)"


        self.belief2tweet_mapping = belief2tweet_mapping 
        self.h_control_mapping = np.eye(self.num_H)
        self.num_obs = [self.num_H] + (self.num_neighbours) * [self.num_H+1] + [self.num_neighbours]                               # list that contains the dimensionalities of each sensory   

        self.num_modalities = len(self.num_obs) # total number of observation modalities
        self.num_states = (1+ self.num_neighbours) * [self.idea_levels] + [self.num_H] + [self.num_neighbours]
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

        start = time.time()
        self.A = self.generate_likelihood()

        self.E_lr = E_lr
        
        if reduce_A:
            self.A_reduced = obj_array(self.num_modalities)
            self.informative_dims = []
            for g in range(self.num_modalities):
                self.A_reduced[g], factor_idx = reduce_a_matrix(self.A[g])
                self.informative_dims.append(factor_idx)
            self.reshape_dims_per_modality, self.tile_dims_per_modality = self.generate_indices_for_policy_updating()
            del self.A
        self.B = self.generate_transition()
        self.C = self.generate_prior_preferences()

        self.E = np.ones(len(self.policies))

        self.policy_mapping = self.generate_policy_mapping()

    def generate_likelihood(self):

        #initialize the A matrix 
        A = obj_array(self.num_modalities)
        for o_idx, o_dim in enumerate(self.num_obs):
            modality_shape = [o_dim] + self.num_states # num_obs[m] rows and as many lagging dimensions as there are hidden states, with each lagging dimension == num_states[i]
            A[o_idx] = np.zeros(modality_shape)
        
        idx_vec_s = [slice(self.num_states[f]) for f in range(self.num_factors)]
        broadcast_dims = [1] + self.num_states # this is template broadcast dimension list

        #iterate over the sensory modalities 
        for o_idx, o_dim in enumerate(self.num_obs):
            #begin with modality 1
            if o_idx == self.focal_h_idx: #this happens for o_idx == 0 -- we are in the 0th sensory modality (my beliefs about H) 

                h_obs = self.num_H #we only observe the two hashtags, not the null observation
                dimensions = [h_obs] + self.num_states #this is the shape of the modality-specific A matrix A[o_idx]
                fill_indices = [0,self.h_control_idx+1] # these are indices of the dimensions we need to fill for this modality
                fill_dimensions = np.delete(dimensions, fill_indices) 
                
                for item in itertools.product(*[list(range(d)) for d in fill_dimensions]):
                    slice_ = list(item)
                    A_indices = insert_multiple(slice_, fill_indices, [slice(0,h_obs), slice(0,self.num_H)]) #here we insert the correct values for the fill indices for this slice                    
                    A[o_idx][tuple(A_indices)] = self.h_control_mapping

            if o_idx in self.neighbour_h_idx: # now we're considering one of the observation modalities corresponding to seeing my neighbour's tweets

                for truth_level in range(self.num_states[self.focal_belief_idx]): # the precision of the mapping is dependent on the truth value of the hidden state 
                                                                    # this reflects the idea that 
                    h_idea_mapping_scaled = np.copy(self.h_idea_mapping)
                    if isinstance(self.ecb_precisions,np.ndarray):
                        h_idea_mapping_scaled[:,truth_level] = softmax(self.ecb_precisions[o_idx-1][truth_level] * self.h_idea_mapping[:,truth_level])
                        if h_idea_mapping_scaled[truth_level,truth_level] < self.h_idea_mapping[truth_level,truth_level]:
                            warnings.warn('ECB precision scaling is not high enough!')
                    idx_vec_o = [slice(0, o_dim)] + idx_vec_s.copy()
                    idx_vec_o[self.focal_belief_idx+1] = slice(truth_level,truth_level+1,None)

                    # augment the h->idea mapping matrix with a row of 0s on top to account for the the null observation (this is for the case when you are sampling the agent whose modality we're considering)
                    h_idea_scaled_with_null = np.zeros((o_dim,self.num_states[o_idx-1]))
                    h_idea_scaled_with_null[1:,:] = np.copy(h_idea_mapping_scaled)

                    h_idea_with_null = np.zeros((o_dim,self.num_states[o_idx-1]))
                    h_idea_with_null[1:,:] = np.copy(self.h_idea_mapping)

                    # create the null matrix to tile throughout the appropriate dimensions (this matrix is for the case when you're _not_ sampling the neighbour whose modality we're considering)
                    null_matrix = np.zeros((o_dim,self.num_states[o_idx-1]))
                    null_matrix[0,:] = np.ones(self.num_states[o_idx-1]) # every observation is the 'null' observation because we're sampling someone else

                    for neighbour_i in range(self.num_states[self.who_idx]):

                        # create a list of which dimensions you need to reshape along for the broadcasted tiling
                        broadcast_dims_specific = broadcast_dims.copy()
                        broadcast_dims_specific[self.focal_belief_idx+1] = 1
                        broadcast_dims_specific[self.who_idx+1] = 1
                        broadcast_dims_specific[neighbour_i+2] = 1

                        idx_vec_o[self.who_idx+1] = slice(neighbour_i,neighbour_i+1,None)

                        reshape_vector = [o_dim] + [1] * self.num_factors
                        # reshape_vector[neighbour_i+2] = self.num_states[neighbour_i+1] # this sets the correspondong factor of the reshape vector (corresponding to neighbour_i's belief states) to the correct number

                        if (o_idx - 1) == neighbour_i: # this is the case when the observation modality in question `o_idx` corresponds to the modality of the neighbour we're sampling `who_i`               
                            for belief_level in range(self.num_states[neighbour_i+1]):
                                if truth_level == belief_level:
                                    idx_vec_o[neighbour_i+2] = slice(belief_level,belief_level+1,None)
                                    belief_level_specific_column = np.reshape(h_idea_scaled_with_null[:,truth_level],reshape_vector)
                                    A[o_idx][tuple(idx_vec_o)] = np.tile(belief_level_specific_column, tuple(broadcast_dims_specific)) 
                                    idx_vec_o[neighbour_i+2] = slice(self.num_states[neighbour_i+1])
                                else:
                                    idx_vec_o[neighbour_i+2] = slice(belief_level,belief_level+1,None)
                                    belief_level_specific_column = np.reshape(h_idea_with_null[:,belief_level],reshape_vector)
                                    A[o_idx][tuple(idx_vec_o)] = np.tile(belief_level_specific_column, tuple(broadcast_dims_specific)) 
                                    idx_vec_o[neighbour_i+2] = slice(self.num_states[neighbour_i+1])
                        else: # this is the case when the observation modality in question `o_idx` corresponds to a modality _other than_ the neighbour we're sampling `who_i` 
                            reshape_vector[neighbour_i+2] = self.num_states[neighbour_i+1]
                            null_matrix_reshaped = np.reshape(null_matrix,reshape_vector)
                            A[o_idx][tuple(idx_vec_o)] = np.tile(null_matrix_reshaped, tuple(broadcast_dims_specific))

            if o_idx == self.who_obs_idx:   

                sampling_A = np.eye(self.num_obs[self.who_obs_idx])

                who_obs = self.num_obs[self.who_obs_idx]

                dimensions = [who_obs] + self.num_states #this is the shape of the modality-specific A matrix A[o_idx]
                fill_indices = [0,self.who_idx+1] # these are the indices of the dimensions we need to fill for this modality
                fill_dimensions = np.delete(dimensions, fill_indices) 
                
                for item in itertools.product(*[list(range(d)) for d in fill_dimensions]):
                    slice_ = list(item)
                    A_indices = insert_multiple(slice_, fill_indices, [slice(0,who_obs), slice(0,self.num_states[self.who_idx])]) #here we insert the correct values for the fill indices for this slice                    
                    A[o_idx][tuple(A_indices)] = sampling_A
        return A


    def generate_transition(self):

        transition_identity = np.eye(self.idea_levels, self.num_H)
        B = obj_array(self.num_factors)

        for f_idx, f_dim in enumerate(self.num_states):

            if f_idx == self.focal_belief_idx:
                transition_identity = np.eye(f_dim, f_dim)
                B[f_idx] = np.expand_dims(softmax(transition_identity * self.env_determinism), axis = 2)
            
            if f_idx in self.neighbour_belief_idx: #the first N+1 hidden state factors are variations of the identity matrix based on belief volatiliy
                
                transition_identity = np.eye(f_dim, f_dim)
                #expand dimension so we can fit with the length of the policy arrays 
                B[f_idx] = np.expand_dims(softmax(transition_identity * self.belief_determinism[f_idx-1]), axis = 2)
            
            if f_idx == self.h_control_idx: #for the hashtag control state we have rows of ones corresponding to the next state

                b_matrix_shape = self.num_H*[f_dim] + [self.num_H]
                B[f_idx] = np.zeros(b_matrix_shape)
                for action in range(self.num_H):
                    B[f_idx][action,:,action] = np.ones(self.num_H)
            
            if f_idx == self.who_idx: #same as above for the who control state

                num_actions = self.num_neighbours
                modality_shape = 2*[self.num_neighbours] + [num_actions] # @ TODO: make dimensionality of control states dissociable from the hidden state dimensionality for this factor
                B[f_idx] = np.zeros(modality_shape)
                for action in range(num_actions): # @ TODO: make dimensionality of control states dissociable from the hidden state dimensionality for this factor
                    B[f_idx][action,:,action] = np.ones(self.num_neighbours)

    
        return B

    def generate_prior_preferences(self):

        C = obj_array(self.num_modalities)

        for o_idx, o_dim in enumerate(self.num_obs): 
            
            C[o_idx] = np.zeros(o_dim)
                
        return C
    
    def generate_prior_states(self, initial_action = None):

        D = obj_array(self.num_factors)

        if initial_action is not None:
            for f_idx, f_dim in enumerate(self.num_states):

                if f_idx == self.focal_belief_idx or f_idx in self.neighbour_belief_idx: #the first N+1 hidden state factors are variations of the identity matrix based on stubborness
                    
                    D[f_idx] = np.ones(f_dim)/f_dim
                    #D[f_idx] = np.random.uniform(0,1,f_dim)
                
                elif f_idx == self.h_control_idx:
                    
                    D[f_idx] = onehot(initial_action[f_idx],f_dim)

                elif f_idx == self.who_idx: 
                    D[f_idx] = onehot(initial_action[f_idx],f_dim)
        else:
            D = obj_array_uniform(self.num_states)
        self.D = D
        return D


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
            #self.belief2tweet_mapping = np.random.uniform(low = 1, high = 9, size=(self.num_H , self.idea_levels))
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