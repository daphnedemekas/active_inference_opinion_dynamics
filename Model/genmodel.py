import numpy as np 
import itertools
import time
from .pymdp.utils import obj_array, obj_array_uniform, insert_multiple, softmax, onehot, reduce_a_matrix
from .pymdp.maths import spm_log
from .pymdp.learning import *
import warnings 
from .generative_model import GenerativeModelSuper


class GenerativeModel(GenerativeModelSuper):

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

        env_determinism = None,
        belief_determinism = None,

        reduce_A = False


    ):

        super().__init__(ecb_precisions, num_neighbours, num_H,idea_levels,initial_action, h_idea_mapping,belief2tweet_mapping ,E_lr ,env_determinism,belief_determinism,reduce_A)


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
                
        return C
    
    def generate_prior_states(self, initial_action = None):
        # Currently prior states are completely unbiased 

        D = obj_array(self.num_factors)

        if initial_action is not None:
            for f_idx, f_dim in enumerate(self.num_states):

                if f_idx == self.focal_belief_idx or f_idx in self.neighbour_belief_idx: #the first N+1 hidden state factors are variations of the identity matrix based on stubborness
                    
                    D[f_idx] = np.ones(f_dim)/f_dim
                    #D[f_idx] = np.random.uniform(0,1,f_dim) #this is if we want the prior preferences to be uniform 
                
                elif f_idx == self.h_control_idx:
                    
                    D[f_idx] = onehot(initial_action[f_idx],f_dim)

                elif f_idx == self.who_idx: 
                    D[f_idx] = onehot(initial_action[f_idx],f_dim)
        else:
            D = obj_array_uniform(self.num_states)
        self.D = D
        return D

