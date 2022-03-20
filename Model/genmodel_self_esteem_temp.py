import numpy as np 
import itertools
import time
from .pymdp.utils import obj_array, obj_array_uniform, insert_multiple, onehot, reduce_a_matrix
from .pymdp.maths import *
from .pymdp.learning import *
import warnings 
from .generative_model import GenerativeModelSuper
from numpy import array 

"""TODO: need to add the ability to choose not to tweet anything
- extend the action space to "say nothing"
probability of performing the do nothing action in B matrix -- if you are less certain about your beliefs, you are less likely to tweet 
- can also include some parameter for each agent which is just inherently how likely they are to tweet (extroverted)
"power dynamics" - people who are confident will be the "loudest" - will want to investigate whether those "extroverted" ideas dominate
"""

class GenerativeModel(GenerativeModelSuper):

    def __init__(
        self,
        num_neighbours, 

        num_H,
        num_idea_levels,

        initial_action = None,
        
        h_idea_mapping = None,

        belief2tweet_mapping = None,
        E_lr = None,

        
        env_determinism = 9,
        belief_determinism = None,

        reduce_A = False, 

        esteem_parameters = [1.5,0.2,-1.5],
        C_params = np.array([0,0,0])



    ):

        super().__init__(num_neighbours=num_neighbours, num_H=num_H,num_idea_levels=num_idea_levels,initial_action=initial_action, h_idea_mapping=h_idea_mapping,belief2tweet_mapping=belief2tweet_mapping ,E_lr=E_lr ,env_determinism=env_determinism,belief_determinism=belief_determinism,reduce_A=reduce_A)

        #observation: [what i tweeted, what my neighbours tweeted, who i observed, my self esteem, neighbours' esteem] 
        self.num_esteem_levels = 3

        self.num_obs = [self.num_H]+ (self.num_neighbours) * [self.num_H+1] +  [self.num_neighbours]  + [self.num_esteem_levels] * (1+self.num_neighbours)# list that contains the dimensionalities of each observation modality 
        self.num_modalities = len(self.num_obs) # total number of observation modalities

        self.focal_esteem_idx = self.who_obs_idx + 1
        self.neighbour_esteem_idx = [self.focal_esteem_idx + 1+ n for n in range(self.num_neighbours)]
        self.num_states = (1+ self.num_neighbours) * [self.num_idea_levels] + [self.num_H] + [self.num_neighbours] #list that contains the dimensionality of each state factor 
        self.num_factors = len(self.num_states) # total number of hidden state factors
        num_controls = [1] * self.num_factors
        num_controls[self.h_control_idx] = self.num_H

        num_controls[self.who_idx] = self.num_neighbours
        self.num_controls = num_controls
        self.policies = self.generate_policies()
        self.B = self.generate_transition()

        self.E = np.ones(len(self.policies))

        self.policy_mapping = self.generate_policy_mapping()

        self.reduce_A = reduce_A

        A_slice = np.zeros((self.num_esteem_levels,self.num_idea_levels,self.num_idea_levels))
        for i in range(A_slice.shape[0]):
            A_slice[i] = softmax(np.eye(self.num_H)*esteem_parameters[i])


        self.initialize_A()
        self.generate_likelihood(A_slice)
        self.C = self.generate_prior_preferences(C_params)

    def get_idea_mapping(self, neighbour_idx: int, o_dim: int) -> array:
        """ This fills in the h_idea_with_null array with the h_idea_mapping """
        h_idea_with_null = np.zeros((o_dim,self.num_states[neighbour_idx-1]))
        h_idea_with_null[1:,:] = np.copy(self.h_idea_mapping) 
        
        return h_idea_with_null  

    def generate_likelihood(self,A_slice):

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


            elif o_idx in self.neighbour_h_idx: # now we're considering one of the observation modalities corresponding to seeing my neighbour's tweets
                null_matrix = self.get_null_matrix(o_dim, o_idx) # create the null matrix to tile throughout the appropriate dimensions (this matrix is for the case when you're _not_ sampling the neighbour whose modality we're considering)

                for idea_level in range(self.num_states[self.focal_belief_idx]): # iterate over the number of idea levels
                    h_idea_with_null = self.get_idea_mapping(o_idx, o_dim)
                    idx_vec_o = [slice(0, o_dim)] + idx_vec_s.copy() #this is a list of the dimensionalities of the observation
                    idx_vec_o[self.focal_belief_idx+1] = slice(idea_level,idea_level+1,None)

                    for neighbour_i in range(self.num_states[self.who_idx]): #iterate over the possible observed neighbours from state factor who_idx
                        # create a list of which dimensions you need to reshape along for the broadcasted tiling
                        broadcast_dims_specific = self.get_broadcast_dims(broadcast_dims, neighbour_i)
                        print(broadcast_dims_specific)

                        idx_vec_o[self.who_idx+1] = slice(neighbour_i,neighbour_i+1,None)
                        reshape_vector = [o_dim] + [1] * self.num_factors

                        print("broadcast dims")
                        print(broadcast_dims_specific)

                        if (o_idx - 1) == neighbour_i: # this is the case when the observation modality in question `o_idx` corresponds to the modality of the neighbour we're sampling `who_i`               
                            A[o_idx] = self.fill_A(A[o_idx], neighbour_i, h_idea_with_null, broadcast_dims_specific, idx_vec_o, reshape_vector)

                        else: # this is the case when the observation modality in question `o_idx` corresponds to a modality _other than_ the neighbour we're sampling `who_i` 
                            reshape_vector[neighbour_i+2] = self.num_states[neighbour_i+1]
                            null_matrix_reshaped = np.reshape(null_matrix,reshape_vector)
                            A[o_idx][tuple(idx_vec_o)] = np.tile(null_matrix_reshaped, tuple(broadcast_dims_specific))

            elif o_idx == self.who_obs_idx:   #this is the observation modality corresponding to which neighbour the agent is samplign 
                A[o_idx] = self.po_who_given_s(A[o_idx], o_idx)


            elif o_idx == self.focal_esteem_idx:

                idx_vec_o = [slice(0, o_dim)] + idx_vec_s.copy()
                #iterate over hashtag state and who_idx state and copy over the template matrix for each neighbour

                for i in range(self.num_neighbours):

                    reshape_vec = np.ones(len(self.num_states), dtype = int)
                    reshape_vec[0] = self.num_esteem_levels #first dimension is the levels of the esteem observation modality 
                    reshape_vec[1] = self.num_idea_levels #second dimension is the focal agent's own belief 
                    reshape_vec[i+2] = self.num_idea_levels #third dimensions is the beliefs of the sampled neighbour  
                    
                    broadcast_dims = np.ones(len(self.num_states), dtype = int) 
                    other_neighbours = list(set(list(range(2,self.num_neighbours+2))) - set([i+2])) #we broadcast over all of the remaining neighbours 
                    broadcast_dims[np.array(other_neighbours)] = int(self.num_idea_levels)

                    broadcast_dims[-1] = int(self.num_H)

                    idx_vec_o[-1] = i

                    A[o_idx][tuple(idx_vec_o)] = np.tile(A_slice.reshape(reshape_vec), tuple(broadcast_dims))

            elif o_idx in self.neighbour_esteem_idx:
                n_idx = o_idx - self.who_obs_idx -2

                idx_vec_o = [slice(0, o_dim)] + idx_vec_s.copy()
                #iterate over hashtag state and who_idx state and copy over the template matrix for each neighbour

                for i in range(self.num_neighbours):
                    if i == n_idx:
                        continue
                    
                    reshape_vec = np.ones(len(self.num_states), dtype = int)
                    reshape_vec[0] = self.num_esteem_levels #first dimension is the levels of the esteem observation modality
                    reshape_vec[n_idx+2] = self.num_idea_levels #second dimension is the esteem-observed neighbours' beliefs 
                    reshape_vec[i+2] = self.num_idea_levels #third dimension is the sampled neighbours' belief 

                    broadcast_dims = np.ones(len(self.num_states), dtype = int) 
                    other_neighbours = list(set([1] + list(range(2,self.num_neighbours+2))) - set([i+2, n_idx+2])) 
                    broadcast_dims[np.array(other_neighbours)] = int(self.num_idea_levels)

                    broadcast_dims[-1] = int(self.num_H)

                    idx_vec_o[-1] = i

                    A[o_idx][tuple(idx_vec_o)] = np.tile(A_slice.reshape(reshape_vec), tuple(broadcast_dims))

        
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

        B = obj_array(self.num_factors)

        for f_idx, f_dim in enumerate(self.num_states): #iterate over the state factors

            if f_idx == self.focal_belief_idx: #the state factor corresponding to what the agent is tweeting 
                B[f_idx] = self.fill_B_states(matrix = np.eye(f_dim, f_dim), precision = self.env_determinism)
                
            if f_idx in self.neighbour_belief_idx: #the first N+1 hidden state factors are variations of the identity matrix based on belief volatiliy    
                B[f_idx] = self.fill_B_states(matrix = np.eye(f_dim, f_dim), precision = self.belief_determinism[f_idx-1])

            if f_idx == self.h_control_idx: #for the hashtag control state we have rows of ones corresponding to the next state
                B[f_idx] = self.fill_B_control_states(num_states = f_dim, num_actions = self.num_H)

            if f_idx == self.who_idx: #same as above for the who control state
                B[f_idx] = self.fill_B_control_states(num_states = self.num_neighbours, num_actions = self.num_neighbours) 
        return B


    def generate_prior_preferences(self, C_params):
        C = obj_array(self.num_modalities)

        for o_idx, o_dim in enumerate(self.num_obs): 
            
            if o_idx == self.focal_esteem_idx: #the agent should prefer reward to rejection
                C[o_idx] = np.array(C_params)
            else :
                C[o_idx] = np.zeros(o_dim)

                
        return C
    
    def generate_prior_states(self, initial_action = None):
        # Currently prior states are completely unbiased 

        D = obj_array(self.num_factors)

        if initial_action is not None:
            for f_idx, f_dim in enumerate(self.num_states):

                if f_idx == self.focal_belief_idx or f_idx in self.neighbour_belief_idx: #the first N+1 hidden state factors are variations of the identity matrix based on stubborness
                    r = np.random.randint(0,2)
                    if r == 0:
                        D[f_idx] = np.random.uniform(0.1,0.4,f_dim) #this is if we want the prior preferences to be uniform 
                    elif r == 1:
                        D[f_idx] = np.random.uniform(0.6,0.9,f_dim) #this is if we want the prior preferences to be uniform 
                    #D[f_idx] = np.ones(f_dim)/f_dim

                elif f_idx == self.h_control_idx:
                    
                    D[f_idx] = onehot(initial_action[f_idx],f_dim)

                elif f_idx == self.who_idx: 
                    D[f_idx] = onehot(initial_action[f_idx],f_dim)
        else:
            D = obj_array_uniform(self.num_states)
        self.D = D
        return D
