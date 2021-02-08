
import numpy as np
import itertools
from utils import obj_array, softmax, insert_multiple

class GenerativeModel(object):


    #imagine we have different types of agents such as agent_types = ['influencer','shy',etc..]
    #this would then feed into different generator functions instead of having random distributions as below

    def __init__(
        self,
        h_control_mapping, 
        precisions, 
        num_neighbours, 
        stubborness_levels,
        h_idea_mapping = None,
        h_true_weights = None, 
        h_false_weights = None, 
        policy_true_weights = None,
        policy_false_weights = None

    ):

        self.h_idea_mapping = h_idea_mapping

        if self.h_idea_mapping is None:
            self.h_idea_mapping = self.create_idea_mapping()

        self.h_idea_mapping = h_idea_mapping
        self.h_control_mapping = h_control_mapping
        self.precisions = precisions
        self.num_neighbours = num_neighbours
        self.num_cohesion_levels = 2 * (self.num_neighbours+1)
        self.stubborness_levels = stubborness_levels

        self.h_true_weights = h_true_weights
        self.h_false_weights = h_false_weights 
        self.policy_true_weights = policy_true_weights 
        self.policy_false_weights = policy_false_weights

        self.num_H = self.h_idea_mapping.shape[0] 
        self.idea_levels = self.h_idea_mapping.shape[1] # number of levels to the truth/falsity belief
        self.num_obs = [self.num_H] + (self.num_neighbours) * [self.num_H+1] + [self.num_cohesion_levels] # list that contains the dimensionalities of each sensory   
        self.num_modalities = len(self.num_obs) # total number of observation modalities
        self.num_states = (1+ self.num_neighbours) * [self.idea_levels] + [self.num_H] + [self.num_neighbours]
        self.num_factors = len(self.num_states) # total number of hidden state factors
        self.num_policies = self.num_H

        self.focal_h_idx = 0 # index of the observation modality corresponding to my observing my own hashtags
        self.neighbour_h_idx = [(self.focal_h_idx + n + 1) for n in range(self.num_neighbours)] # indices of the observation modalities corresponding to observation of my neighbours' hashtags
        self.cohesion_idx = self.neighbour_h_idx[-1] + 1 # index of the observation modality corresponding to seeing the level of discrepancy you have with your neighbours 

        # create variables to name the hidden state factor indices
        self.focal_belief_idx = 0
        self.neighbour_belief_idx = [(self.focal_belief_idx + n + 1) for n in range(self.num_neighbours)]
        self.h_control_idx = self.neighbour_belief_idx[-1] + 1
        self.who_idx = self.h_control_idx + 1


    def generate_likelihood(self):

        if self.h_idea_mapping is None:
            h_idea_mapping = self.create_idea_mapping()

        #initialize the A matrix 
        A = obj_array(self.num_modalities)
        for o_idx, o_dim in enumerate(self.num_obs):
            modality_shape = [o_dim] + self.num_states # num_obs[m] rows and as many lagging dimensions as there are hidden states, with each lagging dimension == num_states[i]
            A[o_idx] = np.zeros(modality_shape)

        #iterate over the sensory modalities 
        for o_idx, o_dim in enumerate(self.num_obs):
            #begin with modality 1
            if o_idx == self.focal_h_idx: #this happens for 0 == 0 -- we are in the 0th sensory modality (my beliefs about H) 

                h_obs = self.num_H #we only observe the two hashtags, not the null observation
                dimensions = [h_obs] + [self.idea_levels] + [self.idea_levels]*self.num_neighbours + [self.num_H] + [self.num_neighbours] #this is the shape of the modality
                fill_indices = [0,self.h_control_idx+1] # these are the dimensions we need to fill for this modality
                fill_dimensions = np.delete(dimensions, fill_indices) 
                
                for item in itertools.product(*[list(range(d)) for d in fill_dimensions]):
                    slice_ = list(item)
                    A_indices = insert_multiple(slice_, fill_indices, [slice(0,h_obs), slice(0,self.num_H)]) #here we insert the correct values for the fill indices for this slice                    
                    A[o_idx][tuple(A_indices)] = self.h_control_mapping

            if o_idx in self.neighbour_h_idx: # now we're considering one of the observation modalities corresponding to seeing my neighbour's tweets
                
                h_obs_with_null = self.num_H + 1 # we now have the null observation 
                dimensions = [h_obs_with_null] + [self.idea_levels] + [self.idea_levels]*self.num_neighbours + [self.num_H] + [self.num_neighbours]

                for truth_level in range(self.num_states[self.focal_belief_idx]): # the precision of the mapping is dependent on the truth value of the hidden state 
                    h_idea_mapping_scaled = np.copy(self.h_idea_mapping)
                    h_idea_mapping_scaled[:,truth_level] = softmax(self.precisions[truth_level] * self.h_idea_mapping[:,truth_level])
                        # augment the h->idea mapping matrix with a row of 0s on top to account for the the null observation (this is for the case when you are sampling the agent whose modality we're considering)
                    h_idea_with_null = np.zeros((o_dim,self.num_states[o_idx-1]))
                    h_idea_with_null[1:,:] = np.copy(h_idea_mapping_scaled)

                    # create the null matrix for the case when you're _not_ sampling the neighbour whose modality we're considering
                    null_matrix = np.zeros((o_dim,self.num_states[o_idx-1]))
                    null_matrix[0,:] = np.ones(self.num_states[o_idx-1]) # every observation is the 'null' observation because we're sampling someone else
                    
                    for neighbour_i in range(self.num_states[self.who_idx]): 
                        fill_indices = [0, self.focal_belief_idx + 1, self.neighbour_h_idx[neighbour_i]+1, self.who_idx+1]
                        fill_dimensions = np.delete(dimensions,tuple(fill_indices)) #the fill dimensions are those which we need to iterate over
                        
                        for item in itertools.product(*[list(range(d)) for d in fill_dimensions]):
                            slice_ = list(item) #now we specify the values of the indices for this specific combination of truth value, neighbour and who_idx
                            A_indices = insert_multiple(slice_, fill_indices, [slice(0,h_obs_with_null), truth_level, slice(0,self.idea_levels), neighbour_i ]) #here we insert the correct values for the fill indices for this slice                    
                            
                            if (o_idx - 1) == neighbour_i: # this is the case when the observation modality in question `o_idx` corresponds to the modality of the neighbour we're sampling `who_i`               
                                A[o_idx][tuple(A_indices)] = h_idea_with_null
                            else:
                                A[o_idx][tuple(A_indices)] = null_matrix

            if o_idx == self.cohesion_idx: #this is the final modality for observing the cohesion of the group's beliefs with respect to my own beliefs
                dimensions = [self.num_cohesion_levels] + [self.idea_levels] + [self.idea_levels]*self.num_neighbours + [self.num_H] + [self.num_neighbours]
                belief_combos = np.array(list(itertools.product([0, 1], repeat=self.num_neighbours+1))) #all combinations of low, medium high cohesiveness beliefs
                pop_sum = belief_combos[:,1:].sum(axis=1)
                cohesion_levels = np.zeros( (2, self.num_neighbours+1, belief_combos.shape[0] ) )
                thresholds = np.linspace(0,1,self.num_neighbours+2)
                
                for truth_level in range(self.num_states[self.focal_belief_idx]): #map the possible combinations to different levels of cohesion 
                    for t_idx in range(len(thresholds[0:-1])):
                        idx = np.logical_and( (belief_combos[:,0]==truth_level), np.logical_and((pop_sum >= thresholds[t_idx]*self.num_neighbours), (pop_sum <= thresholds[t_idx+1]*self.num_neighbours) ) )
                        cohesion_levels[truth_level,t_idx,idx] = 1.0 

                fill_dimensions = np.delete(dimensions,0) #only need to fill the first dimension

                for item in itertools.product(*[list(range(d)) for d in fill_dimensions]):
                    A_indices = list(item)
                    A_indices.insert(0,slice(0,self.num_cohesion_levels+1))
                    combo = A_indices[(self.focal_belief_idx+1):(self.h_control_idx+1)] #the current combination of beliefs 
                    combo_id = np.where(np.all(belief_combos==combo, axis=1)) #find the index of this combination in belief_combos
                    A[o_idx][tuple(A_indices)] = cohesion_levels[:,:,combo_id].flatten()
        return A, self.num_states


    def generate_transition(self):

        transition_identity = np.eye(self.idea_levels, self.num_H)
        B = obj_array(self.num_factors)

        for f_idx, f_dim in enumerate(self.num_states):

            if f_idx == self.focal_belief_idx or f_idx in self.neighbour_belief_idx: #the first N+1 hidden state factors are variations of the identity matrix based on stubborness
                
                transition_identity = np.eye(f_dim, f_dim)
                B[f_idx] = softmax(transition_identity * self.stubborness_levels[f_idx])
            
            if f_idx == self.h_control_idx: #for the hashtag control state we have rows of ones corresponding to the next state

                b_matrix_shape = 2*[f_dim] + [self.num_H]
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
    
    #these are prior preferences over observations so
    #what i would like to see (cohesion, other beliefs)
    def generate_prior_preferences(self, preference_shape = "parabola", cohesion_exp = 2.0, cohesion_temp = 5.0):

        C = obj_array(self.num_modalities)

        for o_idx, o_dim in enumerate(self.num_obs): 
            
            C[o_idx] = np.zeros(o_dim)

            if o_idx == self.cohesion_idx:

                if preference_shape == "one_hot":
                    C[o_idx][0] = 1.0
                    C[o_idx][-1] = 1.0
                    C[o_idx] = softmax(cohesion_temp*C[o_idx])

                if preference_shape == "parabola":
                    C[o_idx] = np.linspace(-1.0, 1.0, o_dim) ** cohesion_exp

                if preference_shape == "linear":
                    C[o_idx] = np.absolute(np.linspace(-1.0, 1.0, o_dim))       
                
        return C
    
    #  generate the policy probability vector E as a mapping from the hidden state factors
    #This is the same function that I wrote to create the h_idea_mapping 
    # it only works if we have 2 idea levels (truth/false) it would need to be adapted to go beyond that 
    #because of the 1-

    #if we want to choose on purpose weights for which our agent will tweet some tweets more than others 
    #based on it being true or false, we can do that in policy_true_weights and policy_false_weights 

    #otherwise we generate them randomly, and if they are random then they are mutually exclusive, but we can 
    #change this as well if we want to 

    def generate_policy_mapping(self):
        policy_mapping = np.zeros((self.num_policies, self.idea_levels))
        if self.policy_true_weights is None:
            self.policy_true_weights = np.random.uniform(low = 1, high = 9, size=self.num_policies)
        if self.policy_false_weights is None:
            self.policy_false_weights = np.ones(self.num_policies) - self.policy_true_weights
        policy_mapping[:,0] = self.policy_true_weights / self.policy_true_weights.sum()    
        policy_mapping[:,1] = self.policy_false_weights / self.policy_false_weights.sum()

        return policy_mapping

    def create_idea_mapping(self):
        h_idea_mapping = np.zeros((self.num_H, self.idea_levels))
        if self.h_true_weights is None:
            self.h_true_weights = np.random.uniform(low = 1, high = 9, size=self.num_H)
        if self.h_false_weights is None:
            self.h_false_weights = np.ones(self.num_H) - self.h_true_weights

        h_idea_mapping[:,0] = self.h_true_weights / self.h_true_weights.sum()    
        h_idea_mapping[:,1] = self.h_false_weights / self.h_false_weights.sum()

        return h_idea_mapping
