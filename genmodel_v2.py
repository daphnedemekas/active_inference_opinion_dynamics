
# %% run
import numpy as np
import itertools
from utils import obj_array, softmax, insert_multiple


def generate_likelihood2(h_idea_mapping, h_control_mapping, precisions, num_neighbours = 2, num_cohesion_levels = 6):

    num_H = h_idea_mapping.shape[0]

    idea_levels = h_idea_mapping.shape[1] # number of levels to the truth/falsity belief

    num_obs = [num_H] + (num_neighbours) * [num_H+1] + [num_cohesion_levels] # list that contains the dimensionalities of each sensory   
    num_modalities = len(num_obs) # total number of observation modalities
    num_states = (1+ num_neighbours) * [idea_levels] + [num_H] + [num_neighbours]


    focal_h_idx = 0 # index of the observation modality corresponding to my observing my own hashtags
    neighbour_h_idx = [(focal_h_idx + n + 1) for n in range(num_neighbours)] # indices of the observation modalities corresponding to observation of my neighbours' hashtags
    cohesion_idx = neighbour_h_idx[-1] + 1 # index of the observation modality corresponding to seeing the level of discrepancy you have with your neighbours 

    # create variables to name the hidden state factor indices
    focal_belief_idx = 0
    neighbour_belief_idx = [(focal_belief_idx + n + 1) for n in range(num_neighbours)]
    h_control_idx = neighbour_belief_idx[-1] + 1
    who_idx = h_control_idx + 1

    #initialize the A matrix 
    A = obj_array(num_modalities)
    for o_idx, o_dim in enumerate(num_obs):
        modality_shape = [o_dim] + num_states # num_obs[m] rows and as many lagging dimensions as there are hidden states, with each lagging dimension == num_states[i]
        A[o_idx] = np.zeros(modality_shape)

    #iterate over the sensory modalities 
    for o_idx, o_dim in enumerate(num_obs):
        #begin with modality 1
        if o_idx == focal_h_idx: #this happens for 0 == 0 -- we are in the 0th sensory modality (my beliefs about H) 

            h_obs = num_H #we only observe the two hashtags, not the null observation
            dimensions = [h_obs] + [idea_levels] + [idea_levels]*num_neighbours + [num_H] + [num_neighbours] #this is the shape of the modality
            fill_indices = [0,h_control_idx+1] # these are the dimensions we need to fill for this modality
            fill_dimensions = np.delete(dimensions, fill_indices) 
            
            for item in itertools.product(*[list(range(d)) for d in fill_dimensions]):
                slice_ = list(item)
                A_indices = insert_multiple(slice_, fill_indices, [slice(0,h_obs), slice(0,num_H)]) #here we insert the correct values for the fill indices for this slice                    
                A[o_idx][tuple(A_indices)] = h_control_mapping

        if o_idx in neighbour_h_idx: # now we're considering one of the observation modalities corresponding to seeing my neighbour's tweets
            
            h_obs_with_null = num_H + 1 # we now have the null observation 
            dimensions = [h_obs_with_null] + [idea_levels] + [idea_levels]*num_neighbours + [num_H] + [num_neighbours]

            for truth_level in range(num_states[focal_belief_idx]): # the precision of the mapping is dependent on the truth value of the hidden state 
                h_idea_mapping_scaled = np.copy(h_idea_mapping)
                h_idea_mapping_scaled[:,truth_level] = softmax(precisions[truth_level] * h_idea_mapping[:,truth_level])
                    # augment the h->idea mapping matrix with a row of 0s on top to account for the the null observation (this is for the case when you are sampling the agent whose modality we're considering)
                h_idea_with_null = np.zeros((o_dim,num_states[o_idx-1]))
                h_idea_with_null[1:,:] = np.copy(h_idea_mapping_scaled)

                # create the null matrix for the case when you're _not_ sampling the neighbour whose modality we're considering
                null_matrix = np.zeros((o_dim,num_states[o_idx-1]))
                null_matrix[0,:] = np.ones(num_states[o_idx-1]) # every observation is the 'null' observation because we're sampling someone else
                
                for neighbour_i in range(num_states[who_idx]): 
                    fill_indices = [0, focal_belief_idx + 1, neighbour_h_idx[neighbour_i]+1, who_idx+1]
                    fill_dimensions = np.delete(dimensions,tuple(fill_indices)) #the fill dimensions are those which we need to iterate over
                    
                    for item in itertools.product(*[list(range(d)) for d in fill_dimensions]):
                        slice_ = list(item) #now we specify the values of the indices for this specific combination of truth value, neighbour and who_idx
                        A_indices = insert_multiple(slice_, fill_indices, [slice(0,h_obs_with_null), slice(0,idea_levels), truth_level,neighbour_i ]) #here we insert the correct values for the fill indices for this slice                    
                        
                        if (o_idx - 1) == neighbour_i: # this is the case when the observation modality in question `o_idx` corresponds to the modality of the neighbour we're sampling `who_i`               
                            A[o_idx][tuple(A_indices)] = h_idea_with_null
                        else:
                            A[o_idx][tuple(A_indices)] = null_matrix

        if o_idx == cohesion_idx: #this is the final modality for observing the cohesion of the group's beliefs with respect to my own beliefs
            dimensions = [num_cohesion_levels] + [idea_levels] + [idea_levels]*num_neighbours + [num_H] + [num_neighbours]
            belief_combos = np.array(list(itertools.product([0, 1], repeat=num_neighbours+1))) #all combinations of low, medium high cohesiveness beliefs
            pop_sum = belief_combos[:,1:].sum(axis=1)
            cohesion_levels = np.zeros( (2, 3, belief_combos.shape[0] ) )
            thresholds = np.linspace(0,1,num_neighbours+2)
            
            for truth_level in range(num_states[focal_belief_idx]): #map the possible combinations to different levels of cohesion 
                for t_idx in range(len(thresholds[0:-1])):
                    idx = np.logical_and( (belief_combos[:,0]==truth_level), np.logical_and((pop_sum >= thresholds[t_idx]*num_neighbours), (pop_sum <= thresholds[t_idx+1]*num_neighbours) ) )
                    cohesion_levels[truth_level,t_idx,idx] = 1.0     
            
            fill_dimensions = np.delete(dimensions,0) #only need to fill the first dimension

            for item in itertools.product(*[list(range(d)) for d in fill_dimensions]):
                A_indices = list(item)
                A_indices.insert(0,slice(0,num_cohesion_levels))
                combo = A_indices[focal_belief_idx+1:h_control_idx+1] #the current combination of beliefs 
                combo_id = np.where(np.all(belief_combos==combo, axis=1)) #find the index of this combination in belief_combos
                A[o_idx][tuple(A_indices)] = cohesion_levels[:,:,combo_id].flatten()
    return A, num_states
