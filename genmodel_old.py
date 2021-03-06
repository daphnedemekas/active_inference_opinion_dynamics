import numpy as np
from scipy import stats
from utils import obj_array, softmax
import itertools

def generate_likelihood(h_idea_mapping, h_control_mapping, precisions, num_neighbours = 1, num_cohesion_levels = 6):
    """
    Docstring @TODO - First attempt at constructing the A matrices 
    
    Parameters:
    ___________
    `h_idea_mapping` - 
    `h_control_mapping` - 
    `num_neighbours` - 
    `num_cohesion_levels` - 

    Returns:
    ___________
    `A`

    """

    """
    Observation modalities:
        - 1 observation modality corresponding for focal agent's own hashtag observations
        - `num_neighbours` hashtag observation modalities corresponding to focal agent's observations of other agent's hashtags (+1 level for `null` outcome)
        # - 1 modality for sensing the overall discrepancy bewteen your own beliefs and those of your neighbours, also whether it concerns the truth/falsity idea
    """

    num_H = h_idea_mapping.shape[0]

    # num_obs = [num_H] + (num_neighbours) * [num_H+1] + [2] + [num_outcast_levels] # list that contains the dimensionalities of each sensory modality
    num_obs = [num_H] + (num_neighbours) * [num_H+1] + [num_cohesion_levels] # list that contains the dimensionalities of each sensory modality

    num_modalities = len(num_obs) # total number of observation modalities

    focal_h_idx = 0 # index of the observation modality corresponding to my observing my own hashtags
    neighbour_h_idx = [(focal_h_idx + n + 1) for n in range(num_neighbours)] # indices of the observation modalities corresponding to observation of my neighbours' hashtags
    cohesion_idx = neighbour_h_idx[-1] + 1 # index of the observation modality corresponding to seeing the level of discrepancy you have with your neighbours 

    """
    Hidden state factors:
    - (1 + num_neighbours) hidden state factors for beliefs about truth/falsity of Idea, 1 per neighbour, + 1 for agent's own beliefs
    - 1 hidden state factor (also controllable) indicating which hashtag the focal agent is currently tweeting about
    - 1 hidden state factor (also controllable) indicating which neighbour the focal agent is currently sampling
    """
    idea_levels = h_idea_mapping.shape[1] # number of levels to the truth/falsity belief
    num_states = (1+ num_neighbours) * [idea_levels] + [num_H] + [num_neighbours]
    num_factors = len(num_states) # total number of hidden state factors

    # create variables to name the hidden state factor indices
    focal_belief_idx = 0
    neighbour_belief_idx = [(focal_belief_idx + n + 1) for n in range(num_neighbours)]
    h_control_idx = neighbour_belief_idx[-1] + 1
    who_idx = h_control_idx + 1

    # Initialise with a bunch of arrays filled with 0s, with the correct dimensions

    A = obj_array(num_modalities)

    for o_idx, o_dim in enumerate(num_obs):
        modality_shape = [o_dim] + num_states # num_obs[m] rows and as many lagging dimensions as there are hidden states, with each lagging dimension == num_states[i]
        A[o_idx] = np.zeros(modality_shape)

    # Now we want to go through and set the appropriate slices of the A matrix

    idx_vec_s = [slice(num_states[f]) for f in range(num_factors)]
    broadcast_dims = [1] + num_states # this is template broadcast dimension list

    for o_idx, o_dim in enumerate(num_obs):

        """
        Modality 1 - observation of my own hashtag output (kinda meaningless actually...it's just a 'copy' of the hashtag-control state)
        """

        if o_idx == focal_h_idx: # this is case when we're considering the focal agent's observation modality of its own hashtags (twitter content)

            idx_vec_o = [slice(0, o_dim)] + idx_vec_s.copy()

            # hashtag content control factor (What I'm tweeting)
            h_control_levels = num_states[h_control_idx] 
            broadcast_dims_specific = broadcast_dims.copy()
            broadcast_dims_specific[h_control_idx+1] = 1 # we won't need to broadcast along the dimension of the matrix that we already have enough levels for - namely, the one correspondign to the number of columns of the hidden-factor-specific A matrix

            reshape_vector = [o_dim] + [1] * num_factors # we reshape the matrix trivially into 1 dimension along all the hidden state factors that it doesn't have to do with
            reshape_vector[h_control_idx+1] = h_control_levels # for the hidden state factor it does have to do with, make sure the reshape length along that dimension is the number of columsn in the matrix
            h_control_mapping_reshaped = np.reshape(h_control_mapping, reshape_vector) # now we reshape it into a [o_dim x 1 x 1 x ... x num_states x 1 x 1 x....] matrix 

            A[o_idx][tuple(idx_vec_o)] = np.tile(h_control_mapping_reshaped,tuple(broadcast_dims_specific)) 
        
        """
        Modalities 2 thru N_{N} - observation of my neighbours' hashtag output (used to make inferences about the beliefs of my neighbours)
        """

        if o_idx in neighbour_h_idx: # now we're considering one of the observation modalities corresponding to seeing my neighbour's tweets

            for truth_level in range(num_states[focal_belief_idx]): # the precision of the mapping is dependent on the truth value of the hidden state 
                                                                    # this reflects the idea that 
                h_idea_mapping_scaled = np.copy(h_idea_mapping)
                h_idea_mapping_scaled[:,truth_level] = softmax(precisions[o_idx, truth_level] * h_idea_mapping[:,truth_level])

                idx_vec_o = [slice(0, o_dim)] + idx_vec_s.copy()
                idx_vec_o[focal_belief_idx+1] = slice(truth_level,truth_level+1,None)

                # augment the h->idea mapping matrix with a row of 0s on top to account for the the null observation (this is for the case when you are sampling the agent whose modality we're considering)
                h_idea_with_null = np.zeros((o_dim,num_states[o_idx-1]))
                h_idea_with_null[1:,:] = np.copy(h_idea_mapping_scaled)

                # create the null matrix to tile throughout the appropriate dimensions (this matrix is for the case when you're _not_ sampling the neighbour whose modality we're considering)
                null_matrix = np.zeros((o_dim,num_states[o_idx-1]))
                null_matrix[0,:] = np.ones(num_states[o_idx-1]) # every observation is the 'null' observation because we're sampling someone else

                for neighbour_i in range(num_states[who_idx]):

                    # create a list of which dimensions you need to reshape along for the broadcasted tiling
                    broadcast_dims_specific = broadcast_dims.copy()
                    broadcast_dims_specific[focal_belief_idx+1] = 1
                    broadcast_dims_specific[who_idx+1] = 1
                    broadcast_dims_specific[neighbour_i+2] = 1

                    idx_vec_o[who_idx+1] = slice(neighbour_i,neighbour_i+1,None)

                    reshape_vector = [o_dim] + [1] * num_factors
                    reshape_vector[neighbour_i+2] = num_states[neighbour_i+1] # this sets the correspondong factor of the reshape vector (corresponding to neighbour_i's belief states) to the correct number

                    if (o_idx - 1) == neighbour_i: # this is the case when the observation modality in question `o_idx` corresponds to the modality of the neighbour we're sampling `who_i`               
                        h_idea_mapping_reshaped = np.reshape(h_idea_with_null,reshape_vector)
                        A[o_idx][tuple(idx_vec_o)] = np.tile(h_idea_mapping_reshaped, tuple(broadcast_dims_specific))   
                    else: # this is the case when the observation modality in question `o_idx` corresponds to a modality _other than_ the neighbour we're sampling `who_i` 
                        null_matrix_reshaped = np.reshape(null_matrix,reshape_vector)
                        A[o_idx][tuple(idx_vec_o)] = np.tile(null_matrix_reshaped, tuple(broadcast_dims_specific))
        
        """
        Last modality (N_{N} + 1) - observations of the 'outcast-i-ness' variable or 'cohesion' variable - how disparate are my tweet outputs from those of my local community
        """

        if o_idx == cohesion_idx:

            belief_combos = np.array(list(itertools.product([0, 1], repeat=num_neighbours+1)))

            pop_sum = belief_combos[:,1:].sum(axis=1)

            cohesion_levels = np.zeros( (2, 3, belief_combos.shape[0] ) )

            for truth_level in range(num_states[focal_belief_idx]):

                idx = np.logical_and( (belief_combos[:,0]==truth_level), (pop_sum < num_neighbours/3) )
                cohesion_levels[truth_level,0,idx] = 1.0 

                idx = np.logical_and( (belief_combos[:,0]==truth_level), np.logical_and( (pop_sum > num_neighbours/3), (pop_sum < 2*(num_neighbours/3)) ) )
                cohesion_levels[truth_level,1,idx] = 1.0

                idx = np.logical_and( (belief_combos[:,0]==truth_level), (pop_sum > 2*(num_neighbours/3)) )
                cohesion_levels[truth_level,2,idx] = 1.0
        
            # create a list of which dimensions you need to reshape along for the broadcasted tiling
            broadcast_dims_specific = broadcast_dims.copy()

            # we're mapping each conditional distribution per belief configuration, so the broadcast dimensions along the hidden state factors that correspond to my beliefs and all my neighbours' beliefs can be set to 1
            broadcast_dims_specific[focal_belief_idx+1] = 1
            for neighbour_i in range(num_neighbours):
                broadcast_dims_specific[neighbour_i+2] = 1

            reshape_vector = [o_dim] + [1] * num_factors
            idx_vec_o = [slice(0, o_dim)] + idx_vec_s.copy()

            for combo_id, combo in enumerate(belief_combos):

                # this filling out of the idx_vec_o is what determines the particular configuration we're considering
                idx_vec_o[focal_belief_idx+1] = slice(combo[0],combo[0]+1,None)

                for neighbour_i in range(num_neighbours):
                    idx_vec_o[neighbour_i+2] = slice(combo[neighbour_i+1],combo[neighbour_i+1]+1,None)
                
                # we reshape this 6 x 1 conditional distribution to have the appropriate number of lagging dimensions (although all trivially-1-dimensional)
                cohesion_outcomes_reshaped = np.reshape(cohesion_levels[:,:,combo_id].flatten(), reshape_vector) # now we reshape it into a [o_dim x 1 x 1 x ... x 1] matrix 
                A[o_idx][tuple(idx_vec_o)] = np.tile(cohesion_outcomes_reshaped, tuple(broadcast_dims_specific))  # now actually do the assignment
    
    return A, num_states

