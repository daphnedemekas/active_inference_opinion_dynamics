import numpy as np
import itertools
import warnings 
from pymdp.maths import softmax
from pymdp.utils import obj_array

def insert_multiple(s, indices, items):
    for idx in range(len(items)):
        s.insert(indices[idx], items[idx])
    return s
    

def initialize_A(num_obs, num_modalities, num_states):
    """initialize the A matrix and fill the modalities with their correct shape"""
    A = obj_array(num_modalities)
    for o_idx, o_dim in enumerate(num_obs): 
        modality_shape = [o_dim] + num_states # num_obs[m] rows and as many lagging dimensions as there are hidden states, with each lagging dimension == num_states[i]
        A[o_idx] = np.zeros(modality_shape)
    return A

def get_null_matrix(o_dim, o_idx, num_states):
    null_matrix = np.zeros((o_dim,num_states[o_idx-1]))
    null_matrix[0,:] = np.ones(num_states[o_idx-1]) # create a matrix that corresponds to NOT sampling the current neighbour -- every observation is the 'null' observation because we're sampling someone else
    return null_matrix


def fill_slice(A_o, A_slice, irrelevant_dimensions, fill_indices, slice_indices):

    for item in itertools.product(*[list(range(d)) for d in irrelevant_dimensions]):
        slice_ = list(item)
        A_indices = insert_multiple(slice_, fill_indices, slice_indices) #here we insert the correct values for the fill indices for this slice                    
        A_o[tuple(A_indices)] = A_slice
    return A_o   

def po_h_given_s(A_0, num_states, h_control_idx, h_control_mapping):
    """ fill the 0th sensory modality which corresponds to the agent's observation of which hashtag it has tweeted
        which should map via h_control_mapping (default identity matrix) to which hashtag state the agent is in """
    num_observable_hashtags = h_control_mapping.shape[0]
    dimensions = [num_observable_hashtags] + num_states #this is the shape of the modality-specific A matrix A[o_idx] the first index is the dimension of the observation modality and then the rest is the full dimensionality of the states
    fill_indices = [0,h_control_idx+1] # these are indices of the dimensions we need to fill for this modality. we have to add 1 to h_control_idx because the first dimensions corresponds to the observation
    
    irrelevant_dimensions = np.delete(dimensions, fill_indices) #these are the lagging dimensions that don't matter for this mapping
    
    A_0 = fill_slice(A_0, h_control_mapping,irrelevant_dimensions, fill_indices, [slice(0,num_observable_hashtags), slice(0,num_observable_hashtags)])

    return A_0

def po_who_given_s(A_o, who_obs_idx, num_obs, num_states, who_idx):
    who_obs = num_obs[who_obs_idx]
    sampling_A = np.eye(who_obs)

    dimensions = [who_obs] + num_states #this is the shape of the modality-specific A matrix A[o_idx]
    fill_indices = [0,who_idx+1] # these are the indices of the dimensions we need to fill for this modality
    irrelevant_dimensions = np.delete(dimensions, fill_indices) 
    A_o = fill_slice(A_o, sampling_A, irrelevant_dimensions, fill_indices, [slice(0,who_obs), slice(0,num_states[who_idx])])
    return A_o

def scale_idea_mapping(h_idea_mapping, num_states, neighbour_idx, truth_level, ecb):
    """ Scales the hashtag to idea mapping based on corresponding beliefs (truth_level) """
    h_idea_mapping_scaled = np.copy(h_idea_mapping)
    h_idea_mapping_scaled[:,truth_level] = softmax(ecb * h_idea_mapping[:,truth_level])
    if h_idea_mapping_scaled[truth_level,truth_level] < h_idea_mapping[truth_level,truth_level]:
        warnings.warn('ECB precision scaling is not high enough!')
    
    # augment the h->idea mapping matrix with a row of 0s on top to account for the the null observation (this is for the case when you are sampling the agent whose modality we're considering)
    h_idea_scaled_with_null = np.zeros((neighbour_idx,num_states[neighbour_idx-1]))
    h_idea_scaled_with_null[1:,:] = np.copy(h_idea_mapping_scaled)

    h_idea_with_null = np.zeros((neighbour_idx,num_states[neighbour_idx-1]))
    h_idea_with_null[1:,:] = np.copy(h_idea_mapping)
    
    return h_idea_scaled_with_null, h_idea_with_null

def get_broadcast_dims(broadcast_dims, focal_belief_idx, who_idx, neighbour_i):               
    broadcast_dims[focal_belief_idx+1] = 1
    broadcast_dims[who_idx+1] = 1
    broadcast_dims[neighbour_i+2] = 1

    return broadcast_dims

def scale_A_by_ecb(A_o, neighbour_i, h_idea_maps, state_dim, broadcast_dims_specific, idx_vec_o, reshape_vector, truth_level):
    h_idea_scaled_with_null, h_idea_with_null = h_idea_maps[0], h_idea_maps[1]
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