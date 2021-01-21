# %% Imports
import numpy as np
from utils import obj_array, spm_dot, dot_likelihood
from genmodel import generate_likelihood
import itertools
# %% Initialize constants
"""
Constant parameters of a focal agent's generative model, e.g. its number of neighbours
"""

num_neighbours = 2 # number of neighbours whose tweets our focal agent can read
num_outcast_levels = 6 # 3 levels of observed discrepancy between focal agent and community, x 2 levels for the truth/falsity of the idea at hand

"""
Specify the mapping between hidden states (truth/falsity of Idea) and Hashtags - this matrix will be used to 'fill out' 
the appropriate high-dimensional slices of the modality-specific A matrices later on.

Note that how you set up this mapping determines:
(1) the number of hashtags (i.e. the dimensionality of the hashtag modality) that an agent can tweet 
(2) the focal agent's beliefs about the 'semantics' of the Idea--> Hashtag content mapping - how do Hashtags provide evidence for the Idea
"""

h_idea_mapping = np.array([[0.6, 0.4], 
                      [0.4, 0.6]])

true_false_precisions = [5.0, 5.0]

num_H = h_idea_mapping.shape[0]# add an extra observation level to include the `null` observation

h_control_mapping = np.array([[1, 0], 
                      [0, 1]])
                      
A = generate_likelihood(h_idea_mapping, h_control_mapping, true_false_precisions, num_neighbours = num_neighbours, num_outcast_levels = num_outcast_levels)
# %%
