#%%
from genmodel_self_esteem import GenerativeModel
import numpy as np
num_neighbours = 3 

num_H = 2
idea_levels = 2
ecb_precisions = np.ones((num_neighbours, idea_levels))*8

genmodel = GenerativeModel(ecb_precisions, num_neighbours, num_H, idea_levels, reduce_A=True)

genmodel.initialize_A()
genmodel.generate_likelihood()


# %%
