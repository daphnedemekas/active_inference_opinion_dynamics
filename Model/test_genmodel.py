#%%
from genmodel_self_esteem import GenerativeModel
import numpy as np
num_neighbours = 3 

num_H = 2
num_idea_levels = 2
ecb_precisions = np.ones((num_neighbours, num_idea_levels))*8

genmodel = GenerativeModel(ecb_precisions, num_neighbours, num_H, num_idea_levels, reduce_A=True)

genmodel.initialize_A()
A = genmodel.generate_likelihood()


# %%


#TODO: copy the active inference demo -- and print stuff to see if this is working
#make it nnice
#then do sequencing
