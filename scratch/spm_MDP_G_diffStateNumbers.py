# %%
import numpy as np
import time
import matplotlib.pyplot as plt
from Model.genmodel import GenerativeModel
from Model.pymdp.utils import obj_array_random, reduce_a_matrix
from Model.pymdp.maths import spm_MDP_G, spm_MDP_G_optim

# list of different numbers of neighbours
num_neighbours_list = list(range(2,11))

computation_times = np.zeros( (4,len(num_neighbours_list)) ) # row 1 is for old version, row 2 is for reduced A matrix version

precisions = np.array([1.0,10.0])

num_idea_levels = 2
num_H = 2

h_idea_mapping = np.eye(num_H)

# loop over different numbers of neighbours and compute spm_MDP_G in four different ways
for idx, nneighbs in enumerate(num_neighbours_list):

    genmodel = GenerativeModel(precisions, nneighbs, num_H, num_idea_levels, h_idea_mapping = h_idea_mapping)

    A_n = genmodel.A[1] 

    qs = obj_array_random(genmodel.num_states)

    t0 = time.time()
    G = spm_MDP_G(A_n, qs)
    t1 = time.time()

    computation_times[0,idx] = t1-t0

    t0 = time.time()
    G = spm_MDP_G_optim(A_n, qs)
    t1 = time.time()

    computation_times[1,idx] = t1-t0

    A_reduced, reduced_factor_idx  = reduce_a_matrix(A_n)
    t0 = time.time()
    G = spm_MDP_G(A_reduced, qs[reduced_factor_idx])
    t1 = time.time()

    computation_times[2,idx] = t1-t0

    t0 = time.time()
    G = spm_MDP_G_optim(A_reduced, qs[reduced_factor_idx])
    t1 = time.time()

    computation_times[3,idx] = t1-t0

plt.plot(computation_times[0,:],label='OG SPM_MDP_G, no reduction')
plt.plot(computation_times[1,:],label='Optimized SPM_MDP_G, no reduction')

plt.plot(computation_times[2,:],label='OG SPM_MDP_G, with reduction')
plt.plot(computation_times[3,:],label='Optimized SPM_MDP_G, with reduction')