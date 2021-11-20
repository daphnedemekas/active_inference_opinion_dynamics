# %% Imports
from Model.pymdp.maths import spm_MDP_G, spm_MDP_G_optim
import numpy as np
from matplotlib import pyplot as plt
# %% Imports

# num_states = [3,4,5]

# A = np.random.rand(2,10)

# x = np.empty(1,dtype=object)

# x[0] = np.random.rand(10)
# #x[1] = np.random.rand(4)
# #x[2] = np.random.rand(4)
# print(A)
# q = spm_MDP_G(A,x)

# q_optim, cost1, cost2, cost3 = spm_MDP_G_optim(A,x)

# print(q)
# print(q_optim)

num_factors_list = range(1,10)
num_iter = 100
num_obs = [2]

cross_costs = np.zeros( (len(num_factors_list), num_iter, 2) )
einsum_costs = np.zeros_like(cross_costs)
dot_prod_costs = np.zeros_like(cross_costs)

for ii, nf in enumerate(num_factors_list):

    num_states = nf * [3]

    A = np.random.rand(*(num_obs + num_states))

    x = np.empty(nf,dtype=object)
    for f in range(nf):
        x[f] = np.random.rand(2)

    for iter_i in range(num_iter):

        _, cross_costs[ii,iter_i,0], einsum_costs[ii,iter_i,0], dot_prod_costs[ii,iter_i,0] = spm_MDP_G_optim(A,x)
        _, cross_costs[ii,iter_i,1], einsum_costs[ii,iter_i,1], dot_prod_costs[ii,iter_i,1] = spm_MDP_G(A,x)

# %%
