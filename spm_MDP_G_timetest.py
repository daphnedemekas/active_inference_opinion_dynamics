import numpy as np
from Model.genmodel import GenerativeModel
from Model.pymdp.utils import obj_array_random, reduce_a_matrix
from Model.pymdp.maths import spm_MDP_G, spm_MDP_G_optim

# list of different numbers of neighbours
num_neighbours = 5

precisions = 5. * np.random.rand(2)

num_idea_levels = 2
num_H = 2

h_idea_mapping = np.eye(num_H)

genmodel = GenerativeModel(precisions, num_neighbours, num_H, num_idea_levels, h_idea_mapping = h_idea_mapping)

A_n = genmodel.A[1] 

qs = obj_array_random(genmodel.num_states)

reduced_A_n, reduced_factor_idx = reduce_a_matrix(A_n)
reduced_qs = qs[reduced_factor_idx]

def test_surprise_OG():

    return spm_MDP_G(A_n, qs)

def test_surprise_OG_optim_G():

    return spm_MDP_G_optim(A_n, qs)

def test_surprise_reduced():

    return spm_MDP_G(reduced_A_n, reduced_qs)

def test_surprise_reduced_optim_G():

    return spm_MDP_G_optim(reduced_A_n, reduced_qs)

if __name__ == '__main__':
    import timeit
    print(timeit.timeit("test_surprise_OG()", setup="from __main__ import test_surprise_OG", number = 1000))
    print(timeit.timeit("test_surprise_OG_optim_G()", setup="from __main__ import test_surprise_OG_optim_G", number = 1000))
    print(timeit.timeit("test_surprise_reduced()", setup="from __main__ import test_surprise_reduced", number = 1000))
    print(timeit.timeit("test_surprise_reduced_optim_G()", setup="from __main__ import test_surprise_reduced_optim_G", number = 1000))

   