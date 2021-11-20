import numpy as np
try:
    from .plots import belief_similarity_matrix, get_KLDs, get_JS, get_cluster_sorted_indices
except:
    from plots import *
import networkx as nx

""" This file contains functions for metrics that can be used to evaluate large statistical samplees of agent behaviour """
# %% function to access the real parameters from the simulation
def davies_bouldin(all_qs): # a low DB index represents low inter cluster and high intra cluster similarity 
    believers = np.where(all_qs[-1,1,:] > 0.5)[0]
    nonbelievers = np.where(all_qs[-1,1,:] < 0.5)[0] 
    centroid1 = np.mean(all_qs[-1,:,believers],axis = 0)
    centroid2 = np.mean(all_qs[-1,:,nonbelievers], axis = 0)
    
    sigma1 = np.mean([KL_div(e[0],e[1], centroid1[0],centroid1[1]) for e in all_qs[-1,:,believers]])
    sigma2 = np.mean([KL_div(e[0],e[1], centroid2[0],centroid2[1])  for e in all_qs[-1,:,nonbelievers]])
    if np.isnan(KL_div(centroid1[0],centroid1[1], centroid2[0], centroid2[1])):
        return np.nan
    else:
        db = 0.5 * ((sigma1 + sigma2) / KL_div(centroid1[0],centroid1[1], centroid2[0], centroid2[1]))
    return db

def clustering_consensus(all_qs): 
    """ The ratio of agents who form clusters out of all agents"""
    cluster = 0
    consensus = 0
    for trial in range(all_qs.shape[0]):
        believers = np.where(all_qs[trial, -1,1,:] > 0.7)[0]
        nonbelievers = np.where(all_qs[trial, -1,1,:] < 0.7)[0] 
        if len(believers) == 0 or len(nonbelievers) == 0:
            consensus += 1
        else:
            cluster += 1
    return cluster / (cluster + consensus)

def resampling_rate(neighbour_samplings_trial):
    #neighbour_samplings_trial[50,12]
    num_agents = neighbour_samplings_trial.shape[-1]
    values = np.zeros(num_agents)
    for a in range(num_agents):
        values[a] = np.max([np.count_nonzero(neighbour_samplings_trial[:,a] == n) for n in range(num_agents)])
    return np.mean(values)

def average_belief_extremity(all_qs):
    change_domain = np.absolute(np.mean(all_qs[:,1,:],axis=0) - 0.5)*2
    return np.mean(change_domain)

def average_belief_difference(all_qs):
    highest_belief = np.max(all_qs[-1,1,:])
    lowest_belief = np.min(all_qs[-1,1,:])
    return np.abs(highest_belief - lowest_belief)

def kld(arr1, arr2):
    return arr1*np.log(arr1/arr2)

def KL_div(array1_0, array1_1, array2_0, array2_1):
    return array1_0 * np.log(array1_0 / array2_0) + array1_1 * np.log(array1_1 / array2_1)


def is_connected(adj_mat):
    return np.where(adj_mat == 1)

def average_belief_extremity(all_qs):
    change_domain = np.absolute(all_qs[-1,1,:] - 0.5)*2
    return np.mean(change_domain)

def time_to_cluster(all_qs):
    times = np.zeros(all_qs.shape[-1])
    for a in range(all_qs.shape[-1]):
        for t in range(all_qs.shape[0]):
            if all_qs[t,1,a] > 0.9 or all_qs[t,1,a] < 0.1:
                times[a] = t
                break
            if t == all_qs.shape[0] - 1:
                times[a] = np.nan
    return np.nanmean(times)


def belief_cluster_sizes(all_qs):
    cluster1 = np.zeros(30)
    cluster2 = np.zeros(30)
    for trial in range(30):
        all_beliefs_t = all_qs[trial,:,:,:] 
        cluster1[trial] = len(np.where(all_beliefs_t[-1,1,:] > 0.5)[0])
        cluster2[trial] = len(np.where(all_beliefs_t[-1,1,:] < 0.5)[0])
    return cluster1, cluster2
        



# %%
