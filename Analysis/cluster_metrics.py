import numpy as np
# %% function to access the real parameters from the simulation
def davies_bouldin(all_qs): # a low DB index represents low inter cluster and high intra cluster similarity 
    believers = np.where(all_qs[-1,1,:] > 0.5)[0]
    nonbelievers = np.where(all_qs[-1,1,:] < 0.5)[0] 
    centroid1 = np.mean(all_qs[-1,:,believers],axis = 0)
    centroid2 = np.mean(all_qs[-1,:,nonbelievers], axis = 0)
    
    sigma1 = np.mean([KL_div(e[0],e[1], centroid1[0],centroid1[1]) for e in all_qs[-1,:,believers]])
    sigma2 = np.mean([KL_div(e[0],e[1], centroid2[0],centroid2[1])  for e in all_qs[-1,:,nonbelievers]])
    if np.isnan(KL_div(centroid1[0],centroid1[1], centroid2[0], centroid2[1])):
        return 1
    else:
        db = 0.5 * ((sigma1 + sigma2) / KL_div(centroid1[0],centroid1[1], centroid2[0], centroid2[1]))
    return db

def kld(arr1, arr2):
    return arr1*np.log(arr1/arr2)
def KL_div(array1_0, array1_1, array2_0, array2_1):
    return array1_0 * np.log(array1_0 / array2_0) + array1_1 * np.log(array1_1 / array2_1)

def cluster_kl(all_qs):
    cluster_metrics = np.zeros((all_qs.shape[0],1))
    believers = np.where(all_qs[-1,1,:] > 0.5)[0]
    nonbelievers = np.where(all_qs[-1,1,:] < 0.5)[0]
    for t in range(all_qs.shape[0]):
        believer_beliefs = all_qs[t,1,believers]
        non_believer_beliefs = all_qs[t,0,nonbelievers]
        if np.sum(believer_beliefs) == 0 or np.sum(non_believer_beliefs) == 0:
            cluster_ratio = 0.5
        else:
            cluster_metric_b = np.mean(np.ones(believer_beliefs.shape[0]) * np.log(np.ones(believer_beliefs.shape[0]) / believer_beliefs))         
            cluster_metric_nb = np.mean(np.ones(non_believer_beliefs.shape[0])* np.log(np.ones(non_believer_beliefs.shape[0]) / non_believer_beliefs) )        
            cluster_metrics[t] = (cluster_metric_b + cluster_metric_nb) / 2
    return cluster_metrics

def conclusion_thresholds(all_qs):
    N = all_qs.shape[2]
    agent_thresholds = np.zeros(N)
    for agent in range(N):
        last_belief = all_qs[-1,1,agent] > 0.5
        for t in reversed(range(all_qs.shape[0])):
            if (all_qs[t,1,agent] > 0.5) != last_belief:
                agent_thresholds[agent] = t
                break
    return agent_thresholds


def cluster_ratios(all_qs):
    cluster_ratios = np.zeros((all_qs.shape[0],1))
    believers = np.where(all_qs[-1,1,:] > 0.5)[0]
    nonbelievers = np.where(all_qs[-1,1,:] < 0.5)[0]
    for t in range(all_qs.shape[0]):
        if np.sum(all_qs[t,0,believers]) == 0 or np.sum(all_qs[t,1,nonbelievers]) == 0:
            cluster_ratio = 0
        else:
            cluster_ratio = np.sum(all_qs[t,0,believers]) / np.sum(all_qs[t,1,nonbelievers])
            cluster_ratio = cluster_ratio if cluster_ratio < 1 else 1/cluster_ratio
        cluster_ratios[t] = cluster_ratio
    return cluster_ratios


#average beliefs in idea over both clusters e
#ratio of sum of believers and sum of nonbelievers 


#def ratio between in and out group samples in different time chunks 

def sampling_ratio(all_qs, agent_view_per_timestep, cluster_sorted_indices, agent_neighbours):
    believers = np.where(all_qs[-1,1,:] > 0.5)[0]
    nonbelievers = np.where(all_qs[-1,1,:] < 0.5)[0]
    N = agent_view_per_timestep.shape[0]
    sample_ratio_in_group = 0    
    #on the last time step, 
    for i, cluster_agent_1 in enumerate(believers):
        viewees = agent_view_per_timestep[i]
        #for v in 
    for i, viewer in enumerate(range(N)):
        for j, viewee in enumerate(range(N)):
            if i != j:
                counts = np.count_nonzero(agent_view_per_timestep[viewer][:] == viewee)
                view_heatmap[i, j] = counts
    print(view_heatmap)
    


#def silhouette_coeff(all_qs):


# %%
