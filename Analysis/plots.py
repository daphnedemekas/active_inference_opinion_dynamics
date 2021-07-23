import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import seaborn as sns

def plot_beliefs_over_time(belief_hist):
    believers = np.where(belief_hist[-1,1,:] > 0.5)[0]
    nonbelievers = np.where(belief_hist[-1,1,:] < 0.5)[0]
    for non_believe_idx in nonbelievers:
        if non_believe_idx == nonbelievers[-1]:
            plt.plot(belief_hist[:,0,non_believe_idx],c='b',lw=2.5, label='Believe in Idea 1')
        else:
            plt.plot(belief_hist[:,0,non_believe_idx],c='b',lw=2.5)

    for believe_idx in believers:
        if believe_idx == believers[-1]:
            plt.plot(belief_hist[:,0,believe_idx],c='orange',lw=2.5, label='Believe in Idea 2')
        else:
            plt.plot(belief_hist[:,0,believe_idx],c='orange',lw=2.5)
    plt.legend(fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel('Time',fontsize=10)
    plt.ylim(0,1)
    plt.ylabel('Strength of belief',fontsize=10)

    
def KL_div(array1_0, array1_1, array2_0, array2_1):
    return array1_0 * np.log(array1_0 / array2_0) + array1_1 * np.log(array1_1 / array2_1)

def KL_div_alt(p, q):
    return p * np.log(p / q) + (1. - p) * np.log((1.-p) / (1.-q))

def JS_div(array1_0, array1_1, array2_0, array2_1):

    m_0 = (array1_0 + array2_0) / 2.
    m_1 = 1.0 - m_0
    return 0.5 * (KL_div(array1_0, array1_1, m_0, m_1) + KL_div(array2_0, array2_1, m_0, m_1))

def get_JS(belief_hist):
    T = belief_hist.shape[0]
    N = belief_hist.shape[2]
    JS_intra_beliefs = np.zeros((N,N,T))

    for a in range(N):
        for n in range(N):
            JS_intra_beliefs[a,n,:] = JS_div(belief_hist[:,0,a], belief_hist[:,1,a], belief_hist[:,0,n], belief_hist[:,1,n])
    return JS_intra_beliefs
    

def get_KLDs(belief_hist):
    T = belief_hist.shape[0]
    N = belief_hist.shape[2]
    KLD_intra_beliefs = np.zeros((N,N,T))

    for a in range(N):
        for n in range(N):
            KLD_intra_beliefs[a,n,:] = KL_div(belief_hist[:,0,a], belief_hist[:,1,a], belief_hist[:,0,n], belief_hist[:,1,n])
    return KLD_intra_beliefs


def belief_similarity_matrix(t, KLD_intra_beliefs, cluster_sorted_indices):
    single_slice = KLD_intra_beliefs[:,:,t]
    sorted_slice = single_slice[cluster_sorted_indices,:][:,cluster_sorted_indices]
    return sorted_slice 

def get_cluster_sorted_indices(all_beliefs):
    believers = np.where(all_beliefs[-1,0,:] > 0.5)
    nonbelievers = np.where(all_beliefs[-1,0,:] < 0.5)
    cluster_sorted_indices = [i for i in believers[0]]
    for j in nonbelievers[0]:
        cluster_sorted_indices.append(j)
    return cluster_sorted_indices

def KL_similarity_matrices(belief_hist):
    T = belief_hist.shape[0]

    KLD_intra_beliefs = get_KLDs(belief_hist)
    cluster_sorted_indices = get_cluster_sorted_indices(belief_hist)
    KLD_plot_images = []
    
    color_map = plt.cm.get_cmap('gray').reversed()

    for t in range(T)[2:-1:2]:
        sorted_slice = belief_similarity_matrix(t, KLD_intra_beliefs, cluster_sorted_indices)
        plt.imshow(sorted_slice, cmap = color_map)
        plt.title("Belief similarity matrix")
        plt.savefig('KLD, t = ' + str(t) + '.png')
        KLD_plot_images.append('KLD, t = ' + str(t) + '.png')
        plt.clf()
    
    return KLD_plot_images, cluster_sorted_indices

    
    
def tweet_similarity_matrices(all_tweets, cluster_sorted_indices):
    T = all_tweets.shape[0]
    N = all_tweets.shape[1]
    tweet_cohesion_matrix = np.zeros((T,N,N))
    
    for a in range(N):
        for n in range(N):
            tweet_cohesion_matrix[:,a,n] = all_tweets[:,a] - all_tweets[:,n]

    tweet_sim_images = []
    color_map = plt.cm.get_cmap('gray').reversed()
    for t in range(T)[2:-1:2]:
        single_slice = tweet_cohesion_matrix[t,:,:]
        sorted_slice = single_slice[cluster_sorted_indices,:][:,cluster_sorted_indices]
        plt.imshow(sorted_slice, cmap = color_map)
        plt.title("Tweet similarity matrix")
        plt.savefig('TSM, t = ' + str(t) + '.png')
        tweet_sim_images.append('TSM, t = ' + str(t) + '.png')
        plt.clf()
    return tweet_sim_images

# def tweet_proportions(all_tweets):
#     T = all_tweets.shape[0]
#     N = all_tweets.shape[1]
#     tweet_proportions = []

#     for t in range(T)[2:-2:2]:
#         sns.heatmap(all_tweets[t,:].reshape(N,1), cmap = "gray", xticklabels = ["hashtag1", "hashtag2"], vmin = 0, vmax = 1)
#         plt.title("Tweet proportions per agent")
#         plt.savefig('TP, t = ' + str(t) + '.png')

#         tweet_proportions.append('TP, t = ' + str(t) + '.png')
#         plt.clf()
    
#     return tweet_proportions

def isolate_metric_by(all_parameters, metric, parameter, idx):
    metric_list = []
    for i, e in enumerate(parameter):
        indices = np.where(all_parameters[:,idx] == e)[0]
        _params = all_parameters[indices]
        metrics = np.nanmean(metric[indices])
        metric_list.append(metrics)
    return metric_list

def plot_bifurcations(all_parameters, ecb_precisions, b_precisions, lr, variance, metric, metric_name):
    fig, axs = plt.subplots(2, 2, figsize=(12,8))
    belief_extremities1 = isolate_metric_by(all_parameters,metric, ecb_precisions, 2)
    belief_extremities2 = isolate_metric_by(all_parameters,metric, b_precisions, 3)
    belief_extremities3 = isolate_metric_by(all_parameters,metric, lr, -1)
    belief_extremities4 = isolate_metric_by(all_parameters,metric, variance, -2)

    axs[0, 0].plot(ecb_precisions, belief_extremities1)
    axs[0, 0].set_xlabel("ECB Precision")
    axs[0, 0].set_ylabel(metric_name)

    axs[1, 0].plot(b_precisions, belief_extremities2)
    axs[1, 0].set_xlabel("Belief Determinism")
    axs[1, 0].set_ylabel(metric_name)

    axs[0, 1].plot(lr, belief_extremities3)
    axs[0, 1].set_xlabel("Learning Rate")
    axs[0, 1].set_ylabel(metric_name)

    axs[1, 1].plot(variance, belief_extremities4)
    axs[1, 1].set_xlabel("Variance")
    axs[1, 1].set_ylabel(metric_name)

def plot_param_histograms(conditional_params):
    fig, axs = plt.subplots(3, 2, figsize=(10,10))
    axs[0, 0].hist(conditional_params[:,0])
    axs[0,0].set_title("Number of Agents")
    axs[0, 1].hist(conditional_params[:,1])
    axs[0,1].set_title("Network Connectedness")

    axs[1, 0].hist(conditional_params[:,2])
    axs[1,0].set_title("ECB Precision")

    axs[1, 1].hist(conditional_params[:,3])
    axs[1,1].set_title("Belief Precision")

    axs[2, 0].hist(conditional_params[:,5])
    axs[2,0].set_title("Variance")

    axs[2, 1].hist(conditional_params[:,6])
    axs[2,1].set_title("Learning Rate")

def scatterplot_metrics(params):
    fig, axs = plt.subplots(3, 2, figsize=(10,10))
    axs[0, 0].scatter(params.insider_outsider_ratios[:,-1], params.avg_belief_extremity)
    axs[0,0].set_xlabel("Outsider to Insider ratios")
    axs[0,0].set_ylabel("Average Belief Extremity")

    axs[0, 1].scatter(params.cluster_kls, params.db_indices)
    axs[0,1].set_xlabel("Cluster KL Divergence")
    axs[0,1].set_ylabel("Davies Bouldin Index")
    
    axs[1, 0].scatter(params.egds, params.cluster_kls)
    axs[1, 0].set_xlabel("Eigenvalue Decay Slopes")
    axs[1, 0].set_ylabel("Cluster KL Divergence")
    
    axs[1, 1].scatter(params.avg_belief_extremity, params.egds)
    axs[1, 1].set_xlabel("Average Belief Extremity")
    axs[1, 1].set_ylabel("Eigenvalue Decay Slopes")
    
    axs[2, 0].scatter(params.avg_belief_extremity, params.cluster_kls)
    axs[2, 0].set_xlabel("Average Belief Extremity")
    axs[2, 0].set_ylabel("Cluster KL Divergence")
    
    axs[2, 1].scatter(params.avg_belief_extremity, params.db_indices)
    axs[2, 1].set_xlabel("Average Belief Extremity")
    axs[2, 1].set_ylabel("Davies Bouldin Index")
    

def get_2d_histogram(param1, param2, conditional_parameters, conditional_metric, param1_index, param2_index):
    hist = np.zeros((len(param1), len(param2)))
    for i, e in enumerate(param1):
        indices = np.where(conditional_parameters[:,param1_index] == e)[0]
        _params = conditional_parameters[indices]
        metrics = conditional_metric[indices]
        for j, l in enumerate(param2):
            p2_indices = np.where(_params[:,param2_index]==l)[0]
            p2_params = _params[p2_indices]
            p2_metrics = metrics[p2_indices]
            avg_metric = np.nanmean(p2_metrics[np.isfinite(p2_metrics)])
            hist[i,j] = avg_metric
    return hist

def plot_2d_histogram(axs, hist, x_label, y_label, param1, param2):

    im2 = axs.imshow(hist)
    axs.set_xlabel(x_label)
    axs.set_ylabel(y_label)
    axs.set_xticks(np.arange(0,len(param2),1))
    axs.set_yticks(np.arange(0,len(param1),1))
    axs.set_xticklabels(param2)
    axs.set_yticklabels(param1)
    return im2
def make_gif(filenames, gif_name):

    with imageio.get_writer('gifs/' + str(gif_name), mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        for filename in set(filenames):
            os.remove(filename)