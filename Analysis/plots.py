import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import seaborn as sns


def KL_div(array1_0, array1_1, array2_0, array2_1):
    return array1_0 * np.log(array1_0 / array2_0) + array1_1 * np.log(array1_1 / array2_1)

def KL_similarity_matrices(belief_hist):
    T = belief_hist.shape[0]
    N = belief_hist.shape[2]
    KLD_intra_beliefs = np.zeros((N,N,T))
    for a in range(N):
        for n in range(N):
            KLD_intra_beliefs[a,n,:] = KL_div(belief_hist[:,0,a], belief_hist[:,1,a], belief_hist[:,0,n], belief_hist[:,1,n])

    KLD_plot_images = []
    believers = np.where(belief_hist[-1,0,:] > 0.5)
    nonbelievers = np.where(belief_hist[-1,0,:] < 0.5)
    cluster_sorted_indices = [i for i in believers[0]]
    for j in nonbelievers[0]:
        cluster_sorted_indices.append(j)
    
    color_map = plt.cm.get_cmap('gray').reversed()

    for t in range(T)[2:-1:2]:

        single_slice = KLD_intra_beliefs[:,:,t]
        sorted_slice = single_slice[cluster_sorted_indices,:][:,cluster_sorted_indices]
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



def tweet_proportions(all_tweets):
    T = all_tweets.shape[0]
    N = all_tweets.shape[1]
    tweet_proportions = []

    for t in range(T)[2:-2:2]:
        print(all_tweets[t].reshape(N,1))
        sns.heatmap(all_tweets[t].reshape(N,1), cmap = "gray", xticklabels = ["hashtag1", "hashtag2"], vmin = 0, vmax = 1)
        plt.title("Tweet proportions per agent")
        plt.savefig('TP, t = ' + str(t) + '.png')

        tweet_proportions.append('TP, t = ' + str(t) + '.png')
        plt.clf()
    
    return tweet_proportions


def make_gif(filenames, gif_name):

    with imageio.get_writer('gifs/' + str(gif_name), mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        for filename in set(filenames):
            os.remove(filename)