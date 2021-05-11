
import numpy as np
from Model.genmodel import GenerativeModel
from Model.agent import Agent
import networkx as nx
from Model.pymdp.utils import obj_array, index_list_to_onehots, sample, to_numpy
from Model.pymdp.maths import spm_dot, dot_likelihood, softmax
import seaborn as sns
from matplotlib import pyplot as plt
import time
import os 
import imageio
from sklearn.cluster import spectral_clustering


def get_belief_metrics(all_beliefs, agents, agent_neighbours,T):
    N = len(agents)
    #print([all_beliefs[:,a] for a in range(N)])
    #print()
    agent_own_beliefs_per_timestep = np.array([[belief[0] for belief in all_beliefs[:,a]] for a in range(N)]) # (N,T,2)
        
    all_neighbour_perceptions = np.zeros((N,N-1,T,2))
    
    #KL_divergences between agents' beliefs
    KLD_intra_beliefs = np.zeros((N,N,T))

    for a in range(len(agents)):
        for n in range(len(agents)):
            KLD_intra_beliefs[a,n,:] = KL_div(agent_own_beliefs_per_timestep[a][:,0], agent_own_beliefs_per_timestep[a][:,1], agent_own_beliefs_per_timestep[n][:,0], agent_own_beliefs_per_timestep[n][:,1])

    agent_hashtag_beliefs_per_timestep = [[belief[-2] for belief in all_beliefs[:,a]] for a in range(N)] # (N,T,2)
    agent_who_idx_beliefs_per_timestep = [[belief[-1] for belief in all_beliefs[:,a]] for a in range(N)] # (N,T,2)

    #proportion of agents believing in idea 1 at the final timestep 
    agent_belief_proportions = np.zeros((N,2))

    for a in range(N):
        idea1 = sum(agent_own_beliefs_per_timestep[a][:,0])
        idea2 = len(agent_own_beliefs_per_timestep[a]) - idea1
        agent_belief_proportions[a] = [idea1/T, idea2/T]

    return agent_own_beliefs_per_timestep, KLD_intra_beliefs, agent_belief_proportions, agent_hashtag_beliefs_per_timestep, agent_who_idx_beliefs_per_timestep


def get_action_metrics(all_actions, agent_neighbours, N,T):
    all_actions = np.array(all_actions) # shape is T, N, 2
    agent_tweets_per_timestep = np.array([[action[0] for action in all_actions[:,a]] for a in range(N)]) # (N,T,2)

    agent_view_per_timestep = np.array([[agent_neighbours[a][int(action[1])] for action in all_actions[:,a][1:]] for a in range(N)]) # (N,T,1)


    tweet_cohesion_matrix = np.zeros((T,N,N))
    
    for a in range(N):
        for n in range(N):
            tweet_cohesion_matrix[:,a,n] = agent_tweets_per_timestep[a] - agent_tweets_per_timestep[n]
    agent_tweet_proportions = np.zeros((T,N,2))

    agent_sample_proportions = np.zeros((N,N))

    for t in range(T)[1:]:
        for a in range(N):
            hashtag1 = sum(agent_tweets_per_timestep[a,:t])
            hashtag2 = len(agent_tweets_per_timestep[a,:t]) - hashtag1
            agent_tweet_proportions[t,a] = [hashtag1/t, hashtag2/t]

    return agent_tweet_proportions, tweet_cohesion_matrix, agent_view_per_timestep

def plot_KLD_similarity_matrix(KLD_intra_beliefs, agent_own_beliefs):
    KLD_plot_images = []
    cluster1_idx = np.where(agent_own_beliefs[:,-1,0] > 0.5)
    cluster2_idx = np.where(agent_own_beliefs[:,-1,0] < 0.5)
    cluster_sorted_indices = [i for i in cluster1_idx[0]]
    for j in cluster2_idx[0]:
        cluster_sorted_indices.append(j)
    color_map = plt.cm.get_cmap('gray').reversed()

    for t in range(T)[2:-1:2]:

        single_slice = KLD_intra_beliefs[:,:,t]
        sorted_slice = single_slice[cluster_sorted_indices,:][:,cluster_sorted_indices]
        plt.imshow(sorted_slice, cmap = color_map)
        plt.title("Belief similarity matrix")
        #plt.savefig('KLD, t = ' + str(t) + '.png')
        KLD_plot_images.append('KLD, t = ' + str(t) + '.png')
        plt.clf()
    return KLD_plot_images, cluster_sorted_indices


def plot_tweet_similarity_matrix(tweet_cohesion_matrix, cluster_sorted_indices):
    tweet_sim_images = []
    color_map = plt.cm.get_cmap('gray').reversed()
    for t in range(T)[2:-1:2]:
        single_slice = tweet_cohesion_matrix[t,:,:]
        sorted_slice = single_slice[cluster_sorted_indices,:][:,cluster_sorted_indices]
        plt.imshow(sorted_slice, cmap = color_map)
        plt.title("Tweet similarity matrix")
        plt.savefig('TSM, t = ' + str(t) + '.png')
        tweet_sim_images.append('TSM, t = ' + str(t) + '.png')
    return tweet_sim_images


def plot_samples(agent_view_per_timestep, cluster_sorted_indices, agent_neighbours):
    #print(cluster_sorted_indices + [8, 9])
    view_heatmap = np.zeros((8,10))
    for i, viewer in enumerate(cluster_sorted_indices):
        for j, viewee in enumerate(cluster_sorted_indices + [8, 9]):
            if i != j:
                counts = np.count_nonzero(agent_view_per_timestep[viewer][250:] == viewee)
                view_heatmap[i, j] = counts
    plt.imshow(view_heatmap, cmap = 'gray')
    #plt.colorbar()
    plt.title("Agent samples over time")
    plt.xlabel("Time")
    plt.ylabel("Agent samples")
    plt.savefig("Agent Samples")

def plot_proportions(tweets, beliefs):
    tweet_proportions = []
    sampled_neighbours = []
    for t in range(T)[2:-2:2]:
        sns.heatmap(tweets[t], cmap = "gray", xticklabels = ["hashtag1", "hashtag2"], vmin = 0, vmax = 1)
        plt.title("Tweet proportions per agent")
        plt.savefig('TP, t = ' + str(t) + '.png')

        tweet_proportions.append('TP, t = ' + str(t) + '.png')
        plt.clf()
    
    return tweet_proportions

def plot_beliefs_over_time(all_actions, agent_own_beliefs, p,T):
    belief_plot_images = []

    clusters = []
    def color_dict(value):
            if value < 0.5:
                return "darkblue"
            else:
                return "coral"
    #time_steps = [2,4,6,10,14,16,20,24,26,28,32,35,40,42,46,48,49,50,52,54,56,60,64,66,70,74,76,78,82,85,90,92,96,98,99,100]
    for t in range(T)[2:-1:2]:
        for a in range(N):
            data = agent_own_beliefs[a][:t,0]
            plt.plot(data, color = color_dict(data[-1]), label = "beliefs in idea 1")
            plt.ylim(0,1)
        plt.title("Connectedness of graph: " +str(p))
        plt.ylabel("Belief that idea is True")
        plt.xlabel("Time")
        plt.ylim([-0.1,1.1])
        plt.savefig('beliefs, t = ' + str(t) + '.png')
        belief_plot_images.append('beliefs, t = ' + str(t) + '.png')
        #plt.show()
    return belief_plot_images

def KL_div(array1_0, array1_1, array2_0, array2_1):
    return array1_0 * np.log(array1_0 / array2_0) + array1_1 * np.log(array1_1 / array2_1)



results = np.load('results/medianodedata.npz', allow_pickle = True) 

all_actions = results['arr_0']
all_beliefs = results['arr_1']
all_observations = results['arr_2']
agent_neighbours = {0: [1, 2, 3, 4, 5, 6, 7, 8, 9], 1: [0, 2, 3, 4, 5, 6, 7, 8, 9], 2: [0, 1, 3, 4, 5, 6, 7, 8, 9], 3: [0, 1, 2, 4, 5, 6, 7, 8, 9], 4: [0, 1, 2, 3, 5, 6, 7, 8, 9], 5: [0, 1, 2, 3, 4, 6, 7, 8, 9], 6: [0, 1, 2, 3, 4, 5, 7, 8, 9], 7: [0, 1, 2, 3, 4, 5, 6, 8, 9], 8: [0, 1, 2, 3, 4, 5, 6, 7, 9], 9: [0, 1, 2, 3, 4, 5, 6, 7, 8]}

T = 300

agents = [1,2,3,4,5,6,7,8]
N = len(agents)
agent_beliefs, KLD_intra_beliefs, belief_proportions, _, _ = get_belief_metrics(all_beliefs, agents, agent_neighbours,T)
tweet_proportions, tweet_cohesion_matrix, agent_view_per_timestep = get_action_metrics(all_actions, agent_neighbours, N, T)
#make plots 



belief_plot_images = plot_beliefs_over_time(all_actions, agent_beliefs, 1, T)
plt.clf()
with imageio.get_writer('belief_plot.gif', mode='I') as writer:
    for filename in belief_plot_images:
        image = imageio.imread(filename)
        writer.append_data(image)
    for filename in set(belief_plot_images):
        os.remove(filename)

KLD_images, cluster_idx = plot_KLD_similarity_matrix(KLD_intra_beliefs, agent_beliefs)
plot_samples(agent_view_per_timestep, cluster_idx, agent_neighbours)

with imageio.get_writer('KLD_plot.gif', mode='I') as writer:
    for filename in KLD_images:
        image = imageio.imread(filename)
        writer.append_data(image)
for filename in set(KLD_images):
    os.remove(filename)

tweet_sim_images = plot_tweet_similarity_matrix(tweet_cohesion_matrix, cluster_idx)
plt.clf()
tweet_proportions = plot_proportions(tweet_proportions, belief_proportions)
plt.clf() 
plot_samples(agent_view_per_timestep)
plt.clf()



with imageio.get_writer('TSM_plot.gif', mode='I') as writer:
    for filename in tweet_sim_images:
        image = imageio.imread(filename)
        writer.append_data(image)

for filename in set(tweet_sim_images):
    os.remove(filename)

with imageio.get_writer('tweet_proportions.gif', mode='I') as writer:
    for filename in tweet_proportions:
        image = imageio.imread(filename)
        writer.append_data(image)

for filename in set(tweet_proportions):
    os.remove(filename)
