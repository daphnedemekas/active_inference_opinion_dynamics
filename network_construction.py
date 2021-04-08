import numpy as np
from Model.genmodel import GenerativeModel
from Model.agent import Agent
from Model.network_tools import create_multiagents, clip_edges, connect_edgeless_nodes
import networkx as nx
from Model.pymdp.utils import obj_array, index_list_to_onehots, sample, to_numpy
from Model.pymdp.maths import spm_dot, dot_likelihood, softmax
import seaborn as sns
from matplotlib import pyplot as plt
import time
import os 
import imageio

def agent_loop(agent, observations = None, initial = False, initial_action = None):  
    qs = agent.infer_states(initial, tuple(observations))
    policy = agent.infer_policies()
    action = agent.sample_action()[-2:]
    who_i_looked_at = int(action[-1]+1)
    what_they_tweeted = observations[int(action[-1])+1]

    if initial == True:
        action = agent.action
    return action, qs

def multi_agent_loop(T, agents, agent_neighbours_local):
    observation_t = [[None] * agent.genmodel.num_modalities for agent in agents] 

    initial = True
    all_actions = obj_array((T,N,))
    all_beliefs = obj_array((T,N))
    all_observations = obj_array((T,N))

    for t in range(T):
        print(str(t) + "/" + str(T))
        actions = []
        for i in range(len(agents)):
            action, belief = agent_loop(agents[i], observation_t[i], initial)
            all_actions[t,i] = action
            all_beliefs[t,i] = belief
            actions.append(action)

        initial = False

        for idx, agent in enumerate(agents):
            for n in range(agent.genmodel.num_neighbours):
                observation_t[idx][n+1] = 0
            my_tweet = int(actions[idx][-2])
            observation_t[idx][0] = my_tweet
            observed_neighbour = int(actions[idx][-1])
            observed_agent = agent_neighbours_local[idx][observed_neighbour]
            observation_t[idx][observed_neighbour+1] = int(actions[observed_agent][-2]) + 1
            observation_t[idx][-1] = int(actions[idx][-1]) 
        all_observations[t,:] = observation_t

        #if timestep == 5 or timestep == 20 or timestep == 10 or timestep == 25 or timestep == 30 :
        #    make_plots(all_actions, all_beliefs,p,timestep)
    return all_actions, all_beliefs, all_observations



def plot_beliefs_over_time(all_actions, agent_own_beliefs, p,T):
    belief_plot_images = []
    def color_dict(value):
            if value < 0.5:
                return "darkblue"
            else:
                return "coral"
    #time_steps = [2,4,6,10,14,16,20,24,26,28,32,35,40,42,46,48,49,50,52,54,56,60,64,66,70,74,76,78,82,85,90,92,96,98,99,100]
    for t in range(T)[2:-1:5]:
        for a in range(N):
            data = agent_own_beliefs[a][:t,0]
            plt.plot(data, color = color_dict(data[-1]), label = "beliefs in idea 1")
            plt.ylim(0,1)
        plt.title("Connectedness of graph: " +str(p))
        plt.ylabel("Belief that idea is True")
        plt.xlabel("Time")
        plt.savefig('beliefs, t = ' + str(t) + '.png')
        belief_plot_images.append('beliefs, t = ' + str(t) + '.png')
        #plt.show()
    return belief_plot_images

def KL_div(array1_0, array1_1, array2_0, array2_1):
    return array1_0 * np.log(array1_0 / array2_0) + array1_1 * np.log(array1_1 / array2_1)

def inference_loop(G,N): #just goes until you get a graph that has the right connectedness
    try:
        G = nx.fast_gnp_random_graph(N,p)
        G, agents_dict, agents, agent_neighbours = create_multiagents(G, N)
        G = connect_edgeless_nodes(G)
        all_actions, all_beliefs, all_observations = multi_agent_loop(T, agents, agent_neighbours)
    except ValueError:
        all_actions, all_beliefs, all_observations, agents, agent_neighbours = inference_loop(G,N)
    
    return all_actions, all_beliefs, all_observations, agents, agent_neighbours


def get_belief_metrics(all_beliefs, agents, agent_neighbours,T):
    N = len(agents)
    agent_own_beliefs_per_timestep = np.array([[belief[0] for belief in all_beliefs[:,a]] for a in range(N)]) # (N,T,2)
        
    all_neighbour_perceptions = np.zeros((N,N-1,T,2))
    
    #KL divergences between agent's beliefs and their beliefs about neighbour's beliefs
    KLD_inter_beliefs = np.zeros((N,N-1,T))

    #KL_divergences between agents' beliefs
    KLD_intra_beliefs = np.zeros((N,N,T))

    #for a in range(len(agents)):
    #    agent_p = []
    #    for i, n in enumerate(agent_neighbours[a]):
    #        all_neighbour_perceptions[a,i] = np.array([belief[n] for belief in all_beliefs[:,a]])
    #        KLD_inter_beliefs[a,i] = KL_div(all_neighbour_perceptions[a,i][:,0], all_neighbour_perceptions[a,i][:,1], agent_own_beliefs_per_timestep[a][:,0] , agent_own_beliefs_per_timestep[a][:,1])

    for a in range(len(agents)):
        for n in range(len(agents)):
            KLD_intra_beliefs[a,n] = KL_div(agent_own_beliefs_per_timestep[a][:,0], agent_own_beliefs_per_timestep[a][:,1], agent_own_beliefs_per_timestep[n][:,0], agent_own_beliefs_per_timestep[n][:,1])

    agent_hashtag_beliefs_per_timestep = [[belief[-2] for belief in all_beliefs[:,a]] for a in range(N)] # (N,T,2)
    agent_who_idx_beliefs_per_timestep = [[belief[-1] for belief in all_beliefs[:,a]] for a in range(N)] # (N,T,2)

    #proportion of agents believing in idea 1 at the final timestep 
    agent_belief_proportions = np.zeros((N,2))

    for a in range(N):
        idea1 = sum(agent_own_beliefs_per_timestep[a][:,0])
        idea2 = len(agent_own_beliefs_per_timestep[a]) - idea1
        agent_belief_proportions[a] = [idea1/T, idea2/T]

    return agent_own_beliefs_per_timestep, KLD_inter_beliefs, KLD_intra_beliefs, agent_belief_proportions, agent_hashtag_beliefs_per_timestep, agent_who_idx_beliefs_per_timestep


def get_action_metrics(all_actions, N,T):
    all_actions = np.array(all_actions) # shape is T, N, 2
    agent_tweets_per_timestep = np.array([[action[0] for action in all_actions[:,a]] for a in range(N)]) # (N,T,2)
    agent_view_per_timestep = np.array([[action[1] for action in all_actions[:,a]] for a in range(N)]) # (N,T,1)


    tweet_cohesion_matrix = np.zeros((T,N,N))
    
    for a in range(N):
        for n in range(N):
            tweet_cohesion_matrix[:,a,n] = agent_tweets_per_timestep[a] - agent_tweets_per_timestep[n]
    agent_tweet_proportions = np.zeros((T,N,2))

    agent_sample_proportions = np.zeros((T,N,N))

    for t in range(T)[1:]:
        for a in range(N):
            hashtag1 = sum(agent_tweets_per_timestep[a,:t])
            hashtag2 = len(agent_tweets_per_timestep[a,:t]) - hashtag1
            agent_tweet_proportions[t,a] = [hashtag1/t, hashtag2/t]

            sampled_agent = int(agent_view_per_timestep[a,t]) #who did they sample 
            agent_sample_proportions[t,a,sampled_agent] = agent_sample_proportions[t-1,a,sampled_agent] + 1 
        agent_sample_proportions[t] = agent_sample_proportions[t] / np.sum(agent_sample_proportions[t])

    return agent_tweet_proportions, tweet_cohesion_matrix, agent_sample_proportions

def plot_KLD_similarity_matrix(KLD_intra_beliefs):
    KLD_plot_images = []
    #time_steps = [2,4,6,10,14,16,20,24,26,28,32,35,40,42,46,48,49,50,52,54,56,60,64,66,70,74,76,78,82,85,90,92,96,98,99]
    for t in range(T)[2:-1:2]:
        plt.imshow(KLD_intra_beliefs[:,:,t], cmap = 'gray')
        plt.title("Belief similarity matrix")
        plt.savefig('KLD, t = ' + str(t) + '.png')
        KLD_plot_images.append('KLD, t = ' + str(t) + '.png')
    return KLD_plot_images

def plot_tweet_similarity_matrix(tweet_cohesion_matrix):
    tweet_sim_images = []
    for t in range(T)[2:-1:2]:
        plt.imshow(tweet_cohesion_matrix[t], cmap = 'gray')
        plt.title("Tweet similarity matrix")
        plt.savefig('TSM, t = ' + str(t) + '.png')
        tweet_sim_images.append('TSM, t = ' + str(t) + '.png')
    return tweet_sim_images

def plot_proportions(tweets, beliefs, samples):
    tweet_proportions = []
    sampled_neighbours = []
    for t in range(T)[2:-2:2]:
        plt.figure(t)
        sns.heatmap(tweets[t], cmap = "gray", xticklabels = ["hashtag1", "hashtag2"], vmin = 0, vmax = 1)
        plt.title("Tweet proportions per agent")
        plt.savefig('TP, t = ' + str(t) + '.png')

        tweet_proportions.append('TP, t = ' + str(t) + '.png')

    plt.show()

    for t in range(T)[2:-2:2]:
        plt.figure(t)
        sns.heatmap(samples[t], cmap = "gray", vmin = 0, vmax = 0.2)
        plt.title("Sampled Neighbours per agent")
        plt.savefig('SN, t = ' + str(t) + '.png')

        sampled_neighbours.append('SN, t = ' + str(t) + '.png')
          
    return tweet_proportions, sampled_neighbours
    #sns.heatmap(beliefs, cmap = "gray", xticklabels = ["idea1", "idea2"])
    #plt.title("Belief proportions per agent")
    #plt.show()

if __name__ == '__main__':

    N = 8 # total number of agents
    idea_levels = 2 
    num_H = 2

    p_vec = np.linspace(0.6,1,1) # different levels of random connection parameter in Erdos-Renyi random graphs
    num_trials = 1 # number of trials per level of the ER parameter
    T = 150
    #fig, axs = plt.subplots(len(p_vec)/2, len(p_vec)/2)
    for param_idx, p in enumerate(p_vec):
        print("p is" + str(p))

        for trial_i in range(num_trials):
            
            G = nx.fast_gnp_random_graph(N,p) # create the graph for this trial & condition

            #this performs the multiagent inference
            all_actions, all_beliefs, all_observations, agents, agent_neighbours = inference_loop(G,N)

            #collect metrics
            agent_beliefs, KLD_inter_beliefs, KLD_intra_beliefs, belief_proportions, _, _ = get_belief_metrics(all_beliefs, agents, agent_neighbours,T)
            tweet_proportions, tweet_cohesion_matrix, agent_sample_proportions = get_action_metrics(all_actions, N, T)
        
            #make plots 
            belief_plot_images = plot_beliefs_over_time(all_actions, agent_beliefs, p, T)
            plt.show()
            KLD_images = plot_KLD_similarity_matrix(KLD_intra_beliefs)
            plt.show()
            tweet_sim_images = plot_tweet_similarity_matrix(tweet_cohesion_matrix)
            plt.show()
            tweet_proportions, sampled_neighbours = plot_proportions(tweet_proportions, belief_proportions, agent_sample_proportions)

    with imageio.get_writer('belief_plot.gif', mode='I') as writer:
        for filename in belief_plot_images:
            image = imageio.imread(filename)
            writer.append_data(image)
    for filename in set(belief_plot_images):
        os.remove(filename)

    with imageio.get_writer('KLD_plot.gif', mode='I') as writer:
        for filename in KLD_images:
            image = imageio.imread(filename)
            writer.append_data(image)

    for filename in set(KLD_images):
        os.remove(filename)

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

    with imageio.get_writer('sampled_neighbours.gif', mode='I') as writer:
        for filename in sampled_neighbours:
            image = imageio.imread(filename)
            writer.append_data(image)

    for filename in set(sampled_neighbours):
        os.remove(filename)