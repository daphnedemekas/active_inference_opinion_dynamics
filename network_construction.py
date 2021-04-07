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
        #print(str(t) + "/" + str(T))
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



def make_plots(all_actions, agent_own_beliefs,p,T):

    for a in range(N):
        data = agent_own_beliefs[a][:,0]
        plt.plot(data, color = color_dict(data[-1]), label = "beliefs in idea 1")

    plt.title("Connectedness of graph: " +str(p))
    plt.ylabel("Belief that idea is True")
    plt.xlabel("Time")
    plt.show()

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

    for a in range(len(agents)):
        agent_p = []
        for i, n in enumerate(agent_neighbours[a]):
            all_neighbour_perceptions[a,i] = np.array([belief[n] for belief in all_beliefs[:,a]])
            KLD_inter_beliefs[a,i] = KL_div(all_neighbour_perceptions[a,i][:,0], all_neighbour_perceptions[a,i][:,1], agent_own_beliefs_per_timestep[a][:,0] , agent_own_beliefs_per_timestep[a][:,1])

    for a in range(len(agents)):
        for n in range(len(agents)):
            KLD_intra_beliefs[a,n] = KL_div(agent_own_beliefs_per_timestep[a][:,0], agent_own_beliefs_per_timestep[a][:,1], agent_own_beliefs_per_timestep[n][:,0], agent_own_beliefs_per_timestep[n][:,1])

    agent_hashtag_beliefs_per_timestep = [[belief[-2] for belief in all_beliefs[:,a]] for a in range(N)] # (N,T,2)
    agent_who_idx_beliefs_per_timestep = [[belief[-1] for belief in all_beliefs[:,a]] for a in range(N)] # (N,T,2)

    #proportion of agents believing in idea 1 at the final timestep 
    final_timestep_beliefs = [0 if agent_own_beliefs_per_timestep[:,-1][a][0] < 0.5 else 1 for a in range(N)]
    idea_1_believers = sum(final_timestep_beliefs) 
    idea_0_believers = len(final_timestep_beliefs) - idea_1_believers
    belief_proportions = [idea_0_believers/N, idea_1_believers/N]

    #rate of change of belief per agent
    #MAKE THESE KL DIVERGENCES
    difference_of_beliefs_per_timestep = np.diff(agent_own_beliefs_per_timestep)
    belief_differences = [np.sum(b) for b in difference_of_beliefs_per_timestep]
    belief_differences_normalised = belief_differences / np.sum(np.abs(np.array(belief_differences)))

    return agent_own_beliefs_per_timestep, KLD_inter_beliefs, KLD_intra_beliefs, belief_proportions, belief_differences_normalised, agent_hashtag_beliefs_per_timestep, agent_who_idx_beliefs_per_timestep


def get_action_metrics(all_actions, N,T):
    all_actions = np.array(all_actions) # shape is T, N, 2
    agent_actions_per_timestep = np.array([[action[0] for action in all_actions[:,a]] for a in range(N)]) # (N,T,2)
    
    agent_tweet_proportions = np.zeros((N,2))

    for a in range(N):
        hashtag1 = sum(agent_actions_per_timestep[a])
        hashtag2 = len(agent_actions_per_timestep[a]) - hashtag1
        agent_tweet_proportions[a] = [hashtag1/T, hashtag2/T]

    return agent_tweet_proportions

if __name__ == '__main__':

    N = 4 # total number of agents
    idea_levels = 2 
    num_H = 2

    p_vec = np.linspace(0.7,1,1) # different levels of random connection parameter in Erdos-Renyi random graphs
    num_trials = 1 # number of trials per level of the ER parameter
    T = 50

    #fig, axs = plt.subplots(len(p_vec)/2, len(p_vec)/2)
    for param_idx, p in enumerate(p_vec):
        print("p is" + str(p))

        for trial_i in range(num_trials):
            
            G = nx.fast_gnp_random_graph(N,p) # create the graph for this trial & condition

            all_actions, all_beliefs, all_observations, agents, agent_neighbours = inference_loop(G,N)

            agent_beliefs, KLD_inter_beliefs, KLD_intra_beliefs, belief_proportions, belief_differences_normalised, _, _ = get_belief_metrics(all_beliefs, agents, agent_neighbours,T)
            make_plots(all_actions, agent_beliefs,p,T)
            
            tweet_proportions = get_action_metrics(all_actions, N, T)
            print(tweet_proportions)
            #plt.imshow(KLD_intra_beliefs[:,:,-1], cmap = 'gray')
            #plt.show()
