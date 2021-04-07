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
    action = agent.sample_action()
    action = action[-2:]
    who_i_looked_at = int(action[-1]+1)
    what_they_tweeted = observations[int(action[-1])+1]


    if initial == True:
        action = agent.action
    return action, qs

def multi_agent_loop(T, agents, agent_neighbours):
    actions = []
    timestep = 0
    action = None

    observations = []

    def reset_observations(agent):
        o = []
        o.append(None)
        for n in range(agent.genmodel.num_neighbours):
            o.append(0)
        o.append(None)
        return o

    for agent in agents:
        observations.append(reset_observations(agent))

    initial = True
    all_actions = obj_array((T,N,))
    all_beliefs = obj_array((T,N))
    all_observations = obj_array((T,N))
    #all_views = np.zeros(N)

    for t in range(T):
        #print(str(t) + "/" + str(T))
        #initial_actions = [[0,1],[1,0],[1,0],[0,2],[1,3],[1,4]]
        actions = []
        for i in range(len(agents)):
            action, belief = agent_loop(agents[i], observations[i], initial)
            all_actions[t,i] = action
            all_beliefs[t,i] = belief
            actions.append(action)

        initial = False

        for idx, agent in enumerate(agents):
            for n in range(agent.genmodel.num_neighbours):
                observations[idx][n+1] = 0
            my_tweet = int(actions[idx][-2])
            observations[idx][0] = my_tweet
            observed_neighbour = int(actions[idx][-1])
            observed_agent = agent_neighbours[idx][observed_neighbour]
            observations[idx][observed_neighbour+1] = int(actions[observed_agent][-2]) + 1
            observations[idx][-1] = int(actions[idx][-1]) 
        all_observations[t,:] = observations

        #if timestep == 5 or timestep == 20 or timestep == 10 or timestep == 25 or timestep == 30 :
        #    make_plots(all_actions, all_beliefs,p,timestep)
    return all_actions, all_beliefs, all_observations

def color_dict(value):
    if value < 0.5:
        return "blue"
    else:
        return "red"
def make_plots(all_actions, agent_own_beliefs,p,T):

    

    tweet_history = np.zeros((N, T))

    for t in range(T):
        for n in G.nodes():
            tweet_history[n,t] = all_actions[t][n][-2]

    
    for a in range(N):
        data = agent_own_beliefs[a][:,0]
        plt.plot(data, color = color_dict(data[-1]), label = "beliefs in idea 1")

    plt.title("Connectedness of graph: " +str(p))
    plt.ylabel("Belief that idea is True")
    plt.xlabel("Time")
    #time.sleep(5)
    plt.show()
    #plt.savefig("p = " + str(p))



    #sns.heatmap(tweet_history, cmap='gray', vmax=1., vmin=0., cbar=True)
    #plt.show()






N = 4 # total number of agents
idea_levels = 2 # the levels of beliefs that agents can have about the idea (e.g. 'True' vs. 'False', in case `idea_levels` ==2)
num_H = 2 #the number of hashtags, or observations that can shed light on the idea

def inference_loop(G,N):
    try:
        G = nx.fast_gnp_random_graph(N,p)
        G, agents_dict, agents, agent_neighbours = create_multiagents(G, N)
        G = connect_edgeless_nodes(G)
        all_actions, all_beliefs, all_observations = multi_agent_loop(T, agents, agent_neighbours)
    except ValueError:
        all_actions, all_beliefs, all_observations, agents, agent_neighbours = inference_loop(G,N)
    
    return all_actions, all_beliefs, all_observations, agents, agent_neighbours

p_vec = np.linspace(0.3,1,5) # different levels of random connection parameter in Erdos-Renyi random graphs
num_trials = 1 # number of trials per level of the ER parameter
T = 5

#fig, axs = plt.subplots(len(p_vec)/2, len(p_vec)/2)
j = 0
for param_idx, p in enumerate(p_vec):
    print("p is" + str(p))
    if param_idx % len(p_vec)/2:
        j += 1

    for trial_i in range(num_trials):
        
        G = nx.fast_gnp_random_graph(N,p) # create the graph for this trial & condition

        all_actions, all_beliefs, all_observations, agents, agent_neighbours = inference_loop(G,N)

        agent_own_beliefs_per_timestep = np.array([[belief[0] for belief in all_beliefs[:,a]] for a in range(len(agents))]) # (N,T,2)
        
        all_neighbour_perceptions = np.zeros((N,N-1,T,2))
        for a in range(len(agents)):
            agent_p = []
            for i, n in enumerate(agent_neighbours[a]):
                all_neighbour_perceptions[a,i] = np.array([belief[n] for belief in all_beliefs[:,a]])

        agent_hashtag_beliefs_per_timestep = [[belief[-2] for belief in all_beliefs[:,a]] for a in range(len(agents))] # (N,T,2)
        agent_who_idx_beliefs_per_timestep = [[belief[-1] for belief in all_beliefs[:,a]] for a in range(len(agents))] # (N,T,2)

        make_plots(all_actions, agent_own_beliefs_per_timestep,p,T)

        #proportion of agents believing in idea 1 at the final timestep 
        final_timestep_beliefs = [0 if agent_own_beliefs_per_timestep[:,-1][a][0] < 0.5 else 1 for a in range(len(agents))]
        idea_1_believers = sum(final_timestep_beliefs) 
        idea_0_believers = len(final_timestep_beliefs) - idea_1_believers

        belief_proportions = [idea_0_believers/N, idea_1_believers/N]

        print("belief proportions")
        print(belief_proportions)
        #rate of change of belief per agent
        strength_of_beliefs_per_timestep = np.diff(agent_own_beliefs_per_timestep)
        total_strengths = [np.sum(b) for b in strength_of_beliefs_per_timestep]
        print("belief strengths")
        total_strengths_normalised = total_strengths / np.sum(np.abs(np.array(total_strengths)))
        print(total_strengths_normalised)

        idea_1_strength = sum(x for x in total_strengths_normalised if x > 0 )
        idea_0_strength = - sum(x for x in total_strengths_normalised if x <0 )

        plt.bar(["idea 0","idea 1"],belief_proportions)
        plt.show()
        plt.bar(["idea 0","idea 1"],[idea_0_strength, idea_1_strength])
        plt.show()




