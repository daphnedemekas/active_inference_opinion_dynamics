import numpy as np
from Model.genmodel import GenerativeModel
from Model.agent import Agent
from Model.network_tools import create_multiagents, clip_edges, connect_edgeless_nodes
import networkx as nx
from Model.pymdp.utils import obj_array, index_list_to_onehots, sample
from Model.pymdp.maths import spm_dot, dot_likelihood, softmax
import seaborn as sns
from matplotlib import pyplot as plt
import time

def agent_loop(agent, observations = None, initial = False, initial_action = None):  
    qs = agent.infer_states(initial, tuple(observations))
    print("qs[0]")
    print(qs[0])
    policy = agent.infer_policies()
    action = agent.sample_action()
    action = action[-2:]
    who_i_looked_at = int(action[-1]+1)
    what_they_tweeted = observations[int(action[-1])+1]


    if initial == True:
        action = agent.action
    return action, qs[0]

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
    all_actions = []
    all_beliefs = []
    all_views = np.zeros(N)

    while timestep < T:
        print(str(timestep) + "/" + str(T))
        actions = []
        beliefs = []
        #initial_actions = [[0,1],[1,0],[1,0],[0,2],[1,3],[1,4]]

        for i in range(len(agents)):
            action, belief = agent_loop(agents[i], observations[i], initial)
            actions.append(action)
            beliefs.append(belief)
        all_actions.append(actions)    
        all_beliefs.append(beliefs)
        initial = False

        for idx, agent in enumerate(agents):
            for n in range(agent.genmodel.num_neighbours):
                observations[idx][n+1] = 0
            my_tweet = int(actions[idx][-2])
            observations[idx][0] = my_tweet
            observed_neighbour = int(actions[idx][-1])
            #which actual agent is that?
            observed_agent = agent_neighbours[idx][observed_neighbour]
            all_views[observed_agent] += 1
            observations[idx][observed_neighbour+1] = int(actions[observed_agent][-2]) + 1
            observations[idx][-1] = int(actions[idx][-1]) 

        timestep += 1
        #if timestep == 5 or timestep == 20 or timestep == 10 or timestep == 25 or timestep == 30 :
        #    make_plots(all_actions, all_beliefs,p,timestep)
    return all_actions, all_beliefs

def color_dict(value):
    if value < 0.5:
        return "blue"
    else:
        return "red"
def make_plots(all_actions, all_beliefs,p,T):

    

    tweet_history = np.zeros((N, T))
    belief_history = np.zeros((N,T))

    for t in range(T):
        for n in G.nodes():
            tweet_history[n,t] = all_actions[t][n][-2]
            belief_history[n,t] = all_beliefs[t][n][0]


    belief_in_idea_1 = []
    belief_in_idea_2 = []

    for t in range(T):
        timestep = []
        timestep_b2 = []
        for a in G.nodes():
            print(all_beliefs[t][a])
            timestep.append(all_beliefs[t][a][0])
            #timestep_b2.append(all_beliefs[t][a][1])
        belief_in_idea_1.append(timestep)
        #belief_in_idea_2.append(timestep_b2)
    
    for a in range(N):
        axs[param_idx, param_idx].plot(np.array(belief_in_idea_1)[:,a], color = color_dict(np.array(belief_in_idea_1)[:,a][-1]), label = "beliefs in idea 1")
    #for a in range(N):
    #    plt.plot(np.array(belief_in_idea_2)[:,a], color = "red", label = "beliefs in idea 2")
    plt.title("Connectedness of graph: " +str(p))
    plt.ylabel("Belief that idea is True")
    plt.xlabel("Time")
    #time.sleep(5)
    plt.show()
    plt.savefig("p = " + str(p))



    #sns.heatmap(tweet_history, cmap='gray', vmax=1., vmin=0., cbar=True)
    #plt.show()






N = 8 # total number of agents
idea_levels = 2 # the levels of beliefs that agents can have about the idea (e.g. 'True' vs. 'False', in case `idea_levels` ==2)
num_H = 2 #the number of hashtags, or observations that can shed light on the idea

def inference_loop(G,N):
    try:
        G = nx.fast_gnp_random_graph(N,p)
        G, agents_dict, agents, agent_neighbours = create_multiagents(G, N)
        G = connect_edgeless_nodes(G)
        all_actions, all_beliefs = multi_agent_loop(T, agents, agent_neighbours)
    except ValueError:
        all_actions, all_beliefs = inference_loop(G,N)
    
    return all_actions, all_beliefs

p_vec = np.linspace(0.3,1,6) # different levels of random connection parameter in Erdos-Renyi random graphs
num_trials = 1 # number of trials per level of the ER parameter
T = 50

fig, axs = plt.subplots(len(p_vec)/2, len(p_vec)/2)
j = 0
for param_idx, p in enumerate(p_vec):
    if param_idx % len(p_vec)/2:
        j += 1

    for trial_i in range(num_trials):
        
        G = nx.fast_gnp_random_graph(N,p) # create the graph for this trial & condition

        all_actions, all_beliefs = inference_loop(G,N)

        make_plots(all_actions, all_beliefs,p,T)

