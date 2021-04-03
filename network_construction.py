import numpy as np
from Model.genmodel import GenerativeModel
from Model.agent import Agent
from Model.network_tools import create_multiagents, clip_edges, connect_edgeless_nodes
import networkx as nx
from Model.pymdp.utils import obj_array, index_list_to_onehots, sample
from Model.pymdp.maths import spm_dot, dot_likelihood, softmax
import seaborn as sns
from matplotlib import pyplot as plt

N = 8 # total number of agents
idea_levels = 2 # the levels of beliefs that agents can have about the idea (e.g. 'True' vs. 'False', in case `idea_levels` ==2)
num_H = 2 #the number of hashtags, or observations that can shed light on the idea
# num_H = 3 #the number of hashtags, or observations that can shed light on the idea

#G = nx.complete_graph(N)
G = nx.fast_gnp_random_graph(N, 0.6)

# Method 1 for cleaning the graph: keep regenerating the graph as long as its not connected. once the graph is connected, we know everyone has at least one edge
#while not nx.is_connected(G):
#     G = nx.fast_gnp_random_graph(N, 0.6)

# Method 2 for cleaning the graph: this doesn't ensure connectedness, but rather just ensures that everyone has at least TWO edges 
G, _ = clip_edges(G, max_degree = 8)
G = connect_edgeless_nodes(G)

##print(list(G.edges()))

G, agents_dict = create_multiagents(G, N)
for n in range(N):
    agents_dict[n]['idea_mapping_params']['h_idea_mapping'] = np.eye(num_H)

hashtags = {0: "#democrat", 1: "#republican"}

#belief_state = {0 : "idea is true", 1: "idea is false"}
#hashtags = {0: "#republican", 1: "#democrat"}

def agent_loop(agent, observations = None, initial = False):  
    #print("Observed tweet: " + str(observations))
    qs = agent.infer_states(initial, tuple(observations))
    #print("updated belief" + str(qs[0]))
    policy = agent.infer_policies(qs)
    action = agent.sample_action()
    #print("ACTION")
    #print(action)
    action = action[-2:]
    #print("What i tweeted: " + str(action[-2]))
    #print("Who i looked at" + str(action[-1]))
    #print()
    who_i_looked_at = int(action[-1]+1)
    what_they_tweeted = observations[int(action[-1])+1]
    #print(agent.genmodel.A[1][what_they_tweeted,:,:,0,0,0])
    #print(agent.genmodel.A[1][what_they_tweeted,:,0,0,0,0])
    #print(agent.genmodel.A[1][what_they_tweeted,:,1,0,0,0])


    if initial == True:
        action = agent.initial_action

    return action, qs[0]

actions = []
agents = []
idea_mappings = []

agent_neighbours = {}

for agent_i in G.nodes():
    agent_neighbours[agent_i] = list(nx.neighbors(G, agent_i))
    agent = Agent(**agents_dict[agent_i])
    agents.append(agent)
    idea_mappings.append(agent.genmodel.h_idea_mapping)

timestep = 0
action = None

observations = []
num_cohesion_levels = []
true_affirmations = []
trues = []
false_affirmations = []
falses = []

for agent in agents:
    o = []
    o.append(None)
    for n in range(agent.genmodel.num_neighbours):
        o.append(0)
    o.append(None)
    observations.append(o)
    num_cohesion_levels.append( 2 * (agent.genmodel.num_neighbours+1))
    true_affirmations.append(0)
    trues.append(0)
    false_affirmations.append(0)
    falses.append(0)

initial = True
all_actions = []
all_beliefs = []
all_views = np.zeros(N)

while timestep < 50:
    print("TIMSTEP")
    print(timestep)
    actions = []
    beliefs = []
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

        my_belief = np.argmax(idea_mappings[idx][my_tweet]) # p(true | h1) p(false | h1) so we just choose whether this agent beliefs the hashtag represents true or false 
        #TODO: we should use the distribution instead 

        if my_belief == 0:
            trues[idx] += 1
            if my_tweet == int(actions[observed_agent][-2]):
                true_affirmations[idx] += 1 #accumulate how many times we have all agreed on the idea being true
            cohesion_level = int((trues[idx] - true_affirmations[idx]) / trues[idx] * (num_cohesion_levels[idx]/2-1))
            #if i tweeted true, then my cohesion level is how much others have been agreeing with me
            # 0 is the most, 2 is the least 

        elif my_belief == 1:
            falses[idx] += 1
            if my_tweet == int(actions[observed_agent][-2]):
                false_affirmations[idx] += 1
            cohesion_level = int(false_affirmations[idx] / falses[idx] * (num_cohesion_levels[idx]/2-1)+num_cohesion_levels[idx]/2)
            #if i tweeted false, then my cohesion level is how much others have been agreeing with me
            # 5 is the most, 3 is the least 
        observations[idx][-1] = cohesion_level

    if timestep == 49:
        colors_dict = {}
        for action_idx in range(len(actions)):
            if actions[action_idx][-2] == 0:
                colors_dict[action_idx] = "blue"
            else:
                colors_dict[action_idx] = "red"

        nx.draw_networkx(G, node_color=colors_dict.values())
        plt.savefig('final_net_0.8.png',dpi=325)

        
    timestep += 1


tweet_history = np.zeros((N, 50))
belief_history = np.zeros((N,50))
reds = []
blues = []

for t in range(50):
    for n in G.nodes():
        tweet_history[n,t] = all_actions[t][n][-2]
        belief_history = all_beliefs[t][n][0]


for agent in all_beliefs:
    for b in agent:
        if b[0] < 0.5:
            reds.append(b[0])
        else:
            blues.append(b[0])
         
plt.figure(figsize=(14, 8))

sizes_r = []
for s in reds:
  map = {5:1,4:2,3:3,2:4,1:5,0:6}
  sizes_r.append(map.get(int(s*10)))

sizes_b = []
for s in blues:
  map = {5:1,6:2,7:3,8:3,9:5,10:6}

  sizes_b.append(map.get(int(s*10)))


sns.heatmap(belief_history, cmap='gray', vmax=1., vmin=0., cbar=True)
plt.savefig('belief_history_06.png',dpi=325)


sns.heatmap(tweet_history, cmap='gray', vmax=1., vmin=0., cbar=True)
plt.savefig('tweet_history_06.png',dpi=325)

plt.figure(figsize=(14, 8))

sns.scatterplot(np.linspace(0.5,1,len(blues)), blues, alpha = 0.8, size = sizes_b, sizes = (100,400)
)
sns.scatterplot(np.flip(np.linspace(0,0.5,len(reds))), reds, alpha = 0.8, size = sizes_r, sizes = (100,400), color = "red"
)

plt.savefig('scatter_06.png',dpi=325)

plt.figure(figsize=(14, 8))

#sns.barplot(range(N), all_views)
#plt.savefig('views_0.6.png',dpi=325)

""" 
fig, axes = plt.subplots(2,2,figsize=(16, 10))

sns.heatmap(idea_mappings[0], ax = axes[0,0], cmap='gray', vmax=1., vmin=0., cbar=True)
sns.heatmap(idea_mappings[1], ax = axes[0,1], cmap='gray', vmax=1., vmin=0., cbar=True)
sns.heatmap(idea_mappings[2], ax = axes[1,0], cmap='gray', vmax=1., vmin=0., cbar=True)
sns.heatmap(idea_mappings[3], ax = axes[1,1], cmap='gray', vmax=1., vmin=0., cbar=True)
plt.savefig('idea_mappings_all_agents.png',dpi=325)
 """






