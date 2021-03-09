import numpy as np
from Model.genmodel import GenerativeModel
from Model.agent import Agent
from Model.network_tools import create_multiagents
import networkx as nx
from Model.pymdp.utils import obj_array, index_list_to_onehots, sample
from Model.pymdp.maths import spm_dot, dot_likelihood, softmax

N = 3 # total number of agents
idea_levels = 2 # the levels of beliefs that agents can have about the idea (e.g. 'True' vs. 'False', in case `idea_levels` ==2)
num_H = 2 #the number of hashtags, or observations that can shed light on the idea

G = nx.complete_graph(N)

G, agents_dict = create_multiagents(G, N = N)


#belief_state = {0 : "idea is true", 1: "idea is false"}
#hashtags = {0: "#republican", 1: "#democrat"}

def agent_loop(agent, state = None):  
    if state == None:
        state = [sample(agent.genmodel.D[s]) for s in agent.genmodel.num_states]
    
    state_vector = index_list_to_onehots(state, agent.genmodel.num_states)
    observations = [sample(spm_dot(agent.genmodel.A[m],state_vector)) for m in range(agent.genmodel.num_modalities)]
    qs = agent.infer_states(0, tuple(observations))
    policy = agent.infer_policies(qs)
    action = agent.sample_action()
    #print(" agent tweeted " + str(hashtags.get(int(action[agent.genmodel.h_control_idx]))))
   # print(" agent will look at neighbour " + str(int(action[agent.genmodel.who_idx])) + "\n")
    new_state = [sample(qs[s]) for s in agent.genmodel.num_states]
    return new_state, action


#print(list(G.edges))
#print(G.graph)

actions = []
agent1 = Agent(**agents_dict[0])
agent2 = Agent(**agents_dict[1])
agent3 = Agent(**agents_dict[2])
agents = [agent1, agent2, agent3]

agent_neighbours = {0:{0: 1, 1:2},1:{0: 0, 1:2}, 2:{0: 0, 1:1}}

timestep = 0
action = None
states = [None, None, None]

while timestep < 200:
    state1, action1 = agent_loop(agent1, states[0])
    state2, action2 = agent_loop(agent1, states[1])
    state3, action3 = agent_loop(agent1, states[2])
    actions = [action1,action2, action3]
    states = [state1, state2, state3]
    for idx, agent in enumerate(agents):
        observed_neighbour = int(actions[idx][agent.genmodel.who_idx])
        #which actual agent is that?
        observed_agent = agent_neighbours[idx][observed_neighbour]
        states[idx][observed_neighbour+1] = int(actions[observed_agent][agents[observed_agent].genmodel.h_control_idx])
    print(states)
    timestep += 1


    #now these actions need to create the next observations 
#print(actions)





