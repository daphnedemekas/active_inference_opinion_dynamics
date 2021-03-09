import numpy as np
from Model.pymdp.utils import obj_array, index_list_to_onehots, sample
from Model.pymdp.maths import spm_dot, dot_likelihood, softmax
from Model.pymdp.inference import average_states_over_policies
from Model.genmodel import GenerativeModel
from Model.agent import Agent
from Model.params import *
import itertools
import time

import networkx as nx

G = nx.Graph()  


#make agents 
mia = Agent(mia_params["neighbour_params"], mia_params["idea_mapping_params"], mia_params["policy_params"], mia_params["C_params"])
vincent = Agent(vincent_params["neighbour_params"], vincent_params["idea_mapping_params"], vincent_params["policy_params"], vincent_params["C_params"])
jules = Agent(jules_params["neighbour_params"], jules_params["idea_mapping_params"], jules_params["policy_params"], jules_params["C_params"])

#G.add_node(mia)
#G.add_node(vincent)
#G.add_node(jules)


#edges = list(itertools.product(*[mia, vincent, jules]))

#G.add_edges_from[edges]

belief_state = {0 : "idea is true", 1: "idea is false"}
hashtags = {0: "#republican", 1: "#democrat"}

def agent_loop(agent, name):  
    print("-------- Initial position --------")
    print(str(name) + " believes: ")
    
    
    starting_state = agent.starting_state
    
    
   # print("the " + str(belief_state.get(starting_state[0])))
   # for idx, state in enumerate(starting_state[1:]):
    #     if idx < len(agent.genmodel.neighbour_h_idx):
    #         print("neighbour " + str(idx + 1) + " believes the " + belief_state.get(agent.starting_state[idx+1]))
    # print()
    # print(str(name) + " is tweeting " + str(hashtags.get(starting_state[agent.genmodel.h_control_idx])))
    sampling_neighbour = starting_state[agent.genmodel.who_idx] 
    # print(str(name) + " is reading neighbour" + str(sampling_neighbour +1)  + "\n")
    # print("-------- Sampling likelihood to get current observations... ---------" + "\n")



    state_vector = index_list_to_onehots(agent.starting_state, agent.genmodel.num_states)
    observations = [sample(spm_dot(agent.genmodel.A[m],state_vector)) for m in range(agent.genmodel.num_modalities)]

    # print("neighbour" + str(sampling_neighbour +1) + " is tweeting " + str(hashtags.get(observations[sampling_neighbour])))
    # print(str(name) + "'s cohesion level is " + str(observations[-1]) + "\n")
    # print(" -------- Inferring beliefs from observations... ------- ")


    qs = agent.infer_states(0, tuple(observations))


    # print(str(name) + " now believes the idea is true with probability " + str(qs[0][0]) + "\n")
    # print(" -------- Inferring policy from beliefs... ------- ")


    policy = agent.infer_policies(qs)

    
    # print(str(name) + " will tweet " + str(hashtags.get(0)) + " with probability " + str(policy[0] + policy[1]) + "\n")
    # print(" -------- Sampling action... ------- " + "\n")
    action = agent.sample_action()
    print(str(name) + " tweeted " + str(hashtags.get(int(action[agent.genmodel.h_control_idx]))))
    print(str(name) + " will look at neighbour " + str(int(action[agent.genmodel.who_idx])) + "\n")




mia_neighbours_indices = [0, 1]

print("MIA")
agent_loop(mia, "mia")
agent_loop(vincent, "vincent")
agent_loop(jules, "jules")


mia_observations = np.zeros(mia.genmodel.num_modalities)
mia_observations[0] = mia.action[mia.genmodel.h_control_idx]

if mia.action[-1] == 0:
    mia_observations[1] = int(vincent.action[vincent.genmodel.h_control_idx])
elif mia.action[-1] == 1:
    mia_observations[2] = int(jules.action[jules.genmodel.h_control_idx])

mia_relative_state_vector = [np.random.randint(idea_levels),   ]
mia_observations[-1] = sample(spm_dot(mia.genmodel.A[-1],))

mia_observations[mia_actions[-1]+1] = all_agents[mia_neighbours_indices[mia_actions[-1]]].action[h_control_idx]