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

G.add_node(mia)
G.add_node(vincent)
G.add_node(jules)



belief_state = {0 : "idea is true", 1: "idea is false"}
hashtags = {0: "republican", 1: "democrat"}
#cohesion_levels = {0 : "low cohesion true", }

def agent_loop(agent, name):    #time.sleep(1)
    print(str(name) + " believes: ")
    starting_state = agent.starting_state
    print("the " + str(belief_state.get(starting_state[0])))

    for idx, state in enumerate(starting_state[1:]):
        if idx < len(agent.genmodel.neighbour_h_idx):
            print("neighbour " + str(idx + 1) + " believes the " + belief_state.get(agent.starting_state[idx+1]))
    #time.sleep(1)
    print()
    print(str(name) + " is tweeting " + str(hashtags.get(starting_state[agent.genmodel.h_control_idx])))

    sampling_neighbour = starting_state[agent.genmodel.who_idx] 
    print(str(name) + " is reading neighbour" + str(sampling_neighbour +1) )

    print()
    print("Sampling likelihood to get current observations...")
    print()
    state_vector = index_list_to_onehots(agent.starting_state, agent.genmodel.num_states)
    #print("Observation: [my tweet, neighbour1s tweet, neighbour2's tweet, neighbour3's tweet, cohesion level]")
    observations = [sample(spm_dot(agent.genmodel.A[m],state_vector)) for m in range(agent.genmodel.num_modalities)]
    print("neighbour" + str(sampling_neighbour +1) + " is tweeting " + str(hashtags.get(observations[sampling_neighbour])))
    print(str(name) + "'s cohesion level is " + str(observations[-1]))
    print()
    print("Inferring next states from observations...")
    qs = agent.infer_states(0, tuple(observations))
    print("new approximate posterior")
    print(qs)
    # print()
    # print("infer policy")
    # action = agent.infer_policies(qs)
    # print(action)

    #now sample new states using this posterior
    #print(index_list_to_onehots(qs, agent.genmodel.num_states))
    #new_states = sample(index_list_to_onehots(qs, agent.genmodel.num_states))
    #print("new states")
    #print(new_states)

print("MIA")
agent_loop(mia, "mia")
print()
print("VINCENT")
agent_loop(vincent, "vincent")
print()
print("JULES")
agent_loop(jules, "jules")