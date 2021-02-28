import numpy as np
from Model.pymdp.utils import obj_array, index_list_to_onehots, sample
from Model.pymdp.maths import spm_dot, dot_likelihood, softmax
from Model.pymdp.inference import average_states_over_policies
from Model.genmodel import GenerativeModel
from Model.agent import Agent
from Model.params import *
import itertools

#make agents 
mia = Agent(mia_params["neighbour_params"], mia_params["idea_mapping_params"], mia_params["policy_params"], mia_params["C_params"])
vincent = Agent(vincent_params["neighbour_params"], vincent_params["idea_mapping_params"], vincent_params["policy_params"], vincent_params["C_params"])
jules = Agent(jules_params["neighbour_params"], jules_params["idea_mapping_params"], jules_params["policy_params"], jules_params["C_params"])

def agent_loop(agent):
        
    state_vector = index_list_to_onehots(agent.starting_state, agent.genmodel.num_states)
    observations = [sample(spm_dot(agent.genmodel.A[m],state_vector)) for m in range(agent.genmodel.num_modalities)]

    qs = agent.infer_states(0, tuple(observations))
    print(qs)

print("MIA")
agent_loop(mia)
print("VINCENT")
agent_loop(vincent)
print("JULES")
agent_loop(jules)