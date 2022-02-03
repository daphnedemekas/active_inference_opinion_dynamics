#%%
import os 
os.chdir('/Users/daphnedemekas/Desktop/Research/active_inference_opinion_dynamics')
#%%


import numpy as np
from model.agent import Agent
import networkx as nx
from model.pymdp import utils
from model.pymdp.utils import obj_array, index_list_to_onehots, sample, reduce_a_matrix
from model.pymdp.maths import softmax, spm_dot, spm_log, get_joint_likelihood, calc_free_energy
from model.pymdp.inference import update_posterior_states
from simulation.simtools import initialize_agent_params, generate_network, initialize_network, run_simulation
import seaborn as sns
from matplotlib import pyplot as plt
from analysis.analysis_tools import collect_idea_beliefs, collect_sampling_history, collect_tweets
#%%


#TODO: figure out why update_poster_policies_reduced_vectorized() isn't working
""""
For the experiment:

- initialize agents with predefined beliefs in one idea v the other rather than making them converge to clusters from ambiguous beliefs
- parameters: 1) ecb 2) belief volatility 3) preference parameter over esteem 4) the softmax parameters for the A matrix slice for the esteem mapping 

- high self esteem builds confidence within the group (group believes what you believe)
- low self esteem creates uncertainty about the outcome of your behaviour (you believe the group disagrees with you / you are an outsider), this should make it more difficult to become certain about the idea

we want to see:
1) isolated agents with low self esteem -- what is the factor that isolates them? (epistemic confirmation bias, number of social connections, certainty about actions)
2) low self esteem agents who choose to bend the groups norms (change their ideas)
3) low self esteem agents who actively seek out members that agree with them 

Then we can also investigate the following questions
- How does self esteem influence the sampling probabilities across the network?
- how does self esteem influence the probability of doing the "say nothing" action?
- how does self esteem observation influence the belief strength / rate of belief increase 
- the influence of self esteem observation on action and whether the agents will change what they believe / tweet as a result of low self esteem observation
Finally, we want to see whether the following behaviour will emerge naturally from the model or if there is something else required to ensure it:

Agents should have a preference to observe that their neighbours have low esteem if their neighbours disagree with them, which should push them to reject those neighbours in order to decrease their esteem
"""

""" Set up the generative model """

idea_levels = 2 # the levels of beliefs that agents can have about the idea (e.g. 'True' vs. 'False', in case `idea_levels` ==2)
num_H = 2 #the number of hashtags, or observations that can shed light on the idea
h_idea_mapping = np.eye(num_H)
h_idea_mapping[:,0] = softmax(h_idea_mapping[:,0]*1.0)
h_idea_mapping[:,1] = softmax(h_idea_mapping[:,1]*1.0)


""" Set parameters and generate agents"""
env_d = 8
c = 0
ecb = 4
belief_d = 4
T = 50 #the number of timesteps 
N = 4
p = 1

G = generate_network(N,p)
esteem_parameters = [1.5,0.2,-1.5]
C_params = [1,0,-2]

model_parameters = { "esteem_parameters": esteem_parameters, "C_params":C_params}

agent_constructor_params = initialize_agent_params(G, h_idea_mapping = h_idea_mapping, \
                                    ecb_precisions = ecb, B_idea_precisions = env_d, \
                                        B_neighbour_precisions = belief_d, model = "self_esteem", model_parameters = model_parameters)

G = initialize_network(G, agent_constructor_params, T = T, model = "self_esteem")


G = run_simulation(G, T = T, model="self_esteem")

#%%

all_qs = collect_idea_beliefs(G)

plt.plot(all_qs[:,0,:])

all_neighbour_samplings = collect_sampling_history(G)
all_tweets = collect_tweets(G)

# %%

def collect_esteems(G, T):
    n = len(G.nodes())
    esteem_observations = np.zeros((T+1, n, n))
    for agent in range(n):
        esteem_observations[:,agent] = G.nodes()[agent]['o'][:, -n:]
    return esteem_observations


""" PLOTTING FOCAL ESTEEMS"""
esteem_observations = collect_esteems(G,T)
for i in range(N):
    plt.plot(esteem_observations[:,0,i])
plt.title("Focal esteems")
plt.show()

# %%
