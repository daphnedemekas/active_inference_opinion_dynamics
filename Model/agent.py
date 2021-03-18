import numpy as np
from .genmodel import GenerativeModel
from .pymdp.inference import update_posterior_states
from .pymdp.control import *
from .pymdp.maths import spm_log
from .pymdp import utils

class Agent(object):

    def __init__(
        self,
        neighbour_params,
        idea_mapping_params,
        policy_params,
        C_params       
        ):            

        self.genmodel = GenerativeModel(**neighbour_params, **idea_mapping_params, **policy_params, **C_params)
        self.set_starting_state_and_priors(policy_params["initial_action"])
        self.action = np.zeros(len(self.genmodel.control_factor_idx))
        self.inference_params = {"num_iter":10, 
                                 "dF":1.0,
                                 "dF_tol":0.001}
        self.policy_hyperparams = {"use_utility": True,
                                   "use_states_info_gain": True,
                                   "use_param_info_gain": False}
        self.initial_action = policy_params["initial_action"]


    def infer_states(self, timestep, observation):
        empirical_prior = utils.obj_array(self.genmodel.num_factors)

        if timestep == True:
            for f in range(self.genmodel.num_factors):
                empirical_prior[f] = spm_log(self.genmodel.D[f])

        else:
            for f, ns in enumerate(self.genmodel.num_states):
                empirical_prior[f] = spm_log(self.genmodel.B[f][:,:, int(self.action[f])].dot(self.qs[f]))
        
        qs = update_posterior_states(observation, self.genmodel.A, prior=empirical_prior, **self.inference_params)

        self.qs = qs

        
        return qs


    def infer_policies(self, qs):


        self.genmodel.E = self.genmodel.get_policy_prior(qs[0]) 
        q_pi, neg_efe = update_posterior_policies(self.qs, self.genmodel.A, self.genmodel.B, self.genmodel.C, self.genmodel.E, self.genmodel.policies, **self.policy_hyperparams)
        self.q_pi = q_pi
        self.neg_efe = neg_efe
        return q_pi

    def sample_action(self):

        action = sample_action(self.q_pi, self.genmodel.policies, self.genmodel.num_states, sampling_type = 'marginal_action') #how does this work? 
        self.action = action
        return action

    def set_starting_state_and_priors(self, initial_action):
        self.genmodel.D = self.genmodel.generate_prior_states(initial_action = initial_action)


