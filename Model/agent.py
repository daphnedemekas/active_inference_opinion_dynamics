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
        self.set_starting_state_and_priors(policy_params["starting_state"])
        self.action = np.zeros(len(self.genmodel.control_factor_idx))
        self.inference_params = {"num_iter":10, 
                                 "dF":1.0,
                                 "dF_tol":0.001}
        self.policy_hyperparams = {"use_utility": True,
                                   "use_states_info_gain": True,
                                   "use_param_info_gain": False}
        self.starting_state = policy_params["starting_state"]


    def infer_states(self, timestep, observation):

        if timestep == 0:
            empirical_prior = utils.obj_array(self.genmodel.num_factors)
            for f in range(self.genmodel.num_factors):
                empirical_prior[f] = spm_log(self.genmodel.D[f])
            print("initial state priors")
            print(self.genmodel.D)
            print()

        else:
            for f, ns in enumerate(self.genmodel.num_states):
                empirical_prior[f] = spm_log(self.B[f][:,:,self.action].dot(self.qs[f]))
        
        qs = update_posterior_states(observation, self.genmodel.A, prior=empirical_prior, **self.inference_params)

        self.qs = qs
        
        return qs


    def infer_policies(self, qs):

        self.genmodel.E = self.genmodel.get_policy_prior(qs) 

        self.q_pi = update_posterior_policies(self.qs, self.genmodel.A, self.genmodel.B, self.genmodel.C, self.genmodel.E, **policy_hyperparams)

        return q_pi

    def sample_action(self):

        action = sample_action(q_pi, self.policies, self.n_control, sampling_type = 'marginal_action') #how does this work? 

        return action

    def set_starting_state_and_priors(self, starting_state):
        self.genmodel.D = self.genmodel.generate_prior_states(starting_state = starting_state)


