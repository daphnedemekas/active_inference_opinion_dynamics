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
        C_params,
        starting_state = None
        ):            

        self.genmodel = GenerativeModel(**neighbour_params, **idea_mapping_params, **policy_params, **C_params)
        self.set_starting_state_and_priors(starting_state)
        self.action = np.zeros(len(self.genmodel.control_factor_idx))
        self.inference_params = {"num_iter":10, 
                                 "dF":1.0,
                                 "dF_tol":0.001}


    def infer_states(self, timestep, observation):

        if timestep == 0:
            empirical_prior = utils.obj_array(self.genmodel.num_factors)
            for f in range(self.genmodel.num_factors):
                empirical_prior[f] = spm_log(self.genmodel.D[f])
        else:
            for f, ns in enumerate(self.genmodel.num_states):
                empirical_prior[f] = self.B[f][:,:,self.action]
        
        qs = update_posterior_states(observation, self.genmodel.A, prior=empirical_prior, **self.inference_params)

        self.qs = qs
        
        return qs


    def infer_policies(self):

        self.genmodel.E = self.genmodel.get_policy_prior() 

        self.q_pi = update_posterior_policies(self.qs, self.genmodel.A, self.genmodel.B, self.genmodel.C, self.genmodel.D, self.genmodel.E, **policy_hyperparams)
        """
        def update_posterior_policies(
        qs,
        A,
        B,
        C,
        policies,
        use_utility=True,
        use_states_info_gain=True,
        use_param_info_gain=False,
        pA=None,
        pB=None,
        gamma=16.0,
        return_numpy=True,
        ):
        """
        return q_pi

    def sample_action(self):

        action = q_pi.sample() #how does this work? 

        return action

    def set_starting_state_and_priors(self, starting_state):
        self.genmodel.D = self.genmodel.generate_prior_states(starting_state = starting_state)


