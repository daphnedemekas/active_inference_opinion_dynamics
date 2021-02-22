import numpy as np
from genmodel import GenerativeModel
from pymdp.inference import update_posterior_states
from pymdp.control import *

class Agent(object):

    def __init__(
        self,
        neighbour_params,
        idea_mapping_params,
        policy_params,
        C_params
        ):            

        self.genmodel = GenerativeModel(**neighbour_params, **idea_mapping_params, **policy_params, **C_params)

        def infer_states(self, observation):

            self.qs = update_posterior_states(self.genmodel.A, self.genmodel.B, observation, self.genmodel.D, **inference_hyperparams)
            #update_posterior_states(obs, A, prior=None, return_numpy=True, **kwargs)

            """ 
            def update_posterior_states_v2(
            A,
            B,
            prev_obs,
            policies,
            prev_actions=None,
            prior=None,
            return_numpy=True,
            policy_sep_prior = True,
            **kwargs,
            ):
            """
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

        def sample_action(self)

            action = q_pi.sample() #how does this work? 

            return action

        def set_starting_state_and_priors(starting_state)
            self.genmodel.D = self.genmodel.generate_prior_states(starting_state)


