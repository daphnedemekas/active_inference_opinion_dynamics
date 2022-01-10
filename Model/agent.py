import numpy as np
from .genmodel_self_esteem import GenerativeModel
from .pymdp.inference import *
from .pymdp.control import *
from .pymdp.learning import *
from .pymdp.maths import spm_log
from .pymdp import utils

class Agent(object):

    def __init__(
        self,
        neighbour_params,
        idea_mapping_params,
        policy_params,
        reduce_A = True,
        reduce_A_inference = True,
        reduce_A_policies = True   
        ):            

        self.genmodel = GenerativeModel(reduce_A = reduce_A, **neighbour_params, **idea_mapping_params, **policy_params)
        self.action = np.zeros(len(self.genmodel.num_states),dtype=int)

        self.reduce_A_inference = reduce_A_inference
        self.reduce_A_policies = reduce_A_policies

        # self.set_starting_state_and_priors()
        self.inference_params = {"num_iter":10, 
                                 "dF":1.0,
                                 "dF_tol":0.001}
        self.policy_hyperparams = {"use_utility": True,
                                   "use_states_info_gain": True,
                                   "use_param_info_gain": False}
        # self.initial_action = policy_params["initial_action"]
        self.action[-2] = policy_params["initial_action"][-2]
        self.action[-1] = policy_params["initial_action"][-1]
        self.genmodel.E += update_E(self.action[1], self.genmodel.who_idx, self.genmodel.policies, learning_rate=policy_params["E_lr"])

        self.set_starting_state_and_priors()

    def infer_states(self, t, observation):
        empirical_prior = utils.obj_array(self.genmodel.num_factors)

        if t == 0:
            for f in range(self.genmodel.num_factors):
                empirical_prior[f] = spm_log(self.genmodel.D[f])

        else:
            for f, ns in enumerate(self.genmodel.num_states):

                empirical_prior[f] = spm_log(self.genmodel.B[f][:,:, int(self.action[f])].dot(self.qs[f]))
        
        if self.reduce_A_inference:
            #print(self.genmodel.informative_dims)
            #print(len(self.genmodel.informative_dims))
            qs = update_posterior_states_factorized(observation, self.genmodel.A_reduced, self.genmodel.informative_dims, self.genmodel.num_states, prior = empirical_prior, **self.inference_params)
            #qs = update_posterior_states(observation, self.genmodel.A, prior=empirical_prior, **self.inference_params)

        else:
            qs = update_posterior_states(observation, self.genmodel.A, prior=empirical_prior, **self.inference_params)

        self.qs = qs

        return qs


    def infer_policies(self):

        belief_component_E = self.genmodel.get_policy_prior(self.qs[0]) 

        neighbour_bias = spm_log(self.genmodel.E / self.genmodel.E.sum())


        posterior_E = belief_component_E + neighbour_bias

       # q_pi, neg_efe = update_posterior_policies_reduced_vectorized(self.qs, self.genmodel.A_reduced, self.genmodel.B, self.genmodel.C, posterior_E, self.genmodel.policies, self.genmodel.informative_dims, self.genmodel.reshape_dims_per_modality, self.genmodel.tile_dims_per_modality, **self.policy_hyperparams)
        q_pi, neg_efe = update_posterior_policies(self.qs, self.genmodel.A,self.genmodel.B, self.genmodel.C, self.genmodel.policies, **self.policy_hyperparams)

        self.q_pi = q_pi
        self.neg_efe = neg_efe
        return q_pi
    
    def sample_action(self):

        action = sample_action(self.q_pi, self.genmodel.policies, self.genmodel.num_states, self.genmodel.control_factor_idx, sampling_type = 'marginal_action') #how does this work? 
        self.action = action

        self.genmodel.E += update_E(action[self.genmodel.who_idx], self.genmodel.who_idx, self.genmodel.policies, self.genmodel.E_lr)

        return action

    def set_starting_state_and_priors(self):
        self.genmodel.D = self.genmodel.generate_prior_states(initial_action = self.action)


