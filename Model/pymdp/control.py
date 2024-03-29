#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member
# pylint: disable=not-an-iterable

""" Functions
__author__: Conor Heins, Alexander Tschantz, Brennan Klein
"""

import itertools 
import numpy as np
from .maths import softmax, spm_dot, spm_wnorm, spm_MDP_G, spm_MDP_G_optim, spm_log
from . import utils
import copy 

def update_posterior_policies_reduced(
    qs,
    A_reduced,
    informative_dims,
    B,
    C,
    E,
    policies,
    use_utility=True,
    use_states_info_gain=True,
    use_param_info_gain=False,
    pA=None,
    pB=None,
    gamma=16.0):
    """ Updates the posterior beliefs about policies based on expected free energy prior. Uses reduced A matrix
        to speed up computation time.
        Parameters
        ----------
        - `qs` [1D numpy array, array-of-arrays, or Categorical (either single- or multi-factor)]:
            Current marginal beliefs about hidden state factors
        - `A_reduced` [numpy ndarray, array-of-arrays (in case of multiple modalities), or Categorical 
                (both single and multi-modality)]:
            Observation likelihood model (beliefs about the likelihood mapping entertained by the agent)
        - `informative_dims` [list of lists]:
            This is a list of length `num_modalities` where each sub-list contains the indices of the hidden
            state factors that have an informative relationship to observations for that modality.
        - `B` [numpy ndarray, array-of-arrays (in case of multiple hidden state factors), or Categorical 
                (both single and multi-factor)]:
                Transition likelihood model (beliefs about the likelihood mapping entertained by the agent)
        - `C` [numpy 1D-array, array-of-arrays (in case of multiple modalities), or Categorical 
                (both single and multi-modality)]:
            Prior beliefs about outcomes (prior preferences)
        - `policies` [list of tuples]:
            A list of all the possible policies, each expressed as a tuple of indices, where a given 
            index corresponds to an action on a particular hidden state factor e.g. policies[1][2] yields the 
            index of the action under policy 1 that affects hidden state factor 2
        - `use_utility` [bool]:
            Whether to calculate utility term, i.e how much expected observation confer with prior expectations
        - `use_states_info_gain` [bool]:
            Whether to calculate state information gain
        - `use_param_info_gain` [bool]:
            Whether to calculate parameter information gain @NOTE requires pA or pB to be specified 
        - `pA` [numpy ndarray, array-of-arrays (in case of multiple modalities), or Dirichlet 
                (both single and multi-modality)]:
            Prior dirichlet parameters for A. Defaults to none, in which case info gain w.r.t. Dirichlet 
            parameters over A is skipped.
        - `pB` [numpy ndarray, array-of-arrays (in case of multiple hidden state factors), or 
            Dirichlet (both single and multi-factor)]:
            Prior dirichlet parameters for B. Defaults to none, in which case info gain w.r.t. 
            Dirichlet parameters over A is skipped.
        - `gamma` [float, defaults to 16.0]:
            Precision over policies, used as the inverse temperature parameter of a softmax transformation 
            of the expected free energies of each policy
       
        Returns
        --------
        - `qp` [1D numpy array or Categorical]:
            Posterior beliefs about policies, defined here as a softmax function of the 
            expected free energies of policies
        - `efe` - [1D numpy array or Categorical]:
            The expected free energies of policies
    """
    n_policies = len(policies)
    neg_efe = np.zeros(n_policies) 
    q_pi = np.zeros((n_policies, 1))

    num_modalities = len(A_reduced)

    qo_pi = utils.obj_array(num_modalities)

    for idx, policy in enumerate(policies):
        qs_pi = get_expected_states(qs, B, policy)
        
        # initialise expected observations
        
        for g in range(num_modalities):
            if not informative_dims[g]:
                qo_pi[g] = A_reduced[g]
            else:
                qo_pi[g] = spm_dot(A_reduced[g], qs_pi[informative_dims[g]])

        if use_utility:
            neg_efe[idx] += calc_expected_utility(qo_pi, C)

        if use_states_info_gain:
            for g in range(num_modalities):
                if informative_dims[g]:
                    spm = spm_MDP_G(A_reduced[g], qs_pi[informative_dims[g]])
                    #spm_optim = spm_MDP_G_optim(A_reduced[g], qs_pi[informative_dims[g]])
                    neg_efe[idx] += spm

    """
    @TODO / @NOTE on 18 May 2021:
    We need to optimize this function to speed it up -- our options (as I see it) include the following 
    (please add to this if there are other ideas, @Daphne):
    1. Continue to use the epistemic value + utility decomposition of the (-ve) EFE, but find a way to speed up
    the inner loop of spm_MDP_G, using workarounds to spm_cross. This is the idea behind Daphne's spm_MDP_G_optim,
    but we can keep going down this rabbit hole and making it faster. I remember Dimi did something where he achieved
    essentially the same output as spm_cross by doing things like :
    output = full_array_of_arrays[0]
    for ii in range(len(full_array_of_arrays)-1):
        output = output[..., None] * full_array_of_arrays[ii + 1]
    2. Try to vectorize across policies (no loop over policies). Dimi did something like this in pomdp+utils in the Space Man repo 
    (`interactive` branch) on Magnus' github https://github.com/MagnusKoudahl/space_man/blob/d3bada1af93b3100b53d6f0cfda31fcccfdabac2/pomdp_utils.py#L324
    @NOTE: This function uses the Ambiguity + Risk decomposition, but I think the same principle could be used for surprise and utility
    3. (-ve) Expected ambiguity + (-ve) expected risk decomposition -- this should be E_q(s)[lnP(o|s)], which I think could be computed using
    ambiguity = 0
    for g in range(num_modalities):
        ambiguity += spm_dot(spm_log(A[g]), qs_pi).sum() # sum across observation levels?
    But I'm not sure how to do the risk one off the top of my head (KLD(Q_pi(o) || P(o)))
    """
    q_pi = softmax(gamma*neg_efe + E)

    q_pi = q_pi / q_pi.sum(axis=0)  # type: ignore
    
    return q_pi, neg_efe

def update_posterior_policies_reduced_vectorized(qs,
                A_reduced,
                B,
                C,
                E,
                policies,
                informative_dims,
                reshape_dims_per_modality,
                tile_dims_per_modality,
                use_utility=True,
                use_states_info_gain=True,
                use_param_info_gain=False,
                pA=None,
                pB=None,
                gamma=16.0):

    n_policies = len(policies)
    
    num_modalities = len(A_reduced)

    num_factors = len(qs)

    num_controls = [B[f].shape[2] for f in range(num_factors)]

    neg_efe = np.zeros(num_controls) 

    qs_pi = utils.obj_array(num_factors)

    for f in range(num_factors):
        qs_pi[f] = (B[f] * qs[f][..., None]).sum(-2)

    qo_pi = utils.obj_array(num_modalities)
    H_pi = utils.obj_array(num_modalities)

    for g in range(num_modalities):
        informative_qs = qs_pi[informative_dims[g]]
        qo_pi[g] = np.einsum('ij...,jl->i...l', A_reduced[g], informative_qs[0])

        H_s = - np.sum(A_reduced[g] * spm_log(A_reduced[g]), 0)
        H_pi[g] = np.einsum('j...,jl->...l', H_s, informative_qs[0])
        for qs_f in informative_qs[1:]:
            qo_pi[g] = np.einsum('ij...,jl->i...l', qo_pi[g], qs_f)
            H_pi[g] = np.einsum('j...,jl->...l', H_pi[g], qs_f)

        # calculate expected utility
        lnC = spm_log(softmax(C[g][:,np.newaxis]))
    
        neg_efe_g = np.einsum('j...,jl->...', qo_pi[g], lnC)
        neg_efe_g -= (qo_pi[g] * spm_log(qo_pi[g])).sum(axis=0)
        neg_efe_g -= H_pi[g]

        reshaped_efe_g = neg_efe_g.reshape(reshape_dims_per_modality[g])
        neg_efe += np.tile(reshaped_efe_g, tile_dims_per_modality[g])

    neg_efe = neg_efe.flatten()
    
    q_pi = softmax(gamma*neg_efe + E)

    q_pi = q_pi / q_pi.sum(axis=0)  # type: ignore
    
    return q_pi, neg_efe


def get_expected_states(qs, B, policy, return_numpy=False):
    """
    Given a posterior density qs, a transition likelihood model B, and a policy, 
    get the state distribution expected under that policy's pursuit

    Parameters
    ----------
    - `qs` [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), or 
    Categorical (either single-factor or AoA)]:
        Current posterior beliefs about hidden states
    - `B` [numpy nd-array, array-of-arrays (where each entry is a numpy nd-array), or Categorical 
        (either single-factor of AoA)]:
        Transition likelihood mapping from states at t to states at t + 1, with different actions 
        (per factor) stored along the lagging dimension
   - `policy` [numpy nd-array]:
        np.array of size (policy_len x n_factors) where each value corrresponds to a control state
    Returns
    -------
    - `qs_pi` [ list of np.arrays with len n_steps, where in case of multiple hidden state factors, 
    each np.array in the list is a 1 x n_factors array-of-arrays, otherwise a list of 1D numpy arrays]:
        Expected states under the given policy - also known as the 'posterior predictive density'

    """
    n_steps, n_factors = policy.shape
    qs = utils.to_numpy(qs, flatten=True)
    B = utils.to_numpy(B)

    # initialise beliefs over expected states
    qs_pi = []
    if utils.is_arr_of_arr(B):
        for t in range(n_steps):
            qs_pi_t = np.empty(n_factors, dtype=object)
            qs_pi.append(qs_pi_t)

        # initialise expected states after first action using current posterior (t = 0)
        for control_factor, control in enumerate(policy[0, :]):
            qs_pi[0][control_factor] = spm_dot(B[control_factor][:, :, control], qs[control_factor])

        # get expected states over time
        if n_steps > 1:
            for t in range(1, n_steps):
                for control_factor, control in enumerate(policy[t, :]):
                    qs_pi[t][control_factor] = spm_dot(
                        B[control_factor][:, :, control], qs_pi[t - 1][control_factor]
                    )
    else:
        # initialise expected states after first action using current posterior (t = 0)
        qs_pi.append(spm_dot(B[:, :, policy[0, 0]], qs))

        # then loop over future timepoints
        if n_steps > 1:
            for t in range(1, n_steps):
                qs_pi.append(spm_dot(B[:, :, policy[t, 0]], qs_pi[t - 1]))

    if len(qs_pi) == 1:
        return qs_pi[0]
    else:
        return qs_pi



def get_expected_obs(qs_pi, A, return_numpy=False):
    """
    Given a posterior predictive density Qs_pi and an observation likelihood model A,
    get the expected observations given the predictive posterior.

    Parameters
    ----------
    qs_pi [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), Categorical 
    (either single-factor or AoA), or list]:
        Posterior predictive density over hidden states. If a list, each entry of the list is the 
        posterior predictive for a given timepoint of an expected trajectory
    A [numpy nd-array, array-of-arrays (where each entry is a numpy nd-array), or Categorical 
    (either single-factor of AoA)]:
        Observation likelihood mapping from hidden states to observations, with different modalities 
        (if there are multiple) stored in different arrays
    Returns
    -------
    qo_pi [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), Categorical 
    (either single-factor or AoA), or list]:
        Expected observations under the given policy. If a list, a list of the expected observations 
        over the time horizon of policy evaluation, where
        each entry is the expected observations at a given timestep. 
    """

    # initialise expected observations
    qo_pi = []
    A = utils.to_numpy(A)

    if isinstance(qs_pi, list):
        n_steps = len(qs_pi)
        for t in range(n_steps):
            qs_pi[t] = utils.to_numpy(qs_pi[t], flatten=True)
    else:
        n_steps = 1
        qs_pi = [utils.to_numpy(qs_pi, flatten=True)]

    if utils.is_arr_of_arr(A):

        num_modalities = len(A)

        for t in range(n_steps):
            qo_pi_t = np.empty(num_modalities, dtype=object)
            qo_pi.append(qo_pi_t)

        # get expected observations over time
        for t in range(n_steps):
            for modality in range(num_modalities):
                qo_pi[t][modality] = spm_dot(A[modality], qs_pi[t])

    else:
        # get expected observations over time
        for t in range(n_steps):
            qo_pi.append(spm_dot(A, qs_pi[t]))

    if n_steps == 1:
        return qo_pi[0]
    else:
        return qo_pi



def calc_expected_utility(qo_pi, C):
    """
    Given expected observations under a policy Qo_pi and a prior over observations C
    compute the expected utility of the policy.

    Parameters
    ----------
    qo_pi [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), 
    Categorical (either single-factor or AoA), or list]:
        Expected observations under the given policy (predictive posterior over outcomes). 
        If a list, a list of the expected observations
        over the time horizon of policy evaluation, where each entry is the expected 
        observations at a given timestep. 
    C [numpy nd-array, array-of-arrays (where each entry is a numpy nd-array):
        Prior beliefs over outcomes, expressed in terms of relative log probabilities
    Returns
    -------
    expected_util [scalar]:
        Utility (reward) expected under the policy in question
    """
    if isinstance(qo_pi, list):
        n_steps = len(qo_pi)
        for t in range(n_steps):
            qo_pi[t] = utils.to_numpy(qo_pi[t], flatten=True)
    else:
        n_steps = 1
        qo_pi = [utils.to_numpy(qo_pi, flatten=True)]

    C = utils.to_numpy(C, flatten=True)

    # initialise expected utility
    expected_util = 0

    # in case of multiple observation modalities, loop over time points and modalities
    if utils.is_arr_of_arr(C):
        num_modalities = len(C)
        for t in range(n_steps):
            for modality in range(num_modalities):
                lnC = np.log(softmax(C[modality][:, np.newaxis]) + 1e-16)
                expected_util += qo_pi[t][modality].dot(lnC)

    # else, just loop over time (since there's only one modality)
    else:
        lnC = np.log(softmax(C[:, np.newaxis]) + 1e-16)
        for t in range(n_steps):
            lnC = np.log(softmax(C[:, np.newaxis] + 1e-16))
            expected_util += qo_pi[t].dot(lnC)

    return expected_util


def calc_states_info_gain(A, qs_pi):
    """
    Given a likelihood mapping A and a posterior predictive density over states Qs_pi,
    compute the Bayesian surprise (about states) expected under that policy
    Parameters
    ----------
    A [numpy nd-array, array-of-arrays (where each entry is a numpy nd-array), or 
    Categorical (either single-factor of AoA)]:
        Observation likelihood mapping from hidden states to observations, with 
        different modalities (if there are multiple) stored in different arrays
    qs_pi [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), 
    Categorical (either single-factor or AoA), or list]:
        Posterior predictive density over hidden states. If a list, each entry of 
        the list is the posterior predictive for a given timepoint of an expected trajectory
    Returns
    -------
    states_surprise [scalar]:
        Surprise (about states) expected under the policy in question
    """

    A = utils.to_numpy(A)

    if isinstance(qs_pi, list):
        n_steps = len(qs_pi)
        for t in range(n_steps):
            qs_pi[t] = utils.to_numpy(qs_pi[t], flatten=True)
    else:
        n_steps = 1
        qs_pi = [utils.to_numpy(qs_pi, flatten=True)]

    states_surprise = 0
    for t in range(n_steps):
        spm_optim = spm_MDP_G_optim(A, qs_pi[t])
        # spm_old   = spm_MDP_G(A, qs_pi[t])
        states_surprise += spm_optim

    return states_surprise


def calc_pA_info_gain(pA, qo_pi, qs_pi):
    """
    Compute expected Dirichlet information gain about parameters pA under a policy
    Parameters
    ----------
    pA [numpy nd-array, array-of-arrays (where each entry is a numpy nd-array), or 
    Dirichlet (either single-factor of AoA)]:
        Prior dirichlet parameters parameterizing beliefs about the likelihood 
        mapping from hidden states to observations, 
        with different modalities (if there are multiple) stored in different arrays.
    qo_pi [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array),
     Categorical (either single-factor or AoA), or list]:
        Expected observations. If a list, each entry of the list is the posterior 
        predictive for a given timepoint of an expected trajectory
    qs_pi [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), 
    Categorical (either single-factor or AoA), or list]:
        Posterior predictive density over hidden states. If a list, each entry of 
        the list is the posterior predictive for a given timepoint of an expected trajectory
    Returns
    -------
    infogain_pA [scalar]:
        Surprise (about dirichlet parameters) expected under the policy in question
    """

    if isinstance(qo_pi, list):
        n_steps = len(qo_pi)
        for t in range(n_steps):
            qo_pi[t] = utils.to_numpy(qo_pi[t], flatten=True)
    else:
        n_steps = 1
        qo_pi = [utils.to_numpy(qo_pi, flatten=True)]

    if isinstance(qs_pi, list):
        for t in range(n_steps):
            qs_pi[t] = utils.to_numpy(qs_pi[t], flatten=True)
    else:
        n_steps = 1
        qs_pi = [utils.to_numpy(qs_pi, flatten=True)]

    
    if utils.is_arr_of_arr(pA):
        num_modalities = len(pA)
        wA = np.empty(num_modalities, dtype=object)
        for modality in range(num_modalities):
            wA[modality] = spm_wnorm(pA[modality])
    else:
        num_modalities = 1
        wA = spm_wnorm(pA)

    pA = utils.to_numpy(pA)
    pA_infogain = 0
    if num_modalities == 1:
        wA = wA * (pA > 0).astype("float")
        for t in range(n_steps):
            pA_infogain = -qo_pi[t].dot(spm_dot(wA, qs_pi[t])[:, np.newaxis])
    else:
        for modality in range(num_modalities):
            wA_modality = wA[modality] * (pA[modality] > 0).astype("float")
            for t in range(n_steps):
                pA_infogain -= qo_pi[t][modality].dot(spm_dot(wA_modality, qs_pi[t])[:, np.newaxis])

    return pA_infogain


def calc_pB_info_gain(pB, qs_pi, qs_prev, policy):
    """
    Compute expected Dirichlet information gain about parameters pB under a given policy
    Parameters
    ----------
    pB [numpy nd-array, array-of-arrays (where each entry is a numpy nd-array), 
    or Dirichlet (either single-factor of AoA)]:
        Prior dirichlet parameters parameterizing beliefs about the likelihood 
        describing transitions bewteen hidden states,
        with different factors (if there are multiple) stored in different arrays.
    qs_pi [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), 
    Categorical (either single-factor or AoA), or list]:
        Posterior predictive density over hidden states. If a list, each entry of 
        the list is the posterior predictive for a given timepoint of an expected trajectory
    qs_prev [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), 
    or Categorical (either single-factor or AoA)]:
        Posterior over hidden states (before getting observations)
    policy [numpy 2D ndarray, of size n_steps x n_control_factors]:
        Policy to consider. Each row of the matrix encodes the action index 
        along a different control factor for a given timestep.  
    Returns
    -------
    infogain_pB [scalar]:
        Surprise (about dirichlet parameters) expected under the policy in question
    """

    if isinstance(qs_pi, list):
        n_steps = len(qs_pi)
        for t in range(n_steps):
            qs_pi[t] = utils.to_numpy(qs_pi[t], flatten=True)
    else:
        n_steps = 1
        qs_pi = [utils.to_numpy(qs_pi, flatten=True)]

    if utils.is_arr_of_arr(pB):
        num_factors = len(pB)
        wB = np.empty(num_factors, dtype=object)
        for factor in range(num_factors):
            wB[factor] = spm_wnorm(pB[factor])
    else:
        num_factors = 1
        wB = spm_wnorm(pB)

    pB = utils.to_numpy(pB)
    pB_infogain = 0
    if num_factors == 1:
        for t in range(n_steps):
            if t == 0:
                previous_qs = qs_prev
            else:
                previous_qs = qs_pi[t - 1]
            a_i = policy[t, 0]
            wB_t = wB[:, :, a_i] * (pB[:, :, a_i] > 0).astype("float")
            pB_infogain = -qs_pi[t].dot(wB_t.dot(qs_prev))
    else:

        for t in range(n_steps):
            # the 'past posterior' used for the information gain about pB here is the posterior
            # over expected states at the timestep previous to the one under consideration
            # if we're on the first timestep, we just use the latest posterior in the
            # entire action-perception cycle as the previous posterior
            if t == 0:
                previous_qs = qs_prev
            # otherwise, we use the expected states for the timestep previous to the timestep under consideration
            else:
                previous_qs = qs_pi[t - 1]

            # get the list of action-indices for the current timestep
            policy_t = policy[t, :]
            for factor, a_i in enumerate(policy_t):
                wB_factor_t = wB[factor][:, :, a_i] * (pB[factor][:, :, a_i] > 0).astype("float")
                pB_infogain -= qs_pi[t][factor].dot(wB_factor_t.dot(previous_qs[factor]))

    return pB_infogain


def construct_policies(n_states, n_control=None, policy_len=1, control_fac_idx=None):
    """Generate a set of policies

    Each policy is encoded as a numpy.ndarray of shape (n_steps, n_factors), where each 
    value corresponds to
    the index of an action for a given time step and control factor. The variable 
    `policies` that is returned
    is a list of each policy-specific numpy nd.array.

    @NOTE: If the list of control state dimensionalities (`n_control`) is not provided, 
    `n_control` defaults to being equal to n_states, except for the indices 
    provided by control_fac_idx, where
    the value of n_control for the indicated factor is 1.

    @TODO think about returning n_control - required arg
    Arguments:
    -------
    - `n_states`: list of dimensionalities of hidden state factors
    - `n_control`: list of dimensionalities of control state factors 
    - `policy_len`: temporal length ('horizon') of policies
    - `control_fac_idx`: list of indices of the hidden state factors 
    that are controllable (i.e. those whose n_control[i] > 1)

    Returns:
    -------
    - `policies`: list of np.ndarrays, where each array within the list is a 
                    numpy.ndarray of shape (n_steps, n_factors).
                Each value in a policy array corresponds to the index of an action for 
                a given timestep and control factor.
    - `n_control`: list of dimensionalities of actions along each hidden 
                    state factor (i.e. control state dimensionalities). 
                The dimensionality of control states whose index is not in control_fac_idx is set to 1.
                This is only returned when n_control is not provided as argument.
    """

    n_factors = len(n_states)
    if control_fac_idx is None:
        control_fac_idx = list(range(n_factors))
    return_n_control = False

    if n_control is None:
        return_n_control = True
        n_control = []
        for c_idx in range(n_factors):
            if c_idx in control_fac_idx:
                n_control.append(n_states[c_idx])
            else:
                n_control.append(1)
        n_control = list(np.array(n_control).astype(int))
    x = n_control * policy_len
    policies = list(itertools.product(*[list(range(i)) for i in x]))
    if policy_len > 1:
        for pol_i in range(len(policies)):
            policies[pol_i] = np.array(policies[pol_i]).reshape(policy_len, n_factors)
    else:
        for pol_i in range(len(policies)):
            policies[pol_i] = np.array(policies[pol_i]).reshape(1, n_factors)

    if return_n_control:
        return policies, n_control
    else:
        return policies

def sample_action(q_pi, policies, n_states, control_indices, sampling_type="marginal_action", alpha = 1.0):
    """
    Samples action from posterior over policies, using one of two methods. 
    Parameters
    ----------
    `q_pi` [1D numpy.ndarray or Categorical]:
        Posterior beliefs about (possibly multi-step) policies.
    `policies` [list of numpy ndarrays]:
        List of arrays that indicate the policies under consideration. Each element 
        within the list is a matrix that stores the 
        the indices of the actions  upon the separate hidden state factors, at 
        each timestep (n_step x n_states)
    `n_states` [list of integers]:
        List of the dimensionalities of the different (controllable)) hidden state factors
    `sampling_type` [string, 'marginal_action' or 'posterior_sample']:
        Indicates whether the sampled action for a given hidden state factor is given by 
        the evidence for that action, marginalized across different policies ('marginal_action')
        or simply the action entailed by a sample from the posterior over policies
    `alpha` [Float]:
        Inverse temperature / precision parameter of action sampling in case that
        `sampling_type` == "marginal_action"
    Returns
    ----------
    selected_policy [1D numpy ndarray]:
        Numpy array containing the indices of the actions along each control factor
    """
    control_factors = [n_states[i] for i in control_indices]
    n_factors = len(n_states)
    n_control_factors = len(control_factors)
    if sampling_type == "marginal_action":

        action_marginals = utils.obj_array(n_factors)
        for f_idx, f_dim in enumerate(n_states):
            action_marginals[f_idx] = np.zeros(f_dim)

        # weight each action according to its integrated posterior probability over policies and timesteps
        for pol_idx, policy in enumerate(policies):
            for t in range(policy.shape[0]):
                for factor_i, action_i in enumerate(policy[t, :]):

                    action_marginals[factor_i][action_i] += q_pi[pol_idx]

        #print("Action marginals")
        selected_policy = np.zeros(n_factors)
        for factor_i in control_indices:
                # # selected_policy[factor_i] = utils.sample(softmax(alpha*action_marginals[factor_i]))
                # selected_policy[factor_i] = utils.sample(action_marginals[factor_i])
            # else:
            selected_policy[factor_i] = np.argmax(action_marginals[factor_i])

            #selected_policy[factor_i] = np.where(np.random.multinomial(1,action_marginals[factor_i]))[0][0]
            #selected_policy[factor_i] = np.argmax(action_marginals[factor_i])
            #selected_policy[factor_i] = utils.sample(softmax(alpha*action_marginals[factor_i]))

    elif sampling_type == "posterior_sample":
        
        policy_index = utils.sample(q_pi)
        selected_policy = policies[policy_index]

    else:
        raise ValueError(f"{sampling_type} not supported")

    return selected_policy

# def sample_action(q_pi, policies, n_states, sampling_type="marginal_action", alpha = 1.0):
#     """
#     Samples action from posterior over policies, using one of two methods. 
#     Parameters
#     ----------
#     `q_pi` [1D numpy.ndarray or Categorical]:
#         Posterior beliefs about (possibly multi-step) policies.
#     `policies` [list of numpy ndarrays]:
#         List of arrays that indicate the policies under consideration. Each element 
#         within the list is a matrix that stores the 
#         the indices of the actions  upon the separate hidden state factors, at 
#         each timestep (n_step x n_states)
#     `n_states` [list of integers]:
#         List of the dimensionalities of the different (controllable)) hidden state factors
#     `sampling_type` [string, 'marginal_action' or 'posterior_sample']:
#         Indicates whether the sampled action for a given hidden state factor is given by 
#         the evidence for that action, marginalized across different policies ('marginal_action')
#         or simply the action entailed by a sample from the posterior over policies
#     `alpha` [Float]:
#         Inverse temperature / precision parameter of action sampling in case that
#         `sampling_type` == "marginal_action"
#     Returns
#     ----------
#     selected_policy [1D numpy ndarray]:
#         Numpy array containing the indices of the actions along each control factor
#     """

#     n_factors = len(n_states)

#     if sampling_type == "marginal_action":

#         action_marginals = utils.obj_array(n_factors)
#         for f_idx, f_dim in enumerate(n_states):
#             action_marginals[f_idx] = np.zeros(f_dim)

#         # weight each action according to its integrated posterior probability over policies and timesteps
#         for pol_idx, policy in enumerate(policies):
#             for t in range(policy.shape[0]):
#                 for factor_i, action_i in enumerate(policy[t, :]):
#                     action_marginals[factor_i][action_i] += q_pi[pol_idx]

#         selected_policy = np.zeros(n_factors)
#         for factor_i in range(n_factors):
#             # selected_policy[factor_i] = np.where(np.random.multinomial(1,action_marginals[factor_i]))[0][0]
#             selected_policy[factor_i] = np.argmax(action_marginals[factor_i])
#             # selected_policy[factor_i] = utils.sample(softmax(alpha*action_marginals[factor_i]))

#     elif sampling_type == "posterior_sample":
        
#         policy_index = utils.sample(q_pi)
#         selected_policy = policies[policy_index]

#     else:
#         raise ValueError(f"{sampling_type} not supported")

#     return selected_policy
