""" Functions
__author__: Conor Heins, Alexander Tschantz, Brennan Klein
"""


import numpy as np
from . import utils
from scipy import special


EPS_VAL = 1e-16 # global constant for use in spm_log() function

def spm_dot(X, x, dims_to_omit=None):
    """ Dot product of a multidimensional array with `x`. The dimensions in `dims_to_omit` 
    will not be summed across during the dot product
    
    Parameters
    ----------
    - `x` [1D numpy.ndarray] - either vector or array of arrays
        The alternative array to perform the dot product with
    - `dims_to_omit` [list :: int] (optional)
        Which dimensions to omit
    
    Returns 
    -------
    - `Y` [1D numpy.ndarray] - the result of the dot product
    """

    # Construct dims to perform dot product on
    if utils.is_arr_of_arr(x):
        dims = (np.arange(0, len(x)) + X.ndim - len(x)).astype(int)
    else:
        dims = np.array([1], dtype=int)
        x = utils.to_arr_of_arr(x)

    # delete ignored dims
    if dims_to_omit is not None:
        if not isinstance(dims_to_omit, list):
            raise ValueError("`dims_to_omit` must be a `list` of `int`")
        dims = np.delete(dims, dims_to_omit)
        if len(x) == 1:
            x = np.empty([0], dtype=object)
        else:
            x = np.delete(x, dims_to_omit)

    # compute dot product
    for d in range(len(x)):
        s = np.ones(np.ndim(X), dtype=int)
        s[dims[d]] = np.shape(x[d])[0]
        X = X * x[d].reshape(tuple(s))
        # X = np.sum(X, axis=dims[d], keepdims=True)

    Y = np.sum(X, axis=tuple(dims.astype(int))).squeeze()
    # Y = np.squeeze(X)

    # check to see if `Y` is a scalar
    if np.prod(Y.shape) <= 1.0:
        Y = Y.item()
        Y = np.array([Y]).astype("float64")

    return Y

def spm_dot_old(X, x, dims_to_omit=None, obs_mode=False):
    """ Dot product of a multidimensional array with `x`. The dimensions in `dims_to_omit` 
    will not be summed across during the dot product
    #TODO: we should look for an alternative to obs_mode
    
    Parameters
    ----------
    - `x` [1D numpy.ndarray] - either vector or array of arrays
        The alternative array to perform the dot product with
    - `dims_to_omit` [list :: int] (optional)
        Which dimensions to omit
    
    Returns 
    -------
    - `Y` [1D numpy.ndarray] - the result of the dot product
    """

    # Construct dims to perform dot product on
    if utils.is_arr_of_arr(x):
        dims = (np.arange(0, len(x)) + X.ndim - len(x)).astype(int)
    else:
        if obs_mode is True:
            """
            @NOTE Case when you're getting the likelihood of an observation under 
                  the generative model. Equivalent to something like self.values[np.where(x),:]
                  when `x` is a discrete 'one-hot' observation vector
            """
            dims = np.array([0], dtype=int)
        else:
            """
            @NOTE Case when `x` leading dimension matches the lagging dimension of `values`
                  E.g. a more 'classical' dot product of a likelihood with hidden states
            """
            dims = np.array([1], dtype=int)

        x = utils.to_arr_of_arr(x)

    # delete ignored dims
    if dims_to_omit is not None:
        if not isinstance(dims_to_omit, list):
            raise ValueError("`dims_to_omit` must be a `list` of `int`")
        dims = np.delete(dims, dims_to_omit)
        if len(x) == 1:
            x = np.empty([0], dtype=object)
        else:
            x = np.delete(x, dims_to_omit)

    # compute dot product
    for d in range(len(x)):
        s = np.ones(np.ndim(X), dtype=int)
        s[dims[d]] = np.shape(x[d])[0]
        X = X * x[d].reshape(tuple(s))
        # X = np.sum(X, axis=dims[d], keepdims=True)

    Y = np.sum(X, axis=tuple(dims.astype(int))).squeeze()
    # Y = np.squeeze(X)

    # check to see if `Y` is a scalar
    if np.prod(Y.shape) <= 1.0:
        Y = Y.item()
        Y = np.array([Y]).astype("float64")

    return Y


def spm_cross(x, y=None, *args):
    """ Multi-dimensional outer product
    
    Parameters
    ----------
    - `x` [np.ndarray] || [Categorical] (optional)
        The values to perfrom the outer-product with. If empty, then the outer-product 
        is taken between x and itself. If y is not empty, then outer product is taken 
        between x and the various dimensions of y.
    - `args` [np.ndarray] || [Categorical] (optional)
        Remaining arrays to perform outer-product with. These extra arrays are recursively 
        multiplied with the 'initial' outer product (that between X and x).
    
    Returns
    -------
    - `z` [np.ndarray] || [Categorical]
          The result of the outer-product
    """

    if len(args) == 0 and y is None:
        if utils.is_arr_of_arr(x):
            z = spm_cross(*list(x))
        elif np.issubdtype(x.dtype, np.number):
            z = x
        else:
            raise ValueError(f"Invalid input to spm_cross ({x})")
        return z

    if utils.is_arr_of_arr(x):
        x = spm_cross(*list(x))

    if y is not None and utils.is_arr_of_arr(y):
        y = spm_cross(*list(y))

    reshape_dims = tuple(list(x.shape) + list(np.ones(y.ndim, dtype=int)))
    A = x.reshape(reshape_dims)

    reshape_dims = tuple(list(np.ones(x.ndim, dtype=int)) + list(y.shape))
    B = y.reshape(reshape_dims)
    z = np.squeeze(A * B)
    for x in args:
        z = spm_cross(z, x)
    return z

def sparse_cross(x,y, *args):
    #reshape_dims = tuple(list(x.shape) + list(np.ones(y.ndim, dtype=int)))
    #A = x.reshape(reshape_dims)

    #reshape_dims = tuple(list(np.ones(x.ndim, dtype=int)) + list(y.shape))
    #B = y.reshape(reshape_dims)
    if len(args) == 0 and y is None:
        if utils.is_arr_of_arr(x):
            z = spm_cross(*list(x))
        elif np.issubdtype(x.dtype, np.number):
            z = x
        else:
            raise ValueError(f"Invalid input to spm_cross ({x})")
        return z
    reshape_dims = tuple(list(x.shape) + list(np.ones(y.ndim, dtype=int)))
    A = x.reshape(reshape_dims)

    reshape_dims = tuple(list(np.ones(x.ndim, dtype=int)) + list(y.shape))
    B = y.reshape(reshape_dims)
    z = A*B
    
    for x in args:
        z = spm_cross(z, x)
    return z



def dot_likelihood(A,obs):

    s = np.ones(np.ndim(A), dtype = int)
    s[0] = obs.shape[0]
    X = A * obs.reshape(tuple(s))
    X = np.sum(X, axis=0, keepdims=True)
    LL = np.squeeze(X)

    # check to see if `LL` is a scalar
    if np.prod(LL.shape) <= 1.0:
        LL = LL.item()
        LL = np.array([LL]).astype("float64")

    return LL


def get_joint_likelihood(A, obs, num_states):
    # deal with single modality case
    if type(num_states) is int:
        num_states = [num_states]
    A = utils.to_arr_of_arr(A)
    obs = utils.to_arr_of_arr(obs)
    ll = np.ones(tuple(num_states))
    for modality in range(len(A)):
        ll = ll * dot_likelihood(A[modality], obs[modality])
    return ll


def get_joint_likelihood_seq(A, obs, num_states):
    ll_seq = np.empty(len(obs), dtype=object)
    for t in range(len(obs)):
        ll_seq[t] = get_joint_likelihood(A, obs[t], num_states)
    return ll_seq


def spm_norm(A):
    """ 
    Returns normalization of Categorical distribution, 
    stored in the columns of A.
    """
    A = A + EPS_VAL
    normed_A = np.divide(A, A.sum(axis=0))
    return normed_A

def spm_log(arr):
    """
    Adds small epsilon value to an array before natural logging it
    """
    return np.log(arr + EPS_VAL)


def spm_wnorm(A):
    """ 
    Returns Expectation of logarithm of Dirichlet parameters over a set of 
    Categorical distributions, stored in the columns of A.
    """
    A = A + EPS_VAL
    norm = np.divide(1.0, np.sum(A, axis=0))
    avg = np.divide(1.0, A)
    wA = norm - avg
    return wA


def spm_betaln(z):
    """ Log of the multivariate beta function of a vector.
     @NOTE this function computes across columns if `z` is a matrix
    """
    return np.sum(special.gammaln(z), axis=0) - special.gammaln(np.sum(z, axis=0))


def softmax(dist):
    """ Computes the softmax function on a set of values
    """
    output = dist - dist.max(axis=0)
    output = np.exp(output)
    output = output / np.sum(output, axis=0)
    return output
  

def calc_free_energy(qs, prior, n_factors, likelihood=None):
    """ Calculate variational free energy
    @TODO Primarily used in FPI algorithm, needs to be made general
    """
    free_energy = 0
    for factor in range(n_factors):
        # Neg-entropy of posterior marginal H(q[f])
        negH_qs = qs[factor].dot(np.log(qs[factor][:, np.newaxis] + 1e-16))
        # Cross entropy of posterior marginal with prior marginal H(q[f],p[f])
        xH_qp = -qs[factor].dot(prior[factor][:, np.newaxis])
        free_energy += negH_qs + xH_qp

    if likelihood is not None:
        accuracy = spm_dot(likelihood, qs)[0]
        free_energy -= accuracy
    return free_energy

def spm_MDP_G(A, x, is_test = False):
    # Probability distribution over the hidden causes: i.e., Q(x)

    #A = agent.genmodel.reduced_A
    _, _, Ng, _ = utils.get_model_dimensions(A=A)

    qx = spm_cross(x)
    G = 0
    qo = 0
    idx = np.array(np.where(qx > np.exp(-16))).T

    if utils.is_arr_of_arr(A):
        # Accumulate expectation of entropy: i.e., E[lnP(o|x)]
        for i in idx:
            nonzeros = []
            shape = []
            # Probability over outcomes for this combination of causes
            po = np.ones(1)
            po_test = np.ones(1)
            for g in range(Ng):
                index_vector = [slice(0, A[g].shape[0])] + list(i)
                ag = (A[g][tuple(index_vector)])
                nonzeros.append(np.nonzero(ag))
                shape.append(ag.shape[0])
                p_nonzero = np.nonzero(po)
                einsum = np.array(np.einsum('i,j->ij', po[p_nonzero], ag).flatten())
                po = np.array(einsum.flatten())
                
                #if is_test:
                 #   po_test = spm_cross(po_test, A[g][tuple(index_vector)])

            indexlength = len(po)
            my_indices = []
            #print(nonzeros)
            #print(np.array(nonzeros).flatten().squeeze())
            #construct the indices with which to place the values
            
            #[0],[1,2],[1,2,3],[1,2,3,4,5,6]
            #[0,0,0,0,0,0],[1,1,1,2,2,2],[1,2,3,1,2,3],[1,2,3,4,5,6]
            print("TEST")
            print([np.repeat(nz[0], int(indexlength/len(nz[0]))) for nz in nonzeros])
            
            for nz in nonzeros:
                nz = list(nz[0])
                num_zeros = len(nz)
                if num_zeros == 1:
                    #print(np.array(list(nz)*indexlength))
                    my_indices.append(np.array(nz*indexlength))
                    #here we have only one nonzero value
                    #do 2,2,2,2,2 for example
                elif num_zeros == 2:
                    my_indices.append(np.array([nz[0]]*int(indexlength/2) + [nz[1]]*int(indexlength/2)))
                    #here we have 2 non zero values 
                    #do 1,1,1,2,2,2
                elif num_zeros == indexlength/2:
                    my_indices.append(np.array(nz*2))
                    #here we have exactly 1/2 of the nonzero valeus
                    #do 1,2,3,1,2,3
                elif num_zeros == indexlength:
                    my_indices.append(np.array(nz))
                    #here we have exactly the length so just becomes itself
                else:
                    print("not taking into account the case of " + str(length) + " zeros")
                    raise
            print("INDICES")
            print(my_indices)
            po_full = np.zeros(tuple(shape))
            po_full[tuple(my_indices)] = po

            #if is_test:
            #    if not np.array_equal(po_full, po_test):
            #        print("spm_MDP_G is not outputting the correct probability over outcomes. Maybe use spm_MDP_old instead.")
            #        raise

            po = (po_full).ravel()
            qo += qx[tuple(i)] * po
            G += qx[tuple(i)] * po.dot(np.log(po + np.exp(-16)))

    else:
        for i in idx:
            po = np.ones(1)
            index_vector = [slice(0, A.shape[0])] + list(i)
            po = spm_cross(po, A[tuple(index_vector)])
            po = po.ravel()
            qo += qx[tuple(i)] * po
            G += qx[tuple(i)] * po.dot(np.log(po + np.exp(-16)))
    # Subtract negative entropy of expectations: i.e., E[lnQ(o)]

    G = G - qo.dot(spm_log(qo))
    return G





def spm_MDP_G_old(A, x):
    """
    Calculates the Bayesian surprise in the same way as spm_MDP_G.m does in 
    the original matlab code.
    
    Parameters
    ----------
    A (numpy ndarray or array-object):
        array assigning likelihoods of observations/outcomes under the various 
        hidden state configurations
    
    x (numpy ndarray or array-object):
        Categorical distribution presenting probabilities of hidden states 
        (this can also be interpreted as the predictive density over hidden 
        states/causes if you're calculating the expected Bayesian surprise)
        
    Returns
    -------
    G (float):
        the (expected or not) Bayesian surprise under the density specified by x --
        namely, this scores how much an expected observation would update beliefs 
        about hidden states x, were it to be observed. 
    """

    # if A.dtype == "object":
    #     Ng = len(A)
    #     AOA_flag = True
    # else:
    #     Ng = 1
    #     AOA_flag = False

    _, _, Ng, _ = utils.get_model_dimensions(A=A)

    # Probability distribution over the hidden causes: i.e., Q(x)
    qx = spm_cross(x)
    G = 0
    qo = 0
    idx = np.array(np.where(qx > np.exp(-16))).T

    if utils.is_arr_of_arr(A):
        # Accumulate expectation of entropy: i.e., E[lnP(o|x)]
        for i in idx:
            # Probability over outcomes for this combination of causes
            po = np.ones(1)
            for g in range(Ng):
                index_vector = [slice(0, A[g].shape[0])] + list(i)
                po = spm_cross(po, A[g][tuple(index_vector)])

            po = po.ravel()
            qo += qx[tuple(i)] * po
            G += qx[tuple(i)] * po.dot(np.log(po + np.exp(-16)))
    else:
        for i in idx:
            po = np.ones(1)
            index_vector = [slice(0, A.shape[0])] + list(i)
            po = spm_cross(po, A[tuple(index_vector)])
            po = po.ravel()
            qo += qx[tuple(i)] * po
            G += qx[tuple(i)] * po.dot(np.log(po + np.exp(-16)))

    # Subtract negative entropy of expectations: i.e., E[lnQ(o)]
    # G = G - qo.dot(np.log(qo + np.exp(-16)))  # type: ignore
    G = G - qo.dot(spm_log(qo))  # type: ignore

    return G