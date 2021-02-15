import numpy as np

def obj_array(shape):
    return np.empty(shape, dtype=object)

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
    if is_arr_of_arr(x):
        dims = (np.arange(0, len(x)) + X.ndim - len(x)).astype(int)
    else:
        dims = np.array([1], dtype=int)
        x = to_arr_of_arr(x)

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
        X = np.sum(X, axis=dims[d], keepdims=True)
    Y = np.squeeze(X)

    # check to see if `Y` is a scalar
    if np.prod(Y.shape) <= 1.0:
        Y = Y.item()
        Y = np.array([Y]).astype("float64")

    return Y

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

def softmax(dist):
    """ 
    Computes the softmax function on a set of values, either a straight numpy
    1-D vector or an array-of-arrays.
    """
    if is_arr_of_arr(dist):
        output = obj_array(len(dist))
        for i in range(len(dist)):
            output[i] = softmax(dist[i])
        return output
    else:
        output = dist - dist.max(axis=0)
        output = np.exp(output)
        output = output / np.sum(output, axis=0)
        return output

def onehot(value, num_values):
    arr = np.zeros(num_values)
    arr[value] = 1.0
    return arr

def is_arr_of_arr(arr):
    return arr.dtype == "object"

def to_arr_of_arr(arr):
    if is_arr_of_arr(arr):
        return arr
    arr_of_arr = np.empty(1, dtype=object)
    arr_of_arr[0] = arr.squeeze()
    return arr_of_arr

def insert_multiple(s, indices, items):
    for idx in range(len(items)):
        s.insert(indices[idx], items[idx])
    return s
