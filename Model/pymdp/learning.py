import numpy as np
from . import utils

def update_E(action_id, control_factor_id, policies, learning_rate = 0):

    num_policies = len(policies)
    E_increment = np.zeros(num_policies)

    # turn policies (list of arrays) into a single array
    policy_array = np.stack(policies, 0).squeeze()

    indices_to_bump = (policy_array[:,control_factor_id] == action_id)

    E_increment[indices_to_bump] = learning_rate

    return E_increment






