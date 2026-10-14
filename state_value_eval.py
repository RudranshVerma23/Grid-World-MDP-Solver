import numpy as np
from induced_kernel import induced_kernel
from induced_reward import induced_reward

import numpy as np
from scipy.sparse import eye # Sparse Identity Matrix

def state_value_eval(policy, kernel, reward_function, gamma=0.99, threshold=1e-10):
    """
    Computes the state-value function V_pi using iterative sparse multiplication.
    """
    num_states = policy.shape[0]
    
    # 1. Precompute induced dynamics
    # Ensure induced_kernel returns a CSR sparse matrix
    P_pi = induced_kernel(kernel, policy)   
    R_pi = induced_reward(reward_function, policy) 
    
    if R_pi.ndim == 1:
        R_pi = R_pi.reshape(-1, 1)

    V = np.zeros((num_states, 1))
    
    # 3. Iterative Policy Evaluation
    while True:
        # Using @ instead of np.dot for Sparse-Dense compatibility
        V_next = R_pi + gamma * (P_pi @ V)
        
        diff = np.max(np.abs(V_next - V))
        V = V_next
        
        if diff < threshold:
            break
            
    return V