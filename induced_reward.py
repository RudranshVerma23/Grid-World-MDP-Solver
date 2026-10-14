import numpy as np

def induced_reward(reward_function, policy):
    """
    Generates the policy-induced reward function R_pi.
    
    Inputs:
        reward_function: np.ndarray of size (|S|, |A|)
        policy: np.ndarray of size (|S|, |A|)
        
    Output:
        induced_R: np.ndarray of size (|S|, 1)
    """
    # 1. Element-wise multiplication: pi(a|s) * R(s, a)
    # 2. Sum along axis 1 (the actions) to get the expected reward per state
    induced_R = np.sum(policy * reward_function, axis=1)
    
    # 3. Reshape to ensure it is (|S|, 1) as requested
    return induced_R.reshape(-1, 1)