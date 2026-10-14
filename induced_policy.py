import numpy as np

def induced_policy(Q):
    """
    Extracts a greedy deterministic policy from a Q-function.
    
    Input:
        Q: np.ndarray of size (|S|, |A|)
        
    Output:
        policy: np.ndarray of size (|S|, |A|) where each row 
                has exactly one '1.0' at the index of the max Q-value.
    """
    num_states, num_actions = Q.shape
    
    # 1. Initialize an empty policy matrix of the same shape as Q
    policy = np.zeros_like(Q)
    
    # 2. Find the index of the maximum value for each state (row)
    best_actions = np.argmax(Q, axis=1)
    
    # 3. Set the probability of the best action to 1.0 for each state
    policy[np.arange(num_states), best_actions] = 1.0
    
    return policy