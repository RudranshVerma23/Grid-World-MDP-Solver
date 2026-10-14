import numpy as np

def policy_iteration(kernel, reward_matrix, gamma=0.99, epsilon=1e-8):
    """
    Policy Iteration using iterative policy evaluation.
    """
    num_states, num_actions = reward_matrix.shape
    # Initialize with a random deterministic policy (e.g., action 0)
    policy_indices = np.zeros(num_states, dtype=int)
    V = np.zeros((num_states, 1))
    
    while True:
        # --- 1. Iterative Policy Evaluation ---
        # Instead of inversion, we iterate until V converges for the fixed policy
        while True:
            # Extract transition and reward for the current policy
            # We map state s to index s*A + policy[s]
            row_indices = np.arange(num_states) * num_actions + policy_indices
            
            # P_pi is (S, S), R_pi is (S, 1)
            P_pi = kernel[row_indices, :]
            R_pi = reward_matrix[np.arange(num_states), policy_indices].reshape(-1, 1)
            
            V_next = R_pi + gamma * (P_pi @ V)
            if np.max(np.abs(V_next - V)) < epsilon:
                break
            V = V_next

        # --- 2. Policy Improvement ---
        # Compute Q for all actions to see if we can improve
        Q_flat = reward_matrix.flatten().reshape(-1, 1) + gamma * (kernel @ V)
        Q = Q_flat.reshape(num_states, num_actions)
        
        new_policy_indices = np.argmax(Q, axis=1)
        
        # Check if policy has stabilized
        if np.array_equal(new_policy_indices, policy_indices):
            break
        
        policy_indices = new_policy_indices
        
    return Q