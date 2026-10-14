import numpy as np
from scipy.sparse import csr_matrix

def value_iteration(kernel, reward_matrix, gamma=0.99, epsilon=1e-8):
    """
    Optimized Value Iteration for large state spaces.
    kernel: (S*A, S) sparse matrix
    reward_matrix: (S, A) dense matrix
    """
    num_states, num_actions = reward_matrix.shape
    # Flatten reward to (S*A, 1) for easier addition with the kernel output
    R = reward_matrix.flatten().reshape(-1, 1)
    
    V = np.zeros((num_states, 1))
    
    while True:
        # Bellman update: Q = R + gamma * P * V
        # kernel @ V results in a (S*A, 1) vector
        Q_flat = R + gamma * (kernel @ V)
        
        # Reshape to (S, A) to find max over actions
        Q = Q_flat.reshape(num_states, num_actions)
        V_next = np.max(Q, axis=1).reshape(-1, 1)
        
        # Convergence check using L-infinity norm
        if np.max(np.abs(V_next - V)) < epsilon:
            break
        V = V_next
    
    # Final Q-function extraction
    Q_final = R + gamma * (kernel @ V)
    return Q_final.reshape(num_states, num_actions)