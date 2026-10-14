from scipy.sparse import csr_matrix
import numpy as np

def induced_kernel(kernel, policy):
    num_states, num_actions = policy.shape
    
    # 1. Prepare indices for the "Selector Matrix" M
    # Each state s will correspond to a row in M
    # Each (s, a) pair corresponds to a column (s * num_actions + a)
    row_indices = np.repeat(np.arange(num_states), num_actions)
    col_indices = np.arange(num_states * num_actions)
    data = policy.flatten()
    
    # 2. Create the Selector Matrix M
    M = csr_matrix((data, (row_indices, col_indices)), 
                   shape=(num_states, num_states * num_actions))
    
    # 3. The induced kernel is simply the matrix product
    induced_P = M @ kernel
    
    return induced_P