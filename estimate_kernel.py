import numpy as np
from scipy.sparse import csr_matrix
from simulator import simulator

def estimate_kernel(N, K):
    """
    Estimates the state-transition kernel using sparse storage.
    """
    num_states = N**4
    num_actions = 5
    
    rows = []
    cols = []
    data = []
    
    # Probability weight of a single sample
    prob_increment = 1.0 / K

    # Convert coordinates to state index
    def get_state_idx(px, py, qx, qy):
        return (px-1)*N**3 + (py-1)*N**2 + (qx-1)*N + (qy-1)

    # Decode index back to coordinates
    def decode_state(idx):
        px = (idx // N**3) + 1
        py = ((idx % N**3) // N**2) + 1
        qx = ((idx % N**2) // N) + 1
        qy = (idx % N) + 1
        return (px, py), (qx, qy)

    # Iterate through every possible state s
    for s_idx in range(num_states):
        pred_loc, prey_loc = decode_state(s_idx)
        
        for a in range(num_actions):
            row_idx = s_idx * num_actions + a
            
            # Sample the simulator K times
            for _ in range(K):
                next_p, next_q, _ = simulator(N, pred_loc, prey_loc, a)
                next_s_idx = get_state_idx(next_p[0], next_p[1], next_q[0], next_q[1])
                
                rows.append(row_idx)
                cols.append(next_s_idx)
                data.append(prob_increment)
            
    # The CSR constructor automatically sums values at identical (row, col) indices
    return csr_matrix((data, (rows, cols)), shape=(num_states * num_actions, num_states))