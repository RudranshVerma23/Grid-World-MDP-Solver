import numpy as np
from scipy.sparse import csr_matrix

def kernel(N):
    num_states = N**4
    num_actions = 5
    
    rows = []
    cols = []
    data = []
    
    # Convert coordinates to state index
    def get_state_idx(px, py, qx, qy):
        return (px-1)*(N**3) + (py-1)*(N**2) + (qx-1)*N + (qy-1)

    # Get valid moves for the prey (including 'stay')
    def get_valid_prey_moves(qx, qy):
        moves = [(qx, qy), (qx, qy+1), (qx, qy-1), (qx-1, qy), (qx+1, qy)]
        return [m for m in moves if 1 <= m[0] <= N and 1 <= m[1] <= N]

    # Pre-calculate cell list to avoid repeated list comprehensions in loops
    all_cells = [(x, y) for x in range(1, N+1) for y in range(1, N+1)]

    for px in range(1, N+1):
        for py in range(1, N+1):
            for qx in range(1, N+1):
                for qy in range(1, N+1):
                    
                    state = get_state_idx(px, py, qx, qy)
                    
                    for a in range(num_actions):
                        row_idx = state * num_actions + a
                        
                        # 1. Deterministic Predator Move
                        npx, npy = px, py
                        if a == 1 and py < N: npy += 1    # Up
                        elif a == 2 and py > 1: npy -= 1  # Down
                        elif a == 3 and px > 1: npx -= 1  # Left
                        elif a == 4 and px < N: npx += 1  # Right
                        
                        # 2. Stochastic Prey Move
                        valid_prey_moves = get_valid_prey_moves(qx, qy)
                        prob_per_move = 1.0 / len(valid_prey_moves)
                        
                        for (nqx, nqy) in valid_prey_moves:
                            # 3. Check for Catch & Respawn
                            if (npx, npy) == (nqx, nqy):
                                # Prey caught! Respawns in any cell EXCEPT predator's new loc
                                num_respawn_cells = N**2 - 1
                                respawn_prob = prob_per_move / num_respawn_cells
                                
                                for (rx, ry) in all_cells:
                                    if (rx, ry) != (npx, npy):
                                        next_state = get_state_idx(npx, npy, rx, ry)
                                        rows.append(row_idx)
                                        cols.append(next_state)
                                        data.append(respawn_prob)
                            else:
                                # Normal move
                                next_state = get_state_idx(npx, npy, nqx, nqy)
                                rows.append(row_idx)
                                cols.append(next_state)
                                data.append(prob_per_move)
                                
    # Create the CSR matrix
    # Construction: csr_matrix((data, (rows, cols)), shape=(Total_Rows, Total_Cols))
    return csr_matrix((data, (rows, cols)), shape=(num_states * num_actions, num_states))