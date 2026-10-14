import numpy as np

def reward(N):
    """
    Generates the expected reward matrix R(s, a) for the predator-prey game.
    Size: |S| x |A|, where |S| = N^4 and |A| = 5.
    """
    num_states = N**4
    num_actions = 5
    reward_matrix = np.zeros((num_states, num_actions))
    
    # Convert coordinates to state index
    def get_state_idx(px, py, qx, qy):
        return (px-1)*N**3 + (py-1)*N**2 + (qx-1)*N + (qy-1)

    # Get valid moves for the prey (including 'stay')
    def get_valid_prey_moves(qx, qy):
        moves = [(qx, qy), (qx, qy+1), (qx, qy-1), (qx-1, qy), (qx+1, qy)]
        return [m for m in moves if 1 <= m[0] <= N and 1 <= m[1] <= N]
    
    # Iterate through every possible state
    for px in range(1, N+1):
        for py in range(1, N+1):
            for qx in range(1, N+1):
                for qy in range(1, N+1):
                    
                    curr_state = get_state_idx(px, py, qx, qy)
                    
                    # 1. Identify all valid moves for the Prey and their count (k)
                    valid_q_moves = get_valid_prey_moves(qx,qy)
                    k = len(valid_q_moves)
                    
                    # 2. Iterate through Predator's actions
                    for a in range(num_actions):
                        # Determine where the predator will land
                        npx, npy = px, py
                        if a == 1 and py < N: npy += 1    # Up
                        elif a == 2 and py > 1: npy -= 1  # Down
                        elif a == 3 and px > 1: npx -= 1  # Left
                        elif a == 4 and px < N: npx += 1  # Right
                        # Action 0 or invalid boundary moves keep predator at (px, py)
                        
                        # 3. Calculate expected reward
                        """R[curr_state, a] = sum over all X { (prob of prey getting to a cell X=(nqx,nqy))*(reward when prey is at X, predator is at the new cell (npx,npy)) }
                                            = prob that prey lands on the cell where predator is updated to"""
                        # If predator's destination is one of the prey's possible next cells:
                        if (npx, npy) in valid_q_moves:
                            # Probability of landing on the same square is 1/k
                            reward_matrix[curr_state, a] = 1.0 / k
                        else:
                            # No overlap possible for this action
                            reward_matrix[curr_state, a] = 0.0
                            
    return reward_matrix

# Example usage:
# R = reward(5)
# print(f"Reward Matrix Shape: {R.shape}")