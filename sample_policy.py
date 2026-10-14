import numpy as np

def sample_policy(N):
    """
    Generates a stationary policy matrix of size |S| x |A|.
    
    The predator takes distance-minimizing actions with 0.5 probability 
    and other actions with 0.5 probability.
    """
    num_states = N**4
    num_actions = 5
    policy = np.zeros((num_states, num_actions))
    
    # Action mapping: 0: Still, 1: Up, 2: Down, 3: Left, 4: Right
    action_offsets = {0: (0, 0), 1: (0, 1), 2: (0, -1), 3: (-1, 0), 4: (1, 0)}

    def get_state_idx(px, py, qx, qy):
        return (px-1)*N**3 + (py-1)*N**2 + (qx-1)*N + (qy-1)
        
    for px in range(1, N+1):
        for py in range(1, N+1):
            for qx in range(1, N+1):
                for qy in range(1, N+1):
                    
                    state = get_state_idx(px, py, qx, qy)
                    distances = []
                    
                    # 1. Calculate the distance resulting from each action
                    for a in range(num_actions):
                        dx, dy = action_offsets[a]
                        # Apply boundary logic: if move is invalid, predator stays put
                        npx, npy = px + dx, py + dy
                        if not (1 <= npx <= N and 1 <= npy <= N):
                            npx, npy = px, py
                        
                        # Manhattan distance to current prey position (qx, qy)
                        dist = abs(npx - qx) + abs(npy - qy)
                        distances.append(dist)
                    
                    # 2. Identify best and other actions
                    min_dist = min(distances)
                    best_actions = [i for i, d in enumerate(distances) if d == min_dist]
                    other_actions = [i for i, d in enumerate(distances) if d > min_dist]
                    
                    # 3. Assign probabilities
                    if not other_actions:
                        # If all actions are equally good/bad, distribute 1.0 uniformly
                        policy[state, :] = 1.0 / num_actions
                    else:
                        # 0.5 split between best actions
                        for a_best in best_actions:
                            policy[state, a_best] = 0.5 / len(best_actions)
                        
                        # 0.5 split between other actions
                        for a_other in other_actions:
                            policy[state, a_other] = 0.5 / len(other_actions)
                            
    return policy

# Verification
# P = sample_policy(3)
# print(f"Policy Matrix Shape: {P.shape}")
# print(f"Row sums are 1.0: {np.allclose(np.sum(P, axis=1), 1.0)}")