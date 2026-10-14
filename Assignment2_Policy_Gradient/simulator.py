import random

def simulator(N, pred_loc, prey_loc, action):
    """
    Simulates one time step in the N x N predator-prey grid game.
    
    Inputs:
        N: Grid size (N x N)
        pred_loc: (x, y) tuple of current predator position
        prey_loc: (x, y) tuple of current prey position
        action: Integer 0-4 (0: Still, 1: Up, 2: Down, 3: Left, 4: Right)
        
    Outputs:
        next_pred_loc: (x, y) tuple of predator's new position
        next_prey_loc: (x, y) tuple of prey's new position (including respawn logic)
        reward: 1 if caught, 0 otherwise
    """
    # 1. Update Predator Location
    # Action mapping (expanded):
    # 0: Stay, 1: Up, 2: Down, 3: Left, 4: Right,
    # 5: Up-Left, 6: Up-Right, 7: Down-Left, 8: Down-Right
    px, py = pred_loc
    action_offsets = {
        0: (0, 0),
        1: (0, 1),   # Up
        2: (0, -1),  # Down
        3: (-1, 0),  # Left
        4: (1, 0),   # Right
        5: (-1, 1),  # Up-Left
        6: (1, 1),   # Up-Right
        7: (-1, -1), # Down-Left
        8: (1, -1),  # Down-Right
    }

    dx, dy = action_offsets.get(action, (0, 0))
    npx, npy = px + dx, py + dy
    # Boundary check: invalid move -> stay in place
    if not (1 <= npx <= N and 1 <= npy <= N):
        npx, npy = px, py

    next_pred_loc = (npx, npy)

    # 2. Update Prey Location
    qx, qy = prey_loc
    # Prey can also move diagonally now (uniform among valid moves)
    potential_prey_moves = [
        (qx, qy),           # Stay
        (qx, qy + 1),       # Up
        (qx, qy - 1),       # Down
        (qx - 1, qy),       # Left
        (qx + 1, qy),       # Right
        (qx - 1, qy + 1),   # Up-Left
        (qx + 1, qy + 1),   # Up-Right
        (qx - 1, qy - 1),   # Down-Left
        (qx + 1, qy - 1),   # Down-Right
    ]
    
    # Filter only moves that stay within the grid {1, ..., N}
    valid_prey_moves = [
        (x, y) for x, y in potential_prey_moves 
        if 1 <= x <= N and 1 <= y <= N
    ]
    
    # Uniform probability: P = 1 / len(valid_prey_moves)
    next_prey_loc = random.choice(valid_prey_moves)

    # 3. Reward and Respawn Logic
    if next_pred_loc == next_prey_loc:
        reward = 1.0
        # Respawn: choose from all cells EXCEPT the current predator location
        all_cells = [(x, y) for x in range(1, N + 1) for y in range(1, N + 1)]
        all_cells.remove(next_pred_loc)
        next_prey_loc = random.choice(all_cells)
    else:
        reward = 0.0

    return next_pred_loc, next_prey_loc, reward