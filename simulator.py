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
    # Action mapping: 0: Still, 1: Up, 2: Down, 3: Left, 4: Right
    px, py = pred_loc
    if action == 1 and py < N: py += 1      # Up
    elif action == 2 and py > 1: py -= 1    # Down
    elif action == 3 and px > 1: px -= 1    # Left
    elif action == 4 and px < N: px += 1    # Right
    # if action is 0 or invalid (e.g. hitting a wall), (px, py) stays the same
    next_pred_loc = (px, py)

    # 2. Update Prey Location
    qx, qy = prey_loc
    potential_prey_moves = [
        (qx, qy),         # Stay still
        (qx, qy + 1),     # Up
        (qx, qy - 1),     # Down
        (qx - 1, qy),     # Left
        (qx + 1, qy)      # Right
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