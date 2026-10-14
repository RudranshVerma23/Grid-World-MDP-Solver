import numpy as np
from state_value_eval import state_value_eval

def q_value_eval(policy, kernel, reward_function, gamma=0.99):
    """
    Computes the action-value function Q_pi.
    
    Inputs:
        policy: Matrix of size |S| x |A|
        kernel: Matrix of size |S||A| x |S|
        reward_function: Matrix of size |S| x |A|
        gamma: Discount factor
        
    Output:
        Q: Matrix of size |S| x |A|
    """
    num_states = policy.shape[0]
    num_actions = policy.shape[1]
    
    # 1. Compute the State-Value Function V_pi (|S| x 1)
    # Using the iterative method from part (g)
    V_pi = state_value_eval(policy, kernel, reward_function, gamma)
    
    # 2. Compute the expected future discounted value for all (s, a) pairs
    # kernel (|S||A| x |S|) @ V_pi (|S| x 1) -> (|S||A| x 1)
    expected_future_value = gamma * np.dot(kernel, V_pi)
    
    # 3. Reshape the flattened state-action results back into an (|S| x |A|) matrix
    future_value_matrix = expected_future_value.reshape(num_states, num_actions)
    
    # 4. Q(s, a) = R(s, a) + gamma * expected_future_V
    Q = reward_function + future_value_matrix
    
    return Q