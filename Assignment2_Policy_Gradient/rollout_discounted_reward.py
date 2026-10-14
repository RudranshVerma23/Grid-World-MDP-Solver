import torch

from policy_network import PredatorPreyPolicyNet
from simulator import simulator

def rollout_discounted_reward(
    policy: PredatorPreyPolicyNet,
    grid_size: int,
    max_steps: int,
    start_predator: tuple[int, int],
    start_prey: tuple[int, int],
    gamma: float,
) -> float:
    """Sample one episode and return the discounted return from t=0."""

    predator_loc = start_predator
    prey_loc = start_prey
    discounted_return = 0.0
    discount = 1.0

    for _ in range(max_steps):
        state = torch.tensor([*predator_loc, *prey_loc], dtype=torch.float32)
        logits = policy(state).squeeze(0)
        action = torch.distributions.Categorical(logits=logits).sample()

        predator_loc, prey_loc, reward = simulator(
            grid_size,
            predator_loc,
            prey_loc,
            int(action.item()),
        )
        discounted_return += discount * float(reward)
        discount *= gamma

        if reward > 0.0:
            break

    return discounted_return