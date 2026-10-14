import torch
from torch import Tensor

from simulator import simulator
from policy_network import PredatorPreyPolicyNet

def gradient_estimate(
    policy: PredatorPreyPolicyNet,
    grid_size: int = 4,
    num_episodes: int = 1,
    gamma: float = 0.99,
    max_steps: int = 200,
    start_predator: tuple[int, int] = (1, 1),
    start_prey: tuple[int, int] | None = None,
) -> dict[str, Tensor]:
    """Return an unbiased Monte Carlo estimate of the policy gradient.

    The estimator uses the REINFORCE identity

        ∇J(θ) = E[\sum_t G_t ∇ log π_θ(a_t | s_t)]

    where G_t is the discounted return from time t onward. The simulator is
    used to generate complete trajectories, and PyTorch autograd computes the
    score-function term through ``Categorical.log_prob``.

    Returns a dictionary with the averaged objective and the detached gradient
    of each parameter tensor.
    """

    if num_episodes < 1:
        raise ValueError("num_episodes must be at least 1")

    if start_prey is None:
        start_prey = (grid_size, grid_size)

    parameter_list = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    gradient_buffers = [torch.zeros_like(parameter) for parameter in parameter_list]

    total_objective = torch.zeros((), dtype=torch.float32)

    for _ in range(num_episodes):
        predator_loc = start_predator
        prey_loc = start_prey
        episode_log_probs: list[Tensor] = []
        episode_rewards: list[float] = []

        for _step in range(max_steps):
            state = torch.tensor([*predator_loc, *prey_loc], dtype=torch.float32)
            logits = policy(state).squeeze(0)
            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.sample()

            episode_log_probs.append(distribution.log_prob(action))

            next_predator_loc, next_prey_loc, reward = simulator(
                grid_size,
                predator_loc,
                prey_loc,
                int(action.item()),
            )
            episode_rewards.append(float(reward))

            predator_loc = next_predator_loc
            prey_loc = next_prey_loc

            if reward > 0.0:
                break

        returns = []
        running_return = 0.0
        for reward in reversed(episode_rewards):
            running_return = reward + gamma * running_return
            returns.append(running_return)
        returns.reverse()

        if not returns:
            continue

        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        
        episode_objective = (torch.stack(episode_log_probs) * returns_tensor).sum()
        total_objective = total_objective + episode_objective.detach()

        gradients = torch.autograd.grad(episode_objective, parameter_list, retain_graph=False)
        for buffer, grad in zip(gradient_buffers, gradients):
            buffer.add_(grad)

    gradient_estimate_tensor = {
        f"param_{index}": buffer / float(num_episodes)
        for index, buffer in enumerate(gradient_buffers)
    }

    return {
        "objective": total_objective / float(num_episodes),
        "gradients": gradient_estimate_tensor,
    }