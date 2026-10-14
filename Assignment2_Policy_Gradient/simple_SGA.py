import torch

from policy_network import PredatorPreyPolicyNet
from gradient_estimate import gradient_estimate
from rollout_discounted_reward import rollout_discounted_reward

def simple_SGA(
    policy: PredatorPreyPolicyNet,
    grid_size: int = 4,
    iterations: int = 200,
    H: int = 8,
    learning_rate: float = 1e-2,
    gamma: float = 0.99,
    max_steps: int = 200,
    start_predator: tuple[int, int] = (1, 1),
    start_prey: tuple[int, int] | None = None,
    eval_episodes: int = 20,
) -> dict[str, list[float] | float | int]:
    """Train the policy using simple stochastic gradient ascent.

    Each ascent step uses the average of H independent gradient estimates,
    which reduces variance compared to using a single sample estimate.
    """

    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if H < 1:
        raise ValueError("H must be at least 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if eval_episodes < 1:
        raise ValueError("eval_episodes must be at least 1")

    if start_prey is None:
        start_prey = (grid_size, grid_size)

    parameter_list = [parameter for parameter in policy.parameters() if parameter.requires_grad]

    discounted_reward_curve: list[float] = []
    objective_curve: list[float] = []

    for _iteration in range(iterations):
        
        estimate = gradient_estimate(
            policy=policy,
            grid_size=grid_size,
            num_episodes=H,
            gamma=gamma,
            max_steps=max_steps,
            start_predator=start_predator,
            start_prey=start_prey,
        )

        objective_avg = float(estimate["objective"].item())
        gradient_dict = estimate["gradients"]

        # after computing averaged_gradients but before parameter update
        if _iteration % 50 == 0 or _iteration == iterations - 1:
            norms = [gradient_dict[f"param_{i}"].norm().item() for i in range(len(parameter_list))]
            print(f"lr={learning_rate}, iter {_iteration}: avg_grad_norm={sum(norms)/len(parameter_list):.4e}, objective={objective_avg:.4f}")
        
        with torch.no_grad():
            for index, parameter in enumerate(parameter_list):
                parameter.add_(learning_rate * gradient_dict[f"param_{index}"])

        average_discounted_reward = 0.0
        for _ in range(eval_episodes):
            average_discounted_reward += rollout_discounted_reward(
                policy=policy,
                grid_size=grid_size,
                max_steps=max_steps,
                start_predator=start_predator,
                start_prey=start_prey,
                gamma=gamma,
            )
        average_discounted_reward /= float(eval_episodes)

        discounted_reward_curve.append(average_discounted_reward)
        objective_curve.append(objective_avg)

    return {
        "learning_rate": learning_rate,
        "H": H,
        "iterations": iterations,
        "discounted_reward_curve": discounted_reward_curve,
        "reward_curve": discounted_reward_curve,
        "objective_curve": objective_curve,
    }