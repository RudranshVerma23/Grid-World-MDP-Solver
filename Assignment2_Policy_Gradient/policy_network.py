"""Neural policy module for the predator-prey grid world.

The state of the environment is represented by four grid coordinates:
predator position (p_x, p_y) and prey position (q_x, q_y). For the
N=4 case used in Assignment 2, the policy network therefore takes a
4-dimensional input and produces logits for the 9 available actions:
stay, up, down, left, right and the four diagonal moves.
"""

from typing import Iterable, Sequence

import torch
from torch import Tensor, nn

class PredatorPreyPolicyNet(nn.Module):
    """Feed-forward policy network for the predator-prey MDP.

    Parameters
    ----------
    grid_size:
        Side length of the square grid.
    hidden_sizes:
        Width of the hidden layers. The default two-layer MLP is compact
        enough for the 256-state N=4 problem while still being expressive.
    action_dim:
        Number of discrete actions in the environment.
    normalize_inputs:
        If True, raw coordinates in [1, grid_size] are scaled to [0, 1].
    """

    def __init__(
        self,
        grid_size: int = 4,
        hidden_sizes: Sequence[int] = (64, 32),
        action_dim: int = 9,
        normalize_inputs: bool = True,
    ) -> None:
        super().__init__()

        if grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer")

        self.grid_size = grid_size
        self.action_dim = action_dim
        self.normalize_inputs = normalize_inputs

        layer_sizes = [4, *hidden_sizes, action_dim]
        layers = []
        for in_features, out_features in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_features, out_features))
            if out_features != action_dim:
                layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

    def _as_tensor(self, value: Tensor | Iterable[float]) -> Tensor:
        tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
        return tensor.to(dtype=torch.float32)

    def encode_coordinates(self, coordinates: Tensor | Iterable[float]) -> Tensor:
        """Convert raw coordinates to the network input format.

        The expected layout is [p_x, p_y, q_x, q_y] using 1-based grid
        coordinates. The returned tensor is batched and optionally normalized.
        """

        tensor = self._as_tensor(coordinates)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)

        if tensor.shape[-1] != 4:
            raise ValueError("coordinates must have four entries: [p_x, p_y, q_x, q_y]")

        if self.normalize_inputs:
            scale = float(self.grid_size - 1)
            tensor = (tensor - 1.0) / scale

        return tensor

    def decode_state_index(self, state_index: Tensor | int) -> Tensor:
        """Map a flattened state index back to coordinates.

        The index convention matches the Assignment 1 code:
        (p_x-1) * N^3 + (p_y-1) * N^2 + (q_x-1) * N + (q_y-1).
        """

        index = torch.as_tensor(state_index, dtype=torch.long)
        if index.ndim == 0:
            index = index.unsqueeze(0)

        n = self.grid_size
        px = (index // (n**3)) + 1
        py = ((index % (n**3)) // (n**2)) + 1
        qx = ((index % (n**2)) // n) + 1
        qy = (index % n) + 1
        return torch.stack((px, py, qx, qy), dim=-1).to(dtype=torch.float32)

    def encode_state_index(self, state_index: Tensor | int) -> Tensor:
        """Convert a flattened state index directly into network features."""

        return self.encode_coordinates(self.decode_state_index(state_index))

    def forward(self, coordinates: Tensor | Iterable[float]) -> Tensor:
        """Return action logits for one state or a batch of states."""

        features = self.encode_coordinates(coordinates)
        return self.network(features)

    def policy_distribution(self, coordinates: Tensor | Iterable[float]) -> Tensor:
        """Return the softmax policy probabilities for each action."""

        return torch.softmax(self.forward(coordinates), dim=-1)

    def greedy_action(self, coordinates: Tensor | Iterable[float]) -> Tensor:
        """Return the greedy action under the current policy parameters."""

        return torch.argmax(self.forward(coordinates), dim=-1)

    def sample_action(self, coordinates: Tensor | Iterable[float]) -> Tensor:
        """Sample an action index from the policy distribution."""

        logits = self.forward(coordinates)
        distribution = torch.distributions.Categorical(logits=logits)
        return distribution.sample()


def build_default_policy(grid_size: int = 4) -> PredatorPreyPolicyNet:
    """Convenience constructor for the N=4 assignment setting."""

    return PredatorPreyPolicyNet(grid_size=grid_size)
