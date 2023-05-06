import torch
from torched.functional.activations.squashed_tanh import squashed_tanh


class SquashedTanh(torch.nn.Module):
    """
    The Squashed tanh activation function, defined as f(x) = .5 * (tanh(x) + 1.0)
    """

    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return squashed_tanh(input)
