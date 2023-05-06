import torch


def squashed_tanh(x: torch.Tensor) -> torch.Tensor:
    """Activation function to apply tanh to labels of {0, 1} by squashing the tanh distribution.

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The squashed output.

    """
    x = torch.tanh(x)
    x = 0.5 * (x + 1.0)  # Map tanh output to binary probability between 0 and 1
    return x
