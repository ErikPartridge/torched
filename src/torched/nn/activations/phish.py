import torch
from torched.functional.activations.phish import phish


class Phish(torch.nn.Module):
    """
    The Phish activation function, defined as f(x) = xTanH(GELU(x))

    Parameters:
    -----------
    reduction: str, optional
        Specifies the type of reduction to apply to the output:
        'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return phish(input, reduction=self.reduction)
