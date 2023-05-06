import torch.nn.functional as F
import torch


def phish(input: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    The Phish activation function, defined as f(x) = xTanH(GELU(x))

    Parameters:
    -----------
    input: torch.Tensor
        The input tensor.
    reduction: str, optional
        Specifies the type of reduction to apply to the output:
        'none' | 'mean' | 'sum'. Default: 'mean'.

    Returns:
    --------
    torch.Tensor
        The Phish activation value.

    Examples:
    ---------
    >>> input = torch.randn(2, 3)
    >>> output = phish_loss(input)
    >>> output.shape
    torch.Size([])

    >>> output = phish_loss(input, reduction='none')
    >>> output.shape
    torch.Size([2, 3])

    >>> output = phish_loss(input, reduction='sum')
    >>> output.shape
    torch.Size([])
    """
    x = input.float()
    gelu = F.gelu(x)
    tanh = torch.tanh(gelu)
    output = x * tanh
    match reduction:
        case "mean":
            return output.mean()
        case "sum":
            return output.sum()
        case _:
            return output
