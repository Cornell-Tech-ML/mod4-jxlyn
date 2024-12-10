from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off

max_reduce = FastOps.reduce(operators.max, -1e9)


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    if not input._tensor.is_contiguous():
        input = input.contiguous()
    new_height = height // kh
    new_width = width // kw
    reshaped_input = input.view(batch, channel, new_height, kh, new_width, kw)
    permuted_input = reshaped_input.permute(0, 1, 2, 4, 3, 5)
    permuted_input = permuted_input.contiguous()
    tiled_input = permuted_input.view(batch, channel, new_height, new_width, kh * kw)

    return tiled_input, new_height, new_width


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    batch, channel, height, width = input.shape
    result = tile(input, kernel)
    output = result[0].mean(4)
    return output.view(batch, channel, result[1], result[2])


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor.

    Args:
    ----
        input: input tensor
        dim: dimension to apply argmax


    Returns:
    -------
        Tensor with 1 where the max is and 0 elsewhere.

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for max operator

        Args:
        ----
            ctx: context
            input: input tensor
            dim: dimension to apply max

        Returns:
        -------
            Tensor with max values

        """
        result = max_reduce(input, int(dim[0]))
        ctx.save_for_backward(input, result)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for max operator

        Args:
        ----
            ctx: context
            grad_output : gradient tensor

        Returns:
        -------
            Tuple with gradient and 0.0

        """
        input, old_result = ctx.saved_values
        result = (input == old_result) * grad_output
        return result, 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction

    Args:
    ----
        input: input tensor
        dim: dimension to apply max

    Returns:
    -------
        Tensor with max values

    """
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor.

    Args:
    ----
        input: input tensor
        dim: dimension to apply softmax

    Returns:
    -------
        Tensor with softmax values

    """
    exp_input = input.exp()
    sum_exp_input = exp_input.sum(dim)
    return exp_input / sum_exp_input


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor.

    Args:
    ----
        input: input tensor
        dim: dimension to apply softmax

    Returns:
    -------
        Tensor with log softmax values

    """
    m = max(input, dim)
    stable_in = input - m
    return input - stable_in.exp().sum(dim).log() - m


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    batch, channel, height, width = input.shape
    middle, new_height, new_width = tile(input, kernel)
    return max(middle, 4).view(batch, channel, new_height, new_width)


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise, include an argument to turn off

    Args:
    ----
        input: input tensor
        p: probability of dropout
        ignore: ignore the dropout and return the input

    Returns:
    -------
        Tensor with dropout applied

    """
    if not ignore:
        prob = rand(input.shape)
        keep = prob > p
        return input * keep
    return input
