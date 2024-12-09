# type: ignore
# Currently pyright doesn't support numba.cuda
from typing import Tuple

import numba
from numba import cuda
from typing import TypeVar, Any
import numba.cuda
from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    Shape,
    Storage,
    Strides,
    UserShape,
    to_index,
    index_to_position,
    broadcast_index,
)
from .tensor_functions import Function
from numba.cuda import jit as _jit

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """Device jit function

    Args:
    ----
        fn : function
        **kwargs : argument

    Returns:
    -------
        jit

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Any, **kwargs: Any) -> FakeCUDAKernel:
    """Jit function

    Args:
    ----
        fn : function
        **kwargs : argument

    Returns:
    -------
        FakeCUDAKernel

    """
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)


def _tensor_conv1d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """Implementation of 1D Convolution.

    Performs 1D convolution on an input tensor using a weight kernel, with optional
    kernel reversal for different anchoring modes.

    Args:
    ----
    out (Storage): Storage for the output tensor.
    out_shape (Shape): Shape of the output tensor (batch, out_channels, width).
    out_strides (Strides): Strides of the output tensor.
    out_size (int): Total size of the output tensor.
    input (Storage): Storage for the input tensor.
    input_shape (Shape): Shape of the input tensor (batch, in_channels, width).
    input_strides (Strides): Strides of the input tensor.
    weight (Storage): Storage for the kernel weights.
    weight_shape (Shape): Shape of the weight tensor (out_channels, in_channels, kernel_width).
    weight_strides (Strides): Strides of the weight tensor.
    reverse (bool): Determines weight orientation (left or right).

    Returns:
    -------
    None: Updates the `out` storage in place.

    Notes:
    -----
    - This method assumes the `out` tensor is pre-allocated with appropriate size and shape.
    - Handles parallel computation using CUDA shared memory.

    """
    # Define block dimensions
    BLOCK_DIM = 16
    BLOCK_DIM2 = 32

    # Extract shapes
    batch_out, out_channels, out_width = out_shape
    batch_in, in_channels, input_width = input_shape
    weight_out_channels, weight_in_channels, kernel_width = weight_shape

    # Sanity checks for shape compatibility
    assert batch_out == batch_in and out_channels == weight_out_channels
    assert in_channels == weight_in_channels and out_width <= input_width

    # Determine thread positions
    width_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    width_start = cuda.blockIdx.x * cuda.blockDim.x
    channel_idx = cuda.blockIdx.z
    px, py = cuda.threadIdx.x, cuda.threadIdx.y

    # Shared memory for kernel and input cache
    weight_cache = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    input_cache = cuda.shared.array((BLOCK_DIM, BLOCK_DIM2), numba.float64)

    # Strides for weight, input, and output tensors
    ws0, ws1, ws2 = weight_strides
    is0, is1, is2 = input_strides
    os0, os1, os2 = out_strides

    # Kernel traversal direction
    kernel_step = -1 if reverse else 1

    for batch_idx in range(batch_out):
        accumulator = 0.0

        # Process channels in blocks
        for channel_start in range(0, in_channels, BLOCK_DIM):
            channel_cache_idx = channel_start + px

            # Process kernel width in blocks
            for kernel_start in range(0, kernel_width, BLOCK_DIM):
                kernel_idx = py + kernel_start

                # Cache kernel weights
                if channel_cache_idx < in_channels and kernel_idx < kernel_width:
                    cache_idx = (
                        channel_idx * ws0 + channel_cache_idx * ws1 + kernel_idx * ws2
                    )
                    weight_cache[px, py] = weight[cache_idx]
                else:
                    weight_cache[px, py] = 0.0

                numba.cuda.syncthreads()

                # Cache input data
                for width_offset in range(0, BLOCK_DIM2, BLOCK_DIM):
                    if reverse:
                        pos = (
                            width_start
                            - kernel_start
                            - BLOCK_DIM
                            + 1
                            + width_offset
                            + py
                        )
                    else:
                        pos = width_start + kernel_start + width_offset + py

                    if channel_cache_idx < in_channels and 0 <= pos < input_width:
                        input_cache_idx = (
                            batch_idx * is0 + channel_cache_idx * is1 + pos * is2
                        )
                        input_cache[px, width_offset + py] = input[input_cache_idx]
                    else:
                        input_cache[px, width_offset + py] = 0.0

                numba.cuda.syncthreads()

                # Compute convolution
                if py == 0 and width_idx < out_width:
                    for channel_idx_inner in range(
                        channel_start, min(in_channels, channel_start + BLOCK_DIM)
                    ):
                        for kernel_idx_inner in range(
                            kernel_start, min(kernel_width, kernel_start + BLOCK_DIM)
                        ):
                            pos = width_idx + kernel_idx_inner * kernel_step

                            if reverse:
                                min_bound = width_start - kernel_start - BLOCK_DIM + 1
                            else:
                                min_bound = width_start + kernel_start

                            max_bound = min_bound + BLOCK_DIM2

                            if min_bound <= pos < max_bound and 0 <= pos < input_width:
                                accumulator += (
                                    weight_cache[
                                        channel_idx_inner - channel_start,
                                        kernel_idx_inner - kernel_start,
                                    ]
                                    * input_cache[
                                        channel_idx_inner - channel_start,
                                        abs(pos - min_bound),
                                    ]
                                )
                numba.cuda.syncthreads()

        # Write output
        if py == 0 and width_idx < out_width:
            output_idx = batch_idx * os0 + channel_idx * os1 + width_idx * os2
            out[output_idx] = accumulator


# JIT compile the function for CUDA
tensor_conv1d = cuda.jit()(_tensor_conv1d)


class Conv1dFun(Function):
    @staticmethod
    def forward_inner(
        output_shape: UserShape, input: Tensor, weight: Tensor, reversed: bool = False
    ) -> Tensor:
        """Perform a 1D convolution (helper for forward).

        Args:
        ----
        output_shape (UserShape): Shape of the output tensor, used to control the convolution length.
        input (Tensor): Input tensor of shape (batch, in_channel, w).
        weight (Tensor): Weight tensor of shape (out_channel, in_channel, kw).
        reversed (bool, optional):
            - If True, anchors weights differently.
            - Computes out[a, b, c] = in[a, :, c:c-kw:-1] * weight[b, :, 0:kw].

        Returns:
        -------
        Tensor: Resulting tensor of shape (batch, out_channel, w).

        """
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2, "Input and weight channels mismatch."

        # Allocate output tensor
        output = input.zeros(output_shape)

        # Define CUDA grid and block dimensions
        THREADS_PER_BLOCK = 16
        blockspergrid = (
            (w + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            1,
            out_channels,
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        # Launch CUDA kernel
        tensor_conv1d[blockspergrid, threadsperblock](
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), reversed
        )
        return output

    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute the forward pass of the 1D convolution.

        Args:
        ----
        ctx (Context): Context object for saving intermediate values.
        input (Tensor): Input tensor of shape (batch, in_channel, w).
        weight (Tensor): Weight tensor of shape (out_channel, in_channel, kw).

        Returns:
        -------
        Tensor: Output tensor of shape (batch, out_channel, w).

        """
        # Save input and weight for backward computation
        ctx.save_for_backward(input, weight)

        # Perform forward convolution
        output = Conv1dFun.forward_inner(
            (input.shape[0], weight.shape[0], input.shape[2]),
            input,
            weight,
            reversed=False,
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the backward pass of the 1D convolution.

        Args:
        ----
        ctx (Context): Context object with saved forward pass values.
        grad_output (Tensor): Gradient of the output tensor.

        Returns:
        -------
        Tuple[Tensor, Tensor]:
            - Gradient with respect to input tensor.
            - Gradient with respect to weight tensor.

        """
        # Retrieve saved input and weight
        input, weight = ctx.saved_values

        # Compute gradient with respect to weight
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        grad_weight = Conv1dFun.forward_inner(
            (weight.shape[1], weight.shape[0], weight.shape[2]),
            new_input,
            new_grad_output,
            reversed=False,
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        # Compute gradient with respect to input
        new_weight = weight.permute(1, 0, 2)
        grad_input = Conv1dFun.forward_inner(
            input.shape, grad_output, new_weight, reversed=True
        )

        return grad_input, grad_weight


# Apply the Conv1dFun class as a function
conv1d = Conv1dFun.apply


def _tensor_conv2d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """2D Convolution Implementation.

    This function computes a 2D convolution for input tensors with the following shapes:
    - Input tensor: (batch, in_channels, height, width)
    - Weight tensor: (out_channels, in_channels, k_height, k_width)
    - Output tensor: (batch, out_channels, height, width)

    Args:
    ----
    out (Storage): Storage for the output tensor.
    out_shape (Shape): Shape of the output tensor.
    out_strides (Strides): Strides of the output tensor.
    out_size (int): Size of the output tensor.
    input (Storage): Storage for the input tensor.
    input_shape (Shape): Shape of the input tensor.
    input_strides (Strides): Strides of the input tensor.
    weight (Storage): Storage for the weight tensor.
    weight_shape (Shape): Shape of the weight tensor.
    weight_strides (Strides): Strides of the weight tensor.
    reverse (bool): Whether the weight is anchored at top-left (False) or bottom-right (True).

    Returns:
    -------
        None: Updates the `out` storage in place.

    """
    # Unpack tensor shapes
    batch_, out_channels, out_height, out_width = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    # Constants for CUDA block dimensions
    BLOCK_DIM = 16
    BLOCK_DIM2 = 32

    # Assertions for shape consistency
    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    ), "Shape mismatch between input, output, and weight tensors."
    assert (
        out_width <= width and out_height <= height
    ), "Output dimensions exceed input dimensions."

    # Calculate thread positions
    width_i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    height_i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    width_cache_start = cuda.blockIdx.x * cuda.blockDim.x
    height_cache_start = cuda.blockIdx.y * cuda.blockDim.y
    out_channel_i = cuda.blockIdx.z
    px = cuda.threadIdx.x
    py = cuda.threadIdx.y

    # Shared memory for weights and input cache
    weight_cache = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    input_cache = cuda.shared.array((BLOCK_DIM2, BLOCK_DIM2), numba.float64)

    # Unpack strides
    ws0, ws1, ws2, ws3 = weight_strides
    is0, is1, is2, is3 = input_strides
    os0, os1, os2, os3 = out_strides

    # Direction multiplier based on `reverse`
    kid = -1 if reverse else 1

    for batch_i in range(batch):
        # Initialize output position and temporary variable
        out_pos = batch_i * os0 + out_channel_i * os1 + height_i * os2 + width_i * os3
        tmp = 0.0

        for in_channel_i in range(in_channels):
            # Cache weights in blocks
            for kh_start in range(0, kh, BLOCK_DIM):
                for kw_start in range(0, kw, BLOCK_DIM):
                    kw_now = kw_start + px
                    kh_now = kh_start + py

                    # Populate weight cache
                    if kh_now < kh and kw_now < kw:
                        weight_cache_pos = (
                            out_channel_i * ws0
                            + in_channel_i * ws1
                            + kh_now * ws2
                            + kw_now * ws3
                        )
                        weight_cache[(px, py)] = weight[weight_cache_pos]
                    else:
                        weight_cache[(px, py)] = 0.0
                    numba.cuda.syncthreads()

                    # Cache input based on kernel size
                    for w_cache_bias in range(0, BLOCK_DIM2, BLOCK_DIM):
                        for h_cache_bias in range(0, BLOCK_DIM2, BLOCK_DIM):
                            if reverse:
                                w_cache_pos = (
                                    width_cache_start
                                    - kw_start
                                    - BLOCK_DIM
                                    + 1
                                    + w_cache_bias
                                    + px
                                )
                                h_cache_pos = (
                                    height_cache_start
                                    - kh_start
                                    - BLOCK_DIM
                                    + 1
                                    + h_cache_bias
                                    + py
                                )
                            else:
                                w_cache_pos = (
                                    width_cache_start + kw_start + w_cache_bias + px
                                )
                                h_cache_pos = (
                                    height_cache_start + kh_start + h_cache_bias + py
                                )

                            if 0 <= w_cache_pos < width and 0 <= h_cache_pos < height:
                                input_cache_pos = (
                                    batch_i * is0
                                    + in_channel_i * is1
                                    + h_cache_pos * is2
                                    + w_cache_pos * is3
                                )
                                input_cache[(w_cache_bias + px, h_cache_bias + py)] = (
                                    input[input_cache_pos]
                                )
                            else:
                                input_cache[(w_cache_bias + px, h_cache_bias + py)] = (
                                    0.0
                                )
                            numba.cuda.syncthreads()

                    # Compute convolution for valid output positions
                    if height_i < out_height and width_i < out_width:
                        for khi in range(kh_start, min(kh, kh_start + BLOCK_DIM)):
                            h_now = height_i + khi * kid
                            height_cache_min = (
                                height_cache_start - kh_start - BLOCK_DIM + 1
                                if reverse
                                else height_cache_start + kh_start
                            )
                            height_cache_max = height_cache_min + BLOCK_DIM2

                            if not (
                                0 <= h_now < height
                                and height_cache_min <= h_now < height_cache_max
                            ):
                                continue

                            for kwi in range(kw_start, min(kw, kw_start + BLOCK_DIM)):
                                w_now = width_i + kwi * kid
                                width_cache_min = (
                                    width_cache_start - kw_start - BLOCK_DIM + 1
                                    if reverse
                                    else width_cache_start + kw_start
                                )
                                width_cache_max = width_cache_min + BLOCK_DIM2

                                if not (
                                    0 <= w_now < width
                                    and width_cache_min <= w_now < width_cache_max
                                ):
                                    continue

                                tmp += (
                                    weight_cache[(kwi - kw_start, khi - kh_start)]
                                    * input_cache[
                                        (
                                            abs(w_now - width_cache_min),
                                            abs(h_now - height_cache_min),
                                        )
                                    ]
                                )
                    numba.cuda.syncthreads()

        # Store result in the output tensor
        if height_i < out_height and width_i < out_width:
            out[out_pos] = tmp


tensor_conv2d = cuda.jit()(_tensor_conv2d)


class Conv2dFun(Function):
    @staticmethod
    def forward_inner(
        output_shape: UserShape, input: Tensor, weight: Tensor, reversed: bool = False
    ) -> Tensor:
        """Compute a 2D Convolution, called by forward.

        Args:
        ----
            output_shape (UserShape): Shape of the output tensor.
            input (Tensor): Input tensor with shape batch x in_channel x h x w.
            weight (Tensor): Weight tensor with shape out_channel x in_channel x kh x kw.
            reversed (bool): If True, reverse the convolution.

        Returns:
        -------
            Tensor: Output tensor with shape batch x out_channel x h x w.

        """
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2, "Input and weight channels do not match."

        output = input.zeros(output_shape)
        THREADS_PER_BLOCK = 16

        # Define grid and block dimensions for CUDA kernel
        blockspergrid = (
            (w + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK,
            (h + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK,
            out_channels,
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        # Invoke CUDA kernel
        tensor_conv2d[blockspergrid, threadsperblock](
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), reversed
        )
        return output

    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Perform forward pass of the 2D convolution.

        Args:
        ----
            ctx (Context): Context to save inputs for backward pass.
            input (Tensor): Input tensor with shape batch x in_channel x h x w.
            weight (Tensor): Weight tensor with shape out_channel x in_channel x kh x kw.

        Returns:
        -------
            Tensor: Output tensor with shape batch x out_channel x h x w.

        """
        ctx.save_for_backward(input, weight)
        output_shape = (input.shape[0], weight.shape[0], input.shape[2], input.shape[3])
        output = Conv2dFun.forward_inner(output_shape, input, weight, reversed=False)
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Perform backward pass of the 2D convolution.

        Args:
        ----
            ctx (Context): Context containing saved tensors from the forward pass.
            grad_output (Tensor): Gradient of the output tensor.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients of the input and weight tensors.

        """
        input, weight = ctx.saved_values

        # Compute gradient with respect to weight
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        grad_weight_shape = (
            weight.shape[1],
            weight.shape[0],
            weight.shape[2],
            weight.shape[3],
        )
        grad_weight = Conv2dFun.forward_inner(
            grad_weight_shape, new_input, new_grad_output, reversed=False
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        # Compute gradient with respect to input
        new_weight = weight.permute(1, 0, 2, 3)
        grad_input = Conv2dFun.forward_inner(
            input.shape, grad_output, new_weight, reversed=True
        )

        return grad_input, grad_weight


conv2d = Conv2dFun.apply
