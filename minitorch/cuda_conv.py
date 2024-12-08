from typing import Tuple

import numpy as np
import numba
from numba import njit, prange, cuda

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Index,
    Shape,
    Storage,
    Strides,
    UserShape,
    to_index,
    index_to_position,
    broadcast_index,
)
from .tensor_functions import Function

to_index = cuda.jit(device=True)(to_index)
index_to_position = cuda.jit(device=True)(index_to_position)
broadcast_index = cuda.jit(device=True)(broadcast_index)


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
    """
    1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at left or right
    """
    BLOCK_DIM = 16
    BLOCK_DIM2 = 32
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    assert out_width <= width
    width_i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    width_cache_start = cuda.blockIdx.x * cuda.blockDim.x
    out_channel_i = cuda.blockIdx.z
    px = cuda.threadIdx.x
    py = cuda.threadIdx.y
    weight_cache = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    input_cache = cuda.shared.array((BLOCK_DIM, BLOCK_DIM2), numba.float64)
    ws0, ws1, ws2 = weight_strides
    is0, is1, is2 = input_strides
    os0, os1, os2 = out_strides

    kwid = -1 if reverse else 1
    for batch_i in range(batch):
        tmp = 0.0
        for in_channel_start in range(0, in_channels, BLOCK_DIM):
            in_channel_cache_pos = in_channel_start + px
            for kw_start in range(0, kw, BLOCK_DIM):
                kw_now = kw_start + py
                if in_channel_cache_pos < in_channels and kw_now < kw:
                    weight_cache_pos = (
                        out_channel_i * ws0 + in_channel_cache_pos * ws1 + kw_now * ws2
                    )
                    weight_cache[(px, py)] = weight[weight_cache_pos]
                else:
                    weight_cache[(px, py)] = 0.0
                numba.cuda.syncthreads()
                for w_cache_bias in range(0, BLOCK_DIM2, BLOCK_DIM):
                    if reverse:
                        w_cache_pos = (
                            width_cache_start
                            - kw_start
                            - BLOCK_DIM
                            + 1
                            + w_cache_bias
                            + py
                        )
                    else:
                        w_cache_pos = width_cache_start + kw_start + w_cache_bias + py
                    if in_channel_cache_pos < in_channels and 0 <= w_cache_pos < width:
                        input_cache_pos = (
                            batch_i * is0
                            + in_channel_cache_pos * is1
                            + w_cache_pos * is2
                        )
                        input_cache[(px, w_cache_bias + py)] = input[input_cache_pos]
                    else:
                        input_cache[(px, w_cache_bias + py)] = 0.0
                numba.cuda.syncthreads()
                if py == 0 and width_i < out_width:
                    for in_channel_i in range(
                        in_channel_start, min(in_channels, in_channel_start + BLOCK_DIM)
                    ):
                        for kwi in range(kw_start, min(kw, kw_start + BLOCK_DIM)):
                            w_now = width_i + kwi * kwid
                            if reverse:
                                width_cache_min = (
                                    width_cache_start - kw_start - BLOCK_DIM + 1
                                )
                            else:
                                width_cache_min = width_cache_start + kw_start
                            width_cache_max = width_cache_min + BLOCK_DIM2
                            if (
                                width_cache_min <= w_now < width_cache_max
                                and 0 <= w_now < width
                            ):
                                tmp += (
                                    weight_cache[
                                        (
                                            in_channel_i - in_channel_start,
                                            kwi - kw_start,
                                        )
                                    ]
                                    * input_cache[
                                        (
                                            in_channel_i - in_channel_start,
                                            abs(w_now - width_cache_min),
                                        )
                                    ]
                                )
                numba.cuda.syncthreads()
        if py == 0 and width_i < out_width:
            out_pos = batch_i * os0 + out_channel_i * os1 + width_i * os2
            out[out_pos] = tmp


tensor_conv1d = cuda.jit()(_tensor_conv1d)


class Conv1dFun(Function):
    @staticmethod
    def forward_inner(
        output_shape: UserShape, input: Tensor, weight: Tensor, reversed: bool = False
    ) -> Tensor:
        """
        Compute a 1D Convolution, called by forward

        Args:
            output_shape: can use output shape to control the conv len
            input : batch x in_channel x h x w
            weight : out_channel x in_channel x kh x kw
            reversed: if True, out[a,b,c,d] = in[a, :, c, d-kd:d] * w[a, e, 0:kd]

        Returns:
            batch x out_channel x h x w
        """
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros(output_shape)
        THREADS_PER_BLOCK = 16
        blockspergrid = (
            (w + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            1,
            out_channels,
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)
        tensor_conv1d[blockspergrid, threadsperblock](
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), reversed
        )
        return output

    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        Compute a 1D Convolution

        Args:
            ctx : Context
            input : batch x in_channel x h x w
            weight : out_channel x in_channel x kh x kw

        Returns:
            batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        output = Conv1dFun.forward_inner(
            (input.shape[0], weight.shape[0], input.shape[2]),
            input,
            weight,
            reversed=False,
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        grad_weight = Conv1dFun.forward_inner(
            (weight.shape[1], weight.shape[0], weight.shape[2]),
            new_input,
            new_grad_output,
            reversed=False,
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        new_weight = weight.permute(1, 0, 2)
        grad_input = Conv1dFun.forward_inner(
            input.shape, grad_output, new_weight, reversed=True
        )
        return grad_input, grad_weight


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
    """
    2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right
    """
    batch_, out_channels, out_height, out_width = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    BLOCK_DIM = 16
    BLOCK_DIM2 = 32

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    assert out_width <= width and out_height <= height
    width_i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    height_i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    width_cache_start = cuda.blockIdx.x * cuda.blockDim.x
    height_cache_start = cuda.blockIdx.y * cuda.blockDim.y
    out_channel_i = cuda.blockIdx.z
    px = cuda.threadIdx.x
    py = cuda.threadIdx.y
    weight_cache = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    input_cache = cuda.shared.array((BLOCK_DIM2, BLOCK_DIM2), numba.float64)
    ws0, ws1, ws2, ws3 = weight_strides
    is0, is1, is2, is3 = input_strides
    os0, os1, os2, os3 = out_strides

    kid = -1 if reverse else 1
    for batch_i in range(batch):
        out_pos = batch_i * os0 + out_channel_i * os1 + height_i * os2 + width_i * os3
        tmp = 0.0
        for in_channel_i in range(in_channels):
            for kh_start in range(0, kh, BLOCK_DIM):
                for kw_start in range(0, kw, BLOCK_DIM):
                    kw_now = kw_start + px
                    kh_now = kh_start + py
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

                    if height_i < out_height and width_i < out_width:
                        for khi in range(kh_start, min(kh, kh_start + BLOCK_DIM)):
                            h_now = height_i + khi * kid
                            if reverse:
                                height_cache_min = (
                                    height_cache_start - kh_start - BLOCK_DIM + 1
                                )
                            else:
                                height_cache_min = height_cache_start + kh_start
                            height_cache_max = height_cache_min + BLOCK_DIM2
                            if not (
                                0 <= h_now < height
                                and height_cache_min <= h_now < height_cache_max
                            ):
                                continue
                            for kwi in range(kw_start, min(kw, kw_start + BLOCK_DIM)):
                                w_now = width_i + kwi * kid
                                if reverse:
                                    width_cache_min = (
                                        width_cache_start - kw_start - BLOCK_DIM + 1
                                    )
                                else:
                                    width_cache_min = width_cache_start + kw_start
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
        if height_i < out_height and width_i < out_width:
            out_pos = (
                batch_i * os0 + out_channel_i * os1 + height_i * os2 + width_i * os3
            )
            out[out_pos] = tmp


tensor_conv2d = cuda.jit()(_tensor_conv2d)


class Conv2dFun(Function):
    @staticmethod
    def forward_inner(
        output_shape: UserShape, input: Tensor, weight: Tensor, reversed: bool = False
    ) -> Tensor:
        """
        Compute a 2D Convolution, called by forward

        Args:
            output_shape: can use output shape to control the conv len
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw
            reversed: if True, out[a,b,c,d] = in[a, :, c, d-kd:d] * w[a, e, 0:kd]

        Returns:
            (:class:`Tensor`) : batch x out_channel x h x w
        """
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros(output_shape)
        THREADS_PER_BLOCK = 16
        blockspergrid = (
            (w + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (h + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out_channels,
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)
        tensor_conv2d[blockspergrid, threadsperblock](
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), reversed
        )
        return output

    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        Compute a 2D Convolution

        Args:
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
            (:class:`Tensor`) : batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        output = Conv2dFun.forward_inner(
            (input.shape[0], weight.shape[0], input.shape[2], input.shape[3]),
            input,
            weight,
            reversed=False,
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values

        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        grad_weight = Conv2dFun.forward_inner(
            (weight.shape[1], weight.shape[0], weight.shape[2], weight.shape[3]),
            new_input,
            new_grad_output,
            reversed=False,
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        new_weight = weight.permute(1, 0, 2, 3)
        grad_input = Conv2dFun.forward_inner(
            input.shape, grad_output, new_weight, reversed=True
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply