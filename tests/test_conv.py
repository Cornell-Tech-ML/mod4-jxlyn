import pytest
from hypothesis import given, settings

import minitorch
from minitorch import Tensor

from .tensor_strategies import tensors
import numba
import numba.cuda
import numpy
import random


@pytest.mark.task4_1
def test_conv1d_simple() -> None:
    t = minitorch.tensor([0, 1, 2, 3]).view(1, 1, 4)
    t.requires_grad_(True)
    t2 = minitorch.tensor([[1, 2, 3]]).view(1, 1, 3)
    out = minitorch.Conv1dFun.apply(t, t2)

    assert out[0, 0, 0] == 0 * 1 + 1 * 2 + 2 * 3
    assert out[0, 0, 1] == 1 * 1 + 2 * 2 + 3 * 3
    assert out[0, 0, 2] == 2 * 1 + 3 * 2
    assert out[0, 0, 3] == 3 * 1


@pytest.mark.task4_1
@given(tensors(shape=(1, 1, 6)), tensors(shape=(1, 1, 4)))
def test_conv1d(input: Tensor, weight: Tensor) -> None:
    print(input, weight)
    minitorch.grad_check(minitorch.Conv1dFun.apply, input, weight)


@pytest.mark.task4_1
@given(tensors(shape=(2, 2, 6)), tensors(shape=(3, 2, 2)))
@settings(max_examples=50)
def test_conv1d_channel(input: Tensor, weight: Tensor) -> None:
    minitorch.grad_check(minitorch.Conv1dFun.apply, input, weight)


@pytest.mark.task4_2
@given(tensors(shape=(1, 1, 6, 6)), tensors(shape=(1, 1, 2, 4)))
def test_conv(input: Tensor, weight: Tensor) -> None:
    minitorch.grad_check(minitorch.Conv2dFun.apply, input, weight)


@pytest.mark.task4_2
@given(tensors(shape=(2, 1, 6, 6)), tensors(shape=(1, 1, 2, 4)))
@settings(max_examples=10)
def test_conv_batch(input: Tensor, weight: Tensor) -> None:
    minitorch.grad_check(minitorch.Conv2dFun.apply, input, weight)


@pytest.mark.task4_2
@given(tensors(shape=(2, 2, 6, 6)), tensors(shape=(3, 2, 2, 4)))
@settings(max_examples=10)
def test_conv_channel(input: Tensor, weight: Tensor) -> None:
    minitorch.grad_check(minitorch.Conv2dFun.apply, input, weight)


@pytest.mark.task4_2
def test_conv2() -> None:
    t = minitorch.tensor([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]).view(
        1, 1, 4, 4
    )
    t.requires_grad_(True)

    t2 = minitorch.tensor([[1, 1], [1, 1]]).view(1, 1, 2, 2)
    t2.requires_grad_(True)
    out = minitorch.Conv2dFun.apply(t, t2)
    out.sum().backward()

    minitorch.grad_check(minitorch.Conv2dFun.apply, t, t2)


if numba.cuda.is_available():
    @pytest.mark.task4_4b
    def test_conv1d_cuda() -> None:
        TEST_SAMPLES = 30  
        for tensor_shape, weight_shape in zip(
            [
                (1, 1, 8),  # Updated tensor shapes
                (2, 2, 5),
                (50, 50, 8),
                (80, 5, 6),
                (10, 100, 3),
                (150, 3, 4),
                (5, 20, 10),
                (25, 60, 8),
            ],
            [
                (1, 1, 5),  # Updated weight shapes
                (4, 2, 3),
                (2, 50, 6),
                (5, 3, 3),
                (3, 100, 2),
                (150, 4, 1),
                (4, 20, 6),
                (3, 60, 5),
            ],
        ):
            for _ in range(TEST_SAMPLES):
                tensor_storage = numpy.array(
                    [
                        random.random() * 2000 - 1000
                        for __ in range(numpy.prod(tensor_shape))
                    ]
                )
                weight_storage = numpy.array(
                    [
                        random.random() * 2000 - 1000
                        for __ in range(numpy.prod(weight_shape))
                    ]
                )
                tensor = Tensor.make(
                    tensor_storage, tensor_shape, backend=minitorch.SimpleBackend
                )
                weight = Tensor.make(
                    weight_storage, weight_shape, backend=minitorch.SimpleBackend
                )
                conva = minitorch.Conv1dFun.apply(tensor, weight)
                convb = minitorch.cuda_conv.Conv1dFun.apply(tensor, weight)
                numpy.testing.assert_allclose(
                    conva._tensor._storage,
                    convb._tensor._storage,
                    1e-3,
                    1e-3,  # Changed tolerance
                )
                minitorch.grad_check(
                    minitorch.cuda_conv.Conv1dFun.apply, tensor, weight
                )

    @pytest.mark.task4_4b
    def test_conv2d_cuda() -> None:
        TEST_SAMPLES = 30  # Changed the number of test samples
        for tensor_shape, weight_shape in zip(
            [
                (1, 1, 8, 8),
                (3, 2, 5, 5),
                (3, 3, 10, 10),
                (2, 40, 40, 4),
                (4, 60, 60, 6),
            ],
            [(1, 1, 3, 5), (2, 2, 3, 3), (4, 2, 2, 6), (3, 40, 40, 5), (2, 60, 3, 4)],
        ):
            for _ in range(TEST_SAMPLES):
                tensor_storage = numpy.array(
                    [
                        random.random() * 2000 - 1000
                        for __ in range(numpy.prod(tensor_shape))
                    ]
                )
                weight_storage = numpy.array(
                    [
                        random.random() * 2000 - 1000
                        for __ in range(numpy.prod(weight_shape))
                    ]
                )
                tensor = Tensor.make(
                    tensor_storage, tensor_shape, backend=minitorch.SimpleBackend
                )
                weight = Tensor.make(
                    weight_storage, weight_shape, backend=minitorch.SimpleBackend
                )
                conva = minitorch.Conv2dFun.apply(tensor, weight)
                convb = minitorch.cuda_conv.Conv2dFun.apply(tensor, weight)
                numpy.testing.assert_allclose(
                    conva._tensor._storage,
                    convb._tensor._storage,
                    1e-3,
                    1e-3,  # Changed tolerance
                )
                minitorch.grad_check(
                    minitorch.cuda_conv.Conv2dFun.apply, tensor, weight
                )
