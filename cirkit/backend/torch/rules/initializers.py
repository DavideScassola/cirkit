# pylint: disable=unused-argument

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from cirkit.backend.compiler import InitializerCompilationSign
from cirkit.backend.torch.initializers import InitializerFunc, copy_from_ndarray_, dirichlet_
from cirkit.symbolic.initializers import (
    CholeskyInitializer,
    ConstantTensorInitializer,
    DirichletInitializer,
    NormalInitializer,
    UniformInitializer,
)

if TYPE_CHECKING:
    from cirkit.backend.torch.compiler import TorchCompiler


def compile_constant_tensor_initializer(
    compiler: "TorchCompiler", init: ConstantTensorInitializer
) -> InitializerFunc:
    if isinstance(init.value, np.ndarray):
        return functools.partial(copy_from_ndarray_, array=init.value)
    return functools.partial(torch.fill_, value=init.value)


def compile_uniform_initializer(
    compiler: "TorchCompiler", init: UniformInitializer
) -> InitializerFunc:
    return functools.partial(nn.init.uniform_, a=init.a, b=init.b)


def compile_normal_initializer(
    compiler: "TorchCompiler", init: NormalInitializer
) -> InitializerFunc:
    return functools.partial(nn.init.normal_, mean=init.mean, std=init.stddev)


def compile_dirichlet_initializer(
    compiler: "TorchCompiler", init: DirichletInitializer
) -> InitializerFunc:
    axis = init.axis if init.axis < 0 else init.axis + 1
    return functools.partial(dirichlet_, alpha=init.alpha, dim=axis)


def _init_cholesky(tensor: torch.Tensor, mean: float, stddev: float) -> torch.Tensor:
    """
    Initializes the tensor with random values and applies softplus to the diagonal
    after zeroing the upper triangle.
    Assumes tensor shape is (..., D, D).
    """
    with torch.no_grad():
        nn.init.normal_(tensor, mean=mean, std=stddev)
        tensor.tril_()  # Zero upper triangle
        diag_idx = torch.arange(tensor.size(-1), device=tensor.device)
        tensor[..., diag_idx, diag_idx] = torch.nn.functional.softplus(
            tensor[..., diag_idx, diag_idx]
        )
    return tensor


def compile_cholesky_initializer(
    compiler: "TorchCompiler", init: CholeskyInitializer
) -> InitializerFunc:
    return functools.partial(_init_cholesky, mean=init.mean, stddev=init.stddev)


DEFAULT_INITIALIZER_COMPILATION_RULES: dict[
    InitializerCompilationSign, Callable[..., InitializerFunc]
] = {
    ConstantTensorInitializer: compile_constant_tensor_initializer,
    UniformInitializer: compile_uniform_initializer,
    NormalInitializer: compile_normal_initializer,
    DirichletInitializer: compile_dirichlet_initializer,
    CholeskyInitializer: compile_cholesky_initializer,
}
