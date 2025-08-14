import contextlib
import functools as ft
import inspect
from typing import TypeAlias, TypeVar, cast, Union

import beartype
import jax
import jax._src.tree_util as private_tree_util
import jax.core
from jaxtyping import Array  # noqa: F401
from jaxtyping import ArrayLike
from jaxtyping import Bool  # noqa: F401
from jaxtyping import DTypeLike  # noqa: F401
from jaxtyping import Float
from jaxtyping import Int  # noqa: F401
from jaxtyping import Key  # noqa: F401
from jaxtyping import Num  # noqa: F401
from jaxtyping import PyTree
from jaxtyping import Real  # noqa: F401
from jaxtyping import UInt8  # noqa: F401
from jaxtyping import config
from jaxtyping import jaxtyped
import jaxtyping._decorator
import torch

# patch jaxtyping to handle https://github.com/patrick-kidger/jaxtyping/issues/277.
# the problem is that custom PyTree nodes are sometimes initialized with arbitrary types (e.g., `jax.ShapeDtypeStruct`,
# `jax.Sharding`, or even <object>) due to JAX tracing operations. this patch skips typechecking when the stack trace
# contains `jax._src.tree_util`, which should only be the case during tree unflattening.
_original_check_dataclass_annotations = jaxtyping._decorator._check_dataclass_annotations  # noqa: SLF001


def _check_dataclass_annotations(self, typechecker):
    if not any(
        frame.frame.f_globals["__name__"] in {"jax._src.tree_util", "flax.nnx.transforms.compilation"}
        for frame in inspect.stack()
    ):
        return _original_check_dataclass_annotations(self, typechecker)
    return None


jaxtyping._decorator._check_dataclass_annotations = _check_dataclass_annotations  # noqa: SLF001

TorchTensor = torch.Tensor
TorchFloat = torch.Tensor  # For float tensors
TorchInt = torch.Tensor    # For int tensors
TorchBool = torch.Tensor   # For bool tensors
TorchUInt8 = torch.Tensor  # For uint8 tensors

# Union type for arrays that can be either JAX arrays or torch tensors
ArrayOrTorch = Union[Array, TorchTensor]

# Custom type annotations that work with both JAX and torch
def FloatOrTorch(shape: str = "...") -> type:
    """Type annotation for float arrays that can be JAX arrays or torch tensors."""
    return Union[Float[Array, shape], TorchFloat]

def IntOrTorch(shape: str = "...") -> type:
    """Type annotation for int arrays that can be JAX arrays or torch tensors."""
    return Union[Int[Array, shape], TorchInt]

def BoolOrTorch(shape: str = "...") -> type:
    """Type annotation for bool arrays that can be JAX arrays or torch tensors."""
    return Union[Bool[Array, shape], TorchBool]

def UInt8OrTorch(shape: str = "...") -> type:
    """Type annotation for uint8 arrays that can be JAX arrays or torch tensors."""
    return Union[UInt8[Array, shape], TorchUInt8]

KeyArrayLike: TypeAlias = jax.typing.ArrayLike
Params: TypeAlias = PyTree[Float[ArrayLike, "..."]]

T = TypeVar("T")


# Custom type checker that handles both JAX arrays and torch tensors
def _torch_aware_typechecker(func):
    """Type checker that can handle both JAX arrays and torch tensors."""
    @ft.wraps(func)
    def wrapper(*args, **kwargs):
        # Use beartype for basic type checking
        return beartype.beartype(func)(*args, **kwargs)
    return wrapper

# runtime type-checking decorator
def typecheck(t: T) -> T:
    return cast(T, ft.partial(jaxtyped, typechecker=_torch_aware_typechecker)(t))


@contextlib.contextmanager
def disable_typechecking():
    initial = config.jaxtyping_disable
    config.update("jaxtyping_disable", True)  # noqa: FBT003
    yield
    config.update("jaxtyping_disable", initial)


def check_pytree_equality(*, expected: PyTree, got: PyTree, check_shapes: bool = False, check_dtypes: bool = False):
    """Checks that two PyTrees have the same structure and optionally checks shapes and dtypes. Creates a much nicer
    error message than if `jax.tree.map` is naively used on PyTrees with different structures.
    """

    if errors := list(private_tree_util.equality_errors(expected, got)):
        raise ValueError(
            "PyTrees have different structure:\n"
            + (
                "\n".join(
                    f"   - at keypath '{jax.tree_util.keystr(path)}': expected {thing1}, got {thing2}, so {explanation}.\n"
                    for path, thing1, thing2, explanation in errors
                )
            )
        )

    if check_shapes or check_dtypes:

        def check(kp, x, y):
            if check_shapes and x.shape != y.shape:
                raise ValueError(f"Shape mismatch at {jax.tree_util.keystr(kp)}: expected {x.shape}, got {y.shape}")

            if check_dtypes and x.dtype != y.dtype:
                raise ValueError(f"Dtype mismatch at {jax.tree_util.keystr(kp)}: expected {x.dtype}, got {y.dtype}")

        jax.tree_util.tree_map_with_path(check, expected, got)
