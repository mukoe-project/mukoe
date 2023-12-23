from typing import Any, Callable, Iterable, Mapping, NamedTuple, Union
import specs

NestedArray = Any
NestedTensor = Any

# The pmap axis name. Data means data parallelization.
PMAP_AXIS_NAME = "data"

NestedSpec = Union[
    specs.Array,
    Iterable["NestedSpec"],
    Mapping[Any, "NestedSpec"],
]

Nest = Union[NestedArray, NestedTensor, NestedSpec]

TensorTransformation = Callable[[NestedTensor], NestedTensor]
TensorValuedCallable = Callable[..., NestedTensor]


class Transition(NamedTuple):
    """Container for a transition."""

    observation: NestedArray
    action: NestedArray
    reward: NestedArray
    discount: NestedArray
    next_observation: NestedArray
    extras: NestedArray = ()
