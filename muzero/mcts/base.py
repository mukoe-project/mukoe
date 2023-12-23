"""Base class for MCTS."""

import dataclasses
from typing import Any, Callable, List, Optional, Tuple, Union, Generic, TypeVar

import numpy as np


@dataclasses.dataclass(frozen=True)
class PredictionFnOutput:
    """The output of prediction network."""

    # The predicted value.
    value: Union[float, int]
    # The predicted per step reward.
    reward: Union[float, int]
    # The predicted action logits.
    action_logits: Optional[np.ndarray] = None


EmbeddingT = TypeVar("EmbeddingT")


@dataclasses.dataclass(frozen=True)
class ModelFunctions(Generic[EmbeddingT]):
    """A collection of functions that are used by the search."""

    # TODO(miaosen): ModelFunctions need a redesign so that it can handle more
    # generic cases and Mz specifics.
    # Callable function for the representation net. Given an observation, it
    # returns the embeddings of the observation and the prediction.
    repr_and_pred: Callable[[Any], Tuple[EmbeddingT, PredictionFnOutput]]

    # Callable function for dynamics net and prediction net, it returns the next
    # state and predictions given the current state and action.
    dyna_and_pred: Callable[[Any, EmbeddingT], Tuple[EmbeddingT, PredictionFnOutput]]

    # Callable function for returning a list of legal action masks, 1 for valid
    # actions, 0 for invalid actions. Input is the action history stored as a list
    # of numbers.
    get_legal_actions_mask: Callable[[List[Union[float, int]]], np.ndarray]
