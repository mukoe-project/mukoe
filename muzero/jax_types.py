import dataclasses
from typing import Any, Callable, Dict, Generic, Mapping, TypeVar

import acme_types as types
import chex
import dm_env
import jax
import jax.numpy as jnp

PRNGKey = jax.random.KeyArray
Networks = TypeVar("Networks")
"""Container for all agent network components."""
Policy = TypeVar("Policy")
"""Function or container for agent policy functions."""
Sample = TypeVar("Sample")
"""Sample from the demonstrations or replay buffer."""
TrainingState = TypeVar("TrainingState")

TrainingMetrics = Mapping[str, jnp.ndarray]
"""Metrics returned by the training step.

Typically these are logged, so the values are expected to be scalars.
"""

Variables = Mapping[str, types.NestedArray]
"""Mapping of variable collections.

A mapping of variable collections, as defined by Learner.get_variables.
The keys are the collection names, the values are nested arrays representing
the values of the corresponding collection variables.
"""


@chex.dataclass(frozen=True, mappable_dataclass=False)
class TrainingStepOutput(Generic[TrainingState]):
    state: TrainingState
    metrics: TrainingMetrics


Seed = int
EnvironmentFactory = Callable[[Seed], dm_env.Environment]


@dataclasses.dataclass
class ModelToSnapshot:
    """Stores all necessary info to be able to save a model.

    Attributes:
      model: a jax function to be saved.
      params: fixed params to be passed to the function.
      dummy_kwargs: arguments to be passed to the function.
    """

    model: Any  # Callable[params, **dummy_kwargs]
    params: Any
    dummy_kwargs: Dict[str, Any]
