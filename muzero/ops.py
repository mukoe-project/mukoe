"""Stateless operations on JAX or numpy arrays."""
import dataclasses

import jax
import jax.numpy as jnp
import rlax

MIN_VALUE_TRANSFORMATION_EPS = 1e-8


@dataclasses.dataclass
class ValueTransformationOptions:
    min_value: float
    max_value: float
    num_bins: int
    value_transformation_epsilon: float


def value_transformation_options(
    value_transformation_epsilon: float = 0.001,
) -> ValueTransformationOptions:
    """Returns the value transformation options."""
    return ValueTransformationOptions(
        max_value=300.0,
        min_value=-300.0,
        num_bins=601,
        value_transformation_epsilon=value_transformation_epsilon,
    )


def inverse_value_transformation(x, eps: float):
    """Implements the inverse of the R2D2 value transformation."""
    _check_value_transformation_eps(eps)
    return rlax.signed_parabolic(x, eps) if eps != 0.0 else x


def _check_value_transformation_eps(eps: float) -> None:
    """Throws if the epsilon for value transformation isn't valid."""
    if eps < 0.0:
        raise ValueError("-ve epsilon ({}) not supported".format(eps))
    elif 0 < eps < MIN_VALUE_TRANSFORMATION_EPS:
        raise ValueError(
            "0 < eps < {} not supported ({})".format(MIN_VALUE_TRANSFORMATION_EPS, eps)
        )


def value_transformation(x, eps: float):
    """Implements the R2D2 value transformation."""
    _check_value_transformation_eps(eps)
    return rlax.signed_hyperbolic(x, eps) if eps != 0 else x


def clip_gradient(x: jnp.ndarray, abs_value: float) -> jnp.ndarray:
    """Clips the gradient of `x` to be in [-abs_value, abs_value]."""

    @jax.custom_gradient
    def wrapped(x: jnp.ndarray):
        return x, lambda g: (jnp.clip(g, -abs_value, abs_value),)

    return wrapped(x)
